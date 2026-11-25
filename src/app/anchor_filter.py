"""OCR 文本 → 地理锚点的**有效性过滤层**（可插拔）。

职责边界
--------
这段逻辑回答一个问题：**OCR 读到的文本里，哪些压根不是地图地名、
不该占用地理编码名额、不该进 RANSAC？** 它是整条配准流水线的第①层
（详见 docs/HANDOVER.md），后面还有「地理编码反馈 → LLM 语义 →
几何一致性裁决」三层。

为什么要模块化
--------------
过滤规则会持续演进：这一版是黑名单 + 正则（零成本、毫秒级），
下一版可能接大模型做残片还原（``杭甬高 → 杭甬高速``）。把接口与
实现分开，替换时**只改注册处一行**，调用方（``ocr_anchor`` /
``pipeline``）无需感知。

扩展方式（见文件末尾 ``register_filter``）
------------------------------------------
1. 实现 :class:`AnchorFilter` 协议（一个 ``reject(text) -> bool`` 方法）；
2. 用 ``register_filter("llm", MyFilter())`` 注册；
3. 在 ``.env`` 设 ``ANCHOR_FILTER=llm``（热读，无需重启）。

任何实现都必须遵守：**宁可漏放，不可误杀真实地名**。
被误杀的地名会直接减少配准锚点——实测把「迪士尼/东柳」挡在门外
会让整张图定不出尺度（2026-11-21 行者宁波）。所以默认实现对
模糊情况一律放行，只拦有强特征的噪声。
"""
from __future__ import annotations

import os
import re
from typing import Callable, Protocol, runtime_checkable

# 环境变量名：选哪个过滤器实现（默认 "rule"）
ANCHOR_FILTER_ENV = "ANCHOR_FILTER"
DEFAULT_FILTER_NAME = "rule"


@runtime_checkable
class AnchorFilter(Protocol):
    """文本有效性过滤器接口。

    ``reject(text)`` 返回 True 表示「这段文本不是有效地名，丢弃」。
    实现应无状态或自行管理缓存；会被高频调用，务必轻量。
    """

    def reject(self, text: str) -> bool: ...


class RuleFilter:
    """默认实现：纯规则，零网络、毫秒级。

    四组强特征，任一命中即判为噪声：

    1. 泛词单独出现（大厦、酒店……）——地名库里同名要素遍地都是；
    2. 运动 App UI 文案（距离、运动时间、导出删除分享……）；
    3. 纯数值/度量文本（34.63、15.67km/h、655米、2:12:39）；
    4. OCR 粘连碎片（A.十、34.67新=江18.29）——中英标点异常混排。

    另外 ``_NON_NAME``（纯符号/数字）也归这里，避免调用方重复判断。
    """

    name = "rule"

    # 道路/水系类后缀的判定仍留在 ocr_anchor.classify_text（它决定锚点
    # 类型而非丢弃），这里只管「压根不是地名」的文本。

    _NON_NAME = re.compile(r"^[\W\d_]+$")

    # 泛词黑名单：单独出现时指代一类设施而非具体地点。
    # 实测「大厦」匹配到 22 公里外同名建筑，2 点模型尺度爆炸。
    _GENERIC_WORDS = frozenset({
        "大厦", "大楼", "酒店", "宾馆", "饭店", "广场", "中心", "超市",
        "银行", "公司", "停车场", "厕所", "餐厅", "便利店", "药店",
        "门诊部",
    })

    # 运动 App 界面文案：统计栏表头/统计项/状态栏/导航按钮。
    # 行者宁波 34.67km 截图（2026-11-21）：这 12 个 UI 文案先于真地名
    # 占满 GEOCODE_TARGET_LIMIT=20 名额，RANSAC 定不出尺度直接失败。
    # 小米运动健康 13.09km 截图（2026-11-23）：「公里」「更慢/更快」
    # 「运动时长」「活动卡路里」等同样占名额，且「活动卡路里/运动时长」
    # 一度成为仅有的 2 个配准锚点（RMSE 260m 的直接推手）。
    _UI_LABELS = frozenset({
        "距离", "速度", "时间", "均速", "配速", "坡度", "心率", "步频",
        "骑行", "跑步", "步行", "游泳", "运动时间", "全程时间",
        "运动均速", "全程均速", "最快速度", "平均速度", "累计上升",
        "累计下降", "累计爬升", "海拔范围", "最高海拔", "最低海拔",
        "平均心率", "最大心率", "消耗卡路里", "卡路里",
        # 小米运动健康统计卡片与图例
        "公里", "千米", "更慢", "更快", "运动时长", "活动卡路里",
        "平均配速", "平均海拔", "平均步频", "平均步幅", "平均心率",
        "户外跑步", "户外骑行", "动态轨迹",
        "中国移动", "中国联通", "中国电信", "返回", "<返回", "地图",
        "分享", "导出", "删除", "详细", "备注", "数据", "图表",
        "保存", "编辑", "开始", "结束", "暂停", "继续", "设置",
        "取消", "确定",
        # 高德分享卡浮动按钮 / 运营位（2026-11-25 虹桥样本实测：
        # 「联系官方」「晒路线」穿透过滤成了 POI 锚点，「晒路线」还被
        # 百度模糊匹配到无关 POI，纯添噪）
        "联系官方", "联系客服", "官方客服", "晒路线", "分享路线",
        "分享运动记录", "生成海报", "保存图片", "身",
    })

    # UI **子串**特征：OCR 粘连块（「小米运动健康一户外跑步」
    # 「中国联通21:50385」）不是全等匹配能拦的。只收「出现在任何文本
    # 里都几乎必是 UI」的强特征串，避免误杀真实地名。
    _UI_SUBSTRINGS = (
        "小米运动健康", "运动健康｜", "户外跑步", "动态轨迹",
    )
    # 状态栏运营商前缀：粘连块「中国联通21:50…」按前缀拦截
    # （真实 POI「中国联通XX营业厅」会被误杀——按「宁可漏放」原则，
    # 牺牲这类低价值锚点换状态栏不再占名额是值得的）。
    _CARRIER_PREFIX = re.compile(r"^中国(?:联通|移动|电信|铁塔)")

    # 纯数值/度量文本
    _NUMERIC_MEASURE = re.compile(
        r"^\s*\d[\d:.,\-/]*\s*"
        r"(?:公里|千米|公里/小时|米|米/秒|米/分|km/h|km|m/s|m|h|min|s|秒)?\s*$",
        re.I,
    )

    # 按钮连写（「导出删除分享」）：两个及以上 UI 动词
    _UI_VERBS = ("导出", "删除", "分享", "返回", "备注", "数据", "图表")

    # OCR 粘连/乱码碎片：中英标点异常混排（A.十、34.67新=江18.29）。
    # 高德对这类串会硬配一个近似 POI，混进锚点后几何自洽性极差。
    _GARBAGE_MIXED = re.compile(
        r"^[A-Za-z]\s*[.．·]\s*[\u4e00-\u9fff]"
        r"|[\d\u4e00-\u9fff]\s*[=＝]\s*[\u4e00-\u9fff\d]",
    )

    def is_ui_noise(self, text: str) -> bool:
        """UI 文案 / 数值 / OCR 碎片（供外部单独调用，语义更窄）。"""
        t = text.strip()
        if t in self._UI_LABELS or self._NUMERIC_MEASURE.fullmatch(t):
            return True
        if any(m in t for m in self._UI_SUBSTRINGS):
            return True
        if self._CARRIER_PREFIX.match(t):
            return True
        if self.is_ocr_garbage(t):
            return True
        return sum(w in t for w in self._UI_VERBS) >= 2 and len(t) <= 16

    def is_ocr_garbage(self, text: str) -> bool:
        """OCR 粘连/乱码碎片（强特征匹配，不碰正常地名）。"""
        t = text.strip()
        if not t or len(t) < 2:
            return True
        return bool(self._GARBAGE_MIXED.search(t))

    def reject(self, text: str) -> bool:
        """统一入口：判断是否丢弃这段文本。"""
        t = text.strip()
        if not t or self._NON_NAME.match(t):
            return True
        if t in self._GENERIC_WORDS or t.endswith("出入口"):
            return True
        if self.is_ui_noise(t):
            return True
        if re.fullmatch(r"[\d:.]+", t):
            return True
        return False


# --- 注册表 / 工厂 ----------------------------------------------------------
_REGISTRY: dict[str, Callable[[], AnchorFilter]] = {}


def register_filter(name: str, factory: Callable[[], AnchorFilter]) -> None:
    """注册一个过滤器实现。重复注册同名会覆盖（便于测试注入）。"""
    _REGISTRY[name] = factory


def get_filter(name: str | None = None) -> AnchorFilter:
    """按名字取过滤器实例。未知名字回落默认规则实现，绝不抛异常。

    ``name`` 缺省时读环境变量 ``ANCHOR_FILTER``（热读，无需重启）。
    """
    key = (name or os.environ.get(ANCHOR_FILTER_ENV, "") or DEFAULT_FILTER_NAME).strip()
    factory = _REGISTRY.get(key)
    if factory is None:
        return _REGISTRY[DEFAULT_FILTER_NAME]()
    try:
        return factory()
    except Exception:  # noqa: BLE001 — 可选实现初始化失败不能拖垮主流程
        return _REGISTRY[DEFAULT_FILTER_NAME]()


register_filter("rule", RuleFilter)

# 模块级共享实例：无状态规则过滤器，复用即可。
DEFAULT = get_filter()


# --- 向后兼容别名 ----------------------------------------------------------
# ocr_anchor / pipeline / 既有测试仍按老函数名调用；内部统一转发到 DEFAULT。
# 这样替换实现时外部调用点无需改动。
def is_ui_noise(text: str) -> bool:
    if hasattr(DEFAULT, "is_ui_noise"):
        return bool(DEFAULT.is_ui_noise(text))  # type: ignore[attr-defined]
    return DEFAULT.reject(text)


def is_ocr_garbage(text: str) -> bool:
    if hasattr(DEFAULT, "is_ocr_garbage"):
        return bool(DEFAULT.is_ocr_garbage(text))  # type: ignore[attr-defined]
    return False
