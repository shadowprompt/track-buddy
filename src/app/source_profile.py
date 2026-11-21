"""输入源画像（Source Profile）：把「截图来自哪个 App」的差异集中到
一个可扩展的数据结构里。

**为什么要有这一层**（2026-11-14 演化决策）：识别链路天然分两层——

* **下游与来源无关**：多图拼接（SIFT / ``llm`` / ``llm_guided`` 后端）、
  轨迹模板飞轮、地名 OCR、地理编码、路网精修、GeoJSON 输出——全部
  只依赖「一张像素轨迹 + 一批地名文本」，新增截图来源时零改动复用。
* **上游与来源绑定**：视觉提取参数（轨迹线掩膜阈值、起/终徽章 HSV、
  形态学核、RANSAC 内点容差）与 OCR 文案启发式（里程正则词表）是
  **每个 App 一套**。美团轨迹是浅荧光绿（``V>200 / S<90`` 可提），
  高德运动轨迹是亮蓝色系，参数互斥，不能混用。

当前注册 ``meituan`` 一个画像（参数值与重构前完全一致，行为零变化）。
未来支持高德/小米/Keep 时：在这里新增 profile + 用真实截图标定参数。

**选择生效画像的两条路径**（优先级从高到低）：

1. **请求级入参**（2026-11-14 起）：``POST /api/recognize`` /
   ``recognize_multi`` 的 ``source`` 表单字段。server 把解析出的画像
   挂到本模块的 ContextVar 上，``CONFIG.source`` / ``CONFIG.vision``
   在请求处理期间自动跟随——识别链路里的读取点一行都不用改。
   单进程可以同时服务多来源请求，但**模板库按画像隔离**
   （见 ``route_template.py``），防止美团模板命中高德截图。
2. **环境级默认**：``SOURCE_PROFILE`` 环境变量（不传入参时生效），
   CLI / 直跑 pipeline 的场景走这条路径。

不提前设计「来源自动识别」——拿到多样本之后再说。

参数标定指南（给未来的自己）：

1. 轨迹线：对样本截图统计轨迹像素的 HSV 分布，调
   ``track_max_value`` / ``track_max_saturation``（当前美团取
   「亮于 200 且饱和度低于 90」——浅色轨迹线，避开彩色虚线与文字）。
2. 徽章：起/终点图标的色相区间写进 ``start_hsv_range`` /
   ``end_hsv_range(+2)``，面积阈值按图标实际像素面积×安全系数。
3. RANSAC 容差 ``ransac_inlier_m``：底图 POI 文字标注中心与真实点位的
   固有偏移决定（美团实测 7-28 像素 ≈ 35-150 米，故取 200 米）。
4. OCR 词表：正面词（明确指向「本次里程」的文案）与干扰词（运营文案/
   累计统计），各来源 App 的页面文案不同。
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass

from .config import VisionConfig


@dataclass(frozen=True)
class SourceProfile:
    """一个截图来源的完整视觉与文案参数集。"""

    id: str
    """机器标识，``SOURCE_PROFILE`` 环境变量的取值。"""

    title: str
    """展示名（诊断、前端、日志里显示）。"""

    vision: VisionConfig
    """视觉提取参数（徽章 HSV / 轨迹掩膜 / 形态学核 / RANSAC 容差）。"""

    distance_positive: tuple[str, ...]
    """里程 OCR 正面词：命中时该文本块里的数字优先（如「本次骑行 9.6 公里」）。"""

    distance_negative: tuple[str, ...]
    """里程 OCR 干扰词：命中时整块排除（如「累计骑行 1024 公里」）。"""

    ride_labels: tuple[str, ...]
    """订单详情页的「本次」字段标签（ride_summary 解析用）。

    语义映射是通用的：含「骑行/里程」→ 本次距离、「消耗/卡」→ 卡路里、
    「碳」→ 减碳克数；标签**文案**随 App 不同，各画像自备。
    """


# 当前唯一注册的画像：美团骑行。参数值与模块化之前完全一致——
# VisionConfig 的默认值就是为美团截图标定的。
MEITUAN = SourceProfile(
    id="meituan",
    title="美团骑行",
    vision=VisionConfig(),
    distance_positive=("里程", "本次", "骑行", "行程", "全程", "共计", "合计", "距离"),
    distance_negative=("免费", "累计", "套餐", "赠送", "优惠", "已骑", "封顶"),
    ride_labels=("本次骑行", "本次消耗", "本次减碳"),
)

# 高德运动（2026-11-16 接入；同日按真实样本精标——上海 3.68km 骑行
# 截图 + 8 图拼接现场）。关键实测结论：**高德轨迹不是单色线，是速度
# 渐变线**（红=最慢 → 橙黄 → 绿=最快，见详情页底部图例条），一条轨迹
# 横跨多个色相带，因此 track_hsv_range 用多色带并集：
#   - 橙红黄带 H[8,30]：实测整条轨迹主体（S p25≈191，S>=150 避开底图）
#   - 高饱和绿带 H[35,70]：快速段（S>=180 避开薄荷绿公园 S≈50-90）
#   - 终点红标 H[0,7] 与轨迹红段 H[8,30] 天然错开，徽章检测不串扰
# 已知残留干扰：详情页底部的速度图例条（同为渐变色）与小装饰会进掩膜，
# 靠连通域筛选排除；里程牌白底胶囊（3KM/2KM/1KM）会切断轨迹线，依赖
# 下游碎片拼接。样本积累后继续收紧：python -m tools.calibrate_profile
AMAP = SourceProfile(
    id="amap",
    title="高德运动",
    vision=VisionConfig(
        track_hsv_range=(
            (8, 150, 120, 30, 255, 255),    # 红→橙→黄（慢→中速段）
            (35, 180, 120, 70, 255, 255),   # 高饱和绿（快速段）
        ),
        # 整屏截图的地图区裁剪（2026-11-16 真实样本实测）：高德地图区
        # 固定在 y≈[6.5%, 44%]，下方是概要卡片 / 运营横幅 / 按钮——
        # 速度图例条（同款红→橙→绿渐变）、红心装饰、内嵌地图的视频
        # 缩略图全在卡片区，不裁会污染掩膜/徽章并带偏 SIFT 拼接。
        map_crop_v=(0.065, 0.44),
        # 起点绿点：实测色相落在通用绿色区间内，沿用美团范围
        start_hsv_range=(35, 90, 90, 85, 255, 255),
        # 终点红标 #E94340 系（H≈0-3）：上界压到 7 与轨迹红段(8-30)划清
        end_hsv_range=(0, 110, 110, 7, 255, 255),
        # 关闭洋红端（170-179）：会命中详情页红心/星星装饰，误判终点
        end_hsv_range2=(0, 0, 0, 0, 0, 0),
        badge_min_area=1500,
        badge_max_area=20000,
        # POI 标注偏移与美团同量级（标注排版决定），容差先沿用 200m。
        ransac_inlier_m=200.0,
    ),
    # 高德骑行详情页文案：「里程 / 用时 / 均速 / 消耗」；干扰侧是
    # 个人页的累计统计与运营位（累计、排行、挑战、会员、红包）。
    distance_positive=("里程", "距离", "骑行", "行程", "全程", "共计", "合计", "本次"),
    distance_negative=("累计", "历史", "排行", "挑战", "会员", "红包", "优惠", "赠送", "已骑"),
    # 运动详情页实测（2026-11-16 OCR）：「骑行 3.68 干米」「运动用时
    # 00:14:14」「消耗热量 79大卡」——距离标签就是「骑行」，热量标签
    # 「消耗热量」；高德骑行页没有减碳字段（ride_summary 宁缺勿错）。
    ride_labels=("骑行", "消耗热量"),
)

# 行者（2026-11-17 接入，按真实样本精标——宁波 34.67km 卫星底图骑行页）。
# 关键实测（与美团/高德都不同）：
#   * 底图是**卫星影像**，深绿/棕色，低饱和中等亮度——美团「上限式」掩膜
#     (V<200,S<90) 会把整张卫星图吞进来（实测 57 万像素 = 全图 79%）。
#   * 轨迹是**鲜艳橙红粗线**（H≈7-9，S≈175-190，线宽 ~8px），色相与底图
#     分离干净，改用区间式掩膜 H[5,22] S[150,255]。
#   * 图上另有**蓝色规划/对比线**（H≈95， filaments 细），必须靠色相排除。
#   * 起点是红色三角旗、终点是白底红字水滴徽章，都与橙红轨迹同色相，
#     现有 find_marker 的「找最大连通域」会把整条轨迹当徽章。解法：
#     启用 badge_min_fill=0.25——徽章是团块(fill≈0.37)、轨迹是细长线
#     (fill≈0.06)，用填充率排除轨迹，精确定位两端（通用改动，其他画像
#     不受影响）。有了真徽章，骨架图搜索能接上起终点引线。
#   * 顶栏「距离 34.67 / 速度 18.29 / 时间 1:53:44」里程**不带单位**——
#     概要正则需要 km/公里字样，故顶栏里程目前不参与尺度校验（见 ride_summary）。
XINGZHE = SourceProfile(
    id="xingzhe",
    title="行者",
    vision=VisionConfig(
        track_hsv_range=(
            (5, 150, 120, 22, 255, 255),    # 橙红轨迹（含起点红旗）
        ),
        # 行者终点是红色团块，与橙红轨迹同色相——靠填充率区分：
        # 实测终点红字 area452/fill0.37，整条轨迹连通域 area16141/fill0.063。
        # min_fill=0.25 把轨迹排除。
        # 起点是「蓝色定位圆点」（H≈104 S≈162 V≈220，area≈775/fill≈0.67），
        # 在终点徽章旁（实测两点相距 38.6px）——闭环赛道，起终点同一位置。
        # 注意：不能沿用红色起点检测（会误抓左边缘的导航指北针箭头）。
        start_hsv_range=(95, 100, 150, 120, 255, 255),
        end_hsv_range=(0, 90, 90, 15, 255, 255),
        end_hsv_range2=(255, 255, 255, 0, 0, 0),
        badge_min_area=150,
        badge_max_area=3000,
        badge_min_fill=0.25,
        # 轨迹线宽 ~8px，闭运算核略收窄防相邻道路粘连
        track_close_kernel=7,
        track_open_kernel=5,
        # 终点邻域碎片：白色「终」徽章 + 蓝色定位点把底部「跑两遍」的
        # U 形折返切成独立小连通域（实测 364px、质心约 (333,1079)）。
        # 闭环管线（loop_course）会把它桥接进主轨迹。
        endpoint_fragment_detect=(100, 80),
        # 闭环赛道：宁波样本起点=终点（用户确认），走中国邮路闭环追踪。
        # 实测：主轨迹弧长 2476px ≈ 32.2km（比例尺 77px=1km），对比
        # App 记录 34.67km 误差 7%；旧开环管线 25.2km 误差 27%。
        loop_course=True,
        # 断裂段桥接：长截图里右上角/左下角轨迹被 POI 文字、比例尺、
        # 图标打断成独立连通域（area 962/721/605，gap 4-131px），
        # 只取最大域会把它们整段丢掉。桥接线状近邻段回主域。
        loop_bridge_fragments=True,
        # 路网吸附交付：闭环描迹的绝对位置偏差（~300-500m）与骨架锯齿
        # 会让画出来的线斜穿街区；吸附后每段沿街（2026-11-18）。
        road_snap_deliver=True,
        # 内容尺度标尺：掩膜核（close_kernel=7）是对旧「起点罗盘箭头」
        # 的直径 39.2px 标定的。起点换成蓝点（直径 31.4）后自动反推会把
        # 核缩到 5，掩膜断口激增——显式锁定标定值。
        scale_badge_d=39.2,
        # 卫星底图 POI 标注偏移与美团同量级，容差沿用。
        ransac_inlier_m=200.0,
    ),
    # 行者顶栏/详情页文案：正面词「距离/里程/骑行距离」；
    # 干扰「累计/总里程/本周/本月」等统计。
    distance_positive=("距离", "里程", "骑行", "行程", "全程", "本次"),
    distance_negative=("累计", "总计", "本周", "本月", "全年", "历史"),
    ride_labels=("距离", "骑行距离"),
)

# 画像注册表：新增来源（xiaomi / keep ...）在这里加条目，
# 并按模块 docstring 的标定指南填参数。**不要**在这里做未标定的占位
# 条目——切换到没标定过的画像等于全链路盲跑。
PROFILES: dict[str, SourceProfile] = {
    "meituan": MEITUAN,
    "amap": AMAP,
    "xingzhe": XINGZHE,
}

DEFAULT_PROFILE_ID = "meituan"


def resolve_source_profile(profile_id: str | None) -> SourceProfile:
    """按 id 解析画像；未知 id 直接报错并列出可用值，绝不静默回退。"""
    pid = (profile_id or DEFAULT_PROFILE_ID).strip().lower() or DEFAULT_PROFILE_ID
    try:
        return PROFILES[pid]
    except KeyError:
        raise ValueError(
            f"未知的输入源画像 {pid!r}；可用值：{', '.join(sorted(PROFILES))}"
        ) from None


# ---------- 请求级画像上下文（入参控制的机制层） --------------------------

# 为什么用 ContextVar 而不是给整条链路加 profile 形参：``CONFIG.source`` /
# ``CONFIG.vision`` 的读取点分散在 pipeline / visuals / multishot /
# input_quality / ride_summary / server 的十几处**调用时刻**读取里，
# 全部改成显式传参是一次大范围签名重构且极易漏改；ContextVar 让
# 「请求处理期间画像生效、请求结束自动还原」对既有代码完全透明。
#
# FastAPI 的 async 端点与 yield 依赖运行在同一个 asyncio task 里——
# 依赖里 set 的值在端点体内可见，请求结束 reset，不会跨请求泄漏。
_ACTIVE_PROFILE: contextvars.ContextVar[SourceProfile | None] = (
    contextvars.ContextVar("active_source_profile", default=None))


def current_active_profile() -> SourceProfile | None:
    """当前请求显式指定的画像；没有（走环境默认）时返回 ``None``。"""
    return _ACTIVE_PROFILE.get()


@contextmanager
def active_source_profile(profile: SourceProfile | None):
    """在 ``with`` 块内把 ``profile`` 设为当前画像；退出自动还原。

    传 ``None`` 表示清除覆盖（回落 ``SOURCE_PROFILE`` 环境变量默认）。
    后台线程（路网精修）不会自动继承调用方的 ContextVar，须在
    线程入口处显式用它重建上下文（见 ``server._run_refine``）。
    """
    token = _ACTIVE_PROFILE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE_PROFILE.reset(token)


def activate_profile_globally(profile: SourceProfile) -> None:
    """在**非请求**上下文（后台线程）里设置画像，随线程生命周期生效。

    ``threading.Thread`` 不继承调用方的 ContextVar：后台线程从一个
    空上下文启动，画像必须在线程入口显式设置。线程只跑单一任务、
    结束即销毁，无需 reset。
    """
    _ACTIVE_PROFILE.set(profile)
