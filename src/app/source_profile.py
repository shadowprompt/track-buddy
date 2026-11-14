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

当前只注册 ``meituan`` 一个画像（参数值与重构前完全一致，行为零变化）。
未来支持高德/小米/Keep 时：在这里新增 profile + 用真实截图标定参数，
再用 ``SOURCE_PROFILE`` 环境变量切换。切换是**全局**的（单进程单来源），
不做逐请求自动检测——来源自动识别属于拿到多样本之后的后续演化，
不提前设计。

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


# 当前唯一注册的画像：美团骑行。参数值与模块化之前完全一致——
# VisionConfig 的默认值就是为美团截图标定的。
MEITUAN = SourceProfile(
    id="meituan",
    title="美团骑行",
    vision=VisionConfig(),
    distance_positive=("里程", "本次", "骑行", "行程", "全程", "共计", "合计", "距离"),
    distance_negative=("免费", "累计", "套餐", "赠送", "优惠", "已骑", "封顶"),
)

# 画像注册表：新增来源（amap / xiaomi / keep ...）在这里加条目，
# 并按模块 docstring 的标定指南填参数。**不要**在这里做未标定的占位
# 条目——切换到没标定过的画像等于全链路盲跑。
PROFILES: dict[str, SourceProfile] = {"meituan": MEITUAN}

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
