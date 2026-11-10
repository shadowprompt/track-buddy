"""P1：宽泛概览图的**前置**判据，以及参考级定位的置信度规则。

为什么必须独立于 ``_assess_zoom_level``
---------------------------------------
``pipeline._assess_zoom_level()`` 要等地理编码**之后**才能判断缩放级别
（它读的是 ``AnchorPair.kind``），可那时垃圾锚点已经进场了：城市级截图上的
行政区、地铁线、高架名会被编码成区中心或不相关 POI，锚点标注位置与真实
点位差 0.8-16 公里，配准尺度随之崩坏（历史上出现过 680km、22m/px、
RMSE 1800m 的结果）。所以 P1 必须在阶段 3 在线地理编码**之前**，只用
OCR 文本就把这类图认出来，先试历史轨迹模板。

实测（用户提供的两张高缩放概览图，1260×2720，同一条路线的往返方向）：

* OCR 42 / 46 条，几乎全是「普陀区 / 静安区 / 徐汇区」、
  「3号线 / 12号线 / 4号线内圈」、「中环路 / 外环高速 / 南北高架路」、
  「上海南站 / 大宁公园 / 西岸艺术中心」；
* 街道级路名只有「龙吴路」等 1-2 条；
* 轨迹提取却有 872 / 829 点、confidence 0.95，形状信息完全够用。

也就是说「看不清细节」不等于「OCR 为空」，也不等于「没有形状」——
真正不可用的是**地名锚点的空间精度**，形状反而是可靠的。
"""
from __future__ import annotations

import re
from typing import Any

# 触发「宽泛概览图」的门槛。宁可漏判（回落原流程，行为与改动前一致），
# 也不能误判（会让街道级截图跳过正常地理编码）。
MIN_OCR_TEXTS = 10
MIN_BROAD_NAMES = 6
MIN_BROAD_RATIO = 0.35
MAX_STREET_NAMES = 2

# ⚠️ 与 pipeline._EXPRESSWAY_MARKERS 保持同一约定：**不要**把「大道」
# 算作大范围对象——世纪大道/龙东大道在城市里仍是有效的街道级定位参考。
_BROAD_MARKERS = (
    "高速", "高架", "环路", "环线", "国道", "省道", "联络线",
    "地铁", "轻轨", "机场",
)
_BROAD_SUFFIXES = (
    "区", "县", "市", "站", "公园", "绿地", "广场", "中心",
    "大学", "医院", "体育馆", "艺术中心",
)
_STREET_SUFFIXES = ("路", "街", "巷", "弄", "道", "桥")
_LINE_RE = re.compile(r"\d+\s*号线|[内外部]圈")


def _text_of(item: Any) -> str:
    """兼容 TextItem / tuple / str 三种 OCR 结果形态。"""
    s = getattr(item, "text", None)
    if s is None and isinstance(item, (tuple, list)) and item:
        s = item[0]
    if s is None and isinstance(item, str):
        s = item
    return str(s or "").strip()


def is_broad_name(name: str) -> bool:
    """行政区 / 地铁线 / 高架 / 大 POI —— 文字标注位置不代表真实点位。"""
    if not name:
        return False
    if name.endswith(_BROAD_SUFFIXES):
        return True
    if _LINE_RE.search(name):
        return True
    return any(k in name for k in _BROAD_MARKERS)


def is_street_name(name: str) -> bool:
    """街道级路名；含「高速/高架/环路」的长道路不算。"""
    if not name or is_broad_name(name):
        return False
    return name.endswith(_STREET_SUFFIXES)


def assess_broad_overview(
    texts: Any, road_names: Any = None
) -> dict[str, Any]:
    """只用 OCR 文本判断是否为「地名密集但空间精度差」的概览图。

    五个条件同时成立才触发：文本足够多、宽泛地名足够多且占比过半门槛、
    街道级路名极少、用户没有手填路名。返回结构直接进 diagnostics，
    方便事后复盘为什么走/没走模板通道。
    """
    names: list[str] = []
    for item in texts or []:
        s = _text_of(item)
        if s and s not in names:
            names.append(s)
    broad = [x for x in names if is_broad_name(x)]
    street = [x for x in names if is_street_name(x)]
    n = len(names)
    ratio = len(broad) / n if n else 0.0
    user_roads = bool([x for x in (road_names or []) if str(x).strip()])
    triggered = bool(
        n >= MIN_OCR_TEXTS
        and len(broad) >= MIN_BROAD_NAMES
        and ratio >= MIN_BROAD_RATIO
        and len(street) <= MAX_STREET_NAMES
        and not user_roads
    )
    return {
        "broad_overview": triggered,
        "n_texts": n,
        "n_broad": len(broad),
        "n_street": len(street),
        "broad_ratio": round(ratio, 3),
        "broad_examples": broad[:6],
        "street_examples": street[:6],
        "user_road_names": user_roads,
    }


def reference_confidence(match: dict[str, Any], cap: float = 0.45) -> float:
    """参考级置信度：形状越贴合略高，但**封顶**且远低于真实配准结果。

    没有任何当次地理核验，所以这个值只表达「形状有多像」，不表达
    「坐标有多准」。消费方必须同时读 diagnostics.source == "template"。
    """
    cov = float(match.get("coverage") or 0.0)
    res = float(match.get("residual") or 1.0)
    value = 0.30 + 0.15 * cov - 0.5 * res
    return round(max(0.10, min(cap, value)), 3)
