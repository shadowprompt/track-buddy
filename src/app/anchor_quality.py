"""锚点质量：把「OCR 残片」和「系统性偏移」从配准里分出来。

背景
----
``_KIND_TOLERANCE["road"]`` 给了 800m 的米制下限，本意是道路是**线对象**、
标注点可以落在沿线任意位置。但 OCR 残片（截断的路名、半个地名）也会被
判成 road，并以 600m 级残差混进内点：实测一张单图 9 个内点里混了
5 个残片，RMSE 被拉到 270.8m，诊断数字完全不能反映配准质量。

本模块做三件事：

1. ``is_road_fragment()`` —— road 类锚点必须是一个**完整的道路名**，
   否则判为残片，不参与 RANSAC（仍保留在诊断/离群列表里，可追溯）；
2. ``split_anchor_quality()`` —— 批量分流；
3. ``systematic_bias()`` —— 用内点残差**向量**的方向一致性识别
   「整组锚点被同一个源系统性带偏」，只输出诊断，不自动改坐标。

判据只看名称与锚点类型，与城市、分辨率、具体路名无关。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from .geo.coords import haversine, lonlat_to_mercator

# 只做类型标注，运行时不导入 register —— register 会导入本模块，
# 直接 import 会形成循环。
if TYPE_CHECKING:
    from .register import AnchorPair

# 常见道路通名后缀。不含任何具体地名，换城市/换语言可整体替换。
ROAD_SUFFIXES: tuple[str, ...] = (
    "路", "街", "大道", "大街", "巷", "弄", "里", "道", "桥",
    "高速", "公路", "环路", "快速路", "辅路", "支路", "隧道",
    "立交", "大道", "街坊", "胡同", "路东", "路西", "路南", "路北",
)

# 完整路名至少 3 个字符（「X路」两字也合法，故单独放宽）。
_MIN_FULL_ROAD_LEN = 2

# 方向相干性 / 平均残差幅值同时超阈值，才提示「系统性偏移」。
_BIAS_COHERENCE = 0.80
_BIAS_MAGNITUDE_M = 100.0

# 只由「点状」锚点计算可信 RMSE。road 是线对象，其残差含沿线任意性，
# 不该与 POI/车站的点位精度混在一起。
POINT_KINDS: tuple[str, ...] = ("station", "poi", "icon")


def is_road_fragment(p: "AnchorPair") -> bool:
    """road 类锚点的名字是不是被 OCR 截断的残片。

    完整道路名必须以通名后缀结尾（含「XX路」「XX街」「XX高速」等）；
    像「云岭东」「虹桥基」「水城」这种不以通名收尾的，基本都是残片。
    """
    if p.kind != "road":
        return False
    name = (p.name or "").strip()
    if len(name) < _MIN_FULL_ROAD_LEN:
        return True
    return not any(name.endswith(suffix) for suffix in ROAD_SUFFIXES)


def split_anchor_quality(
    pairs: Iterable["AnchorPair"],
) -> tuple[list["AnchorPair"], list["AnchorPair"]]:
    """分成 (可参与配准, 被剔除的残片)。"""
    kept: list["AnchorPair"] = []
    rejected: list["AnchorPair"] = []
    for p in pairs:
        (rejected if is_road_fragment(p) else kept).append(p)
    return kept, rejected


def _residual_vectors(reg: Any) -> list[tuple[Any, np.ndarray]]:
    """[(锚点, 预测−真值的墨卡托残差向量), ...]"""
    out: list[tuple[Any, np.ndarray]] = []
    for p in reg.inliers:
        if p.lon is None or p.lat is None:
            continue
        px, py = reg.pixel_to_mercator(p.px, p.py)
        dx, dy = lonlat_to_mercator(p.lon, p.lat)
        out.append((p, np.array([px - dx, py - dy], dtype=float)))
    return out


def _vector_stats(v: np.ndarray) -> dict[str, Any]:
    """一组残差向量的相干性 / 平均偏移 / 平均幅值。"""
    norms = np.linalg.norm(v, axis=1)
    unit = v / np.maximum(norms[:, None], 1e-9)
    coherence = float(np.linalg.norm(unit.mean(axis=0)))
    return {
        "n": int(len(v)),
        "coherence": round(coherence, 3),
        "bias_e_m": round(float(v[:, 0].mean()), 1),
        "bias_n_m": round(float(v[:, 1].mean()), 1),
        "magnitude_m": round(float(norms.mean()), 1),
    }


def systematic_bias(reg: Any) -> dict[str, Any]:
    """内点残差向量的方向一致性与平均偏移（墨卡托米）。

    ``coherence`` 是单位残差向量均值的模，取值 0~1：
      * 接近 0 → 各方向随机，是正常的标注偏移；
      * 接近 1 → 所有锚点都朝同一方向偏，通常意味着某个锚点源
        的坐标基准与底图不一致（跨库系统差），而不是配准没拟合好。
    """
    items = _residual_vectors(reg)
    if not items:
        return {"n": 0, "coherence": 0.0, "bias_e_m": 0.0,
                "bias_n_m": 0.0, "magnitude_m": 0.0, "systematic": False}
    stats = _vector_stats(np.asarray([v for _, v in items], dtype=float))
    stats["systematic"] = bool(stats["coherence"] >= _BIAS_COHERENCE
                               and stats["magnitude_m"] >= _BIAS_MAGNITUDE_M)
    return stats


def point_anchor_rmse(reg: Any) -> tuple[float, int]:
    """只用点状锚点算 RMSE 与数量（供置信度使用）。"""
    vals = [
        float(r) for p, r in zip(reg.inliers, reg.residuals)
        if p.kind in POINT_KINDS
    ]
    if not vals:
        return 0.0, 0
    return float(np.sqrt(np.mean(np.square(vals)))), len(vals)


# --- 跨坐标源一致性 -------------------------------------------------------
#
# 词典是多年累积的：同一地名可能同时存着高德、天地图、OSM 与用户确认的
# 坐标。在线侧的 CRS 已统一换算到 GCJ-02，但**历史条目**不一定。
# 同名跨源候选若相距几百米，说明至少有一套基准是错的；此时不能让
# RANSAC 靠「候选多」刷票，而应整体降权，把决定权交给几何一致性。
_CROSS_SOURCE_SEP_M = 200.0
_CROSS_SOURCE_PENALTY = 0.75
_SOURCE_SPREAD_WARN_M = 200.0


def apply_source_consistency(
    pairs: Iterable["AnchorPair"],
    max_sep_m: float = _CROSS_SOURCE_SEP_M,
    penalty: float = _CROSS_SOURCE_PENALTY,
) -> tuple[list["AnchorPair"], dict[str, Any]]:
    """同名跨源候选坐标不一致时整体降权，并返回冲突报告。

    只改 ``match_quality``（影响候选排序 / 代表选择），不改坐标、
    不删候选。返回新列表，避免污染调用方持有的对象。
    """
    items_all = list(pairs)
    by_name: dict[str, list["AnchorPair"]] = defaultdict(list)
    for p in items_all:
        by_name[p.name or ""].append(p)

    conflict: dict[str, float] = {}
    for name, items in by_name.items():
        sources = {p.source or "unknown" for p in items}
        if len(sources) < 2 or len(items) < 2:
            continue
        sep = 0.0
        for a, b in combinations(items, 2):
            if (a.source or "") == (b.source or ""):
                continue
            if None in (a.lon, a.lat, b.lon, b.lat):
                continue
            sep = max(sep, haversine(a.lon, a.lat, b.lon, b.lat))
        if sep > max_sep_m:
            conflict[name] = round(sep, 1)

    out = [
        replace(p, match_quality=float(p.match_quality) * penalty)
        if p.name in conflict else p
        for p in items_all
    ]
    report = {
        "n_names": len(by_name),
        "n_conflicted": len(conflict),
        "conflicted": conflict,
        "penalty": penalty,
        "max_sep_m": max_sep_m,
    }
    return out, report


# --- 跨源冲突「硬剔除」 -------------------------------------------------------
#
# ``apply_source_consistency`` 只降权、留待几何一致性裁决（见上方与测试
# ``test_cross_source_conflict_is_downweighted_not_deleted``）。但实测在
# 「仅 3 个内点」的退化情形里，RANSAC 会选错那个冲突候选，把尺度撑大数倍
# （22 m/px vs 真值 ~2 m/px）：1-2km 的短轨迹被拉伸成 18km，RMSE 1800m。
# 几何裁决在退化情形下失效，所以冲突严重时必须**从源头剔除**低可靠源的候选，
# 只保留最可靠源的坐标，杜绝「基准打架」引发的尺度崩塌。
#
# 仅作用于**同一地名**出现多源冲突的情形；不同地名各自单源、或多源坐标彼此
# 一致（< max_sep_m）的情形完全不受影响，因此不会误伤「同名异地地点」这类
# 合法多候选。
_SOURCE_RELIABILITY: dict[str, int] = {
    "user": 50, "verified": 50,
    "amap": 40, "tmap": 40, "baidu": 40, "tianditu": 40,
    "lexicon": 30,
    "osm": 10,
}


def _source_rank(source: str) -> int:
    return _SOURCE_RELIABILITY.get((source or "").lower(), 0)


def dedupe_cross_source_candidates(
    pairs: Iterable["AnchorPair"],
    max_sep_m: float = _CROSS_SOURCE_SEP_M,
) -> tuple[list["AnchorPair"], dict[str, Any]]:
    """同名锚点跨源坐标冲突时，只保留最可靠源的候选。

    返回 (保留的候选, 报告)。不改变坐标、不降权——与 ``apply_source_consistency``
    分工：它负责「同存但降权」，本函数负责「冲突严重时剔除非优选源」。

    可靠性排序：用户核验 > 在线源(amap/tmap/baidu/tianditu) > 本地词典 >
    OSM 冷启动。同档时取 ``match_quality`` 高者。OSM 冷启动坐标系统性偏差
    最大，是跨源冲突的主要来源，优先剔除它。
    """
    items = list(pairs)
    idx_of = {id(p): i for i, p in enumerate(items)}
    by_name: dict[str, list["AnchorPair"]] = defaultdict(list)
    for p in items:
        by_name[p.name or ""].append(p)

    drop_idx: set[int] = set()
    conflicted: dict[str, dict[str, Any]] = {}
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        sources = {p.source or "unknown" for p in group}
        if len(sources) < 2:
            continue
        sep = 0.0
        for a, b in combinations(group, 2):
            if (a.source or "") == (b.source or ""):
                continue
            if None in (a.lon, a.lat, b.lon, b.lat):
                continue
            sep = max(sep, haversine(a.lon, a.lat, b.lon, b.lat))
        if sep <= max_sep_m:
            continue  # 多源一致，留待几何挑选
        # 冲突：选最可靠源（同档取 match_quality 高者）
        def _pref(src: str) -> tuple[int, float]:
            best = max(
                (p for p in group if (p.source or "") == src),
                key=lambda p: float(p.match_quality),
            )
            return (_source_rank(src), float(best.match_quality))

        preferred = max(sources, key=_pref)
        dropped_srcs: list[str] = []
        for p in group:
            if (p.source or "") != preferred:
                drop_idx.add(idx_of[id(p)])
                s = p.source or "unknown"
                if s not in dropped_srcs:
                    dropped_srcs.append(s)
        conflicted[name] = {
            "sep_m": round(sep, 1),
            "preferred_source": preferred,
            "dropped_sources": dropped_srcs,
        }

    kept = [p for i, p in enumerate(items) if i not in drop_idx]
    report = {
        "n_names": len(by_name),
        "n_conflicted_dropped": len(conflicted),
        "dropped_count": len(drop_idx),
        "conflicted": conflicted,
        "max_sep_m": max_sep_m,
    }
    return kept, report


def source_bias_report(reg: Any) -> dict[str, Any]:
    """按坐标源分组统计内点残差，识别「混用基准」的系统性偏差。

    ``spread_m`` 是各源平均残差向量中心之间的最大距离。若内点混用
    多个源且 spread 超过阈值，说明这些源的坐标基准彼此不一致，
    此时**不要**把偏差当成配准误差去平移，而应统一锚点来源。
    """
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for p, vec in _residual_vectors(reg):
        groups[p.source or "unknown"].append(vec)

    by_source: dict[str, Any] = {}
    for src, vecs in groups.items():
        by_source[src] = _vector_stats(np.asarray(vecs, dtype=float))

    spread = 0.0
    for a, b in combinations(list(by_source.values()), 2):
        spread = max(spread, float(np.hypot(
            a["bias_e_m"] - b["bias_e_m"],
            a["bias_n_m"] - b["bias_n_m"])))
    dominant = max(by_source, key=lambda s: by_source[s]["n"]) if by_source else ""
    mixed = len(by_source) >= 2
    return {
        "by_source": by_source,
        "mixed": mixed,
        "dominant": dominant,
        "spread_m": round(spread, 1),
        "warning": bool(mixed and spread >= _SOURCE_SPREAD_WARN_M),
    }
