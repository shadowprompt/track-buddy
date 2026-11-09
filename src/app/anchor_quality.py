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

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from .geo.coords import lonlat_to_mercator

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


def systematic_bias(reg: Any) -> dict[str, Any]:
    """内点残差向量的方向一致性与平均偏移（墨卡托米）。

    ``coherence`` 是单位残差向量均值的模，取值 0~1：
      * 接近 0 → 各方向随机，是正常的标注偏移；
      * 接近 1 → 所有锚点都朝同一方向偏，通常意味着某个锚点源
        的坐标基准与底图不一致（跨库系统差），而不是配准没拟合好。
    """
    vecs: list[tuple[float, float]] = []
    for p in reg.inliers:
        if p.lon is None or p.lat is None:
            continue
        px, py = reg.pixel_to_mercator(p.px, p.py)
        dx, dy = lonlat_to_mercator(p.lon, p.lat)
        vecs.append((px - dx, py - dy))
    if not vecs:
        return {"n": 0, "coherence": 0.0, "bias_e_m": 0.0,
                "bias_n_m": 0.0, "magnitude_m": 0.0, "systematic": False}
    v = np.asarray(vecs, dtype=float)
    norms = np.linalg.norm(v, axis=1)
    unit = v / np.maximum(norms[:, None], 1e-9)
    coherence = float(np.linalg.norm(unit.mean(axis=0)))
    magnitude = float(norms.mean())
    return {
        "n": int(len(v)),
        "coherence": round(coherence, 3),
        "bias_e_m": round(float(v[:, 0].mean()), 1),
        "bias_n_m": round(float(v[:, 1].mean()), 1),
        "magnitude_m": round(magnitude, 1),
        "systematic": bool(coherence >= _BIAS_COHERENCE
                           and magnitude >= _BIAS_MAGNITUDE_M),
    }


def point_anchor_rmse(reg: Any) -> tuple[float, int]:
    """只用点状锚点算 RMSE 与数量（供置信度使用）。"""
    vals = [
        float(r) for p, r in zip(reg.inliers, reg.residuals)
        if p.kind in POINT_KINDS
    ]
    if not vals:
        return 0.0, 0
    return float(np.sqrt(np.mean(np.square(vals)))), len(vals)
