"""锚点优先于路网贴合的通用护栏测试。"""

import numpy as np

from app.geo.coords import gcj02_to_wgs84, mercator_to_lonlat
from app.roadfit import refine_with_roads

_OX, _OY = 1.0e7, 3.0e6


class _LinearReg:
    def pixel_to_mercator(self, x: float, y: float) -> tuple[float, float]:
        return (_OX + float(x), _OY + float(y))


def _trajectory(length_m: float, n: int = 60) -> np.ndarray:
    xs = np.linspace(0.0, length_m, n)
    return np.column_stack([xs, np.zeros_like(xs)])


def _parallel_road(offset_m: float, length_m: float = 4000.0) -> list:
    xs = np.linspace(-length_m / 4, length_m * 3 / 4, 240)
    pts = []
    for x in xs:
        lon_gcj, lat_gcj = mercator_to_lonlat(_OX + x, _OY + offset_m)
        pts.append(gcj02_to_wgs84(lon_gcj, lat_gcj))
    return [("合成道路", pts)]


def test_road_fit_rejects_shift_that_breaks_good_anchors():
    """已有可靠锚点时，不能只因 OSM 更近就搬到平行道路。"""
    traj = _trajectory(4000.0)
    roads = _parallel_road(276.0)
    # 当前配准的可靠锚点残差很小；道路候选需要向北 276m 才能贴合，
    # 但这会把截图与地理锚点的直接对应关系破坏掉。
    anchors = np.array([[12.0, 0.0], [-18.0, 8.0], [25.0, -6.0]])
    fit = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0,
        shift_span_ratio=0.15,
        anchor_points_m=anchors,
        anchor_guard_m=100.0,
        anchor_worsen_m=80.0,
        anchor_worsen_ratio=2.5,
    )
    assert not fit.ok
    assert np.hypot(fit.shift_e_m, fit.shift_n_m) == 0.0
    assert fit.anchor_before_m is not None
    assert fit.anchor_after_m is not None
    assert fit.anchor_after_m > fit.anchor_before_m
    assert any("锚点一致性" in note for note in fit.notes)


def test_road_fit_can_shift_when_anchors_show_systematic_error():
    """锚点本身有明显系统误差时，合理的路网修正仍可采纳。"""
    traj = _trajectory(4000.0)
    roads = _parallel_road(276.0)
    # 这些残差意味着当前位置整体偏南约 276m；向北平移会让锚点更接近。
    anchors = np.array([[0.0, 276.0], [15.0, 270.0], [-12.0, 281.0]])
    fit = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0,
        shift_span_ratio=0.15,
        anchor_points_m=anchors,
        anchor_guard_m=100.0,
        anchor_worsen_m=80.0,
        anchor_worsen_ratio=2.5,
    )
    assert fit.ok, fit.notes
    assert abs(fit.shift_n_m - 276.0) < 50.0
    assert fit.anchor_after_m is not None
    assert fit.anchor_before_m is not None
    assert fit.anchor_after_m < fit.anchor_before_m


def test_anchor_guard_is_optional_for_existing_callers():
    """旧调用方不传锚点时，原有尺度护栏行为保持不变。"""
    traj = _trajectory(700.0)
    roads = _parallel_road(620.0)
    fit = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0,
        shift_span_ratio=0.15,
    )
    assert not fit.ok
    assert np.hypot(fit.shift_e_m, fit.shift_n_m) <= 0.15 * 700.0 + 5.0
