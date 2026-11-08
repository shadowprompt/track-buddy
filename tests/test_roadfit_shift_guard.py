"""路网精修的「位移 / 轨迹尺度」护栏。

为什么不针对某张截图
--------------------
曾经只用一个绝对搜索半径（±800 米）+「贴路距离变小」就采纳精修。
路网是稀疏的平行线族：把一条几百米的短轨迹整体搬几百米，一样能压在
**另一条**平行道路上，贴路距离从几百米降到几米，所有判据都满足，
位置却彻底错了。这类过拟合与城市、截图、分辨率都无关，只与
「位移相对轨迹自身长度的比例」有关，所以用合成几何做回归测试。
"""

import numpy as np

from app.geo.coords import gcj02_to_wgs84, mercator_to_lonlat
from app.roadfit import refine_with_roads

# 任意原点：只影响绝对经纬度，不影响相对位移判定
_OX, _OY = 1.0e7, 3.0e6


class _LinearReg:
    """1 像素 = 1 米的线性配准，便于用米直接构造合成轨迹。"""

    def pixel_to_mercator(self, x: float, y: float) -> tuple[float, float]:
        return (_OX + float(x), _OY + float(y))


def _trajectory(length_m: float, n: int = 60) -> np.ndarray:
    xs = np.linspace(0.0, length_m, n)
    return np.column_stack([xs, np.zeros_like(xs)])


def _parallel_road(offset_m: float, length_m: float = 4000.0) -> list:
    """造一条与轨迹平行、在北侧 offset_m 米的道路（WGS-84 经纬度）。

    RoadNetwork 内部会把 WGS-84 转成 GCJ-02，所以这里反向转换一次，
    保证最终落回目标墨卡托坐标。
    """
    xs = np.linspace(-length_m / 4, length_m * 3 / 4, 240)
    pts = []
    for x in xs:
        lon_gcj, lat_gcj = mercator_to_lonlat(_OX + x, _OY + offset_m)
        pts.append(gcj02_to_wgs84(lon_gcj, lat_gcj))
    return [("合成道路", pts)]


def _shift(fit) -> float:
    return float(np.hypot(fit.shift_e_m, fit.shift_n_m))


def test_short_track_is_not_moved_to_another_parallel_road() -> None:
    """短轨迹：620 米外的平行路不能被当成「贴路改善」。"""
    traj = _trajectory(700.0)
    roads = _parallel_road(620.0)

    # 先看「不限幅」时确实会搬过去——证明这不是合成数据造不出来
    loose = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0, shift_span_ratio=10.0,
    )
    assert _shift(loose) > 500.0, (
        f"对照失效：不限幅时应出现大幅位移，实际 {_shift(loose):.0f}m"
    )

    guarded = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0, shift_span_ratio=0.15,
    )
    limit = 0.15 * 700.0
    assert _shift(guarded) <= limit + 5.0, (
        f"短轨迹被整体搬走 {_shift(guarded):.0f}m，超过尺度上限 {limit:.0f}m"
    )
    assert not guarded.ok, (
        "位移不足却仍采纳精修，说明过拟合解没有被拦住："
        f"{guarded.notes}"
    )


def test_long_track_still_accepts_a_reasonable_shift() -> None:
    """长轨迹：276 米的整体平移仍应正常采纳（护栏不能一刀切禁掉）。"""
    traj = _trajectory(4000.0)
    roads = _parallel_road(276.0)
    fit = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0, shift_span_ratio=0.15,
    )
    assert fit.ok, f"正常的贴路平移被误杀：{fit.notes}"
    assert abs(fit.shift_n_m - 276.0) < 50.0, (
        f"平移量异常：北 {fit.shift_n_m:.0f}m"
    )
    assert _shift(fit) <= 0.15 * 4000.0 + 5.0


def test_ratio_is_configurable_and_scale_independent() -> None:
    """比例调到更小，长轨迹的大位移同样要被拦；判据与分辨率无关。"""
    traj = _trajectory(4000.0)
    roads = _parallel_road(276.0)
    fit = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0, shift_span_ratio=0.01,
    )
    assert _shift(fit) <= 0.01 * 4000.0 + 5.0
    assert not fit.ok
