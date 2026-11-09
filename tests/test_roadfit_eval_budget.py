"""位移搜索的「评估点抽样」回归测试。

12 图拼接画布的轨迹约 1.1 万个像素点，粗搜 4225 个位移候选就是
4700 万次 KD 树查询。抽样只用于估计中位贴路距离，**搜索空间、护栏、
轨迹长度（位移限幅依据）都不能变**。这里用合成几何验证抽样前后
得到的平移量与护栏判定一致，避免「提速了但结果悄悄变了」。
"""

from __future__ import annotations

import numpy as np

from app.geo.coords import gcj02_to_wgs84, mercator_to_lonlat
from app.roadfit import refine_with_roads

_OX, _OY = 1.0e7, 3.0e6


class _LinearReg:
    """1 像素 = 1 米的线性配准，便于用米直接构造合成轨迹。"""

    def pixel_to_mercator(self, x: float, y: float) -> tuple[float, float]:
        return (_OX + float(x), _OY + float(y))


def _trajectory(length_m: float, n: int) -> np.ndarray:
    xs = np.linspace(0.0, length_m, n)
    return np.column_stack([xs, np.zeros_like(xs)])


def _parallel_road(offset_m: float, length_m: float = 4000.0) -> list:
    xs = np.linspace(-length_m / 4, length_m * 3 / 4, 240)
    pts = []
    for x in xs:
        lon_gcj, lat_gcj = mercator_to_lonlat(_OX + x, _OY + offset_m)
        pts.append(gcj02_to_wgs84(lon_gcj, lat_gcj))
    return [("合成道路", pts)]


def test_sampling_keeps_shift_and_distances() -> None:
    """3000 点抽到 200 点，最优平移与贴路距离基本不变。"""
    traj = _trajectory(4000.0, n=3000)
    roads = _parallel_road(100.0)
    kw = dict(max_shift_m=300.0, shift_span_ratio=0.15)

    full = refine_with_roads(_LinearReg(), traj, roads,
                             eval_budget=0, **kw)
    sampled = refine_with_roads(_LinearReg(), traj, roads,
                                eval_budget=200, **kw)

    assert full.ok and sampled.ok, (full.notes, sampled.notes)
    assert abs(full.shift_n_m - sampled.shift_n_m) < 5.0
    assert abs(full.shift_e_m - sampled.shift_e_m) < 5.0
    assert abs(full.median_before_m - sampled.median_before_m) < 1.0
    assert abs(full.median_after_m - sampled.median_after_m) < 1.0
    # 路名分段始终按**原始折线**算，抽样不能影响分段数量
    assert len(full.segments) == len(sampled.segments)


def test_budget_above_track_size_is_exact_noop() -> None:
    """轨迹点少于预算时，抽样分支不该产生任何差异。"""
    traj = _trajectory(700.0, n=60)
    roads = _parallel_road(50.0)
    kw = dict(max_shift_m=300.0, shift_span_ratio=0.15)

    a = refine_with_roads(_LinearReg(), traj, roads, eval_budget=2000, **kw)
    b = refine_with_roads(_LinearReg(), traj, roads, eval_budget=0, **kw)
    assert a.to_dict() == b.to_dict()


def test_sampling_does_not_loosen_shift_guard() -> None:
    """短轨迹 + 小预算：护栏仍按完整轨迹长度限幅，不能搬到平行路。"""
    traj = _trajectory(700.0, n=1000)
    roads = _parallel_road(620.0)

    fit = refine_with_roads(
        _LinearReg(), traj, roads,
        max_shift_m=800.0, shift_span_ratio=0.15, eval_budget=50,
    )
    assert float(np.hypot(fit.shift_e_m, fit.shift_n_m)) <= 0.15 * 700.0 + 5.0
    assert not fit.ok
