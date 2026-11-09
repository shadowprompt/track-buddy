"""被护栏否决的路网精修：保留候选位移并换算成截图像素偏移。

用户需要「把未采纳的精修路线也画到截图上、换个颜色」才能自己判断
护栏拒得对不对。旧代码在三道护栏里把 ``shift_e_m/n_m`` 直接归零，
搜索出的最优解随之丢失，前端无从画起。这里锁住四件事：

1. 否决时输出位移仍为 0（结果语义不变），但候选解完整保留；
2. 采纳成功时 ``to_dict()`` 不新增任何字段（旧契约不变）；
3. 三道护栏中的任意一道否决都要保留候选；
4. 米制候选位移按配准矩阵的逆换算成像素位移，退化矩阵安全降级。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app import roadfit as roadfit_mod
from app.config import AppConfig
from app.pipeline import RidePipeline, _candidate_shift_px
from app.register import Registration
from app.roadfit import RoadFit, refine_with_roads

_OX, _OY = 1.35e7, 3.65e6


class _LinearReg:
    """1 像素 = 1 米的线性配准，便于用米直接构造合成轨迹。"""

    def pixel_to_mercator(self, x: float, y: float) -> tuple[float, float]:
        return (_OX + float(x), _OY + float(y))


def _trajectory(length_m: float, n: int = 60) -> np.ndarray:
    xs = np.linspace(0.0, length_m, n)
    return np.column_stack([xs, np.zeros_like(xs)])


def _parallel_road(offset_m: float, length_m: float = 4000.0) -> list:
    from app.geo.coords import gcj02_to_wgs84, mercator_to_lonlat

    xs = np.linspace(-length_m / 4, length_m * 3 / 4, 240)
    pts = []
    for x in xs:
        lon_gcj, lat_gcj = mercator_to_lonlat(_OX + x, _OY + offset_m)
        pts.append(gcj02_to_wgs84(lon_gcj, lat_gcj))
    return [("合成道路", pts)]


def _anchor_rejected(**kwargs):
    """锚点一致性护栏：可靠锚点 + 北侧 276m 的平行道路。"""
    return refine_with_roads(
        _LinearReg(), _trajectory(4000.0), _parallel_road(276.0),
        max_shift_m=800.0, shift_span_ratio=0.15,
        anchor_points_m=np.array([[12.0, 0.0], [-18.0, 8.0], [25.0, -6.0]]),
        anchor_guard_m=100.0, anchor_worsen_m=80.0,
        anchor_worsen_ratio=2.5, **kwargs,
    )


def _reg(matrix=None, translation=None) -> Registration:
    return Registration(
        ok=True,
        matrix=np.array(matrix if matrix is not None
                        else [[2.0, 0.0], [0.0, -2.0]], dtype=float),
        translation=np.array(translation if translation is not None
                             else [_OX, _OY], dtype=float),
    )


def test_rejected_fit_keeps_candidate_but_zero_output_shift():
    """否决后输出位移仍是 0，候选位移与候选贴路距离都留着。"""
    fit = _anchor_rejected()
    assert not fit.ok
    d = fit.to_dict()
    assert d["shift_e_m"] == 0.0 and d["shift_n_m"] == 0.0
    assert d["median_after_m"] == d["median_before_m"]
    assert d["candidate_shift_n_m"] > 100.0
    assert d["candidate_median_after_m"] is not None
    assert any("锚点一致性" in n for n in d["notes"])


def test_accepted_fit_has_no_candidate_keys():
    """采纳成功时键集必须与改动前完全一致。"""
    fit = refine_with_roads(
        _LinearReg(), _trajectory(4000.0), _parallel_road(276.0),
        max_shift_m=800.0, shift_span_ratio=0.15,
        anchor_points_m=np.array([[0.0, 276.0], [15.0, 270.0],
                                  [-12.0, 281.0]]),
        anchor_guard_m=100.0, anchor_worsen_m=80.0,
        anchor_worsen_ratio=2.5,
    )
    assert fit.ok
    assert not any(k.startswith("candidate_") for k in fit.to_dict())


def test_shift_limit_guard_also_keeps_candidate():
    """第二道护栏（位移限幅）同样要保留候选，不能只修锚点那一支。"""
    fit = refine_with_roads(
        _LinearReg(), _trajectory(700.0), _parallel_road(620.0),
        max_shift_m=800.0, shift_span_ratio=0.15,
    )
    assert not fit.ok
    d = fit.to_dict()
    assert "candidate_shift_e_m" in d
    assert "candidate_shift_n_m" in d
    assert d["shift_e_m"] == 0.0 and d["shift_n_m"] == 0.0


def test_to_dict_outputs_candidate_only_when_present():
    plain = RoadFit(ok=True, shift_e_m=10.0).to_dict()
    assert "candidate_shift_e_m" not in plain
    marked = RoadFit(ok=False, candidate_shift_e_m=10.0,
                     candidate_shift_n_m=-4.0,
                     candidate_median_after_m=7.0).to_dict()
    assert marked["candidate_shift_e_m"] == 10.0
    assert marked["candidate_shift_n_m"] == -4.0
    assert marked["candidate_median_after_m"] == 7.0


def test_candidate_shift_px_uses_inverse_registration_matrix():
    """M = [[2,0],[0,-2]]，M⁻¹·[100,-50] = [50,25] 像素。"""
    out = _candidate_shift_px(
        _reg(), {"candidate_shift_e_m": 100.0,
                 "candidate_shift_n_m": -50.0},
    )
    assert out == {"candidate_shift_px": [50.0, 25.0]}


def test_candidate_shift_px_handles_missing_or_degenerate_matrix():
    fit = {"candidate_shift_e_m": 100.0, "candidate_shift_n_m": -50.0}
    assert _candidate_shift_px(_reg(), {"ok": True}) == {}
    assert _candidate_shift_px(_reg([[0.0, 0.0], [0.0, 0.0]]), fit) == {}
    assert _candidate_shift_px(SimpleNamespace(), fit) == {}
    assert _candidate_shift_px(_reg(), {"candidate_shift_e_m": 1.0}) == {}


def test_road_refine_attaches_pixel_shift_to_result(monkeypatch):
    """接线检查：_road_refine 返回给 server 的 dict 里必须带着像素位移。"""
    rejected = RoadFit(
        ok=False, median_before_m=30.0, median_after_m=30.0,
        candidate_shift_e_m=100.0, candidate_shift_n_m=-50.0,
        candidate_median_after_m=5.0, notes=["测试否决"],
    )
    monkeypatch.setattr(roadfit_mod, "fetch_roads", lambda *a, **k: [])
    monkeypatch.setattr(roadfit_mod, "refine_with_roads",
                        lambda *a, **k: rejected)
    pipe = RidePipeline(AppConfig())
    reg = _reg()
    track = SimpleNamespace(
        polyline=np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]]),
        bbox=(0.0, 0.0, 200.0, 10.0),
    )
    out, _ = pipe._road_refine(reg, track)
    assert out["ok"] is False
    assert out["shift_e_m"] == 0.0
    assert out["candidate_shift_px"] == [50.0, 25.0]
    # 否决不能偷偷改配准矩阵
    assert np.allclose(reg.translation, [_OX, _OY])
