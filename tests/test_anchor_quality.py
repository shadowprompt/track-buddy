"""锚点质量：OCR 残片不进 RANSAC，可信 RMSE 只由点状锚点计算。

守住的真实 bug：road 类锚点有 800m 米制容差，截断路名（OCR 残片）
以 600m 级残差混进内点，RMSE 与置信度一起失真。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.anchor_quality import (
    POINT_KINDS,
    is_road_fragment,
    point_anchor_rmse,
    split_anchor_quality,
    systematic_bias,
)
from app.eval import run_synthetic_benchmark, synthetic_case
from app.geo.coords import mercator_to_lonlat
from app.register import AnchorPair, register


def _p(name, kind="road", lon=121.0, lat=31.0, px=0.0, py=0.0):
    return AnchorPair(name=name, px=px, py=py, weight=1.0, kind=kind,
                      lon=lon, lat=lat, match_quality=1.0)


@pytest.mark.parametrize("name,fragment", [
    ("片段0东", True),
    ("云岭东", True),
    ("虹桥基", True),
    ("水城", True),
    ("测试路", False),
    ("建国大街", False),
    ("外环高速", False),
    ("", True),
])
def test_road_fragment_detection(name, fragment):
    assert is_road_fragment(_p(name, "road")) is fragment


def test_non_road_anchors_are_never_rejected():
    for kind in ("poi", "station", "district", "transit", "icon"):
        assert is_road_fragment(_p("随便什么", kind)) is False


def test_split_keeps_fragments_traceable():
    pairs = [_p("基准路", "poi"), _p("片段东", "road"), _p("完整路", "road")]
    kept, rejected = split_anchor_quality(pairs)
    assert [p.name for p in kept] == ["基准路", "完整路"]
    assert [p.name for p in rejected] == ["片段东"]


class _FakeReg:
    """只提供 systematic_bias / point_anchor_rmse 需要的最小接口。"""

    def __init__(self, shifts, kinds=None, residuals=None):
        kinds = kinds or ["poi"] * len(shifts)
        residuals = residuals or [0.0] * len(shifts)
        self.residuals = list(residuals)
        self.inliers = []
        for i, (dx, dy) in enumerate(shifts):
            lon, lat = mercator_to_lonlat(1000.0 + i * 10 + dx, 2000.0 + dy)
            self.inliers.append(_p(f"p{i}", kinds[i], lon, lat,
                                   px=1000.0 + i * 10, py=2000.0))

    def pixel_to_mercator(self, px, py):
        return (px, py)


def test_systematic_bias_detects_consistent_direction():
    reg = _FakeReg([(150.0, 0.0)] * 6)
    b = systematic_bias(reg)
    assert b["n"] == 6
    assert b["coherence"] > 0.95
    assert b["systematic"] is True
    assert b["bias_e_m"] < -100  # 预测 - 真值


def test_systematic_bias_ignores_random_directions():
    reg = _FakeReg([(150.0, 0.0), (-150.0, 0.0), (0.0, 150.0), (0.0, -150.0)])
    b = systematic_bias(reg)
    assert b["coherence"] < 0.2
    assert b["systematic"] is False


def test_point_anchor_rmse_excludes_road():
    reg = _FakeReg(
        [(0, 0)] * 4,
        kinds=["poi", "poi", "road", "district"],
        residuals=[10.0, 20.0, 600.0, 900.0],
    )
    rmse, n = point_anchor_rmse(reg)
    assert n == 2
    assert rmse == pytest.approx(np.sqrt((100 + 400) / 2))
    assert "road" not in POINT_KINDS


def test_register_rejects_fragments_and_improves_precision():
    case = synthetic_case(seed=3)
    reg = register(case.pairs)
    assert reg.ok
    assert len(reg.fragment_rejected) >= 1
    names = {p.name for p in reg.inliers}
    rejected = {p.name for p in reg.fragment_rejected}
    assert not (names & rejected), "残片不应出现在内点里"
    assert all(p in [q.name for q in reg.outliers] for p in rejected)
    assert "coherence" in reg.anchor_bias
    assert reg.n_reliable >= 2
    # 可信 RMSE 不应被 road 的大残差撑大
    assert reg.rmse_reliable_m <= reg.rmse_m + 1e-6


def test_benchmark_precision_after_cleanup():
    """合成基准的内点精确率应高于清洗前基线（P50 0.778）。"""
    report = run_synthetic_benchmark(n_cases=6, seed0=0)
    assert report["aggregate"]["precision"]["p50"] >= 0.95
    assert report["aggregate"]["recall"]["p50"] >= 0.95
    assert report["aggregate"]["scale_rel_error"]["p50"] < 0.05
