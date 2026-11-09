"""准确率评测指标与合成基准的回归测试。

守住的约定：
  * 同样的折线必须得 0 误差；整体平移必须被 endpoint / Fréchet 同时抓到；
  * Fréchet 对**形状**敏感，不能退化成「平均最近点距离」；
  * 合成基准可复现（同 seed 同结果），这是 A/B 的前提；
  * 真值文件支持 GPX / GeoJSON / 坐标数组三种常见来源。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from app.eval import (
    discrete_frechet_m,
    endpoint_errors,
    evaluate_files,
    evaluate_tracks,
    inlier_scores,
    load_prediction,
    load_truth,
    polyline_length_m,
    registration_errors,
    run_synthetic_benchmark,
    synthetic_case,
)
from app.register import AnchorPair, register

ROOT = Path(__file__).resolve().parent.parent


def _shift(points, dlon=0.0, dlat=0.0):
    return [(a + dlon, b + dlat) for a, b in points]


def test_identical_tracks_have_zero_error():
    line = [(121.0 + 0.0001 * i, 31.0 + 0.0002 * i) for i in range(30)]
    m = evaluate_tracks(line, line)
    assert m["start_m"] < 1e-6
    assert m["end_m"] < 1e-6
    assert m["frechet_m"] < 1e-6
    assert m["length_rel_error"] < 1e-9
    assert m["n_points"] == 30


def test_shift_is_captured_by_both_endpoint_and_frechet():
    line = [(121.0 + 0.0001 * i, 31.0 + 0.0002 * i) for i in range(30)]
    moved = _shift(line, dlon=0.001)  # 约 95 米
    err = endpoint_errors(moved, line)
    assert 80 < err["start_m"] < 110
    assert 80 < err["end_m"] < 110
    assert 80 < discrete_frechet_m(moved, line) < 110


def test_frechet_is_shape_sensitive():
    """中间一段横移：起终点不变，Fréchet 必须显著增大。"""
    line = [(121.0 + 0.0001 * i, 31.0) for i in range(40)]
    bent = list(line)
    for i in range(12, 28):
        bent[i] = (bent[i][0], bent[i][1] + 0.002)  # 约 220 米
    assert endpoint_errors(bent, line)["max_m"] < 1e-6
    assert discrete_frechet_m(bent, line) > 150


def test_length_and_degenerate_inputs():
    line = [(121.0, 31.0), (121.001, 31.0), (121.002, 31.0)]
    assert polyline_length_m(line) == pytest.approx(191.0, abs=5.0)
    assert polyline_length_m([]) == 0.0
    assert endpoint_errors([], line)["start_m"] == math.inf
    assert discrete_frechet_m([], line) == math.inf
    assert evaluate_tracks([(121.0, 31.0)], line)["n_points"] == 1


def test_registration_errors_use_known_transform():
    case = synthetic_case(seed=7)
    reg = register(case.pairs)
    assert reg.ok
    m = registration_errors(reg, case.matrix, case.translation, case.probe)
    assert m["scale_rel_error"] < 0.05
    assert m["center_error_m"] < 60.0
    assert m["matrix_max_diff"] < 0.5


def test_inlier_scores_count_names_not_candidates():
    inliers = [
        AnchorPair(name="甲", px=0, py=0, weight=1, kind="poi"),
        AnchorPair(name="甲", px=0, py=0, weight=1, kind="poi"),
        AnchorPair(name="乙", px=1, py=1, weight=1, kind="poi"),
    ]
    s = inlier_scores(inliers, {"甲", "丙"})
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert s["n_inlier_names"] == 2


def test_synthetic_case_is_reproducible():
    a = synthetic_case(seed=42)
    b = synthetic_case(seed=42)
    assert [(p.name, p.lon, p.lat) for p in a.pairs] == [
        (p.name, p.lon, p.lat) for p in b.pairs]
    assert np.allclose(a.matrix, b.matrix)
    assert a.truth_track == b.truth_track


def test_synthetic_benchmark_runs_and_aggregates():
    report = run_synthetic_benchmark(n_cases=3, seed0=0)
    assert report["n_cases"] == 3
    assert report["n_ok"] >= 2
    assert set(report["aggregate"]) >= {
        "center_error_m", "scale_rel_error", "precision", "recall",
        "frechet_m", "length_rel_error",
    }
    assert report["aggregate"]["scale_rel_error"]["p50"] < 0.05
    assert len(report["cases"]) == 3


@pytest.mark.parametrize("kind", ["geojson", "array", "gpx"])
def test_load_truth_formats(tmp_path, kind):
    line = [(121.0, 31.0), (121.001, 31.0), (121.002, 31.0)]
    if kind == "geojson":
        payload = {"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "LineString", "coordinates": [list(p) for p in line]},
        }]}
        text = json.dumps(payload)
        suffix = ".geojson"
    elif kind == "array":
        text = json.dumps([list(p) for p in line])
        suffix = ".json"
    else:
        pts = "".join(f'<trkpt lat="{b}" lon="{a}"></trkpt>' for a, b in line)
        text = f'<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk>{pts}</trk></gpx>'
        suffix = ".gpx"
    p = tmp_path / f"truth{suffix}"
    p.write_text(text, encoding="utf-8")
    assert load_truth(p) == line


def test_evaluate_files_with_recognition_json(tmp_path):
    line = [(121.0, 31.0), (121.001, 31.0), (121.002, 31.0)]
    pred = tmp_path / "pred.json"
    truth = tmp_path / "truth.json"
    pred.write_text(json.dumps({"geojson": {"features": [{
        "geometry": {"coordinates": [list(p) for p in line]}}]}}),
        encoding="utf-8")
    truth.write_text(json.dumps([list(p) for p in line]), encoding="utf-8")
    m = evaluate_files(pred, truth)
    assert m["start_m"] < 1e-6
    assert m["frechet_m"] < 1e-6
    assert load_prediction(pred) == line


def test_cli_smoke():
    """CLI 至少能在最小样本数下跑通并产出聚合结果。"""
    script = ROOT / "src" / "tools" / "eval_accuracy.py"
    import subprocess

    cp = subprocess.run(
        [sys.executable, str(script), "--synthetic", "1", "--json"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert cp.returncode == 0, cp.stderr
    data = json.loads(cp.stdout)
    assert data["n_cases"] == 1
    assert "aggregate" in data
