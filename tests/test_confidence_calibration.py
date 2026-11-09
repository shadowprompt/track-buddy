"""置信度校准：保序、收缩、安全降级与 CLI 冒烟。

最重要的一条契约：**没有真值样本时，置信度必须逐值不变**。
项目里所有置信度断言、前端展示、词典防毒门槛都建立在它之上。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app import calibration as cal

ROOT = Path(__file__).resolve().parent.parent


def _good_metrics() -> dict:
    return {"start_m": 60.0, "end_m": 90.0,
            "frechet_m": 150.0, "length_rel_error": 0.08}


def test_label_from_metrics_uses_external_errors_only():
    assert cal.label_from_metrics(_good_metrics()) == 1
    for key, bad in (("start_m", 900.0), ("end_m", 900.0),
                     ("frechet_m", 900.0), ("length_rel_error", 0.9)):
        m = _good_metrics()
        m[key] = bad
        assert cal.label_from_metrics(m) == 0, key
    assert cal.label_from_metrics(None) == 0
    m = _good_metrics()
    m["start_m"] = float("inf")
    assert cal.label_from_metrics(m) == 0
    m = _good_metrics()
    del m["end_m"]
    assert cal.label_from_metrics(m) == 0
    # 自洽指标不参与标签：贴路距离再好，没有真值误差也算没测到
    assert cal.label_from_metrics({"median_after_m": 3.0}) == 0


def test_isotonic_is_monotone_even_with_noisy_samples():
    samples = [(0.20, 0), (0.25, 1), (0.30, 0),
               (0.70, 1), (0.75, 0), (0.90, 1)]
    knots = cal.fit_isotonic(samples)
    assert knots
    rates = [k["rate"] for k in knots]
    assert rates == sorted(rates)
    assert sum(k["n"] for k in knots) == len(samples)


def test_fit_calibration_requires_minimum_truth_samples():
    assert cal.fit_calibration([(0.5, 1)] * 3) is None
    assert cal.fit_calibration([]) is None
    model = cal.fit_calibration([(0.2, 0)] * 5 + [(0.9, 1)] * 5)
    assert model is not None
    assert model["n_samples"] == 10
    assert model["n_positive"] == 5
    assert model["version"] == cal.MODEL_VERSION


def test_apply_shrinks_not_overwrites_and_stays_monotone(tmp_path):
    model = cal.fit_calibration([(0.9, 0)] * 10)
    assert model is not None
    p = tmp_path / "model.json"
    cal.save_calibration(model, p)
    loaded = cal.load_calibration(p)
    assert loaded is not None
    # 经验率是 0，但 10 个样本只能把 0.9 拉低一部分，不能直接打成 0
    out = loaded.apply(0.9)
    assert 0.4 < out < 0.9
    vals = [loaded.apply(i / 20) for i in range(21)]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 0.99 for v in vals)


def test_missing_invalid_thin_or_non_monotone_model_is_ignored(tmp_path):
    assert cal.load_calibration(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    assert cal.load_calibration(bad) is None

    thin = {"version": cal.MODEL_VERSION, "n_samples": 2,
            "min_samples": 8,
            "knots": [{"x": 0.5, "rate": 0.2, "n": 2}]}
    thin_p = tmp_path / "thin.json"
    thin_p.write_text(json.dumps(thin), encoding="utf-8")
    assert cal.load_calibration(thin_p) is None

    broken = dict(thin)
    broken["n_samples"] = 10
    broken["knots"] = [{"x": 0.2, "rate": 0.9, "n": 5},
                       {"x": 0.8, "rate": 0.1, "n": 5}]
    broken_p = tmp_path / "nonmonotone.json"
    broken_p.write_text(json.dumps(broken), encoding="utf-8")
    assert cal.load_calibration(broken_p) is None


def test_no_truth_means_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(cal.CALIBRATION_PATH_ENV,
                       str(tmp_path / "none.json"))
    cal._STORES.clear()
    try:
        for raw in (0.0, 0.2, 0.735, 0.99, 1.0):
            out, status = cal.apply_calibration(raw)
            assert out == raw
            assert status["status"] == "uncalibrated"
            assert status["enabled"] is False
    finally:
        cal._STORES.clear()


def test_calibration_model_is_applied_when_present(tmp_path, monkeypatch):
    p = tmp_path / "model.json"
    model = cal.fit_calibration([(0.9, 0)] * 8 + [(0.2, 1)] * 4)
    assert model is not None
    cal.save_calibration(model, p)
    monkeypatch.setenv(cal.CALIBRATION_PATH_ENV, str(p))
    cal._STORES.clear()
    try:
        out, status = cal.apply_calibration(0.9)
        assert status["enabled"] is True
        assert status["n_samples"] == 12
        assert out < 0.9
    finally:
        cal._STORES.clear()


def _write_case(d: Path, idx: int, correct: bool) -> tuple[str, str]:
    line = [(121.40, 31.20), (121.41, 31.21), (121.42, 31.22)]
    coords = (line if correct else
              [(lo + 0.01, la) for lo, la in line])  # 约 1 km 东偏
    pred = {
        "confidence": 0.3 + 0.07 * idx,
        "geojson": {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "LineString", "coordinates": coords}}]},
    }
    truth = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "LineString", "coordinates": line}}]}
    pp, tp = d / f"pred_{idx}.json", d / f"truth_{idx}.json"
    pp.write_text(json.dumps(pred), encoding="utf-8")
    tp.write_text(json.dumps(truth), encoding="utf-8")
    return str(pp), str(tp)


def test_cli_builds_model_from_truth_cases(tmp_path):
    for i in range(8):
        _write_case(tmp_path, i, i < 5)
    cases = [{"pred": str(tmp_path / f"pred_{i}.json"),
              "truth": str(tmp_path / f"truth_{i}.json")} for i in range(8)]
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    out = tmp_path / "calib.json"
    script = ROOT / "src" / "tools" / "calibrate_confidence.py"
    cp = subprocess.run(
        [sys.executable, str(script), "--cases", str(manifest),
         "--out", str(out), "--json"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert cp.returncode == 0, cp.stderr
    report = json.loads(cp.stdout)
    assert report["n_samples"] == 8
    assert report["n_positive"] == 5
    assert report["model_written"] is True
    assert cal.load_calibration(out) is not None


def test_cli_refuses_to_write_model_without_enough_truth(tmp_path):
    _write_case(tmp_path, 0, True)
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": [
        {"pred": str(tmp_path / "pred_0.json"),
         "truth": str(tmp_path / "truth_0.json")}]}), encoding="utf-8")
    out = tmp_path / "calib.json"
    script = ROOT / "src" / "tools" / "calibrate_confidence.py"
    cp = subprocess.run(
        [sys.executable, str(script), "--cases", str(manifest),
         "--out", str(out), "--json"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert cp.returncode == 1
    report = json.loads(cp.stdout)
    assert report["model_written"] is False
    assert not out.exists()
