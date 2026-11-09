"""坐标源贯通与跨源一致性：不静默混用基准，也不静默改坐标。

守住的点：
  * 词典条目的真实来源要透传到锚点，而不是被抹成 "lexicon"；
  * 同名跨源候选相距过远时整体降权，由几何一致性挑选；
  * 按源输出残差报告，混用基准时提示「统一来源」而不是去平移。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.anchor_quality import apply_source_consistency, source_bias_report
from app.eval import synthetic_case
from app.geo.coords import mercator_to_lonlat
from app.register import AnchorPair, register

ROOT = Path(__file__).resolve().parent.parent


def _p(name, source, lon, lat, quality=1.0, px=0.0, py=0.0):
    return AnchorPair(name=name, px=px, py=py, weight=1.0, kind="poi",
                      lon=lon, lat=lat, match_quality=quality, source=source)


def test_cross_source_conflict_is_downweighted_not_deleted():
    a = _p("甲", "amap", 121.0000, 31.0000)
    b = _p("甲", "tianditu", 121.0050, 31.0000)  # 约 470 米
    c = _p("乙", "amap", 121.0010, 31.0000)
    out, report = apply_source_consistency([a, b, c])
    assert report["n_conflicted"] == 1
    assert "甲" in report["conflicted"]
    assert report["conflicted"]["甲"] > 200
    assert len(out) == 3
    assert out[0].match_quality == pytest.approx(0.75)
    assert out[1].match_quality == pytest.approx(0.75)
    assert out[2].match_quality == 1.0
    assert a.match_quality == 1.0, "不能污染调用方持有的对象"
    assert (out[0].lon, out[1].lon) == (a.lon, b.lon), "坐标不能被改"


def test_single_source_candidates_are_untouched():
    a = _p("甲", "amap", 121.0000, 31.0000)
    b = _p("甲", "amap", 121.0050, 31.0000)
    out, report = apply_source_consistency([a, b], max_sep_m=50.0)
    assert report["n_conflicted"] == 0
    assert all(p.match_quality == 1.0 for p in out)


class _SrcReg:
    def __init__(self, rows):
        self.inliers = []
        self.residuals = [0.0] * len(rows)
        for i, (src, dx, dy) in enumerate(rows):
            px, py = 1000.0 + i * 10, 2000.0
            lon, lat = mercator_to_lonlat(px + dx, py + dy)
            self.inliers.append(_p(f"p{i}", src, lon, lat, px=px, py=py))

    def pixel_to_mercator(self, px, py):
        return (px, py)


def test_source_bias_report_flags_mixed_baselines():
    reg = _SrcReg([("amap", 300.0, 0.0)] * 3
                  + [("tianditu", -300.0, 0.0)] * 3)
    rep = source_bias_report(reg)
    assert rep["mixed"] is True
    assert set(rep["by_source"]) == {"amap", "tianditu"}
    assert rep["spread_m"] > 500
    assert rep["warning"] is True
    assert rep["dominant"] in ("amap", "tianditu")


def test_source_bias_report_silent_for_single_source():
    reg = _SrcReg([("amap", 300.0, 0.0)] * 4)
    rep = source_bias_report(reg)
    assert rep["mixed"] is False
    assert rep["warning"] is False
    assert rep["by_source"]["amap"]["coherence"] > 0.95


def test_register_reports_sources_without_changing_fit():
    reg = register(synthetic_case(seed=3).pairs)
    assert reg.ok
    assert reg.source_bias["mixed"] is True
    assert "tianditu" in reg.source_bias["by_source"]
    assert "osm" in reg.source_bias["by_source"]
    assert "n_names" in reg.source_consistency
    assert all(p.source for p in reg.inliers)


def test_lexicon_result_keeps_real_source(tmp_path):
    from app.geocode import Lexicon

    path = tmp_path / "lex.json"
    path.write_text(json.dumps({"甲": [
        {"lon": 121.0, "lat": 31.0, "weight": 2.0, "count": 2,
         "source": "tianditu", "address": "", "city": "上海"},
    ]}, ensure_ascii=False), encoding="utf-8")
    results = Lexicon(path).lookup("甲")
    assert results and results[0].source == "tianditu"


def _load_checker():
    import importlib.util

    path = ROOT / "src" / "tools" / "check_lexicon_sources.py"
    spec = importlib.util.spec_from_file_location("_lexicon_source_check", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_lexicon_source_checker_and_cli(tmp_path):
    data = {
        "甲": [
            {"lon": 121.0, "lat": 31.0, "source": "amap"},
            {"lon": 121.005, "lat": 31.0, "source": "tianditu"},
        ],
        "乙": [
            {"lon": 121.0, "lat": 31.0, "source": "amap"},
            {"lon": 121.0001, "lat": 31.0, "source": "amap"},
        ],
    }
    checker = _load_checker()
    rep = checker.inspect_lexicon(data, sep_m=200.0)
    assert rep["n_names"] == 2
    assert rep["n_conflicts"] == 1
    assert rep["n_near_conflicts"] == 1
    assert rep["n_far_conflicts"] == 0
    assert rep["conflicts"][0]["name"] == "甲"
    assert rep["conflicts"][0]["category"] == "near"
    assert rep["source_counts"]["amap"] == 3
    assert rep["n_multi_source_names"] == 1

    lex = tmp_path / "lex.json"
    lex.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    cp = subprocess.run(
        [sys.executable, str(ROOT / "src" / "tools" / "check_lexicon_sources.py"),
         "--path", str(lex), "--top", "2"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert cp.returncode == 0, cp.stderr
    assert "冲突 1 组" in cp.stdout
