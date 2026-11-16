"""独立校验：截图里程 / 外部路由 / 输入信息量。

守住的点：外部证据**缺失时绝不惩罚**，只有明确不一致才轻罚；
低信息量输入（起终点直线）必须在上传/识别初期就给出重截建议。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.input_quality import (
    assess_input_quality,
    extract_distance_km,
    route_validation_factor,
    scale_consistency,
)
from app.pipeline import RidePipeline


class _Text:
    def __init__(self, text):
        self.text = text


def test_extract_distance_from_text_items_and_strings():
    assert extract_distance_km([_Text("骑行 12.6公里"), _Text("用时 48 分钟")]) == 12.6
    assert extract_distance_km(["总里程 8.2 km"]) == 8.2
    assert extract_distance_km(["3号线", "地铁 2号线"]) is None
    assert extract_distance_km([]) is None


def test_extract_distance_uses_median_and_filters_unreasonable():
    texts = [_Text("3公里内免费"), _Text("本次 10.5公里"), _Text("累计 999公里")]
    assert extract_distance_km(texts) == 10.5
    assert extract_distance_km([_Text("0.01公里"), _Text("9999 km")]) is None


def test_extract_distance_ignores_track_milestone_labels():
    """轨迹里程牌（纯整数+KM 的无上下文标签）不是骑行总里程。

    回归：2026-11-16 高德 8 图实测，「3KM/2KM/1KM」三个里程牌取中位
    = 2.0，与几何里程 3.65km 比值 1.826 触发误报、置信度被压到 0.2。
    """
    # 只有里程牌 → 视为「截图未识别到里程」，跳过尺度校验而不是误报
    assert extract_distance_km(["3KM", "2KM", "1KM"]) is None
    assert extract_distance_km([_Text("3KM"), _Text("2 KM")]) is None
    # 带上下文或小数的读数不受影响
    assert extract_distance_km(["骑行 3.68千米"]) == 3.68
    assert extract_distance_km([_Text("2.0km")]) == pytest.approx(2.0)
    # 里程牌混在有上下文的真值里：真值仍被读出
    assert extract_distance_km(["3KM", "2KM", _Text("里程 4.2公里")]) == 4.2


def test_scale_consistency():
    ok = scale_consistency(10_500.0, 10.5)
    assert ok["available"] and ok["consistent"]
    assert ok["ratio"] == pytest.approx(1.0)
    bad = scale_consistency(52_000.0, 10.5)
    assert bad["available"] and not bad["consistent"]
    assert bad["ratio"] > 1.7
    missing = scale_consistency(10_500.0, None)
    assert not missing["available"] and missing["consistent"] is True


def test_route_factor_never_punishes_missing_result():
    assert route_validation_factor(None)["factor"] == 1.0
    assert route_validation_factor({"osrm_verify": None})["factor"] == 1.0
    bad = route_validation_factor({"external_verify": {
        "engine": "valhalla", "hits": [], "match_ratio": 0.1,
        "order_consistency": 0.2,
    }})
    assert bad["available"] and bad["factor"] < 1.0
    good = route_validation_factor({"external_verify": {
        "engine": "valhalla", "hits": ["甲路", "乙路"], "match_ratio": 0.8,
        "order_consistency": 1.0,
    }})
    assert good["factor"] > 1.0


def test_input_quality_levels():
    ok = assess_input_quality(image_size=(1260, 2720), straightness=0.4,
                              bbox_px=(10, 10, 400, 300),
                              has_road_names=True, n_ocr_texts=20)
    assert ok["level"] == "ok" and ok["usable"]
    low = assess_input_quality(image_size=(1260, 2720), straightness=0.83,
                               bbox_px=(269, 1313, 710, 95),
                               has_road_names=False, n_ocr_texts=8)
    assert low["level"] == "reject" and not low["usable"]
    assert low["suggestions"]
    small = assess_input_quality(image_size=(400, 300), n_ocr_texts=2)
    assert small["level"] == "warn" and len(small["reasons"]) >= 2


def _stub_pipeline():
    p = RidePipeline.__new__(RidePipeline)
    p._self_check = lambda reg, pairs: (0.0, 0)
    return p


def _stub_reg():
    return SimpleNamespace(
        rmse_reliable_m=10.0, n_reliable=3, rmse_m=10.0, scale=2.0,
        inliers=[SimpleNamespace(name=f"a{i}", kind="poi") for i in range(3)],
        outliers=[], residuals=[10.0] * 3,
    )


def test_confidence_applies_independent_checks():
    p = _stub_pipeline()
    reg, track = _stub_reg(), SimpleNamespace(confidence=1.0)
    args = (reg, track, [], [], None, None, None)
    base = p._score_confidence(*args)
    bad = p._score_confidence(*args, scale_check={"consistent": False})
    good = p._score_confidence(
        *args, route_check={"available": True, "factor": 1.03})
    assert bad < base
    assert good > base
    assert p._score_confidence(
        *args, scale_check={"available": False, "consistent": True}) == base
