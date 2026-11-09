"""词典防毒：低置信 / 不自洽 / 低信息量结果不得反哺词典。

同时锁定 pipeline.run() 的「会话包裹」契约：自动写入必须等最终
置信度出来才落盘，用户显式纠正不受影响。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import AppConfig
from app.geocode import Lexicon
from app.lexicon_guard import (
    AUTO_WEIGHT_CAP, MIN_AUTO_CONFIDENCE, evaluate_write,
)
from app.pipeline import PipelineResult, RidePipeline, _result_is_self_consistent

LON, LAT = 121.40, 31.20


def _lex(tmp_path: Path) -> Lexicon:
    return Lexicon(tmp_path / "lexicon.json")


def _auto(name: str = "示例地标", **kw):
    base = dict(name=name, lon=LON, lat=LAT, city="上海",
                source="tianditu", boost=0.5)
    base.update(kw)
    return base


def test_user_correction_bypasses_every_auto_gate(tmp_path):
    lex = _lex(tmp_path)
    d = lex.remember(**_auto(confidence=0.0, self_consistent=False,
                             input_ok=False, source="user", boost=2.0))
    assert d.allowed and d.reason == "user_verified"
    assert d.weight_cap is None
    e = lex.lookup_exact("示例地标")[0]
    assert e["weight"] == 2.0
    assert e["verified"] is True
    assert lex.stats()["verified_points"] == 1


@pytest.mark.parametrize("kwargs,reason", [
    ({"confidence": 0.1}, "low_confidence"),
    ({"self_consistent": False}, "self_inconsistent"),
    ({"input_ok": False}, "low_information_input"),
    ({"name": "3公里内免费"}, "name_is_ui_text"),
    ({"name": "甲"}, "name_not_place"),
    ({"lon": 999.0}, "invalid_coordinate"),
    ({"lat": float("nan")}, "invalid_value"),
])
def test_auto_writes_are_rejected_by_external_evidence(kwargs, reason):
    d = evaluate_write(
        key=Lexicon.normalize(kwargs.pop("name", "示例地标")),
        lon=kwargs.pop("lon", LON), lat=kwargs.pop("lat", LAT),
        source="tianditu", boost=0.5, entries=[],
        confidence=kwargs.pop("confidence", 0.8),
        self_consistent=kwargs.pop("self_consistent", True),
        input_ok=kwargs.pop("input_ok", True),
    )
    assert d.allowed is False
    assert d.reason == reason


def test_cross_source_near_conflict_blocked_but_far_allowed():
    existing = [{"lon": LON, "lat": LAT, "weight": 5.0,
                 "source": "user", "verified": True}]
    near = evaluate_write(
        key="示例地标", lon=LON + 0.01, lat=LAT, source="osm",
        boost=0.2, entries=existing, confidence=0.9)
    assert near.allowed is False
    assert near.reason == "cross_source_conflict"
    far = evaluate_write(
        key="示例地标", lon=LON + 1.0, lat=LAT, source="osm",
        boost=0.2, entries=existing, confidence=0.9)
    assert far.allowed is True  # 同名异地，本来就该保留多个候选
    same_source = evaluate_write(
        key="示例地标", lon=LON + 0.01, lat=LAT, source="user",
        boost=0.2, entries=existing, confidence=0.9)
    assert same_source.allowed is True  # 用户来源不受冲突护栏限制


def test_unverified_weight_is_capped_and_user_upgrade_unlocks(tmp_path):
    lex = _lex(tmp_path)
    for _ in range(12):
        lex.remember(**_auto(confidence=0.9))
    e = lex.lookup_exact("示例地标")[0]
    assert e["weight"] == pytest.approx(AUTO_WEIGHT_CAP)
    assert e["verified"] is False
    assert e["last_confidence"] == 0.9
    assert lex.lookup("示例地标")[0].quality < 1.0

    lex.remember(**_auto(source="user", boost=2.0, confidence=0.9))
    upgraded = lex.lookup_exact("示例地标")[0]
    assert upgraded["verified"] is True
    assert upgraded["weight"] > AUTO_WEIGHT_CAP
    assert upgraded["source"] == "user"


def test_no_confidence_still_cold_starts_but_slowly(tmp_path):
    lex = _lex(tmp_path)
    d = lex.remember(**_auto())  # 非流水线调用，无最终置信度
    assert d.allowed and d.reason == "no_confidence_capped"
    assert lex.lookup_exact("示例地标")[0]["weight"] < 0.5
    assert lex.lookup_exact("示例地标")[0]["weight"] <= AUTO_WEIGHT_CAP


def test_capture_queues_auto_but_user_writes_immediately(tmp_path):
    lex = _lex(tmp_path)
    path = lex.path
    lex.begin_capture()
    assert lex.remember(**_auto(name="自动条目")).reason == "queued"
    assert lex.remember(**_auto(name="人工纠正", source="user",
                                boost=2.0)).reason == "user_verified"
    assert "自动条目" not in json.loads(path.read_text(encoding="utf-8"))
    assert "人工纠正" in json.loads(path.read_text(encoding="utf-8"))

    report = lex.flush_capture(confidence=0.1)
    assert report["pending"] == 1
    assert report["written"] == 0
    assert report["reasons"] == {"low_confidence": 1}
    assert len(lex.lookup_exact("自动条目")) == 0

    lex.begin_capture()
    lex.remember(**_auto(name="另一个条目"))
    assert lex.discard_capture() == 1
    assert len(lex.lookup_exact("另一个条目")) == 0


def test_self_consistency_helper_uses_only_independent_evidence():
    assert _result_is_self_consistent({})
    assert _result_is_self_consistent({
        "input_quality": {"level": "warn", "usable": True},
        "scale_check": {"available": True, "consistent": True},
        "anchor_bias": {"systematic": False},
    })
    assert not _result_is_self_consistent({
        "input_quality": {"level": "reject"}})
    assert not _result_is_self_consistent({
        "scale_check": {"consistent": False}})
    assert not _result_is_self_consistent({
        "anchor_bias": {"systematic": True, "coherence": 0.95}})
    assert not _result_is_self_consistent({
        "zoom": {"zoom_too_low": True, "position_ok": False}})
    # 位置已被路名修正时低缩放不该一票否决
    assert _result_is_self_consistent({
        "zoom": {"zoom_too_low": True, "position_ok": True}})


def test_pipeline_run_flushes_capture_with_final_confidence(tmp_path, monkeypatch):
    cfg = AppConfig(lexicon_path=tmp_path / "lexicon.json")
    pipe = RidePipeline(cfg)
    state = {"confidence": 0.1}

    def fake_impl(self, img, **kwargs):
        self.lexicon.remember(
            "示例地标", LON, LAT, city="上海", source="tianditu", boost=0.5)
        return PipelineResult(
            geojson={}, confidence=state["confidence"],
            diagnostics={
                "input_quality": {"level": "ok", "usable": True},
                "scale_check": {"consistent": True},
                "anchor_bias": {"systematic": False},
            },
        )

    monkeypatch.setattr(RidePipeline, "_run_impl", fake_impl)
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    low = pipe.run(img)
    assert low.diagnostics["lexicon_write_guard"]["reasons"] == {
        "low_confidence": 1}
    assert len(pipe.lexicon.lookup_exact("示例地标")) == 0
    assert low.diagnostics["confidence_calibration"]["enabled"] is False

    state["confidence"] = 0.8
    high = pipe.run(img)
    assert high.diagnostics["lexicon_write_guard"]["written"] == 1
    assert len(pipe.lexicon.lookup_exact("示例地标")) == 1
    assert "lexicon" in high.diagnostics


def test_pipeline_rejects_low_information_result_even_if_confident(
        tmp_path, monkeypatch):
    cfg = AppConfig(lexicon_path=tmp_path / "lexicon.json")
    pipe = RidePipeline(cfg)

    def fake_impl(self, img, **kwargs):
        self.lexicon.remember(
            "示例地标", LON, LAT, city="上海", source="osm", boost=0.2)
        return PipelineResult(
            geojson={}, confidence=0.9,
            diagnostics={"input_quality": {"level": "reject", "usable": False}},
        )

    monkeypatch.setattr(RidePipeline, "_run_impl", fake_impl)
    res = pipe.run(np.zeros((8, 8, 3), dtype=np.uint8))
    guard = res.diagnostics["lexicon_write_guard"]
    assert guard["input_ok"] is False
    assert guard["self_consistent"] is False
    assert guard["reasons"] == {"self_inconsistent": 1}
    assert len(pipe.lexicon.lookup_exact("示例地标")) == 0


def test_pipeline_discards_pending_writes_on_exception(tmp_path, monkeypatch):
    cfg = AppConfig(lexicon_path=tmp_path / "lexicon.json")
    pipe = RidePipeline(cfg)

    def boom(self, img, **kwargs):
        self.lexicon.remember(
            "示例地标", LON, LAT, city="上海", source="osm", boost=0.2)
        raise RuntimeError("识别中断")

    monkeypatch.setattr(RidePipeline, "_run_impl", boom)
    with pytest.raises(RuntimeError):
        pipe.run(np.zeros((8, 8, 3), dtype=np.uint8))
    assert pipe.lexicon._capture is None
    assert len(pipe.lexicon.lookup_exact("示例地标")) == 0
