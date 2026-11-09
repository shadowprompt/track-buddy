"""HMM 地图匹配：能贴回道路，但绝不替用户把直线「补」成一条路。

守住的三道护栏：低信息量直线输入拒绝、吸附距离过大拒绝、
匹配后长度/方向剧变拒绝。默认开关关闭，主结果不受影响。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import CONFIG
from app.eval import synthetic_case
from app.geo.coords import gcj02_to_wgs84
from app.mapmatch import hmm_match
from app.register import register
from app.roadfit import RoadNetwork

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "src" / "app" / "server.py"
PIPELINE = ROOT / "src" / "app" / "pipeline.py"
HTML = ROOT / "src" / "web" / "index.html"


def _case(offset_m=25.0, seed=0):
    c = synthetic_case(seed=seed)
    reg = register(c.pairs)
    assert reg.ok
    road_gcj = [reg.pixel_to_lonlat(float(px), float(py))
                for px, py in c.track_px]
    road_wgs = [gcj02_to_wgs84(lo, la) for lo, la in road_gcj]
    net = RoadNetwork([("测试路", road_wgs)])
    noisy = c.track_px + np.array([offset_m / reg.scale, 0.0])
    return reg, noisy, net


def test_match_snaps_noisy_track_back_to_road():
    reg, noisy, net = _case(offset_m=25.0)
    m = hmm_match(reg, noisy, net)
    assert m.ok, m.reason
    assert len(m.coordinates_gcj02) == len(noisy)
    assert len(m.coordinates_px) == len(noisy)
    assert 10 < m.mean_snap_m < 45
    assert m.roads == ["测试路"]
    assert m.length_before_m > 0 and m.length_after_m > 0
    assert 0.7 < m.length_after_m / m.length_before_m < 1.4


def test_match_reduces_distance_to_road():
    reg, noisy, net = _case(offset_m=30.0)
    m = hmm_match(reg, noisy, net)
    assert m.ok
    before = np.asarray([reg.pixel_to_mercator(float(a), float(b))
                         for a, b in noisy])
    chosen = np.asarray([reg.pixel_to_mercator(float(a), float(b))
                         for a, b in m.coordinates_px])
    # 匹配后的像素点重新投影回墨卡托，应更贴近原真值轨迹
    truth = np.asarray([reg.pixel_to_mercator(float(a), float(b))
                        for a, b in noisy - np.array([30.0 / reg.scale, 0.0])])
    assert np.mean(np.linalg.norm(chosen - truth, axis=1)) < \
        np.mean(np.linalg.norm(before - truth, axis=1))


def test_low_shape_info_is_rejected():
    reg, noisy, net = _case()
    m = hmm_match(reg, noisy, net, low_shape_info=True)
    assert not m.ok
    assert "直线" in m.reason


def test_far_from_network_is_rejected():
    reg, noisy, net = _case(offset_m=300.0)
    m = hmm_match(reg, noisy, net, max_snap_m=60.0)
    assert not m.ok
    assert "吸附距离" in m.reason
    assert m.mean_snap_m > 60


def test_empty_network_and_short_track():
    reg, noisy, _ = _case()
    assert not hmm_match(reg, noisy, RoadNetwork([])).ok
    assert not hmm_match(reg, noisy[:2], RoadNetwork([])).ok


def test_to_dict_keys_and_default_switch(monkeypatch):
    monkeypatch.delenv("MAP_MATCH_ENABLED", raising=False)
    reg, noisy, net = _case()
    d = hmm_match(reg, noisy, net).to_dict()
    assert set(d) >= {"ok", "reason", "crs", "coordinates",
                      "coordinates_px", "roads", "mean_snap_m", "n_points"}
    assert d["crs"] == "GCJ-02"
    assert CONFIG.map_match_enabled is False


def test_switch_can_be_enabled_by_env(monkeypatch):
    monkeypatch.setenv("MAP_MATCH_ENABLED", "1")
    assert CONFIG.map_match_enabled is True


def test_payload_and_pipeline_wiring():
    server = SERVER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")
    assert server.count('"map_matched": result.map_matched') == 2
    assert "def _map_match_optional(" in pipeline
    assert "map_matched=map_matched" in pipeline
    assert "low_shape_info=low" in pipeline
    assert "dg.map_match && dg.map_match.ok" in html
