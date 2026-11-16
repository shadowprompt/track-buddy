"""输入源画像的**请求级入参控制**（2026-11-14）。

三块契约：

1. ``active_source_profile`` 上下文：``CONFIG.source`` / ``CONFIG.vision`` /
   ``CONFIG.source_profile_id`` 在 ``with`` 块内跟随显式画像，退出还原；
2. 模板库按画像隔离：``remember``/``register_pair`` 写入画像 id，
   ``locate``/``find_similar`` 只召回同画像模板，历史数据（无 profile
   字段）视为默认画像 meituan；
3. 服务端：``source`` 表单入参 → 请求期画像生效（diagnostics 透传）、
   未知 id → 400 并列出可用值、不传 → 环境级默认。

全部桩件化：不启动真实 OCR / 在线地理编码，运行期目录与模板库指向
tmp_path。
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.server as S
from app.config import CONFIG
from app.route_template import SHAPE_DOMAIN_OVERVIEW, TemplateStore
from app.source_profile import (
    PROFILES,
    active_source_profile,
    current_active_profile,
    resolve_source_profile,
)


# --- 1. 请求级画像上下文 -------------------------------------------------

def test_active_profile_drives_config_source() -> None:
    default_id = CONFIG.source_profile_id
    meituan = resolve_source_profile("meituan")
    with active_source_profile(meituan):
        assert current_active_profile() is meituan
        assert CONFIG.source_profile_id == "meituan"
        assert CONFIG.source is meituan
        # vision 属性跟随画像：同一画像身份相等
        assert CONFIG.vision is meituan.vision
    # 退出还原：回落环境默认
    assert current_active_profile() is None
    assert CONFIG.source_profile_id == default_id


def test_active_profile_none_clears_override() -> None:
    meituan = resolve_source_profile("meituan")
    with active_source_profile(meituan):
        with active_source_profile(None):
            assert current_active_profile() is None
            assert CONFIG.source.id == CONFIG.source_profile_id
        assert current_active_profile() is meituan


def test_unknown_profile_id_raises_with_available_list() -> None:
    with pytest.raises(ValueError) as ei:
        resolve_source_profile("keep")
    assert "meituan" in str(ei.value)


def test_vision_identity_for_meituan_matches_legacy_default() -> None:
    """美团的画像参数与模块化前的字段默认值完全一致（零回归锚点）。"""
    v = CONFIG.vision
    assert v.track_max_value == 200
    assert v.track_max_saturation == 90
    assert v.ransac_inlier_m == 200.0


# --- 2. 模板库按画像隔离 -------------------------------------------------

def _bent(n: int = 160) -> list[list[float]]:
    return [[float(400 * math.sin(i / 9) + 250 * math.sin(i / 23)),
             float(320 * math.cos(i / 7) + 180 * math.sin(i / 3))]
            for i in range(n)]


def _render_detail(px: list[list[float]]) -> list[list[float]]:
    out = [[x * 1.6 + 30, y * 1.25 - 20] for x, y in px]
    return [[x + 1.5 * math.sin(i / 3), y + 1.5 * math.cos(i / 4)]
            for i, (x, y) in enumerate(out)]


def _geo(px: list[list[float]]) -> list[list[float]]:
    a = np.asarray(px, dtype=float)
    x0, y0 = a.min(axis=0)
    w = max(1.0, float(a[:, 0].max() - a[:, 0].min()))
    h = max(1.0, float(a[:, 1].max() - a[:, 1].min()))
    return [[121.20 + (x - x0) / w * 0.20, 31.30 - (y - y0) / h * 0.10]
            for x, y in px]


def _register(store: TemplateStore, px, profile: str = "") -> dict:
    detail = _render_detail(px)
    return store.register_pair(
        overview_px=px, detail_px=detail, geo_polyline=_geo(detail),
        confidence=0.8, city="上海", shot_id="probe", profile=profile)


def test_template_profile_recorded_and_isolated(tmp_path) -> None:
    px = _bent()
    store = TemplateStore(tmp_path / "t.json")
    out = _register(store, px, profile="meituan")
    assert out["written"] is True
    assert store.templates[0]["profile"] == "meituan"

    # 同画像可召回；其它画像（未来注册 amap）查不到——即使形状完全一样
    hit_meituan = store.locate(px[::-1], city_hint="上海",
                               shape_domain=SHAPE_DOMAIN_OVERVIEW,
                               profile="meituan")
    assert hit_meituan is not None
    assert hit_meituan["template"]["id"] == out["id"]

    hit_other = store.locate(px[::-1], city_hint="上海",
                             shape_domain=SHAPE_DOMAIN_OVERVIEW,
                             profile="amap")
    assert hit_other is None


def test_template_without_profile_treated_as_default(tmp_path) -> None:
    """历史模板（2026-11-14 之前沉淀，无 profile 字段）视为 meituan。"""
    px = _bent()
    store = TemplateStore(tmp_path / "t.json")
    _register(store, px, profile="")
    # 模拟历史数据：内存里去掉 profile 字段后落盘，再重新加载
    for t in store._data.get("templates", []):
        t.pop("profile", None)
    store.save()
    store2 = TemplateStore(tmp_path / "t.json")
    assert all("profile" not in t for t in store2.templates)

    hit = store2.locate(px[::-1], city_hint="上海",
                        shape_domain=SHAPE_DOMAIN_OVERVIEW,
                        profile="meituan")
    assert hit is not None
    other = store2.locate(px[::-1], city_hint="上海",
                          shape_domain=SHAPE_DOMAIN_OVERVIEW,
                          profile="amap")
    assert other is None


def test_find_similar_respects_profile(tmp_path) -> None:
    px = _bent()
    store = TemplateStore(tmp_path / "t.json")
    _register(store, px, profile="meituan")
    assert store.find_similar(px, profile="meituan")
    assert not store.find_similar(px, profile="amap")


def test_dedup_only_within_same_profile(tmp_path) -> None:
    """同形状不同画像各自成模板，不合并——地理真值互不相关。"""
    px = _bent()
    store = TemplateStore(tmp_path / "t.json")
    assert _register(store, px, profile="meituan")["written"] is True
    out2 = _register(store, px, profile="amap")
    assert out2["written"] is True, out2
    assert len(store.templates) == 2


# --- 3. 服务端 source 入参 ----------------------------------------------

@pytest.fixture
def server_stub(monkeypatch, tmp_path):
    px = _bent()
    detail = _render_detail(px)
    geo = _geo(detail)
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "LineString", "coordinates": geo}}]}
    fake = SimpleNamespace(
        confidence=0.8, track_points=len(px), geojson=gj, geojson_gcj02=gj,
        crs="WGS-84", distance_m=15000.0, anchors_used=[], anchors_rejected=[],
        road_attribution={}, road_fit={"ok": True, "shift_m": 0.0},
        road_name_constraints=[], road_name_fit={}, alternatives=[],
        diagnostics={"input_quality": {"usable": True}}, warnings=[],
        elapsed_ms=1, track_polyline_px=detail, map_matched=False,
        image_size=(100, 100, 3))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    store = TemplateStore(tmp_path / "route_templates.json")
    monkeypatch.setattr(S, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(S, "DATA_DIR", tmp_path)
    monkeypatch.setattr(S, "_templates", store)
    # 桩件 run 在**请求上下文内**执行：捕获当时生效的画像 id，验证
    # source 入参经 ContextVar 真正传到了识别链路（真实流程由
    # pipeline._base_diagnostics 把它写进 diagnostics.source）。
    captured: dict[str, str] = {}

    def fake_run(*a, **k):
        captured["profile_id"] = CONFIG.source_profile_id
        return fake

    monkeypatch.setattr(S._pipeline, "run", fake_run)
    monkeypatch.setattr(S, "extract_track",
                        lambda img, cfg: _track(px[::-1]))
    return store, fake, captured


def _track(px):
    from app.track import TrackResult
    return TrackResult(
        polyline=np.asarray(px, float), start_marker=(0, 0),
        end_marker=(1, 1), mask=np.zeros((10, 10), np.uint8),
        bbox=(0, 0, 10, 10), confidence=0.95, notes=[])


def _jpg() -> bytes:
    _, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 255, np.uint8))
    return buf.tobytes()


def test_recognize_rejects_unknown_source(server_stub) -> None:
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg"))],
        data={"source": "keep"})
    assert r.status_code == 400
    assert "keep" in r.json()["detail"] and "meituan" in r.json()["detail"]


def test_recognize_without_source_uses_env_default(server_stub) -> None:
    store, fake, captured = server_stub
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    # 请求上下文内生效的画像 = 环境级默认（未传 source）
    assert captured["profile_id"] == CONFIG.source_profile_id


def test_recognize_with_explicit_source_applies_profile(server_stub) -> None:
    store, fake, captured = server_stub
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg"))],
        data={"source": "meituan"})
    assert r.status_code == 200, r.text
    assert captured["profile_id"] == "meituan"


def test_recognize_multi_accepts_source_param(server_stub) -> None:
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize_multi",
        files=[("files", ("d1.jpg", _jpg(), "image/jpeg")),
               ("files", ("d2.jpg", _jpg(), "image/jpeg"))],
        data={"source": "unknown_app"})
    assert r.status_code == 400
    assert "unknown_app" in r.json()["detail"]


def test_health_lists_available_profiles(server_stub) -> None:
    client = TestClient(S.app)
    d = client.get("/api/health").json()
    sp = d["source_profile"]
    assert sp["available"], "health 必须下发画像清单供前端渲染选择器"
    ids = {p["id"] for p in sp["available"]}
    assert "meituan" in ids
