"""轨迹模板（P0）：概览图独立通道、模板沉淀、同类但不完全一样的形状匹配。

不启动真实服务、不触发 OCR / 在线地理编码：接口层用进程内 TestClient，
拼接与流水线换成桩件；形状判定全部是纯函数。

覆盖「同类但不完全一样」的四种情形：不同缩放平移、反向往返、只截一段、
以及**必须拒识**的两种：不同路线、只截一小块碎片。
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.pipeline import PipelineResult
from app.route_template import (
    N_SAMPLES,
    TemplateStore,
    match_shape,
    normalize_shape,
    resample_by_arc,
    shape_distance,
    signature_distance,
)


def _curve(n: int = 120, a: float = 12.0, b: float = 5.0,
           amp: float = 100.0):
    return [[float(i * 10),
             float(amp * math.sin(i / a) + 40 * math.cos(i / b))]
            for i in range(n)]


def _other_curve(n: int = 120):
    return [[float(i * 10),
             float(200 * math.cos(i / 7) + 30 * math.sin(i / 3))]
            for i in range(n)]


def _third_curve(n: int = 120):
    return [[float(i * 10),
             float(60 * math.sin(i / 20) + 150 * math.cos(i / 2.5))]
            for i in range(n)]


def _geo(n: int = 120):
    return [[121.30 + i * 0.00008,
             31.20 + 0.00004 * math.sin(i / 12)] for i in range(n)]


def _scaled(pts, k: float = 1.7, dx: float = 50.0, dy: float = -30.0):
    return [[x * k + dx, y * k + dy] for x, y in pts]


@pytest.fixture
def store(tmp_path):
    return TemplateStore(tmp_path / "route_templates.json")


def _seed(store, pts=None, **kw):
    pts = pts if pts is not None else _curve()
    args = dict(px_polyline=pts, geo_polyline=_geo(len(pts)),
                confidence=0.82, city="上海")
    args.update(kw)
    return store.remember(**args)


# --- 折线归一化 ---------------------------------------------------------

def test_resample_by_arc_is_even_and_keeps_endpoints():
    rs = resample_by_arc(_curve(), N_SAMPLES)
    assert rs is not None and rs.shape == (N_SAMPLES, 2)
    assert np.allclose(rs[0], [0.0, 40.0])
    assert np.allclose(rs[-1], [1190.0, _curve()[-1][1]])
    seg = np.linalg.norm(np.diff(rs, axis=0), axis=1)
    assert seg.max() / max(seg.min(), 1e-9) < 1.05


def test_resample_rejects_degenerate_input():
    assert resample_by_arc([[1, 1]]) is None
    assert resample_by_arc([[1, 1], [1, 1], [1, 1]]) is None


def test_normalize_shape_is_scale_and_translation_invariant():
    p = _curve()
    a = normalize_shape(resample_by_arc(p))
    b = normalize_shape(resample_by_arc(_scaled(p)))
    assert a is not None and b is not None
    assert shape_distance(a, b) < 1e-6


# --- 同类但不完全一样 ---------------------------------------------------

def test_same_shape_matches(store):
    _seed(store)
    hits = store.find_similar(_curve())
    assert len(hits) == 1
    assert hits[0]["reason"] == "ok"
    assert hits[0]["coverage"] == 1.0
    assert hits[0]["residual"] < 0.005


def test_different_zoom_and_offset_matches(store):
    _seed(store)
    hits = store.find_similar(_scaled(_curve(), k=2.4, dx=-300, dy=90))
    assert len(hits) == 1 and hits[0]["reversed"] is False


def test_reversed_route_matches(store):
    _seed(store)
    hits = store.find_similar(_curve()[::-1])
    assert len(hits) == 1
    assert hits[0]["reversed"] is True
    assert hits[0]["signature"] < 0.05


def test_partial_crop_matches_with_coverage(store):
    """只截中间 62.5% —— 这是「同类但不完全一样」的核心场景。"""
    _seed(store)
    hits = store.find_similar(_curve()[20:95])
    assert len(hits) == 1
    assert 0.60 <= hits[0]["coverage"] < 1.0
    assert hits[0]["span"] != [0, N_SAMPLES]


def test_different_route_is_rejected(store):
    _seed(store)
    assert store.find_similar(_other_curve()) == []
    m = match_shape(_other_curve(), store.templates[0])
    assert m.accepted is False
    assert m.reason in ("direction_mismatch", "shape_mismatch", "low_coverage")


def test_tiny_fragment_is_rejected(store):
    """只截 15% 的碎片即使某段形状碰巧相似，覆盖不足也必须拒识。"""
    _seed(store)
    assert store.find_similar(_curve()[40:58]) == []


def test_order_matters_not_only_point_cloud():
    """打乱点序后点集不变，但有序方向签名必须显著变差。"""
    p = _curve()
    a = normalize_shape(resample_by_arc(p))
    shuffled = p[::2] + p[1::2]
    b = normalize_shape(resample_by_arc(shuffled))
    assert a is not None and b is not None
    assert signature_distance(a, b) > 0.45


# --- 沉淀裁决 / 去重 ----------------------------------------------------

def test_remember_writes_and_reloads(store, tmp_path):
    out = _seed(store)
    assert out["written"] is True
    again = TemplateStore(store.path)
    assert again.stats()["templates"] == 1
    assert len(again.find_similar(_curve())) == 1
    assert store.path.exists()


def test_remember_rejects_low_confidence(store):
    assert _seed(store, confidence=0.20)["reason"] == "low_confidence"
    assert store.stats()["templates"] == 0


def test_remember_rejects_inconsistent_and_low_information(store):
    assert _seed(store, self_consistent=False)["reason"] == "self_inconsistent"
    assert _seed(store, input_ok=False)["reason"] == "low_information_input"
    assert store.stats()["templates"] == 0


def test_verified_bypasses_auto_gates(store):
    args = dict(px_polyline=_curve(), geo_polyline=_geo(),
                confidence=0.05, self_consistent=False, input_ok=False,
                verified=True)
    out = store.remember(**args)
    assert out["written"] is True
    assert store.templates[0]["verified"] is True


def test_remember_rejects_bad_coordinates_and_short_tracks(store):
    assert _seed(store, geo_polyline=[[999, 999]] * 120)["reason"] in (
        "invalid_coordinate", "not_enough_points")
    assert _seed(store, px_polyline=[[0, 0], [1, 1]])["reason"] == "not_enough_points"


def test_duplicate_shape_increments_hits_instead_of_new_template(store):
    assert _seed(store)["written"] is True
    out = _seed(store, pts=_scaled(_curve()), confidence=0.9)
    assert out["reason"] == "deduped"
    assert out["templates"] == 1
    assert store.stats()["hits"] == 2
    assert store.templates[0]["confidence"] == 0.9


def test_distinct_shapes_are_kept_and_cap_applies(tmp_path):
    s = TemplateStore(tmp_path / "t.json", max_templates=2)
    for pts in (_curve(), _other_curve(), _third_curve()):
        s.remember(px_polyline=pts, geo_polyline=_geo(len(pts)),
                   confidence=0.8, city="上海")
    assert s.stats()["templates"] == 2
    assert len(s.find_similar(_curve(), limit=5)) >= 1


def test_corrupted_file_degrades_safely(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{broken json", encoding="utf-8")
    s = TemplateStore(p)
    assert s.stats()["templates"] == 0
    assert s.find_similar(_curve()) == []


def test_find_similar_requires_verified_when_asked(store):
    _seed(store)
    assert store.find_similar(_curve(), require_verified=True) == []


def test_find_similar_honors_city_hint(store):
    _seed(store, city="上海")
    assert store.find_similar(_curve(), city_hint="北京") == []
    assert len(store.find_similar(_curve(), city_hint="上海")) == 1


# --- 服务端接入 ---------------------------------------------------------

@pytest.fixture
def server_env(monkeypatch, tmp_path):
    from app import server

    small = np.zeros((10, 10, 3), dtype=np.uint8)
    px = _curve()
    gj = {"features": [{"geometry": {"coordinates": _geo(len(px))},
                        "properties": {}}]}
    res = PipelineResult(
        geojson=gj,
        geojson_gcj02=json.loads(json.dumps(gj)),
        confidence=0.82,
        track_points=len(px),
        track_polyline_px=px,
        diagnostics={"input_quality": {"usable": True, "level": "ok"}},
        warnings=[],
        image_size=(1200, 400),
    )
    seen: dict = {}

    def fake_stitch(imgs):
        seen["n_stitch_inputs"] = len(imgs)
        return SimpleNamespace(
            ok=True, canvas=small,
            to_dict=lambda: {"ok": True, "notes": [], "used": [0, 1]},
        )

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(server, "imread_unicode", lambda _p: small)
    monkeypatch.setattr(server, "stitch_images", fake_stitch)
    monkeypatch.setattr(server._pipeline, "run", lambda *a, **kw: res)
    monkeypatch.setattr(server, "_templates",
                        TemplateStore(tmp_path / "templates.json"))
    return server, seen


def _jpg(name):
    return (name, b"fake", "image/jpeg")


def test_overview_is_not_stitched_and_template_is_captured(server_env):
    server, seen = server_env
    files = [
        ("files", _jpg("s0.jpg")),
        ("files", _jpg("s1.jpg")),
        ("overview", _jpg("overview.jpg")),
    ]
    with TestClient(server.app) as client:
        resp = client.post(
            "/api/recognize_multi", files=files,
            data={"city": "上海", "refine": "off"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # 概览图进了独立通道，没有混进 SIFT 拼接的 imgs
    assert seen["n_stitch_inputs"] == 2
    assert data["n_shots"] == 2
    assert data["stitch"]["overview_present"] is True
    assert "overview_url" in data
    assert data["template_capture"]["written"] is True
    assert server._templates.stats()["templates"] == 1


def test_multi_without_overview_keeps_old_behavior(server_env):
    server, seen = server_env
    files = [("files", _jpg("s0.jpg")), ("files", _jpg("s1.jpg"))]
    with TestClient(server.app) as client:
        resp = client.post(
            "/api/recognize_multi", files=files,
            data={"city": "上海", "refine": "off"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert seen["n_stitch_inputs"] == 2
    assert "overview_present" not in data["stitch"]
    assert "overview_url" not in data
    assert data["template_capture"]["written"] is True


def test_single_image_also_captures_template(server_env):
    server, _ = server_env
    with TestClient(server.app) as client:
        resp = client.post(
            "/api/recognize", files={"file": _jpg("s0.jpg")},
            data={"city": "上海", "refine": "off"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["template_capture"]["written"] is True


def test_health_reports_template_stats(server_env):
    server, _ = server_env
    with TestClient(server.app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "templates" in resp.json()
    assert resp.json()["templates"]["templates"] == 0
