"""P2：双上传入口 + 配对注册 + 形状域过滤。

不启动服务、不读本地截图、不触发真实 OCR / 在线地理编码：合成折线做
几何 fixture，服务端用进程内 TestClient + 桩件，运行期目录与模板库一律
指向 tmp_path。

重点覆盖四件事：
1. 「概览单图形状 ↔ 多图地理真值」按弧长配对注册成功，之后相似概览图
   （含反向）能直接命中，不必再带多图；
2. 形状域隔离：旧的 detail 模板（细节图 / 拼接画布）不得参与概览图定位，
   堵住 §22.45 实测的跨缩放假阳性；
3. 多图未覆盖整条路线、形状不符、点数 / 坐标 / 置信度不达标时一律拒收，
   且拒绝绝不影响识别请求；
4. ``/api/recognize`` 纯单图路径零回归，配对路径返回拼接画布叠加图 +
   overview_url，仅多图旧端点仍可用。
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

import app.pipeline as P
import app.route_template as RT
import app.server as S
from app.config import CONFIG
from app.route_template import (
    MIN_PAIR_COVERAGE,
    SHAPE_DOMAIN_DETAIL,
    SHAPE_DOMAIN_OVERVIEW,
    ShapeMatch,
    TemplateStore,
)
from app.track import TrackResult

HTML = Path(__file__).resolve().parent.parent / "src" / "web" / "index.html"

# 用户提供的真实高缩放概览图 OCR 文本：行政区 / 地铁线 / 高架主导，
# 街道级路名只有「龙吴路」一条，恰好触发 template_fallback 的宽泛判据。
OCR_OVERVIEW = [
    "南北高架路", "中环路", "外环高速", "虹梅高架路", "机场联络线",
    "静安区", "徐汇区", "闵行区", "长宁区", "黄浦区", "普陀区",
    "3号线", "7号线", "12号线", "15号线", "1号线", "5号线",
    "上海南站", "上海西站", "上海站", "锦江乐园", "西岸艺术中心",
    "起", "终", "美团地图", "龙吴路",
]


def _bent(n: int = 160) -> list[list[float]]:
    """来回绕行、aspect 接近 1 的折线，形状信息量充足。"""
    return [[float(400 * math.sin(i / 9) + 250 * math.sin(i / 23)),
             float(320 * math.cos(i / 7) + 180 * math.sin(i / 3))]
            for i in range(n)]


def _render_detail(px: list[list[float]]) -> list[list[float]]:
    """模拟「同一条路线的多图拼接画布」：非均匀缩放 + 平移 + 抖动。"""
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


def _store(tmp_path: Path) -> TemplateStore:
    return TemplateStore(tmp_path / "route_templates.json")


def _jpg() -> bytes:
    _, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 255, np.uint8))
    return buf.tobytes()


# --- 配对注册 -----------------------------------------------------------

def test_register_pair_writes_overview_template(tmp_path):
    px = _bent()
    detail = _render_detail(px)
    store = _store(tmp_path)
    out = store.register_pair(
        overview_px=px, detail_px=detail, geo_polyline=_geo(detail),
        confidence=0.8, city="上海", shot_id="probe")
    assert out["written"] is True, out
    assert out["shape_domain"] == SHAPE_DOMAIN_OVERVIEW
    assert out["pair_match"]["accepted"] is True
    assert out["pair_match"]["coverage"] >= MIN_PAIR_COVERAGE
    assert out["pair_match"]["mode"] in ("query_in_template",
                                         "template_in_query")
    assert store.stats()["overview"] == 1
    tpl = store.templates[0]
    assert tpl["shape_domain"] == SHAPE_DOMAIN_OVERVIEW
    assert tpl["source"] == "paired"
    assert len(tpl["px_points"]) == len(tpl["geo_points"]) == 64


def test_registered_pair_locates_similar_reversed_overview(tmp_path):
    px = _bent()
    detail = _render_detail(px)
    store = _store(tmp_path)
    assert store.register_pair(overview_px=px, detail_px=detail,
                               geo_polyline=_geo(detail), confidence=0.8,
                               city="上海")["written"] is True
    hit = store.locate(px[::-1], city_hint="上海",
                       shape_domain=SHAPE_DOMAIN_OVERVIEW)
    assert hit is not None
    assert hit["match"]["reversed"] is True
    assert len(hit["coords_gcj"]) == 64
    assert store.locate(px[::-1], shape_domain=SHAPE_DOMAIN_DETAIL) is None


def test_default_remember_is_detail_domain_and_never_locates_overview(tmp_path):
    """P0 沉淀的细节图模板不得被概览图查询命中（跨缩放假阳性）。"""
    px = _bent()
    store = _store(tmp_path)
    assert store.remember(px_polyline=px, geo_polyline=_geo(px),
                          confidence=0.8, city="上海")["written"] is True
    assert store.templates[0]["shape_domain"] == SHAPE_DOMAIN_DETAIL
    assert store.stats()["overview"] == 0
    assert store.locate(px, shape_domain=SHAPE_DOMAIN_OVERVIEW) is None
    assert store.locate(px, shape_domain=SHAPE_DOMAIN_DETAIL) is not None


def test_pair_dedup_upgrades_existing_template_domain(tmp_path):
    px = _bent()
    store = _store(tmp_path)
    assert store.remember(px_polyline=px, geo_polyline=_geo(px),
                          confidence=0.4, city="上海")["written"] is True
    out = store.register_pair(overview_px=px, detail_px=_render_detail(px),
                              geo_polyline=_geo(_render_detail(px)),
                              confidence=0.8, city="上海")
    assert out["reason"] == "deduped", out
    assert store.templates[0]["shape_domain"] == SHAPE_DOMAIN_OVERVIEW
    assert store.stats()["overview"] == 1


# --- 拒识 ---------------------------------------------------------------

@pytest.mark.parametrize("coverage,mode", [
    (0.70, "query_in_template"),
    (0.80, "template_in_query"),
])
def test_partial_route_is_rejected(monkeypatch, tmp_path, coverage, mode):
    px = _bent()
    detail = _render_detail(px)
    monkeypatch.setattr(
        RT, "match_shape",
        lambda q, t: ShapeMatch(True, 0.02, coverage, False, (0, 45),
                                "ok", 0.1, mode))
    out = _store(tmp_path).register_pair(
        overview_px=px, detail_px=detail, geo_polyline=_geo(detail),
        confidence=0.8)
    assert out["written"] is False
    assert out["reason"] == "partial_route"
    assert out["pair_match"]["coverage"] == coverage


def test_unrelated_route_pair_is_rejected(tmp_path):
    px = _bent()
    other = [[float(i * 9), float(260 * math.cos(i / 6) + 60 * math.sin(i / 2))]
             for i in range(160)]
    out = _store(tmp_path).register_pair(
        overview_px=px, detail_px=other, geo_polyline=_geo(other),
        confidence=0.8)
    assert out["written"] is False
    assert out["reason"].startswith("shape_") or out["reason"] == "degenerate_shape"


def test_pair_rejects_short_or_invalid_data(tmp_path):
    px = _bent()
    detail = _render_detail(px)
    geo = _geo(detail)
    store = _store(tmp_path)
    assert store.register_pair(overview_px=px[:5], detail_px=detail,
                               geo_polyline=geo, confidence=0.8)["reason"] \
        == "not_enough_points"
    # 保持折线形状合法，只把经度整体推出 [-180, 180]。
    bad = [[lo + 1000.0, la] for lo, la in geo]
    assert store.register_pair(overview_px=px, detail_px=detail,
                               geo_polyline=bad, confidence=0.8)["reason"] \
        == "invalid_coordinate"
    assert store.stats()["templates"] == 0


@pytest.mark.parametrize("kwargs,reason", [
    ({"confidence": 0.2}, "low_confidence"),
    ({"self_consistent": False}, "self_inconsistent"),
    ({"input_ok": False}, "low_information_input"),
])
def test_pair_inherits_auto_write_guards(tmp_path, kwargs, reason):
    px = _bent()
    detail = _render_detail(px)
    args = {"overview_px": px, "detail_px": detail,
            "geo_polyline": _geo(detail), "confidence": 0.8}
    args.update(kwargs)
    out = _store(tmp_path).register_pair(**args)
    assert out["written"] is False
    assert out["reason"] == reason


def test_pair_projection_failure_is_reported(monkeypatch, tmp_path):
    px = _bent()
    detail = _render_detail(px)
    monkeypatch.setattr(RT, "project_to_geo", lambda *a, **k: None)
    out = _store(tmp_path).register_pair(
        overview_px=px, detail_px=detail, geo_polyline=_geo(detail),
        confidence=0.8)
    assert out["written"] is False
    assert out["reason"] == "projection_failed"


# --- 主流程注入 ---------------------------------------------------------

def test_pipeline_requests_overview_domain(monkeypatch, tmp_path):
    px = _bent()
    track = TrackResult(
        polyline=np.asarray(px, float), start_marker=(0, 0), end_marker=(1, 1),
        mask=np.zeros((10, 10), np.uint8), bbox=(0, 0, 10, 10),
        confidence=0.95, notes=[])
    monkeypatch.setattr(P, "extract_track", lambda img, cfg: track)
    monkeypatch.setattr(P, "ocr_texts",
                        lambda img, cfg: [SimpleNamespace(text=t)
                                         for t in OCR_OVERVIEW])
    monkeypatch.setattr(P, "build_anchors", lambda img, items, cfg: [])
    monkeypatch.setattr(P.RidePipeline, "_resolve_anchors",
                        lambda self, a, c, ao, w: [])
    calls: list[str | None] = []

    class SpyStore:
        def locate(self, query_px, *, city_hint="", shape_domain=None):
            calls.append(shape_domain)
            return None

    cfg = replace(CONFIG, lexicon_path=tmp_path / "lexicon.json")
    pipe = P.RidePipeline(cfg, templates=SpyStore())
    pipe.run(np.zeros((100, 100, 3), np.uint8), city="上海",
             allow_online=False, output_crs="WGS-84", refine=False)
    assert calls == [SHAPE_DOMAIN_OVERVIEW], calls


# --- 服务端三分支 -------------------------------------------------------

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
        road_attribution={}, road_fit={}, road_name_constraints=[],
        road_name_fit={}, alternatives=[],
        diagnostics={"input_quality": {"usable": True}}, warnings=[],
        elapsed_ms=1, track_polyline_px=detail, map_matched=False,
        image_size=(100, 100, 3))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    store = _store(tmp_path)
    monkeypatch.setattr(S, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(S, "DATA_DIR", tmp_path)
    monkeypatch.setattr(S, "_templates", store)
    monkeypatch.setattr(S._pipeline, "run", lambda *a, **k: fake)
    monkeypatch.setattr(
        S, "extract_track",
        lambda img, cfg: TrackResult(
            polyline=np.asarray(px[::-1], float), start_marker=(0, 0),
            end_marker=(1, 1), mask=np.zeros((10, 10), np.uint8),
            bbox=(0, 0, 10, 10), confidence=0.95, notes=[]))
    monkeypatch.setattr(S, "stitch_images", lambda imgs: None)
    return store, fake


def _single_files(with_details: bool = False):
    files = [("file", ("overview.jpg", _jpg(), "image/jpeg"))]
    if with_details:
        files += [("details", ("d1.jpg", _jpg(), "image/jpeg")),
                  ("details", ("d2.jpg", _jpg(), "image/jpeg"))]
    return files


def test_single_shot_path_is_unchanged(server_stub):
    store, _ = server_stub
    client = TestClient(S.app)
    r = client.post("/api/recognize", files=_single_files(),
                    data={"city": "上海", "refine": "off"})
    assert r.status_code == 200
    d = r.json()
    assert "template_capture" in d
    assert "paired_registration" not in d
    assert "overview_url" not in d
    assert d["shot_url"].endswith(".jpg")
    assert store.stats()["overview"] == 0


def test_paired_path_registers_overview_template(server_stub):
    store, _ = server_stub
    client = TestClient(S.app)
    r = client.post("/api/recognize", files=_single_files(True),
                    data={"city": "上海", "refine": "off"})
    assert r.status_code == 200
    d = r.json()
    cap = d["paired_registration"]
    assert cap["written"] is True, cap
    assert cap["shape_domain"] == SHAPE_DOMAIN_OVERVIEW
    assert cap["pair_match"]["accepted"] is True
    assert d["shot_url"].endswith("_stitched.png")
    assert d["overview_url"].endswith(".jpg")
    assert d["n_shots"] == 2
    assert "template_capture" not in d
    assert any("已注册" in w for w in d["warnings"])
    assert store.stats()["overview"] == 1


def test_paired_failure_does_not_break_response(server_stub, monkeypatch):
    client = TestClient(S.app)
    monkeypatch.setattr(S, "_capture_paired_template",
                        lambda **k: {"written": False, "reason": "partial_route"})
    r = client.post("/api/recognize", files=_single_files(True),
                    data={"city": "上海", "refine": "off"})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["paired_registration"]["written"] is False
    assert any("下次同类图仍需补传" in w for w in d["warnings"])


def test_multi_only_endpoint_still_works(server_stub):
    client = TestClient(S.app)
    r = client.post("/api/recognize_multi",
                    files=[("files", ("d1.jpg", _jpg(), "image/jpeg"))],
                    data={"city": "上海", "refine": "off"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


# --- 前端两个上传入口 ---------------------------------------------------

def test_frontend_has_two_upload_entries():
    h = HTML.read_text(encoding="utf-8")
    assert '<input type="file" id="file" accept="image/*" hidden>' in h
    assert '<input type="file" id="details" accept="image/*" multiple hidden>' in h
    assert 'id="dropDetails"' in h
    assert 'id="detailsOnly"' in h
    assert "fd.append('file', picked, picked.name);" in h
    assert "fd.append('details', f, f.name);" in h
    assert "fd.append('files', f, f.name);" in h
    assert "detailsOnly ? '/api/recognize_multi' : '/api/recognize'" in h
    assert "以后形状接近的概览图只需单独上传" in h
    assert "概览单图（必选）" in h and "街道级细节图（可选）" in h
