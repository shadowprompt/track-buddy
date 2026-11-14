"""P2：双上传入口 + 配对注册 + 形状域过滤。

不启动服务、不读本地截图、不触发真实 OCR / 在线地理编码：合成折线做
几何 fixture，服务端用进程内 TestClient + 桩件，运行期目录与模板库一律
指向 tmp_path。

重点覆盖五件事：
1. 「概览单图形状 ↔ 多图地理真值」按弧长配对注册成功，之后相似概览图
   （含反向）能直接命中，不必再带多图；
2. 形状域隔离：旧的 detail 模板（细节图 / 拼接画布）不得参与概览图定位，
   堵住 §22.45 实测的跨缩放假阳性；
3. 多图未覆盖整条路线、形状不符、点数 / 坐标 / 里程不达标时一律拒收，
   且拒绝绝不影响识别请求；画布扁长、低置信、input_ok 不再连坐 paired
   注册（§22.56 死锁 A/C 解除），自洽性仍是硬门；
4. 召回触发双通道：宽泛概览图照旧；库里已有 overview 模板时普通截图也
   探测一次，但只有「近复刻」（残差/覆盖率达标）才采信，否则静默回落；
5. ``/api/recognize`` 纯单图路径零回归，配对路径返回拼接画布叠加图 +
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
    # paired 的写入门槛只剩自洽性；低置信与 input_ok 不再连坐（§22.56）。
    ({"self_consistent": False}, "self_inconsistent"),
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


def test_paired_low_confidence_registers_and_locates(tmp_path):
    """配对真值可信度来自多图拼接证据：单画布置信度封顶（0.2）不得连坐。

    无街道名锚点的长路线整图置信度被 pipeline 封到 0.2，旧门槛
    0.2 < 0.35 会把「概览↔真值」配对拒写（死锁 A）；本用例锁定：
    低置信 paired 模板可注册、可定位。
    """
    px = _bent()
    detail = _render_detail(px)
    store = _store(tmp_path)
    out = store.register_pair(
        overview_px=px, detail_px=detail, geo_polyline=_geo(detail),
        confidence=0.2, city="上海")
    assert out["written"] is True, out
    tpl = store.templates[0]
    assert tpl["source"] == "paired"
    assert tpl["confidence"] == 0.2
    # 读侧同样不得用 0.35 连坐 paired 模板。
    hit = store.locate(px, city_hint="上海",
                       shape_domain=SHAPE_DOMAIN_OVERVIEW)
    assert hit is not None
    assert hit["template"]["id"] == tpl["id"]


def test_canvas_thin_shape_no_longer_vetoes_pair(monkeypatch, tmp_path):
    """画布轨迹扁长（aspect≥4）不再否决配对：对应质量由 match_shape 把关。

    实测沿河近直路线：概览↔画布 residual 0.0116 / coverage 1.0，却被
    「画布侧 shape_has_enough_info」一刀切拒成 degenerate_shape（§22.56）。
    用 spy 模拟「概览侧形状合格、画布侧退化」，锁定配对仍成立。
    """
    px = _bent()
    detail = _render_detail(px)
    calls = {"n": 0}
    real = RT.shape_has_enough_info

    def spy(p):
        calls["n"] += 1
        return real(p) if calls["n"] == 1 else False

    monkeypatch.setattr(RT, "shape_has_enough_info", spy)
    out = _store(tmp_path).register_pair(
        overview_px=px, detail_px=detail, geo_polyline=_geo(detail),
        confidence=0.8)
    # 只剩概览侧一次调用：画布侧的退化否决已从 register_pair 移除。
    assert calls["n"] == 1, calls
    assert out["written"] is True, out


def test_geo_length_out_of_range_is_rejected(tmp_path):
    """地理真值里程异常（<100m）说明配准解不可信，不能固化成模板。"""
    px = _bent()
    detail = _render_detail(px)
    tiny = [[121.20 + i * 1e-6, 31.30] for i in range(64)]
    out = _store(tmp_path).register_pair(
        overview_px=px, detail_px=detail, geo_polyline=tiny,
        confidence=0.8)
    assert out["written"] is False
    assert out["reason"] == "geo_length_out_of_range"


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
        def locate(self, query_px, *, city_hint="", shape_domain=None, profile=""):
            calls.append(shape_domain)
            return None

    cfg = replace(CONFIG, lexicon_path=tmp_path / "lexicon.json")
    pipe = P.RidePipeline(cfg, templates=SpyStore())
    pipe.run(np.zeros((100, 100, 3), np.uint8), city="上海",
             allow_online=False, output_crs="WGS-84", refine=False)
    assert calls == [SHAPE_DOMAIN_OVERVIEW], calls


# --- 召回触发双通道（§22.56）--------------------------------------------

def _located_stub(residual: float, coverage: float = 1.0) -> dict:
    """构造一个 locate() 命中返回值：上海范围内的 64 点折线。"""
    coords = [[121.20 + i * 0.002, 31.30 + 0.001 * (i % 4)] for i in range(64)]
    return {
        "coords_gcj": coords,
        "match": {"accepted": True, "residual": residual,
                  "coverage": coverage, "signature": 0.1, "reversed": False,
                  "span": [0, 64], "mode": "query_in_template", "reason": "ok"},
        "template": {"id": "tpl_x", "city": "上海", "confidence": 0.2,
                     "verified": False, "hit_count": 1},
        "length_m": 5200.0,
    }


def _probe_pipeline(monkeypatch, tmp_path, store, texts):
    track = TrackResult(
        polyline=np.asarray(_bent(), float), start_marker=(0, 0),
        end_marker=(1, 1), mask=np.zeros((10, 10), np.uint8),
        bbox=(0, 0, 10, 10), confidence=0.95, notes=[])
    monkeypatch.setattr(P, "extract_track", lambda img, cfg: track)
    monkeypatch.setattr(
        P, "ocr_texts",
        lambda img, cfg: [SimpleNamespace(text=t) for t in texts])
    monkeypatch.setattr(P, "build_anchors", lambda img, items, cfg: [])
    monkeypatch.setattr(P.RidePipeline, "_resolve_anchors",
                        lambda self, a, c, ao, w: [])
    cfg = replace(CONFIG, lexicon_path=tmp_path / "lexicon.json")
    return P.RidePipeline(cfg, templates=store)


def test_non_broad_overview_hits_near_duplicate_template(monkeypatch, tmp_path):
    """库里有 overview 模板时，普通截图也探测；近复刻命中直跳参考级。

    实测概览图 OCR 宽泛占比 0.373 < 0.5，宽泛判据不触发，但「上次拼过
    的路线又来了」——只要形状近乎复刻就该免传细节图。
    """
    street_texts = ["祁连山南路", "金沙江路", "淞虹路", "北翟路", "天山西路"]
    calls: list[str | None] = []

    class HitStore:
        def stats(self):
            return {"templates": 1, "overview": 1}

        def locate(self, query_px, *, city_hint="", shape_domain=None, profile=""):
            calls.append(shape_domain)
            return _located_stub(residual=0.010, coverage=1.0)

    pipe = _probe_pipeline(monkeypatch, tmp_path, HitStore(), street_texts)
    res = pipe.run(np.zeros((100, 100, 3), np.uint8), city="上海",
                   allow_online=False, output_crs="WGS-84", refine=False)
    assert calls == [SHAPE_DOMAIN_OVERVIEW], calls
    assert res.diagnostics.get("source") == "template"
    assert res.diagnostics.get("template_match") is not None
    assert any("近复刻命中" in w for w in res.warnings)


def test_non_broad_overview_far_match_falls_through(monkeypatch, tmp_path):
    """探测到了相似模板但不够「近复刻」：静默回落原流程，绝不硬套参考级。"""
    street_texts = ["祁连山南路", "金沙江路", "淞虹路", "北翟路", "天山西路"]
    calls: list[str | None] = []

    class FarStore:
        def stats(self):
            return {"templates": 1, "overview": 1}

        def locate(self, query_px, *, city_hint="", shape_domain=None, profile=""):
            calls.append(shape_domain)
            return _located_stub(residual=0.20, coverage=1.0)

    pipe = _probe_pipeline(monkeypatch, tmp_path, FarStore(), street_texts)
    res = pipe.run(np.zeros((100, 100, 3), np.uint8), city="上海",
                   allow_online=False, output_crs="WGS-84", refine=False)
    assert calls == [SHAPE_DOMAIN_OVERVIEW], calls
    assert res.diagnostics.get("source") != "template"
    assert not any("近复刻命中" in w for w in res.warnings)


def test_no_templates_no_probe(monkeypatch, tmp_path):
    """库里没有 overview 模板且非宽泛图：不做探测，行为与旧版一致。"""
    street_texts = ["祁连山南路", "金沙江路", "淞虹路", "北翟路", "天山西路"]
    calls: list[str | None] = []

    class EmptyStore:
        def stats(self):
            return {"templates": 0, "overview": 0}

        def locate(self, query_px, *, city_hint="", shape_domain=None, profile=""):
            calls.append(shape_domain)
            return None

    pipe = _probe_pipeline(monkeypatch, tmp_path, EmptyStore(), street_texts)
    pipe.run(np.zeros((100, 100, 3), np.uint8), city="上海",
             allow_online=False, output_crs="WGS-84", refine=False)
    assert calls == [], calls


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
    # 这个 fixture 验证的是「配对路径会注册概览图模板」，需要拼接成功。
    # 以前打成 None（拼接失败）是不真实的设定：现在拼接失败会被
    # 「不能拿单图结果当多图真值注册模板」的护栏挡下，故补一个成功画布。
    monkeypatch.setattr(
        S, "stitch_images",
        lambda imgs: SimpleNamespace(
            ok=True, canvas=np.zeros((10, 10, 3), np.uint8),
            to_dict=lambda: {"ok": True, "notes": [], "used": [0, 1]},
        ),
    )
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
    assert any("已记入历史记录" in w for w in d["warnings"])
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


def test_capture_paired_requires_recognition_and_evidence(server_stub):
    """配对注册的证据门：识别成功 + 徽章守恒 + 逐像素复核缺一不可。

    画布 input_quality 与置信度封顶不再参与（§22.56 死锁 A/C 解除）。
    """
    store, fake = server_stub
    detail = _render_detail(_bent())
    base = dict(overview_img=np.zeros((10, 10, 3), np.uint8),
                truth_result=fake, city="上海", shot_id="x")
    ok_payload = {"ok": True, "track_polyline_px": detail,
                  "geojson_gcj02": fake.geojson_gcj02}
    r1 = S._capture_paired_template(
        truth_payload={"ok": False}, stitch_info={}, **base)
    assert r1 == {"written": False, "reason": "recognition_failed"}
    r2 = S._capture_paired_template(
        truth_payload=dict(ok_payload),
        stitch_info={"badge_conservation": {"ok": False}}, **base)
    assert r2["written"] is False
    assert r2["reason"] == "self_inconsistent"
    r3 = S._capture_paired_template(
        truth_payload=dict(ok_payload),
        stitch_info={"verify": {"s1": 0.03, "s2": 1.0}}, **base)
    assert r3["written"] is False
    assert r3["reason"] == "self_inconsistent"
    r4 = S._capture_paired_template(
        truth_payload=dict(ok_payload),
        stitch_info={"verify": {"s1": 1.0, "s2": 0.9}}, **base)
    assert r4["written"] is True, r4
    assert store.stats()["overview"] == 1


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
    assert "轨迹概览图（必选）" in h and "轨迹街道级细节图（可选）" in h
