"""模板召回劫持真值路径事故回归（2026-11-14 18:18 实测）。

事故：用户传概览图 + 5 张细节图，拼接已成功、地理真值在即，但
pipeline 阶段 2.5 在配准之前执行，画布形状被历史模板「近复刻」误命中
（reversed=True、residual 0.0266、signature 0.3632）后直接 return——
真值路径被短路，payload 同时出现「⚠️ 参考级定位（未核验）」与
「已用 5 张细节图拼接配准，本次结果为地理真值」两句自相矛盾的告警，
地图上展示的是一条完全无关的历史路线。

三块契约：

1. 近复刻门槛收紧：方向签名门 + 反向匹配门槛减半（形状距离分不出
   「走法不同」的两条路，转向序列可以）；
2. paired 模式（传了细节图）必须跳过模板召回——用户愿意传细节图，
   就一定走当次地理配准；模板召回只服务「免传细节图」的复用场景，
   绝不与真值路径竞争；
3. 两个识别端点（单图带细节 / 多图）都把 skip 旗标传到了 pipeline。
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.pipeline as P
import app.server as S
from app.config import CONFIG
from app.pipeline import _near_template_hit


# --- 1. 近复刻门槛 -------------------------------------------------------

def _m(residual: float, coverage: float = 0.98,
       signature: float = 0.05, reversed_: bool = False) -> dict:
    return {"match": {"residual": residual, "coverage": coverage,
                      "signature": signature, "reversed": reversed_}}


def test_genuine_forward_replica_passes() -> None:
    """同图重传实测 residual≈0.0116：必须继续直跳（飞轮核心价值）。"""
    assert _near_template_hit(_m(0.0116, signature=0.02)) is True


def test_genuine_reversed_replica_passes() -> None:
    """同路线反向重骑：residual 与正向同量级（~0.01），仍可复用。"""
    assert _near_template_hit(_m(0.008, signature=0.02,
                                 reversed_=True)) is True


def test_incident_reversed_false_positive_rejected() -> None:
    """2026-11-14 事故原值：residual 0.0266 / signature 0.3632 /
    reversed=True——三道门各自都能拦下它。"""
    hit = _near_template_hit(_m(0.0266, coverage=0.9844,
                                signature=0.3632, reversed_=True))
    assert hit is False


def test_reversed_with_low_signature_but_loose_residual_rejected() -> None:
    """反向匹配即使签名很小，residual 门槛也减半——反向假阳性率更高。"""
    assert _near_template_hit(_m(0.0266, signature=0.10,
                                 reversed_=True)) is False


def test_high_signature_forward_rejected() -> None:
    """形状相近但「走法」不同（signature > 0.30）：拒绝。"""
    assert _near_template_hit(_m(0.010, signature=0.40)) is False


def test_loose_coverage_rejected() -> None:
    assert _near_template_hit(_m(0.010, coverage=0.80)) is False


def test_missing_or_garbage_match_rejected() -> None:
    assert _near_template_hit({}) is False
    assert _near_template_hit({"match": {"residual": "x"}}) is False


# --- 2. paired 模式跳过召回（端点接线） ----------------------------------

def _bent(n: int = 160) -> list[list[float]]:
    return [[float(400 * math.sin(i / 9) + 250 * math.sin(i / 23)),
             float(320 * math.cos(i / 7) + 180 * math.sin(i / 3))]
            for i in range(n)]


def _track(px):
    from app.track import TrackResult
    return TrackResult(
        polyline=np.asarray(px, float), start_marker=(0, 0),
        end_marker=(1, 1), mask=np.zeros((10, 10), np.uint8),
        bbox=(0, 0, 10, 10), confidence=0.95, notes=[])


def _jpg() -> bytes:
    _, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 255, np.uint8))
    return buf.tobytes()


@pytest.fixture
def recall_stub(monkeypatch, tmp_path):
    """捕获 pipeline.run 的 kwargs 与 templates.locate 的调用。"""
    px = _bent()
    fake = SimpleNamespace(
        confidence=0.8, track_points=len(px),
        geojson={"type": "FeatureCollection", "features": []},
        geojson_gcj02={"type": "FeatureCollection", "features": []},
        crs="WGS-84", distance_m=15000.0, anchors_used=[],
        anchors_rejected=[], road_attribution={},
        road_fit={"ok": True, "shift_m": 0.0}, road_name_constraints=[],
        road_name_fit={}, alternatives=[],
        diagnostics={"input_quality": {"usable": True}}, warnings=[],
        elapsed_ms=1, track_polyline_px=px, map_matched=False,
        image_size=(100, 100, 3))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(S, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(S, "DATA_DIR", tmp_path)

    captured: dict[str, object] = {"locate_calls": 0}

    def fake_run(img, **kwargs):
        captured["skip_template_recall"] = kwargs.get(
            "skip_template_recall")
        return fake

    monkeypatch.setattr(S._pipeline, "run", fake_run)

    class SpyTemplates:
        def locate(self, *a, **k):
            captured["locate_calls"] += 1
            return None

    monkeypatch.setattr(S, "_templates", SpyTemplates())
    return captured


def test_single_with_details_skips_recall(recall_stub) -> None:
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg")),
               ("details", ("d1.jpg", _jpg(), "image/jpeg")),
               ("details", ("d2.jpg", _jpg(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    assert recall_stub["skip_template_recall"] is True


def test_single_without_details_keeps_recall(recall_stub) -> None:
    """免传细节图是模板召回的核心价值：不带 details 时旗标必须为 False。"""
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    assert recall_stub["skip_template_recall"] is False


def test_multi_endpoint_skips_recall(recall_stub) -> None:
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize_multi",
        files=[("files", ("d1.jpg", _jpg(), "image/jpeg")),
               ("files", ("d2.jpg", _jpg(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    assert recall_stub["skip_template_recall"] is True


# --- 2b. template 三态入参（2026-11-14 用户需求） ------------------------
#
# auto：默认策略（无细节图才召回）；prefer：优先模板（paired 也先探测）；
# off：禁用模板。payload["template_mode"] 回显本次生效策略。

def test_template_prefer_overrides_paired_skip(recall_stub) -> None:
    """「优先模板」：带细节图也先探测召回（用户显式选择，允许命中）。"""
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg")),
               ("details", ("d1.jpg", _jpg(), "image/jpeg"))],
        data={"template": "prefer"})
    assert r.status_code == 200, r.text
    assert recall_stub["skip_template_recall"] is False
    assert r.json()["template_mode"] == "prefer"


def test_template_off_forces_skip(recall_stub) -> None:
    """「禁用模板」：即使没传细节图也不探测。"""
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg"))],
        data={"template": "off"})
    assert r.status_code == 200, r.text
    assert recall_stub["skip_template_recall"] is True
    assert r.json()["template_mode"] == "off"


def test_template_auto_default_keeps_paired_skip(recall_stub) -> None:
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg")),
               ("details", ("d1.jpg", _jpg(), "image/jpeg"))],
        data={"template": "auto"})
    assert r.status_code == 200, r.text
    assert recall_stub["skip_template_recall"] is True
    assert r.json()["template_mode"] == "auto"


def test_template_invalid_value_rejected(recall_stub) -> None:
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize",
        files=[("file", ("overview.jpg", _jpg(), "image/jpeg"))],
        data={"template": "always"})
    assert r.status_code == 400
    assert "prefer" in r.json()["detail"] and "off" in r.json()["detail"]


def test_multi_template_prefer(recall_stub) -> None:
    client = TestClient(S.app)
    r = client.post(
        "/api/recognize_multi",
        files=[("files", ("d1.jpg", _jpg(), "image/jpeg")),
               ("files", ("d2.jpg", _jpg(), "image/jpeg"))],
        data={"template": "prefer"})
    assert r.status_code == 200, r.text
    assert recall_stub["skip_template_recall"] is False


def test_prefer_hit_does_not_claim_ground_truth(recall_stub) -> None:
    """prefer 模式近复刻命中时：pair_notes 绝不能宣称「地理真值」。

    §22.65 事故的展示层教训——模板插值结果混一句「本次结果为地理
    真值」是最误导人的组合，必须由「优先模板已命中」文案替代。
    """
    from app.pipeline import TEMPLATE_CONFIDENCE_CAP
    diag = {
        "input_quality": {"usable": False},
        "template_derived": True,
        "source": "template",
        "template_match": {"accepted": True, "residual": 0.01,
                           "coverage": 0.99, "signature": 0.02,
                           "reversed": False, "span": [0, 63],
                           "mode": "query_in_template", "reason": "ok",
                           "template": {"id": "tpl_x", "city": "上海",
                                        "confidence": 0.9}},
    }
    client = TestClient(S.app)
    # 直接改桩件返回模板结果：pipeline 内部由 _near_template_hit 决定
    # 采信，这里只验证 server 层的文案分流。
    recall_stub  # fixture 已就绪

    orig = S._pipeline.run

    def fake_run(img, **kwargs):
        return SimpleNamespace(
            confidence=TEMPLATE_CONFIDENCE_CAP, track_points=64,
            geojson={"type": "FeatureCollection", "features": []},
            geojson_gcj02={"type": "FeatureCollection", "features": []},
            crs="GCJ-02", distance_m=9278.7, anchors_used=[],
            anchors_rejected=[], road_attribution={},
            road_fit={"ok": False, "shift_m": 0.0},
            road_name_constraints=[], road_name_fit={}, alternatives=[],
            diagnostics=diag,
            warnings=["⚠️ 参考级定位（未核验）：近复刻命中"],
            elapsed_ms=1, track_polyline_px=[], map_matched=False,
            image_size=(100, 100, 3))

    S._pipeline.run = fake_run
    try:
        r = client.post(
            "/api/recognize",
            files=[("file", ("overview.jpg", _jpg(), "image/jpeg")),
                   ("details", ("d1.jpg", _jpg(), "image/jpeg"))],
            data={"template": "prefer"})
    finally:
        S._pipeline.run = orig
    assert r.status_code == 200, r.text
    notes = " ".join(r.json().get("warnings") or [])
    assert "参考级" in notes and "地理真值" not in notes.replace(
        "未做当次地理配准", "")


# --- 3. pipeline 层行为：旗标置位时根本不探测 ----------------------------

def test_run_impl_skips_locate_when_flag_set(monkeypatch) -> None:
    """直接验证阶段 2.5 的分支条件：skip=True 时 locate 一次都不调用。

    用最小桩件驱动 _run_impl 到阶段 2.5 之后立刻抛出一个哨兵异常，
    避免跑完后续重量级阶段（地理编码/配准）。
    """
    from app.track import TrackResult

    px = [[float(i), float(50 + 30 * math.sin(i / 8))] for i in range(200)]
    calls: dict[str, int] = {"locate": 0}

    class SpyTemplates:
        def stats(self):
            return {"overview": 1}

        def locate(self, *a, **k):
            calls["locate"] += 1
            return None

    pipe = P.RidePipeline.__new__(P.RidePipeline)
    pipe.cfg = CONFIG
    pipe.templates = SpyTemplates()
    pipe.lexicon = SimpleNamespace(
        begin_capture=lambda: None, stats=lambda: {})
    pipe.geocoder = SimpleNamespace(
        _online_failures=[], reset_online_failure=lambda: None)

    img = np.full((600, 800, 3), 255, np.uint8)
    monkeypatch.setattr(P, "extract_track", lambda im, cfg: TrackResult(
        polyline=np.asarray(px, float), start_marker=(5, 5),
        end_marker=(795, 595), mask=np.zeros((600, 800), np.uint8),
        bbox=(0, 0, 800, 600), confidence=0.9, notes=[]))
    # OCR 与锚点阶段同样桩掉：阶段 2.5 只依赖 texts 与 road_names
    monkeypatch.setattr(P, "ocr_texts", lambda im, cfg: ["长宁路", "伊犁路"])
    monkeypatch.setattr(P, "build_anchors", lambda im, texts, cfg: [])
    sentinel = RuntimeError("sentinel-stop-after-stage-2.5")
    monkeypatch.setattr(
        P, "assess_broad_overview",
        lambda texts, road_names: {
            "broad_overview": False, "n_texts": 2, "n_broad": 0,
            "n_street": 2, "broad_ratio": 0.0, "broad_examples": [],
            "street_examples": texts, "position_ok": True,
            "user_road_names": False})

    def boom(*a, **k):
        raise sentinel

    # _resolve_anchors 是阶段 2.5 之后的第一个重量级调用点；
    # 走到这里说明 skip 生效（未命中召回、未提前 return）。
    monkeypatch.setattr(pipe, "_resolve_anchors", boom)

    with pytest.raises(RuntimeError, match="sentinel"):
        pipe._run_impl(img, skip_template_recall=True)
    assert calls["locate"] == 0

    # 对照组：不置旗标时探测会发生（locate 被调、未命中后继续走到哨兵）
    with pytest.raises(RuntimeError, match="sentinel"):
        pipe._run_impl(img)
    assert calls["locate"] == 1


# --- 4. 前端契约（静态源码断言） -----------------------------------------

def test_web_template_mode_select_and_display() -> None:
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1]
            / "src" / "web" / "index.html").read_text(encoding="utf-8")
    # 三态下拉与提交
    assert 'id="templateMode"' in html
    for v in ("auto", "prefer", "off"):
        assert f'<option value="{v}"' in html
    assert "fd.append('template'" in html
    # 顶部 stat 卡展示是否参考模板
    assert "template_derived" in html
    assert "参考历史记录" in html and "本次独立配准" in html


def test_miniprogram_template_mode_wired() -> None:
    root = (Path(__file__).resolve().parents[1] / "wxmin")
    api = (root / "utils" / "api.js").read_text(encoding="utf-8")
    assert "opts.templateMode" in api
    assert "'template'" in api
    idx_js = (root / "pages" / "index" / "index.js").read_text(encoding="utf-8")
    assert "templateModes" in idx_js and "onTemplateModeChange" in idx_js
    idx_wxml = (root / "pages" / "index" / "index.wxml").read_text(
        encoding="utf-8")
    assert "templateModes" in idx_wxml
    res_js = (root / "pages" / "result" / "result.js").read_text(
        encoding="utf-8")
    assert "template_derived" in res_js and "参考历史记录" in res_js


# --- 4. 前端契约（静态源码断言） -----------------------------------------

def test_web_template_mode_select_and_display() -> None:
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1]
            / "src" / "web" / "index.html").read_text(encoding="utf-8")
    # 三态下拉与提交
    assert 'id="templateMode"' in html
    for v in ("auto", "prefer", "off"):
        assert f'<option value="{v}"' in html
    assert "fd.append('template'" in html
    # 顶部 stat 卡展示是否参考模板
    assert "template_derived" in html
    assert "参考历史记录" in html and "本次独立配准" in html


def test_miniprogram_template_mode_wired() -> None:
    root = (Path(__file__).resolve().parents[1] / "wxmin")
    api = (root / "utils" / "api.js").read_text(encoding="utf-8")
    assert "opts.templateMode" in api
    assert "'template'" in api
    idx_js = (root / "pages" / "index" / "index.js").read_text(encoding="utf-8")
    assert "templateModes" in idx_js and "onTemplateModeChange" in idx_js
    idx_wxml = (root / "pages" / "index" / "index.wxml").read_text(
        encoding="utf-8")
    assert "templateModes" in idx_wxml
    res_js = (root / "pages" / "result" / "result.js").read_text(
        encoding="utf-8")
    assert "template_derived" in res_js and "参考历史记录" in res_js
