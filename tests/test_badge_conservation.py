"""拼接层徽章守恒诊断 + 失败归因文案（2026-11-12）。

真实事故：7 张细节图里第 7 张有「起」，贴图时被同深度的另一张盖掉，
画布上 0 枚。`track.py` 只看最终画布，于是报「未检测到起点徽章」；
用户自然以为是自己截图没截到——归因错了整整一层。

这组测试守住三件事：
1. `stitch()` 的产物级不变量：输入有 ⇒ 画布有；
2. `_explain_badge_loss()`：不守恒时说「拼接丢失」，守恒时一字不改；
3. 配对模式失败时不再宣称「本次结果为地理真值」，注册失败按原因分流。

不依赖网络、不依赖用户真实截图、不写运行期词典/模板数据。
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import app.server as S  # noqa: E402
from app.config import VisionConfig  # noqa: E402
from app.multishot import (  # noqa: E402
    _badge_conservation, _badge_protection_items, stitch,
)


def _textured_base(h: int = 560, w: int = 1100, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 238, np.uint8)
    for y in range(0, h, 37):
        cv2.line(img, (0, y), (w, y), (205, 205, 205), 1)
    for x in range(0, w, 41):
        cv2.line(img, (x, 0), (x, h), (210, 210, 210), 1)
    for _ in range(220):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(2, 8)),
                   tuple(int(v) for v in rng.integers(70, 190, 3)), -1)
    return img


def _shots() -> tuple[np.ndarray, np.ndarray]:
    base = _textured_base()
    a = base[:, 0:700].copy()
    b = base[:, 300:1000].copy()
    cv2.circle(b, (100, 280), 28, (60, 200, 60), -1)   # 绿「起」，落在重叠区
    cv2.circle(b, (560, 280), 28, (60, 60, 235), -1)  # 红「终」，落在独占区
    return a, b


def _erase_green(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (35, 90, 90), (85, 255, 255))
    out[m > 0] = (245, 245, 245)
    return out


# --- 1. 产物级不变量 -----------------------------------------------------

def test_stitch_reports_conservation_when_badges_survive() -> None:
    a, b = _shots()
    res = stitch([a, b])
    assert res.ok, res.notes
    cons = res.badge_conservation
    assert cons["ok"] is True, cons
    assert cons["missing"] == []
    assert cons["input"]["start"] == [1], cons
    assert cons["canvas"] == {"start": True, "end": True}, cons
    assert res.to_dict()["badge_conservation"] == cons
    assert not any("拼接产物丢失了" in n for n in res.notes), res.notes


def test_conservation_flags_only_categories_present_in_input() -> None:
    """终点保住、起点丢失：只报 start；输入本来没有的类别不误报。"""
    a, b = _shots()
    canvas = _erase_green(b)
    cons = _badge_conservation(canvas, {"start": [1], "end": [1]})
    assert cons["ok"] is False
    assert cons["missing"] == ["start"], cons
    assert cons["canvas"] == {"start": False, "end": True}, cons

    none = _badge_conservation(canvas, {"start": [], "end": []})
    assert none["ok"] is True and none["missing"] == [], none

    only_end = _badge_conservation(
        _erase_green(canvas), {"start": [], "end": [1]}
    )
    assert only_end["ok"] is True, only_end


def test_excluded_shot_badge_still_counts_as_input() -> None:
    """被判「无重叠」整张排除的图，上面的徽章也算输入里的信息。"""
    a, b = _shots()  # 只有 b（index 1）有徽章
    items, inv = _badge_protection_items([a, b], [0], scan_indices=[0, 1])
    assert inv["start"] == [1] and inv["end"] == [1], inv
    assert 1 not in [i for i, _k, _c in items], "没变换矩阵的图不能写回"

    # 不传 scan_indices 时保持旧行为（只扫参与贴图的图）
    _, legacy = _badge_protection_items([a, b], [0])
    assert legacy == {"start": [], "end": []}, legacy


# --- 2. 失败归因文案 -----------------------------------------------------

def test_explain_badge_loss_replaces_stale_warning() -> None:
    cons = {
        "input": {"start": [6], "end": [0]},
        "canvas": {"start": False, "end": True},
        "ok": False,
        "missing": ["start"],
    }
    warnings = [
        "未检测到起点徽章（绿色「起」）",
        "轨迹形状不合理：连通域宽 2883px",
        "未能提取到有效轨迹线",
        "概览图未能注册（not_enough_points）",
    ]
    out = S._explain_badge_loss(warnings, cons)
    assert "拼接阶段丢失了起点徽章" in out[0], out
    assert "输入第 7 张图有" in out[0], out
    assert "不是截图没截到" in out[0], out
    assert not any("未检测到起点徽章" in w for w in out), out
    assert out[1:] == warnings[1:], "连锁告警必须保留，只是不再当主因"


@pytest.mark.parametrize("cons", [None, {}, {"ok": True, "missing": []}])
def test_explain_badge_loss_is_noop_when_conserved(cons) -> None:
    warnings = ["未检测到起点徽章（绿色「起」）", "未能提取到有效轨迹线"]
    assert S._explain_badge_loss(warnings, cons) == warnings


def test_paired_failure_note_is_reason_specific() -> None:
    stale = "请确认细节图覆盖整条路线且识别自洽"
    note = S._paired_failure_note("not_enough_points")
    assert "本次轨迹提取未成功" in note and stale not in note, note
    assert "下次同类图仍需补传" in note

    assert "只覆盖了路线的一部分" in S._paired_failure_note("partial_route")
    assert "置信度不足" in S._paired_failure_note("low_confidence")
    assert "形状配对未通过" in S._paired_failure_note("shape_coverage")
    assert "模板注册过程出错" in S._paired_failure_note("error:ValueError")
    assert "可只传单图" in S._paired_failure_note("deduped")
    assert "unknown" in S._paired_failure_note(None)


# --- 3. 端到端：失败时不能宣称「地理真值」 ------------------------------

def _jpg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((120, 80, 3), 235, np.uint8))
    assert ok
    return buf.tobytes()


def _failed_result() -> SimpleNamespace:
    empty = {"type": "FeatureCollection", "features": []}
    return SimpleNamespace(
        confidence=0.0, track_points=0, geojson=empty, geojson_gcj02={},
        crs="GCJ-02", distance_m=0.0, anchors_used=[], anchors_rejected=[],
        road_attribution={}, road_fit={}, road_name_constraints=[],
        road_name_fit={}, alternatives=[], diagnostics={},
        warnings=["未检测到起点徽章（绿色「起」）", "未能提取到有效轨迹线"],
        elapsed_ms=1, track_polyline_px=[], map_matched=False,
        image_size=(120, 80, 3),
    )


def test_failed_pair_does_not_claim_ground_truth(monkeypatch, tmp_path) -> None:
    cons = {
        "input": {"start": [1], "end": [0]},
        "canvas": {"start": False, "end": True},
        "ok": False,
        "missing": ["start"],
    }
    st = SimpleNamespace(
        ok=True,
        canvas=np.full((120, 160, 3), 235, np.uint8),
        to_dict=lambda: {"ok": True, "notes": [], "badge_conservation": cons},
    )
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(S, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(S, "DATA_DIR", tmp_path)
    monkeypatch.setattr(S._pipeline, "run", lambda *a, **k: _failed_result())
    monkeypatch.setattr(S, "stitch_images", lambda imgs: st)
    monkeypatch.setattr(
        S, "_capture_paired_template",
        lambda **k: {"written": False, "reason": "not_enough_points"},
    )

    files = [("file", ("overview.jpg", _jpg(), "image/jpeg"))]
    files += [("details", (f"d{i}.jpg", _jpg(), "image/jpeg")) for i in (1, 2)]
    r = TestClient(S.app).post(
        "/api/recognize", files=files,
        data={"city": "上海", "refine": "off"},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is False
    assert d["stitch"]["badge_conservation"]["ok"] is False
    warns = d["warnings"]
    assert "拼接阶段丢失了起点徽章" in warns[0], warns
    assert not any("未检测到起点徽章（绿色「起」）" in w for w in warns), warns
    assert not any("本次结果为地理真值" in w for w in warns), warns
    assert any("但本次识别未成功" in w for w in warns), warns
    assert any("本次轨迹提取未成功" in w for w in warns), warns
