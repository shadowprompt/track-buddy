"""服务端拼接后端接入的回归测试。

重点不是「能不能拼」，而是**失败时会不会骗人**：用户选了大模型后端
却没配服务，结果里必须明确写「后端失败 + 未回退」，不能拿 SIFT 的
画布冒充大模型的成绩。
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.server import (
    _canvas_preview_id,
    _stitch_with_backend,
    app,
    recognize,
    recognize_multi,
    stitch_backends,
)

client = TestClient(app)


def _imgs():
    return [np.zeros((40, 60, 3), np.uint8) for _ in range(2)]


def test_backends_endpoint_reports_ids_and_availability() -> None:
    d = client.get("/api/stitch-backends")
    assert d.status_code == 200
    body = d.json()
    assert body["default"] == "sift"
    assert [b["id"] for b in body["backends"]] == [
        "sift", "llm", "llm_guided"]
    assert all("available" in b and "reason" in b for b in body["backends"])
    assert body["backends"][0]["available"] is True
    assert stitch_backends()["default"] == "sift"


def test_both_recognize_endpoints_accept_stitch_backend_form() -> None:
    for fn in (recognize_multi, recognize):
        assert "stitch_backend" in inspect.signature(fn).parameters
    schema = client.get("/openapi.json").json()
    assert "stitch_backend" in json.dumps(schema, ensure_ascii=False)


def test_unknown_backend_fails_loudly_and_never_falls_back() -> None:
    info, canvas = _stitch_with_backend(_imgs(), "does-not-exist")
    assert info["ok"] is False
    assert info["backend"] == "does-not-exist"
    assert info["backend_error"] is True
    assert canvas is None
    assert any("未回退到其它拼接后端" in n for n in info["notes"])


def test_llm_without_key_fails_loudly_and_never_falls_back(monkeypatch) -> None:
    # 强制模拟「没配 key」，无论开发机上 .env 有没有配置都不会发真实请求。
    class NoLLM:
        available = False
        model = ""

        def chat_json(self, *a, **k):
            raise AssertionError("不可用时不应调用模型")

    monkeypatch.setattr(
        "app.stitch_backends.registry.LLMClient", lambda: NoLLM())
    info, canvas = _stitch_with_backend(_imgs(), "llm")
    assert info["ok"] is False
    assert info["backend"] == "llm"
    assert info["backend_error"] is True
    assert canvas is None
    assert "LLM_BASE_URL" in info["notes"][0]
    assert any("未回退到其它拼接后端" in n for n in info["notes"])


def test_plain_sift_failure_is_not_a_backend_error() -> None:
    """SIFT 拼不上属于正常失败，不能标成「后端不可用」。"""
    info, _canvas = _stitch_with_backend(_imgs(), "sift")
    assert info["backend"] == "sift"
    assert "backend_error" not in info
    assert any("拼接" in n for n in info["notes"])


def test_default_sift_path_keeps_legacy_patch_point(monkeypatch) -> None:
    """默认后端必须继续走 `server.stitch_images`，否则既有回归全断。"""
    from types import SimpleNamespace

    import app.server as S

    fake = SimpleNamespace(
        ok=True, canvas=np.zeros((10, 10, 3), np.uint8),
        to_dict=lambda: {"ok": True, "notes": []},
    )
    monkeypatch.setattr(S, "stitch_images", lambda imgs: fake)
    info, canvas = _stitch_with_backend(_imgs(), "sift")
    assert info["ok"] is True
    assert canvas is not None
    assert info["backend"] == "sift"
    assert info["backend_diagnostics"]["pairs_attempted"] == 1


def test_stitch_failure_note_explains_why_no_template() -> None:
    from app.server import _paired_failure_note

    note = _paired_failure_note("stitch_failed")
    assert "拼接未成功" in note
    assert "地理真值" in note
    assert "下次同类图仍需补传" in note


def test_run_stitch_carries_plan_diagnostics_on_error(monkeypatch) -> None:
    """后端报错时诊断不能丢，否则页面「模型诊断」整块是空的。"""
    from app.stitch_backends import base, registry

    diag = {"model": "gpt-6.1-sol", "elapsed_ms": 440, "pairs_accepted": 0}
    plan = base.StitchPlan(error="HTTP 400 invalid_parameter", diagnostics=diag)
    spec = SimpleNamespace(
        id="llm", label="大模型语义配准", requires_llm=False,
        plan=lambda *a, **k: plan,
    )
    monkeypatch.setattr(registry, "get_backend", lambda name: spec)
    with pytest.raises(base.StitchBackendError) as ei:
        registry.run_stitch(_imgs(), backend="llm")
    assert ei.value.diagnostics == diag


def test_backend_failure_exposes_diagnostics_in_result(monkeypatch) -> None:
    from app.stitch_backends.base import StitchBackendError

    diag = {
        "model": "gpt-6.1-sol", "elapsed_ms": 440,
        "pairs_accepted": 0, "error": "旧值会被覆盖",
    }

    def boom(images, backend):
        raise StitchBackendError(
            "[大模型语义配准（纯 VLM）] HTTP 400: "
            "Field 'max_tokens' must be at most 65536",
            diag,
        )

    monkeypatch.setattr("app.server.run_stitch", boom)
    info, canvas = _stitch_with_backend(_imgs(), "llm")
    assert info["ok"] is False and canvas is None
    assert info["backend_error"] is True
    bd = info["backend_diagnostics"]
    assert bd["model"] == "gpt-6.1-sol"
    assert bd["elapsed_ms"] == 440
    assert "must be at most 65536" in bd["error"]
    assert any("未回退到其它拼接后端" in n for n in info["notes"])


def test_canvas_preview_id_distinguishes_real_stitch_from_fallback() -> None:
    """拼接失败时预览图不能还叫 _stitched（会让人误以为拼上了）。"""
    assert _canvas_preview_id("abc", True) == "abc_stitched"
    assert _canvas_preview_id("abc", False) == "abc_fallback"
    for fn in (recognize_multi, recognize):
        assert "_canvas_preview_id(" in inspect.getsource(fn)


def test_stitch_failure_never_registers_a_paired_template() -> None:
    """拼接失败时不能把第 1 张图的单图结果固化成模板（静态接线契约）。

    模板一旦写错会带偏以后所有同形状概览图，属于「错误自我强化」，
    因此 `recognize()` 必须在 `stitch_failed` 时直接给出
    `written=False / stitch_failed`，而不是调 `_capture_paired_template()`。
    """
    src = inspect.getsource(recognize)
    assert 'cap = {"written": False, "reason": "stitch_failed"}' in src
    guard = src.index('if stitch_failed:')
    call = src.index('_capture_paired_template(')
    assert guard < call, "保护分支必须在模板注册调用之前"
