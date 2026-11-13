"""LLMClient 流式请求回归。

背景（2026-11-13 实测）：中转端点对「非流式 + 7 张图」的请求在约 30 秒
返回 `HTTP 102` 空响应，而推理型模型实际需要约 59 秒才输出完；改成
`stream=True` 后 58.97 秒拿到合法 JSON（reasoning_tokens=2668）。
这组测试锁住：流式参数、SSE 拼接、中文编码、reasoning 不污染答案。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.llm import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    MAX_TOKENS_CEILING,
    LLMClient,
)
from app.stitch_backends.llm import plan_llm  # noqa: E402


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        lines: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
        body: dict | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self.headers = headers or {}
        self._body = body or {}
        self.text = text

    def iter_lines(self):
        return iter(self._lines)

    def json(self):
        return self._body


def _client(monkeypatch) -> LLMClient:
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT", "600")
    return LLMClient()


def _sse(chunks: list[dict], tail: bytes = b"data: [DONE]") -> list[bytes]:
    return [b"data: " + json.dumps(c, ensure_ascii=False).encode("utf-8") for c in chunks] + [tail]


def test_stream_payload_and_sse_joining(monkeypatch) -> None:
    c = _client(monkeypatch)
    lines = _sse([
        {"choices": [{"delta": {"reasoning_content": "这段是思考，不能进答案"}}]},
        {"choices": [{"delta": {"content": '{"pairs":[{"i":1,"j":2,'}}]},
        {"choices": [{"delta": {"content":
            '"dx_ratio":0.75,"dy_ratio":0,"confidence":0.9}],"notes":"中文说明"}'}}]},
    ])
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse(lines=lines, headers={"content-type": "text/event-stream"})

    c._session.post = fake_post
    res = c.chat_json("system", "user")

    assert res.ok, res.error
    assert res.data is not None
    assert res.data["pairs"][0]["confidence"] == 0.9
    assert res.data["notes"] == "中文说明"          # UTF-8 不能解成乱码
    assert "这段是思考" not in res.text              # reasoning_content 不得混入
    assert captured["json"]["stream"] is True
    assert captured["json"]["max_tokens"] == min(
        DEFAULT_MAX_TOKENS, MAX_TOKENS_CEILING)
    assert captured["json"]["temperature"] == 0
    assert captured["stream"] is True
    assert captured["timeout"] == 600.0


def test_non_200_is_reported_not_swallowed(monkeypatch) -> None:
    c = _client(monkeypatch)
    c._session.post = lambda url, **kw: _FakeResponse(
        status_code=102, headers={"content-type": "text/event-stream"}, text="")
    res = c.chat_json("system", "user")
    assert not res.ok
    assert "HTTP 102" in res.error


def test_plain_json_endpoint_still_works(monkeypatch) -> None:
    """换到不支持 SSE 的 OpenAI 兼容端点时，也要能读出答案。"""
    c = _client(monkeypatch)
    c._session.post = lambda url, **kw: _FakeResponse(
        headers={"content-type": "application/json"},
        body={"choices": [{"message": {"content": '{"ok": true}'}}]},
    )
    res = c.chat_json("system", "user")
    assert res.ok and res.data == {"ok": True}


def test_timeout_default_is_ten_minutes(monkeypatch) -> None:
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)
    assert LLMClient().timeout == 600.0


def test_max_tokens_stays_within_endpoint_limit() -> None:
    """端点实测硬上限 65536；默认 16384 必须落在里面。"""
    assert MAX_TOKENS_CEILING == 65536
    assert 0 < DEFAULT_MAX_TOKENS <= MAX_TOKENS_CEILING


def test_max_tokens_is_clamped_even_if_constant_is_raised(monkeypatch) -> None:
    """本地把常量调成 512000 也不能真发出去（实测会 HTTP 400）。

    2026-11-13 09:28 的真实事故：有人把 ``DEFAULT_MAX_TOKENS`` 手动改成
    512000，重启后纯 VLM 后端每次都立刻返回
    ``invalid_parameter: Field 'max_tokens' must be at most 65536``。
    运行时钳制保证这类手调不再让整个后端必然失败。
    """
    c = _client(monkeypatch)
    monkeypatch.setattr("app.llm.DEFAULT_MAX_TOKENS", 512000)
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _FakeResponse(
            lines=_sse([{"choices": [{"delta": {"content": '{"ok":true}'}}]}]),
            headers={"content-type": "text/event-stream"},
        )

    c._session.post = fake_post
    res = c.chat_json("system", "user")
    assert res.ok, res.error
    assert captured["json"]["max_tokens"] == MAX_TOKENS_CEILING == 65536


def test_http_400_keeps_the_reason_instead_of_truncating_it(monkeypatch) -> None:
    """错误文本不能再只留前 200 字：参数原因常在 JSON 后半段。"""
    c = _client(monkeypatch)
    body = "x" * 260 + "Field 'max_tokens' must be at most 65536"
    c._session.post = lambda url, **kw: _FakeResponse(
        status_code=400,
        headers={"content-type": "application/json"},
        text=body,
    )
    res = c.chat_json("system", "user")
    assert not res.ok
    assert "HTTP 400" in res.error
    assert "must be at most 65536" in res.error


def test_stream_response_feeds_semantic_pairs(monkeypatch) -> None:
    """真实 7 图响应格式 → _semantic_pairs → StitchPlan.links 全链路。"""
    c = _client(monkeypatch)
    answer = json.dumps({
        "pairs": [
            {"i": 1, "j": 7, "dx_ratio": -0.25, "dy_ratio": -0.05, "confidence": 0.82},
            {"i": 3, "j": 5, "dx_ratio": 0.75, "dy_ratio": -0.03, "confidence": 0.9},
            {"i": 4, "j": 6, "dx_ratio": -0.57, "dy_ratio": 0.0, "confidence": 0.93},
        ],
        "notes": "依据道路连续性判断",
    }, ensure_ascii=False)
    lines = _sse([
        {"choices": [{"delta": {"reasoning_content": "推理中"}}]},
        {"choices": [{"delta": {"content": answer[:60]}}]},
        {"choices": [{"delta": {"content": answer[60:]}}]},
    ])
    c._session.post = lambda url, **kw: _FakeResponse(
        lines=lines, headers={"content-type": "text/event-stream"})

    imgs = [np.zeros((80, 50, 3), np.uint8) for _ in range(7)]
    plan = plan_llm(imgs, client=c)

    assert not plan.error, plan.error
    assert plan.diagnostics["model"] == "test-model"
    assert plan.diagnostics["images_sent"] == 7
    assert plan.diagnostics["pairs_accepted"] == 3
    # notes 是字符串时不能按字符拆开
    assert plan.diagnostics["model_notes"] == ["依据道路连续性判断"]
    assert len(plan.links or {}) == 3
    assert all(link.source == "llm" for link in (plan.links or {}).values())
