"""可选的大模型客户端（OpenAI 兼容协议）。

为什么要有这个模块
------------------
实测证明：**外部路径规划 API 不能代替用户的语义输入**。
用高德「步行路径规划」规划全程，14 步里只命中用户路名 1 步
（它走的是枫林路/建国西路/华山路/延安西路/天山路——
那是「步行最短路」，不是用户实际骑的路线）。
逐对连接相邻路名倒是 6/9 成功（小木桥路→龙华中路经零陵路 ✓），
但长段会失败，且它给的是「最优路」而非「用户走过的路」。

所以分工是清楚的：

    **大模型负责语义层** —— 用户的路名序列合不合理、顺序对不对、
    哪条路可能是记错的、截图的缩放级别如何、底图上还有哪些地名。
    **代码负责坐标层** —— OSM 几何、DP 对齐、起终点锚定、相似变换。

大模型**不能**提供精确坐标（它没有 OSM 级别的几何），
也**不能**提供用户的实际路线（那只能来自用户本人）。

配置（都可选，缺省则整个模块跳过，现有行为完全不变）::

    LLM_BASE_URL=https://api.example.com/v1
    LLM_API_KEY=sk-xxxx
    LLM_MODEL=gpt-4o-mini
    LLM_TIMEOUT=600          # 秒；推理型多模态模型跑 7 张图实测约 59 秒

请求一律走 ``stream=True``：实测非流式请求在约 30 秒被中转网关掐断
（``HTTP 102`` 空响应），而模型实际要 ~59 秒才输出完；流式分片会持续
保持连接。响应里的 ``reasoning_content`` 不是答案，只收集 ``content``。
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# 推理型多模态模型会把绝大部分 token 花在 reasoning 上（实测 7 张图任务
# reasoning_tokens=2668 / completion_tokens=2794）。上限给小了，模型在
# 「想完」之前就被截断，content 为空，JSON 永远解析不出来。
DEFAULT_MAX_TOKENS = 16384
# 默认请求超时（秒）。这个值在流式请求下既是「单次 read 超时」，也是
# 整次调用的总时长上限；25 秒对推理型模型的 7 图任务必然不够。
DEFAULT_TIMEOUT = 600.0


@dataclass
class LLMResult:
    """一次 LLM 调用的结果。"""

    ok: bool = False
    text: str = ""
    data: dict[str, Any] | None = None
    error: str = ""
    elapsed_ms: int = 0
    model: str = ""


class LLMClient:
    """极简的 OpenAI 兼容客户端（只依赖 requests）。"""

    def __init__(self, cfg: Any = None) -> None:
        self.base_url = _pick(cfg, "LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
        self.api_key = _pick(cfg, "LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
        self.model = _pick(cfg, "LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-4o-mini"))
        self.timeout = float(_pick(
            cfg, "LLM_TIMEOUT", os.environ.get("LLM_TIMEOUT", str(DEFAULT_TIMEOUT))))
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "meituan-ride/0.1 (+local)"})
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    @staticmethod
    def _read_chat_content(r: requests.Response, deadline: float) -> str:
        """把一次响应统一读成模型输出的文本。

        流式端点返回 SSE（``data: {...}`` 分片 + ``[DONE]``）；非流式端点
        返回普通 JSON。只收集 ``content`` 分片——推理型模型的
        ``reasoning_content`` 不是答案，拼进去会破坏 JSON 解析。
        逐行按 UTF-8 解码，避免 ``decode_unicode=True`` 把中文解成乱码。
        """
        ctype = (r.headers.get("content-type") or "").lower()
        if "event-stream" not in ctype:
            body = r.json()
            choice = (body.get("choices") or [{}])[0]
            return str((choice.get("message") or {}).get("content", "") or "")
        parts: list[str] = []
        for raw in r.iter_lines():
            if time.time() > deadline:
                raise TimeoutError("流式响应超过总时长上限")
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except (ValueError, TypeError):
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            parts.append(str(delta.get("content") or ""))
        return "".join(parts)

    def chat_json(
        self,
        system: str,
        user: str,
        images: list[str] | None = None,
        schema_hint: str = "",
    ) -> LLMResult:
        """调用一次模型，要求返回 JSON。失败时返回 ok=False（不抛异常）。"""
        res = LLMResult(model=self.model)
        if not self.available:
            res.error = "未配置 LLM_BASE_URL / LLM_API_KEY"
            return res
        t0 = time.time()
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for path in images or []:
            data_url = _to_data_url(path)
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [{"role": "system", "content": system}]
        if schema_hint:
            messages.append({"role": "system", "content": schema_hint})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": DEFAULT_MAX_TOKENS,
            # 非流式请求实测在 ~30s 被网关掐断（HTTP 102 空响应/一直挂住），
            # 而 7 张图的推理要 ~59s；流式分片能持续保持连接，是跑完的前提。
            "stream": True,
        }
        try:
            deadline = t0 + self.timeout
            r = self._session.post(
                self.base_url.rstrip("/") + "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
                stream=True,
            )
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}: {r.text[:200]}"
                self.last_error = res.error
                return res
            res.text = self._read_chat_content(r, deadline)
            res.data = _extract_json(res.text)
            res.ok = res.data is not None
            if not res.ok:
                res.error = "模型未返回可解析的 JSON"
        except Exception as exc:  # noqa: BLE001
            res.error = f"{type(exc).__name__}: {exc}"
        self.last_error = res.error
        res.elapsed_ms = int((time.time() - t0) * 1000)
        return res


def _pick(cfg: Any, attr: str, fallback: Any) -> Any:
    if cfg is not None:
        v = getattr(cfg, attr, None)
        if v:
            return v
    return fallback


def _to_data_url(path: str) -> str:
    """把本地图片转成 data URL（多模态调用用）。"""
    p = Path(path)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON（容忍 ```json 包裹和前后废话）。"""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    # 退而求其次：找第一个 {...}
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(s[a:b + 1])
            return obj if isinstance(obj, dict) else None
        except ValueError:
            return None
    return None
