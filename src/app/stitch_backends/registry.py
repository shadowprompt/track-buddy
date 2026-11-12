"""拼接后端注册表与统一入口。

服务端只认 ``run_stitch()`` 一个函数，新增后端只要实现 ``plan()`` 并
``register()`` 一下，不必改接口、前端也由 ``list_backends()`` 自动列出。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import CONFIG
from ..llm import LLMClient
from ..multishot import StitchResult, stitch as _stitch
from .base import BackendSpec, StitchBackendError, UnknownStitchBackend

DEFAULT_BACKEND = "sift"
_SPECS: dict[str, BackendSpec] = {}


def register(spec: BackendSpec) -> None:
    if spec.id in _SPECS:
        raise ValueError(f"拼接后端重复注册: {spec.id}")
    _SPECS[spec.id] = spec


def get_backend(name: str) -> BackendSpec:
    key = (name or "").strip().lower()
    if key not in _SPECS:
        raise UnknownStitchBackend(
            f"未知的拼接后端 {name!r}；可选: {', '.join(sorted(_SPECS))}")
    return _SPECS[key]


def llm_status(cfg: Any = None, client: Any = None) -> tuple[bool, str]:
    """大模型后端当前是否具备运行条件（只查配置，不发网络请求）。"""
    c = cfg or CONFIG
    if not getattr(c, "llm_enabled", True):
        return False, "配置里 LLM 已关闭（llm_enabled=False）"
    llm = client or LLMClient()
    if not getattr(llm, "available", False):
        return False, (
            "未配置大模型服务：请在 .env 里设置 LLM_BASE_URL、LLM_API_KEY"
            "（可选 LLM_MODEL），保存后自行重启服务")
    return True, ""


def list_backends(cfg: Any = None, client: Any = None) -> list[dict[str, Any]]:
    """给前端的能力清单：哪些能选、不能选的原因是什么。"""
    ok, reason = llm_status(cfg, client)
    out: list[dict[str, Any]] = []
    for spec in _SPECS.values():  # dict 保序 = 注册顺序 = 前端展示顺序
        available = ok if spec.requires_llm else True
        out.append({
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "requires_llm": spec.requires_llm,
            "skips_sift": spec.skips_sift,
            "available": available,
            "reason": "" if available else reason,
        })
    return out


def run_stitch(
    images: list[np.ndarray],
    backend: str = DEFAULT_BACKEND,
    *,
    cfg: Any = None,
    client: Any = None,
    **stitch_kwargs: Any,
) -> StitchResult:
    """按指定后端拼接，返回与后端无关的同一个 ``StitchResult``。

    ⚠️ 任何失败都抛出 ``StitchBackendError``，**不会**改用别的后端出结果。
    调用方必须把失败如实呈现给用户，否则 A/B 对比会被悄悄污染。
    """
    name = (backend or DEFAULT_BACKEND).strip().lower()
    spec = get_backend(name)
    c = cfg or CONFIG
    llm = client or LLMClient()

    if spec.requires_llm:
        ok, reason = llm_status(c, llm)
        if not ok:
            raise StitchBackendError(f"[{spec.label}] {reason}")

    if len(images) <= 1:
        # 单图无需拼接，也就不该为了「拼接」去调一次模型、白花钱。
        res = _stitch(images, backend=name, **stitch_kwargs)
        res.backend_diagnostics = {
            "single_image_no_stitch": True,
            "requested_backend": name,
        }
        return res

    plan = spec.plan(images, cfg=c, client=llm)
    if plan.error:
        raise StitchBackendError(f"[{spec.label}] {plan.error}")

    res = _stitch(
        images,
        links_override=plan.links,
        pair_candidates=plan.pair_candidates,
        backend=name,
        **stitch_kwargs,
    )
    res.backend_diagnostics = dict(plan.diagnostics)
    if plan.notes:
        res.notes = list(plan.notes) + res.notes
    return res
