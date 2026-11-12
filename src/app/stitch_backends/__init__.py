"""可插拔的多图拼接后端。

服务端用法::

    from .stitch_backends import run_stitch

    st = run_stitch(imgs, backend="sift")        # 项目现用方式
    st = run_stitch(imgs, backend="llm")         # 纯大模型语义配准
    st = run_stitch(imgs, backend="llm_guided")  # 模型候选 + SIFT 精配准

`run_stitch()` 永远返回 `multishot.StitchResult`，后端差异只体现在
「边从哪来」与 `backend_diagnostics`，便于直接做 A/B 对比。
"""

from __future__ import annotations

from .base import (
    BackendSpec,
    StitchBackendError,
    StitchPlan,
    UnknownStitchBackend,
)
from .registry import (
    DEFAULT_BACKEND,
    get_backend,
    list_backends,
    llm_status,
    register,
    run_stitch,
)

# 导入即注册；顺序决定前端下拉的展示顺序（sift → llm → llm_guided）
from . import sift as _sift  # noqa: F401
from . import llm as _llm  # noqa: F401

__all__ = [
    "DEFAULT_BACKEND",
    "BackendSpec",
    "StitchBackendError",
    "StitchPlan",
    "UnknownStitchBackend",
    "get_backend",
    "list_backends",
    "llm_status",
    "register",
    "run_stitch",
]
