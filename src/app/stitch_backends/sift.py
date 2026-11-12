"""项目现用的默认后端：逐对 SIFT + RANSAC。

这个后端**不提供任何边**，等于告诉 `stitch()` 走原生路径。
之所以也注册成一个后端，是为了让三种方式共用同一个入口、
同一套结果字段，A/B 时只有「选中的 id」不同。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BackendSpec, StitchPlan
from .registry import register


def plan_sift(
    images: list[np.ndarray], cfg: Any = None, client: Any = None,
) -> StitchPlan:
    return StitchPlan(diagnostics={
        "strategy": "全组合逐对 SIFT + RANSAC + 候选峰 NCC",
        "pairs_attempted": len(images) * (len(images) - 1) // 2,
    })


register(BackendSpec(
    id="sift",
    label="SIFT 特征配准（项目现用）",
    description=(
        "逐对全组合特征匹配求变换，按内点×相似度选最大生成树；"
        "确定性、可复现，不依赖外部服务。"),
    plan=plan_sift,
    requires_llm=False,
    skips_sift=False,
))
