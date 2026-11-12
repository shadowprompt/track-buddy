"""拼接后端的公共契约。

## 为什么把「边的来源」抽出来

多图拼接的真正分歧只在**第一步**：图与图之间的重叠关系（以及变换矩阵）
从哪里来。拿到 `link_map` 之后，连通块、位姿树、贴图、逐像素复核、
起终点徽章守恒都是同一套代码，与来源无关。

把这一步做成可插拔后端，A/B 对比才是公平的：
- `sift`      —— 21 对全组合 SIFT + RANSAC（项目现用方式）
- `llm`       —— 多模态模型直接判断相邻关系与相对位移，跳过 SIFT
- `llm_guided`—— 模型只圈定候选图对，仍由 SIFT 在这些对上求精确变换

三种后端产出的是**同一个 `StitchResult`**，所以 `verify` / `verify_score` /
`canvas_size` / `badge_conservation` / `used` 可以直接横向比较。

## 一条硬规矩：绝不静默回退

用户选了某个后端，却因为「没配 key / 模型失败 / 返回空」偷偷用 SIFT 出结果，
A/B 数据就被污染了，而且看不出来。所以后端不可用时 `StitchPlan.error`
必须非空，由 `run_stitch()` 抛给调用方，明确报失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..multishot import PairLink


@dataclass
class StitchPlan:
    """一个后端对「边从哪来」的回答。

    links           直接给出变换矩阵（纯语义配准用），此时跳过逐对配准
    pair_candidates 只给出需要尝试的图对（引导式配准用），仍走 SIFT
    diagnostics     写入结果的诊断信息，供 A/B 复盘
    notes           追加到拼接 notes 的说明
    error           非空表示后端无法工作，调用方必须报失败而非回退
    """

    links: dict[tuple[int, int], PairLink] | None = None
    pair_candidates: list[tuple[int, int]] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class BackendSpec:
    """一个可选择的拼接后端。"""

    id: str
    label: str
    description: str
    plan: Callable[..., StitchPlan]
    requires_llm: bool = False
    # 是否跳过 SIFT 逐对配准（用于结果里正确解读 links 的 inliers）
    skips_sift: bool = False


class StitchBackendError(Exception):
    """后端不可用或执行失败。绝不回退到其它后端。"""


class UnknownStitchBackend(StitchBackendError):
    """请求了未注册的后端名。"""
