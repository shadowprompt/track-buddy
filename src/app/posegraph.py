"""多图拼接的位姿图优化：把链式累积漂移**分摊**到每张图。

问题
----
``multishot.stitch()`` 用 BFS 生成树把 pairwise 变换一路乘下去：
``T[j] = T[i] · inv(m_ij)``。每对配准都有几像素噪声，链长 10 时
末端误差会线性累积；而**非生成树的冗余边**（图 1 和图 4 也重叠）
被完全丢弃，本来可以用来闭环平差。

做法
----
把每张图当作一个节点、每条重叠关系当作一条边，求一组整体平移
修正量 δ_i，最小化所有边的加权残差：

    Σ w_ij · ‖ (p_j + δ_j) − (p_i + δ_i) − d_ij ‖²

其中 d_ij 是边测量给出的平移差，p_i 是 BFS 初值。这是标准拉普拉斯
最小二乘，固定根节点 δ_root = 0 消除 gauge 自由度。

⚠️ 只优化**平移**（累积漂移主要是平移），不改各图的旋转/尺度，
也不改主请求后续的地理配准。默认关闭，开启后仅影响拼接画布。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class PoseEdge:
    """一条从 i 到 j 的相对平移观测。"""

    i: int
    j: int
    dx: float
    dy: float
    weight: float = 1.0


@dataclass
class PoseGraphResult:
    ok: bool
    deltas: list[tuple[float, float]] = field(default_factory=list)
    residual_before: float = 0.0
    residual_after: float = 0.0
    n_edges: int = 0
    n_nodes: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "deltas": [[round(float(a), 2), round(float(b), 2)]
                       for a, b in self.deltas],
            "residual_before": round(self.residual_before, 3),
            "residual_after": round(self.residual_after, 3),
            "n_edges": self.n_edges,
            "n_nodes": self.n_nodes,
            "reason": self.reason,
        }


def _weighted_rms(res: np.ndarray, weights: np.ndarray) -> float:
    if len(res) == 0:
        return 0.0
    w = np.maximum(weights, 1e-9)
    return float(np.sqrt(np.sum(w[:, None] * np.square(res)) / w.sum() / 2.0))


def optimize_translations(
    n_nodes: int,
    edges: Iterable[PoseEdge],
    init_positions: Sequence[Sequence[float]] | None = None,
    root: int = 0,
) -> PoseGraphResult:
    """加权最小二乘求各节点平移修正量。

    ``init_positions`` 给定时用它算优化前的边残差（用于 A/B 对比）；
    不给时按全 0 初值处理。
    """
    es = [e for e in edges if 0 <= e.i < n_nodes and 0 <= e.j < n_nodes
          and e.i != e.j and e.weight > 0]
    empty = PoseGraphResult(ok=False, n_edges=len(es), n_nodes=n_nodes,
                            deltas=[(0.0, 0.0)] * n_nodes)
    if n_nodes < 3 or len(es) < n_nodes - 1:
        empty.reason = "节点或边不足，位姿图优化退化为原链式结果"
        return empty

    p = (np.asarray(init_positions, dtype=float)
         if init_positions is not None else np.zeros((n_nodes, 2), dtype=float))
    if p.shape != (n_nodes, 2):
        empty.reason = "初始平移维度不匹配"
        return empty

    idx_i = np.asarray([e.i for e in es], dtype=int)
    idx_j = np.asarray([e.j for e in es], dtype=int)
    d = np.asarray([[e.dx, e.dy] for e in es], dtype=float)
    w = np.asarray([e.weight for e in es], dtype=float)
    residual = d - (p[idx_j] - p[idx_i])       # (m, 2)，δ 要逼近它
    before = _weighted_rms(residual, w)

    A = np.zeros((len(es), n_nodes), dtype=float)
    rows = np.arange(len(es))
    A[rows, idx_i] = -1.0
    A[rows, idx_j] = 1.0
    sw = np.sqrt(w)
    free = [k for k in range(n_nodes) if k != root]
    try:
        sol, *_ = np.linalg.lstsq(A[:, free] * sw[:, None],
                                  residual * sw[:, None], rcond=None)
    except np.linalg.LinAlgError as exc:
        empty.reason = f"最小二乘求解失败：{type(exc).__name__}"
        empty.residual_before = round(before, 3)
        return empty

    delta = np.zeros((n_nodes, 2), dtype=float)
    delta[free] = sol
    after_res = residual - (delta[idx_j] - delta[idx_i])
    return PoseGraphResult(
        ok=True,
        deltas=[(float(a), float(b)) for a, b in delta],
        residual_before=round(before, 3),
        residual_after=round(_weighted_rms(after_res, w), 3),
        n_edges=len(es),
        n_nodes=n_nodes,
        reason="按全部重叠边做全局平差，漂移分摊到各图",
    )
