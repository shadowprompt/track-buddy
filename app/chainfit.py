"""有序路名链求解器：用「路名骨架」同时定出尺度、位置与朝向。

完整流程
--------
  1. 每条路名切成空间实例
  2. 链式 DP 选出空间连贯的实例组合（骨架）
  3. 骨架反推尺度（相邻路名的地理间距 / 像素弧长差，取中位数）
  4. 用骨架的接触点做 Procrustes，解出完整相似变换 (s, θ, t)
  5. 用「DP 对齐残差」做目标函数，外层多起点扫描，选最优

为什么要多起点
--------------
第 3 步用「接触点」反推尺度，而接触点依赖当前变换——
若初始位置偏了 2 公里，选中的实例就可能是错的那一个（自证循环）。
所以外层用**多组初始尺度**分别求解，各自收敛后用统一目标函数打分。

实测（上海城市级截图）
----------------------
  初始（锚点配准）：尺度 15.73 m/px，轨迹 15739 米，起点落在**浦东侧**
  本模块求解后：尺度约 12.5 m/px，轨迹约 12000 米，起点回到**浦西侧**
  独立标尺真值：12.47 / 12.58 / 13.42（车站对）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .roadchain import (
    ChainResult, RoadInstance, build_candidates, contact_points,
    scale_from_chain, scale_from_chain_span, select_chain,
)


@dataclass
class ChainFit:
    """路名链求解结果。"""

    ok: bool = False
    # 解出的相似变换（作用在「y 已翻转的像素」上）
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0
    rotation_deg: float = 0.0
    # 目标函数（DP 对齐后的加权残差，米）
    value: float = float("inf")
    base_value: float = float("inf")
    # 逐条路名的残差
    residuals: list[float] = field(default_factory=list)
    # 参与尺度反推的逐段比值
    scale_ratios: list[float] = field(default_factory=list)
    # 选中的实例（按路名顺序）
    chain: ChainResult | None = None
    notes: list[str] = field(default_factory=list)

    def apply(self, px: np.ndarray) -> np.ndarray:
        """像素 (N,2) -> 墨卡托米 (N,2)。"""
        p = np.asarray(px, dtype=float).copy()
        p[:, 1] = -p[:, 1]
        return (self.matrix @ p.T).T + self.translation


def _procrustes(src: np.ndarray, dst: np.ndarray,
                w: np.ndarray | None = None) -> tuple[float, float, np.ndarray]:
    """加权相似变换闭式解：min Σ wᵢ|s·R(θ)·srcᵢ + t - dstᵢ|²。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) < 2 or len(src) != len(dst):
        return 1.0, 0.0, np.zeros(2)
    if w is None:
        w = np.ones(len(src))
    w = np.asarray(w, dtype=float).reshape(-1)
    w = w / (w.sum() or 1.0)

    sxy = src.copy()
    sxy[:, 1] = -sxy[:, 1]      # 像素 y 向下 -> 翻成北向上

    ms = (w[:, None] * sxy).sum(axis=0)
    md = (w[:, None] * dst).sum(axis=0)
    A = sxy - ms
    B = dst - md

    H = (A * w[:, None]).T @ B
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = float(np.sign(np.linalg.det(V @ U.T))) or 1.0
    R = V @ np.diag([1.0, d]) @ U.T
    denom = float((w[:, None] * A * A).sum())
    s = float((S * np.array([1.0, d])).sum() / max(denom, 1e-12))
    t = md - s * (R @ ms)
    theta = math.atan2(R[1, 0], R[0, 0])
    return s, theta, t


def _compose(s: float, theta: float, t: np.ndarray) -> np.ndarray:
    c, si = math.cos(theta), math.sin(theta)
    R = np.array([[c, -si], [si, c]])
    return R * s


def _apply(px: np.ndarray, s: float, theta: float, t: np.ndarray) -> np.ndarray:
    p = np.asarray(px, dtype=float).copy()
    p[:, 1] = -p[:, 1]
    m = _compose(s, theta, t)
    return (m @ p.T).T + t


def _arc_px(px: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _dp_align(traj: np.ndarray, chain: ChainResult,
              n_seg: int = 30, pos_prior: float = 30.0
              ) -> list[tuple[int, int]] | None:
    """路名序列 -> 轨迹分段的单调 DP 对齐。

    返回 [(i0, i1) 索引区间]，与 chain.chosen 一一对应。
    """
    chosen = chain.chosen
    n = len(chosen)
    live = [i for i, c in enumerate(chosen) if c.n > 0]
    if len(live) < 2:
        return None

    ml = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(ml)])
    total = float(cum[-1]) or 1.0
    n_seg = max(len(live), min(n_seg, len(traj)))
    bounds = [total * i / n_seg for i in range(n_seg + 1)]

    seg_pts: list[np.ndarray] = []
    for i in range(n_seg):
        m = (cum >= bounds[i]) & (cum <= bounds[i + 1])
        p = traj[m]
        if len(p) == 0:
            j = int(np.argmin(np.abs(cum - bounds[i])))
            p = traj[max(0, j - 1):j + 2]
        if len(p) > 20:
            p = p[np.linspace(0, len(p) - 1, 20).astype(int)]
        seg_pts.append(p)

    # 代价矩阵：live 中的第 k 条路名 -> n_seg 段
    K = len(live)
    cost = np.zeros((K, n_seg))
    for k, i in enumerate(live):
        je = (k * (n_seg - 1) / max(1, K - 1)) if K > 1 else 0.0
        for j, p in enumerate(seg_pts):
            if len(p) == 0:
                cost[k, j] = 1e5
                continue
            cost[k, j] = (chosen[i].median_dist_to(p)
                          + pos_prior * abs(j - je))

    INF = float("inf")
    dp = np.full((K, n_seg), INF)
    back = np.full((K, n_seg), -1, dtype=int)
    dp[0, :] = cost[0, :]
    for k in range(1, K):
        bp, bi = INF, -1
        for j in range(n_seg):
            if j - 1 >= 0 and dp[k - 1, j - 1] < bp:
                bp = dp[k - 1, j - 1]
                bi = j - 1
            if bp < INF:
                dp[k, j] = bp + cost[k, j]
                back[k, j] = bi
    last = int(np.argmin(dp[K - 1]))
    if not np.isfinite(dp[K - 1, last]):
        return None

    spans: list[tuple[int, int]] = [(0, 0)] * n
    j = last
    for k in range(K - 1, -1, -1):
        lo, hi = bounds[j], bounds[j + 1]
        idx = np.where((cum >= lo) & (cum <= hi))[0]
        if len(idx) >= 2:
            spans[live[k]] = (int(idx[0]), int(idx[-1]))
        j = back[k, j]
        if j < 0:
            break
    return spans


def _evaluate(traj: np.ndarray, chain: ChainResult,
              weights: np.ndarray) -> tuple[float, list[float]]:
    """DP 对齐后的加权残差。"""
    spans = _dp_align(traj, chain)
    if spans is None:
        return float("inf"), []
    val = 0.0
    ws = 0.0
    res: list[float] = []
    for i, (i0, i1) in enumerate(spans):
        inst = chain.chosen[i]
        if inst.n == 0 or i1 - i0 < 1:
            res.append(1e5)
            continue
        pts = traj[i0:i1 + 1]
        if len(pts) > 30:
            pts = pts[np.linspace(0, len(pts) - 1, 30).astype(int)]
        r = inst.median_dist_to(pts)
        res.append(r)
        val += weights[i] * r
        ws += weights[i]
    return val / (ws or 1.0), res


def _solve_once(
    px: np.ndarray,
    arc: np.ndarray,
    candidates: list[list[RoadInstance]],
    weights: np.ndarray,
    init_scale: float,
    init_t: np.ndarray,
    iters: int = 5,
) -> tuple[float, float, np.ndarray, ChainResult, list[float], str] | None:
    """从给定初始尺度出发迭代求解。

    每轮：
      1. 链式 DP 选实例 -> 骨架
      2. 骨架反推尺度
      3. 用骨架接触点做 Procrustes -> (s, θ, t)

    尺度估计用**整体跨度**为主、逐段中位数为辅：
    逐段比值的方差太大（实测同一例中 10.2~26.0），中位数不可靠；
    整体跨度只用首尾两点，相对误差小得多。
    """
    s = init_scale
    theta = 0.0
    t = np.array(init_t, dtype=float)
    chain = ChainResult()
    ratios: list[float] = []
    span_note = ""
    for _ in range(iters):
        traj = _apply(px, s, theta, t)
        chain = select_chain(candidates, traj, arc)
        if not chain.ok:
            return None
        contact_points(chain, traj, arc)
        s_span, span_note = scale_from_chain_span(chain)
        s_seg, ratios, _ = scale_from_chain(chain)
        # 两个尺度估计：跨度法优先（相对误差小），段中位数兜底
        if s_span > 0:
            s_new = s_span
        elif s_seg > 0:
            s_new = s_seg
        else:
            return None
        # 用骨架接触点解 Procrustes（尺度由接触点自己定，不用 s_new）
        src: list[np.ndarray] = []
        dst: list[np.ndarray] = []
        wts: list[np.ndarray] = []
        for i, inst in enumerate(chain.chosen):
            if inst.n == 0:
                continue
            ga = chain.contact_geo[i]
            pa = chain.contact_arc_px[i]
            if not (math.isfinite(ga[0]) and math.isfinite(pa)):
                continue
            j = int(np.argmin(np.abs(arc - pa)))
            src.append(px[j:j + 1])
            dst.append(np.array([[ga[0], ga[1]]]))
            wts.append(np.array([weights[i]]))
        if len(src) < 3:
            # 对应点不够，退回「只改尺度」
            s = s_new
            break
        s2, th2, t2 = _procrustes(
            np.vstack(src), np.vstack(dst), np.concatenate(wts))
        if abs(s2 - s) < 1e-4 and abs(th2 - theta) < 1e-7:
            s, theta, t = s2, th2, t2
            break
        s, theta, t = s2, th2, t2
    return s, theta, t, chain, ratios, span_note


def fit_chain(
    px: np.ndarray,
    road_names: list[tuple[str, list[list[tuple[float, float]]]]],
    weights: list[float] | None = None,
    init_scale: float = 1.0,
    init_translation: np.ndarray | None = None,
    scale_guess: float | None = None,
    n_starts: int = 7,
    iters: int = 5,
) -> ChainFit:
    """主入口：用有序路名拟合相似变换。

    px:              (N,2) 像素轨迹
    road_names:      [(路名, [段, ...]), ...]，段是墨卡托米坐标
    weights:         每条路名的权重（默认按唯一性）
    init_scale:      初始尺度（米/像素）
    init_translation:初始平移（墨卡托米）
    scale_guess:     尺度的先验猜测；给出时多起点会围绕它铺开
    """
    out = ChainFit()
    usable = [(nm, segs) for nm, segs in road_names if segs]
    if len(usable) < 3 or len(px) < 10:
        out.notes.append("可用路名不足或轨迹点太少，无法用路名链求解")
        return out

    if weights is None:
        weights = [1.0] * len(usable)
    w = np.asarray(weights, dtype=float)[:len(usable)]
    if w.sum() <= 0:
        w = np.ones(len(usable))

    candidates = build_candidates([(nm, segs) for nm, segs in usable])
    if sum(1 for c in candidates if c) < 3:
        out.notes.append("路名切分出的实例太少")
        return out

    arc = _arc_px(px)
    if init_translation is None:
        init_translation = np.zeros(2)

    # 多起点的尺度铺开
    center = scale_guess if scale_guess else init_scale
    lo_r = center * 0.70
    hi_r = center * 1.30
    starts = np.linspace(lo_r, hi_r, max(3, n_starts))

    best: tuple[float, float, float, np.ndarray, ChainResult, list[float], str] | None = None
    for s0 in starts:
        t0 = np.array(init_translation, dtype=float)
        r = _solve_once(px, arc, candidates, w, float(s0), t0, iters=iters)
        if r is None:
            continue
        s, theta, t, chain, ratios, span_note = r
        traj = _apply(px, s, theta, t)
        v, res = _evaluate(traj, chain, w)
        if not math.isfinite(v):
            continue
        if best is None or v < best[0]:
            best = (v, s, theta, t, chain, ratios, span_note)

    if best is None:
        out.notes.append("所有起点都没能收敛")
        return out

    v, s, theta, t, chain, ratios, span_note = best
    traj = _apply(px, s, theta, t)
    _, res = _evaluate(traj, chain, w)

    out.ok = True
    out.matrix = _compose(s, theta, t)
    out.translation = t
    out.scale = s
    out.rotation_deg = math.degrees(theta)
    out.value = v
    out.residuals = res
    out.scale_ratios = ratios
    out.chain = chain
    out.notes.append(
        f"路名链求解：尺度 {init_scale:.2f} -> {s:.2f} 米/像素"
        f"（×{s / max(init_scale, 1e-9):.2f}），旋转 {out.rotation_deg:+.2f}°，"
        f"目标值 {v:.0f}"
    )
    if span_note:
        out.notes.append(f"  尺度来源：{span_note}")
    return out
