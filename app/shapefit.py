"""形状重建：把「城市级截图里的直线轨迹」还原成沿真实街道的走向。

问题
----
城市级缩放（一张图覆盖 20 公里）时，11 公里的骑行在截图上只占 ~680 像素，
美团画的轨迹线**本身就是一条直线**——街道锯齿在截图阶段就丢了。
路名能修正**位置**（起终点锚定，实测准到 300-900 米），但修不了**形状**。

路线约束（物理上必须成立）
------------------------
1. 真实路径长度 >= 直线距离；
2. 轨迹在某条路上那一段的弧长，**不应超过该路名实例的总长**。

旧的 `_align_segments` 违反了第 2 条：它把轨迹等分成 30 段（每段 437 米），
每条路名只匹配**恰好 1 段**，于是
  * 中山南二路（实例长 4963 米）只拿到 437 米（0.09 倍）；
  * 船厂路（实例长 350 米）拿到 437 米（1.25 倍，超出）。
用这种对齐重建形状，只会把轨迹拉歪（实测残差 28266 -> 33913 米）。

本模块的做法
------------
1. `roadchain.select_chain` 先选出**空间连贯**的实例（同名路全市多实例，
   「虹桥路」有 149 段，必须先切分再链式 DP 选一组）；
2. 改进版 DP 对齐：路名可匹配**连续区间**，代价 = 区间内距离中位数
   + 长度超限罚 + 位置先验；
3. 用对齐结果把每段重建到对应路名的真实几何上；
4. **安全阀**：重建后到各路名的残差若变差，直接放弃并返回原轨迹。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ShapeFitResult:
    """形状重建结果。"""

    ok: bool = False
    points: np.ndarray | None = None          # 重建后的轨迹（墨卡托米）
    segments: list[tuple[int, int, str]] = field(default_factory=list)
    before_m: float = 0.0                    # 重建前残差合计
    after_m: float = 0.0                     # 重建后残差合计
    covered_ratio: float = 0.0               # 被路名覆盖的轨迹点比例
    max_shift_m: float = 0.0                 # 重建带来的位移（90 分位，米）
    end_shift_m: float = 0.0                 # 起终点的位移（米）
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "before_m": round(self.before_m, 1),
            "after_m": round(self.after_m, 1),
            "covered_ratio": round(self.covered_ratio, 3),
            "max_shift_m": round(self.max_shift_m, 1),
            "end_shift_m": round(self.end_shift_m, 1),
            "n_segments": len(self.segments),
            "notes": self.notes,
        }


def _split_runs(points: np.ndarray, jump_m: float = 500.0) -> list[np.ndarray]:
    """把按顺序的点拆成连续子折线（跨 OSM segment 处有跳跃）。"""
    if len(points) < 2:
        return []
    runs: list[list[np.ndarray]] = [[points[0]]]
    for i in range(1, len(points)):
        if float(np.hypot(points[i][0] - points[i - 1][0],
                          points[i][1] - points[i - 1][1])) > jump_m:
            runs.append([points[i]])
        else:
            runs[-1].append(points[i])
    return [np.array(r) for r in runs if len(r) >= 2]


def main_axis(points: np.ndarray, jump_m: float = 500.0) -> np.ndarray:
    """取最长的连续子折线作为主轴，并**去掉相邻重复点**。

    ⚠️ **去重是必须的**：OSM 的相邻 way 共享端点，
    `split_instances` 又对每段独立采样，于是折线里塞满了重复点
    （实测「船厂路」22 点里只有 13 个唯一点）。
    重复点会让投影塌陷——DP 在同一位置反复打转，
    算出来的轨迹长度为 0（实测 10 段里有 5 段长度为 0）。
    """
    runs = _split_runs(points, jump_m)
    if not runs:
        return points
    best = max(runs, key=lambda r: float(np.linalg.norm(
        np.diff(r, axis=0), axis=1).sum()))
    keep = [best[0]]
    for i in range(1, len(best)):
        if math.hypot(float(best[i][0] - keep[-1][0]),
                      float(best[i][1] - keep[-1][1])) > 1.0:
            keep.append(best[i])
    return np.array(keep) if len(keep) >= 2 else best


def _arc(poly: np.ndarray) -> np.ndarray:
    """折线的累积弧长。"""
    if len(poly) < 2:
        return np.zeros(len(poly))
    return np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def _interp_along(poly: np.ndarray, arc: np.ndarray,
                  targets: np.ndarray) -> np.ndarray:
    """在折线上按弧长插值出坐标。"""
    out = np.empty((len(targets), 2))
    total = float(arc[-1])
    for k, t in enumerate(targets):
        t = float(np.clip(t, 0.0, total))
        j = int(np.searchsorted(arc, t, side="right") - 1)
        j = max(0, min(j, len(poly) - 2))
        span = arc[j + 1] - arc[j]
        f = 0.0 if span < 1e-9 else (t - arc[j]) / span
        out[k] = poly[j] + (poly[j + 1] - poly[j]) * f
    return out


def align_by_length(
    traj: np.ndarray,
    arc: np.ndarray,          # ⚠️ 必须是**米制**弧长（墨卡托米）
    road_polys: list[np.ndarray],
    road_lengths: list[float],
    names: list[str],
    n_seg: int = 120,
    pos_prior_m: float = 15.0,
    len_penalty: float = 1.5,
    max_span_ratio: float = 1.0,
) -> list[tuple[int, int]]:
    """把路名对齐到轨迹的连续区间（带长度一致性约束）。

    与旧版 `_align_segments` 的四点不同：
      1. 路名可匹配**连续区间**（旧版只匹配恰好 1 段）；
      2. 区间代价用**中位距离**（旧版是单段中位，区间一长就失真）；
      3. 区间弧长与**目标覆盖长度**的偏差计入代价——
         旧版只罚「超长」不罚「太短」，DP 于是给每条路名挑最短的、
         恰好贴合的一小段，重建等于没做（实测残差 28266 -> 28103）；
      4. 保留**跳过机制**（用户可能没提某些路段）。

    目标覆盖长度怎么定：按各路名实例长**按比例**分配总轨迹长。
    物理依据——用户沿这些路骑行，**路越长走得越长**。
    实测本例：中山南二路（实例 4963 米）应占约 3757 米，
    船厂路（350 米）约 265 米。

    返回 [(seg_start, seg_end), ...]，与 names 一一对应（段索引）。
    """
    n_names = len(names)
    n_seg = max(n_names, min(n_seg, len(traj) - 1))
    total = float(arc[-1]) or 1.0
    bounds = np.linspace(0.0, total, n_seg + 1)

    # --- 目标覆盖长度：按实例长按比例分配总轨迹长 ---
    lens = np.array([max(1.0, float(x)) for x in road_lengths])
    scale = total / float(lens.sum()) if lens.sum() > 0 else 1.0
    targets = lens * scale

    # --- 段采样点 ---
    seg_pts: list[np.ndarray] = []
    for i in range(n_seg):
        m = (arc >= bounds[i]) & (arc <= bounds[i + 1])
        pts = traj[m]
        if len(pts) < 2:
            j = int(np.argmin(np.abs(arc - bounds[i])))
            pts = traj[max(0, j - 1):j + 2]
        seg_pts.append(pts)

    # --- 代价矩阵：路名 i 到段 j 的中位距离 ---
    cost = np.zeros((n_names, n_seg), dtype=float)
    for i, poly in enumerate(road_polys):
        if len(poly) < 2:
            cost[i, :] = 1e5
            continue
        tree = cKDTree(poly)
        for j, pts in enumerate(seg_pts):
            if len(pts) == 0:
                cost[i, j] = 1e5
                continue
            d, _ = tree.query(pts)
            cost[i, j] = float(np.median(d))

    # 位置先验：第 i 个路名大致落在第 i/(n-1) 处（很弱，只用于打破平局）
    for i in range(n_names):
        j_exp = (i * (n_seg - 1) / max(1, n_names - 1)) if n_names > 1 else 0.0
        cost[i] += pos_prior_m * np.abs(np.arange(n_seg) - j_exp)

    # --- DP ---
    #
    # dp[i][j] = 前 i+1 个路名都处理完，第 i 个路名负责的区间**结束于**段 j。
    #
    # ⚠️ **不允许「跳过」**：上一版给跳过设了 25 米的代价，
    # 而几何残差是几百米，DP 于是全选跳过——所有路名挤到同一段
    # （实测 10 条路名全落在段 12，覆盖率只有 10%）。
    #
    # 改成：区间**连续铺满**轨迹（路名 i 的区间 = [k+1, j]，
    # 前一个结束于 k）。某条路名若在轨迹上找不到对应段，
    # 就让它占一个**空区间**，代价 EMPTY_COST——
    # 这本身是个真实的错误信号（说明该路名离轨迹很远）。
    #
    # 好处：所有轨迹段都归某个路名管，重建时不会有「没人负责」的空洞；
    # 而且区间长度之和恒等于轨迹总长，target 分配自动自洽。
    EMPTY_COST = 3000.0
    INF = float("inf")
    dp = np.full((n_names, n_seg), INF)
    back = np.full((n_names, n_seg, 2), -1, dtype=int)
    for j in range(n_seg):
        dp[0, j] = _span_cost(cost[0], 0, j, targets[0],
                               bounds, len_penalty, max_span_ratio)
        back[0, j] = (0, j)

    for i in range(1, n_names):
        for j in range(i, n_seg):
            best = INF
            bk = (-1, -1)
            # (a) 路名 i 负责区间 [k+1, j]
            for k in range(i - 1, j):
                if not np.isfinite(dp[i - 1, k]):
                    continue
                c = _span_cost(cost[i], k + 1, j, targets[i],
                               bounds, len_penalty, max_span_ratio)
                v = dp[i - 1, k] + c
                if v < best:
                    best = v
                    bk = (k, j)
            # (b) 路名 i 占空区间（找不到对应段）
            if np.isfinite(dp[i - 1, j]):
                v = dp[i - 1, j] + EMPTY_COST
                if v < best:
                    best = v
                    bk = (j, -1)
            dp[i, j] = best
            back[i, j] = bk

    last = int(np.argmin(dp[n_names - 1]))
    if not np.isfinite(dp[n_names - 1, last]):
        return []
    # 回溯
    out: list[tuple[int, int]] = [(0, 0)] * n_names
    j = last
    for i in range(n_names - 1, -1, -1):
        a, b = back[i, j]
        if b < 0:
            out[i] = (j + 1, j)          # 空区间
            continue
        if a < 0:
            return []
        out[i] = (int(a), int(b))
        j = a
    return out


def _span_cost(
    cost_row: np.ndarray,
    a: int,
    b: int,
    target_len: float,
    bounds: np.ndarray,
    len_penalty: float,
    max_span_ratio: float,
) -> float:
    """区间 [a, b] 的代价：距离项 + 覆盖长度偏差。

    ⚠️ **必须同时罚「太长」和「太短」**：
    只罚太短的话，DP 会给每条路名挑最短的、恰好贴合的一小段，
    重建后轨迹几乎没变（实测残差 28266 -> 28103，等于没做）。
    """
    if b < a:
        return 1e9
    seg = cost_row[a:b + 1]
    if len(seg) == 0:
        return 1e9
    # 稳健距离项：取最小 60% 的均值（偏向真正贴合的部分）
    srt = np.sort(seg)
    k = max(1, int(len(srt) * 0.6))
    dist = float(srt[:k].mean())
    # 覆盖长度项：区间弧长 vs 目标长度（双向罚）
    span_len = float(bounds[b + 1] - bounds[a])
    cap = max(target_len, 1.0) * max_span_ratio
    over = max(0.0, span_len - cap)
    under = max(0.0, target_len - span_len)
    return dist + len_penalty * (over + under)


def _map_by_ratio(seg: np.ndarray, poly: np.ndarray,
                  arc: np.ndarray) -> np.ndarray:
    """把段内点按弧长比例映射到几何的对应片段上。

    锚点用**段首末点各自在几何上的最近点**（不是段中点居中——
    那会让起终点跟着被挪动，实测偏 375 米）。

    **正反双向取优**：OSM 的 way 有方向，用户的骑行方向可能与之相反
    （实测「船厂路」的最近点索引是递减的 6,6,5,5,3,2,2,2）。
    两个方向都试，取投影总距离小的那个。

    为什么不用 DP 保序投影：方向相反时单调约束会让 DP 走不通，
    只能走「跳跃」转移，路径塌成一点（实测 10 段里 5 段长度为 0）。
    """
    best: np.ndarray | None = None
    best_score = float("inf")
    best_len = 0.0
    seg_len = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    for cand in (poly, poly[::-1]):
        if len(cand) < 2:
            continue
        ca = _arc(cand)
        tree = cKDTree(cand)
        _, j0 = tree.query(seg[0])
        _, j1 = tree.query(seg[-1])
        j0, j1 = int(j0), int(j1)
        if j1 == j0:
            continue
        # ⚠️ **几何片段的长度以「原段长」为准**，不是首末最近点之间的距离。
        #
        # 物理依据：真实路径长度 >= 直线距离。像素轨迹在城市级截图里
        # 是一条直线，所以它的弧长是**直线距离**；而真实骑行沿路走，
        # 路径**更长**（绕行）。若按「首末最近点之间的距离」取片段，
        # 映射后必然把轨迹**压短**——实测 10 段里 6 段被压短 25%+，
        # 物理上不可能，只能跳过。
        if j1 > j0:
            lo_t = float(ca[j0])
            hi_t = min(float(ca[-1]), lo_t + seg_len)
        else:
            hi_t = float(ca[j0])
            lo_t = max(float(ca[0]), hi_t - seg_len)
        if hi_t - lo_t < 1.0:
            continue
        cum_in = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(seg, axis=0), axis=1))])
        frac = cum_in / max(cum_in[-1], 1e-6)
        mapped = _interp_along(cand, ca, lo_t + frac * (hi_t - lo_t))
        score = float(np.sum(np.linalg.norm(mapped - seg, axis=1)))
        if score < best_score:
            best_score = score
            best = mapped
            best_len = float(hi_t - lo_t)
    if best is None:
        return seg.copy()
    return best


def _project_ordered(seg: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """把段内点投影到几何折线上，并保证**顺序单调**（map matching）。

    **双向尝试**：OSM 的 way 有方向，用户的骑行方向可能与之相反
    （实测「船厂路」的最近点索引是递减的 10,10,8,8,5,3,3,3）。
    单向 DP 在这种情况下走不通，只能走「跳跃」转移，
    于是路径塌陷。所以正反两个方向都试，取投影总距离小的那个。
    """
    fwd = _project_ordered_core(seg, poly)
    rev = _project_ordered_core(seg, poly[::-1])
    if len(fwd) != len(rev):
        return fwd
    df = float(np.sum(np.linalg.norm(fwd - seg, axis=1)))
    dr = float(np.sum(np.linalg.norm(rev - seg, axis=1)))
    return fwd if df <= dr else rev


def _project_ordered_core(seg: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """把段内点投影到几何折线上，并保证**顺序单调**（map matching）。

    为什么不用「按弧长比例映射」：
    那种做法要求「轨迹段长 = 几何片段长」严格成立，一旦几何走向与轨迹
    相反（道路单向、跨路口、OSM 分段顺序不同），映射就会错位——
    实测 10 段里有 4 段「变差」（残差没降），还有 1 段被压短一半。

    投影 + DP 保序是交通领域 map matching 的标准做法：
      1. 每个点找几何上的最近点；
      2. DP 找一条**单调**的对应序列（不允许轨迹前进时几何后退）；
      3. 限制单步跳跃幅度，避免「从路的一端跳到另一端」。

    返回与 seg 等长的坐标数组。
    """
    n, m = len(seg), len(poly)
    if m < 2 or n < 2:
        return seg.copy()
    tree = cKDTree(poly)
    dist, j0 = tree.query(seg)
    j0 = j0.astype(int)

    # 限制单步跳跃：轨迹前进 n 步，索引最多前进 ceil(m/n * 4) + 8
    max_jump = max(4, int(m / n * 4) + 8)

    INF = float("inf")
    dp = np.full((n, m), INF)
    back = np.full((n, m), -1, dtype=int)
    dp[0] = dist[0]
    # 「跳跃」转移的惩罚（米）：允许索引跨过一段，但要付出代价。
    # 没有它，窗口太窄时会死锁——实测「中山南二路」整段投影失败。
    JUMP_PENALTY = 50.0
    for t in range(1, n):
        lo = max(0, j0[t] - max_jump)
        hi = min(m, j0[t] + max_jump + 1)
        # 前缀最小：允许跳到更早的索引（带惩罚）
        #
        # ⚠️ **必须同时记录最小值所在的索引**（prefix_idx）。
        # 只记值不记位置的话，回溯时 `back[t,i]` 指向的索引
        # 其 dp 值并不是那个最小值——链条断裂，DP 会沿一条
        # 「虚假最优」路径收敛到同一个索引，投影结果塌成一点
        # （实测 10 段里有 5 段长度变成 0）。
        prev = dp[t - 1]
        run = INF
        run_idx = -1
        prefix = np.full(m, INF)
        prefix_idx = np.full(m, -1)
        for i in range(m):
            if prev[i] < run:
                run = prev[i]
                run_idx = i
            prefix[i] = run
            prefix_idx[i] = run_idx
        for i in range(lo, hi):
            pl = max(0, i - max_jump)
            seg_dp = prev[pl:i + 1]
            if not np.isfinite(seg_dp).any():
                continue
            k = int(np.argmin(seg_dp)) + pl
            v = prev[k] + dist[t]
            # 跳跃方案：从前缀最小处跳过来（i>=1 才有前缀可跳）
            if i >= 1 and prefix_idx[i - 1] >= 0:
                j_best = int(prefix_idx[i - 1])
                v2 = prefix[j_best] + dist[t] + JUMP_PENALTY
                if v2 < v:
                    v = v2
                    k = j_best
            dp[t, i] = v
            back[t, i] = k
    last = int(np.argmin(dp[n - 1]))
    if not np.isfinite(dp[n - 1, last]):
        return seg.copy()
    path = np.zeros(n, dtype=int)
    i = last
    for t in range(n - 1, 0, -1):
        path[t] = i
        i = back[t, i]
        if i < 0:
            return seg.copy()
    path[0] = i          # ⚠️ 第 0 行没有前驱，不能再往前回溯
    return poly[path]


def rebuild(
    traj: np.ndarray,
    arc: np.ndarray,          # ⚠️ 必须是**米制**弧长（墨卡托米）
    road_polys: list[np.ndarray],
    road_arcs: list[np.ndarray],
    spans: list[tuple[int, int]],
    n_seg: int,
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """按对齐结果把轨迹重建到路名几何上。"""
    total = float(arc[-1]) or 1.0
    bounds = np.linspace(0.0, total, n_seg + 1)
    out = traj.copy()
    segs: list[tuple[int, int, str]] = []
    skipped: list[tuple[int, str]] = []
    for i, (a, b) in enumerate(spans):
        poly = road_polys[i]
        if len(poly) < 2 or b <= a:
            continue
        lo_len = float(bounds[a])
        hi_len = float(bounds[b + 1])
        m = (arc >= lo_len) & (arc <= hi_len)
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        seg = traj[idx]
        # 起终点是**用户给的锚**（起终点锚定已钉准，误差 300-900 米），
        # 重建不该动它们——动了就说明整条轨迹被挪了，而不是「掰弯」。
        # 实测不锁定时起终点偏了 744 米。
        keep_first = (i == 0)
        keep_last = (i == len(spans) - 1)
        work = seg[1:] if keep_first else seg
        if keep_last and len(work) > 1:
            work = work[:-1]
        if len(work) < 2:
            continue
        mapped = _map_by_ratio(work, poly, road_arcs[i])
        write_idx = idx[1:] if keep_first else idx
        if keep_last:
            write_idx = write_idx[:-1]
        # 逐段守门：真实路径长度 >= 直线距离。
        # 若这一段重建后**变短**，说明轨迹是「斜穿」这条路而不是
        # 「沿着」走——贴上去反而会压缩轨迹。这时**跳过该段**
        # （保持原样），而不是整体放弃：其余段已经重建好了
        # （实测 9/10 段残差降到 0）。
        old_len = float(np.linalg.norm(
            np.diff(seg, axis=0), axis=1).sum())
        new_len = float(np.linalg.norm(
            np.diff(mapped, axis=0), axis=1).sum()) if len(mapped) > 1 else 0.0
        if new_len < old_len * 0.92:
            skipped.append((i, f"重建后变短 {old_len:.0f}->{new_len:.0f} 米"))
            continue
        out[write_idx] = mapped
        segs.append((int(idx[0]), int(idx[-1]), f"seg{i}"))
    return out, segs, skipped


def fit_shape(
    traj: np.ndarray,
    arc: np.ndarray,          # ⚠️ 必须是**米制**弧长（墨卡托米）
    road_polys: list[np.ndarray],
    road_arcs: list[np.ndarray],
    road_lengths: list[float],
    names: list[str],
    residual_fn,
    n_seg: int = 120,
    max_shift_m: float = 800.0,
    max_end_shift_m: float = 300.0,
    n_iter: int = 3,
) -> ShapeFitResult:
    """完整的形状重建流程（含安全阀）。

    ⚠️ **残差必须只在「被覆盖的段」上评估**。
    对整条轨迹求中位距离是错的：中段那几公里直线本来就不贴任何路名，
    中位数被未覆盖段主导，重建效果完全看不出来
    （实测整条残差 28266 -> 28146，看不出差别）。

    `residual_fn(points, idx, spans)` 需计算「**逐段**残差之和」——
    每一段只对它自己对应的路名算距离。
    ⚠️ **不能用「到所有路名的残差合计」**：
    重建后该段贴着自己的路名（残差 0），但离其他路名可能更远，
    合计值反而变差（实测 28201 -> 31788），会误杀正确的重建。
    逐段残差才是直接指标（实测 9334 -> 7022，改善 25%）。
    """
    res = ShapeFitResult()
    # ---- 不动点迭代 ----
    #
    # 为什么要迭代：像素轨迹在城市级截图里是**直线**，所以它的弧长是
    # 「直线距离」；而真实骑行沿路走，**路径更长**（绕行）。
    # 于是「按直线弧长分配给各路名」天然偏低——某条路名分到的弧长
    # 超过了它的实例长度能覆盖的范围，重建时就被压短（实测 10 段里 6 段）。
    #
    # 迭代可以自洽：重建一次后，新轨迹的弧长是**路径长度**，
    # 用它重新分配，各路名分到的长度就更贴近实际，���建质量随之提高。
    cur_traj = traj
    cur_arc = arc
    before = float("inf")
    after = float("inf")
    spans: list[tuple[int, int]] = []
    new_pts = traj
    segs: list[tuple[int, int, str]] = []
    skipped: list[tuple[int, str]] = []
    idx = np.array([], dtype=int)
    best_pts = None
    best_after = float("inf")
    best_spans: list[tuple[int, int]] = []
    best_segs: list[tuple[int, int, str]] = []

    for it in range(max(1, n_iter)):
        spans = align_by_length(cur_traj, cur_arc, road_polys, road_lengths,
                                names, n_seg=n_seg)
        if not spans:
            res.notes.append("改进版对齐失败")
            return res
        total = float(cur_arc[-1]) or 1.0
        bounds = np.linspace(0.0, total, n_seg + 1)
        covered: set[int] = set()
        for a, b in spans:
            lo, hi = bounds[a], bounds[b + 1]
            for k in np.where((cur_arc >= lo) & (cur_arc <= hi))[0]:
                covered.add(int(k))
        if len(covered) < 8:
            res.notes.append("被覆盖的轨迹点太少（%d），放弃" % len(covered))
            return res
        idx = np.array(sorted(covered))
        cand, segs, skipped = rebuild(
            cur_traj, cur_arc, road_polys, road_arcs, spans, n_seg)
        after = residual_fn(cand, idx, spans)
        if it == 0:
            before = residual_fn(cur_traj, idx, spans)
        if after < best_after:
            best_after = after
            best_pts = cand
            best_spans = spans
            best_segs = segs
        # 用新轨迹的弧长进入下一轮
        cur_traj = cand
        cur_arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(cand, axis=0), axis=1))])

    new_pts = best_pts if best_pts is not None else traj
    spans = best_spans
    segs = best_segs
    after = best_after
    for k, why in skipped:
        res.notes.append(f"第 {k} 段跳过重建：{why}")
    # 守门：**起终点必须不动**。
    #
    # 为什么不能用「整体位移」当守门条件：把直线掰成 Z 字形**必然**
    # 带来 1 公里级的位移（直线与 Z 字形本来就差这么多），
    # 实测 90 分位 2929 米——那是正确的重建结果，不是错误。
    #
    # 但「跑到别的路上」同样是 1 公里级位移，靠位移大小区分不了。
    # 能区分的是：**起终点是用户给的锚**（起点钉在小木桥路、终点钉在安龙路），
    # 重建不该动它们。动了就说明整条轨迹被平移了，不是「掰弯」。
    shift = np.linalg.norm(new_pts[idx] - cur_traj[idx], axis=1)
    res.max_shift_m = float(np.percentile(shift, 90))
    end_shift = float(max(
        np.hypot(*(new_pts[0] - traj[0])),
        np.hypot(*(new_pts[-1] - traj[-1]))))
    res.end_shift_m = end_shift
    if end_shift > max_end_shift_m:
        res.notes.append(
            f"重建后起终点偏移 {end_shift:.0f} 米"
            f"（> {max_end_shift_m:.0f} 米），已放弃")
        return res
    # 物理约束：真实路径长度 >= 直线距离。
    # 重建后若某段比原来**更短**，说明这一段被压缩了——映射错位。
    for k, (a, b) in enumerate(spans):
        if b <= a:
            continue
        lo, hi = bounds[a], bounds[b + 1]
        ii = np.where((cur_arc >= lo) & (cur_arc <= hi))[0]
        if len(ii) < 2:
            continue
        old_len = float(np.linalg.norm(
            np.diff(cur_traj[ii], axis=0), axis=1).sum())
        new_len = float(np.linalg.norm(
            np.diff(new_pts[ii], axis=0), axis=1).sum())
        if new_len < old_len * 0.92:
            res.notes.append(
                f"「{names[k]}」段重建后变短（{old_len:.0f} -> {new_len:.0f} 米），"
                f"已放弃")
            return res
    res.before_m = before
    res.after_m = after
    res.segments = segs
    res.covered_ratio = len(covered) / max(1, len(traj))
    # ⚠️ **必须拿「最好的一轮」比较**，不是最后一轮。
    # 不动点迭代里中间某轮可能更好（后面的轮次因为轨迹已变形反而变差），
    # 用最后一轮会误杀正确的重建（实测逐段残差 6663 -> 6663 被判失败）。
    if best_after > before * 0.95:
        res.notes.append(
            f"重建后残差未改善（覆盖段 {before:.0f} -> {best_after:.0f} 米），已放弃")
        return res
    res.ok = True
    res.points = new_pts
    res.notes.append(
        f"形状重建：覆盖 {res.covered_ratio:.0%} 的轨迹点，"
        f"覆盖段残差 {before:.0f} -> {after:.0f} 米")
    return res


# ----------------------------------------------------------------------
# 与 OSRM 融合：段内用路名几何，段间用 OSRM 的连接路径
# ----------------------------------------------------------------------

def _align_similar(
    src_a: np.ndarray, src_b: np.ndarray,
    dst_a: np.ndarray, dst_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """两点定**刚性**变换（平移 + 旋转，**不缩放**）。

    把 `src_a -> src_b` 的向量旋转到 `dst_a -> dst_b` 的方向。
    返回 (R, t)，用法：`out = src @ R.T + t`。

    ⚠️ **不能用带缩放的相似变换**。
    物理约束：连接两点的**路网路径长度 >= 直线距离**。
    若允许缩放（s = |dst|/|src| < 1），就会把连接段**压短**——
    实测融合后轨迹从 17553 米缩到 14998 米，比直线还短，
    物理上不可能。

    刚性变换下，连接段保持 OSRM 给的真实长度，
    末点与下一段首点可能差几米到几十米（可接受）。
    """
    ds = src_b - src_a
    dd = dst_b - dst_a
    if float(np.hypot(ds[0], ds[1])) < 1e-6:
        return np.eye(2), dst_a - src_a
    ang = math.atan2(dd[1], dd[0]) - math.atan2(ds[1], ds[0])
    c, si = math.cos(ang), math.sin(ang)
    R = np.array([[c, -si], [si, c]])
    t = dst_a - R @ src_a
    return R, t


def fuse_with_osrm(
    rebuilt: np.ndarray,
    seg_index: list[tuple[int, int]],
    osrm_legs: list[list[tuple[float, float]]],
    leg_wgs_to_merc,
) -> np.ndarray:
    """把 OSRM 的连接路径插进「段内重建」的轨迹里。

    参数
    ----
    rebuilt       段内重建后的轨迹（墨卡托米）
    seg_index     每段在 rebuilt 里的点下标区间 [(lo, hi), ...]
    osrm_legs     OSRM 各 leg 的几何（WGS-84 经纬度）
    leg_wgs_to_merc  WGS-84 -> 墨卡托米 的换算函数

    逻辑
    ----
    OSRM 的 leg[i] 是「第 i 个途经点 → 第 i+1 个途经点」的连接，
    而途经点就是各路名与轨迹的接触点——所以 leg[i] 恰好对应
    「路名 i 的段 → 路名 i+1 的段」之间的过渡。

    用两点相似变换把 leg[i] 对齐到「段 i 的末点 → 段 i+1 的首点」，
    再插进去。这样：
      * 段内仍然精确贴住用户说的那条路；
      * 段间变成真实的路口连接（而不是直线跳跃）。

    ⚠️ **必须从后往前替换**：前面插入的点会让后面的下标整体后移。
    """
    if not osrm_legs or len(seg_index) < 2:
        return rebuilt
    segs = [(int(a), int(b)) for a, b in seg_index]
    out = rebuilt.copy()
    for i in range(len(segs) - 2, -1, -1):
        if i >= len(osrm_legs):
            continue
        leg = osrm_legs[i]
        if len(leg) < 2:
            continue
        leg_m = np.array([leg_wgs_to_merc(lo, la) for lo, la in leg])
        hi_i = segs[i][1]
        lo_j = segs[i + 1][0]
        if hi_i >= len(rebuilt) or lo_j >= len(rebuilt) or lo_j <= hi_i:
            continue
        dst_a = rebuilt[hi_i]
        dst_b = rebuilt[lo_j]
        if float(np.hypot(dst_b[0] - dst_a[0], dst_b[1] - dst_a[1])) < 1.0:
            continue
        R, t = _align_similar(leg_m[0], leg_m[-1], dst_a, dst_b)
        mapped = leg_m @ R.T + t
        bridge = mapped[1:-1]
        if len(bridge) == 0:
            continue
        out = np.vstack([out[:hi_i + 1], bridge, out[lo_j:]])
        shift = len(bridge)
        for k in range(i + 1, len(segs)):
            segs[k] = (segs[k][0] + shift, segs[k][1] + shift)
    return out
