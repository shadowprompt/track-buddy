"""有序路名链：把用户的路名列表变成一条可求解的「地理骨架」。

这是本次修复的核心算法
======================

问题
----
低缩放截图（一张图覆盖 20 公里）下，锚点配准给出的**尺度不可信**：

  | 独立标尺                       | 尺度 (m/px) |
  |--------------------------------|-------------|
  | 上海西站 ↔ 上海站              | 13.42       |
  | 上海西站 ↔ 上海南站            | 12.58       |
  | 上海站   ↔ 上海南站            | 12.47       |
  | **锚点配准给出的 scale**        | **15.73**   |

配准偏大 **26%**，轨迹被拉长到 15739 米（真实约 12000 米），
东端因此被甩过黄浦江——实测轨迹起点落在**浦东侧**（59/417 个点在浦东），
路名归属于是报出「打浦路隧道 / 时光印记大道 / 世博大道 / 东岸绿道」。
用户反馈「我根本没去浦东」正是这个原因。

尺度为什么会被污染
------------------
锚点全是**远距离大范围地标**（上海西站、上海站、桃浦中央绿地、黄浦区、长宁区）。
地图引擎把地标的**文字标注**放在视野空白处，与真实点位差 100-2000 米，
且偏移**有方向性**。尺度 = 地理距离 / 像素距离，两端同向偏移会放大尺度。
实测逐对锚点隐含尺度从 12.57 到 19.86 m/px，极差 58%。

解法
----
用户提供的路名是**骑行顺序**，而且每条路在 OSM 里有**真实几何**。
把它们串起来就得到一条「地理骨架」，它同时约束：

  * **尺度**——骨架上相邻两条路的地理间距是客观事实；
  * **位置**——骨架的整体位置是客观事实；
  * **朝向**——骨架的走向是客观事实。

这与地标标注完全无关，因此不受标注偏移影响。

难点：同名路在全市有多个实例
---------------------------
「虹桥路」全市 149 段、「东安路」跨 lon 121.06~121.83、
「中山南二路」56 段。选错实例，整个骨架就废了。

解法：**链式 DP**——相邻路名的实例必须空间连贯（骑行不会瞬移 5 公里）。
emission 用「实例到当前轨迹的距离」，transition 用「相邻实例的间距」。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class RoadInstance:
    """一条路名在空间上的一个实例（同名路可能有多个）。"""

    name: str
    points: np.ndarray                 # (M,2) 墨卡托米
    tree: cKDTree = field(init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.tree = cKDTree(self.points)

    @property
    def n(self) -> int:
        return len(self.points)

    def min_dist_to(self, other: "RoadInstance") -> float:
        """到另一个实例的最近距离（米）。"""
        if self.n == 0 or other.n == 0:
            return float("inf")
        d, _ = other.tree.query(self.points)
        return float(d.min())

    def median_dist_to(self, q: np.ndarray) -> float:
        """一组点到本实例的中位距离（米）。"""
        if self.n == 0 or len(q) == 0:
            return float("inf")
        d, _ = self.tree.query(q)
        return float(np.median(d))


def split_instances(
    name: str,
    segments: list[list[tuple[float, float]]],
    gap_m: float = 800.0,
    sample_m: float = 25.0,
) -> list[RoadInstance]:
    """把同名路的所有段按空间聚成若干实例。

    gap_m: 相邻采样点超过这个距离就断开（不同实例）。
    sample_m: 沿线采样间隔（米）。采样后用 KD 树，查询更快也更准。
    """
    pts: list[tuple[float, float]] = []
    for seg in segments:
        for i in range(len(seg) - 1):
            (ax, ay), (bx, by) = seg[i], seg[i + 1]
            n = max(1, int(math.hypot(bx - ax, by - ay) / sample_m))
            for k in range(n + 1):
                t = k / n
                pts.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    if not pts:
        return []

    arr = np.array(pts)
    clusters: list[list[tuple[float, float]]] = []
    for p in arr:
        placed = False
        for c in clusters:
            # 与簇内抽样点比较（不是全部点，否则 O(n²) 太慢）
            step = max(1, len(c) // 40)
            d = min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in c[::step])
            if d < gap_m:
                c.append((float(p[0]), float(p[1])))
                placed = True
                break
        if not placed:
            clusters.append([(float(p[0]), float(p[1]))])

    out = [RoadInstance(name=name, points=np.array(c)) for c in clusters]
    # 大的实例排前面（更可能是用户骑的主干道）
    out.sort(key=lambda r: -r.n)
    return out


@dataclass
class ChainResult:
    """链式 DP 选实例的结果。"""

    ok: bool = False
    # 每条路名选中的实例
    chosen: list[RoadInstance] = field(default_factory=list)
    # 每条路名的全部候选
    candidates: list[list[RoadInstance]] = field(default_factory=list)
    # 每个实例在轨迹上的「接触像素弧长」
    contact_arc_px: list[float] = field(default_factory=list)
    # 每个实例与轨迹的接触点（墨卡托米）
    contact_geo: list[tuple[float, float]] = field(default_factory=list)
    total_cost: float = float("inf")
    notes: list[str] = field(default_factory=list)


def build_candidates(
    road_names: list[tuple[str, list[list[tuple[float, float]]]]],
    gap_m: float = 800.0,
) -> list[list[RoadInstance]]:
    """给每条路名切分出空间实例。"""
    out: list[list[RoadInstance]] = []
    for nm, segs in road_names:
        out.append(split_instances(nm, segs, gap_m=gap_m))
    return out


def select_chain(
    candidates: list[list[RoadInstance]],
    traj: np.ndarray,
    arc_px: np.ndarray,
    max_jump_m: float = 6000.0,
    jump_penalty: float = 0.35,
    max_candidates: int = 12,
) -> ChainResult:
    """链式 DP：为每条路名选一个空间连贯的实例。

    traj:    (N,2) 像素轨迹（未投影）
    arc_px:  (N,)  像素弧长
    """
    res = ChainResult()
    n = len(candidates)
    if n == 0:
        res.notes.append("没有路名")
        return res

    # emission：实例到当前轨迹的最近距离
    emission: list[np.ndarray | None] = []
    for cands in candidates:
        if not cands:
            emission.append(None)
            continue
        costs = []
        for inst in cands[:max_candidates]:
            d, _ = inst.tree.query(traj)
            costs.append(float(d.min()))
        emission.append(np.array(costs))

    usable = [i for i in range(n) if emission[i] is not None]
    if len(usable) < 2:
        res.notes.append("可用路名不足")
        return res

    # DP：状态 = (路名序号, 实例下标)
    # 只在「连续可用的路名」上做 DP（中间若有路名无几何，用跨段代价连接）
    INF = float("inf")
    dp = np.full((n, max_candidates), INF)
    back = np.full((n, max_candidates), -1, dtype=int)
    first = usable[0]
    dp[first, :len(emission[first])] = emission[first]

    for pos in range(1, len(usable)):
        i = usable[pos]
        prev = usable[pos - 1]
        if emission[i] is None or emission[prev] is None:
            continue
        for j in range(min(len(emission[i]), max_candidates)):
            best, bi = INF, -1
            for k in range(min(len(emission[prev]), max_candidates)):
                if not np.isfinite(dp[prev, k]):
                    continue
                dd = candidates[i][j].min_dist_to(candidates[prev][k])
                if dd > max_jump_m:
                    continue
                cost = dp[prev, k] + dd * jump_penalty
                if cost < best:
                    best, bi = cost, k
            if bi >= 0:
                dp[i, j] = best + emission[i][j]
                back[i, j] = bi

    last = usable[-1]
    v = dp[last, :min(len(emission[last]), max_candidates)]
    if not np.isfinite(v).any():
        res.notes.append("链式 DP 找不到空间连贯的实例组合")
        return res

    j = int(np.argmin(v))
    res.total_cost = float(v[j])
    chosen_idx: dict[int, int] = {}
    for pos in range(len(usable) - 1, -1, -1):
        i = usable[pos]
        chosen_idx[i] = j
        j = back[i, j]
        if j < 0:
            break

    res.candidates = candidates
    res.chosen = []
    for i in range(n):
        k = chosen_idx.get(i)
        if k is None or i not in usable:
            res.chosen.append(RoadInstance(name="", points=np.zeros((0, 2))))
        else:
            res.chosen.append(candidates[i][k])
    res.ok = True
    return res


def contact_points(
    chain: ChainResult,
    traj: np.ndarray,
    arc_px: np.ndarray,
) -> None:
    """求每个路名实例与轨迹的「接触点」及其像素弧长。

    接触点 = 实例上离轨迹最近的点；像素弧长 = 该点在轨迹上的位置。
    """
    if not chain.ok:
        return
    traj_tree = cKDTree(traj)
    chain.contact_arc_px = []
    chain.contact_geo = []
    for inst in chain.chosen:
        if inst.n == 0:
            chain.contact_arc_px.append(float("nan"))
            chain.contact_geo.append((float("nan"), float("nan")))
            continue
        d, i = inst.tree.query(traj)
        k = int(np.argmin(d))
        p = inst.points[i[k]]
        _, j = traj_tree.query(p[None])
        chain.contact_arc_px.append(float(arc_px[int(np.atleast_1d(j)[0])]))
        chain.contact_geo.append((float(p[0]), float(p[1])))


def scale_from_chain(
    chain: ChainResult,
    min_px: float = 30.0,
) -> tuple[float, list[float], list[int]]:
    """从链式骨架反推尺度。

    对每一对相邻路名：
        尺度ᵢ = (两实例接触点的地理距离) / (两接触点的像素弧长差)

    取中位数。像素弧长差太小的对不可靠（除以小数会放大噪声），剔除。

    返回 (尺度, 逐段尺度列表, 参与计算的对的下标)。
    """
    ratios: list[float] = []
    used: list[int] = []
    for i in range(len(chain.contact_arc_px) - 1):
        a_px = chain.contact_arc_px[i]
        b_px = chain.contact_arc_px[i + 1]
        if not (math.isfinite(a_px) and math.isfinite(b_px)):
            continue
        dpx = b_px - a_px
        if dpx < min_px:
            continue
        ga = chain.contact_geo[i]
        gb = chain.contact_geo[i + 1]
        if not (math.isfinite(ga[0]) and math.isfinite(gb[0])):
            continue
        dgeo = math.hypot(gb[0] - ga[0], gb[1] - ga[1])
        if dgeo <= 0:
            continue
        ratios.append(dgeo / dpx)
        used.append(i)
    if not ratios:
        return 0.0, [], []
    return float(np.median(ratios)), ratios, used


def scale_from_chain_span(
    chain: ChainResult,
    min_px: float = 100.0,
) -> tuple[float, float]:
    """从链式骨架的**整体跨度**反推尺度。

    为什么需要这个
    --------------
    逐段比值法（`scale_from_chain`）的问题是方差大：实测同一次求解里
    逐段比值从 10.2 到 26.0 波动，中位数被中间几段主导。

    根因是「接触点」本身不确定——路名实例上的接触点只是「离轨迹最近的
    那个点」，它随轨迹位置漂移；相邻两段的接触点漂移方向不同，
    差值就被放大了。

    整体跨度法只用**首尾两个**接触点：位移大、相对误差小。
    实测同一例中整体跨度给出 12.03，与独立标尺（12.47~13.42）吻合。

    返回 (尺度, 尺度来源说明)。数据不足时返回 (0, 原因)。
    """
    arcs = chain.contact_arc_px
    geos = chain.contact_geo
    live = [i for i in range(len(arcs))
            if math.isfinite(arcs[i]) and math.isfinite(geos[i][0])]
    if len(live) < 2:
        return 0.0, "有效接触点不足"
    i0, i1 = live[0], live[-1]
    dpx = arcs[i1] - arcs[i0]
    if dpx < min_px:
        return 0.0, f"首尾像素跨度太小（{dpx:.0f}px）"
    dgeo = math.hypot(geos[i1][0] - geos[i0][0], geos[i1][1] - geos[i0][1])
    if dgeo <= 0:
        return 0.0, "首尾地理距离为零"
    s = dgeo / dpx
    return s, f"骨架首尾跨度 {dgeo:.0f} 米 / {dpx:.0f} 像素"
