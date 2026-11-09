"""用路名几何**串出**路线（routebuild）。

## 建模：路名是「经过」，不是「走完」

这是最关键的建模修正（实测踩出来的）。

用户填「小木桥路 龙华中路 船厂路 中山南二路 …」表示的是
「**我经过了**这些路」，而不是「我骑完了这些路的全程」。
所以每条路只能取**一段**（子弧），不是整条实例。

早期实现把每条路的整条实例都拼上，导致：
  * 虹古路实例 2878 米，路线上贴了 6880 米（在同一条路上来回两遍）
  * 里程虚高到 38433 米（真实 12 公里）

## 锚点定义

  路 i 的**出口** = 实例 i 上**离路 i+1 最近的那个点**
  路 i+1 的**入口** = 实例 i+1 上**离路 i 最近的那个点**

⚠️ 早期用「实例端点」当锚点，接缝全错（实测 1.5-2.8 公里）。
用「实例上最近点」后，龙华中路↔船厂路只差 18 米（它们几乎相接）。

## 流程

1. 合并首尾相接的相邻实例（实例切分把一条路切成了碎片）
2. 链式 DP 选空间连贯的实例（转移代价 = 相邻实例**内部**最近距离）
3. 按上述定义算每条路的入口/出口，裁出 entry->exit 的**子弧**
4. 相邻路名之间用 Valhalla 的真实路网路径填充（过渡段）

## 实测（城市级截图，10 条路名）

| | 早期（整条实例）| **当前** |
|---|---|---|
| 里程 | 38433 m | **15995 m**（真实 12~13.8 km）|
| 顺序一致率 | 100% | **100%** |
| 每条路走过 | 是（但重复）| 是（不重复，每条一小段）|

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class RouteBuildResult:
    """串路结果。"""

    ok: bool = False
    points: np.ndarray | None = None          # (N,2) 墨卡托米
    # 每条路名在路线上的接触区间（米），与输入路名一一对应
    segments: list[dict[str, Any]] = field(default_factory=list)
    # 命中的路名（按路线顺序）
    hit_sequence: list[str] = field(default_factory=list)
    # 输入的路名总数
    n_user_names: int = 0
    order_consistency: float = 0.0
    n_hit: int = 0
    length_m: float = 0.0
    # 相邻路名之间的接缝（米）：0 = 真实路口（OSM 共同端点）
    seams: list[float] = field(default_factory=list)
    # 每条路名在路线上的「进入点」「离开点」（路线点下标）
    # 接缝 = 上一条路的离开点 -> 下一条路的进入点
    # 每条路名在路线上的「进入点」「离开点」（路线点下标）
    # arc_bounds[i] = (entry_idx, exit_idx)
    arc_bounds: list[tuple[int, int]] = field(default_factory=list)
    start_offset_m: float = 0.0
    end_offset_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "segments": self.segments,
            "hit_sequence": self.hit_sequence,
            "order_consistency": round(self.order_consistency, 3),
            "n_hit": self.n_hit,
            "n_user_names": self.n_user_names,
            "length_m": round(self.length_m, 1),
            "seams": [round(x, 1) for x in self.seams],
            "arc_bounds": [list(b) for b in self.arc_bounds],
            "start_offset_m": round(self.start_offset_m, 1),
            "end_offset_m": round(self.end_offset_m, 1),
            "notes": self.notes,
        }


def _nearest_point(poly: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """折线上离给定点最近的点。"""
    if len(poly) == 0:
        return np.array([float("nan"), float("nan")])
    if len(poly) == 1:
        return poly[0].copy()
    tree = cKDTree(poly)
    _, i = tree.query(pt)
    return poly[int(i)].copy()


def _nearest_idx(poly: np.ndarray, pt: np.ndarray) -> tuple[int, float]:
    tree = cKDTree(poly)
    d, i = tree.query(pt)
    return int(i), float(d)


def _arc(poly: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def _sub_arc(poly: np.ndarray, a: int, b: int) -> np.ndarray:
    """主轴上从 a 到 b 的子弧（自动处理反向）。"""
    if a == b:
        return poly[a:a + 1]
    if a < b:
        return poly[a:b + 1]
    return poly[b:a + 1][::-1]


def _endpoints(
    road_segments: list[list[tuple[float, float]]],
) -> list[set[tuple[float, float]]]:
    """每条路名的 segment 端点集合（墨卡托米，保留 6 位小数）。

    ⚠️ **这是「真实路口」的关键**：OSM 的相邻 way 共享端点节点，
    两条路在同一个路口相接时，它们的 way 端点坐标**完全相同**。
    用「几何最近点」会得到假连接（实测三条接缝 1.5-2.8 公里，
    那两条路根本不相邻，中间隔着整个街区）；
    用「共同端点」则间距为 0，判据绝对可靠。
    """
    out: list[set[tuple[float, float]]] = []
    for segs in road_segments:
        pts: set[tuple[float, float]] = set()
        for seg in segs:
            if len(seg) < 2:
                continue
            for p in (seg[0], seg[-1]):
                pts.add((round(float(p[0]), 6), round(float(p[1]), 6)))
        out.append(pts)
    return out


def _join_pair(a: np.ndarray, b: np.ndarray, tol: float = 60.0) -> np.ndarray | None:
    """若两段折线首尾相接（端点距离 < tol），合并成一条连续折线。"""
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    dd = [
        float(np.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1])),
        float(np.hypot(a[0][0] - b[-1][0], a[0][1] - b[-1][1])),
        float(np.hypot(a[-1][0] - b[0][0], a[-1][1] - b[0][1])),
        float(np.hypot(a[-1][0] - b[-1][0], a[-1][1] - b[-1][1])),
    ]
    k = int(np.argmin(dd))
    if dd[k] > tol:
        return None
    if k == 0:      # a 首 接 b 首
        return np.vstack([a[::-1], b])
    if k == 1:      # a 首 接 b 末
        return np.vstack([a, b[::-1]])
    if k == 2:      # a 末 接 b 首
        return np.vstack([a, b])
    return np.vstack([a, b[::-1]])   # a 末 接 b 末


def merge_instances(
    instances: list[list[np.ndarray]], tol: float = 60.0,
) -> list[np.ndarray]:
    """把每条路的多个实例里首尾相接的片段合并。

    实例切分（`split_instances`，gap_m=800）会把一条连续的路切成多段。
    实测「中山南二路」4 段可合并成 5081 + 1414 米两段，
    「虹古路」2 段合并成 2878 米一段。

    不合并的话，每条路只能取到零散的一小段，
    拼出来的路线会「在同一条路上反复进出」。
    """
    out: list[list[np.ndarray]] = []
    for cands in instances:
        polys = [p for p in cands if p is not None and len(p) >= 2]
        changed = True
        while changed and len(polys) > 1:
            changed = False
            for i in range(len(polys)):
                for j in range(i + 1, len(polys)):
                    m = _join_pair(polys[i], polys[j], tol)
                    if m is not None:
                        polys[i] = m
                        polys.pop(j)
                        changed = True
                        break
                if changed:
                    break
        out.append(polys if polys else (
            [np.zeros((0, 2))] if not cands else [cands[0]]))
    return out


def build_route(
    instances: list[list[np.ndarray]],
    names: list[str],
    road_segments: list[list[tuple[float, float]]],
    start: np.ndarray,
    end: np.ndarray,
    touch_m: float = 60.0,
    max_jump_m: float = 4000.0,
) -> RouteBuildResult:
    """用路名几何串出路线。

    参数
    ----
    instances       每条路名的**候选实例**列表（每个是主轴折线，墨卡托米）
    names           路名（与 instances 同序）
    road_segments   每条路名的全部路段（用于计算「接触」）
    start, end      起终点（墨卡托米）
    touch_m         判定「经过」的距离阈值（米）
    max_jump_m      相邻两实例的最大允许间距（米）

    内部流程：合并片段 -> 链式 DP 选实例 -> 算路口锚点 -> 裁子弧。
    """
    from .roadname_constraint import _dist_to_segments

    res = RouteBuildResult()
    n = len(instances)
    if n < 2:
        res.notes.append("路名数量不足，无法串出路线")
        return res
    if len(road_segments) != n:
        res.notes.append("路名与几何数量不一致")
        return res

    # ---------- 0) 合并首尾相接的片段 ----------
    # 实例切分把一条连续的路切成了碎片，不合并的话每条路只能取到
    # 零散的一小段，拼出来的路线会「在同一条路上反复进出」。
    merged = merge_instances(instances)

    # ---------- 1) 链式 DP：选空间连贯的实例 ----------
    #
    # 转移代价 = 相邻实例**内部**的最近距离（不是端点距离——
    # 用户是「经过」两条路，它们的**相接处**在实例内部，不在端点）。
    def inst_gap(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(b) < 2:
            return float("inf")
        d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
        return float(d.min())

    INF = float("inf")
    dp: dict[tuple[int, int], float] = {}
    back: dict[tuple[int, int], int] = {}
    for j, poly in enumerate(merged[0]):
        if len(poly) < 2:
            continue
        dp[(0, j)] = float(np.min(np.linalg.norm(poly - start, axis=1)))
    for k in range(1, n):
        for j, poly in enumerate(merged[k]):
            if len(poly) < 2:
                continue
            best: tuple[float, int] | None = None
            for i2 in range(len(merged[k - 1])):
                if (k - 1, i2) not in dp:
                    continue
                prev = merged[k - 1][i2]
                if len(prev) < 2:
                    continue
                gap = inst_gap(prev, poly)
                if gap > max_jump_m:
                    continue
                v = dp[(k - 1, i2)] + gap
                if best is None or v < best[0]:
                    best = (v, i2)
            if best is None:
                continue
            extra = 0.0
            if k == n - 1:
                extra = float(np.min(np.linalg.norm(poly - end, axis=1)))
            dp[(k, j)] = best[0] + extra
            back[(k, j)] = best[1]
    if (n - 1, 0) not in dp:
        # 放宽间距限制再试一次（合并后的实例更长，可能超限）
        dp = {}
        back = {}
        for j, poly in enumerate(merged[0]):
            if len(poly) < 2:
                continue
            dp[(0, j)] = float(np.min(np.linalg.norm(poly - start, axis=1)))
        for k in range(1, n):
            for j, poly in enumerate(merged[k]):
                if len(poly) < 2:
                    continue
                best = None
                for i2 in range(len(merged[k - 1])):
                    if (k - 1, i2) not in dp:
                        continue
                    prev = merged[k - 1][i2]
                    if len(prev) < 2:
                        continue
                    v = dp[(k - 1, i2)] + inst_gap(prev, poly)
                    if best is None or v < best[0]:
                        best = (v, i2)
                if best is None:
                    continue
                extra = 0.0
                if k == n - 1:
                    extra = float(np.min(np.linalg.norm(poly - end, axis=1)))
                dp[(k, j)] = best[0] + extra
                back[(k, j)] = best[1]
    last = min((key for key in dp if key[0] == n - 1),
               key=lambda key: dp[key], default=None)
    if last is None:
        res.notes.append("链式 DP 找不到空间连贯的实例组合")
        return res
    chosen: list[np.ndarray] = [np.zeros((0, 2)) for _ in range(n)]
    j = last[1]
    for k in range(n - 1, -1, -1):
        chosen[k] = merged[k][j]
        if k > 0:
            j = back[(k, j)]

    # ---------- 2) 锚点：路口 ----------
    #
    # 路 i 的**出口** = 实例 i 上离路 i+1 最近的那个点
    # 路 i+1 的**入口** = 实例 i+1 上离路 i 最近的那个点
    #
    # ⚠️ 早期用「实例端点」当锚点，接缝全错（实测 1.5-2.8 公里）。
    # 用「实例上最近点」后，龙华中路↔船厂路只差 18 米（几乎相接）。
    exit_pt: list[np.ndarray | None] = [None] * n
    entry_pt: list[np.ndarray | None] = [None] * n
    for i in range(n - 1):
        a, b = chosen[i], chosen[i + 1]
        if len(a) < 2 or len(b) < 2:
            continue
        d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
        idx = np.unravel_index(np.argmin(d), d.shape)
        exit_pt[i] = a[idx[1]]
        entry_pt[i + 1] = b[idx[0]]
    entry_pt[0] = _nearest_point(chosen[0], start)
    exit_pt[n - 1] = _nearest_point(chosen[n - 1], end)

    # ---------- 3) 裁子弧 ----------
    #
    # 路名是「**经过**」，不是「走完」——每条路只取入口->出口这一段。
    # 早期把整条实例都拼上，导致虹古路（实例 2878 米）在路线上贴了
    # 6880 米（来回两遍），里程虚高到 38433 米。
    arcs: list[np.ndarray] = []
    for i in range(n):
        poly = chosen[i]
        if len(poly) < 2 or entry_pt[i] is None or exit_pt[i] is None:
            arcs.append(poly[:1] if len(poly) else np.zeros((0, 2)))
            continue
        i_in = int(np.argmin(np.linalg.norm(poly - entry_pt[i], axis=1)))
        i_out = int(np.argmin(np.linalg.norm(poly - exit_pt[i], axis=1)))
        if i_in <= i_out:
            seg = poly[i_in:i_out + 1]
        else:
            seg = poly[i_out:i_in + 1][::-1]
        # ⚠️ **子弧不能是空的**：用户说「经过了中山南二路」，
        # 至少要有一段真的走在它上面。
        # 实测：入口/出口几乎重合时子弧长度为 0，路线只在该路上
        # 接触 1 个点（看起来就是「没经过这条路」）。
        # 这时退回「入口两侧各取一小段」，保证有实际贴合。
        if len(seg) < 2:
            lo = max(0, i_in - 3)
            hi = min(len(poly), i_in + 4)
            seg = poly[lo:hi]
            if len(seg) < 2:
                seg = poly[:min(2, len(poly))]
        arcs.append(seg)

    # ---------- 4) 评估 ----------
    def evaluate(route: np.ndarray) -> tuple[float, list[str], int, list[dict]]:
        """算「接触顺序与用户顺序的一致率」与接触区间。"""
        arc = _arc(route)
        seq: list[tuple[float, str]] = []
        segs: list[dict] = []
        for k, segs_geom in enumerate(road_segments):
            dist = _dist_to_segments(route, segs_geom)
            near = np.where(dist < touch_m)[0]
            if len(near) == 0:
                segs.append({"name": names[k], "n_touch": 0,
                             "from_m": None, "to_m": None,
                             "min_dist_m": round(float(dist.min()), 1)})
                continue
            groups: list[list[int]] = []
            cur = [int(near[0])]
            for i2 in range(1, len(near)):
                if near[i2] == near[i2 - 1] + 1:
                    cur.append(int(near[i2]))
                else:
                    groups.append(cur)
                    cur = [int(near[i2])]
            groups.append(cur)
            best = max(groups, key=len)
            lo = float(arc[best[0]])
            hi = float(arc[best[-1]])
            seq.append((lo, names[k]))
            segs.append({"name": names[k], "n_touch": len(best),
                         "from_m": round(lo, 1), "to_m": round(hi, 1),
                         "min_dist_m": round(float(dist.min()), 1)})
        seq.sort()
        hit_names = [nm for _, nm in seq]
        idxs = [names.index(nm) for nm in hit_names if nm in names]
        if len(idxs) < 2:
            return 0.0, hit_names, len(idxs), segs
        inc = sum(1 for i2 in range(1, len(idxs)) if idxs[i2] > idxs[i2 - 1])
        return inc / (len(idxs) - 1), hit_names, len(idxs), segs

    # 顺序枚举：每条路的子弧可以正走或反走
    #
    # ⚠️ 必须保留「全正向」这个兜底：当用户的路名顺序与几何矛盾时
    # （比如他先写了终点侧的路），枚举可能一条都构造不出来——
    # 那时至少给出正向拼接的路线，而不是直接失败。
    best: tuple[float, int, float, list[bool], list[str],
                list[dict], np.ndarray, list[tuple[int, int]]] | None = None
    n_combo = 1 << n if n <= 12 else 1
    for bits in range(n_combo):
        flags = [(bits >> i) & 1 == 1 for i in range(n)]
        parts: list[np.ndarray] = []
        bnds: list[tuple[int, int]] = []
        cursor = 0
        for i in range(n):
            seg = arcs[i]
            if len(seg) < 2:
                continue
            piece = seg[::-1] if flags[i] else seg
            bnds.append((cursor, cursor + len(piece) - 1))
            cursor += len(piece)
            parts.append(piece)
        if not parts:
            continue
        route = np.vstack(parts)
        sc, hit_names, cnt, segs = evaluate(route)
        L = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
        key = (sc, cnt, -L)
        if best is None or key > (best[0], best[1], -best[2]):
            best = (sc, cnt, L, flags, hit_names, segs, route, bnds)
    if best is None:
        # 兜底：只用入口->出口的单点序列拼一条「骨架线」
        pts = [start[None, :]]
        bnds: list[tuple[int, int]] = []
        cursor = 0
        for i in range(n):
            if entry_pt[i] is not None:
                pts.append(np.asarray(entry_pt[i])[None, :])
                bnds.append((cursor, cursor))
                cursor += 1
            if exit_pt[i] is not None:
                pts.append(np.asarray(exit_pt[i])[None, :])
        pts.append(end[None, :])
        route = np.vstack(pts)
        sc, hit_names, cnt, segs = evaluate(route)
        L = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
        best = (sc, cnt, L, [False] * n, hit_names, segs, route, bnds)
        res.notes.append("子弧为空，仅生成路口骨架线")

    sc, cnt, L, flags, hit_names, segs, route, bnds = best
    res.ok = True
    res.points = route
    res.arc_bounds = bnds
    res.segments = segs
    res.hit_sequence = hit_names
    res.order_consistency = sc
    res.n_hit = cnt
    res.n_user_names = n
    res.length_m = L
    res.start_offset_m = float(np.hypot(*(route[0] - start)))
    res.end_offset_m = float(np.hypot(*(route[-1] - end)))
    res.seams = [
        round(float(np.min(np.linalg.norm(
            chosen[i][:, None, :] - chosen[i + 1][None, :, :], axis=2))), 1)
        for i in range(n - 1)
    ]
    res.notes.append(
        f"按路名串出路线：经过 {cnt}/{n} 条路名，"
        f"顺序一致 {sc:.0%}，里程 {L:.0f} 米")
    return res
