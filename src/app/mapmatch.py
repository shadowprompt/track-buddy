"""地图匹配（HMM / Viterbi）：把像素轨迹吸附成「沿路折线」。

与 ``roadfit.refine_with_roads`` 的区别
--------------------------------------
精修是**单一刚体平移**：整条轨迹一起挪，只能修正整体位置偏差，
对局部漂移、弯曲路段、链式拼接的累积误差无能为力。
地图匹配把每个轨迹点当作一次观测，在道路采样点里找一条
「既贴近观测、又符合相邻点间行进距离」的最优状态序列，
输出的是沿路折线。

⚠️ 这是用**路网先验补形状**，天然有风险：起终点连线（美团概览页）
会被强行贴成某条看起来合理的路。因此本模块默认关闭，且带三道护栏：

1. 低信息量直线输入直接拒绝（调用方传 ``low_shape_info``）；
2. 平均吸附距离超过阈值 → 拒绝（路网不覆盖 / 轨迹本不该贴路）；
3. 匹配后长度或方向剧变 → 拒绝（防止折叠或跳到平行道路）。

输出是**备选显示层**：主 GeoJSON、置信度、里程一律不改。

数据源沿用 ``roadfit.RoadNetwork``（OSM 几何，仅用于几何参考，
不作为底图瓦片），坐标系统一在 GCJ-02 墨卡托米上计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geo.coords import mercator_to_lonlat


@dataclass
class MapMatch:
    """地图匹配结果（可选展示层）。"""

    ok: bool
    reason: str = ""
    coordinates_gcj02: list[list[float]] = field(default_factory=list)
    coordinates_px: list[list[float]] = field(default_factory=list)
    roads: list[str] = field(default_factory=list)
    mean_snap_m: float = 0.0
    median_snap_m: float = 0.0
    n_points: int = 0
    length_before_m: float = 0.0
    length_after_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "crs": "GCJ-02",
            "coordinates": self.coordinates_gcj02,
            "coordinates_px": self.coordinates_px,
            "roads": self.roads,
            "mean_snap_m": self.mean_snap_m,
            "median_snap_m": self.median_snap_m,
            "n_points": self.n_points,
            "length_before_m": self.length_before_m,
            "length_after_m": self.length_after_m,
        }


def _length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def hmm_match(
    reg: Any,
    polyline_px: Any,
    net: Any,
    max_points: int = 300,
    k_candidates: int = 4,
    sigma_m: float = 25.0,
    beta_m: float = 30.0,
    same_road_bonus: float = 0.35,
    max_snap_m: float = 60.0,
    low_shape_info: bool = False,
) -> MapMatch:
    """隐马尔可夫地图匹配，返回吸附后的备选折线。

    ``net`` 需提供 ``samples`` / ``owner`` / ``tree``（见 RoadNetwork）。
    """
    if low_shape_info:
        return MapMatch(ok=False, reason="起终点直线输入不做地图匹配（会把直线强行贴成一条路）")
    pts = np.asarray(polyline_px, dtype=float)
    if len(pts) < 3:
        return MapMatch(ok=False, reason="轨迹点不足")
    tree = getattr(net, "tree", None)
    samples = getattr(net, "samples", None)
    owner = getattr(net, "owner", None)
    if tree is None or not samples or owner is None:
        return MapMatch(ok=False, reason="无可用路网几何")
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]

    traj = np.asarray(
        [reg.pixel_to_mercator(float(px), float(py)) for px, py in pts],
        dtype=float,
    )
    k = int(min(k_candidates, len(samples)))
    dists, idxs = tree.query(traj, k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)
    sample_arr = np.asarray(samples, dtype=float)
    states = sample_arr[idxs]                     # (n, k, 2)
    owners = [[owner[int(i)] for i in row] for row in idxs]
    emit = -0.5 * np.square(dists / max(sigma_m, 1e-6))

    n = len(traj)
    dp = emit[0].copy()
    parent = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        step = float(np.linalg.norm(traj[t] - traj[t - 1]))
        nxt = np.full(k, -np.inf, dtype=float)
        par = np.zeros(k, dtype=int)
        for j in range(k):
            move = np.linalg.norm(states[t, j] - states[t - 1], axis=1)
            trans = -np.abs(move - step) / max(beta_m, 1e-6)
            trans += np.where(
                np.asarray(owners[t - 1]) == owners[t][j],
                same_road_bonus, 0.0,
            )
            score = dp + trans + emit[t, j]
            i = int(np.argmax(score))
            nxt[j] = score[i]
            par[j] = i
        dp, parent[t] = nxt, par

    path = [int(np.argmax(dp))]
    for t in range(n - 1, 0, -1):
        path.append(int(parent[t, path[-1]]))
    path.reverse()
    path_arr = np.asarray(path, dtype=int)
    chosen = states[np.arange(n), path_arr]
    snap = dists[np.arange(n), path_arr]

    mean_snap = float(snap.mean())
    if mean_snap > max_snap_m:
        return MapMatch(
            ok=False,
            reason=f"平均吸附距离 {mean_snap:.0f}m 超过 {max_snap_m:.0f}m，"
                   "路网可能不覆盖此轨迹",
            mean_snap_m=round(mean_snap, 1),
            n_points=n,
        )

    len_before, len_after = _length(traj), _length(chosen)
    if len_before > 0 and not (0.5 <= len_after / len_before <= 2.0):
        return MapMatch(
            ok=False, reason="匹配后轨迹长度剧变，疑似折叠或跳到平行道路",
            mean_snap_m=round(mean_snap, 1), n_points=n,
            length_before_m=round(len_before, 1),
            length_after_m=round(len_after, 1),
        )
    if len(traj) > 2:
        d0 = traj[-1] - traj[0]
        d1 = chosen[-1] - chosen[0]
        if float(np.dot(d0, d1)) <= 0:
            return MapMatch(
                ok=False, reason="匹配后首尾方向反转，拒绝该解",
                mean_snap_m=round(mean_snap, 1), n_points=n,
            )

    coords = [mercator_to_lonlat(float(x), float(y)) for x, y in chosen]
    px_out = [reg.mercator_to_pixel(float(x), float(y)) for x, y in chosen]
    road_seq = [owners[t][path_arr[t]] for t in range(n)]
    roads = [r for i, r in enumerate(road_seq) if i == 0 or r != road_seq[i - 1]]
    return MapMatch(
        ok=True,
        coordinates_gcj02=[[float(a), float(b)] for a, b in coords],
        coordinates_px=[[float(a), float(b)] for a, b in px_out],
        roads=roads,
        mean_snap_m=round(mean_snap, 1),
        median_snap_m=round(float(np.median(snap)), 1),
        n_points=n,
        length_before_m=round(len_before, 1),
        length_after_m=round(len_after, 1),
    )
