"""把 Valhalla 的逐对连接路径插入串路结果的接缝处。

## 为什么需要

串路（`app/routebuild.py`）只解决了「每条路走哪一段」，
但相邻两条路之间是**直线跳跃**——真实骑行必然经过别的路。

实测（同一张城市级截图）：
  船厂路 → 中山南二路   直线 2053 米 | Valhalla 路径 2202 米
                              经由 零陵路、大木桥路、斜土路、合宝路
  中山南二路 → 东安路   直线 2550 米 | Valhalla 路径 2786 米
                              经由 合宝路、正阳路、医学院路、枫林路…
  虹桥路 → 水城路       直线 3403 米 | Valhalla 路径 3666 米

所以「接缝」本身就是一段真实路网路径——把它插进去，
整条路线才是一条**可骑行**的路线，而不是「十条直线的拼接」。

## 定位必须用下标，不能用「最近点」

⚠️ 踩过的坑：最初用「路线上离接缝首尾最近的点」来定位插入位置，
结果长度从 14905 米暴涨到 27688 米——因为两条路的**几何**很接近，
「最近点」匹配到了错误的段（比如把接缝插到了回头路上）。

正确做法：用 `routebuild` 记录的 `exit_indices`（每条路名在路线上的
离开点下标），**按下标精确插入**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SeamResult:
    """接缝插入结果。"""

    ok: bool = False
    points: np.ndarray | None = None
    n_inserted: int = 0
    n_total: int = 0
    length_before_m: float = 0.0
    length_after_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_inserted": self.n_inserted,
            "n_total": self.n_total,
            "length_before_m": round(self.length_before_m, 1),
            "length_after_m": round(self.length_after_m, 1),
            "notes": self.notes,
        }


def insert_seams(
    route: np.ndarray,
    seam_paths: list[list[np.ndarray] | None],
    entry_indices: list[int],
    exit_indices: list[int],
    max_growth: float = 1.5,
) -> SeamResult:
    """把接缝路径按**进入/离开点下标**插入路线。

    参数
    ----
    route         原路线（墨卡托米，(N,2)）
    seam_paths    每条接缝的路径（墨卡托米点序列）；None 表示无可用路径
    entry_indices 每条路名在 route 上的进入点下标
    exit_indices  每条路名在 route 上的离开点下标
    max_growth    插入后长度相对原长度的上限

    ⚠️ **接缝 i 的区间是 [exit_indices[i], entry_indices[i+1]]**——
    即「上一条路的最后一个点」到「下一条路的第一个点」。
    之前误用「exit[i] -> exit[i+1]」，那覆盖的是**下一条路的整段**，
    导致插入后长度暴涨到 49644 米。
    """
    res = SeamResult()
    if route is None or len(route) < 2:
        res.notes.append("路线太短")
        return res
    if not seam_paths or all(p is None for p in seam_paths):
        res.notes.append("没有接缝路径")
        return res
    if not exit_indices or not entry_indices:
        res.notes.append("缺少进入/离开点锚点")
        return res
    if len(entry_indices) != len(exit_indices):
        res.notes.append("进入点与离开点数量不一致")
        return res

    length_before = float(np.linalg.norm(
        np.diff(route, axis=0), axis=1).sum())
    res.length_before_m = length_before

    # 从后往前插入——前面的插入会让后面的下标整体后移
    # ⚠️ **必须用「原始下标」一次算好所有插入位置**，
    # 然后从后往前执行。
    #
    # 踩过的坑：边插边改锚点，导致后面用的 j_from/j_to 是「已插入后的
    # 下标」，而 out 的长度也在变——实测区间下标飙到 [271..272]，
    # 远超路线长度 129，插入后长度从 14905 暴涨到 56047 米。
    #
    # 正确做法：先把每个接缝在**原路线**上的区间算好，
    # 从后往前插入时，后面的插入不影响前面的下标。
    mids: list[np.ndarray | None] = []
    for i in range(len(seam_paths)):
        path = seam_paths[i]
        if path is None or len(path) < 2 or i + 1 >= len(entry_indices):
            mids.append(None)
            continue
        # 语义：mid 是「插到离开点与下一条路进入点之间」的点。
        #   路径有 >=3 点 -> 取中间的 path[1:-1]（首尾由路线本身提供）
        #   路径有 2 点   -> 取 1 个点 path[1:2]（否则无法体现绕行）
        mid = path[1:-1] if len(path) >= 3 else path[1:2]
        if len(mid) == 0:
            mids.append(None)
            continue
        mids.append(mid)

    out = route.copy()
    inserted = 0
    n_route = len(route)
    for i in range(len(mids) - 1, -1, -1):
        mid = mids[i]
        if mid is None:
            continue
        # 用**原始路线**的下标（从后往前插，前面不受影响）
        j_from = max(0, min(exit_indices[i], n_route - 1))
        j_to = max(0, min(entry_indices[i + 1], n_route - 1))
        if j_to <= j_from:
            continue
        out = np.vstack([out[:j_from + 1], mid, out[j_to:]])
        inserted += 1

    length_after = float(np.linalg.norm(
        np.diff(out, axis=0), axis=1).sum())
    res.n_inserted = inserted
    res.n_total = len(seam_paths)
    res.length_after_m = length_after
    if length_after > length_before * max_growth:
        res.notes.append(
            f"插入后长度暴涨（{length_before:.0f} -> {length_after:.0f} 米，"
            f"超过 {max_growth} 倍上限），已放弃")
        return res
    res.ok = True
    res.points = out
    res.notes.append(
        f"插入 {inserted}/{len(seam_paths)} 段真实路网连接，"
        f"里程 {length_before:.0f} -> {length_after:.0f} 米")
    return res
