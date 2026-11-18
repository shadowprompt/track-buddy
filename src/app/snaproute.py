"""路网温和吸附：把像素描迹的点局部投影到道路上，修「画出来不在马路上」。

为什么不是全局地图匹配（HMM + 路网最短路）
------------------------------------------
实测（宁波闭环样本，绝对位置偏差 ~300-500m 结构性）：HMM 选状态 +
Dijkstra 连接的全局方案会把轨迹重塑进错误的平行路族，长度虚增
29-42%、百余处跨街跳段——**在锚点基准统一之前，全局重匹配是在跟
系统性偏差搏斗，注定失败**。

温和吸附的哲学：轨迹本身已经大致贴路（实测描迹到 OSM 路网中位
52m），不需要路网先验重塑形状；只需要把「离路 ≤ 容差」的点投影到
路上、离路太远的点**保持原样**（诚实：路网吸附修不了绝对位置偏差，
强行吸附只会伪造）。

三步
----
1. 逐点找最近路网采样点；距离 ≤ ``tol_m`` 才吸附（投影到 6m 采样格，
   误差 ≤3m）；
2. **窗口多数表决**：吸附点的道路身份若与邻域多数不一致且吸附距离
   不小，则回退原始点——压制平行路族处的道路抖动；
3. **跳变回退**：吸附后相邻点间距 > ``max_jump_m`` 的，把后者回退为
   原始坐标——十字路口/双平行路处的吸附翻转不会产生跨街直线。

坐标系统一 GCJ-02 墨卡托米（``reg.pixel_to_mercator`` 与 OSM→GCJ 同
基准，前端腾讯/高德底图同为 GCJ-02，显示一致）。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def gentle_snap(
    reg: Any,
    polyline_px: Any,
    net: Any,
    *,
    tol_m: float = 60.0,
    window: int = 2,
    max_jump_m: float = 100.0,
    max_points: int = 1500,
) -> dict[str, Any]:
    """把像素轨迹温和吸附到路网，返回可 JSON 化的结果字典。

    成功时 ``ok=True``，``coordinates_gcj02`` 为吸附后的 GCJ-02 经纬度
    （作为交付轨迹），``coordinates_px`` 为对应像素（overlay 用），
    ``coordinates_raw_gcj02`` 为原始轨迹（对照）。
    """
    empty: dict[str, Any] = {
        "ok": False, "reason": "", "method": "gentle",
        "coordinates_gcj02": [], "coordinates_px": [],
        "mean_snap_m": 0.0, "median_snap_m": 0.0, "n_points": 0,
        "n_snapped": 0, "length_before_m": 0.0, "length_after_m": 0.0,
    }
    pts = np.asarray(polyline_px, dtype=float)
    if len(pts) < 3:
        return {**empty, "reason": "轨迹点不足"}
    tree = getattr(net, "tree", None)
    samples = getattr(net, "samples", None)
    if tree is None or not samples:
        return {**empty, "reason": "无可用路网几何"}

    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    traj = np.asarray(
        [reg.pixel_to_mercator(float(px), float(py)) for px, py in pts],
        dtype=float,
    )

    d, idx = tree.query(traj)
    sample_arr = np.asarray(samples, dtype=float)
    owner = getattr(net, "owner", None)
    ways_owner = [owner[i] for i in idx] if owner is not None else []

    # --- 1. 距离容差内吸附 ---
    snap_mask = d <= tol_m

    # --- 2. 窗口多数表决（压道路抖动） ---
    n = len(traj)
    for i in range(n):
        if not snap_mask[i]:
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neigh = [ways_owner[j] for j in range(lo, hi)
                 if j != i and snap_mask[j]]
        if neigh and ways_owner[i] not in neigh:
            # 邻域都不在这条路上：孤点吸附多为平行路误吸
            if d[i] > tol_m * 0.5:
                snap_mask[i] = False

    out = traj.copy()
    out[snap_mask] = sample_arr[idx[snap_mask]]

    # --- 3. 跳变回退 ---
    reverted = 0
    for _ in range(3):  # 有限轮迭代，防止连锁
        steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
        bad = np.where(steps > max_jump_m)[0]
        if len(bad) == 0:
            break
        moved = False
        for i in bad:
            j = i + 1
            if snap_mask[j]:
                out[j] = traj[j]      # 回退后者
                snap_mask[j] = False
                moved = True
                reverted += 1
            elif snap_mask[i]:
                out[i] = traj[i]      # 后者已是原始点，回退前者
                snap_mask[i] = False
                moved = True
                reverted += 1
        if not moved:
            break

    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    max_step = float(steps.max()) if len(steps) else 0.0

    def _len(a: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))

    len_before = _len(traj)
    len_after = _len(out)
    snapped_d = d[snap_mask]
    coords = []
    from .geo.coords import mercator_to_lonlat
    coords = [mercator_to_lonlat(float(x), float(y)) for x, y in out]
    px_out = [reg.mercator_to_pixel(float(x), float(y)) for x, y in out]
    return {
        "ok": True,
        "reason": "",
        "method": "gentle",
        "crs": "GCJ-02",
        "coordinates_gcj02": [[float(a), float(b)] for a, b in coords],
        "coordinates_px": [[float(a), float(b)] for a, b in px_out],
        "mean_snap_m": round(float(snapped_d.mean()), 1) if len(snapped_d) else 0.0,
        "median_snap_m": round(float(np.median(snapped_d)), 1) if len(snapped_d) else 0.0,
        "n_points": n,
        "n_snapped": int(snap_mask.sum()),
        "n_reverted": reverted,
        "max_step_m": round(max_step, 1),
        "length_before_m": round(len_before, 1),
        "length_after_m": round(len_after, 1),
    }
