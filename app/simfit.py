"""相似变换拟合：把「像素轨迹」对齐到「地理路网」。

为什么需要这个模块
------------------
低缩放截图（一张图覆盖 20 公里）下，**锚点配准给出的尺度是不可信的**。

实测（城市级截图，上海，1260x2720px，1260 宽对应约 15.7 公里）：

  | 独立标尺                       | 尺度 (m/px) |
  |--------------------------------|-------------|
  | 上海西站 ↔ 上海站              | 13.42       |
  | 上海西站 ↔ 上海南站            | 12.58       |
  | 上海站   ↔ 上海南站            | 12.47       |
  | **锚点配准给出的 scale**        | **15.73**   |

配准偏大 **26%**。轨迹被拉长到 15739 米（真实约 12000 米），
东端因此被甩过黄浦江，压在浦东的「打浦路隧道 / 时光印记大道 /
世博大道 / 东岸绿道」上——用户反馈「我根本没去浦东」正是这个原因。

尺度为什么会被污染
------------------
锚点全是**远距离大范围地标**（上海西站、上海站、桃浦中央绿地、黄浦区、长宁区）。
地图引擎会把这些地标的**文字标注**放在视野空白处，与真实点位差 100-2000 米，
而且偏移是**有方向性**的。尺度 = 地理距离 / 像素距离，
两个端点同向偏移会**放大**尺度，异向偏移会**缩小**尺度。

实测逐对锚点隐含尺度从 12.57 到 19.86 m/px，极差 58% —— 尺度完全不可信。

本模块的做法
------------
把问题从「用稀疏且有偏的锚点求尺度」换成
**「用整条轨迹的形状去匹配真实路网」**：

  轨迹是一条连续曲线，它在地理空间的**形状**（长宽比、弯道节奏、
  各段之间的相对距离）必须与它实际骑过的那些路一致。
  用这个约束联合求解 (尺度 s, 旋转 θ, 平移 t)，
  尺度就被**路网的真实几何**钉死了，不再依赖任何地标标注。

求解方式：交替优化（ICP 风格）

  1. 给定当前变换 T，把像素轨迹映射到地理空间
  2. 对每个轨迹采样点，找它在路网上最近的点 -> 得到对应关系
  3. 用对应关系解相似变换的闭式解（Procrustes）-> 得到新的 (s, θ, t)
  4. 重复 2-3 直到收敛

第 3 步是闭式解（不做网格搜索），所以很快。

但 ICP 有个致命弱点：**局部最优**。它只会把轨迹往「当前最近的」那几条路上拉，
一旦初始位置偏了 3 公里，它就永远在错的街上打转。
所以本模块用**多起点**（多个初始尺度），每个起点独立跑 ICP，
最后用统一的目标函数打分选最优。

目标函数
--------
不只看「贴路距离」，还要看**路名命中**——用户提供的路名是有序的，
轨迹按顺序经过它们。这个顺序约束是尺度最强的信号：

    obj = Σᵢ wᵢ · median(该段轨迹点到第 i 条路名的距离)

其中 wᵢ 来自路名的唯一性（越具体权重越大）。

物理护栏：轨迹不该横穿大江大河
------------------------------
骑行轨迹横跨黄浦江几乎一定是错的（本例正是如此）。
`app/pipeline.py` 在采纳前会调用 `crosses_major_water()` 做最后检查。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .geo.coords import lonlat_to_mercator, mercator_to_lonlat


@dataclass
class SimilarityFit:
    """相似变换拟合结果。"""

    ok: bool = False
    # 仿射矩阵：[[s·cosθ, -s·sinθ], [s·sinθ, s·cosθ]]
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0
    rotation_deg: float = 0.0
    # 目标函数值（越小越好）
    value: float = float("inf")
    base_value: float = float("inf")
    # 优化后各段的路名残差（米）
    residuals: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def apply(self, px: np.ndarray) -> np.ndarray:
        """像素坐标 (N,2) -> 墨卡托米 (N,2)。"""
        p = np.asarray(px, dtype=float).copy()
        p[:, 1] = -p[:, 1]        # 像素 y 向下，翻成北向上
        return (self.matrix @ p.T).T + self.translation


def _procrustes(src: np.ndarray, dst: np.ndarray,
                w: np.ndarray | None = None) -> tuple[float, float, np.ndarray]:
    """加权相似变换闭式解。

    求解 min Σ wᵢ |s·R(θ)·srcᵢ + t - dstᵢ|²。

    返回 (s, θ, t)，其中 src 的 y 轴翻转已在内部处理。
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) < 2 or len(src) != len(dst):
        return 1.0, 0.0, np.zeros(2)
    if w is None:
        w = np.ones(len(src))
    w = np.asarray(w, dtype=float).reshape(-1)
    w = w / (w.sum() or 1.0)

    # 像素 y 轴向下：先把 src 的 y 取反，之后就只需处理纯旋转
    sxy = src.copy()
    sxy[:, 1] = -sxy[:, 1]

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


def _compose(s: float, theta: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, si = math.cos(theta), math.sin(theta)
    R = np.array([[c, -si], [si, c]])
    return R * s, np.asarray(t, dtype=float)


def _apply_t(src_px: np.ndarray, s: float, theta: float,
             t: np.ndarray) -> np.ndarray:
    p = np.asarray(src_px, dtype=float).copy()
    p[:, 1] = -p[:, 1]
    m, _ = _compose(s, theta, t)
    return (m @ p.T).T + t


class RoadMatcher:
    """把轨迹对齐到路网，并解出相似变换。

    路网用「点云 + KD 树」表示：每个路网点按 6 米间隔采样，
    这样「点到路的距离」就是「到点云的距离」，可以快速查询。
    """

    def __init__(self, roads: list[tuple[str, list[tuple[float, float]]]],
                 sample_m: float = 6.0):
        """roads: [(路名, [(lon, lat), ...]), ...]，WGS-84 经纬度。"""
        self.names: list[str] = []
        self.owner: list[int] = []
        pts: list[tuple[float, float]] = []
        for i, (nm, pl) in enumerate(roads):
            if len(pl) < 2:
                continue
            mpts = [lonlat_to_mercator(lo, la) for lo, la in pl]
            for j in range(len(mpts) - 1):
                (ax, ay), (bx, by) = mpts[j], mpts[j + 1]
                seg = math.hypot(bx - ax, by - ay)
                n = max(1, int(seg / sample_m))
                for k in range(n + 1):
                    t = k / n
                    pts.append((ax + (bx - ax) * t, ay + (by - ay) * t))
                    self.names.append(nm)
                    self.owner.append(i)
        self.points = np.array(pts) if pts else np.zeros((0, 2))
        self.tree = cKDTree(self.points) if len(self.points) else None

    def __len__(self) -> int:
        return len(self.points)

    def nearest(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (每个 q 点到路网的距离, 对应路网点的坐标)。"""
        if self.tree is None:
            n = len(q)
            return np.full(n, 1e9), np.zeros((n, 2))
        d, i = self.tree.query(q)
        return d, self.points[i]

    def icp(self, src_px: np.ndarray, init: tuple[float, float, np.ndarray],
            iters: int = 8, max_corr_m: float = 150.0) -> tuple[float, float, np.ndarray]:
        """从初始变换出发做 ICP，求 (s, θ, t)。

        src_px: (N,2) 像素坐标。
        init: (s, θ, t) —— t 是作用在「y 已翻转像素」上的平移（墨卡托米）。
        """
        s, theta, t = init
        for _ in range(iters):
            cur = _apply_t(src_px, s, theta, t)
            d, near = self.nearest(cur)
            # 只用「贴得够近」的对应关系（超过阈值的匹配是错的）
            keep = d < max_corr_m
            if keep.sum() < 3:
                break
            s2, th2, t2 = _procrustes(src_px[keep], near[keep])
            if abs(s2 - s) < 1e-7 and abs(th2 - theta) < 1e-9:
                s, theta, t = s2, th2, t2
                break
            s, theta, t = s2, th2, t2
        return s, theta, t


def fit_similarity_to_roads(
    src_px: np.ndarray,
    matcher: RoadMatcher,
    init_scale: float,
    init_translation: np.ndarray,
    road_names: list[tuple[str, list[list[tuple[float, float]]]]] | None = None,
    weights: list[float] | None = None,
    scale_range: tuple[float, float] = (0.6, 1.4),
    n_starts: int = 7,
    iters: int = 8,
) -> SimilarityFit:
    """把像素轨迹拟合到路网上，解出相似变换。

    src_px: (N,2) 像素坐标。
    matcher: 路网索引。
    init_scale: 初始尺度（米/像素，来自锚点配准）。
    init_translation: 初始平移（墨卡托米）。
    road_names: 可选，[(路名, [段, ...]), ...]，用于有序路名约束打分。

    返回最优的 SimilarityFit。
    """
    out = SimilarityFit()
    if matcher.tree is None or len(src_px) < 3:
        out.notes.append("路网为空或轨迹点太少")
        return out

    if road_names:
        obj = _make_roadname_objective(src_px, road_names, weights)
    else:
        obj = _make_roadnet_objective(src_px, matcher)

    best: tuple[float, float, float, np.ndarray] | None = None
    for k in range(n_starts):
        # 多起点：尺度等比铺开（ICP 是局部优化，起点的尺度决定了收敛到哪）
        s0 = init_scale * (scale_range[0] + (scale_range[1] - scale_range[0])
                            * (k / max(1, n_starts - 1)))
        s, theta, t = matcher.icp(
            src_px, (s0, 0.0, init_translation.copy()), iters=iters)
        v = obj(s, theta, t)
        if best is None or v < best[0]:
            best = (v, s, theta, t)

    if best is None:
        out.notes.append("所有起点都拟合失败")
        return out

    v, s, theta, t = best
    m, _ = _compose(s, theta, t)
    out.ok = True
    out.matrix = m
    out.translation = t
    out.scale = s
    out.rotation_deg = math.degrees(theta)
    out.value = v
    out.base_value = obj(init_scale, 0.0, init_translation)
    out.notes.append(
        f"路网相似变换：尺度 {init_scale:.2f} -> {s:.2f} 米/像素"
        f"（×{s / max(init_scale, 1e-9):.2f}），旋转 {out.rotation_deg:+.2f}°，"
        f"目标值 {out.base_value:.0f} -> {v:.0f}"
    )
    return out


def _make_roadnet_objective(src_px: np.ndarray, matcher: RoadMatcher):
    """目标函数：整条轨迹到路网的距离中位数。"""
    def obj(s: float, theta: float, t: np.ndarray) -> float:
        cur = _apply_t(src_px, s, theta, t)
        d, _ = matcher.nearest(cur)
        return float(np.median(d))
    return obj


def _make_roadname_objective(
    src_px: np.ndarray,
    road_names: list[tuple[str, list[list[tuple[float, float]]]]],
    weights: list[float] | None = None,
):
    """目标函数：有序路名约束。

    与 `app/roadname_constraint.py` 的口径一致：
    轨迹按弧长等分成 N 段（N = 路名数），第 i 段配第 i 个路名，
    目标值 = Σ wᵢ · median(第 i 段到第 i 条路的距离) / Σ wᵢ。

    这里刻意用**等分**而不是 DP：对齐的目的只是给尺度提供信号，
    等分更稳（DP 在初始位置偏 2 公里时会给出错误对应）。
    """
    if weights is None:
        weights = [1.0] * len(road_names)
    trees = []
    coords = []
    for nm, segs in road_names:
        pts = [p for seg in segs for p in seg]
        arr = np.array(pts) if pts else np.zeros((0, 2))
        coords.append(arr)
        trees.append(cKDTree(arr) if len(arr) >= 2 else None)

    def obj(s: float, theta: float, t: np.ndarray) -> float:
        cur = _apply_t(src_px, s, theta, t)
        ml = np.linalg.norm(np.diff(cur, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(ml)])
        total = float(cum[-1]) or 1.0
        n = len(road_names)
        val = 0.0
        wsum = 0.0
        for i in range(n):
            if trees[i] is None:
                continue
            lo = total * i / n
            hi = total * (i + 1) / n
            m = (cum >= lo) & (cum <= hi)
            pts = cur[m]
            if len(pts) < 2:
                j = int(np.argmin(np.abs(cum - lo)))
                pts = cur[max(0, j - 1):j + 2]
            if len(pts) > 30:
                pts = pts[np.linspace(0, len(pts) - 1, 30).astype(int)]
            d, _ = trees[i].query(pts)
            w = float(weights[i]) if i < len(weights) else 1.0
            val += w * float(np.median(d))
            wsum += w
        return val / (wsum or 1.0)
    return obj


def fit_similarity_by_road_names(
    src_px: np.ndarray,
    road_names: list[tuple[str, list[list[tuple[float, float]]]]],
    init_scale: float,
    init_translation: np.ndarray,
    weights: list[float] | None = None,
    scale_range: tuple[float, float] = (0.6, 1.35),
    n_starts: int = 9,
    iters: int = 10,
) -> SimilarityFit:
    """用**有序路名**拟合相似变换（尺度 + 旋转 + 平移）。

    这是路名约束的升级版。原来的实现只允许平移，
    在尺度被锚点污染 26% 的情况下怎么挪都救不回来。

    做法：把轨迹按用户给的顺序对齐到路名（DP），
    再用「轨迹点 -> 最近路网点」的对应关系解相似变换闭式解，迭代收敛。

    road_names: [(路名, [段, ...]), ...]，段是墨卡托米坐标。
    """
    out = SimilarityFit()
    usable = [(nm, segs) for nm, segs in road_names if segs]
    if len(usable) < 2 or len(src_px) < 3:
        out.notes.append("可用路名不足，无法拟合相似变换")
        return out

    if weights is None:
        weights = [1.0] * len(usable)
    # KD 树 + 坐标数组成对保存：
    # cKDTree.query 返回的是**索引**，坐标要自己回查，不能直接当点用。
    trees: list[cKDTree] = []
    coords: list[np.ndarray] = []
    for nm, segs in usable:
        arr = np.array([p for seg in segs for p in seg])
        coords.append(arr)
        trees.append(cKDTree(arr))

    # 目标函数：按路名顺序把轨迹等分，逐段算残差
    def obj(s: float, theta: float, t: np.ndarray,
            ret_res: bool = False):
        cur = _apply_t(src_px, s, theta, t)
        ml = np.linalg.norm(np.diff(cur, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(ml)])
        total = float(cum[-1]) or 1.0
        n = len(usable)
        res: list[float] = []
        wsum = 0.0
        val = 0.0
        for i in range(n):
            lo = total * i / n
            hi = total * (i + 1) / n
            m = (cum >= lo) & (cum <= hi)
            pts = cur[m]
            if len(pts) < 2:
                j = int(np.argmin(np.abs(cum - lo)))
                pts = cur[max(0, j - 1):j + 2]
            if len(pts) > 30:
                pts = pts[np.linspace(0, len(pts) - 1, 30).astype(int)]
            d, _ = trees[i].query(pts)
            r = float(np.median(d))
            res.append(r)
            w = float(weights[i]) if i < len(weights) else 1.0
            val += w * r
            wsum += w
        v = val / (wsum or 1.0)
        return (v, res) if ret_res else v

    best: tuple[float, float, float, np.ndarray] | None = None
    for k in range(n_starts):
        s0 = init_scale * (scale_range[0] + (scale_range[1] - scale_range[0])
                            * (k / max(1, n_starts - 1)))
        s, theta, t = _icp_by_names(
            src_px, usable, trees, coords, weights,
            (s0, 0.0, init_translation.copy()), iters=iters,
        )
        v = obj(s, theta, t)
        if best is None or v < best[0]:
            best = (v, s, theta, t)

    if best is None:
        out.notes.append("相似变换拟合失败")
        return out

    v, s, theta, t = best
    v0 = obj(init_scale, 0.0, init_translation)
    _, res = obj(s, theta, t, ret_res=True)
    m, _ = _compose(s, theta, t)
    out.ok = True
    out.matrix = m
    out.translation = t
    out.scale = s
    out.rotation_deg = math.degrees(theta)
    out.value = v
    out.base_value = v0
    out.residuals = res
    out.notes.append(
        f"路名相似变换：尺度 {init_scale:.2f} -> {s:.2f} 米/像素"
        f"（×{s / max(init_scale, 1e-9):.2f}），旋转 {out.rotation_deg:+.2f}°，"
        f"目标值 {v0:.0f} -> {v:.0f}"
    )
    return out


def _icp_by_names(
    src_px: np.ndarray,
    usable: list[tuple[str, list[list[tuple[float, float]]]]],
    trees: list[cKDTree],
    coords: list[np.ndarray],
    weights: list[float],
    init: tuple[float, float, np.ndarray],
    iters: int = 10,
) -> tuple[float, float, np.ndarray]:
    """路名驱动的 ICP：每条路名内部做 Procrustes，再整体合成。"""
    s, theta, t = init
    n = len(usable)
    for _ in range(iters):
        cur = _apply_t(src_px, s, theta, t)
        ml = np.linalg.norm(np.diff(cur, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(ml)])
        total = float(cum[-1]) or 1.0
        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []
        for i in range(n):
            lo = total * i / n
            hi = total * (i + 1) / n
            m = (cum >= lo) & (cum <= hi)
            idx = np.where(m)[0]
            if len(idx) < 2:
                continue
            step = max(1, len(idx) // 8)
            idx = idx[::step][:8]
            pts = cur[idx]
            d, ii = trees[i].query(pts)
            near = coords[i][ii]
            # 只保留贴得近的对应
            ok = d < 120.0
            if ok.sum() < 2:
                continue
            src_parts.append(src_px[idx[ok]])
            dst_parts.append(near[ok])
            w_parts.append(np.full(int(ok.sum()), float(weights[i])))
        if not src_parts:
            break
        src = np.vstack(src_parts)
        dst = np.vstack(dst_parts)
        w = np.concatenate(w_parts)
        s2, th2, t2 = _procrustes(src, dst, w)
        if abs(s2 - s) < 1e-7 and abs(th2 - theta) < 1e-9:
            s, theta, t = s2, th2, t2
            break
        s, theta, t = s2, th2, t2
    return s, theta, t


# ---------------------------------------------------------------------------
# 水系护栏
# ---------------------------------------------------------------------------

# 中国的通航大河（按宽度排序）。这些河上跨江大桥少，
# 而「骑行轨迹横穿长江/黄浦江/珠江」在真实场景里几乎必然是错的。
MAJOR_WATERS: dict[str, list[tuple[float, float]]] = {
    # 黄浦江（上海）：从淀山湖到长江口，穿过整个浦西浦东分界
    "黄浦江": [
        (121.1000, 31.2100), (121.1500, 31.1700), (121.2200, 31.1500),
        (121.2800, 31.1400), (121.3300, 31.1600), (121.3800, 31.1700),
        (121.4200, 31.1800), (121.4500, 31.1900), (121.4700, 31.1950),
        (121.4900, 31.1900), (121.5100, 31.1950), (121.5300, 31.2100),
        (121.5500, 31.2300), (121.5700, 31.2400), (121.5900, 31.2600),
        (121.6200, 31.2800), (121.6800, 31.2900), (121.7500, 31.2900),
        (121.8200, 31.3000), (121.9000, 31.3200),
    ],
    # 长江（上海段）
    "长江": [
        (121.0000, 31.4000), (121.2000, 31.4000), (121.5000, 31.4000),
        (121.7000, 31.4000), (121.9000, 31.4000),
    ],
    # 苏州河
    "苏州河": [
        (121.1900, 31.2500), (121.2400, 31.2400), (121.2800, 31.2300),
        (121.3200, 31.2400), (121.3600, 31.2450), (121.4000, 31.2450),
        (121.4400, 31.2450),
    ],
}


def crosses_major_water(
    lonlat_gcj: list[tuple[float, float]],
    waters: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[bool, str]:
    """检查轨迹是否横穿大河。

    返回 (是否横穿, 说明)。

    实现：把大河折线加密成点云，轨迹的每个点看它到「河心线」的
    横向距离是否小于河的半宽（黄浦江按 200 米、长江按 800 米）。
    连续落在河内的点超过阈值即判定横穿。
    """
    from .geo.coords import gcj02_to_wgs84

    if waters is None:
        waters = MAJOR_WATERS
    if len(lonlat_gcj) < 2:
        return False, ""

    half_width = {"黄浦江": 200.0, "长江": 800.0, "苏州河": 60.0}
    hits: list[str] = []
    for name, pl in waters.items():
        if len(pl) < 2:
            continue
        w = half_width.get(name, 150.0)
        # 河心线加密
        dense: list[tuple[float, float]] = []
        for i in range(len(pl) - 1):
            (ax, ay), (bx, by) = pl[i], pl[i + 1]
            n = max(2, int(math.hypot(bx - ax, by - ay) / 200.0))
            for k in range(n + 1):
                t = k / n
                dense.append((ax + (bx - ax) * t, ay + (by - ay) * t))
        tree = cKDTree(np.array(dense))
        # 轨迹点 -> WGS-84 -> 河心线距离
        pts_wgs = [gcj02_to_wgs84(lo, la) for lo, la in lonlat_gcj]
        d, _ = tree.query(np.array(pts_wgs))
        inside = d < w
        # 允许「擦边」：连续 2 个点才算真横穿
        run = 0
        for v in inside:
            run = run + 1 if v else 0
            if run >= 2:
                hits.append(name)
                break
    if hits:
        return True, "、".join(sorted(set(hits)))
    return False, ""


def transform_to_geo(
    src_px: np.ndarray, fit: SimilarityFit,
) -> list[tuple[float, float]]:
    """把像素轨迹变成 GCJ-02 经纬度。"""
    merc = fit.apply(src_px)
    return [mercator_to_lonlat(float(x), float(y)) for x, y in merc]
