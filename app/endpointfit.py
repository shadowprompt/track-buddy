"""起终点锚定：用「起终点徽章 + 用户路名」定出相似变换。

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

配准偏大 **26%**，轨迹被拉长到 15739 米（真实约 12500 米），
东端因此被甩过黄浦江——实测轨迹起点落在**浦东侧**（59/417 个点在浦东），
路名归属报出「打浦路隧道 / 时光印记大道 / 世博大道 / 东岸绿道」。
用户反馈「我根本没去浦东」正是这个原因。

尺度为什么会被污染
------------------
锚点全是**远距离大范围地标**（上海西站、上海站、桃浦中央绿地、黄浦区、长宁区）。
地图引擎把地标的**文字标注**放在视野空白处，与真实点位差 100-2000 米，
且偏移**有方向性**。尺度 = 地理距离 / 像素距离，两端同向偏移会放大尺度。
实测逐对锚点隐含尺度从 12.57 到 19.86 m/px，极差 58%。

为什么用「起终点徽章」
--------------------
前几种思路都失败在同一个地方——**残差类目标函数对尺度不敏感**：

  * 路网贴路距离：低缩放下路网密集、平行路多，尺度偏 26% 时
    轨迹仍能贴在某条路上（实测尺度从 10.5 扫到 17.0，残差单调下降）；
  * 路名链骨架跨度：接触点随位置漂移，逐段比值从 10.2 到 26.0 波动；
  * 投影覆盖长度：链式 DP 选出的实例随尺度跳变，曲线不连续。

但**起终点徽章是纯色块检测出来的像素位置**——它没有「文字标注偏移」
这个问题（地标文字标注偏移实测可达 100-2000 米）。

用户提供的路名列表则给出语义锚点：
  * 起点在列表**前几条**路上
  * 终点在列表**后几条**路上

于是：起终点徽章的像素距离（精确）+ 两端路名的真实地理距离
-> 直接定出尺度。这与地标标注完全无关。

算法（两阶段，保证速度）
------------------------
阶段 1：**链式 DP** 选出空间连贯的路名实例（`app/roadchain.py`）
         ——一次 DP 就能把同名路的多实例问题解决掉
阶段 2：在**选定实例内**枚举起终点锚点对，两点定相似变换，
         用固定实例的残差打分

阶段 2 不再重跑链式 DP（实例已定），所以快得多：
实测整轮 < 3 秒，而「先枚举再 DP」的朴素做法要几分钟。

最后再迭代一次「用新变换重跑链式 DP + 重选实例」，让两者相互收敛。

实测（上海城市级截图）
----------------------
  配准：尺度 15.73 m/px，轨迹 15739 米，起点落在**浦东侧**
  本模块：尺度 13.15 m/px，轨迹 13056 米，起点回到**浦西侧**（121.459）
  独立标尺真值：12.47 / 12.58 / 13.42
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .roadchain import ChainResult, RoadInstance, build_candidates, select_chain


@dataclass
class AnchorFit:
    """起终点锚定拟合结果。"""

    ok: bool = False
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0
    rotation_deg: float = 0.0
    value: float = float("inf")          # 路名链残差（米）
    residuals: list[float] = field(default_factory=list)
    start_road: str = ""
    end_road: str = ""
    chain: ChainResult | None = None
    notes: list[str] = field(default_factory=list)

    def apply(self, px: np.ndarray) -> np.ndarray:
        """像素 (N,2) -> 墨卡托米 (N,2)。"""
        p = np.asarray(px, dtype=float).copy()
        p[:, 1] = -p[:, 1]
        return (self.matrix @ p.T).T + self.translation


def two_point_similarity(
    p_from: tuple[float, float], p_to: tuple[float, float],
    g_from: tuple[float, float], g_to: tuple[float, float],
) -> tuple[float, float, np.ndarray] | None:
    """两点定相似变换。

    像素点 p_from -> 地理点 g_from，p_to -> g_to。
    4 自由度（尺度 s、旋转 θ、平移 tx、ty），4 个方程，唯一闭式解。

    返回 (s, θ, t)，其中 t 作用在**原始像素**（y 向下）上，
    与 `app/register.py` 的 `Registration.matrix` 约定完全一致。

    ⚠️ **坐标系约定必须与 Registration 一致**（踩过坑）：
    `Registration.pixel_to_mercator` 直接做 `matrix @ [px, py] + translation`，
    其中 matrix 含 y 翻转（`[[s,0],[0,-s]]`）。
    若这里返回的是「作用在已翻转像素上」的矩阵，
    拼进 reg 后整条轨迹会**南北镜像**——
    实测起点纬度 31.191 变成 31.214、终点 31.213 变成 31.192（正好互换），
    后续平移精修被迫移动 34 公里才能补偿。

    解法：先在「y 已翻转」空间里解（那里是标准相似变换），
    再把矩阵与平移换算回「原始像素」空间。
    """
    # --- 步骤 1：在翻转空间解标准相似变换 ---
    # geo = R(θ) · s · (px, -py) + t_flip
    sp = np.array([p_from[0], -p_from[1]], dtype=float)
    ep = np.array([p_to[0], -p_to[1]], dtype=float)
    dp = ep - sp
    dg = np.array([g_to[0] - g_from[0], g_to[1] - g_from[1]], dtype=float)
    n = float(math.hypot(dp[0], dp[1]))
    if n < 1e-9:
        return None
    s = float(math.hypot(dg[0], dg[1])) / n
    th = math.atan2(dg[1], dg[0]) - math.atan2(dp[1], dp[0])

    # --- 步骤 2：换算回原始像素空间 ---
    # M_raw = R(θ) · diag(s, -s)，使 M_raw · [px, py] = R(θ) · s · (px, -py)
    c, si = math.cos(th), math.sin(th)
    R = np.array([[c, -si], [si, c]])
    M_raw = R @ np.diag([s, -s])
    sp_raw = np.array([p_from[0], p_from[1]], dtype=float)
    t_raw = np.array(g_from, dtype=float) - M_raw @ sp_raw
    return s, th, t_raw


def _apply(px: np.ndarray, s: float, theta: float, t: np.ndarray) -> np.ndarray:
    """像素 (N,2) -> 墨卡托米 (N,2)，与 Registration 约定一致。"""
    c, si = math.cos(theta), math.sin(theta)
    R = np.array([[c, -si], [si, c]])
    M = R @ np.diag([s, -s])
    p = np.asarray(px, dtype=float)
    return (p @ M.T) + t


def _compose(s: float, theta: float, t: np.ndarray) -> np.ndarray:
    """返回可直接赋给 `Registration.matrix` 的矩阵。"""
    c, si = math.cos(theta), math.sin(theta)
    R = np.array([[c, -si], [si, c]])
    return R @ np.diag([s, -s])


def _residual_by_fixed_instances(
    traj: np.ndarray,
    instances: list[RoadInstance],
    weights: np.ndarray,
) -> tuple[float, list[float]]:
    """用**固定**的实例集合算残差（不重跑链式 DP，快）。

    按弧长等分逐段算中位距离。等分而非 DP：这里只是打分，
    需要的是「轨迹整体离这组路名有多远」，不是精确逐段对应。
    """
    ml = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(ml)])
    total = float(cum[-1]) or 1.0
    n = len(instances)
    val = 0.0
    ws = 0.0
    res: list[float] = []
    for i, inst in enumerate(instances):
        if inst.n == 0:
            res.append(1e5)
            continue
        lo = total * i / n
        hi = total * (i + 1) / n
        m = (cum >= lo) & (cum <= hi)
        p = traj[m]
        if len(p) < 2:
            j = int(np.argmin(np.abs(cum - lo)))
            p = traj[max(0, j - 1):j + 2]
        if len(p) > 30:
            p = p[np.linspace(0, len(p) - 1, 30).astype(int)]
        r = inst.median_dist_to(p)
        res.append(r)
        val += weights[i] * r
        ws += weights[i]
    return val / (ws or 1.0), res


def _anchor_samples(inst: RoadInstance, k: int) -> np.ndarray:
    """从实例上取 k 个均匀分布的候选锚点。"""
    if inst.n == 0:
        return np.zeros((0, 2))
    step = max(1, inst.n // k)
    return inst.points[::step][:k]


def fit_by_endpoints(
    px: np.ndarray,
    arc_px: np.ndarray,
    start_px: tuple[float, float],
    end_px: tuple[float, float],
    road_names: list[tuple[str, list[list[tuple[float, float]]]]],
    weights: list[float] | None = None,
    start_span: int = 3,
    end_span: int = 2,
    scale_range: tuple[float, float] = (6.0, 22.0),
    n_anchors: int = 14,
    outer_rounds: int = 2,
) -> AnchorFit:
    """主入口：用起终点徽章 + 路名锚定相似变换。

    px:         (N,2) 像素轨迹
    arc_px:     (N,) 像素弧长
    start_px:   起点徽章像素坐标
    end_px:     终点徽章像素坐标
    road_names: [(路名, [段, ...]), ...]，段是墨卡托米坐标，按骑行顺序
    start_span: 起点信任列表前几条路
    end_span:   终点信任列表后几条路
    """
    out = AnchorFit()
    usable = [(nm, segs) for nm, segs in road_names if segs]
    if len(usable) < 3 or len(px) < 10:
        out.notes.append("可用路名不足或轨迹点太少")
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

    # 起点/终点信任的路名范围
    s_hi = max(1, min(start_span, len(usable)))
    e_lo = max(0, len(usable) - max(1, min(end_span, len(usable))))
    start_ids = list(range(s_hi))
    end_ids = list(range(e_lo, len(usable)))

    # --- 阶段 1：链式 DP 选实例 ---
    # 用「初始近似变换」跑一次 DP。这里用一个粗略的近似：
    # 假设轨迹起点落在第一条路名上，终点落在最后一条路名上，
    # 用两点定相似变换得到初始解。
    chain: ChainResult | None = None
    best: tuple[float, float, float, np.ndarray, ChainResult, list[float],
                str, str] | None = None
    n_eval = 0

    for round_i in range(max(1, outer_rounds)):
        # 本轮用来做链式 DP 的参考变换
        if chain is None:
            # 用「首条路 ↔ 末条路」的两点估计做初始参考
            s_inst = candidates[start_ids[0]][0] if candidates[start_ids[0]] else None
            e_inst = candidates[end_ids[-1]][0] if candidates[end_ids[-1]] else None
            if s_inst is None or e_inst is None:
                out.notes.append("起终点路名没有可用实例")
                return out
            ps = s_inst.points[len(s_inst.points) // 2]
            pe = e_inst.points[len(e_inst.points) // 2]
            r0 = two_point_similarity(start_px, end_px,
                                      (float(ps[0]), float(ps[1])),
                                      (float(pe[0]), float(pe[1])))
            if r0 is None:
                out.notes.append("起终点锚定初始解失败")
                return out
            ref_traj = _apply(px, r0[0], r0[1], r0[2])
        else:
            ref_traj = _apply(px, best[1], best[2], best[3])

        chain = select_chain(candidates, ref_traj, arc_px)
        if not chain.ok:
            if best is not None:
                break
            out.notes.append("链式 DP 找不到空间连贯的实例组合")
            return out

        # --- 阶段 2：在选定实例内枚举起终点锚点 ---
        for si in start_ids:
            if si >= len(chain.chosen) or chain.chosen[si].n == 0:
                continue
            pts_s = _anchor_samples(chain.chosen[si], n_anchors)
            for ei in end_ids:
                if ei >= len(chain.chosen) or chain.chosen[ei].n == 0:
                    continue
                pts_e = _anchor_samples(chain.chosen[ei], n_anchors)
                for ps in pts_s:
                    for pe in pts_e:
                        r = two_point_similarity(
                            start_px, end_px,
                            (float(ps[0]), float(ps[1])),
                            (float(pe[0]), float(pe[1])))
                        if r is None:
                            continue
                        s, th, t = r
                        if not (scale_range[0] <= s <= scale_range[1]):
                            continue
                        traj = _apply(px, s, th, t)
                        v, res = _residual_by_fixed_instances(
                            traj, chain.chosen, w)
                        if not math.isfinite(v):
                            continue
                        n_eval += 1
                        if best is None or v < best[0]:
                            best = (v, s, th, t, chain, res,
                                    usable[si][0], usable[ei][0])
        if best is None:
            out.notes.append("起终点锚定找不到可行解")
            return out
        # 本轮最优已定，下一轮用它重跑链式 DP（让实例与变换相互收敛）
        if round_i + 1 >= max(1, outer_rounds):
            break

    v, s, th, t, chain, res, s_road, e_road = best
    out.ok = True
    out.matrix = _compose(s, th, t)
    out.translation = t
    out.scale = s
    out.rotation_deg = math.degrees(th)
    out.value = v
    out.residuals = res
    out.start_road = s_road
    out.end_road = e_road
    out.chain = chain
    out.notes.append(
        f"起终点锚定：尺度 {s:.2f} 米/像素，旋转 {out.rotation_deg:+.2f}°，"
        f"起点锚在「{s_road}」、终点锚在「{e_road}」，"
        f"路名链残差 {v:.0f} 米（评估 {n_eval} 个组合）"
    )
    return out
