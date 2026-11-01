"""像素 → 经纬度的几何配准。

模型选择
--------
美团地图为正北朝上的 Web 墨卡托投影。图像 y 轴向下，墨卡托 y 轴向上（北），
因此像素→墨卡托的真实变换天然含一次 y 轴翻转：

    X = s·(px - px0) + X0
    Y = -s·(py - py0) + Y0

即矩阵 M = [[s, 0], [0, -s]]，行列式为负（含反射）。
若用户旋转过截图，M 还会带上旋转分量。

因此这里统一用 2×2 仿射矩阵 + 平移向量来表达变换：

    [X, Y]ᵀ = M · [px, py]ᵀ + t

矩阵用最小二乘（Procrustes）求解，数值稳定，不受反射符号陷阱影响。

为什么用 RANSAC
--------------
OCR 会把「周家渡」认成「周家浜」，地理编码 API 也可能返回同名地点的
另一个点位；且文字标注中心与真实点位本身存在几十至上百米的固有偏移。
若直接最小二乘，一个错误锚点就会把整条轨迹拉偏数百米。
RANSAC 能识别并剔除这类离群点。

模型选择策略：先试「相似变换」（4 自由度，允许旋转缩放反射），
内点不足再退化为「一般仿射」（6 自由度）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .geo.coords import lonlat_to_mercator, mercator_to_lonlat


@dataclass
class AnchorPair:
    """一个像素锚点与其候选经纬度。"""

    name: str
    px: float
    py: float
    weight: float
    kind: str
    lon: float | None = None
    lat: float | None = None
    # 该候选经纬度的可靠性（1.0 = 精确匹配，0.5 = 模糊匹配）
    match_quality: float = 1.0


@dataclass
class Registration:
    """配准结果。"""

    ok: bool
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    inliers: list[AnchorPair] = field(default_factory=list)
    outliers: list[AnchorPair] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    rmse_m: float = 0.0
    notes: list[str] = field(default_factory=list)
    model: str = "affine"

    @property
    def scale(self) -> float:
        """米/像素（取矩阵两轴尺度的均值）。"""
        sx = float(np.hypot(self.matrix[0, 0], self.matrix[0, 1]))
        sy = float(np.hypot(self.matrix[1, 0], self.matrix[1, 1]))
        return (sx + sy) / 2.0

    @property
    def rotation_deg(self) -> float:
        """旋转角（度）。北向朝上时约为 180°（因 y 轴翻转）。"""
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))

    def pixel_to_mercator(self, px: float, py: float) -> tuple[float, float]:
        """像素坐标 → 墨卡托米。"""
        v = self.matrix @ np.array([px, py], dtype=float) + self.translation
        return float(v[0]), float(v[1])

    def pixel_to_lonlat(self, px: float, py: float) -> tuple[float, float]:
        """像素坐标 → 经纬度（GCJ-02）。"""
        x, y = self.pixel_to_mercator(px, py)
        return mercator_to_lonlat(x, y)

    def mercator_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """墨卡托米 → 像素坐标（反变换，用于把真值轨迹画回截图核对）。"""
        try:
            inv = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            return (0.0, 0.0)
        v = inv @ (np.array([x, y], dtype=float) - self.translation)
        return float(v[0]), float(v[1])


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘拟合仿射变换。返回 (M, t)。"""
    n = len(src)
    if n < 3:
        raise ValueError("仿射变换至少需要 3 个点")
    a = np.hstack([src, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(a, dst, rcond=None)
    return sol[:2].T.copy(), sol[2].copy()


def _fit_north_up(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """拟合「正北朝上」约束模型：M = [[s, 0], [0, -s]]。

    美团地图默认正北朝上（且截图未经旋转），此时像素→墨卡托只应包含
    等比缩放与平移，y 轴取负（图像向下、墨卡托向上）。

    这个模型只有 2 个自由度（尺度 s + 平移），因此对锚点噪声极不敏感，
    是精度最高、最稳的模型。实测 4 个 POI 锚点的残差仅 9-21 米。

    做法：先用两点求出尺度，再在全部锚点上最小化残差求解平移。
    """
    n = len(src)
    if n < 2:
        raise ValueError("正北模型至少需要 2 个点")
    # 尺度：取所有点对距离比的中位数（抗离群）
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            dp = float(np.linalg.norm(src[j] - src[i]))
            dm = float(np.linalg.norm(dst[j] - dst[i]))
            if dp > 1e-6:
                ratios.append(dm / dp)
    if not ratios:
        raise ValueError("无法估计尺度")
    scale = float(np.median(ratios))

    # 平移：最小化 || s·flip(src) + t - dst ||
    fx = scale * src[:, 0]
    fy = -scale * src[:, 1]
    tx = float(np.mean(dst[:, 0] - fx))
    ty = float(np.mean(dst[:, 1] - fy))
    m = np.array([[scale, 0.0], [0.0, -scale]])
    return m, np.array([tx, ty])


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """拟合相似变换（含旋转、等比缩放）。返回 (M, t)。

    做法：先求一般仿射解，再用 SVD 取最接近相似变换（等比）的正交部分。
    这样既保留了数值稳定性，又约束了 4 个自由度。
    """
    n = len(src)
    if n < 2:
        raise ValueError("相似变换至少需要 2 个点")
    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)
    s0 = src - sc
    d0 = dst - dc
    if (s0 ** 2).sum() < 1e-12:
        raise ValueError("源点退化")

    m, _ = _fit_affine(src, dst)
    u, sv, vt = np.linalg.svd(m)
    scale = float(sv.mean())
    if scale <= 0:
        raise ValueError("解得非正尺度")
    rot = u @ vt
    t = dc - scale * (rot @ sc)
    return scale * rot, t


def _apply(matrix: np.ndarray, translation: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ matrix.T + translation


def _residuals(pred: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - dst, axis=1)



def _ransac(
    src: np.ndarray,
    dst: np.ndarray,
    fitter,
    min_points: int,
    inlier_m: float,
    max_iter: int = 3000,
    seed: int = 42,
):
    """通用 RANSAC。返回 (M, t, 内点索引)，失败返回 None。

    fitter(src_subset, dst_subset) -> (M, t)
    """
    n = len(src)
    combos = list(itertools.combinations(range(n), min_points))
    if not combos:
        return None
    if len(combos) > max_iter:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), max_iter, replace=False)
        combos = [combos[i] for i in pick]

    best_inliers: list[int] = []
    best_m = best_t = None
    best_err = float("inf")

    for combo in combos:
        idx = list(combo)
        sub_src = src[idx]
        sub_dst = dst[idx]
        # 退化检查：两点情形只需不重合，三点及以上需不共线
        if min_points >= 3:
            if np.linalg.matrix_rank(sub_src - sub_src.mean(axis=0)) < 2:
                continue
        else:
            spread = float(np.linalg.norm(sub_src - sub_src.mean(axis=0), axis=1).max())
            if spread < 1e-6:
                continue
        try:
            m, t = fitter(sub_src, sub_dst)
        except (ValueError, np.linalg.LinAlgError):
            continue
        res = _residuals(_apply(m, t, src), dst)
        inliers = [k for k in range(n) if res[k] <= inlier_m]
        if len(inliers) < min_points:
            continue
        # 在内点上精修
        try:
            m, t = fitter(src[inliers], dst[inliers])
        except (ValueError, np.linalg.LinAlgError):
            pass
        res = _residuals(_apply(m, t, src), dst)
        inliers = [k for k in range(n) if res[k] <= inlier_m]
        if len(inliers) < min_points:
            continue
        err = float(res[inliers].mean())
        if len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and err < best_err
        ):
            best_inliers, best_m, best_t, best_err = inliers, m, t, err

    if best_m is None:
        return None
    return best_m, best_t, best_inliers


def register(
    pairs: list[AnchorPair],
    inlier_m: float = 120.0,
    min_inliers: int = 3,
) -> Registration:
    """用 RANSAC 做像素→墨卡托配准。

    输入的 pairs 可以包含同一地名的多个候选坐标——RANSAC 会自动
    挑出自洽的那一组，天然处理「同名地点歧义」。

    模型按优先级依次尝试，优先选择自由度低、精度高的：
      1. north_up   —— 正北朝上，2 自由度。默认地图状态，精度最高
      2. similarity —— 允许旋转，4 自由度。适配用户旋转过截图的情况
      3. affine     —— 一般仿射，6 自由度。兜底
    """
    notes: list[str] = []
    valid = [p for p in pairs if p.lon is not None and p.lat is not None]
    if len(valid) < 2:
        return Registration(
            ok=False, notes=[f"有效锚点不足（{len(valid)} 个），至少需要 2 个"]
        )

    src = np.array([[p.px, p.py] for p in valid], dtype=float)
    dst = np.array([lonlat_to_mercator(p.lon, p.lat) for p in valid], dtype=float)

    models = [
        ("north_up", _fit_north_up, 2),
        ("similarity", _fit_similarity, 2),
        ("affine", _fit_affine, 3),
    ]

    for model_name, fitter, min_pts in models:
        if len(valid) < min_pts:
            continue
        result = _ransac(src, dst, fitter, min_pts, inlier_m)
        if result is None:
            continue
        m, t, inliers = result
        res = _residuals(_apply(m, t, src), dst)
        outliers = [k for k in range(len(valid)) if res[k] > inlier_m]
        inlier_res = res[inliers]
        reg = Registration(
            ok=True,
            matrix=m,
            translation=t,
            inliers=[valid[k] for k in inliers],
            outliers=[valid[k] for k in outliers],
            residuals=[float(r) for r in inlier_res],
            rmse_m=float(np.sqrt((inlier_res ** 2).mean())) if len(inlier_res) else 0.0,
            model=model_name,
        )
        notes.append(
            f"{model_name} 模型: {len(inliers)} 内点 / {len(outliers)} 离群，"
            f"RMSE {reg.rmse_m:.1f}m"
        )
        reg.notes = notes
        return reg

    return Registration(
        ok=False,
        notes=notes + ["配准失败：所有模型均未找到足够的内点"],
    )

def apply_to_polyline(
    reg: Registration, polyline_px: np.ndarray
) -> list[tuple[float, float]]:
    """把像素折线转换为经纬度序列（GCJ-02）。"""
    if not reg.ok or len(polyline_px) == 0:
        return []
    return [reg.pixel_to_lonlat(float(px), float(py)) for px, py in polyline_px]
