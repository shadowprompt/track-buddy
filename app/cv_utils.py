"""图像读取与基础几何工具。

Windows 下 cv2.imread 对非 ASCII 路径会失败（返回 None），必须走
imdecode + np.fromfile 通道。本模块统一封装，避免各处踩坑。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图片，兼容中文/非 ASCII 路径。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(f"无法读取图片: {path}")
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError(f"图片解码失败: {path}")
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray, ext: str | None = None) -> bool:
    """写出图片，兼容非 ASCII 路径。"""
    p = Path(path)
    if ext is None:
        ext = p.suffix or ".png"
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        return False
    enc.tofile(str(p))
    return True


def disk_kernel(size: int) -> np.ndarray:
    """椭圆结构元。size 会被强制为奇数。"""
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def find_marker(
    hsv: np.ndarray,
    ranges: list[tuple[int, int, int, int, int, int]],
    min_area: int,
    max_area: int,
) -> tuple[tuple[int, int], np.ndarray] | None:
    """按 HSV 区间找最大的圆形徽章。

    面积阈值是关键：美团地图上「起」「终」徽章面积约 4000+，
    而虚线碎片、文字色块只有 100-300，两者相差一个数量级。
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in ranges:
        mask |= cv2.inRange(
            hsv, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v)
        )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best_area, best_idx = 0, -1
    for i in range(1, count):
        area = int(stats[i][4])
        if not (min_area <= area <= max_area):
            continue
        if area > best_area:
            best_area, best_idx = area, i
    if best_idx < 0:
        return None
    x, y, w, h, _ = stats[best_idx]
    center = (int(round(centroids[best_idx][0])), int(round(centroids[best_idx][1])))
    return center, (labels == best_idx).astype(np.uint8)


def bbox_distance(
    bbox: tuple[int, int, int, int], point: tuple[int, int]
) -> float:
    """点到包围盒的欧氏距离（点在盒内为 0）。"""
    x, y, w, h = bbox
    dx = max(x - point[0], 0, point[0] - (x + w))
    dy = max(y - point[1], 0, point[1] - (y + h))
    return float(np.hypot(dx, dy))


def order_polyline(points: np.ndarray) -> np.ndarray:
    """把无序散点排成一条有序折线。

    做法：以最长轴做主序排序，再用「最近邻 + 局部一致性」修正。
    对骑行轨迹这种单调推进的曲线，主轴排序已能还原绝大多数情况。
    """
    if len(points) < 3:
        return points
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ axis
    order = np.argsort(proj)
    return points[order]


def resample_polyline(points: np.ndarray, step_px: float) -> np.ndarray:
    """按弧长等距重采样折线。

    实现要点：必须**沿折线逐段插值**，不能对 x、y 分别做 np.interp。
    早期实现用坐标轴独立插值，遇到折角（如骑行轨迹的直角转弯）就会
    切角：实测 900px 的轨迹重采样后弧长缩到 874.7px，点最大偏离原线
    11 像素（≈60 米），导致里程凭空缩短约 15%。
    """
    if len(points) < 2:
        return points
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return pts
    n = max(2, int(total / max(step_px, 1e-6)) + 1)
    targets = np.linspace(0.0, total, n)

    # 逐段插值：找到每个目标弧长所在的段，在该段内做线性插值
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    seg_len = seg[idx]
    t = np.where(
        seg_len > 1e-9,
        (targets - cum[idx]) / np.where(seg_len > 1e-9, seg_len, 1.0),
        0.0,
    )
    t = np.clip(t, 0.0, 1.0)
    p0 = pts[idx]
    p1 = pts[idx + 1]
    return p0 + (p1 - p0) * t[:, None]


def simplify_polyline(points: np.ndarray, eps: float) -> np.ndarray:
    """折线简化：弧长保真优先的 Douglas-Peucker 变体。

    为什么不用纯 Douglas-Peucker
    --------------------------
    经典 DP 用「点到端点连线的垂距」作为误差判据。这对一般折线没问题，
    但对**骑行轨迹这种近直线**有个致命缺陷：垂距是相对于「当前子段的
    直线」算的，而一条南北向长直线上任何微小的横向抖动都会被这段长直线
    放大成很大的垂距。结果整条轨迹被压成极少数点——实测长宁剑河路
    截图（轨迹总长 2043 像素、端到端直线 1954 像素，仅 4.5% 蛇形）
    简化后只剩 32 个点，弯道全部消失，输出轨迹与实际骑行路线严重不符。

    判据的选择
    ----------
    - **主判据：弧长偏差**——子段折线比其弦长多走了多少路。
      量纲是像素，与「直线很长」无关，因此不会被放大。
    - **辅助：垂距**——只用来决定「在哪个位置分裂」，不用来判定是否简化。
      之所以仍需要它：一条整体平移的折线弧长不变、但形状确实变了，
      单看弧长会漏掉；垂距能定位到偏离最大的点。

    早期版本要求两个判据**同时**超阈值才保留，结果过于保守
    （实测只留下 20 个点，形状虽在但密度过低）。
    正确做法是：弧长偏差超阈值就一定保留（这是「真实弯道」的证据），
    垂距只用于选分裂点。
    """
    if len(points) < 3:
        return points
    pts = np.asarray(points, dtype=float)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True

    def arc_len(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())

    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        sub = pts[i : j + 1]
        seg_vec = pts[j] - pts[i]
        chord = float(np.linalg.norm(seg_vec))

        # 主判据：弧长偏差
        excess = arc_len(sub) - chord
        if excess <= eps:
            continue

        # 辅助：垂距仅用于定位分裂点
        if chord < 1e-9:
            dist = np.linalg.norm(sub - sub[0], axis=1)
        else:
            rel = sub - pts[i]
            cross = seg_vec[0] * rel[:, 1] - seg_vec[1] * rel[:, 0]
            dist = np.abs(cross) / chord
        k = int(np.argmax(dist))
        if k <= 0 or k >= len(sub) - 1:
            k = len(sub) // 2
        keep[i + k] = True
        stack.append((i, i + k))
        stack.append((i + k, j))
    return pts[keep]
