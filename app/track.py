"""轨迹提取：从截图中分离出真实骑行轨迹的像素折线。

关键设计（均已在真实截图上验证）：

1. 轨迹线是「深色描边 + 浅灰填充」的双层结构，不能用暗阈值直接提取。
   改用「低饱和 + 中等亮度」掩膜：地图底色是蓝灰色（饱和度高），
   文字笔画虽暗但只有 2-3px 宽，而轨迹线宽 8-10px，
   因此闭运算补断点后、开运算去细笔画即可分离。

2. 红色虚线（规划路线 / 运营区围栏）饱和度高，被饱和度条件天然排除。
   即便混入，靠「必须连接起终点徽章」的拓扑规则也会被排除。

3. 真实轨迹线是单一连通域，因此提取后无需做路径搜索。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .config import VisionConfig
from .cv_utils import bbox_distance, disk_kernel, find_marker


@dataclass
class TrackResult:
    """轨迹提取结果（像素坐标）。"""

    polyline: np.ndarray  # (N,2) 有序折线
    start_marker: tuple[int, int] | None
    end_marker: tuple[int, int] | None
    mask: np.ndarray  # 二值掩膜
    bbox: tuple[int, int, int, int]
    confidence: float
    notes: list[str]


def detect_markers(
    img: np.ndarray, cfg: VisionConfig
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, list[str]]:
    """检测绿色「起」与红色「终」徽章。"""
    notes: list[str] = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    start = find_marker(
        hsv,
        [cfg.start_hsv_range],
        cfg.badge_min_area,
        cfg.badge_max_area,
    )
    end = find_marker(
        hsv,
        [cfg.end_hsv_range, cfg.end_hsv_range2],
        cfg.badge_min_area,
        cfg.badge_max_area,
    )

    start_pt = start[0] if start else None
    end_pt = end[0] if end else None
    if start_pt is None:
        notes.append("未检测到起点徽章（绿色「起」）")
    if end_pt is None:
        notes.append("未检测到终点徽章（红色「终」）")
    if start_pt and end_pt:
        notes.append(f"起终点徽章已定位: 起{start_pt} 终{end_pt}")
    return start_pt, end_pt, notes


def build_track_mask(img: np.ndarray, cfg: VisionConfig) -> np.ndarray:
    """构造轨迹线候选掩膜。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v < cfg.track_max_value) & (s < cfg.track_max_saturation)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(cfg.track_close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_kernel(cfg.track_open_kernel))
    return mask


def _select_component(
    mask: np.ndarray,
    start_pt: tuple[int, int] | None,
    end_pt: tuple[int, int] | None,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """挑选轨迹连通域。

    核心判据是**拓扑**：真正的轨迹一定同时贴着「起」和「终」两个徽章。

    早期版本额外要求「横向延展」（aspect > 1.6），那只是碰巧对上一张
    东西向的截图成立——遇到对角线或南北向的路线就会整条漏掉
    （实测一条 aspect≈1.03 的对角线路线被判为「未找到连通域」）。
    现在改成：先用包围盒粗筛，再对候选做**像素级精确测距**
    （连通域到两个徽章的最近像素距离），取总距离最小者。

    退化情况（未检出徽章）：取面积最大的「细长」连通域。
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None, None

    cands: list[tuple[float, int]] = []   # (粗筛距离, 标签 id)
    for i in range(1, count):
        area = int(stats[i][4])
        if area < 800:
            continue
        x, y, w, h = (int(stats[i][0]), int(stats[i][1]), int(stats[i][2]), int(stats[i][3]))
        bbox = (x, y, w, h)
        if start_pt and end_pt:
            d = bbox_distance(bbox, start_pt) + bbox_distance(bbox, end_pt)
            if d > 700:
                continue
        else:
            # 无徽章时用「细长程度」粗筛：线状连通域的填充率很低
            fill = area / float(max(w * h, 1))
            if fill > 0.42 and (w / max(h, 1)) < 2.0:
                continue
            d = -float(area)
        cands.append((d, i))

    if not cands:
        return None, None

    # 对最接近的若干候选做像素级精确测距
    cands.sort(key=lambda t: t[0])
    best = None
    best_cost = float("inf")
    best_key = None
    for d, i in cands[:12]:
        ys, xs = np.where(labels == i)
        pts = np.column_stack([xs, ys]).astype(float)
        x, y, w, h = (int(stats[i][0]), int(stats[i][1]), int(stats[i][2]), int(stats[i][3]))
        area = int(stats[i][4])
        bbox = (x, y, w, h)
        if start_pt and end_pt:
            tree = cKDTree(pts)
            ds = float(tree.query([list(start_pt)])[0][0])
            de = float(tree.query([list(end_pt)])[0][0])
            cost = ds + de
        else:
            ds = de = 0.0
            cost = -float(area)
        # 同样代价时优先面积大的
        key = (cost, -area)
        if best_key is None or key < best_key:
            best_key, best, best_cost = key, i, cost

    if best is None:
        return None, None

    x, y, w, h = (int(stats[best][0]), int(stats[best][1]), int(stats[best][2]), int(stats[best][3]))
    if start_pt and end_pt and best_cost > 260:
        # 精确测距后仍然离两个徽章都很远 —— 不是轨迹
        return None, None
    return (labels == best).astype(np.uint8), (x, y, w, h)


def extract_track(
    img: np.ndarray, cfg: VisionConfig
) -> TrackResult:
    """提取轨迹折线（像素坐标）。"""
    notes: list[str] = []
    start_pt, end_pt, mnotes = detect_markers(img, cfg)
    notes.extend(mnotes)

    mask = build_track_mask(img, cfg)
    comp, bbox = _select_component(mask, start_pt, end_pt)

    if comp is None:
        notes.append("未找到轨迹连通域，尝试放宽掩膜条件")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v, s = hsv[:, :, 2], hsv[:, :, 1]
        relaxed = ((v < 215) & (s < 120)).astype(np.uint8)
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, disk_kernel(13))
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, disk_kernel(5))
        comp, bbox = _select_component(relaxed, start_pt, end_pt)
        if comp is None:
            notes.append("轨迹提取失败：掩膜为空")
            return TrackResult(
                polyline=np.zeros((0, 2)),
                start_marker=start_pt,
                end_marker=end_pt,
                mask=np.zeros(img.shape[:2], np.uint8),
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                notes=notes,
            )
        mask = relaxed
        notes.append("已启用放宽掩膜")
        confidence = 0.35
    else:
        confidence = 0.9 if (start_pt and end_pt) else 0.55

    if start_pt and end_pt and bbox:
        d = bbox_distance(bbox, start_pt) + bbox_distance(bbox, end_pt)
        if d < 200:
            confidence = min(0.97, confidence + 0.05)
            notes.append("轨迹连通域通过起终点拓扑校验")
        else:
            confidence *= 0.7
            notes.append(f"轨迹连通域与起终点距离较大 (d={d:.0f}px)，置信度下调")

    # 骨架化 → 提取中心线
    line = comp.copy()
    line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, disk_kernel(3))
    skeleton = _thin(line)
    pts = np.argwhere(skeleton > 0)
    if len(pts) == 0:
        pts = np.argwhere(comp > 0)
        skeleton = None

    polyline = _points_to_polyline(pts, start_pt, end_pt, skeleton)

    # 把起终点徽章接到折线两端（徽章中心到轨迹端点有一段引线）
    polyline = _attach_markers(polyline, start_pt, end_pt)

    return TrackResult(
        polyline=polyline,
        start_marker=start_pt,
        end_marker=end_pt,
        mask=comp,
        bbox=bbox or (0, 0, 0, 0),
        confidence=round(confidence, 3),
        notes=notes,
    )


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化，得到单像素中心线。"""
    img = (mask > 0).astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        return img

    def nb(p2, p3, p4, p5, p6, p7, p8, p9):
        return (
            img[0:h - 2, 1:w - 1] * p2,
            img[0:h - 2, 2:w] * p3,
            img[1:h - 1, 2:w] * p4,
            img[2:h, 2:w] * p5,
            img[2:h, 1:w - 1] * p6,
            img[2:h, 0:w - 2] * p7,
            img[1:h - 1, 0:w - 2] * p8,
            img[0:h - 2, 0:w - 2] * p9,
        )

    changed = True
    it = 0
    while changed and it < 40:
        changed = False
        it += 1
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = nb(1, 1, 1, 1, 1, 1, 1, 1)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            a = np.zeros_like(b)
            for i in range(8):
                a += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (img[1:h - 1, 1:w - 1] == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if rm.any():
                img[1:h - 1, 1:w - 1][rm] = 0
                changed = True
    return img


def _points_to_polyline(
    pts: np.ndarray,
    start_pt: tuple[int, int] | None,
    end_pt: tuple[int, int] | None,
) -> np.ndarray:
    """把散点排成从起点到终点的有序折线。

    策略：以「起点→终点」方向为基准投影排序（骑行轨迹总体单调推进），
    偏离过大的点作为支路剔除。
    """
    if len(pts) == 0:
        return np.zeros((0, 2))
    pts = pts[:, ::-1].astype(float)  # (row, col) -> (x, y)

def _points_to_polyline(
    pts: np.ndarray,
    start_pt: tuple[int, int] | None,
    end_pt: tuple[int, int] | None,
    skeleton: np.ndarray | None = None,
) -> np.ndarray:
    """把骨架像素排成一条从起点到终点的有序折线。

    这里必须用「图路径搜索」而不是「按投影排序」。

    早期实现按起终点连线方向对骨架点做投影排序，对大体沿水平方向的轨迹
    会出错：轨迹中的垂直段与水平段的投影区间重叠，排序后被交错排列，
    形成锯齿状折线。实测该缺陷让轨迹长度虚增 5 倍（4848px vs 实际约
    750px），里程从约 4km 误报为 22.8km。

    正确做法：把骨架看作 8 邻接图，在图上搜索从起点徽章到终点徽章的
    最短路径。骨架本身是单像素曲线，搜索结果即真实轨迹。
    """
    if len(pts) == 0:
        return np.zeros((0, 2))

    # 优先使用骨架图搜索
    if skeleton is not None and skeleton.any():
        path = _trace_graph_path(skeleton, start_pt, end_pt)
        if path is not None and len(path) >= 2:
            return path

    # 兜底：按最长轴排序
    pts = pts[:, ::-1].astype(float)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return pts[np.argsort(centered @ axis)]


def _attach_markers(
    polyline: np.ndarray,
    start_pt: tuple[int, int] | None,
    end_pt: tuple[int, int] | None,
) -> np.ndarray:
    """把起终点徽章位置接到折线两端。

    美团的「起」「终」是水滴形图标，色块中心比真实锚点（尖端）偏离
    约 20-30 像素。这里把徽章色块中心与折线端点连线，取靠近折线的一端
    （即水滴尖端方向）作为真实锚点。

    注意：**不能直接改写折线端点**。早期实现把 pts[0] 移动到尖 tips，
    结果端点与原轨迹之间产生一条 22 像素的直线跳跃，重采样时会把点
    分布在这条直线上，最大偏离原轨迹 11 像素（≈60 米）。
    正确做法是**追加**徽章尖端作为新的首/末点，保留原轨迹端点不变。
    """
    if len(polyline) < 2:
        return polyline
    pts = polyline.copy()
    head: list[np.ndarray] = []
    tail: list[np.ndarray] = []

    for marker, at_start in ((start_pt, True), (end_pt, False)):
        if marker is None:
            continue
        m = np.array(marker, dtype=float)
        end = pts[0] if at_start else pts[-1]
        d = float(np.linalg.norm(end - m))
        if d < 1e-6 or d > 120:
            continue
        # 水滴尖端大约在色块中心与轨迹端点之间 55% 处
        tip = m + (end - m) * 0.55
        if at_start:
            head.append(tip)
        else:
            tail.append(tip)

    if head:
        pts = np.vstack([np.array(head), pts])
    if tail:
        pts = np.vstack([pts, np.array(tail)])
    return pts


def _trace_graph_path(
    skeleton: np.ndarray,
    start_pt: tuple[int, int] | None,
    end_pt: tuple[int, int] | None,
) -> np.ndarray | None:
    """在骨架 8 邻接图上搜索起点→终点的最短路径。"""
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 2:
        return None
    index: dict[tuple[int, int], int] = {}
    for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist())):
        index[(x, y)] = i
    n = len(index)

    # 构建邻接表
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    offsets = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    for (x, y), i in index.items():
        for dx, dy in offsets:
            j = index.get((x + dx, y + dy))
            if j is not None:
                w = (dx * dx + dy * dy) ** 0.5
                adj[i].append((j, w))

    # 找起点/终点对应的骨架节点
    def nearest(pt: tuple[int, int] | None) -> int | None:
        if pt is None:
            return None
        best, best_d = None, float("inf")
        for (x, y), i in index.items():
            d = (x - pt[0]) ** 2 + (y - pt[1]) ** 2
            if d < best_d:
                best_d, best = d, i
        return best

    s = nearest(start_pt)
    e = nearest(end_pt)
    if s is None or e is None:
        return None
    if s == e:
        return np.array([[xs[s], ys[s]]], dtype=float)

    # Dijkstra
    import heapq

    dist = [float("inf")] * n
    prev = [-1] * n
    dist[s] = 0.0
    heap = [(0.0, s)]
    visited = [False] * n
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if u == e:
            break
        for v, w in adj[u]:
            if not visited[v] and d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    if not visited[e] and prev[e] < 0:
        return None
    path: list[int] = []
    cur = e
    while cur != -1:
        path.append(cur)
        if cur == s:
            break
        cur = prev[cur]
    path.reverse()
    if len(path) < 2:
        return None
    return np.array([[xs[i], ys[i]] for i in path], dtype=float)

