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
from .scale import Scale


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


def _marker_component_diameter(found) -> float | None:
    """从徽章连通域面积反推等效直径，作为「内容尺度」的实测标尺。

    徽章是 UI 元素，其尺寸随单张截图的缩放级别变化，但**不随拼接张数**
    变化。用它当标尺，轨迹线宽、贴合容差、形态学核就能自动适配任意分辨率。
    """
    if not found:
        return None
    comp = found[1]
    area = float((comp > 0).sum())
    if area <= 0:
        return None
    return float(np.sqrt(4.0 * area / np.pi))


def _detect_markers_ex(
    img: np.ndarray, cfg: VisionConfig
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, list[str], Scale]:
    """检测徽章，同时构造本图的几何尺度。"""
    notes: list[str] = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    start = find_marker(
        hsv, [cfg.start_hsv_range], cfg.badge_min_area, cfg.badge_max_area
    )
    end = find_marker(
        hsv,
        [cfg.end_hsv_range, cfg.end_hsv_range2],
        cfg.badge_min_area,
        cfg.badge_max_area,
    )

    diameters = [
        d for d in (
            _marker_component_diameter(start),
            _marker_component_diameter(end),
        ) if d is not None
    ]
    # 取较大者：误检的 UI 圆点通常比真徽章小，较大值更接近真实内容尺度
    scale = Scale.from_image(img, max(diameters) if diameters else None)

    start_pt = start[0] if start else None
    end_pt = end[0] if end else None
    if start_pt is None:
        notes.append("未检测到起点徽章（绿色「起」）")
    if end_pt is None:
        notes.append("未检测到终点徽章（红色「终」）")
    if start_pt and end_pt:
        notes.append(f"起终点徽章已定位: 起{start_pt} 终{end_pt}")
    return start_pt, end_pt, notes, scale


def detect_markers(
    img: np.ndarray, cfg: VisionConfig
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, list[str]]:
    """检测绿色「起」与红色「终」徽章（公开接口，保持三元组返回）。"""
    start_pt, end_pt, notes, _ = _detect_markers_ex(img, cfg)
    return start_pt, end_pt, notes


def build_track_mask(
    img: np.ndarray, cfg: VisionConfig, scale: Scale | None = None
) -> np.ndarray:
    """构造轨迹线候选掩膜；形态学核按轨迹线宽自适应。

    两种掩膜模式（互斥，区间式优先）：

    * **区间式** ``cfg.track_hsv_range``：``cv2.inRange`` 按完整 HSV
      区间圈定，适合高德等彩色渐变轨迹。支持两种形态——单个 6 元组
      ``(h_lo, s_lo, v_lo, h_hi, s_hi, v_hi)``，或**多色带**（6 元组的
      元组，各带取并集）；后者服务于「速度渐变线」这类一条轨迹横跨
      多个色相带的来源（高德：红→橙→黄→绿）。
    * **上限式**（美团默认）：``V < track_max_value 且 S <
      track_max_saturation``，提浅荧光绿这类低饱和高亮线。
    """
    if scale is None:
        scale = Scale.from_image(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rng = getattr(cfg, "track_hsv_range", None)
    if rng is not None:
        bands = rng if rng and isinstance(rng[0], (tuple, list)) else (rng,)
        if len(bands) == 1:
            mask = cv2.inRange(
                hsv,
                np.array(bands[0][:3], np.uint8),
                np.array(bands[0][3:6], np.uint8),
            )
        else:
            parts = [
                cv2.inRange(hsv, np.array(b[:3], np.uint8), np.array(b[3:6], np.uint8))
                for b in bands
            ]
            mask = parts[0]
            for part in parts[1:]:
                mask = cv2.bitwise_or(mask, part)
    else:
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        mask = ((v < cfg.track_max_value) & (s < cfg.track_max_saturation)).astype(
            np.uint8
        )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(scale.close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_kernel(scale.open_kernel))
    return mask


def _select_with_markers(
    mask: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    count: int,
    start_pt: tuple[int, int],
    end_pt: tuple[int, int],
    scale: Scale,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """有徽章时挑轨迹连通域，允许「两段分别贴两端」时补线合并。

    ## 为什么需要补线

    拼接画布上，相邻截图之间难免有空洞或亚像素级错位，轨迹线可能在接缝处
    断成两段：一段贴起点、一段贴终点。这类断裂的**空隙是内容尺度的**
    （与单张截图的步距同量级），而**两端的跨度是画布尺度的**（随张数增长）；
    因此这里允许补缝，但上限由 `Scale.bridge_gap` 按路线跨度自适应给出，
    不使用固定像素数。

    ## 判据始终靠拓扑，不靠「谁大就选谁」

    * 单一连通域同时贴两端 → 直接返回；
    * 否则分别找「贴起点段」与「贴终点段」，两段都必须**像素级**贴住
      对应徽章，且空隙小于允许值 → 按轨迹线宽画一条短线把缝补上；
    * 都不满足 → 返回 None，底图色块（绿地/水面）照旧被拦住。

    「贴住」的容差按徽章直径缩放，接缝补线宽度按轨迹线宽缩放，
    所以任意分辨率、任意拼接张数都用同一套系数。
    """
    cands: list[tuple[float, int, tuple[int, int, int, int]]] = []
    for i in range(1, count):
        area = int(stats[i][4])
        if area < scale.min_area:
            continue
        x, y, w, h = (int(stats[i][0]), int(stats[i][1]),
                      int(stats[i][2]), int(stats[i][3]))
        bbox = (x, y, w, h)
        # 至少贴住一个徽章即可候选（绝不能再要求「两端距离之和」）
        d_near = min(bbox_distance(bbox, start_pt), bbox_distance(bbox, end_pt))
        if d_near > scale.coarse_dist:
            continue
        cands.append((d_near, i, bbox))
    if not cands:
        return None, None
    cands.sort(key=lambda t: t[0])

    measured: list[tuple[float, float, int, tuple[int, int, int, int], np.ndarray]] = []
    for _, i, bbox in cands[:12]:
        x, y, w, h = bbox
        # 只在 ROI 内取像素，避免对 6600 万像素画布反复全图扫描
        ys, xs = np.where(labels[y:y + h, x:x + w] == i)
        pts = np.column_stack([xs + x, ys + y]).astype(float)
        if len(pts) == 0:
            continue
        tree = cKDTree(pts)
        ds = float(tree.query([list(start_pt)])[0][0])
        de = float(tree.query([list(end_pt)])[0][0])
        measured.append((ds, de, i, bbox, pts))

    if not measured:
        return None, None

    touch = scale.touch_dist
    both = [m for m in measured if m[0] < touch and m[1] < touch]
    if both:
        best = min(both, key=lambda m: m[0] + m[1])
        return (labels == best[2]).astype(np.uint8), best[3]

    near_s = [m for m in measured if m[0] < touch]
    near_e = [m for m in measured if m[1] < touch]
    if not near_s or not near_e:
        return None, None
    a = min(near_s, key=lambda m: m[0])
    b = min(near_e, key=lambda m: m[1])
    if a[2] == b[2]:
        return None, None

    # 两段之间最近的一对像素，按轨迹线宽画一条短直线把接缝补上。
    # 允许空隙按路线跨度自适应（远小于整条路线的跨度才算「接缝」）。
    tree_b = cKDTree(b[4])
    dist, idx = tree_b.query(a[4])
    k = int(np.argmin(dist))
    if float(dist[k]) > scale.bridge_gap(start_pt, end_pt):
        return None, None
    pa = (int(a[4][k][0]), int(a[4][k][1]))
    pb = (int(b[4][idx[k]][0]), int(b[4][idx[k]][1]))
    comp = ((labels == a[2]) | (labels == b[2])).astype(np.uint8)
    cv2.line(comp, pa, pb, 1, scale.bridge_width)
    xa, ya, wa, ha = a[3]
    xb, yb, wb, hb = b[3]
    x0, y0 = min(xa, xb), min(ya, yb)
    x1, y1 = max(xa + wa, xb + wb), max(ya + ha, yb + hb)
    return comp, (x0, y0, x1 - x0, y1 - y0)


def _merge_track_fragments(
    labels: np.ndarray,
    stats: np.ndarray,
    count: int,
    scale: Scale,
    badge_pts: list[tuple[int, int]],
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """把被里程牌 / 拼接接缝切断的轨迹碎片按「端点邻近」串成一条链。

    ## 为什么需要（2026-11-16 高德实测）

    高德轨迹上盖着 3KM/2KM/1KM 白底胶囊里程牌，加上拼接接缝，会把
    轨迹线切成 3-5 段互不连通的碎片。美团的「单一连通域 + 两端补线」
    假设在这里失效——只取最大段会丢掉 30-40% 的路线。

    ## 策略

    1. 候选碎片：面积 ≥ ``min_area`` 且（线状 或 贴住某个徽章）；
    2. 种子：有徽章时取徽章像素级最近的碎片，否则取面积最大的线状碎片；
    3. 迭代：反复把「距当前链任一端点 ≤ ``bridge_gap``」的最近碎片
       并入，并按轨迹线宽画连接线；没有可并的即停（宁缺勿错接）。

    候选池被画像掩膜先行净化（高德橙/绿色带基本只剩轨迹与徽章），
    端点邻近 + 距离上限双保险防止把平行道路碎片错接进来。
    """
    cands: dict[int, tuple[int, int, int, int]] = {}
    for i in range(1, count):
        area = int(stats[i][4])
        if area < scale.min_area:
            continue
        x, y, w, h = (int(stats[i][0]), int(stats[i][1]),
                      int(stats[i][2]), int(stats[i][3]))
        fill = area / float(max(w * h, 1))
        near_badge = any(
            bbox_distance((x, y, w, h), bp) <= scale.coarse_dist for bp in badge_pts
        )
        # 面状团块（绿地/色块/图例条）不当碎片：既不细长也不贴徽章
        if fill > 0.42 and (w / max(h, 1)) < 2.0 and not near_badge:
            continue
        cands[i] = (x, y, w, h)
    if not cands:
        return None, None

    def pixels(i: int) -> np.ndarray:
        x, y, w, h = cands[i]
        ys, xs = np.where(labels[y:y + h, x:x + w] == i)
        return np.column_stack([xs + x, ys + y]).astype(float)

    trees = {i: cKDTree(pixels(i)) for i in cands}

    # --- 种子选择 ---
    if badge_pts:
        best_i, best_d = None, float("inf")
        for i, tree in trees.items():
            for bp in badge_pts:
                d = float(tree.query([list(bp)])[0][0])
                if d < best_d:
                    best_i, best_d = i, d
        if best_i is None or best_d > scale.touch_dist * 3.0:
            seed = max(cands, key=lambda i: int(stats[i][4]))
        else:
            seed = best_i
    else:
        seed = max(cands, key=lambda i: int(stats[i][4]))

    chain_ids = {seed}
    chain_pts = pixels(seed)
    bridges: list[tuple[np.ndarray, np.ndarray]] = []  # (链端点, 碎片点)

    def tips_of(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """链的两端：主轴投影的极值点。"""
        ctr = pts.mean(axis=0)
        q = pts - ctr
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        axis = vt[0]
        proj = q @ axis
        return pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]

    for _ in range(12):  # 碎片数有限，防意外死循环
        remaining = [i for i in cands if i not in chain_ids]
        if not remaining:
            break
        tip_a, tip_b = tips_of(chain_pts)
        tree_tips = cKDTree(np.vstack([tip_a, tip_b]))
        best = None  # (dist, cand_id, 链端点, 碎片最近点)
        for i in remaining:
            dist, idx = trees[i].query(tree_tips.data)
            k = int(np.argmin(dist))
            if best is None or dist[k] < best[0]:
                best = (float(dist[k]), i,
                        tree_tips.data[k], trees[i].data[int(idx[k])])
        if best is None:
            break
        dist, i, tip_pt, cand_pt = best
        # 允许空隙以「当前链两端跨度」为尺度自适应（两个方向等价，取一次）
        gap = scale.bridge_gap(
            (int(tip_a[0]), int(tip_a[1])), (int(tip_b[0]), int(tip_b[1]))
        )
        if dist > gap:
            break  # 剩余碎片都太远：宁缺勿错接
        bridges.append((tip_pt, cand_pt))
        chain_ids.add(i)
        chain_pts = np.vstack([chain_pts, pixels(i)])

    comp = np.zeros(labels.shape, np.uint8)
    for i in chain_ids:
        comp[labels == i] = 1
    for pa, pb in bridges:
        cv2.line(comp, (int(pa[0]), int(pa[1])),
                 (int(pb[0]), int(pb[1])), 1, scale.bridge_width)
    x0 = min(cands[i][0] for i in chain_ids)
    y0 = min(cands[i][1] for i in chain_ids)
    x1 = max(cands[i][0] + cands[i][2] for i in chain_ids)
    y1 = max(cands[i][1] + cands[i][3] for i in chain_ids)
    return comp, (x0, y0, x1 - x0, y1 - y0)


def _select_component(
    mask: np.ndarray,
    start_pt: tuple[int, int] | None,
    end_pt: tuple[int, int] | None,
    scale: Scale | None = None,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """挑选轨迹连通域。

    有双徽章时先走 `_select_with_markers`（拓扑 + 接缝补线）；失败或
    徽章不全时走 ``_merge_track_fragments``（碎片串接，应对里程牌/接缝
    把轨迹切成多段的高德式截图）。`scale` 为空时按掩膜尺寸自建
    （供独立调用 / 实验脚本使用）。
    """
    if scale is None:
        scale = Scale.from_image(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None, None
    if start_pt and end_pt:
        res = _select_with_markers(
            mask, labels, stats, count, start_pt, end_pt, scale
        )
        if res[0] is not None:
            return res
        # 双徽章场景：_select_with_markers 拒绝是有像素级测量依据的。
        # 串接可以再试，但结果必须像素级贴住**两端**才算翻案成功，
        # 否则维持拒绝语义（实测腾讯截图：误检徽章 + 底图色块正是
        # 靠「只贴一端的串接链」混进来）。
        merged = _merge_track_fragments(
            labels, stats, count, scale, [start_pt, end_pt])
        if merged[0] is not None:
            ys, xs = np.where(merged[0] > 0)
            tree = cKDTree(np.column_stack([xs, ys]).astype(float))
            touches_both = all(
                float(tree.query([list(bp)])[0][0]) < scale.touch_dist
                for bp in (start_pt, end_pt)
            )
            if touches_both:
                return merged
        return None, None
    badges = [p for p in (start_pt, end_pt) if p]
    return _merge_track_fragments(labels, stats, count, scale, badges)


def extract_track(
    img: np.ndarray, cfg: VisionConfig
) -> TrackResult:
    """提取轨迹折线（像素坐标）。"""
    notes: list[str] = []
    start_pt, end_pt, mnotes, scale = _detect_markers_ex(img, cfg)
    notes.extend(mnotes)

    mask = build_track_mask(img, cfg, scale)
    comp, bbox = _select_component(mask, start_pt, end_pt, scale)

    if comp is None:
        notes.append("未找到轨迹连通域，尝试放宽掩膜条件")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v, s = hsv[:, :, 2], hsv[:, :, 1]
        # 这里放宽的是 HSV 颜色区间（无量纲颜色量，不随分辨率变化）
        relaxed = ((v < 215) & (s < 120)).astype(np.uint8)
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, disk_kernel(scale.close_kernel))
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, disk_kernel(scale.open_kernel))
        comp, bbox = _select_component(relaxed, start_pt, end_pt, scale)
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
        if d < scale.conf_touch:
            confidence = min(0.97, confidence + 0.05)
            notes.append("轨迹连通域通过起终点拓扑校验")
        else:
            confidence *= 0.7
            notes.append(f"轨迹连通域与起终点距离较大 (d={d:.0f}px)，置信度下调")

    # 骨架化 → 提取中心线
    line = comp.copy()
    line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, disk_kernel(scale.skeleton_kernel))
    skeleton = _thin(line)
    pts = np.argwhere(skeleton > 0)
    if len(pts) == 0:
        pts = np.argwhere(comp > 0)
        skeleton = None

    # ⚠️ **形状合理性检查**
    #
    # 淡色底图里的大块绿地（公园、沿河绿带）会被「低饱和 + 中等亮度」
    # 掩膜捕获，骨架化后成为面状团块。纯形状判据区分不了它和真实轨迹：
    # 两者都可能细长、骨架填充率都很低；而拼接画布上真实轨迹的包围盒
    # 本来就接近整幅图宽，用「宽度」判据更是必然误杀。
    #
    # 因此**拓扑是主导信号**：真实轨迹一定同时贴住「起」「终」徽章，
    # 底图元素不会；`_select_component` 已用像素级精确测距验证过这一点
    # （贴合容差按徽章直径缩放）。只要两个徽章都在，被选中的连通域
    # 就必然是轨迹，不该再被形状否决。
    #
    # 形状判据只在「缺徽章、无拓扑可依」时兜底。其中「宽到离谱」用
    # 相对图宽的比例、「团块」用无量纲的填充率与长宽比，都与分辨率无关；
    # 拓扑距离则按画布对角线缩放——画布越大，两端天然相距越远。
    if bbox and bbox[2] > 0 and bbox[3] > 0:
        bw, bh = bbox[2], bbox[3]
        img_w = img.shape[1]
        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))
        ratio = long_side / short_side
        fill = len(pts) / float(bw * bh) if (bw * bh) else 0.0

        # 拓扑校验：两个徽章都在，且连通域同时贴近它们。
        # 阈值随画布对角线缩放（宁可放过，不可误杀）。
        has_both = bool(start_pt and end_pt)
        topo_ok = False
        if has_both:
            d_bb = bbox_distance(bbox, start_pt) + bbox_distance(bbox, end_pt)
            topo_ok = d_bb < scale.topo_dist
        elif start_pt or end_pt:
            # 单徽章（高德细节画布常见——细节图里没有起终点圆点，只有
            # 概览才有）：像素级贴住现有徽章即视为拓扑通过。真实轨迹
            # 一定从其终点徽章长出来，底图色块不会；否则横跨整幅拼接
            # 画布的真实轨迹会被下面的 wide 判据必然误杀。
            # ⚠️ 豁免必须同时要求形状像线（长宽比 ≥2.5）：误检徽章 +
            # 底图团块的组合（实测腾讯截图）就是靠贴住徽章混进来的。
            pts_xy = pts[:, ::-1] if len(pts) else np.zeros((0, 2))
            if len(pts_xy):
                tree_pts = cKDTree(pts_xy)
                touches = all(
                    float(tree_pts.query([list(bp)])[0][0]) < scale.touch_dist
                    for bp in (start_pt, end_pt) if bp
                )
                topo_ok = touches and ratio >= 2.5

        # 形状判据（仅在缺徽章时生效）：
        #   wide：横跨整幅图 —— 单图里的骑行轨迹不该这么宽
        #   blob：填充率高且长宽比小 —— 明显是面状色块
        wide = bw >= img_w * 0.75
        blob = fill > 0.45 and ratio < 2.5

        if not topo_ok and (wide or blob):
            reason = []
            if wide:
                reason.append(f"连通域宽 {bw}px ≈ 整幅图宽（{img_w}px）")
            if blob:
                reason.append(f"填充率 {fill:.2f}、长宽比 {ratio:.1f}（是团块不是线条）")
            if start_pt or end_pt:
                # 徽章存在但没连上 → 说清楚是哪一类问题
                notes.append(
                    "轨迹形状不合理：" + "、".join(reason) +
                    "——这个区域没有连上起终点徽章，"
                    "多半是底图上的道路/河道/绿地边界被误当成轨迹。"
                )
            else:
                notes.append(
                    "轨迹形状不合理：" + "、".join(reason) +
                    "——多半是底图色块（绿地/水面/建筑/文字）被误当成轨迹。"
                    "请确认截图里**有骑行轨迹线**（骑行详情页才有，"
                    "单纯浏览地图不会画路线）。"
                )
            return TrackResult(
                polyline=np.zeros((0, 2)),
                start_marker=start_pt,
                end_marker=end_pt,
                mask=comp,
                bbox=bbox,
                confidence=0.0,
                notes=notes,
            )

    polyline = _points_to_polyline(pts, start_pt, end_pt, skeleton)

    # 把起终点徽章接到折线两端（徽章中心到轨迹端点有一段引线）
    polyline = _attach_markers(polyline, start_pt, end_pt, scale)

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
    scale: Scale,
) -> np.ndarray:
    """把起终点徽章位置接到折线两端。

    美团的「起」「终」是水滴形图标，色块中心比真实锚点（尖端）偏离
    约 20-30 像素。这里把徽章色块中心与折线端点连线，取靠近折线的一端
    （即水滴尖端方向）作为真实锚点。

    注意：**不能直接改写折线端点**。早期实现把 pts[0] 移动到尖 tips，
    结果端点与原轨迹之间产生一条明显的直线跳跃，重采样时会把点散布在
    这条并不存在的直线上，造成可观的定位偏差。
    正确做法是**追加**徽章尖端作为新的首/末点，保留原轨迹端点不变。
    引线长度上限按徽章直径缩放，适配不同分辨率的截图。
    """
    if len(polyline) < 2:
        return polyline
    lead = scale.marker_lead
    pts = polyline.copy()
    head: list[np.ndarray] = []
    tail: list[np.ndarray] = []

    for marker, at_start in ((start_pt, True), (end_pt, False)):
        if marker is None:
            continue
        m = np.array(marker, dtype=float)
        end = pts[0] if at_start else pts[-1]
        d = float(np.linalg.norm(end - m))
        if d < 1e-6 or d > lead:
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

