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

from collections import deque
from dataclasses import dataclass
import heapq
import itertools

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .config import VisionConfig
from .cv_utils import bbox_distance, disk_kernel, find_marker, find_marker_candidates
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
    ``found`` 可为 ``find_marker`` 的 ``(中心, 掩膜)`` 或
    ``find_marker_candidates`` 的 ``(中心, 面积)``。
    """
    if not found:
        return None
    comp = found[1]
    if isinstance(comp, np.ndarray):
        area = float((comp > 0).sum())
    else:
        area = float(comp)
    if area <= 0:
        return None
    return float(np.sqrt(4.0 * area / np.pi))


def _pick_adjacent_badge_pair(
    start_cands: list[tuple[tuple[int, int], int]],
    end_cands: list[tuple[tuple[int, int], int]],
    gap_hi: float,
) -> tuple[tuple[tuple[int, int], np.ndarray] | None,
           tuple[tuple[int, int], np.ndarray] | None] | None:
    """从蓝/红徽章候选里挑出「相邻的一对」。

    闭环赛道起终点必相邻（实测 39-47px），这是比「各自面积最大」更强的
    几何判据：长截图里 UI 蓝标签和红色指北针面积都碾压真徽章，各自最大
    会双双重定位。返回与 ``find_marker`` 同构的 ``(中心, 掩膜)`` 二元组。

    ``gap_hi`` 为允许的最大间距（内容尺度，徽章直径的 2.5 倍）；超出一律
    视为无关团块。评分为「面积和 ÷ 距离惩罚」，让近邻的真徽章对胜出。
    """
    if not start_cands or not end_cands:
        return None
    best = None
    best_score = -1.0
    for (sp, sa) in start_cands:
        for (ep, ea) in end_cands:
            gap = float(np.hypot(sp[0] - ep[0], sp[1] - ep[1]))
            if gap > gap_hi:
                continue
            # 完全重合（gap≈0）多半是同一团块被两种 HSV 区间各命中一次，
            # 不是真起终点；给一个很小的下限距离避免退化。
            if gap < 3.0:
                continue
            score = (sa + ea) / (1.0 + gap / max(gap_hi, 1.0))
            if score > best_score:
                best_score = score
                best = (sp, ep)
    if best is None:
        return None
    sp, ep = best
    # 掩膜这里用 1×1 占位即可：调用方只取 [0] 中心，直径由画像的
    # scale_badge_d 显式给定（行者）或从其它候选反推，不依赖此掩膜面积。
    one = np.ones((1, 1), dtype=np.uint8)
    return (sp, one), (ep, one)


def _detect_markers_ex(
    img: np.ndarray, cfg: VisionConfig
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, list[str], Scale]:
    """检测徽章，同时构造本图的几何尺度。"""
    notes: list[str] = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    min_fill = getattr(cfg, "badge_min_fill", 0.0)
    scale_badge_d = getattr(cfg, "scale_badge_d", None)
    start = find_marker(
        hsv, [cfg.start_hsv_range], cfg.badge_min_area, cfg.badge_max_area,
        min_fill=min_fill,
    )
    # 起终点**同色**的来源（终点检测要排除起点团块本身，否则两处都会
    # 命中较大的那个）。排除半径按图高比例（归一化尺度）。
    # ⚠️ 只在区间相同时互斥：行者闭环赛道的起点是蓝色定位点、终点是
    # 红徽章，两点只相距 ~39px——若按旧逻辑（min_fill>0 就排除），蓝色
    # 起点会把 136px 半径内的红终点一起排除掉，终点检测必然失败。
    start_pt_tmp = start[0] if start else None
    excl = None
    excl_r = 0.0
    if (min_fill > 0 and start_pt_tmp is not None
            and cfg.start_hsv_range == cfg.end_hsv_range):
        excl = start_pt_tmp
        excl_r = max(24.0, img.shape[0] * 0.12)
    end = find_marker(
        hsv,
        [cfg.end_hsv_range, cfg.end_hsv_range2],
        cfg.badge_min_area,
        cfg.badge_max_area,
        min_fill=min_fill,
        exclude_pt=excl,
        exclude_radius=excl_r,
    )
    # 闭环赛道：起终点徽章**必须成对相邻**（实测相距 39-47px，约 1 个
    # 徽章直径）。这是比「面积最大」强得多的几何约束——长截图带详情面板
    # 时，面板里的蓝色 UI 元素（(318,1117) area 2464）比真起点蓝点
    # (389,586) area 768 大得多，左上角红色指北针 (99,309) area 1329
    # 也比真「终」徽章 (391,539) area 327 大，按最大挑必然双双重定位
    # （2026-11-21 实测）。解法：枚举蓝×红候选的所有配对，取距离落在
    # [下限, 上限] 内、且面积之和最大的一对——真徽章对面积/位置双优，
    # 而「面板蓝 + 指北针红」相距 837px 直接被上限排除。
    # ⚠️ 小米等来源的起终点相距百余 px（棒棒糖走廊），相邻约束不成立，
    # 会把走廊上的红色轨迹块配成「终点」（2026-11-23 余姚样本：配到
    # (363,852)，真徽章在 (324,929)）——用 pair_loop_badges=False 关闭，
    # 回退常规「面积最大」检测。
    if (getattr(cfg, "loop_course", False)
            and getattr(cfg, "pair_loop_badges", True)):
        start_cands = find_marker_candidates(
            hsv, [cfg.start_hsv_range],
            cfg.badge_min_area, cfg.badge_max_area, min_fill=min_fill,
        )
        end_cands = find_marker_candidates(
            hsv, [cfg.end_hsv_range, cfg.end_hsv_range2],
            cfg.badge_min_area, cfg.badge_max_area, min_fill=min_fill,
        )
        # 相邻上限按内容尺度（徽章直径）取，而非图高比例：长截图图高
        # 1830 时 15% 图高=274px，会把远处的无关红蓝团块也配进来。
        gap_hi = scale_badge_d * 2.5 if scale_badge_d else img.shape[0] * 0.08
        pair = _pick_adjacent_badge_pair(start_cands, end_cands, gap_hi)
        if pair is not None:
            start, end = pair
            start_pt_tmp = start[0] if start else None
            notes.append(
                f"闭环赛道起终点成对定位: 起{start[0]} 终{end[0]}"
            )

    diameters = [
        d for d in (
            _marker_component_diameter(start),
            _marker_component_diameter(end),
        ) if d is not None
    ]
    # 取较大者：误检的 UI 圆点通常比真徽章小，较大值更接近真实内容尺度。
    # 画像可用 scale_badge_d 显式覆盖（行者：内容尺度是对罗盘箭头标定的，
    # 起点换成蓝点后自动反推会缩水，见 config.scale_badge_d 注释）。
    scale_d = scale_badge_d
    if scale_d is None:
        scale_d = max(diameters) if diameters else None
    scale = Scale.from_image(img, scale_d)

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
    # 近黑区域挖除：视频缩略图内嵌迷你地图的轨迹线与真轨迹同色系，
    # 不挖除会被碎片串接接进主链（见 VisionConfig.track_mask_exclude_dark_v）。
    _dark_v = getattr(cfg, "track_mask_exclude_dark_v", None)
    if _dark_v:
        # ⚠️ 必须 ×255 成 0/255：astype 直接得 0/1，bitwise_not(1)=254，
        # `255 & 254 = 254` 非零——挖除会静默失效（2026-11-25 实测踩坑）。
        dark = ((hsv[:, :, 2] < int(_dark_v)).astype(np.uint8)) * 255
        if dark.any():
            dark = cv2.dilate(
                dark, disk_kernel(max(3, int(scale.badge_d) // 2)))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(dark))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(scale.close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_kernel(scale.open_kernel))
    return mask


def _endpoint_fragment_mask(
    mask: np.ndarray,
    end_pt: tuple[int, int] | None,
    cfg: VisionConfig,
    badge_d: float | None = None,
) -> tuple[int, np.ndarray]:
    """检测终点徽章邻域内、与主轨迹断开的轨迹碎片。

    终点徽章 + 定位点会把「跑两遍」的折返末段切成独立小连通域。本函数从
    **主掩膜本身**分离出这些碎片（颜色已验证、已开运算去噪，不做颜色重提
    以免混入徽章红字与卫星底图黄线），返回 ``(像素数, 碎片掩膜)``。

    闭环管线（``loop_course=True``）用碎片掩膜做桥接接入；开环时上层只用
    像素数做诊断报告。
    参数来源：``cfg.endpoint_fragment_detect_rel`` = (radius_per_badge,
    min_area_per_badge_d2)，按内容尺度换算（需传入 badge_d）；未配置时
    回退绝对像素形式 ``cfg.endpoint_fragment_detect`` = (radius_px,
    min_area_px)（标定分辨率专用）。
    未配置 / 无终点时返回 ``(0, 空掩膜)``。
    """
    empty = np.zeros_like(mask)
    if end_pt is None:
        return 0, empty
    rel = getattr(cfg, "endpoint_fragment_detect_rel", None)
    spec = getattr(cfg, "endpoint_fragment_detect", None)
    if rel and badge_d:
        radius = float(rel[0]) * float(badge_d)
        min_area = float(rel[1]) * float(badge_d) ** 2
    elif spec:
        radius, min_area = spec[:2]
    else:
        return 0, empty
    h, w = mask.shape[:2]
    ex, ey = int(end_pt[0]), int(end_pt[1])
    x1, y1 = max(0, int(round(ex - radius))), max(0, int(round(ey - radius)))
    x2, y2 = min(w, int(round(ex + radius))), min(h, int(round(ey + radius)))
    if x2 <= x1 or y2 <= y1:
        return 0, empty

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return 0, empty
    main_id = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    frag = ((mask[y1:y2, x1:x2] > 0)
            & (labels[y1:y2, x1:x2] != main_id)).astype(np.uint8)
    fn, frag_labels, fs, _ = cv2.connectedComponentsWithStats(frag, 8)
    total = 0
    out = np.zeros_like(mask)
    for i in range(1, fn):
        if int(fs[i, cv2.CC_STAT_AREA]) >= min_area:
            # ⚠️ 组件像素必须从 **frag_labels** 取（frag 是 0/1 二值掩膜，
            # frag==i 对 i≥2 恒为空——实测导致碎片掩膜恒空、接入失败）。
            fy, fx = np.where(frag_labels == i)
            total += int(fs[i, cv2.CC_STAT_AREA])
            out[fy + y1, fx + x1] = 1
    return total, out


def detect_endpoint_fragment(
    mask: np.ndarray,
    end_pt: tuple[int, int] | None,
    cfg: VisionConfig,
    badge_d: float | None = None,
) -> int:
    """检测终点徽章邻域内、与主轨迹断开的轨迹碎片像素数（诊断接口）。

    闭环赛道（``loop_course=True``）下该碎片会被桥接进主轨迹，见
    ``extract_track``；开环时仅报告。详见 ``_endpoint_fragment_mask``。
    """
    total, _ = _endpoint_fragment_mask(mask, end_pt, cfg, badge_d=badge_d)
    return total


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


def _merge_line_fragments(
    mask: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    main_id: int,
    scale: Scale,
) -> tuple[np.ndarray, list[str]]:
    """把被地图文字/图标打断的线状轨迹段桥接回主连通域。

    闭环长截图里，骑行轨迹会被 POI 文字标注、比例尺、UI 图标切断成
    多个连通域。``_select_loop_component`` 只取最大域，导致右上角、
    左下角等真实轨迹段被整段丢弃（行者宁波实测丢了 3 段共 2288px，
    主域 bbox 只到 x=364/y=899，而实际绿线延伸到 x=614/y=908）。

    只接纳满足全部条件的碎片，宁可漏接不可错接：
      1. 面积 ≥ 主域 1.5%（滤掉抗锯齿小点）；
      2. 线状：长宽比 ≥1.8 或填充率 ≤0.35（底图团块填充率高、接近方形）；
      3. 到主域最近距离 ≤4×badge_d（实测最大断口 131px < 157px）；
      4. 分多次迭代，每轮重算主域，允许链式桥接（先接最近的，再接下一段）。

    返回合并后的掩膜与桥接说明列表。
    """
    from scipy.spatial import cKDTree

    comp = (labels == main_id).astype(np.uint8)
    notes: list[str] = []
    main_area = float(stats[main_id, cv2.CC_STAT_AREA])
    min_area = max(40.0, main_area * 0.015)
    max_gap = 4.0 * scale.badge_d

    n = int(labels.max())
    merged: set[int] = {main_id}
    for _ in range(12):  # 最多桥接 12 段，防异常图无限扩张
        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            break
        tree = cKDTree(np.column_stack([xs, ys]).astype(float))
        best: tuple[float, int, tuple[int, int], tuple[int, int]] | None = None
        for i in range(1, n + 1):
            if i in merged:
                continue
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x, y, w, h = (int(stats[i, k]) for k in range(4))
            ratio = max(w, h) / max(1, min(w, h))
            fill = area / max(1, w * h)
            # 真实轨迹是**细线**，填充率低（行者新图各段 fill 0.07-0.20）。
            # 底图图例/指北针/色块虽有长边，但填充率高（实测旧样本误接的
            # 图例段 area575/bbox50x33 → fill 0.35）。用 fill 作主判据，
            # 长宽比只作辅助（0.35~0.5 区间需同时足够细长才放行）。
            if fill > 0.25 and not (fill <= 0.42 and ratio >= 3.0):
                continue  # 面状团块/图例，不是细线轨迹
            fy, fx = np.where(labels == i)
            d, idx = tree.query(np.column_stack([fx, fy]).astype(float), k=1)
            j = int(np.argmin(d))
            if d[j] > max_gap:
                continue
            if best is None or d[j] < best[0]:
                best = (float(d[j]), i, (int(xs[idx[j]]), int(ys[idx[j]])),
                        (int(fx[j]), int(fy[j])))
        if best is None:
            break
        gap, seg_id, p_main, p_frag = best
        seg_area = int(stats[seg_id, cv2.CC_STAT_AREA])
        cv2.line(comp, p_main, p_frag, 1, 1)
        comp[labels == seg_id] = 1
        merged.add(seg_id)
        notes.append(
            f"闭环断裂段桥接：area={seg_area}px 的线状段已接回主轨迹"
            f"（gap {gap:.0f}px）"
        )
    return comp, notes


def _select_loop_component(
    mask: np.ndarray,
    end_pt: tuple[int, int],
    scale: Scale,
    bridge: bool = False,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, list[str]]:
    """闭环赛道：取终点徽章贴住的**最大**连通域。

    起点与终点是同一位置（相距 ~39px）时，双徽章拓扑选择
    （``_select_with_markers``）会把徽章旁的小碎片当成「同时贴住两端的
    轨迹」（实测选中 6043px 碎片区而非 13970px 主轨迹）。闭环下拓扑
    约束已由「起终点重合」语义取代，直接取终点贴住的最大连通域即可。

    ``bridge=True`` 时把被地图文字/图标打断的线状近邻段并回主域，
    修「识别轨迹比原图少一截」（见 ``_merge_line_fragments``）。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None, None, []
    main_id = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ex, ey = int(end_pt[0]), int(end_pt[1])
    r = max(8.0, scale.touch_dist)
    x1, y1 = max(0, ex - int(r)), max(0, ey - int(r))
    x2, y2 = min(mask.shape[1], ex + int(r) + 1), min(mask.shape[0], ey + int(r) + 1)
    if (labels[y1:y2, x1:x2] == main_id).any():
        comp = (labels == main_id).astype(np.uint8)
        notes: list[str] = []
        if bridge:
            comp, notes = _merge_line_fragments(mask, labels, stats, main_id, scale)
        ys, xs = np.where(comp > 0)
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        return comp, bbox, notes
    return None, None, []


def extract_track(
    img: np.ndarray, cfg: VisionConfig
) -> TrackResult:
    """提取轨迹折线（像素坐标）。"""
    notes: list[str] = []
    start_pt, end_pt, mnotes, scale = _detect_markers_ex(img, cfg)
    notes.extend(mnotes)
    loop_mode = bool(getattr(cfg, "loop_course", False)) and end_pt is not None

    mask = build_track_mask(img, cfg, scale)
    if not loop_mode:
        # 开环：终点折返段仅检测报告（强行接入会乱线/里程虚高，§24.8）
        frag_px = detect_endpoint_fragment(
            mask, end_pt, cfg, badge_d=scale.badge_d)
        if frag_px > 0:
            notes.append(
                f"检测到终点邻域折返碎片 {frag_px}px（疑似「跑两遍」的末段，"
                "被徽章/定位点遮挡而与主轨迹断开）——形状不可靠，未纳入轨迹"
            )
    if loop_mode:
        comp, bbox, bridge_notes = _select_loop_component(
            mask, end_pt, scale,
            bridge=bool(getattr(cfg, "loop_bridge_fragments", False)),
        )
        notes.extend(bridge_notes)
        if comp is not None:
            # 终点邻域碎片桥接接入：被「终」徽章+定位点遮挡的进出段
            frag_px, frag_mask = _endpoint_fragment_mask(
                mask, end_pt, cfg, badge_d=scale.badge_d)
            if frag_px > 0:
                cy_, cx_ = np.where(comp > 0)
                tree_c = cKDTree(np.column_stack([cx_, cy_]).astype(float))
                fys, fxs = np.where(frag_mask > 0)
                d, ii = tree_c.query(np.column_stack([fxs, fys]).astype(float), k=1)
                j = int(np.argmin(d))
                if d[j] < 60:
                    pm = (int(cx_[ii[j]]), int(cy_[ii[j]]))
                    pf = (int(fxs[j]), int(fys[j]))
                    cv2.line(comp, pm, pf, 1, 1)
                    comp[frag_mask > 0] = 1
                    notes.append(
                        f"闭环赛道：终点邻域折返碎片 {frag_px}px 已桥接接入"
                        f"（gap {d[j]:.0f}px）——起点=终点，此段即「跑两遍」"
                        "的进出通道"
                    )
                else:
                    notes.append(
                        f"闭环赛道：终点邻域碎片 {frag_px}px 距主轨迹过远 "
                        f"({d[j]:.0f}px)，未接入"
                    )
    else:
        comp, bbox = _select_component(mask, start_pt, end_pt, scale)

    if comp is None:
        notes.append("未找到轨迹连通域，尝试放宽掩膜条件")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v, s = hsv[:, :, 2], hsv[:, :, 1]
        # 这里放宽的是 HSV 颜色区间（无量纲颜色量，不随分辨率变化）
        relaxed = ((v < 215) & (s < 120)).astype(np.uint8)
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, disk_kernel(scale.close_kernel))
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, disk_kernel(scale.open_kernel))
        if loop_mode:
            comp, bbox, _ = _select_loop_component(relaxed, end_pt, scale)
        else:
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
    # 抹除起终点徽章/定位点团块：小米的绿定位点与红徽章颜色落在轨迹
    # 色带内，掩膜里形成 ~26px 厚团块——骨架化后生出辐条，邮路 walk
    # 在团块里原地徘徊（余姚样本底段 153px 直线走了 518px，弧长虚高
    # ~20%，连带尺度校准失准）。抹除后骨架端点直指徽章中心，缺口由
    # _reconnect_marker_blobs 按邻域像素向中心连弦补线。
    erase_r = int(getattr(cfg, "erase_marker_blobs_r", 0) or 0)
    # 分辨率归一化：抹除半径是按某一参考分辨率实测的像素值，而徽章
    # 直径（scale.badge_d）是实测内容尺度——两者同源，比值即缩放系数。
    # 2026-11-24 实测：1080×2400 = 864×1920 的 1.25× 等比图，45px 固定
    # 半径挖不净花瓣晕圈，骨架生出「绕洞的环」，walk 弧长虚高且
    # 棒棒糖重排因锚点容差超限整体失效。标定基准由
    # ``marker_erase_ref_badge_d`` 给出（= 标定那张图的实测 badge_d）。
    _erase_ref_d = getattr(cfg, "marker_erase_ref_badge_d", None)
    if erase_r > 0 and _erase_ref_d:
        erase_r = int(round(erase_r * scale.badge_d / float(_erase_ref_d)))
    blob_marks = []
    if erase_r > 0 and loop_mode:
        for bp in (start_pt, end_pt):
            if bp is not None:
                blob_marks.append(bp)
                cv2.circle(line, (int(bp[0]), int(bp[1])), erase_r, 0, -1)
    skeleton = _thin(line)
    if blob_marks:
        _reconnect_marker_blobs(skeleton, blob_marks, erase_r)
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

    polyline = None
    if loop_mode and skeleton is not None:
        polyline = _trace_loop_path(skeleton, end_pt, badge_d=scale.badge_d)
        if polyline is not None and len(polyline) >= 10:
            arc = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())
            notes.append(
                f"闭环赛道排序完成：{len(polyline)} 点，弧长 {arc:.0f}px"
                "（中国邮路：奇度配对自动把重合往返段计两遍，"
                "起点=终点，无跳步）"
            )
            # 棒棒糖路线（起点/终点在走廊上且不重合）：邮路 walk 只覆盖
            # 每条边一次，走廊不翻倍、欧拉端点落在环底结点/末端尖角而非
            # 真实起终点。按语义重排为 起点→走廊→环→走廊→起点→终点。
            if (getattr(cfg, "split_loop_delivery", False)
                    and start_pt is not None and end_pt is not None):
                reordered = _lollipop_order(
                    polyline, start_pt, end_pt, badge_d=scale.badge_d)
                if reordered is not None:
                    polyline = reordered
                    arc2 = float(np.linalg.norm(
                        np.diff(polyline, axis=0), axis=1).sum())
                    notes.append(
                        "棒棒糖路线交付排序：起点→走廊→环→走廊→起点→终点"
                        f"，{len(polyline)} 点，弧长 {arc2:.0f}px"
                        "（走廊计两遍=原路返回，端点对齐起终点徽章）"
                    )
        elif polyline is not None:
            polyline = None
    if polyline is None:
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


def _reconnect_marker_blobs(
    skeleton: np.ndarray,
    marks: list[tuple[int, int]],
    erase_r: int,
) -> None:
    """团块抹除后把骨架缺口接回徽章中心（就地修改）。

    抹除在掩膜上开了半径 ``erase_r`` 的孔：路线末端（红徽章）只从一侧
    进入，补到中心即得到干净的 deg-1 端点；路线中段（绿定位点在走廊上）
    从两侧进入，各向中心连弦形成直通。比 ``_bridge_skeleton_breaks``
    （max_geo=15px）能补更大的孔，且端点精确落在徽章中心。
    """
    ys, xs = np.where(skeleton > 0)
    if len(ys) == 0:
        return
    P = np.column_stack([xs, ys]).astype(float)
    for mx, my in marks:
        d = np.hypot(P[:, 0] - mx, P[:, 1] - my)
        ring = np.where((d >= erase_r - 1) & (d <= erase_r + 6))[0]
        if len(ring) == 0:
            continue
        # 环带像素按 8-连通性分侧（像素数少，BFS 足够）：只有中段点
        # （走廊穿过定位点）会有 ≥2 侧，末端点只有 1 侧。
        ring_set = {tuple(int(v) for v in P[i]) for i in ring}
        sides: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for seed in list(ring_set):
            if seed in seen:
                continue
            comp = [seed]
            seen.add(seed)
            dq = deque([seed])
            while dq:
                cx, cy = dq.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        q = (cx + dx, cy + dy)
                        if q in ring_set and q not in seen:
                            seen.add(q)
                            comp.append(q)
                            dq.append(q)
            j = min(
                comp,
                key=lambda t: (t[0] - mx) ** 2 + (t[1] - my) ** 2,
            )
            sides.append(j)
        for sx, sy in sides[:3]:
            cv2.line(skeleton, (sx, sy), (int(mx), int(my)), 1, 1)


def _lollipop_order(
    polyline: np.ndarray,
    start_pt: tuple[int, int],
    end_pt: tuple[int, int],
    badge_d: float | None = None,
) -> np.ndarray | None:
    """棒棒糖路线交付排序（loop_course + split_loop_delivery）。

    邮路 walk 形如 ``[尖端 → 起点 → 走廊 → 环底结点 → 环 → 环底结点]``：
    终端（walk[-1]）在途中被访问过两次（j 处）。真实跑法是
    ``起点 → 走廊 → 环 → 走廊 → 起点 → 终点``，全部片段都在 walk 里：

        walk[S:j+1]            起点→走廊→结点
        walk[j:]               结点→环→结点
        walk[S:j][::-1]        结点→走廊→起点（走廊第二遍）
        walk[0:S+1][::-1]      起点→尖端（=终点）

    条件不满足（结构不符 / 起终点不在 walk 上）返回 None 回退原 walk。

    ``badge_d``：实测内容尺度（徽章直径）。锚点/重访容差曾是固定像素
    （30/6px，按 864×1920 标定），2026-11-24 在 1.25× 等比截图上
    walk 头部离红徽章 46px > 30px，重排整体失效、弧长少 25%。改为
    badge_d 相对值：锚点容差 ≈0.95×badge_d（徽章团块抹除半径的量级），
    重访容差 ≈0.15×badge_d。
    """
    p = polyline
    n = len(p)
    if n < 20:
        return None
    if badge_d:
        tol_revisit = max(6.0, badge_d * 0.15)
        tol_anchor = max(30.0, badge_d * 0.95)
    else:
        tol_revisit, tol_anchor = 6.0, 30.0
    # walk 终端必须是途中 revisit 的点
    tail = p[-1]
    d_tail = np.hypot(p[:-1, 0] - tail[0], p[:-1, 1] - tail[1])
    j = int(np.argmin(d_tail))
    if d_tail[j] > tol_revisit or j < 10 or j > n - 10:
        return None
    d_s = np.hypot(p[:j, 0] - start_pt[0], p[:j, 1] - start_pt[1])
    S = int(np.argmin(d_s))
    if S < 5 or S > j - 5 or d_s[S] > tol_anchor:
        return None
    d_e = np.hypot(p[0, 0] - end_pt[0], p[0, 1] - end_pt[1])
    if d_e > tol_anchor:
        # 终点徽章不在 walk 头部：结构不符，回退
        return None
    return np.vstack([
        p[S:j + 1],           # 起点→走廊→环底结点
        p[j:],                # 结点→环→结点
        p[S:j][::-1],         # 结点→走廊→起点（走廊第二遍=原路返回）
        p[0:S + 1][::-1],     # 起点→尖端（红徽章=终点）
    ])


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


# ---------- 闭环赛道：骨架清理 + 中国邮路闭环追踪 ----------
#
# 背景（行者宁波 34.67km 样本，用户确认起点=终点）：
#   Zhang-Suen 细化对 45° 斜线会留下 **2px 厚的锯齿双列带**（算法已知缺陷，
#   斜向光栅化让细化无法收敛到单像素）。实测该样本骨架上有 641 个度≥3
#   「分叉」、524 个奇度节点——绝大多数是双列带边缘的伪分叉。任何在此
#   脏骨架上直接做图搜索/邮路的尝试都会里程虚高 20%-150%（HANDOVER
#   §24.8 的 6 种失败方案）。
#
# 三步清理把骨架还原成真实拓扑（实测 junc 641→19、奇度 524→28、单连通）：
#   1. ``_press_skeleton``        删除 2px 带冗余像素（逐删+即时连通重查）
#   2. ``_weld_skeleton_parts``   分量间最近点对桥接（<max_gap）
#   3. ``_bridge_skeleton_breaks`` 奇度端点「物理近距但图距远」的掩膜断口
#
# 然后 ``_trace_loop_path`` 做中国邮路闭环追踪：
#   - 奇度节点按最短路贪心配对 → 复制配对路径 → 全偶多重图。
#     **跑两遍的重合往返段**（掩膜上一条线）天然被计两遍；
#   - 自然开放端（起终点徽章遮挡处的两个端头，起点=终点）不配对，
#     欧拉路径从一个端头走到另一个端头；
#   - Hierholzer 栈式遍历输出闭合/开放 walk，像素级无跳步。

_OFFS8 = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
]
_OFFS4 = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _skel_pixels(skeleton: np.ndarray) -> set[tuple[int, int]]:
    ys, xs = np.where(skeleton > 0)
    return set(zip(xs.tolist(), ys.tolist()))


def _skel_components(S: set[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    comp_of: dict[tuple[int, int], tuple[int, int]] = {}
    for p in S:
        if p in comp_of:
            continue
        dq = deque([p])
        comp_of[p] = p
        while dq:
            cx, cy = dq.popleft()
            for dx, dy in _OFFS8:
                q = (cx + dx, cy + dy)
                if q in S and q not in comp_of:
                    comp_of[q] = p
                    dq.append(q)
    return comp_of


def _press_skeleton(skeleton: np.ndarray) -> int:
    """删除 2px 带中的冗余像素（保持连通性），返回删除像素数。

    判据：像素存在 4-邻接邻居（2px 带证据）且删除后其 8 邻域仍连通。
    **逐个删除 + 即时重查**——批量标记再统一删会把「先删者破坏后删者的
    局部连通」变成真断链（实测把单连通骨架切成 128 片）。
    """
    removed_total = 0
    for _ in range(10):
        S = _skel_pixels(skeleton)
        degs = {p: sum(1 for dx, dy in _OFFS8 if (p[0] + dx, p[1] + dy) in S) for p in S}
        cands = [
            p for p in S
            if degs[p] >= 2
            and any((p[0] + dx, p[1] + dy) in S for dx, dy in _OFFS4)
        ]
        # 度数高的像素冗余概率大，优先处理
        cands.sort(key=lambda p: -degs[p])
        removed = 0
        for (x, y) in cands:
            S_now = _skel_pixels(skeleton)
            if (x, y) not in S_now:
                continue
            nbrs = [
                (x + dx, y + dy) for dx, dy in _OFFS8
                if (x + dx, y + dy) in S_now and (dx, dy) != (0, 0)
            ]
            if len(nbrs) <= 1:
                skeleton[y, x] = 0
                removed += 1
                continue
            S2 = S_now - {(x, y)}
            seen = {nbrs[0]}
            dq = deque([nbrs[0]])
            while dq:
                cx, cy = dq.popleft()
                for dx, dy in _OFFS8:
                    q = (cx + dx, cy + dy)
                    if q in nbrs and q not in seen and q in S2:
                        seen.add(q)
                        dq.append(q)
            if len(seen) == len(nbrs):
                skeleton[y, x] = 0
                removed += 1
        removed_total += removed
        if removed == 0:
            break
    return removed_total


def _weld_skeleton_parts(skeleton: np.ndarray, max_gap: float = 20.0) -> int:
    """把不连通的骨架分量按最近点对桥接起来（gap ≤ max_gap），返回焊接数。

    压薄可能在个别瓶颈处留下小碎片（虽然逐删策略已尽量避免），行者样本
    实测焊接 0-1 处。返回后骨架应为单连通；仍有 >max_gap 的孤立分量时
    放弃（诚实失败——那多半是底图误检，不该硬连）。
    """
    welds = 0
    while True:
        S = _skel_pixels(skeleton)
        comp_of = _skel_components(S)
        roots = set(comp_of.values())
        if len(roots) <= 1:
            break
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for p, c in comp_of.items():
            if c not in groups:
                groups[c] = []
            groups[c].append(p)
        keys = sorted(groups, key=lambda k: -len(groups[k]))
        big = groups[keys[0]]
        best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
        for k in keys[1:]:
            for pb in groups[k]:
                for pa in big:
                    d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                    if best is None or d < best[0]:
                        best = (d, pa, pb)
        if best is None or best[0] > max_gap:
            break
        d, pa, pb = best
        cv2.line(skeleton, pa, pb, 1, 1)
        welds += 1
    return welds


def _bridge_skeleton_breaks(skeleton: np.ndarray, max_geo: float = 15.0) -> int:
    """桥接「物理近距但图距远」的奇度端点对（掩膜被 POI/文字切断处）。

    判据：两端点直距 ≤ max_geo，但骨架图上最短路 > 50px——说明它们物理
    上相邻、只是掩膜断开（行者样本左中部实测 gap 8px / 图距 2166px）。
    返回桥接数。桥线可能引入新的 2px 带，桥后需再压薄一轮。
    """
    bridges = 0
    while True:
        S = _skel_pixels(skeleton)
        degs = {p: sum(1 for dx, dy in _OFFS8 if (p[0] + dx, p[1] + dy) in S) for p in S}
        odd = [p for p, d in degs.items() if d % 2 == 1]
        idx = {p: i for i, p in enumerate(sorted(S))}
        adj = {
            idx[p]: [idx[(p[0] + dx, p[1] + dy)] for dx, dy in _OFFS8
                     if (p[0] + dx, p[1] + dy) in idx]
            for p in S
        }

        def graph_steps(src: int, dst: int, limit: int = 50) -> int:
            """骨架图上 src→dst 的步数；>limit 或不可达返回 limit+1。"""
            d0 = {src: 0}
            dq = deque([src])
            while dq:
                u = dq.popleft()
                if u == dst:
                    return d0[u]
                if d0[u] >= limit:
                    continue
                for v in adj[u]:
                    if v not in d0:
                        d0[v] = d0[u] + 1
                        dq.append(v)
            return limit + 1

        # 每轮只桥一对（桥线改变图，重查后再继续）
        best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
        for a, b in itertools.combinations(odd, 2):
            dgeo = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            if dgeo > max_geo:
                continue
            if graph_steps(idx[a], idx[b]) > 50:
                if best is None or dgeo < best[0]:
                    best = (dgeo, a, b)
        if best is None:
            break
        dgeo, a, b = best
        cv2.line(skeleton, a, b, 1, 1)
        _press_skeleton(skeleton)
        bridges += 1
    return bridges


def _trace_loop_path(
    skeleton: np.ndarray,
    anchor_pt: tuple[int, int],
    badge_d: float | None = None,
) -> np.ndarray | None:
    """闭环赛道专用：清理骨架 + 中国邮路追踪，返回有序折线（像素坐标）。

    ``skeleton`` 会被就地修改（压薄/焊接/桥接）。流程：
      1. 压薄 2px 带 → 焊接分量 → 桥接断口（循环至稳定）；
      2. 奇度节点贪心最短路配对（复制 ≤2.4×badge_d，标定图上 ≈100px；
         更长的对视为自然开放端，即被起终点徽章遮挡的「起点=终点」
         衔接处）；
      3. 复制路径入多重图，Hierholzer 栈式遍历（起点取开放端或锚点）；
      4. 输出像素级 walk（每步 1-1.4px，无跳步，首尾=起终点）。
    失败（骨架为空 / 不连通且 gap 超限）返回 None，上层回退开环排序。
    """
    _press_skeleton(skeleton)
    _weld_skeleton_parts(skeleton)
    _bridge_skeleton_breaks(skeleton)

    S = _skel_pixels(skeleton)
    if len(S) < 10:
        return None
    comp_of = _skel_components(S)
    if len(set(comp_of.values())) > 1:
        return None

    S_list = sorted(S)
    idx = {p: i for i, p in enumerate(S_list)}
    n = len(S_list)
    adj = {
        idx[p]: [
            (idx[(p[0] + dx, p[1] + dy)], (dx * dx + dy * dy) ** 0.5)
            for dx, dy in _OFFS8 if (p[0] + dx, p[1] + dy) in idx
        ]
        for p in S_list
    }

    def dij(src: int) -> dict[int, float]:
        d = {src: 0.0}
        h = [(0.0, src)]
        while h:
            dd, u = heapq.heappop(h)
            if dd > d.get(u, float("inf")):
                continue
            for v, wt in adj[u]:
                if dd + wt < d.get(v, float("inf")):
                    d[v] = dd + wt
                    heapq.heappush(h, (dd + wt, v))
        return d

    def path_px(s: int, t: int) -> list[int]:
        d = {s: 0.0}
        prev: dict[int, int] = {}
        h = [(0.0, s)]
        while h:
            dd, u = heapq.heappop(h)
            if dd > d.get(u, float("inf")):
                continue
            for v, wt in adj[u]:
                if dd + wt < d.get(v, float("inf")):
                    d[v] = dd + wt
                    prev[v] = u
                    heapq.heappush(h, (dd + wt, v))
        p, cur = [], t
        while cur in prev:
            p.append(cur)
            cur = prev[cur]
        p.append(s)
        return p[::-1]

    degs = {i: len(adj[i]) for i in range(n)}
    oddset = {i for i in range(n) if degs[i] % 2 == 1}

    # 奇度配对：全局贪心最近对，成本 ≤ 配对上限；超出的一对作为自然
    # 开放端（起终点徽章遮挡处——起点=终点语义下 walk 从一端头走到
    # 另一端头）。上限曾是固定 100px（864×1920 标定），2026-11-24 在
    # 1.25× 等比截图上几何距离同比例放大，固定上限会相对收紧、把本该
    # 桥接的真断口误判为开放端。改为 badge_d 相对值：2.4×badge_d 在
    # 标定图上 ≈100px（行为不变），其余分辨率随内容尺度缩放。
    pair_cap = max(100.0, badge_d * 2.4) if badge_d else 100.0
    pairs: list[tuple[int, int]] = []
    open_ends: list[int] = []
    cand: list[tuple[float, int, int]] = []
    for s in sorted(oddset):
        D = dij(s)
        for t in sorted(oddset):
            if t > s:
                c = D.get(t, float("inf"))
                if c < float("inf"):
                    cand.append((c, s, t))
    cand.sort()
    used: set[int] = set()
    for c, s, t in cand:
        if s in used or t in used:
            continue
        if c > pair_cap and len(open_ends) < 2:
            open_ends = [s, t]
            used |= {s, t}
            continue
        used |= {s, t}
        pairs.append((s, t))
    if len(oddset) - len(used) > 0:
        # 孤立奇度点（不连通）——放弃闭环，回退开环
        return None

    # 复制配对路径入多重图
    cnt: dict[tuple[int, int], int] = {}
    for u in range(n):
        for v, _ in adj[u]:
            e = (min(u, v), max(u, v))
            cnt[e] = cnt.get(e, 0) + 1
    # 每条无向边计了 2 次（两个方向）——修正为 1 次
    for e in cnt:
        cnt[e] //= 2
    for s, t in pairs:
        pp = path_px(s, t)
        for a, b in zip(pp, pp[1:]):
            e = (min(a, b), max(a, b))
            cnt[e] = cnt.get(e, 0) + 1

    nad: dict[int, list[int]] = {}
    for (a, b), c in cnt.items():
        nad.setdefault(a, []).extend([b] * c)
        nad.setdefault(b, []).extend([a] * c)

    if open_ends:
        start = open_ends[0]
    else:
        ax, ay = anchor_pt
        start = min(range(n), key=lambda i: (S_list[i][0] - ax) ** 2 + (S_list[i][1] - ay) ** 2)

    walk: list[int] = []
    stack = [start]
    while stack:
        u = stack[-1]
        nxt = None
        for v in nad.get(u, []):
            e = (min(u, v), max(u, v))
            if cnt[e] > 0:
                nxt = v
                break
        if nxt is None:
            walk.append(u)
            stack.pop()
        else:
            cnt[(min(u, nxt), max(u, nxt))] -= 1
            stack.append(nxt)
    if len(walk) < 10:
        return None
    walk.reverse()
    poly = np.array([[S_list[i][0], S_list[i][1]] for i in walk], dtype=float)
    steps = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    # 覆盖校验：Hierholzer 正确时应走完全部边（cnt 归零）；剩余边 >2%
    # 说明遍历被中断（图有不可达部分），放弃闭环回退开环。
    left = sum(1 for e in cnt if cnt[e] > 0)
    if left > len(cnt) * 0.02:
        return None
    if steps.max() > 3.0:
        return None
    return poly


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
    if s is None:
        return None

    # Dijkstra：从起点跑**完整**最短路树（不提前在终点 break），
    # 这样既拿到 s→e 路径，也知道整棵树的覆盖情况。
    import heapq

    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    dist[s] = 0.0
    heap = [(0.0, s)]
    visited = [False] * n
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, w in adj[u]:
            if not visited[v] and d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    def trace_back(target: int) -> np.ndarray | None:
        path: list[int] = []
        cur = target
        while cur != -1:
            path.append(cur)
            if cur == s:
                break
            cur = prev[cur]
        path.reverse()
        if len(path) < 2 or path[0] != s:
            return None
        return np.array([[xs[i], ys[i]] for i in path], dtype=float)

    if e is None or dist[e] == INF:
        # 终点徽章接不上骨架（引线断开）——只能从起点出发取最长路径。
        e = None

    if e is not None:
        direct = trace_back(e)
        # 环形/自交检测：若 s→e 最短路只覆盖了骨架的一小部分，说明它在
        # 环上抄了近道、漏掉另一侧的大段轨迹（实测行者宁波 34.67km：
        # 最短路 976 点、骨架 2273 点，左半环整段漏掉）。此时改取「从起点
        # 出发可达的最远点」回溯——对环形/折返轨迹，最远点天然给出绕行
        # 全程；对普通单调轨迹，最远点仍是正确的终点，行为不变。
        reachable = [i for i in range(n) if dist[i] < INF]
        cover = len(direct) / max(len(reachable), 1) if direct is not None else 0.0
        if direct is not None and cover >= 0.7:
            return direct

    # 从起点可达的最远点（环形全程 / 无终点时的兜底）
    reachable = [i for i in range(n) if dist[i] < INF]
    if len(reachable) < 2:
        return None
    far = max(reachable, key=lambda i: dist[i])
    return trace_back(far)

