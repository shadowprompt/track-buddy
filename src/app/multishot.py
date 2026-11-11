"""多图拼接：把沿路线滑动截取的街道级截图拼成一张大图。

背景
----
城市级截图里 11 公里骑行只占约 680 像素，底图上的轨迹线**本身就是直线**，
街道锯齿在截图阶段就已丢失 —— 位置能恢复，形状恢复不了。

街道级截图能看清形状，但一张图盖不住整条路线（11 公里要 5-8 张）。
解法：沿路线滑动截多张有重叠的图，**拼成一张大图**再走现有流程。

拼接还有第二个好处
------------------
大图上的 POI 标注铺开在数公里范围内，尺度估计的条件数远好于单张图。
实测单张街道级图的尺度会算成 16 米/像素（偏差 8 倍），
而大图上锚点分布在一个大圆周上，尺度可以被稳定地定出来。

怎么拼
------
1. 对每一对图做 SIFT 匹配 + RANSAC 拟合相似变换；
2. **剔除静止 UI 区域**（关键，见下）；
3. 把配准成功的图用并查集连成最大连通块，BFS 求出每张图到画布的变换；
4. 按 BFS 深度从小到大往画布上贴，同一像素只保留深度最小的那张 ——
   避免像素级混合产生重影。

为什么必须剔除静止 UI
---------------------
App 的状态栏、右侧按钮、底部数据卡片在每张截图里**位置完全相同**，
它们提供大量「位移为 0」的对应。而单位变换本身也是一个**合法的相似变换**，
于是 RANSAC 的「内点最多」准则会被 UI 骗过去：
只要 UI 特征数超过地图重叠特征数，选出来的就是单位变换 → 拼接完全错位。

实测（两张平移 932px 的合成图）：UI 对应 58 对，地图内点 64 个——
只差 6 个，UI 一多就翻车。

剔除办法不靠猜 App 布局，而是靠一个事实：
**静止 UI 在两张图的同一像素位置内容完全相同**，
而地图内容即使重叠，在同一像素位置也不同（那是不同的世界位置）。
所以「同位置内容相同」的区域就是 UI，直接从特征匹配里排除。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .cv_utils import disk_kernel, find_marker
from .scale import Scale, canvas_pixel_budget


@dataclass
class PairLink:
    """两张图之间的配准结果。"""

    i: int
    j: int
    matrix: np.ndarray          # 3x3，把 i 的像素坐标映射到 j 的像素坐标
    inliers: int
    good: int
    ncc: float
    ui_px: int = 0              # 被判定为静止 UI 而剔除的像素数

    @property
    def scale(self) -> float:
        return float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))

    def to_dict(self) -> dict[str, Any]:
        return {
            "i": self.i,
            "j": self.j,
            "inliers": self.inliers,
            "good": self.good,
            "ncc": round(self.ncc, 4),
            "scale": round(self.scale, 6),
            "rotation_deg": round(self.rotation_deg, 4),
            "tx": round(float(self.matrix[0, 2]), 1),
            "ty": round(float(self.matrix[1, 2]), 1),
            "ui_px": self.ui_px,
        }


@dataclass
class StitchShot:
    """一张图在画布上的落位。"""

    index: int
    depth: int
    x0: int
    y0: int
    x1: int
    y1: int
    inliers: int = 0
    ncc: float = 0.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "depth": self.depth,
            "box": [self.x0, self.y0, self.x1, self.y1],
            "inliers": self.inliers,
            "ncc": round(self.ncc, 4),
        }


@dataclass
class StitchResult:
    """拼接结果。"""

    ok: bool = False
    canvas: np.ndarray | None = None
    shots: list[StitchShot] = field(default_factory=list)
    used: list[int] = field(default_factory=list)
    unused: list[int] = field(default_factory=list)
    links: list[PairLink] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # 可选：位姿图全局平差的统计（默认关闭时为空 dict）
    pose_graph: dict[str, Any] = field(default_factory=dict)
    # 独立复核：把每张图 warp 回画布后，与画布内容逐像素比对的结果。
    #
    # 为什么需要它：SIFT + RANSAC 求出的变换**理论上**对，
    # 但用户看到的是「拼出来的大图」，有没有错位、有没有重影，
    # 光看接口返回的数字不直观。这组数字是**像素级**的客观判据：
    # 相似度高 = 拼缝对得上；相似度低 = 拼错了。
    verify: list[dict[str, Any]] = field(default_factory=list)
    # 整体复核得分（各图相似度的中位数）
    verify_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "shots": [s.to_dict() for s in self.shots],
            "used": self.used,
            "unused": self.unused,
            "links": [l.to_dict() for l in self.links],
            "notes": self.notes,
            "pose_graph": self.pose_graph,
            "verify": self.verify,
            "verify_score": round(self.verify_score, 4),
            "canvas_size": (
                [int(self.canvas.shape[1]), int(self.canvas.shape[0])]
                if self.canvas is not None
                else None
            ),
        }


# --- 内部工具 -------------------------------------------------------------


def _badge_protection_items(
    images: list[np.ndarray], indices: list[int]
) -> list[tuple[int, str, np.ndarray]]:
    """挑出需要在画布上保住的「起/终」徽章，每类只留面积最大的一枚。

    ⚠️ 为什么需要这一步（2026-11-11 实测事故）
    ------------------------------------------
    贴图时同一像素只留**先贴**的那张图。若含「起」徽章的截图恰好排在
    重叠区后面，徽章会被一张「同区域但没有徽章」的截图完整盖掉：
    7 张细节图里 4270 个绿色像素 100% 丢失，`extract_track()` 随后
    报「未检测到起点徽章」并直接放弃整个识别。徽章是轨迹提取的拓扑锚，
    不能当作普通像素任由覆盖顺序决定。

    每类只保留最大一枚，是为了与 `track.find_marker()` 的语义一致
    （它在画布上也只取最大的连通域），避免写回多枚互相矛盾的「起」。
    """
    from .config import CONFIG  # 局部导入，避免与 server 的模块级循环

    cfg = CONFIG.vision
    best: dict[str, tuple[int, np.ndarray, int]] = {}
    for idx in indices:
        hsv = cv2.cvtColor(images[idx], cv2.COLOR_BGR2HSV)
        for kind, ranges in (
            ("start", [cfg.start_hsv_range]),
            ("end", [cfg.end_hsv_range, cfg.end_hsv_range2]),
        ):
            found = find_marker(
                hsv, ranges, cfg.badge_min_area, cfg.badge_max_area
            )
            if found is None:
                continue
            comp = found[1].astype(np.uint8)
            # 徽章中央的白色「起/终」字在 HSV 掩膜上是个孔洞；填洞再外扩，
            # 才能把整枚徽章（含白字与描边）原样写回，而不是留个空心圈。
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, disk_kernel(15))
            comp = cv2.dilate(comp, disk_kernel(5))
            area = int((comp > 0).sum())
            if kind not in best or area > best[kind][2]:
                best[kind] = (idx, comp, area)
    return [(idx, kind, comp) for kind, (idx, comp, _area) in best.items()]


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _describe(img: np.ndarray, nfeatures: int = 8000):
    """提取 SIFT 特征。只做一次，多对复用。"""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(_gray(img), None)
    return kp, des


def _static_ui_mask(a: np.ndarray, b: np.ndarray, tol: int = 6) -> np.ndarray:
    """找出两张图里「同一像素位置内容相同」的区域（= 静止 UI）。

    地图内容即使重叠，在同一像素位置也是不同的世界位置；
    只有固定不动的 UI 才会逐像素相同。
    """
    ga, gb = _gray(a), _gray(b)
    h = min(ga.shape[0], gb.shape[0])
    w = min(ga.shape[1], gb.shape[1])
    diff = cv2.absdiff(ga[:h, :w], gb[:h, :w])
    same = (diff <= tol).astype(np.uint8) * 255
    # 腐蚀：容忍 JPEG 压缩噪声，同时避免把地图上的巧合相同区域算进来
    same = cv2.erode(same, np.ones((7, 7), np.uint8))
    mask = np.zeros(ga.shape, bool)
    mask[:h, :w] = same > 0
    return mask


def _match_pair(
    a: np.ndarray,
    b: np.ndarray,
    ka: tuple,
    kb: tuple,
    ratio: float = 0.75,
    ransac_px: float = 4.0,
) -> PairLink | None:
    """配准一对图。返回 None 表示配不上。"""
    kpa, da = ka
    kpb, db = kb
    if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
        return None

    # 剔除静止 UI 区域上的特征
    ui = _static_ui_mask(a, b)
    pts_a = np.array([k.pt for k in kpa], dtype=float)
    pts_b = np.array([k.pt for k in kpb], dtype=float)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None

    src = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    keep = ~ui[np.clip(src.reshape(-1, 2)[:, 1].astype(int), 0, ui.shape[0] - 1),
                np.clip(src.reshape(-1, 2)[:, 0].astype(int), 0, ui.shape[1] - 1)]
    n_ui = int((~keep).sum())
    if keep.sum() >= 8:
        src = src[keep]
        dst = dst[keep]

    # ⚠️ **必须复核位移**（实测踩过，最隐蔽的一个坑）
    #
    # 城市街道网格高度规律，底图里有大量**周期性重复结构**。
    # 实测（上海街区、两图重叠 240px）：
    #     真解 dx=-560 有 393 个匹配
    #     假解 dx=+449 有 **509** 个匹配 ← 更多！
    # RANSAC 选内点最多的，于是选了假解。
    #
    # 而且**光靠 NCC 也救不了**：用假解算出的 NCC 是 0.9987——
    # 因为 NCC 是拿「RANSAC 选出的变换」去比，数学上自洽，语义上却错。
    #
    # 正确做法：**从位移直方图里找出几个候选峰，逐个算 NCC，选最高的**。
    # 实测真解 NCC=1.0000、假解 NCC=0.1866 —— 区分度极高。
    best = _pick_best_shift(src, dst, a, b)
    if best is None:
        return None
    M, n_in, ncc = best

    m3 = np.vstack([M, [0, 0, 1]]).astype(float)
    return PairLink(
        i=0, j=0, matrix=m3, inliers=n_in, good=len(good), ncc=ncc, ui_px=n_ui
    )


def _pick_best_shift(
    src: np.ndarray,
    dst: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    max_candidates: int = 4,
) -> tuple[np.ndarray, int, float] | None:
    """从位移直方图里挑出真正解。

    ## 为什么不能只信 RANSAC

    城市街道网格规则，底图里有大量周期性重复结构。
    实测真解 393 个匹配、假解 509 个——RANSAC 会选后者。

    ## 为什么不能只信 NCC

    NCC 是拿「RANSAC 选出的变换」去算的，数学上自洽、语义上却错
    （实测假解的 NCC 也有 0.9987）。

    ## 正确做法

    位移直方图里会有几个明显的峰（每个峰是一个候选解）。
    逐个候选算 NCC，取最高的那个——
    真解是「真的重叠」，warp 后像素级吻合（NCC≈1）；
    假解是「碰巧长得像」，warp 后毫不相似（NCC≈0.19）。
    """
    d = (dst.reshape(-1, 2) - src.reshape(-1, 2))
    if len(d) < 8:
        return None

    zero_tol = Scale.from_image(a).zero_shift_tol
    span = float(d[:, 0].max() - d[:, 0].min())
    if span < 1:
        return None
    bins = max(20, int(span / 4))
    hist, edges = np.histogram(d[:, 0], bins=bins)
    bw = float(edges[1] - edges[0]) if len(edges) > 1 else 4.0
    # 取峰高最大的几个（按峰高降序）
    order = np.argsort(hist)[::-1][:max_candidates]

    best: tuple[np.ndarray, int, float] | None = None
    for i in order:
        if hist[i] < 8:
            continue
        center = float(edges[i] + bw / 2)
        # 峰附近 ±一个 bin 宽的匹配都算这个候选的支持者
        sel = np.abs(d[:, 0] - center) <= bw
        if sel.sum() < 8:
            continue
        M, inl = cv2.estimateAffinePartial2D(
            src[sel], dst[sel], method=cv2.RANSAC, ransacReprojThreshold=4.0
        )
        if M is None or inl is None:
            continue
        if not _is_valid_similarity(M):
            continue
        n_in = int(inl.sum())
        if n_in < 8:
            continue
        tx, ty = float(M[0, 2]), float(M[1, 2])
        # ⚠️ **两张不同视角的截图不可能配出 0 位移**。
        #
        # 静止 UI 图标与规则街网会让 SIFT 选出一个「单位变换」假解，
        # 且它的 NCC 可能与真解非常接近，靠相似度分不出来。若放进 link，
        # 下游判重或断链都会把真实的相邻关系弄断，整组图裂成两块。
        # 因此必须在**候选峰层面**排除，而不是事后补救。
        #
        # 容差按图短边缩放（不同分辨率的截图，配准残差量级不同）。
        # 唯一例外：两图像素级近重复（同一视角传了两次）——此时 0 位移
        # 就是真解，必须保留，否则真正的重复图反而检测不出来。
        if (
            abs(tx) < zero_tol
            and abs(ty) < zero_tol
            and not _is_near_duplicate(a, b)
        ):
            continue
        m3 = np.vstack([M, [0, 0, 1]]).astype(float)
        ncc = _ncc(a, b, m3)
        if ncc <= 0.0:
            continue
        if best is None or ncc > best[2]:
            best = (M, n_in, ncc)
    return best


def _is_near_duplicate(a: np.ndarray, b: np.ndarray, tol: float = 6.0) -> bool:
    """两张图是否「几乎完全一样」（同一视角的重复上传）。

    ## 为什么需要这个兜底

    完全相同的两张图，SIFT 特征描述子会**一一对应**，
    ratio test（要求次优距离大于最优的 1.33 倍）会把它们**全部落选**——
    于是 `_match_pair` 返回 None，重复图就检测不出来了。

    ## 判据（两条都要满足，都是尺度无关的量）

    1. 尺寸相同 + **逐像素平均差异**极小（tol=6，容忍 JPEG 压缩噪声）；
    2. **灰度结构相关** > 0.90（归一化相关系数，与分辨率无关）。

    只看第 1 条不够：淡色底图大部分面积是浅色背景，**不同区域**的两张
    截图平均像素差也可能很小，会把路线两端误判成同一视角。结构相关能
    把「均值接近但布局不同」区分开。两者都是无量纲/逐像素量，
    因此不需要随图像尺寸缩放。
    """
    if a.shape != b.shape:
        return False
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
    if float(diff) >= tol:
        return False
    ga = _gray(a).astype(np.float32).ravel()
    gb = _gray(b).astype(np.float32).ravel()
    ga = (ga - ga.mean()) / (ga.std() + 1e-6)
    gb = (gb - gb.mean()) / (gb.std() + 1e-6)
    return float((ga * gb).mean()) > 0.90


def _is_valid_similarity(m: np.ndarray) -> bool:
    """校验 2x3 相似变换矩阵是否退化。

    退化矩阵（行列式为 0、或尺度为 0）会让 warp 把图压成一条线，
    此时所有点都变成「内点」，内点数失去筛选意义。
    """
    a = np.array(m, dtype=float)
    if a.shape != (2, 3):
        return False
    lin = a[:, :2]
    det = float(np.linalg.det(lin))
    if abs(det) < 1e-6:
        return False
    r0 = float(np.hypot(lin[0, 0], lin[0, 1]))
    r1 = float(np.hypot(lin[1, 0], lin[1, 1]))
    if r0 < 1e-6 or r1 < 1e-6:
        return False
    # 两行模长应相等（相似变换保角），差异过大说明不是相似变换
    if abs(r0 - r1) / max(r0, r1) > 0.05:
        return False
    return True


def _ncc(a: np.ndarray, b: np.ndarray, m_ab: np.ndarray) -> float:
    """把 b 按 m_ab（a→b）叠回 a，算重叠区的归一化互相关。

    ⚠️ **方向**（实测踩过）：`cv2.warpAffine(src, M, dsize)` 默认把 M
    当作 **src→dst** 使用，与 `warpPerspective` 一致。

    我们要把 b（b 坐标系）搬到 a（a 坐标系）上，
    所以需要一个 **b→a** 的变换。而 m_ab 是 a→b，
    因此要传 **m_ab⁻¹**。

    实测：传 m_ab 时算出的 NCC 仍有 0.99——因为两张图重叠度高，
    错位一点也像。但**接缝处会明显对不上**，肉眼可见。
    """
    try:
        inv = np.linalg.inv(m_ab)
    except np.linalg.LinAlgError:
        return 0.0
    h, w = a.shape[:2]
    warped = cv2.warpAffine(b, inv[:2], (w, h))
    ga = _gray(a).astype(np.float32)
    gb = _gray(warped).astype(np.float32)
    mask = cv2.warpAffine(np.ones_like(ga), inv[:2], (w, h)) > 0.5
    if mask.sum() < 1000:
        return 0.0
    x = ga[mask]
    y = gb[mask]
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return float((x * y).mean())


# --- 主流程 ---------------------------------------------------------------


def stitch(
    images: list[np.ndarray],
    min_inliers: int = 10,
    min_ncc: float = 0.15,
    max_pixels: int | None = None,
    pose_graph: bool | None = None,
) -> StitchResult:
    """把一组截图拼成一张大图。

    参数
    ----
    images         各张截图（顺序无所谓，会自动找相邻关系）
    min_inliers    两图配准所需的最少内点数（特征数量，与尺度无关）
    min_ncc        配准后的重叠区相似度下限（归一化量，与尺度无关）
    max_pixels    画布像素上限。默认 None = 按**内存预算 + 输入总像素**
                  自适应推导（见 `scale.canvas_pixel_budget`），
                  而不是写死一个像素数：张数少不浪费预算，张数多也不会
                  因为超过固定上限就被误判为「画布过大」。
    pose_graph     None = 读 POSE_GRAPH_ENABLED 配置；True/False 显式覆盖。
                   开启后用全部重叠边做全局平差，分摊链式累积漂移。
    """
    res = StitchResult()
    n = len(images)
    if n == 0:
        res.notes.append("没有图片")
        return res
    if n == 1:
        res.ok = True
        res.canvas = images[0]
        res.used = [0]
        res.shots = [
            StitchShot(index=0, depth=0, x0=0, y0=0,
                       x1=images[0].shape[1], y1=images[0].shape[0])
        ]
        res.notes.append("只有一张图，无需拼接")
        return res

    # --- 1. 逐对配准 ---
    feats = [_describe(im) for im in images]
    all_links: dict[tuple[int, int], PairLink] = {}
    for i, j in itertools.combinations(range(n), 2):
        link = _match_pair(images[i], images[j], feats[i], feats[j])
        if link is None:
            # SIFT 配不上不代表不是重复图——完全相同的两张图
            # 特征描述子会一一对应，反而可能因 ratio test 全部落选。
            # 退化情形（两张几乎一样）要用**像素比对**兜底。
            if _is_near_duplicate(images[i], images[j]):
                link = PairLink(
                    i=i, j=j, matrix=np.eye(3), inliers=999,
                    good=0, ncc=1.0, ui_px=0,
                )
                link.matrix[0, 2] = 0.0
                link.matrix[1, 2] = 0.0
            else:
                continue
        link.i, link.j = i, j
        all_links[(i, j)] = link

    if not all_links:
        res.notes.append("找不到任何可用的重叠关系，无法拼接")
        return res

    # --- 1.5 识别重复图 ---
    #
    # 判重是**破坏性操作**（会删掉一张图），判据只认「内容一致」：
    # 两张图逐像素 + 灰度结构都几乎一样（`_is_near_duplicate`）。
    #
    # ⚠️ **绝不能按「配准位移 ≈ 0」判重或删 link**：单位变换假解与真实
    # 相邻关系可能共存于同一组图里，按位移处理会误删含徽章的图，
    # 或切断连接两簇图的唯一一座桥。0 位移假解已在 `_pick_best_shift`
    # 的候选峰层面排除，这里只做纯粹的重复图检测。
    dup_groups: list[list[int]] = []
    dropped: set[int] = set()
    for (i, j) in all_links:
        if not _is_near_duplicate(images[i], images[j]):
            continue
        if i not in dropped and j not in dropped:
            dropped.add(j)
            dup_groups.append([i, j])
    for g in dup_groups:
        res.notes.append(
            f"第{'、'.join(str(k + 1) for k in g)} 张是**同一视角的重复**，只保留一张"
        )
    if dropped:
        keep = [k for k in range(n) if k not in dropped]
        remap = {old: new for new, old in enumerate(keep)}
        images = [images[k] for k in keep]
        all_links = {
            (remap[i], remap[j]): link
            for (i, j), link in all_links.items()
            if i in remap and j in remap
        }
        n = len(images)
        res.used = [remap[k] for k in res.used if k in remap]

    # --- 1.6 可信度过滤 ---
    link_map: dict[tuple[int, int], PairLink] = {}
    for (i, j), link in all_links.items():
        if link.inliers < min_inliers or link.ncc < min_ncc:
            res.notes.append(
                f"图{i + 1}↔图{j + 1} 配准不可信"
                f"（内点 {link.inliers}，相似度 {link.ncc:.2f}），已断开"
            )
            continue
        link_map[(i, j)] = link
        res.links.append(link)

    if not link_map:
        res.notes.append("找不到任何可用的重叠关系，无法拼接")
        return res

    # --- 2. 连通块（并查集）---
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for (i, j) in link_map:
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    root, members = max(groups.items(), key=lambda kv: len(kv[1]))

    if len(members) < n:
        res.unused = sorted(set(range(n)) - set(members))
        res.notes.append(
            f"有 {len(res.unused)} 张图与主图无重叠，已排除"
            f"（第 {'、'.join(str(k + 1) for k in res.unused)} 张）"
        )
    res.used = sorted(members)

    # --- 3. BFS 求每张图到画布的变换 ---
    # T_i 把第 i 张图的像素坐标映射到画布坐标。
    # 对一条 link(i→j)：m 把 i 映射到 j，则
    #     T_i = T_j · m   ⇒   T_j = T_i · m⁻¹
    order: list[int] = [root]
    depth: dict[int, int] = {root: 0}
    T: dict[int, np.ndarray] = {root: np.eye(3)}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for (i, j), link in link_map.items():
            if i == cur and j not in depth:
                depth[j] = depth[cur] + 1
                T[j] = T[cur] @ np.linalg.inv(link.matrix)
                order.append(j)
                queue.append(j)
            elif j == cur and i not in depth:
                depth[i] = depth[cur] + 1
                T[i] = T[cur] @ link.matrix
                order.append(i)
                queue.append(i)

    # --- 3.5 位姿图全局平差（可选） ---
    #
    # BFS 只用生成树边，链长 10 时配准噪声会线性累积；而其余重叠边
    # （图 1 与图 4 也重叠）被完全丢弃，本来可以拿来闭环平差。
    # 这里用全部边做加权最小二乘，把漂移分摊到各图。
    # ⚠️ 只改**平移**，不动旋转/尺度，也不影响后续地理配准。
    from .config import CONFIG  # 局部导入，避免与 server 的模块级循环

    use_pg = CONFIG.pose_graph_enabled if pose_graph is None else pose_graph
    if use_pg and len(T) >= 3:
        from .posegraph import PoseEdge, optimize_translations

        ids = sorted(T)
        local = {k: idx for idx, k in enumerate(ids)}
        edges: list[PoseEdge] = []
        for (i, j), link in link_map.items():
            if i not in local or j not in local:
                continue
            # m 把 i 映射到 j：T_i = T_j · m
            # ⇒ b_j − b_i = −A_j · t_m
            A_j = T[j][:2, :2]
            t_m = np.asarray(link.matrix[:2, 2], dtype=float)
            d = -(A_j @ t_m)
            edges.append(PoseEdge(
                i=local[i], j=local[j],
                dx=float(d[0]), dy=float(d[1]),
                weight=max(1.0, float(link.inliers))
                * max(0.05, float(link.ncc)),
            ))
        init_p = np.asarray([T[k][:2, 2] for k in ids], dtype=float)
        pg = optimize_translations(len(ids), edges, init_p,
                                   root=local.get(root, 0))
        res.pose_graph = pg.to_dict()
        if pg.ok:
            for li, k in enumerate(ids):
                T[k][0, 2] += pg.deltas[li][0]
                T[k][1, 2] += pg.deltas[li][1]
            res.notes.append(
                f"位姿图全局平差：{pg.n_edges} 条重叠边，"
                f"边残差 {pg.residual_before:.1f} → {pg.residual_after:.1f}px"
                "（链式累积漂移已分摊到各图）")
        else:
            res.notes.append(f"位姿图优化跳过：{pg.reason}")

    # --- 4. 画布尺寸 ---
    corners: list[np.ndarray] = []
    for idx in res.used:
        h, w = images[idx].shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        hom = np.hstack([pts, np.ones((4, 1))])
        out = (T[idx] @ hom.T).T
        corners.append(out[:, :2])
    allc = np.vstack(corners)
    min_x = float(np.floor(allc[:, 0].min()))
    min_y = float(np.floor(allc[:, 1].min()))
    max_x = float(np.ceil(allc[:, 0].max()))
    max_y = float(np.ceil(allc[:, 1].max()))
    W = int(max_x - min_x)
    H = int(max_y - min_y)
    if W <= 0 or H <= 0:
        res.notes.append("拼出的画布尺寸非法")
        return res
    budget = max_pixels
    if budget is None:
        total_input_px = sum(im.shape[0] * im.shape[1] for im in images)
        budget = canvas_pixel_budget(total_input_px)
    if W * H > budget:
        res.notes.append(
            f"拼出的画布过大（{W}×{H}，{W * H // 10000} 万像素），"
            f"超过内存预算 {budget // 10000} 万像素，放弃"
        )
        return res

    # 徽章保护：先找出各图上的「起/终」，贴图后按各自变换强制写回。
    badge_items = _badge_protection_items(images, res.used)
    badge_indices = {item[0] for item in badge_items}

    canvas = np.full((H, W, 3), 245, np.uint8)
    filled = np.zeros((H, W), bool)
    transforms: dict[int, np.ndarray] = {}

    # --- 5. 按深度从小到大贴图，同一像素只留深度最小的 ---
    #
    # ⚠️ **warpPerspective 的矩阵方向**（实测踩过，差点没发现）
    #
    # `cv2.warpPerspective(src, M, dsize)` 默认把 M 当作 **src→dst**
    # 直接使用（实测：src 的标记在 x=25，传 T=平移200 后落在 dst 的 x=225，
    # 符合预期；传 T⁻¹ 或加 WARP_INVERSE_MAP 都会落到画布外）。
    #
    # 而 `t` 正是「原图像素 → 画布像素」的 src→dst 变换，
    # 所以**直接传 t**，不要取逆。
    #
    # 取逆的后果不是报错，而是**内容被错位地摆上去**：
    # 实测传 t⁻¹ 时，A 左上角的标记出现在画布 x=983（正确应为 476），
    # 差 507 像素——肉眼几乎看不出来，但轨迹叠加会完全对不上，
    # 用户只会看到「图拼了但轨迹不贴」。
    # 同深度时让含徽章的图先贴（能少一次写回）；跨深度的覆盖仍由 6.5 兜底。
    for idx in sorted(
        res.used,
        key=lambda k: (depth[k], 0 if k in badge_indices else 1, k),
    ):
        t = T[idx].copy()
        t[0, 2] -= min_x
        t[1, 2] -= min_y
        transforms[idx] = t.copy()
        h, w = images[idx].shape[:2]
        warped = cv2.warpPerspective(
            images[idx], t, (W, H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
        mask = cv2.warpPerspective(
            np.full((h, w), 255, np.uint8), t, (W, H),
            flags=cv2.INTER_NEAREST, borderValue=0,
        ) > 0
        sel = mask & ~filled
        canvas[sel] = warped[sel]
        filled |= mask

        box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        hom = np.hstack([box, np.ones((4, 1))])
        out = (t @ hom.T).T
        # ⚠️ 用 floor/ceil 而不是 int()：int() 是**向零截断**，
        # 坐标 -0.5 会变成 0（应为 -1）、0.5 会变成 0（应为 1），
        # 导致位置框与画布差一个像素、甚至出现负数。
        res.shots.append(
            StitchShot(
                index=idx,
                depth=depth[idx],
                x0=int(np.floor(out[:, 0].min())),
                y0=int(np.floor(out[:, 1].min())),
                x1=int(np.ceil(out[:, 0].max())),
                y1=int(np.ceil(out[:, 1].max())),
            )
        )

    # --- 6. 裁掉外侧空白 ---
    #
    # 画布尺寸是所有图的**外接矩形**，但错位 / 阶梯状排列时，
    # 外接矩形里会有大量从未被任何图覆盖的区域。
    # 那部分空白对用户毫无意义，只是让图看起来「中间少了一块」。
    #
    # 这里裁到**已覆盖区域的包围盒**——外侧空白全部去掉。
    # 注意：图**之间**的空洞仍然保留（那是几何事实，裁不掉），
    # 但填的是中性灰 245 而不是纯白，一眼能看出「这里没有图」。
    ys, xs = np.nonzero(filled)
    crop_off = (0, 0)
    if len(xs):
        cx0, cx1 = int(xs.min()), int(xs.max()) + 1
        cy0, cy1 = int(ys.min()), int(ys.max()) + 1
        if (cx0, cy0) != (0, 0) or (cx1, cy1) != (W, H):
            canvas = canvas[cy0:cy1, cx0:cx1]
            # 位置框同步平移
            for s in res.shots:
                s.x0 -= cx0
                s.x1 -= cx0
                s.y0 -= cy0
                s.y1 -= cy0
            W, H = canvas.shape[1], canvas.shape[0]
            crop_off = (cx0, cy0)
            res.notes.append(f"已裁掉外侧空白 → {W}×{H}")
        # 位置框可能略微超出画布（变换是浮点，取整会有 1 像素误差；
        # 也可能有图整体落在画布外）。钳制到画布范围内，
        # 否则前端画框时会画到画布外面。
        for s in res.shots:
            s.x0 = max(0, min(s.x0, W - 1))
            s.y0 = max(0, min(s.y0, H - 1))
            s.x1 = max(s.x0 + 1, min(s.x1, W))
            s.y1 = max(s.y0 + 1, min(s.y1, H))
        # 统计图之间的空洞比例
        gap = 1.0 - float(filled[cy0:cy1, cx0:cx1].mean())
        if gap > 0.02:
            res.notes.append(
                f"图之间有 {gap * 100:.0f}% 的空洞"
                f"（滑动步子太大或截图之间有遗漏）"
            )

    # --- 6.5 起终点徽章强制写回 ---
    #
    # 第 5 步只保证**同深度**时含徽章的图先贴；跨深度的图仍可能先占住
    # 那片像素（真实事故里就是两张 depth=2 的图互相覆盖）。这里用每张图
    # 自己的变换，把徽章连同白字、描边原样盖回裁剪后的画布。
    if badge_items:
        warped_cache: dict[int, np.ndarray] = {}
        restored = 0
        for idx, _kind, comp in badge_items:
            tt = transforms[idx].copy()
            tt[0, 2] -= crop_off[0]
            tt[1, 2] -= crop_off[1]
            wm = cv2.warpPerspective(
                comp * 255, tt, (W, H),
                flags=cv2.INTER_NEAREST, borderValue=0,
            ) > 0
            if not wm.any():
                continue
            if idx not in warped_cache:
                warped_cache[idx] = cv2.warpPerspective(
                    images[idx], tt, (W, H), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
            canvas[wm] = warped_cache[idx][wm]
            restored += 1
        if restored:
            res.notes.append(
                f"已保护 {restored} 枚起终点徽章（重叠区强制写回，"
                "避免轨迹提取丢失拓扑锚）"
            )

    res.canvas = canvas
    res.ok = True
    res.notes.append(
        f"已拼接 {len(res.used)} 张图 → {W}×{H} 画布，"
        f"基于 {len(res.links)} 对重叠关系"
    )
    # 复核要用**裁剪后**的画布，所以变换要再减去裁剪偏移
    res.verify = _verify_canvas(
        images, canvas, res.used, T,
        min_x + crop_off[0], min_y + crop_off[1],
    )
    scores = [v["similarity"] for v in res.verify if v.get("similarity") is not None]
    if scores:
        res.verify_score = float(np.median(scores))
    return res


def _verify_canvas(
    images: list[np.ndarray],
    canvas: np.ndarray,
    used: list[int],
    T: dict[int, np.ndarray],
    min_x: float,
    min_y: float,
) -> list[dict[str, Any]]:
    """独立复核：每张图按算出的变换贴回画布，与画布内容逐像素比对。

    ## 为什么必须有这一步

    SIFT + RANSAC 求出的变换**理论上**对，但用户看到的是拼出来的大图。
    「拼得对不对」需要**像素级**的客观判据，而不是接口返回的数字。

    这个复核还有一个更重要的作用：**自动抓住方向性错误**。
    实测踩过一次 `warpPerspective` 传参方向反了，
    画面「看起来」还行，但接缝处内容错位——靠肉眼很难发现，
    靠这个相似度一眼就能看出来。
    """
    out: list[dict[str, Any]] = []
    H, W = canvas.shape[:2]
    for idx in used:
        t = T[idx].copy()
        t[0, 2] -= min_x
        t[1, 2] -= min_y
        h, w = images[idx].shape[:2]
        warped = cv2.warpPerspective(
            images[idx], t, (W, H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
        mask = cv2.warpPerspective(
            np.full((h, w), 255, np.uint8), t, (W, H),
            flags=cv2.INTER_NEAREST, borderValue=0,
        ) > 0
        # 只在「这张图独占的区域」上比——重叠区可能被另一张盖住，
        # 比那里会得到偏低的分（不是错，是正常的覆盖关系）。
        own = mask & ~_covered_by_others(used, idx, T, min_x, min_y, (W, H), images)
        # 这两个是**统计样本量下限**：归一化互相关需要几百个像素才稳定，
        # 量纲是「样本数」而非「空间距离」，因此不随分辨率/画布尺度缩放。
        min_samples, min_valid = 500, 100
        if own.sum() < min_samples:
            own = mask
        if own.sum() < min_valid:
            out.append({"index": idx, "similarity": None, "pixels": 0})
            continue
        a = _gray(canvas)[own].astype(np.float32)
        b = _gray(warped)[own].astype(np.float32)
        # 归一化互相关
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        sim = float((a * b).mean())
        out.append({
            "index": idx,
            "similarity": round(sim, 4),
            "pixels": int(own.sum()),
        })
    return out


def _covered_by_others(
    used: list[int],
    idx: int,
    T: dict[int, np.ndarray],
    min_x: float,
    min_y: float,
    size: tuple[int, int],
    images: list[np.ndarray],
) -> np.ndarray:
    """其它所有图在画布上覆盖的区域（用于剔除重叠区）。"""
    W, H = size
    acc = np.zeros((H, W), bool)
    for other in used:
        if other == idx:
            continue
        t = T[other].copy()
        t[0, 2] -= min_x
        t[1, 2] -= min_y
        h, w = images[other].shape[:2]
        m = cv2.warpPerspective(
            np.full((h, w), 255, np.uint8), t, (W, H),
            flags=cv2.INTER_NEAREST, borderValue=0,
        ) > 0
        acc |= m
    return acc
