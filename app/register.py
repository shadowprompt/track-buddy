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
    # 只用「可靠类型」锚点（车站 / POI）算出的 RMSE，以及其数量。
    #
    # 行政区锚点的标注位置会被地图引擎挪到视野空白处，与真实几何质心
    # 可差 1-2 公里——那是**标注偏移**，不是配准误差。把它混进总 RMSE
    # 会让整体拟合质量被严重低估（实测城市级截图 RMSE 1212m，
    # 而其中三个 POI 锚点的残差只有 100-300 米）。
    rmse_reliable_m: float = 0.0
    n_reliable: int = 0
    # 备选解：低置信度时交给用户选。
    #
    # ⚠️ 为什么需要它：低缩放截图（一张图覆盖 20 公里）下，
    # 几何信号本质上是**多峰**的——底图匹配实测前 12 个候选的
    # Chamfer 分数差只有 0.1%（3.52 vs 3.52 米），但地理位置
    # 分散在 900-1600 米范围内。只输出最优解等于**替用户乱猜**，
    # 不如把几个候选摆出来让他点。
    alternatives: list["Registration"] = field(default_factory=list)

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

    这个模型有 **3 个自由度**（尺度 s + 平移 tx + 平移 ty），因此定解
    至少需要 3 个点。早期实现只用 2 个点采样（见 `_ransac` 的 min_points），
    2 点只能定出尺度、平移被迫取均值，必然留下系统性偏差——实测在城市级
    低缩放截图上直接选出了偏差 10% 的错误尺度。

    做法：先用点对距离比的中位数求出尺度（抗离群），再在全部锚点上
    最小化残差求解平移。
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
    # 用**中位数**而不是均值：中位数对个别偏斜的标注点免疫。
    # 行政区锚点（静安区/徐汇区）的标注位置可能偏离真实质心 1-2 公里，
    # 均值会被单个离群点拽偏 500 米以上，中位数则稳定得多。
    fx = scale * src[:, 0]
    fy = -scale * src[:, 1]
    tx = float(np.median(dst[:, 0] - fx))
    ty = float(np.median(dst[:, 1] - fy))
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


# --- 锚点容差表 -----------------------------------------------------------
#
# 内点容差必须用**像素**而不是绝对米数，才能跨尺度自适应：
#
# 地图标注的偏移是「渲染层面」的——标签避让会把文字挪开若干像素，
# 这个偏移量与截图的缩放级别无关。实测（城市级截图，15.2 米/像素）：
#   * 上海站   32.6 px    * 上海西站  42.1 px
#   * 桃浦绿地 61.8 px    * 长宁区    28.7 px
#   * 静安区   93.4 px    * 徐汇区    92.0 px
#
# 若用固定米数阈值：
#   * 200 米 → 城市级只够 13 像素，可靠 POI 全被判成离群，
#     只剩行政区撑场，模型被带偏；
#   * 2500 米 → 街道级又太松，混进的离群点会污染模型。
#
# 所以：容差 = max(像素上限 × 当前尺度, 米制下限)。
# 像素上限按锚点类型区分——行政区标签被挪到视野空白处，挪得更远；
# POI/车站由图标锚定真实点位，挪动很小。
_KIND_TOLERANCE: dict[str, tuple[float, float]] = {
    # kind: (像素上限, 米制下限)
    "station": (60.0, 200.0),
    "poi": (60.0, 200.0),
    "icon": (60.0, 200.0),
    "district": (110.0, 800.0),
    "road": (150.0, 800.0),
    "transit": (150.0, 800.0),
}
_DEFAULT_TOLERANCE: tuple[float, float] = (60.0, 250.0)
# 粗筛后进入留一验证精排的模型数量上限。LOO 要重拟合，代价较高，
# 只对最有希望的少数模型做即可。
_LOO_TOP_N = 40
# 粗筛阶段的容差倍数。粗筛要**宽松**，否则由几个准确 POI 拟合出的
# 正确模型会因为「行政区锚点把尺度带偏、POI 残差超标」而进不了候选；
# 真正的把关交给精排阶段的严格容差 + 留一验证。
_ROUGH_TOL_SCALE = 2.0
# 同一锚点（同一屏幕位置）的多个候选之间若相距过近，
# 说明它们指向的是同一个实体，不应被当成多个独立证据。
_SAME_ANCHOR_MERGE_M = 200.0
# 「可靠类型」锚点：位置由图标锚定，标注点与真实点只差几十米。
# 行政区（大范围对象）不在此列——它的标注位置会被地图引擎挪动。
_RELIABLE_KINDS = ("station", "poi", "icon", "road")


def _model_scale(m: np.ndarray) -> float:
    """从变换矩阵取尺度（米/像素）。"""
    sx = float(np.hypot(m[0, 0], m[0, 1]))
    sy = float(np.hypot(m[1, 0], m[1, 1]))
    return (sx + sy) / 2.0


def _tolerance_vec(kinds: list[str], scale: float) -> np.ndarray:
    """按「当前模型的尺度」算出逐候选的内点容差（米）。

    每个候选模型用自己的尺度来判定内点——这正是尺度自适应的关键：
    同一张图上，模型说 15 米/像素时容差就该是几百米，
    说 0.5 米/像素时就该是几十米。
    """
    out = np.empty(len(kinds), dtype=float)
    for i, k in enumerate(kinds):
        px_cap, m_floor = _KIND_TOLERANCE.get(k, _DEFAULT_TOLERANCE)
        out[i] = max(m_floor, px_cap * max(scale, 1e-6))
    return out


def _anchor_key(p: AnchorPair) -> tuple[int, int]:
    """屏幕锚点分组键：同一像素位置的候选属于同一个锚点。"""
    return (int(round(p.px)), int(round(p.py)))


def _group_candidates(
    pairs: list[AnchorPair], max_per_group: int = 3
) -> list[list[AnchorPair]]:
    """按屏幕锚点分组，并合并相距过近的重复候选。

    为什么必须分组：Nominatim 对同一个地名常常返回多个同名 POI
    （实测「外环高速」5 个、「上海南站」5 个、「机场联络线」3 个），
    它们的屏幕像素完全相同、地理坐标彼此相差几十到几百米。
    若把它们当成独立证据计入内点数，一个错误的、由两三个近距点
    撑起来的模型就能「刷票」压过真正的解——这正是城市级截图
    配准到错误城区的直接原因。
    """
    groups: dict[tuple[int, int], list[AnchorPair]] = {}
    for p in pairs:
        groups.setdefault(_anchor_key(p), []).append(p)

    out: list[list[AnchorPair]] = []
    for _, items in groups.items():
        items.sort(key=lambda x: -x.match_quality)
        kept: list[AnchorPair] = []
        for p in items:
            # 与已保留的候选地理距离过近 → 视为同一实体，丢弃
            if any(
                p.lon is not None and p.lat is not None
                and q.lon is not None and q.lat is not None
                and _haversine(p.lon, p.lat, q.lon, q.lat) < _SAME_ANCHOR_MERGE_M
                for q in kept
            ):
                continue
            kept.append(p)
        if kept:
            out.append(kept[:max_per_group])
    return out


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    from .geo.coords import haversine

    return haversine(lon1, lat1, lon2, lat2)


def _estimate_scale_prior(
    src: np.ndarray,
    dst: np.ndarray,
    groups: list[list[int]],
    kinds: list[str],
    rounds: int = 4,
) -> tuple[float | None, list[list[int]]]:
    """用可靠锚点的两两距离比「投票」估计尺度。

    尺度 = 地理距离 / 像素距离。只用可靠锚点（POI / 车站），
    因为它们的标注偏移小；行政区锚点的偏移方向随机，会把尺度带偏。

    但可靠锚点也可能匹配到同名异地地点，而且**同一锚点的多个候选
    会在投票里刷票**：「上海南站」的 5 个候选横跨 3 公里，
    与其它锚点两两组合会产生十几个比值，把众数簇整个带偏
    （实测尺度投票给出 13.03，真值 15.73）。

    四处修正：
      1. **每个锚点对只贡献一个比值**（该对当前代表之间的比值），
         同名候选再多也不会刷票；
      2. **代表是迭代选出来的**：先用当前尺度把每组里残差最小的
         候选选作代表，再重算比值。不能固定用「组内中位数」——
         组内第二候选会把锚点对的比值中位数拉低 3.7%，这个误差在
         远距离锚点上会被放大成 4 公里的残差；
      3. **以「锚点对的自洽性」筛锚点**：只有参与了一致性最高那簇
         锚点对的锚点才可信。用残差筛是循环的——平移一旦被污染，
         坏锚点的某个候选反而会「达标」；
      4. **迭代**：剔除后重新投票。

    实测城市级截图：「上海南站」（标注偏 4 公里）、「外环高速」
    （候选横跨 15 公里）都被剔除，尺度收敛到 15.69（真值 15.73）。
    """
    rep_lists = [
        list(idxs) for idxs in groups
        if all(kinds[i] in _RELIABLE_KINDS for i in idxs)
    ]
    if len(rep_lists) < 2:
        return None, []

    active = list(rep_lists)
    reps = [g[0] for g in active]
    scale: float | None = None
    for _ in range(rounds):
        # 用当前代表算锚点对比值
        pair_ratios: list[tuple[int, int, float]] = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dp = float(np.linalg.norm(src[reps[j]] - src[reps[i]]))
                dm = float(np.linalg.norm(dst[reps[j]] - dst[reps[i]]))
                if dp > 1e-6 and dm > 1e-6:
                    pair_ratios.append((i, j, dm / dp))
        if not pair_ratios:
            break

        arr = np.array([r[2] for r in pair_ratios])
        best_mask: np.ndarray | None = None
        for r in arr:
            mask = np.abs(arr - r) <= 0.10 * r
            if best_mask is None or int(mask.sum()) > int(best_mask.sum()):
                best_mask = mask
        if best_mask is None or int(best_mask.sum()) < 2:
            break
        scale = float(np.median(arr[best_mask]))

        # 用当前尺度换代表、剔除不可靠的锚点组
        m0, t0 = _fit_from_scale_prior(src, dst, kinds, scale, active)
        tol0 = _tolerance_vec(kinds, scale)
        res0 = _residuals(_apply(m0, t0, src), dst)
        new_reps: list[int] = []
        keep: list[int] = []
        for gi, g in enumerate(active):
            best = min(g, key=lambda i: res0[i])
            if res0[best] <= tol0[best]:
                keep.append(gi)
                new_reps.append(best)
        if len(keep) < 2:
            break
        unchanged = (keep == list(range(len(active)))) and (new_reps == reps)
        active = [active[i] for i in keep]
        reps = new_reps
        if unchanged:
            break
    return scale, active


def _fit_from_scale_prior(
    src: np.ndarray,
    dst: np.ndarray,
    kinds: list[str],
    scale: float,
    groups_subset: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """给定尺度，用**可靠锚点**的中位数残差确定平移。

    尺度投票已经给出可信的尺度（见 `_estimate_scale_prior`），
    此时不该再让行政区锚点参与尺度估计——它们的偏移方向随机。
    平移同样用可靠锚点的**中位数**残差，抗离群。

    这样得到的模型完全由可靠锚点定义，行政区只在后续精修中作为
    「附加约束」参与，而不是主导参数。
    """
    m = np.array([[scale, 0.0], [0.0, -scale]])
    fx = scale * src[:, 0]
    fy = -scale * src[:, 1]
    dx = dst[:, 0] - fx
    dy = dst[:, 1] - fy
    if groups_subset:
        # 两轮迭代选代表：
        #   第一轮：用组内**全部候选**残差的中位数定一个粗平移；
        #   第二轮：用粗平移给每个锚点组挑残差最小的那个候选，
        #           再用这些代表定精平移。
        #
        # 不能直接「用未平移的残差挑代表」——那是在拿绝对坐标比大小，
        # 同一个锚点组里地理坐标较大的候选会被误选为代表，
        # 平移随之偏掉，最后整组都掉出内点容差。
        rel_all = [i for g in groups_subset for i in g]
        if not rel_all:
            rel_all = list(range(len(src)))
        tx0 = float(np.median(dx[rel_all]))
        ty0 = float(np.median(dy[rel_all]))
        rel = []
        for g in groups_subset:
            rel.append(min(g, key=lambda i: abs(dx[i] - tx0) + abs(dy[i] - ty0)))
    else:
        rel = [i for i in range(len(src)) if kinds[i] in _RELIABLE_KINDS]
    if not rel:
        rel = list(range(len(src)))
    tx = float(np.median(dx[rel]))
    ty = float(np.median(dy[rel]))
    return m, np.array([tx, ty])


def _combination_alternatives(
    src: np.ndarray,
    dst: np.ndarray,
    groups: list[list[int]],
    gid: list[int],
    kinds: list[str],
    scale_prior: float | None,
    flat: list[AnchorPair],
    min_points: int = 3,
) -> list[Registration]:
    """生成「同一批锚点的不同候选组合」备选解。

    动机：低缩放下真正的分歧不是「用什么模型」，而是
    **「这个地名指的是哪个地方」**。Nominatim 对「上海站」返回 5 个、
    对「外环高速」返回 5 个，彼此相距几公里。选了哪个就决定了落点在哪。

    做法：固定物理模型（north_up），让**每个多候选的锚点组都选它的
    第 k 个候选**（k = 1, 2, ...），重新拟合，检查是否仍然自洽。
    仍然自洽 → 这是一个真实存在的备选位置。

    这样得到的备选与「同名地点的其它候选」直接对应，
    用户看到「候选 2」时能立刻判断它对应的是哪个同名地点。
    """
    out: list[Registration] = []
    if len(groups) < min_points:
        return out

    # 只用有多候选的组——单候选的组换不了
    multi = {gi for gi, g in enumerate(groups) if len(g) >= 2}
    if len(multi) < min_points:
        return out

    max_k = max(len(groups[gi]) for gi in multi)
    for k in range(1, max_k):
        # 每个多候选组都取第 k 个候选；单候选组保持不变
        idxs: list[int] = []
        ok = True
        for gi, g in enumerate(groups):
            if gi in multi:
                if k >= len(g):
                    ok = False
                    break
                idxs.append(g[k])
            else:
                idxs.append(g[0])
        if not ok or len(idxs) < min_points:
            continue

        sub_src = src[idxs]
        sub_dst = dst[idxs]
        if np.linalg.matrix_rank(sub_src - sub_src.mean(axis=0)) < 2:
            continue
        try:
            m, t = _fit_north_up(sub_src, sub_dst)
        except (ValueError, np.linalg.LinAlgError):
            continue

        sc = _model_scale(m)
        tol = _tolerance_vec(kinds, sc)
        res = _residuals(_apply(m, t, src), dst)
        inl = [j for j in range(len(src)) if res[j] <= tol[j]]
        reps = _anchor_representatives(gid, inl, res)
        if len(reps) < min_points:
            continue
        consistent, _med_loo = _loo_score(
            src, dst, gid, reps, res, _fit_north_up, tol,
            fixed_scale=(scale_prior if scale_prior else sc),
            kinds=kinds,
        )
        if consistent < min_points:
            continue

        inl_res = res[inl]
        rel = [j for j in inl if kinds[j] in _RELIABLE_KINDS]
        inl_set = set(inl)
        out.append(Registration(
            ok=True,
            matrix=m,
            translation=t,
            inliers=[flat[j] for j in inl],
            outliers=[flat[j] for j in range(len(flat)) if j not in inl_set],
            residuals=[float(r) for r in inl_res],
            rmse_m=(float(np.sqrt((inl_res ** 2).mean()))
                    if len(inl_res) else 0.0),
            rmse_reliable_m=(
                float(np.sqrt((res[rel] ** 2).mean()))
                if len(rel) else 0.0),
            n_reliable=len(rel),
            model="north_up",
        ))
    return out


def _anchor_representatives(
    gid: list[int], inliers: list[int], res: np.ndarray
) -> dict[int, int]:
    """每个锚点只保留残差最小的那一个内点候选。

    同一屏幕锚点可能有多个地理候选（Nominatim 对同名地点常返回多个），
    它们在模型下几乎同时「自洽」。若都计入内点，内点数会虚增，
    精修时还会给这个锚点加权，把模型往错误方向拽。
    """
    best: dict[int, int] = {}
    for k in inliers:
        g = gid[k]
        if g not in best or res[k] < res[best[g]]:
            best[g] = k
    return best


def _loo_score(
    src: np.ndarray,
    dst: np.ndarray,
    gid: list[int],
    reps: dict[int, int],
    res: np.ndarray,
    fitter,
    tol: np.ndarray,
    fixed_scale: float | None = None,
    kinds: list[str] | None = None,
) -> tuple[int, float]:
    """留一交叉验证：每个锚点单独拿掉，用其余锚点重新拟合再预测它。

    返回 (自洽锚点数, 留一残差中位数)。

    为什么必须做这一步：模型可以通过「一个偏得离谱的锚点 + 一堆
    碰巧一致的锚点」刷出更多内点，而这类点**只靠自己撑着**——
    拿掉它，其余点根本拟合不出同样的模型。留一验证能把这种点揪出来。

    实测城市级样本：「每万佛讲寺」的标注位置与其它锚点差了一个尺度
    （7.29 vs 5.3 墨卡托米/像素）。内点数看着最多（9 个），
    但留一后它自己撑不住，真正的自洽锚点只有 8 个——
    去掉它反而得到更准的模型。
    """
    keys = list(reps.keys())
    # 锚点少于 3 个时无法做留一验证（留出 1 个只剩 1 个，拟合不出模型）。
    # 此时退化为「内点锚点数 + 残差中位数」评分——
    # 两个互相一致的锚点仍足以定出一个可用模型，后面还有路网精修兜底。
    if len(reps) < 3:
        vals = [float(res[k]) for k in reps.values()]
        return len(reps), float(np.median(vals)) if vals else float("inf")
    loo: list[float] = []
    consistent = 0
    for g in keys:
        k = reps[g]
        others = [j for j in reps.values() if j != k]
        if len(others) < 2:
            loo.append(float(res[k]))
            continue
        try:
            if fixed_scale is not None and kinds is not None:
                # 固定尺度、只重估平移：尺度由可靠锚点的投票决定，
                # 留一验证不应该重新放开尺度，否则验证的就不是同一个模型。
                m2, t2 = _fit_from_scale_prior(
                    src[others], dst[others],
                    [kinds[i] for i in others], fixed_scale,
                )
            else:
                m2, t2 = fitter(src[others], dst[others])
        except (ValueError, np.linalg.LinAlgError):
            loo.append(float(res[k]))
            continue
        pred = _apply(m2, t2, src[k:k + 1])[0]
        r = float(np.linalg.norm(pred - dst[k]))
        loo.append(r)
        if r <= tol[k]:
            consistent += 1
    med = float(np.median(loo)) if loo else float("inf")
    return consistent, med


def _enumerate_models(
    src: np.ndarray,
    dst: np.ndarray,
    groups: list[list[int]],
    gid: list[int],
    kinds: list[str],
    fitter,
    min_points: int,
    max_combos: int = 60000,
    scale_prior: float | None = None,
    prior_tol: float = 0.12,
    loo_scale: float | None = None,
):
    """确定性枚举「K 个不同锚点」的组合，逐个拟合并评分。

    为什么不用随机 RANSAC：
      1. 随机采样会漏掉正确解（正确解往往只由 3 个锚点定出，
         组合占比随锚点数急剧下降）；
      2. 同一锚点的多个候选参与采样时，会产生大量「近重合三点」，
         拟合出的模型尺度由极小的距离比决定，尺度误差可达数倍；
      3. 确定性枚举每次结果一致，便于回归测试。

    评分分两级：
      * 粗筛：不同锚点的内点数越多越好，残差中位数越小越好；
      * 精排：对粗筛靠前的模型做**留一交叉验证**，看哪些锚点是真的
        互相支撑（自洽），而不是靠一个离群点刷出来的数量。
    """
    n_groups = len(groups)
    if n_groups < min_points:
        return None

    # 枚举：选 min_points 个不同组，每组取一个候选
    combos = []
    for gsel in itertools.combinations(range(n_groups), min_points):
        per_group = [groups[g] for g in gsel]
        for choices in itertools.product(*[range(len(g)) for g in per_group]):
            combos.append([per_group[i][choices[i]] for i in range(min_points)])
            if len(combos) >= max_combos:
                break
        if len(combos) >= max_combos:
            break

    rough: list[tuple] = []
    for combo in combos:
        idx = list(combo)
        sub_src = src[idx]
        sub_dst = dst[idx]
        if min_points >= 3:
            if np.linalg.matrix_rank(sub_src - sub_src.mean(axis=0)) < 2:
                continue
        else:
            # 两点只需不重合
            spread = float(
                np.linalg.norm(sub_src - sub_src.mean(axis=0), axis=1).max()
            )
            if spread < 1e-6:
                continue
        try:
            m, t = fitter(sub_src, sub_dst)
        except (ValueError, np.linalg.LinAlgError):
            continue
        sc = _model_scale(m)
        # 尺度先验：可靠锚点单独定出的尺度是可信的，
        # 偏离太多的模型直接丢弃（行政区锚点的偏移方向随机，
        # 混进采样会把尺度带偏几个百分点）。
        if scale_prior is not None:
            if not (scale_prior * (1.0 - prior_tol) <= sc
                    <= scale_prior * (1.0 + prior_tol)):
                continue
        # 每个模型用自己的尺度算内点容差（尺度自适应），粗筛阶段放宽
        tol = _tolerance_vec(kinds, sc * _ROUGH_TOL_SCALE)
        res = _residuals(_apply(m, t, src), dst)
        inliers = [k for k in range(len(src)) if res[k] <= tol[k]]
        reps = _anchor_representatives(gid, inliers, res)
        if len(reps) < min_points:
            continue
        med = float(np.median(res[inliers]))
        rough.append(((len(reps), -med), m, t, inliers, res, tol))

    if not rough:
        return None

    # 粗筛后只对前若干个模型做留一验证（LOO 比全量重拟合便宜得多）
    rough.sort(key=lambda x: x[0], reverse=True)
    best = None
    for _key, m, t, inliers, res, tol in rough[:_LOO_TOP_N]:
        # 用内点精修（按锚点去重，避免同一锚点重复加权）
        reps = _anchor_representatives(gid, inliers, res)
        rep_idx = list(reps.values())
        try:
            m2, t2 = fitter(src[rep_idx], dst[rep_idx])
            tol2 = _tolerance_vec(kinds, _model_scale(m2))
            res2 = _residuals(_apply(m2, t2, src), dst)
            inliers2 = [k for k in range(len(src)) if res2[k] <= tol2[k]]
            reps2 = _anchor_representatives(gid, inliers2, res2)
            if len(reps2) >= min_points:
                m, t, inliers, res, tol = m2, t2, inliers2, res2, tol2
                reps = reps2
        except (ValueError, np.linalg.LinAlgError):
            pass
        consistent, med_loo = _loo_score(
            src, dst, gid, reps, res, fitter, tol,
            fixed_scale=loo_scale, kinds=kinds,
        )
        final = (consistent, -med_loo)
        if best is None or final > best[0]:
            best = (final, m, t, inliers, res)
    if best is None:
        return None
    final, m, t, inliers, res = best
    return m, t, inliers, res, final[0], -final[1]


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

    保留给旧调用方；新的 `register` 走 `_enumerate_models`。
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
    """像素→墨卡托配准。

    输入的 pairs 可以包含同一地名的多个候选坐标——这里按屏幕锚点分组，
    同组内地理距离过近的候选会被合并，再由几何一致性挑选。

    模型按优先级依次尝试，优先选择自由度低、精度高的：
      1. north_up   —— 正北朝上，3 自由度（尺度 + 平移）。默认地图状态，精度最高
      2. similarity —— 允许旋转，4 自由度。适配用户旋转过截图的情况
      3. affine     —— 一般仿射，6 自由度。兜底

    内点判定用**按锚点类型 + 按模型尺度**的容差（`_KIND_TOLERANCE`）：
    容差以像素为主、米数为下限，因此同一套参数在街道级截图
    （0.5 米/像素）与城市级截图（15 米/像素）上都成立。
    """
    notes: list[str] = []
    valid = [p for p in pairs if p.lon is not None and p.lat is not None]
    if len(valid) < 2:
        return Registration(
            ok=False, notes=[f"有效锚点不足（{len(valid)} 个），至少需要 2 个"]
        )

    # --- 按屏幕锚点分组 ---
    group_pairs = _group_candidates(valid)
    flat: list[AnchorPair] = []
    groups: list[list[int]] = []
    for g in group_pairs:
        idxs = []
        for p in g:
            idxs.append(len(flat))
            flat.append(p)
        groups.append(idxs)

    src = np.array([[p.px, p.py] for p in flat], dtype=float)
    dst = np.array([lonlat_to_mercator(p.lon, p.lat) for p in flat], dtype=float)
    gid = [i for i, g in enumerate(groups) for _ in g]
    kinds = [p.kind for p in flat]

    # --- 阶段一：只用「可靠锚点」定尺度 ---
    #
    # 可靠锚点（POI / 车站）由图标锚定真实点位，标注偏移小且方向随机；
    # 行政区锚点则被地图引擎挪到视野空白处，偏移可达 1-2 公里。
    # 两类混在一起估尺度时，中位数会被行政区带偏几个百分点
    # （实测城市级截图：正确尺度 15.73 墨卡托米/像素，
    #  混入行政区后被拉成 14.5~14.7）。
    #
    # 所以先用可靠锚点单独拟合出一个尺度，再把它作为先验
    # 去约束全锚点的模型搜索。
    scale_prior, prior_groups = _estimate_scale_prior(src, dst, groups, kinds)

    models = [
        # (名称, 拟合器, 采样点数, 自由度档位)
        #
        # 采样点数必须 ≥ 模型自由度，否则是欠定/过参数化：
        #   * north_up 3 自由度（尺度 + 平移 x/y）→ 至少 3 点
        #   * similarity 4 自由度 → 3 点定解（尺度+旋转由 2 点定，平移由 3 点定）
        #   * affine 6 自由度 → **必须 4 点**。早期用 3 点采样 affine，
        #     任何 3 个点都能被拟合成一个「完美」的仿射变换，于是模型
        #     凭一两个错误锚点凑出更多内点，压过了物理上正确的 north_up。
        #
        # 档位同时决定门槛：自由度越高的模型，需要更多**不同锚点**的内点
        # 才能被采信，而且**优先采用低自由度模型**——美团地图截图默认
        # 正北朝上无旋转，north_up 是物理真值；affine 拟合出的轻微旋转
        # （实测 -1.67°）和被拉大的残差（786m vs north_up 的 200m 内）
        # 都是过参数化的产物，不是真实信息。
        # (名称, 拟合器, 采样点数, 所需的不同锚点内点数)
        #
        # `north_up_2pt` 是**兜底**：可靠锚点只剩 2 个时（实测有样本
        # 只匹配到 3 个 POI 锚点，其中 2 个自洽），3 点模型不成立，
        # 但两个一致锚点仍足以定出模型——后面还有路网精修兜底。
        # 它排在最后，且 3 点的 north_up 一旦成立就不会被它顶掉。
        ("north_up", _fit_north_up, 3, 3),
        ("similarity", _fit_similarity, 3, 4),
        ("affine", _fit_affine, 4, 5),
        ("north_up_2pt", _fit_north_up, 2, 2),
    ]

    # --- 首选路径：由尺度投票直接构造模型 ---
    #
    # 尺度投票只用可靠锚点，比任何「枚举 + 打分」都更贴近物理真相。
    # 用它构造出的模型再和枚举结果比 LOO 分数，取优者。
    best_reg: Registration | None = None
    best_key: tuple = ()
    best_n_anchor = 0
    # 所有达到门槛的候选，按评分排序。用于给用户「换个位置」的选项。
    alternatives: list[Registration] = []
    if scale_prior:
        m0, t0 = _fit_from_scale_prior(src, dst, kinds, scale_prior, prior_groups)
        tol0 = _tolerance_vec(kinds, scale_prior)
        res0 = _residuals(_apply(m0, t0, src), dst)
        inl0 = [k for k in range(len(src)) if res0[k] <= tol0[k]]
        reps0 = _anchor_representatives(gid, inl0, res0)
        if len(reps0) >= min_inliers:
            consistent0, med_loo0 = _loo_score(
                src, dst, gid, reps0, res0, _fit_north_up, tol0,
                fixed_scale=scale_prior, kinds=kinds,
            )
            key0 = (consistent0, -med_loo0)
            best_reg = Registration(
                ok=True,
                matrix=m0,
                translation=t0,
                inliers=[flat[k] for k in inl0],
                outliers=[flat[k] for k in range(len(flat)) if k not in set(inl0)],
                residuals=[float(r) for r in res0[inl0]],
                rmse_m=float(np.sqrt((res0[inl0] ** 2).mean())) if inl0 else 0.0,
                rmse_reliable_m=float(np.sqrt((res0[list(reps0.values())] ** 2).mean())),
                n_reliable=len(reps0),
                model="north_up",
            )
            best_key = key0
            best_n_anchor = len(reps0)
            alternatives.append(best_reg)

    for model_name, fitter, min_pts, required in models:
        if len(groups) < min_pts:
            continue
        result = _enumerate_models(
            src, dst, groups, gid, kinds, fitter, min_pts,
            scale_prior=scale_prior, loo_scale=scale_prior,
        )
        if result is None:
            continue
        m, t, inliers, res, consistent, med_loo = result
        n_anchor = len({gid[k] for k in inliers})
        if consistent < required:
            continue
        # 评分口径必须与「尺度先验」路径一致：都用
        # (留一自洽锚点数, -留一残差中位数)，否则两条路径无法比较。
        key = (consistent, -med_loo, -required)
        inlier_res = res[inliers]
        inlier_set = set(inliers)
        # 可靠类型内点（车站 / POI / 道路）——排除行政区这类
        # 「标注位置可能被人为挪动」的大范围对象
        rel = [k for k in inliers if flat[k].kind in _RELIABLE_KINDS]
        rmse_rel = (
            float(np.sqrt((res[rel] ** 2).mean())) if rel else 0.0
        )
        cand_reg = Registration(
            ok=True,
            matrix=m,
            translation=t,
            inliers=[flat[k] for k in inliers],
            outliers=[flat[k] for k in range(len(flat)) if k not in inlier_set],
            residuals=[float(r) for r in inlier_res],
            rmse_m=float(np.sqrt((inlier_res ** 2).mean())) if len(inlier_res) else 0.0,
            rmse_reliable_m=rmse_rel,
            n_reliable=len(rel),
            model=model_name,
        )
        alternatives.append(cand_reg)
        # 物理优先级：**列表里第一个成立且达标的模型就是最优解**。
        # 美团地图默认正北朝上，north_up 是物理真值；高自由度模型
        # 拟合出的轻微旋转与更小残差都是过参数化的产物（实测旋转角
        # 被伪造成 -1.67°），不能因为分数好看就顶掉它。
        #
        # ⚠️ 早期版本在这里直接 break（后面的模型完全不看），那是对的；
        # 但这样就收集不到备选解。现在改成「继续跑完收集备选，
        # 但最优解仍取第一个达标者」。
        if best_reg is None:
            best_reg = cand_reg
            best_key = key
            best_n_anchor = n_anchor

    if best_reg is None:
        return Registration(
            ok=False,
            notes=notes + ["配准失败：所有模型均未找到足够的内点"],
        )

    reg = best_reg
    notes.append(
        f"{reg.model} 模型: {len(reg.inliers)} 内点（{best_n_anchor} 个不同锚点，"
        f"{best_key[0]} 个通过留一验证）/ {len(reg.outliers)} 离群，"
        f"RMSE {reg.rmse_m:.1f}m"
    )
    reg.notes = notes

    # --- 备选解 ---
    #
    # ⚠️ **备选不能只来自「不同模型层」**：north_up / similarity / affine
    # 往往给出相近的解（实测街道级样本上三者相差不到 200 米），
    # 真正的分歧来自「同一批锚点的不同地理候选组合」——
    # 比如「上海站」匹配到 5 个同名地点，选了哪个就决定了落点在哪。
    #
    # 所以备选的两个来源：
    #   1. **模型层备选**（已有）：不同模型给出的解；
    #   2. **候选组合备选**（新增）：固定物理模型（north_up），
    #      强制用「每个锚点的第 k 个候选」重新拟合，
    #      得到「如果这个地名指的是另一个地方，轨迹会落在哪」。
    #
    # 判据（都要满足）：
    #   * 评分没有明显更差（留一自洽锚点数相当）；
    #   * 地理上离最优解足够远（> 300 米），否则是同一个解的抖动；
    #   * 与已收集的候选不重复。
    #
    # 为什么这是必要的：低置信度时前端要显示「换个位置」，
    # 但展示一堆相距 50 米的候选毫无意义——用户要选的是
    # **不同的街区**，不是同一个街区的微调。
    scored = sorted(
        alternatives,
        key=lambda r: (
            len(r.inliers),
            -float(np.median(r.residuals)) if r.residuals else 0.0,
        ),
        reverse=True,
    )
    scored.extend(_combination_alternatives(
        src, dst, groups, gid, kinds, scale_prior, flat))
    # 用「全部锚点像素的质心」作为比较基准点——它与具体图片无关，
    # 不需要给 register 传轨迹参数。
    _anchor_cx = float(src[:, 0].mean())
    _anchor_cy = float(src[:, 1].mean())
    _probe = np.array([_anchor_cx, _anchor_cy], dtype=float)

    def _center_of(r: Registration) -> np.ndarray:
        return r.matrix @ _probe + r.translation

    _reg_center = _center_of(reg)
    picked: list[Registration] = []
    for alt in scored:
        if alt is reg:
            continue
        try:
            sep_m = float(np.linalg.norm(_center_of(alt) - _reg_center))
        except Exception:
            sep_m = 0.0
        if sep_m < 300.0:
            continue
        # 与已收集的候选也保持 300 米以上间隔
        if any(
            float(np.linalg.norm(_center_of(alt) - _center_of(p))) < 300.0
            for p in picked
        ):
            continue
        picked.append(alt)
        if len(picked) >= 4:
            break
    reg.alternatives = picked
    if picked:
        notes.append(
            f"另有 {len(picked)} 个地理上分离的备选位置（相距 >300 米），"
            "可在前端「换个位置」中切换"
        )
    return reg

def apply_to_polyline(
    reg: Registration, polyline_px: np.ndarray
) -> list[tuple[float, float]]:
    """把像素折线转换为经纬度序列（GCJ-02）。"""
    if not reg.ok or len(polyline_px) == 0:
        return []
    return [reg.pixel_to_lonlat(float(px), float(py)) for px, py in polyline_px]
