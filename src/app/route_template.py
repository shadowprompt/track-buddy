"""轨迹模板：把一次成功识别的「像素折线 ↔ 地理折线」沉淀下来。

与地名词典的区别
----------------
``Lexicon`` 存的是「地名 → 点坐标」，只能让**同一个地名再次出现**时更准；
它没有形状维度，也无法回答「一张没有地名的概览图像哪条历史轨迹」。

这里存的是另一种资产：一次成功识别后，把像素空间的轨迹折线与 GCJ-02
地理折线**按弧长参数等距重采样到同样点数**，形成一一对应。于是模板本身
就是一张「像素 → 地理」的查找表：将来只要把新图的形状对齐到模板的像素
折线，就能按弧长插值出地理坐标，完全不必理解 ``Registration`` 的矩阵格式。

为什么必须支持「同类但不完全一样」
----------------------------------
固定骑行路线每次截图都会有差别：裁切范围不同、少截一段、多绕一小段、
往返方向相反。整条折线全局比对只能认出「同一张图」，所以这里同时提供：

* 尺度/平移归一化 —— 不同缩放等级、不同裁切；
* 反向变体 —— 同一条路线反着骑；
* **连续子段滑窗** —— 新图只是历史轨迹的一部分，或历史轨迹只是新图的
  一部分；用 ``coverage`` 记录「匹配上的弧长占较短折线的比例」，低于门槛
  一律拒识，避免「一小段形状碰巧相似」被当成整条路线。

P1 起本模块额外提供 ``TemplateStore.locate()``：形状命中后按弧长反推
GCJ-02 坐标。但它**只在主流程判定为宽泛概览图时**才被调用，且所有拒识
门槛通过后才会给出坐标；未命中一律回落原流程，绝不硬凑。城市级全域
形状搜索已被证明不可行（HANDOVER §21.4：前 12 候选分数差 0.1%，位置差
900-1600m），因此匹配范围严格限定在**已沉淀的模板集合**内。反推出来的
坐标没有当次地理核验，调用方必须标成「参考级/未核验」并封顶置信度。

P2 起增加 ``shape_domain`` 与 ``register_pair()``：一次请求同时给出
「概览单图」和「可选的多张街道级细节图」时，用多图配准出的真值地理折线
按弧长投影到概览图的形状上，注册成 ``overview`` 域模板。之后相似的
概览图只需单独上传即可命中，不必再带多图。``locate()`` 查询概览图时
只比对 ``overview`` 域，旧的 ``detail`` 域模板保留但不参与定位。
"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# 重采样点数：64 点足以保留街道级拐弯，又让 Chamfer 是 64×64 的小矩阵。
N_SAMPLES = 64
# 滑窗步长：起点/长度各按 8 点步进，避免组合爆炸。
WINDOW_STEP = 8
MIN_TRACK_POINTS = 10
# 自动沉淀门槛，与 lexicon_guard.MIN_AUTO_CONFIDENCE 对齐。
MIN_TEMPLATE_CONFIDENCE = 0.35
# 归一化空间（bbox 对角 = 1）的双向 Chamfer 距离上限。
MAX_SHAPE_RESIDUAL = 0.085
# 方向签名平均角差上限（弧度）。细长轨迹按 bbox 对角归一化后，纵向差异
# 会被压扁，光靠点集距离会把两条不同的波浪路线判成同类；有序方向序列
# 能区分「走法不同」，实测同类 0.00、异类 1.13 rad。
MAX_SIGNATURE_DIFF = 0.45
SIGNATURE_PENALTY = 0.05
# 匹配弧长占较短折线的比例下限；低于它说明只碰巧像了一小段。
MIN_MATCH_COVERAGE = 0.60
# 去重：几乎完全相同的轨迹不新增条目，改为累计命中次数。
DEDUP_RESIDUAL = 0.045
DEDUP_COVERAGE = 0.92
MAX_TEMPLATES = 200
# 选窗口时对「只匹配一小段」加惩罚，否则短子段归一化后更容易拟合，
# 会把整条完全相同的轨迹也判成局部匹配。
COVERAGE_PENALTY = 0.03

# --- P1：由形状反推坐标的额外门槛 -------------------------------------
# 实测教训：一条 2km 的「起终点连线」与十几公里的概览图轨迹，归一化后
# residual 仅 0.0438、signature 0.3873，会被判成同类——形状假阳性。
# 细长直线本身不携带可区分的走向信息，任一侧是这种形状就禁止用它定位。
MIN_SHAPE_STRAIGHTNESS = 0.80
MIN_SHAPE_ASPECT = 4.0
# template_in_query 模式下新图首尾有一段没有地理对应，只允许极小的缺口，
# 否则宁可拒识也不凭空外推坐标。
MIN_LOCATE_COVERAGE_B = 0.85
# 参考级定位的置信度上限：形状对齐没有任何当次地理核验。
TEMPLATE_CONFIDENCE_CAP = 0.45

# --- P2：形状域与配对注册 ---------------------------------------------
# 实测教训（§22.45）：模板的 px_points 来自细节图/拼接画布，而 locate()
# 的 query 永远是概览图，两个形状域混用会让 3.7km 的局部路线误命中
# 十几公里的概览图。配对注册把「概览图形状 ↔ 多图地理真值」直接绑定，
# 之后 key 与 query 同域，假阳性自然收敛。
SHAPE_DOMAIN_OVERVIEW = "overview"
SHAPE_DOMAIN_DETAIL = "detail"
# 配对时多图真值必须覆盖单图整条路线的比例下限。低于它说明只截了局部，
# 模板的地理折线会不完整，project_to_geo 会把子段外夹到端点、固化错误。
MIN_PAIR_COVERAGE = 0.85
# 模板地理里程的合理区间（米），挡住被污染的错误模板。
MIN_TEMPLATE_LENGTH_M = 100.0
MAX_TEMPLATE_LENGTH_M = 200_000.0


def _as_points(raw: Any) -> np.ndarray:
    """把 [[x, y], ...] 转成有限值的 float 数组，并去掉连续重复点。"""
    pts: list[list[float]] = []
    if raw is None:
        return np.empty((0, 2), dtype=float)
    for item in raw:
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if pts and abs(pts[-1][0] - x) < 1e-9 and abs(pts[-1][1] - y) < 1e-9:
            continue
        pts.append([x, y])
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def resample_by_arc(pts: Any, n: int = N_SAMPLES) -> np.ndarray | None:
    """按弧长等距重采样；退化（点数不足 / 总长为零）返回 None。"""
    p = _as_points(pts)
    if len(p) < 2:
        return None
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-9))
    p, seg = p[keep], seg[keep[:-1]]
    if len(p) < 2:
        return None
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if not math.isfinite(total) or total <= 1e-9:
        return None
    t = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=float)
    for d in range(2):
        out[:, d] = np.interp(t, cum, p[:, d])
    return out


def normalize_shape(pts: Any) -> np.ndarray | None:
    """平移到质心、按 bbox 对角线整体缩放（各向同性，不扭曲形状）。"""
    p = _as_points(pts)
    if len(p) < 2:
        return None
    q = p - p.mean(axis=0)
    diag = float(np.linalg.norm(q.max(axis=0) - q.min(axis=0)))
    if not math.isfinite(diag) or diag <= 1e-9:
        return None
    return q / diag


def direction_signature(pts: Any, k: int = 16) -> list[float]:
    """按弧长切成 k 段，返回每段的方向角（粗筛 / 审计用）。"""
    p = resample_by_arc(pts, max(k + 1, 2))
    if p is None:
        return []
    out: list[float] = []
    for i in range(k):
        a, b = p[i], p[i + 1]
        out.append(round(float(math.atan2(b[1] - a[1], b[0] - a[0])), 4))
    return out


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """双向平均最近点距离（输入应已归一化到同一尺度）。"""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0)


def ordered_distance(a: np.ndarray, b: np.ndarray) -> float:
    """按弧长序号逐点对应距离；对折线方向、连接顺序敏感。"""
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """形状距离 = 点集容错 + 有序约束。

    只用 Chamfer 会把「点集分布相同但走法不同」的两条路线判成同一条，
    反向变体也会失去意义；只用逐点对应又对中间多绕一小段过于敏感。
    两者加权后：局部容错由 Chamfer 提供，方向/顺序由有序项把关。
    """
    return 0.6 * chamfer(a, b) + 0.4 * ordered_distance(a, b)


def signature_distance(a: np.ndarray, b: np.ndarray, k: int = 16) -> float:
    """按弧长分段的**有序**方向角平均差（弧度，0 表示走法完全一致）。"""
    sa, sb = direction_signature(a, k), direction_signature(b, k)
    if len(sa) < k or len(sb) < k:
        return 0.0
    return float(sum(
        abs((x - y + math.pi) % (2 * math.pi) - math.pi)
        for x, y in zip(sa, sb)) / k)


def _round_points(p: np.ndarray, nd: int = 3) -> list[list[float]]:
    return [[round(float(x), nd), round(float(y), nd)] for x, y in p]


@dataclass(frozen=True)
class ShapeMatch:
    """一次形状比对的结果（不产出坐标，只回答「像不像、像了多少」）。"""

    accepted: bool
    residual: float
    coverage: float
    reversed: bool
    # 匹配子段在「较长折线」上的 [起点, 终点]（重采样点下标）。
    span: tuple[int, int]
    reason: str
    signature: float = 0.0
    # P1 投影必须知道「谁是谁的子段」，否则弧长映射方向会反：
    #   query_in_template —— 整条新图落在模板的 [span] 子段内；
    #   template_in_query —— 整条模板落在新图的 [span] 子段内。
    mode: str = "query_in_template"

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "residual": round(self.residual, 5),
            "coverage": round(self.coverage, 4),
            "signature": round(self.signature, 4),
            "reversed": self.reversed,
            "span": list(self.span),
            "mode": self.mode,
            "reason": self.reason,
        }


def shape_has_enough_info(px: Any) -> bool:
    """形状是否携带足够的「走向」信息，可用于反推坐标。

    判据与 ``pipeline._track_lacks_shape_info`` 保持一致：直线度 ≥0.80
    且 bbox 长宽比 ≥4.0 的扁长直线，本质是起终点连线，不同路线归一化后
    几乎长得一样。实测一条 2km 的连线与十几公里的概览图 residual 仅
    0.0438、signature 0.3873 会被判成同类，所以**任一侧**是这种形状
    都必须拒识，不能只靠 residual / signature 把关。
    """
    p = _as_points(px)
    if len(p) < MIN_TRACK_POINTS:
        return False
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(p[-1] - p[0]))
    straightness = chord / max(length, 1e-9)
    w = float(p[:, 0].max() - p[:, 0].min())
    h = float(p[:, 1].max() - p[:, 1].min())
    aspect = max(w, h) / max(1e-6, min(w, h))
    return not (straightness >= MIN_SHAPE_STRAIGHTNESS
                and aspect >= MIN_SHAPE_ASPECT)


def polyline_length_m(coords: Any) -> float:
    """同城尺度下的折线长度（米），用于模板里程的合理性检查。"""
    a = _as_points(coords)
    if len(a) < 2:
        return 0.0
    lat0 = math.radians(float(a[:, 1].mean()))
    d = np.diff(a, axis=0).copy()
    d[:, 0] *= 111_320.0 * math.cos(lat0)
    d[:, 1] *= 110_540.0
    return float(np.linalg.norm(d, axis=1).sum())


def project_to_geo(query_px: Any, template: dict[str, Any],
                   match: ShapeMatch) -> np.ndarray | None:
    """把当前图的像素折线按弧长映射到模板的 GCJ-02 折线。

    模板的 ``px_points`` 与 ``geo_points`` 是等弧长重采样的一一对应，所以
    这里不需要理解配准矩阵：只要求出「新图第 i 个点落在模板弧长的哪个
    位置」，在 geo_points 上线性插值即可。

    * ``query_in_template``：整条新图对应模板子段，按 span 精确取段；
    * ``template_in_query``：整条模板对应新图子段，子段外**夹到端点**
      （绝不外推坐标），因此 ``locate()`` 只在 coverage≥0.85 时放行。

    ``reversed`` 时按模板原方向的逆序取段，保证 GeoJSON 首点仍是起点。
    """
    q = resample_by_arc(query_px)
    geo = _as_points(template.get("geo_points"))
    tpx = _as_points(template.get("px_points"))
    if q is None or len(q) < 2 or len(geo) < 2 or len(tpx) < 2:
        return None
    n_q, n_t, n_g = len(q), len(tpx), len(geo)
    start = max(0, min(n_t - 2, int(match.span[0])))
    end = max(start + 1, min(n_t, int(match.span[1])))

    def geo_at(u: float) -> np.ndarray:
        """模板弧长参数 u∈[0,1] 处的 GCJ-02 坐标。"""
        u = max(0.0, min(1.0, float(u)))
        idx = u * (n_g - 1)
        i0 = int(math.floor(idx))
        i1 = min(n_g - 1, i0 + 1)
        w = idx - i0
        return geo[i0] * (1.0 - w) + geo[i1] * w

    out: list[np.ndarray] = []
    if match.mode == "query_in_template":
        # span 是模板（可能已反转）上的下标；换算回原模板弧长区间。
        if match.reversed:
            u_lo = (n_t - end) / (n_t - 1)
            u_hi = (n_t - 1 - start) / (n_t - 1)
        else:
            u_lo = start / (n_t - 1)
            u_hi = (end - 1) / (n_t - 1)
        u_from = u_hi if match.reversed else u_lo
        u_to = u_lo if match.reversed else u_hi
        for i in range(n_q):
            f = i / (n_q - 1)
            out.append(geo_at(u_from + f * (u_to - u_from)))
    else:
        for i in range(n_q):
            f = (i - start) / max(1.0, (end - 1) - start)
            f = max(0.0, min(1.0, f))
            out.append(geo_at(1.0 - f if match.reversed else f))

    arr = np.asarray(out, dtype=float)
    if arr.shape != (n_q, 2) or not np.isfinite(arr).all():
        return None
    return arr


def _window_span(n: int) -> list[tuple[int, int]]:
    """枚举连续子段：起点按步长滑动，长度至少覆盖 60%。"""
    floor_len = max(2, int(math.ceil(MIN_MATCH_COVERAGE * n)))
    spans: list[tuple[int, int]] = []
    for start in range(0, n // 2 + 1, WINDOW_STEP):
        for length in range(floor_len, n + 1, WINDOW_STEP):
            end = min(n, start + length)
            if end - start >= floor_len:
                spans.append((start, end))
    spans.append((0, n))
    return sorted(set(spans), key=lambda s: (s[0], s[1]))


_SPANS = _window_span(N_SAMPLES)


def match_shape(query_px: Any, template: dict[str, Any]) -> ShapeMatch:
    """把一张图的像素折线与一条模板比对，支持局部子段与反向变体。

    ``coverage`` 取「匹配子段占其所在折线的比例」的最小值：模板 10km、
    新图 2km 时即使这 2km 形状很像，覆盖也只有 0.2，会被拒掉。
    """
    q = resample_by_arc(query_px)
    t = _as_points(template.get("px_points"))
    if q is None or len(t) < MIN_TRACK_POINTS:
        return ShapeMatch(False, float("inf"), 0.0, False, (0, 0), "no_shape")
    t_rs = resample_by_arc(t)
    if t_rs is not None:
        t = t_rs
    if len(t) < 2:
        return ShapeMatch(False, float("inf"), 0.0, False, (0, 0), "no_shape")
    qn = normalize_shape(q)
    if qn is None:
        return ShapeMatch(False, float("inf"), 0.0, False, (0, 0), "degenerate_shape")

    best = (float("inf"), 0.0, False, (0, 0), 0.0, "query_in_template")
    best_score = float("inf")
    for rev in (False, True):
        tv = t[::-1] if rev else t
        tv_n = normalize_shape(tv)
        if tv_n is None:
            continue
        # A) 整条 query 对齐模板的某个子段（新图少截/只截了一段）
        for start, end in _SPANS:
            sub = tv[start:end]
            sub_n = normalize_shape(resample_by_arc(sub) if len(sub) != N_SAMPLES else sub)
            if sub_n is None:
                continue
            r = shape_distance(qn, sub_n)
            sig = signature_distance(qn, sub_n)
            cov = (end - start) / N_SAMPLES
            score = (r + COVERAGE_PENALTY * (1.0 - cov)
                     + SIGNATURE_PENALTY * sig)
            if score < best_score:
                best_score = score
                best = (r, cov, rev, (start, end), sig, "query_in_template")
        # B) 整条模板对齐 query 的某个子段（新图多绕了一段）
        for start, end in _SPANS:
            sub = q[start:end]
            sub_n = normalize_shape(resample_by_arc(sub) if len(sub) != N_SAMPLES else sub)
            if sub_n is None:
                continue
            r = shape_distance(sub_n, tv_n)
            sig = signature_distance(sub_n, tv_n)
            cov = (end - start) / N_SAMPLES
            score = (r + COVERAGE_PENALTY * (1.0 - cov)
                     + SIGNATURE_PENALTY * sig)
            if score < best_score:
                best_score = score
                best = (r, cov, rev, (start, end), sig, "template_in_query")

    r, cov, rev, span, sig, mode = best
    if not math.isfinite(r):
        return ShapeMatch(False, float("inf"), 0.0, False, (0, 0), "no_shape")
    if cov < MIN_MATCH_COVERAGE:
        return ShapeMatch(False, r, cov, rev, span, "low_coverage", sig, mode)
    if r > MAX_SHAPE_RESIDUAL:
        return ShapeMatch(False, r, cov, rev, span, "shape_mismatch", sig, mode)
    if sig > MAX_SIGNATURE_DIFF:
        return ShapeMatch(False, r, cov, rev, span, "direction_mismatch", sig, mode)
    return ShapeMatch(True, r, cov, rev, span, "ok", sig, mode)


class TemplateStore:
    """轨迹模板库（``data/route_templates.json``，含真实位置，不入库）。"""

    def __init__(self, path: Path, *, max_templates: int = MAX_TEMPLATES):
        self.path = Path(path)
        self.max_templates = max_templates
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": 1, "templates": []}
        self.load()

    # --- 持久化 ---------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                items = raw.get("templates") if isinstance(raw, dict) else None
                self._data = {"version": 1, "templates": list(items or [])}
            except (json.JSONDecodeError, OSError, AttributeError):
                self._data = {"version": 1, "templates": []}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )

    @property
    def templates(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._data.get("templates") or [])

    def stats(self) -> dict[str, Any]:
        items = self.templates
        return {
            "templates": len(items),
            "verified": sum(1 for x in items if x.get("verified")),
            "hits": sum(int(x.get("hit_count", 0) or 0) for x in items),
            "overview": sum(
                1 for x in items
                if (x.get("shape_domain") or SHAPE_DOMAIN_DETAIL)
                == SHAPE_DOMAIN_OVERVIEW),
        }

    # --- 沉淀 -----------------------------------------------------------
    def remember(
        self,
        *,
        px_polyline: Sequence[Sequence[float]],
        geo_polyline: Sequence[Sequence[float]],
        confidence: float,
        city: str = "",
        shot_id: str = "",
        source: str = "auto",
        verified: bool = False,
        self_consistent: bool = True,
        input_ok: bool = True,
        shape_domain: str = SHAPE_DOMAIN_DETAIL,
    ) -> dict[str, Any]:
        """裁决并沉淀一条模板；失败也返回原因，绝不抛给调用方。

        ``shape_domain`` 区分模板形状来自哪一种图：``detail`` 是细节图/
        拼接画布（P0 默认），``overview`` 是概览单图（P2 配对注册）。
        ``locate()`` 查询概览图时只比对后者，避免跨缩放形状假阳性。
        """
        px = _as_points(px_polyline)
        geo = _as_points(geo_polyline)
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        if len(px) < MIN_TRACK_POINTS or len(geo) < MIN_TRACK_POINTS:
            return {"written": False, "reason": "not_enough_points"}
        if not (np.isfinite(px).all() and np.isfinite(geo).all()):
            return {"written": False, "reason": "invalid_value"}
        if not (np.all(geo[:, 0] >= -180) and np.all(geo[:, 0] <= 180)
                and np.all(geo[:, 1] >= -90) and np.all(geo[:, 1] <= 90)):
            return {"written": False, "reason": "invalid_coordinate"}
        if not math.isfinite(conf):
            return {"written": False, "reason": "invalid_value"}
        if not verified and source != "user":
            if not self_consistent:
                return {"written": False, "reason": "self_inconsistent"}
            if not input_ok:
                return {"written": False, "reason": "low_information_input"}
            if conf < MIN_TEMPLATE_CONFIDENCE:
                return {"written": False, "reason": "low_confidence"}

        px_rs = resample_by_arc(px)
        geo_rs = resample_by_arc(geo)
        shape = normalize_shape(px_rs) if px_rs is not None else None
        if px_rs is None or geo_rs is None or shape is None:
            return {"written": False, "reason": "degenerate_shape"}

        with self._lock:
            items: list[dict[str, Any]] = self._data.setdefault("templates", [])
            hit = self._find_duplicate(items, px, shape)
            if hit is not None:
                hit["hit_count"] = int(hit.get("hit_count", 0) or 0) + 1
                if conf > float(hit.get("confidence", 0) or 0) or verified:
                    hit.update({
                        "px_points": _round_points(px_rs),
                        "geo_points": _round_points(geo_rs, 6),
                        "shape": _round_points(shape, 5),
                        "confidence": round(conf, 4),
                        "verified": bool(verified or hit.get("verified")),
                        "source": source,
                        "shot_id": shot_id,
                        # 配对注册可以把一条旧的 detail 模板升级成 overview，
                        # 否则它会永远无法参与概览图定位。
                        "shape_domain": str(shape_domain or SHAPE_DOMAIN_DETAIL),
                    })
                self.save()
                return {"written": False, "reason": "deduped",
                        "id": hit.get("id"), "templates": len(items)}

            tpl = {
                "id": f"tpl_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                "created_at": int(time.time()),
                "city": str(city or ""),
                "source": str(source or "auto"),
                "shape_domain": str(shape_domain or SHAPE_DOMAIN_DETAIL),
                "verified": bool(verified or source == "user"),
                "confidence": round(conf, 4),
                "hit_count": 1,
                "shot_id": str(shot_id or ""),
                "length_px": round(float(np.linalg.norm(
                    np.diff(px, axis=0), axis=1).sum()), 2),
                "px_points": _round_points(px_rs),
                "geo_points": _round_points(geo_rs, 6),
                "shape": _round_points(shape, 5),
                "signature": direction_signature(px_rs),
            }
            items.append(tpl)
            if len(items) > self.max_templates:
                items.sort(key=lambda x: (bool(x.get("verified")),
                                          float(x.get("confidence", 0) or 0),
                                          int(x.get("hit_count", 0) or 0)),
                           reverse=True)
                del items[self.max_templates:]
            self.save()
            return {"written": True, "reason": "ok", "id": tpl["id"],
                    "templates": len(items)}

    def _find_duplicate(self, items: list[dict[str, Any]],
                        px: np.ndarray, shape: np.ndarray) -> dict[str, Any] | None:
        q_rs = resample_by_arc(px)
        qn = normalize_shape(q_rs if q_rs is not None else px)
        if qn is None:
            return None
        for tpl in items:
            tn = _as_points(tpl.get("shape"))
            if len(tn) < 2:
                continue
            # 快速预筛：全局差异明显（正向/反向都不像）就跳过，避免滑窗。
            quick = min(shape_distance(qn, tn), shape_distance(qn, tn[::-1]))
            if quick > DEDUP_RESIDUAL * 3:
                continue
            m = match_shape(px, tpl)
            if m.accepted and m.coverage >= DEDUP_COVERAGE and m.residual <= DEDUP_RESIDUAL:
                return tpl
        return None

    # --- 召回（P0 只判定，不产出坐标）-----------------------------------
    def find_similar(
        self,
        query_px: Sequence[Sequence[float]],
        *,
        city_hint: str = "",
        require_verified: bool = False,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """找出形状足够接近的模板；没有就返回空列表（绝不硬凑一个）。"""
        out: list[dict[str, Any]] = []
        for tpl in self.templates:
            if require_verified and not tpl.get("verified"):
                continue
            if city_hint and tpl.get("city") and city_hint not in str(tpl["city"]):
                continue
            m = match_shape(query_px, tpl)
            if not m.accepted:
                continue
            if (not tpl.get("verified")
                    and float(tpl.get("confidence", 0) or 0) < MIN_TEMPLATE_CONFIDENCE):
                continue
            row = m.as_dict()
            row.update({
                "id": tpl.get("id"),
                "confidence": tpl.get("confidence"),
                "verified": bool(tpl.get("verified")),
                "hit_count": tpl.get("hit_count"),
            })
            out.append(row)
        out.sort(key=lambda x: (x["residual"] - 0.02 * x["coverage"]))
        return out[:max(0, limit)]

    # --- 定位（P1：命中才产出坐标）--------------------------------------
    def locate(
        self,
        query_px: Sequence[Sequence[float]],
        *,
        city_hint: str = "",
        shape_domain: str | None = None,
    ) -> dict[str, Any] | None:
        """形状命中后直接给出 GCJ-02 折线；任何一道门槛不过就返回 ``None``。

        绝不「凑一个最像的模板」：宁可让调用方回落到原流程，也不能把
        错误坐标包装成一个看起来精确的结果。相对 ``find_similar`` 额外加：

        * 新图与模板**两侧**都要有足够形状信息（挡细长直线假阳性）；
        * 模板地理里程必须落在合理区间；
        * ``template_in_query`` 需要 coverage≥0.85，避免大段无地理对应；
        * 投影坐标必须合法；
        * 指定 ``shape_domain`` 时只比对同域模板（概览图不碰细节图模板）。
        """
        if not shape_has_enough_info(query_px):
            return None
        best: dict[str, Any] | None = None
        for tpl in self.templates:
            if shape_domain is not None and (
                    tpl.get("shape_domain") or SHAPE_DOMAIN_DETAIL) != shape_domain:
                continue
            if city_hint and tpl.get("city") and city_hint not in str(tpl["city"]):
                continue
            if not shape_has_enough_info(tpl.get("px_points")):
                continue
            length_m = polyline_length_m(tpl.get("geo_points"))
            if not (MIN_TEMPLATE_LENGTH_M <= length_m <= MAX_TEMPLATE_LENGTH_M):
                continue
            if (not tpl.get("verified")
                    and float(tpl.get("confidence", 0) or 0) < MIN_TEMPLATE_CONFIDENCE):
                continue
            m = match_shape(query_px, tpl)
            if not m.accepted:
                continue
            if m.mode == "template_in_query" and m.coverage < MIN_LOCATE_COVERAGE_B:
                continue
            coords = project_to_geo(query_px, tpl, m)
            if coords is None:
                continue
            if not (np.all(coords[:, 0] >= -180) and np.all(coords[:, 0] <= 180)
                    and np.all(coords[:, 1] >= -90) and np.all(coords[:, 1] <= 90)):
                continue
            row = {
                "coords_gcj": [[round(float(lo), 7), round(float(la), 7)]
                               for lo, la in coords],
                "match": m.as_dict(),
                "template": {
                    "id": tpl.get("id"),
                    "city": tpl.get("city"),
                    "confidence": tpl.get("confidence"),
                    "verified": bool(tpl.get("verified")),
                    "hit_count": tpl.get("hit_count"),
                },
                "length_m": round(length_m, 1),
            }
            score = m.residual - 0.02 * m.coverage
            if best is None or score < (
                    best["match"]["residual"] - 0.02 * best["match"]["coverage"]):
                best = row
        return best

    def register_pair(
        self,
        *,
        overview_px: Sequence[Sequence[float]],
        detail_px: Sequence[Sequence[float]],
        geo_polyline: Sequence[Sequence[float]],
        confidence: float,
        city: str = "",
        shot_id: str = "",
        self_consistent: bool = True,
        input_ok: bool = True,
    ) -> dict[str, Any]:
        """「概览单图 + 多图真值」配对注册，产出 ``overview`` 域模板。

        ``detail_px`` 与 ``geo_polyline`` 来自同一次多图配准，二者按弧长
        一一对应。这里复用 :func:`project_to_geo` 把真值投影到概览图的
        弧长参数化上，再以「概览形状 ↔ 地理折线」沉淀，于是模板的 key
        与将来 ``locate()`` 的 query 都是概览图，形状域统一。

        失败一律返回 ``{"written": False, "reason": ...}``，不抛异常——
        配对注册是附加能力，绝不能让识别请求失败。
        """
        px = _as_points(overview_px)
        dp = _as_points(detail_px)
        geo = _as_points(geo_polyline)
        if (len(px) < MIN_TRACK_POINTS or len(dp) < MIN_TRACK_POINTS
                or len(geo) < MIN_TRACK_POINTS):
            return {"written": False, "reason": "not_enough_points"}
        if not (np.isfinite(px).all() and np.isfinite(dp).all()
                and np.isfinite(geo).all()):
            return {"written": False, "reason": "invalid_value"}
        if not (np.all(geo[:, 0] >= -180) and np.all(geo[:, 0] <= 180)
                and np.all(geo[:, 1] >= -90) and np.all(geo[:, 1] <= 90)):
            return {"written": False, "reason": "invalid_coordinate"}
        if not shape_has_enough_info(px) or not shape_has_enough_info(dp):
            return {"written": False, "reason": "degenerate_shape"}

        # match_shape / project_to_geo 都以「重采样后 64 点」的下标解释
        # span，临时真值模板必须先按弧长重采样，不能喂原始点数（78 点等）。
        dp_rs = resample_by_arc(dp)
        geo_rs = resample_by_arc(geo)
        if dp_rs is None or geo_rs is None:
            return {"written": False, "reason": "degenerate_shape"}
        truth = {"px_points": dp_rs, "geo_points": geo_rs}

        m = match_shape(px, truth)
        info = m.as_dict()
        if not m.accepted:
            return {"written": False, "reason": f"shape_{m.reason}",
                    "pair_match": info}
        # coverage 是「匹配子段占所在折线的比例」，两种 mode 下都表示
        # 多图真值与概览图的覆盖是否一致；低于门槛说明多图只截了局部，
        # 或概览图有大段没有地理对应。project_to_geo 的子段外一律夹到
        # 端点（绝不外推），0.85 以上的缺口最多影响 15% 的采样点。
        # ⚠️ 不能再限定 mode：实测完全同形的两条折线，滑窗打分会偏好
        # template_in_query，写死 query_in_template 会把正常配对全拒掉。
        if m.coverage < MIN_PAIR_COVERAGE:
            return {"written": False, "reason": "partial_route",
                    "pair_match": info}
        coords = project_to_geo(px, truth, m)
        if coords is None:
            return {"written": False, "reason": "projection_failed",
                    "pair_match": info}
        if not (np.all(coords[:, 0] >= -180) and np.all(coords[:, 0] <= 180)
                and np.all(coords[:, 1] >= -90) and np.all(coords[:, 1] <= 90)):
            return {"written": False, "reason": "invalid_coordinate",
                    "pair_match": info}

        out = self.remember(
            px_polyline=px,
            geo_polyline=coords,
            confidence=confidence,
            city=city,
            shot_id=shot_id,
            source="paired",
            self_consistent=self_consistent,
            input_ok=input_ok,
            shape_domain=SHAPE_DOMAIN_OVERVIEW,
        )
        out["pair_match"] = info
        out["shape_domain"] = SHAPE_DOMAIN_OVERVIEW
        return out
