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

⚠️ 本模块在 P0 只负责「沉淀 + 相似度判定」，**不产出坐标、不改变主流程
识别结果**。把它接进无地名兜底路径是 P1 的事。城市级全域形状搜索已被
证明不可行（HANDOVER §21.4：前 12 候选分数差 0.1%，位置差 900-1600m），
因此匹配范围严格限定在**已沉淀的模板集合**内，且必须带拒识门槛。
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "residual": round(self.residual, 5),
            "coverage": round(self.coverage, 4),
            "signature": round(self.signature, 4),
            "reversed": self.reversed,
            "span": list(self.span),
            "reason": self.reason,
        }


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

    best = (float("inf"), 0.0, False, (0, 0), 0.0)
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
                best = (r, cov, rev, (start, end), sig)
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
                best = (r, cov, rev, (start, end), sig)

    r, cov, rev, span, sig = best
    if not math.isfinite(r):
        return ShapeMatch(False, float("inf"), 0.0, False, (0, 0), "no_shape")
    if cov < MIN_MATCH_COVERAGE:
        return ShapeMatch(False, r, cov, rev, span, "low_coverage", sig)
    if r > MAX_SHAPE_RESIDUAL:
        return ShapeMatch(False, r, cov, rev, span, "shape_mismatch", sig)
    if sig > MAX_SIGNATURE_DIFF:
        return ShapeMatch(False, r, cov, rev, span, "direction_mismatch", sig)
    return ShapeMatch(True, r, cov, rev, span, "ok", sig)


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
    ) -> dict[str, Any]:
        """裁决并沉淀一条模板；失败也返回原因，绝不抛给调用方。"""
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
                    })
                self.save()
                return {"written": False, "reason": "deduped",
                        "id": hit.get("id"), "templates": len(items)}

            tpl = {
                "id": f"tpl_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                "created_at": int(time.time()),
                "city": str(city or ""),
                "source": str(source or "auto"),
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
