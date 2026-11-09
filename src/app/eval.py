"""准确率评测：把「看起来准」变成可测量、可回归的量。

动机
----
流水线里原有的质量指标都是**自洽指标**：贴路中位距离、锚点残差、
confidence。它们在候选平移把直线搬到平行主干道时会一起变好，
但真实位置反而更错（交接文档 §22.38）。没有外部真值，
任何「提升准确率」的说法都无法证伪。

本模块提供两类可比较的量：

1. **轨迹级**（需要真值）：起终点误差、离散 Fréchet 距离、
   里程相对误差；
2. **配准级**（合成真值即可）：尺度相对误差、中心位移、
   内点精确率 / 召回率。

``synthetic_case()`` 生成带已知变换、可控噪声、离群点与 OCR 残片的
锚点集合，``run_synthetic_benchmark()`` 直接对 ``register()`` 打分。
合成真值不依赖任何用户截图，可在 CI 里跑；真实 ``(截图, GPX)``
对由 ``load_truth()`` / ``evaluate_tracks()`` 接入。

坐标约定：输入折线是 ``(lon, lat)``；几何量在**以首点为原点的局部
平面米**上计算（小范围等距近似，误差远小于本项目 10-30m 精度量级）。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .geo.coords import haversine, mercator_to_lonlat
from .register import AnchorPair, register

Point = tuple[float, float]

# Fréchet 计算前把折线等距重采样到这个点数上限，避免 700×700 的 DP。
_FRECHET_MAX_POINTS = 160


def _as_points(points: Iterable[Sequence[float]]) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if p is None or len(p) < 2:
            continue
        try:
            lon, lat = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(lon) and math.isfinite(lat):
            out.append((lon, lat))
    return out


def _local_xy(
    points: Sequence[Point], origin: Point | None = None
) -> np.ndarray:
    """经纬度 → 局部平面米。

    ⚠️ 比较两条折线时必须传**同一个 origin**：若各用自己的首点做原点，
    整体平移会被悄悄消掉，Fréchet 距离恒为 0（实测踩过）。
    """
    if not points:
        return np.zeros((0, 2), dtype=float)
    lon0, lat0 = origin if origin is not None else points[0]
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 111320.0
    return np.array(
        [[(lon - lon0) * kx, (lat - lat0) * ky] for lon, lat in points],
        dtype=float,
    )


def polyline_length_m(points: Iterable[Sequence[float]]) -> float:
    """折线的地表长度（米），用 haversine 累加。"""
    pts = _as_points(points)
    if len(pts) < 2:
        return 0.0
    return float(sum(haversine(*a, *b) for a, b in zip(pts, pts[1:])))


def _resample_xy(xy: np.ndarray, n: int) -> np.ndarray:
    """按累计弧长等距重采样，保持首末点。"""
    if len(xy) <= 1 or len(xy) <= n:
        return xy
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return xy[[0, -1]]
    target = np.linspace(0.0, cum[-1], n)
    out = np.empty((n, 2), dtype=float)
    for axis in (0, 1):
        out[:, axis] = np.interp(target, cum, xy[:, axis])
    return out


def endpoint_errors(
    pred: Iterable[Sequence[float]], truth: Iterable[Sequence[float]]
) -> dict[str, float]:
    """起终点误差（米）。方向由 GeoJSON 的 first_point_is_start 约定保证。"""
    a, b = _as_points(pred), _as_points(truth)
    if not a or not b:
        return {"start_m": float("inf"), "end_m": float("inf"),
                "max_m": float("inf")}
    start = haversine(*a[0], *b[0])
    end = haversine(*a[-1], *b[-1])
    return {"start_m": float(start), "end_m": float(end),
            "max_m": float(max(start, end))}


def discrete_frechet_m(
    pred: Iterable[Sequence[float]], truth: Iterable[Sequence[float]]
) -> float:
    """离散 Fréchet 距离（米）：衡量整条折线的形状偏差。

    与「平均最近点距离」不同，Fréchet 对顺序敏感——把轨迹整体挪到
    平行道路上、或把中间一段贴错路，都会显著增大。
    """
    p, t = _as_points(pred), _as_points(truth)
    if not p or not t:
        return float("inf")
    # 共用真值首点做原点，否则整体平移会被消掉
    A = _resample_xy(_local_xy(p, origin=t[0]), _FRECHET_MAX_POINTS)
    B = _resample_xy(_local_xy(t, origin=t[0]), _FRECHET_MAX_POINTS)
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    n, m = len(A), len(B)
    dp = np.full((n, m), np.inf, dtype=float)
    dp[0, 0] = cost[0, 0]
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], cost[0, j])
    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], cost[i, 0])
        for j in range(1, m):
            dp[i, j] = max(
                cost[i, j],
                min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]),
            )
    return float(dp[-1, -1])


def evaluate_tracks(
    pred: Iterable[Sequence[float]], truth: Iterable[Sequence[float]]
) -> dict[str, float]:
    """轨迹级指标汇总。"""
    p, t = _as_points(pred), _as_points(truth)
    ep = endpoint_errors(p, t)
    len_p, len_t = polyline_length_m(p), polyline_length_m(t)
    return {
        **ep,
        "frechet_m": discrete_frechet_m(p, t),
        "length_m": len_p,
        "truth_length_m": len_t,
        "length_rel_error": (abs(len_p - len_t) / len_t) if len_t > 0 else float("inf"),
        "n_points": len(p),
    }


def registration_errors(
    reg: Any,
    truth_matrix: np.ndarray,
    truth_translation: np.ndarray,
    probe_px: Sequence[float],
) -> dict[str, float]:
    """配准级指标：尺度相对误差 + 锚点质心处的绝对位移。"""
    truth_scale = float(
        (np.hypot(truth_matrix[0, 0], truth_matrix[0, 1])
         + np.hypot(truth_matrix[1, 0], truth_matrix[1, 1])) / 2.0
    )
    probe = np.asarray(probe_px, dtype=float)
    center_err = float(np.linalg.norm(
        (reg.matrix @ probe + reg.translation)
        - (np.asarray(truth_matrix) @ probe + np.asarray(truth_translation))
    ))
    return {
        "scale_m_per_px": reg.scale,
        "truth_scale_m_per_px": truth_scale,
        "scale_rel_error": abs(reg.scale - truth_scale) / max(truth_scale, 1e-9),
        "center_error_m": center_err,
        "matrix_max_diff": float(np.max(np.abs(
            reg.matrix - np.asarray(truth_matrix)))),
    }


def inlier_scores(
    inliers: Sequence[AnchorPair], truth_names: set[str]
) -> dict[str, float]:
    """内点判定的精确率 / 召回率（按地名字去重）。"""
    got = {p.name for p in inliers}
    tp = len(got & truth_names)
    return {
        "precision": tp / len(got) if got else 0.0,
        "recall": tp / len(truth_names) if truth_names else 0.0,
        "n_inlier_names": len(got),
        "n_truth_names": len(truth_names),
    }


@dataclass
class SyntheticCase:
    """一组带已知真值的锚点 / 轨迹，用于不依赖截图的回归评测。"""

    pairs: list[AnchorPair]
    matrix: np.ndarray
    translation: np.ndarray
    probe: np.ndarray
    truth_names: set[str]
    truth_track: list[Point]
    track_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def synthetic_case(
    seed: int = 0,
    n_good: int = 7,
    n_outliers: int = 2,
    n_fragments: int = 3,
    base_xy: tuple[float, float] = (13_500_000.0, 3_600_000.0),
    noise_m: float = 25.0,
) -> SyntheticCase:
    """构造合成基准：正北朝上变换 + 高斯噪声 + 离群点 + OCR 残片。

    残片锚点用**不带道路后缀**的截断名，坐标带 500-800m 偏移，
    模拟「云岭东」这类残片被宽松的 road 容差混进内点的情形。
    """
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(1.5, 6.0))
    matrix = np.array([[scale, 0.0], [0.0, -scale]], dtype=float)
    t = np.array(base_xy, dtype=float)
    px = rng.uniform(100.0, 1100.0, size=n_good)
    py = rng.uniform(100.0, 2100.0, size=n_good)
    good_px = np.column_stack([px, py])
    truth_names: set[str] = set()
    pairs: list[AnchorPair] = []

    for i, p in enumerate(good_px):
        xy = matrix @ p + t + rng.normal(0.0, noise_m, size=2)
        lon, lat = mercator_to_lonlat(float(xy[0]), float(xy[1]))
        name = f"基准{i}"
        truth_names.add(name)
        pairs.append(AnchorPair(
            name=name, px=float(p[0]), py=float(p[1]), weight=1.0,
            kind="poi", lon=lon, lat=lat, match_quality=1.0,
        ))

    for i in range(n_outliers):
        p = rng.uniform(100.0, 1100.0, size=2)
        xy = matrix @ p + t + rng.normal(0.0, 600.0, size=2)
        lon, lat = mercator_to_lonlat(float(xy[0]), float(xy[1]))
        pairs.append(AnchorPair(
            name=f"离群{i}", px=float(p[0]), py=float(p[1]), weight=1.0,
            kind="poi", lon=lon, lat=lat, match_quality=0.6,
        ))

    for i in range(n_fragments):
        p = rng.uniform(100.0, 1100.0, size=2)
        xy = matrix @ p + t + rng.normal(0.0, 650.0, size=2)
        lon, lat = mercator_to_lonlat(float(xy[0]), float(xy[1]))
        pairs.append(AnchorPair(
            name=f"片段{i}东", px=float(p[0]), py=float(p[1]), weight=1.0,
            kind="road", lon=lon, lat=lat, match_quality=0.5,
        ))

    # 真值轨迹：像素折线 → 经纬度，后续用 reg 反算的轨迹与它比较
    track_px = np.column_stack([
        np.linspace(150.0, 950.0, 24),
        np.linspace(300.0, 1800.0, 24)
        + 120.0 * np.sin(np.linspace(0.0, 3.0, 24)),
    ])
    truth_track = []
    for p in track_px:
        x, y = matrix @ p + t
        truth_track.append(mercator_to_lonlat(float(x), float(y)))

    probe = good_px.mean(axis=0)
    return SyntheticCase(
        pairs=pairs, matrix=matrix, translation=t, probe=probe,
        truth_names=truth_names, truth_track=truth_track, track_px=track_px,
    )


def run_synthetic_benchmark(n_cases: int = 20, seed0: int = 0) -> dict[str, Any]:
    """对 register() 打合成基准分，返回逐样本与聚合结果。

    这是 A/B 的基线：任何改动前后跑同一 seed，比较
    ``center_error_m`` / ``scale_rel_error`` / ``precision`` / ``recall``
    的 P50 / P90，而不是只看「贴路距离变好了」。
    """
    cases: list[dict[str, Any]] = []
    for k in range(max(0, n_cases)):
        c = synthetic_case(seed=seed0 + k)
        reg = register(c.pairs)
        row: dict[str, Any] = {"seed": seed0 + k, "ok": bool(reg.ok)}
        if reg.ok:
            row.update(registration_errors(reg, c.matrix, c.translation, c.probe))
            row.update(inlier_scores(reg.inliers, c.truth_names))
            pred = [reg.pixel_to_lonlat(float(p[0]), float(p[1]))
                    for p in c.track_px]
            row.update(evaluate_tracks(pred, c.truth_track))
        cases.append(row)

    def _agg(key: str) -> dict[str, float]:
        vals = sorted(float(r[key]) for r in cases
                      if r.get("ok") and math.isfinite(float(r.get(key, float("inf")))))
        if not vals:
            return {"p50": float("nan"), "p90": float("nan"), "mean": float("nan")}
        return {
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "mean": float(np.mean(vals)),
        }

    keys = ("center_error_m", "scale_rel_error", "precision", "recall",
            "start_m", "end_m", "frechet_m", "length_rel_error")
    return {
        "n_cases": len(cases),
        "n_ok": sum(1 for r in cases if r["ok"]),
        "aggregate": {k: _agg(k) for k in keys},
        "cases": cases,
    }


def load_truth(path: str | Path) -> list[Point]:
    """加载真值折线：GeoJSON / 坐标数组 JSON / GPX。"""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".gpx", ".xml"):
        root = ET.fromstring(raw)
        ns = {"g": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        pts: list[Point] = []
        for tag in ("trkpt", "rtept", "wpt"):
            for node in root.iterfind(f".//g:{tag}", ns) if ns else root.iterfind(f".//{tag}"):
                lat, lon = node.get("lat"), node.get("lon")
                if lat and lon:
                    pts.append((float(lon), float(lat)))
        return _as_points(pts)
    data = json_loads(raw)
    if isinstance(data, list):
        return _as_points(data)
    features = data.get("features") or []
    if features:
        geom = features[0].get("geometry") or {}
        return _as_points(geom.get("coordinates") or [])
    return _as_points(data.get("coordinates") or [])


def load_prediction(path: str | Path) -> list[Point]:
    """从识别结果 JSON 或 GeoJSON 里取主轨迹（WGS-84）。"""
    data = json_loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return _as_points(data)
    features = (data.get("geojson") or data).get("features") or []
    if features:
        geom = features[0].get("geometry") or {}
        return _as_points(geom.get("coordinates") or [])
    return _as_points(data.get("coordinates") or [])


def json_loads(raw: str) -> Any:
    import json

    return json.loads(raw)


def evaluate_files(pred_path: str | Path, truth_path: str | Path) -> dict[str, Any]:
    """真实样本 A/B 入口：两份文件 → 轨迹级指标。"""
    return evaluate_tracks(load_prediction(pred_path), load_truth(truth_path))
