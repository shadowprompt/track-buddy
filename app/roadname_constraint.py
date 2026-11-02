"""路名线约束：用用户提供的路名作为「线约束」重新定位。

为什么需要这个
--------------
低缩放截图（一张图覆盖 20 公里）的定位信号本质上是**多峰**的：
OCR 出来的只有行政区名和地铁线号，底图匹配实测前 12 个候选的
匹配分数差只有 0.1%，但地理位置分散在 900-1600 米。

用户提供的路名是**极强的约束**，而且比让他填经纬度现实得多
（他答不上经纬度，但答得上「我骑了古北路」）。

关键设计：路名不能当「点」用
------------------------------
全市范围实测（见 docs/HANDOVER.md 21.9）：

| 路名 | 段数 | 点锚点散布 |
|---|---|---|
| 打浦路隧道 | 8 | 1.8 km |
| 龙东大道 | 13 | 1.7 km |
| 内环高架路 | 117 | 11.8 km |
| 世纪大道 | 154 | 62.4 km |

「世纪大道」有 154 段，当点锚点几乎没用。
但当**线约束**（让整条轨迹贴着这条路的几何走）就非常有效。

实测效果（城市级样本，用全部路网作独立裁判）：
  原落点（纯 OCR）      轨迹到全路网中位 191 米
  单路名约束后                             98 米
  多路名约束后                            100 米

即：用户输入路名让定位精度几乎翻倍。

与 `app/roadfit.py` 的区别
--------------------------
- `roadfit.py`：回答「这条轨迹骑的是哪几条路」（路名**归属**）
- 本模块：回答「如果我走过 XX 路，那轨迹应该在哪」（路名**约束**）
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geo.coords import gcj02_to_wgs84, lonlat_to_mercator

# Overpass 公共镜像。主站繁忙时轮换，且每个镜像给较短超时，
# 快速失败换下一个而不是死等。
_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# 全市检索用的粗 bbox 会在调用处按城市给定；这里只放兜底值。
_DEFAULT_BBOX = (120.85, 30.65, 121.60, 31.55)


@dataclass
class RoadNameConstraint:
    """一个路名约束的结果。"""

    name: str
    ok: bool = False
    # 该路名在检索范围内找到的路段（GCJ-02 墨卡托米）
    segments: list[list[tuple[float, float]]] = field(default_factory=list)
    # 路段数量（衡量这个路名的唯一性）
    n_segments: int = 0
    # 该路名总长（米）
    total_len_m: float = 0.0
    # 备注（失败原因等）
    notes: list[str] = field(default_factory=list)

    @property
    def uniqueness(self) -> float:
        """唯一性：0（完全没有）到 1（全市只有一段）。

        路段越多、散布越广，这个值越低——它直接决定约束强度。
        """
        if not self.ok or not self.segments:
            return 0.0
        if self.n_segments == 1:
            return 1.0
        # 段数越多越不唯一；用对数衰减，10 段约 0.5，100 段约 0.25
        return max(0.05, 1.0 / (1.0 + math.log10(self.n_segments)))


def _overpass(query: str, timeout: float = 90.0) -> dict[str, Any]:
    data = urllib.parse.urlencode({"data": query}).encode()
    last: Exception | None = None
    for i, mirror in enumerate(_OVERPASS_MIRRORS):
        per = timeout if i == 0 else max(10.0, timeout / len(_OVERPASS_MIRRORS))
        req = urllib.request.Request(
            mirror, data=data, headers={"User-Agent": "track-buddy/0.1"}
        )
        try:
            txt = urllib.request.urlopen(req, timeout=per).read().decode()
            # Overpass 繁忙时返回的是 HTML 网关错误页而不是 JSON
            if txt.lstrip()[:1] != "{":
                raise ValueError("响应不是 JSON")
            return json.loads(txt)
        except Exception as exc:  # noqa: BLE001 — 换下一个镜像
            last = exc
            time.sleep(0.8)
    raise RuntimeError(f"Overpass 全部镜像失败：{last}")


def fetch_road_name(
    name: str,
    bbox: tuple[float, float, float, float] = _DEFAULT_BBOX,
    timeout: float = 90.0,
) -> RoadNameConstraint:
    """按路名检索该路在给定范围内的全部路段。

    bbox 是 **WGS-84** 的 (west, south, east, north)。
    返回的 segments 是 **GCJ-02 墨卡托米**——与配准结果同基准，
    可以直接比较。
    """
    out = RoadNameConstraint(name=name)
    nm = name.strip()
    if not nm:
        out.notes.append("路名为空")
        return out

    # OSM 的 name 精确匹配。加引号转义，避免特殊字符破坏查询。
    safe = nm.replace('"', '\\"')
    q = (
        f'[out:json][timeout:{int(timeout)}];'
        f'way["name"="{safe}"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});out geom;'
    )
    try:
        data = _overpass(q, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        out.notes.append(f"查询失败：{type(exc).__name__}")
        return out

    for e in data.get("elements", []):
        g = e.get("geometry") or []
        if len(g) < 2:
            continue
        # OSM 是 WGS-84，转到 GCJ-02（与美团底图同基准）
        pts = [
            lonlat_to_mercator(*gcj02_to_wgs84(p["lon"], p["lat"]))
            for p in g
        ]
        seg_len = sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )
        if seg_len < 30.0:
            continue    # 忽略极短的碎片
        out.segments.append(pts)
        out.total_len_m += seg_len

    out.n_segments = len(out.segments)
    if not out.segments:
        out.notes.append(f"OSM 里查不到「{nm}」的几何")
        return out
    out.ok = True
    return out


def _dist_to_segments(
    points: np.ndarray, segments: list[list[tuple[float, float]]]
) -> np.ndarray:
    """每个点到「所有路段」的最近距离（米）。

    返回形状与 points 相同的向量。
    """
    if not segments:
        return np.full(len(points), 1e9)
    best = None
    for poly in segments:
        for i in range(1, len(poly)):
            x1, y1 = poly[i - 1]
            x2, y2 = poly[i]
            dx, dy = x2 - x1, y2 - y1
            seg2 = dx * dx + dy * dy
            rel = points - np.array([x1, y1])
            if seg2 < 1e-9:
                t = np.zeros(len(points))
            else:
                t = np.clip((rel[:, 0] * dx + rel[:, 1] * dy) / seg2, 0.0, 1.0)
            proj = np.stack([x1 + t * dx, y1 + t * dy], axis=1)
            d = np.linalg.norm(points - proj, axis=1)
            best = d if best is None else np.minimum(best, d)
    return best if best is not None else np.full(len(points), 1e9)


def refine_by_road_names(
    traj_merc: np.ndarray,
    constraints: list[RoadNameConstraint],
    max_shift_m: float = 4000.0,
    coarse_step_m: float = 200.0,
    sample_n: int = 150,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """用路名约束搜索最优平移。

    目标函数：min  Σ w_i · median(轨迹点到第 i 条路名的距离)

    返回 (平移向量 dx,dy, 目标函数值, 备注列表)。

    ⚠️ **只平移，不改尺度**。尺度由锚点配准给出（那是几何上更可靠的
    约束）；路名约束用来修正**绝对位置**，这两件事分开做更稳。
    """
    notes: list[str] = []
    usable = [c for c in constraints if c.ok and c.segments]
    if not usable:
        return np.zeros(2), float("inf"), ["没有可用的路名约束"]

    if len(traj_merc) < 2:
        return np.zeros(2), float("inf"), ["轨迹点太少"]
    step_idx = max(1, len(traj_merc) // sample_n)
    sample = traj_merc[::step_idx]

    if weights is None:
        # 唯一性越高、段数越少的路名，权重越大
        weights = [c.uniqueness for c in usable]
    tot_w = sum(weights) or 1.0

    def objective(dx: float, dy: float) -> float:
        moved = sample + np.array([dx, dy])
        val = 0.0
        for c, w in zip(usable, weights):
            d = _dist_to_segments(moved, c.segments)
            val += w * float(np.median(d))
        return val / tot_w

    base = objective(0.0, 0.0)
    best = (base, 0.0, 0.0)
    _best_sq = 0

    # 粗搜
    n = int(max_shift_m / coarse_step_m)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if i * i + j * j > n * n:
                continue
            v = objective(i * coarse_step_m, j * coarse_step_m)
            # ⚠️ 平局时取**离原位更近**的解
            #
            # 一条长路（例如高架）沿线任意平移都能让轨迹贴上去，
            # 目标值完全相同。若不加这个判据，搜索会滑到搜索范围的
            # 边缘（实测一条 10 公里的直路能让东西向平移跑到 -950 米），
            # 轨迹被平移到路的另一段上去——距离虽然还是 0，位置却错了。
            #
            # 平局取近的，等价于「在同样贴路的前提下优先不移动」。
            if v < best[0] - 1e-9 or (
                abs(v - best[0]) <= 1e-9
                and (i * i + j * j) < _best_sq
            ):
                best = (v, i * coarse_step_m, j * coarse_step_m)
                _best_sq = i * i + j * j

    # 精搜
    val, bx, by = best
    for _ in range(5):
        improved = False
        for di in range(-10, 11):
            for dj in range(-10, 11):
                sx = bx + di * coarse_step_m / 10
                sy = by + dj * coarse_step_m / 10
                v = objective(sx, sy)
                if v < val - 1e-6:
                    val, bx, by = v, sx, sy
                    improved = True
        if not improved:
            break

    notes.append(
        f"路名约束：{len(usable)} 条路，目标值 {base:.0f} → {val:.0f} 米，"
        f"平移 东{bx:+.0f} 北{by:+.0f} 米"
    )
    for c in usable:
        notes.append(
            f"  「{c.name}」{c.n_segments} 段 / {c.total_len_m/1000:.1f} km"
            f"，唯一性 {c.uniqueness:.2f}"
        )
    return np.array([bx, by], dtype=float), val, notes
