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
import re
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


def _cache_load(path: str) -> dict[str, Any]:
    """读路名几何缓存。损坏时返回空 dict（不影响主流程）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cache_store(path: str, key: str, value: dict[str, Any]) -> None:
    """写路名几何缓存。失败不影响主流程。"""
    try:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = _cache_load(path)
        data[key] = value
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


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
    cache_path: str | None = None,
    roads_cache_path: str | None = None,
) -> RoadNameConstraint:
    """按路名检索该路在给定范围内的全部路段。

    ⚠️ **数据源策略（踩过坑才这么定）**：

    最初的做法是「每个路名查一次 Overpass」，实测暴露三个问题：
      1. **慢**：单次查询 36-60 秒，10 个路名就是 5-10 分钟；
      2. **不稳**：Overpass 繁忙时返回 **HTTP 504 / 部分结果**，
         同一个路名这次查到下次查不到（实测「小木桥路」「虹桥路」
         「中山南二路」都出现过「上次 6 段、下次 0 条」）；
      3. **UTF-8 缺陷**：Overpass QL 的精确匹配 `["name"="龙华中路"]`
         对中文普遍失效，必须用正则 `~"^...$"` 才行——很容易被
         误判成「OSM 里没这条路」。

    正解：**一次拉全量具名道路，然后本地按名字过滤**。
    `fetch_roads` 的查询不带 name 过滤（只按 highway/name），完全绕开
    UTF-8 问题；一次查询覆盖整个区域，本地匹配任意路名都免费。

    实测同一批路名：
      逐个查 Overpass：命中 3/10
      拉全量再本地匹配：命中 **9/10**（只有「安龙新村」缺失——它是小区不是路）

    返回的 segments 是 **GCJ-02 墨卡托米**——与配准结果同基准。
    """
    out = RoadNameConstraint(name=name)
    nm = name.strip()
    if not nm:
        out.notes.append("路名为空")
        return out

    # --- 优先用本地路网缓存（一次全量查询 + 本地匹配） ---
    if roads_cache_path:
        roads = _load_roads_cache(roads_cache_path, bbox, timeout)
        if roads:
            matched = _match_by_name(roads, nm)
            if matched:
                out.segments = matched
                out.n_segments = len(matched)
                out.total_len_m = _segments_len(matched)
                out.ok = True
                return out

    # --- 兜底：直接按路名查 Overpass（正则匹配，绕开 UTF-8 缺陷） ---
    key = f"{nm}|{bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}"
    if cache_path:
        hit = _cache_load(cache_path).get(key)
        if hit is not None:
            out.ok = bool(hit.get("ok"))
            out.segments = [[(float(x), float(y)) for x, y in seg]
                            for seg in hit.get("segments", [])]
            out.n_segments = len(out.segments)
            out.total_len_m = float(hit.get("total_len_m", 0.0))
            out.notes = list(hit.get("notes", []))
            return out

    safe = re.escape(nm)
    q = (
        f'[out:json][timeout:{int(timeout)}];'
        f'way["name"~"^{safe}$",i]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});'
        f'out geom;'
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
        if cache_path:
            _cache_store(cache_path, key, {
                "ok": False, "segments": [], "total_len_m": 0.0,
                "notes": out.notes,
            })
        return out
    out.ok = True
    if cache_path:
        _cache_store(cache_path, key, {
            "ok": True,
            "segments": [[[x, y] for x, y in seg] for seg in out.segments],
            "total_len_m": out.total_len_m,
            "notes": [],
        })
    return out


# 小区 / 园区类后缀——用户常报「安龙新村」这种，但 OSM 里只有「安龙路」
_SUFFIX_WORDS = (
    "新村", "小区", "花园", "苑", "家园", "公寓", "广场", "中心",
    "园区", "公寓区", "别墅区", "城", "府", "台", "湾", "里",
)


def _strip_suffix(name: str) -> str:
    """去掉小区类后缀，得到可能的道路名。"""
    out = name
    for suf in _SUFFIX_WORDS:
        if out.endswith(suf) and len(out) > len(suf) + 1:
            out = out[: -len(suf)]
            break
    return out


def _match_by_name(
    roads: list[tuple[str, list[tuple[float, float]]]], name: str
) -> list[list[tuple[float, float]]]:
    """在全量路网里按名字找路段（GCJ-02 墨卡托米）。

    匹配优先级：
      1. **精确匹配**——OSM 名与输入完全一致（「水城路」）
      2. **去后缀匹配**——输入是小区名时去掉「新村」等后缀再匹配
         （「安龙新村」→「安龙路」）
      3. **无「弄/号」变体匹配**——「虹桥路」→「虹桥路」但排除
         「虹桥路996弄」这类支弄

    返回的坐标是 GCJ-02 墨卡托米（`fetch_roads` 给的是 WGS-84 经纬度）。
    """
    def _conv(pts):
        return [lonlat_to_mercator(*gcj02_to_wgs84(lo, la)) for lo, la in pts]

    # 1) 精确
    exact = [_conv(pts) for nm, pts in roads if nm == name]
    if exact:
        return exact

    # 2) 去后缀
    stripped = _strip_suffix(name)
    if stripped != name:
        got = [_conv(pts) for nm, pts in roads if nm == stripped]
        if got:
            return got

    # 3) 去掉「xxx弄」这类支弄后缀
    got = []
    for nm, pts in roads:
        base = re.sub(r"\d+弄$", "", nm)
        base = re.sub(r"\d+号.*$", "", base)
        if base == name or base == stripped:
            got.append(_conv(pts))
    if got:
        return got

    # 4) 「小区名」→「小区旁的路」：去后缀后再加常见路尾
    #    「安龙新村」→「安龙路」（实测 OSM 里只有「安龙路」）
    if stripped != name:
        for tail in ("路", "街", "道", "东路", "西路", "南路", "北路"):
            cand = stripped + tail
            got = [_conv(pts) for nm, pts in roads if nm == cand]
            if got:
                return got

    # 5) 前缀匹配：OSM 名以输入开头且长度接近
    #    「虹桥」→「虹桥路」「虹桥路996弄」
    for nm, pts in roads:
        if nm.startswith(name) and len(nm) <= len(name) + 3:
            got.append(_conv(pts))
    if got:
        return got
    return []


def _segments_len(segments: list[list[tuple[float, float]]]) -> float:
    total = 0.0
    for poly in segments:
        for i in range(1, len(poly)):
            total += math.hypot(poly[i][0] - poly[i - 1][0],
                                poly[i][1] - poly[i - 1][1])
    return total


def _load_roads_cache(
    cache_path: str,
    bbox: tuple[float, float, float, float],
    timeout: float,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """从路网缓存里取覆盖 bbox 的具名道路；没有就查一次并落盘。

    复用 `app/roadfit.py` 的 `fetch_roads`——它的查询不带 name 过滤，
    绕开 Overpass 的 UTF-8 精确匹配缺陷。
    """
    from .roadfit import fetch_roads

    try:
        return fetch_roads(bbox, timeout=timeout, cache_path=cache_path)
    except Exception:
        return []


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


def _align_segments(
    traj_merc: np.ndarray,
    constraints: list[RoadNameConstraint],
    weights: list[float],
    n_seg: int = 30,
) -> list[tuple[int, int]]:
    """用动态规划把路名序列对齐到轨迹分段。

    为什么需要这个：用户给的路名是**骑行顺序**，但每条路占的弧长
    比例**不是 1/N**。等分分段会错位——实测 10 个路名等分后，
    「小木桥路」被分到的段里离它 2855 米（而整条轨迹上离它只有 27 米）。

    做法：把轨迹分成 n_seg 段，求「路名序列 → 段序列」的单调最优匹配
    （类似序列对齐 / DTW）。允许路名跳过一些段，但不允许交叉。

    返回 [(seg_start, seg_end), ...]，与 constraints 一一对应。
    """
    n_names = len(constraints)
    if n_names == 0:
        return []
    n_seg = max(n_names, min(n_seg, len(traj_merc)))

    seg_len = np.linalg.norm(np.diff(traj_merc, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1]) or 1.0
    bounds = [total * i / n_seg for i in range(n_seg + 1)]

    # 段采样点
    seg_pts: list[np.ndarray] = []
    for i in range(n_seg):
        m = (cum >= bounds[i]) & (cum <= bounds[i + 1])
        pts = traj_merc[m]
        if len(pts) == 0:
            j = int(np.argmin(np.abs(cum - bounds[i])))
            pts = traj_merc[max(0, j - 1):j + 2]
        if len(pts) > 20:
            idx = np.linspace(0, len(pts) - 1, 20).astype(int)
            pts = pts[idx]
        seg_pts.append(pts)

    # cost[i][j] = 第 i 个路名解释第 j 段的代价
    cost = np.zeros((n_names, n_seg), dtype=float)
    for i, c in enumerate(constraints):
        for j, pts in enumerate(seg_pts):
            if len(pts) == 0:
                cost[i, j] = 1e5
                continue
            d = _dist_to_segments(pts, c.segments)
            cost[i, j] = float(np.median(d))

    # DP：dp[i][j] = 前 i+1 个路名都匹配好，且第 i 个路名落在第 j 段的最小代价
    INF = float("inf")
    dp = np.full((n_names, n_seg), INF)
    back = np.zeros((n_names, n_seg), dtype=int)
    for i in range(n_names):
        for j in range(n_seg):
            if j < i:
                continue    # 前 i 个路名至少占 i+1 段
            if i == 0:
                dp[i, j] = cost[i, j]
                back[i, j] = -1
            else:
                # 从 dp[i-1][j-1]（前一个路名在更早的段）转移
                if dp[i - 1, j - 1] < INF:
                    dp[i, j] = dp[i - 1, j - 1] + cost[i, j]
                    back[i, j] = j - 1
    # 最后一个路名放在最靠后的可行位置
    last = int(np.argmin(dp[n_names - 1]))
    if not np.isfinite(dp[n_names - 1, last]):
        # 退化成等分
        return [(total * i / n_names, total * (i + 1) / n_names)
                for i in range(n_names)]

    # 回溯
    pairs: list[tuple[int, int]] = [(0, 0)] * n_names
    j = last
    for i in range(n_names - 1, -1, -1):
        pairs[i] = (bounds[j], bounds[j + 1])
        j = back[i, j]
        if j < 0:
            break
    # 回溯顺序反了，且首段起点应为 0
    pairs[0] = (0.0, pairs[0][1])
    return pairs


def segment_base_value(
    traj_merc: np.ndarray,
    constraints: list[RoadNameConstraint],
    weights: list[float] | None = None,
) -> float:
    """按「DP 对齐 + 分段」口径计算原位目标值。

    ⚠️ 必须与 `refine_by_road_names` 用**完全相同**的口径——
    否则会把两种量纲不同的数放在一起比（实测踩过：分段约束的
    最终值 866 米 vs 整体约束的基准值 3534 米，改善率算出来是假的）。
    """
    usable = [c for c in constraints if c.ok and c.segments]
    if not usable or len(traj_merc) < 2:
        return float("inf")
    if weights is None:
        weights = [c.uniqueness for c in usable]
    tot_w = sum(weights) or 1.0

    seg_len = np.linalg.norm(np.diff(traj_merc, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = float(cum[-1])
    n_seg = len(usable)
    try:
        spans = _align_segments(traj_merc, usable, weights)
    except Exception:
        spans = [(total_len * i / n_seg, total_len * (i + 1) / n_seg)
                 for i in range(n_seg)]

    val = 0.0
    for (lo, hi), c, w in zip(spans, usable, weights):
        m = (cum >= lo) & (cum <= hi)
        pts = traj_merc[m]
        if len(pts) < 2:
            j = int(np.argmin(np.abs(cum - lo)))
            pts = traj_merc[max(0, j - 1):j + 2]
        if len(pts) > 30:
            idx = np.linspace(0, len(pts) - 1, 30).astype(int)
            pts = pts[idx]
        d = _dist_to_segments(pts, c.segments)
        val += w * float(np.median(d))
    return val / tot_w


def refine_by_road_names(
    traj_merc: np.ndarray,
    constraints: list[RoadNameConstraint],
    max_shift_m: float = 4000.0,
    coarse_step_m: float = 200.0,
    sample_n: int = 150,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """用路名约束搜索最优平移。

    ⚠️ **两种模式**（这是本模块最关键的设计）：

    ### 模式 A：整体约束（ordered=False）
    目标函数 = min Σ wᵢ · median(轨迹全部点到第 i 条路的距离)

    适用于「用户只给了一条路，轨迹整体应该在那条路上」。

    ### 模式 B：分段约束（ordered=True，默认）
    目标函数 = min Σ wᵢ · median(第 i 段轨迹点到第 i 条路的距离)

    **为什么必须分段**：一条 16 公里的骑行轨迹会经过 10 条不同的路，
    但它**不可能同时贴着这 10 条路**——只在每条路的**一小段**上。
    整体约束在这种输入下必然失败（实测目标值只从 3534 降到 3161，
    改善 11%，达不到采纳门槛）。

    分段的做法：用户给的路名是**骑行顺序**（小木桥路 → 龙华中路 → … → 安龙新村），
    所以把轨迹按弧长等分成 N 段（N = 路名数），第 i 段配第 i 个路名。
    这样每个路名只约束它**实际负责的那一段**，几何上自洽。

    返回 (平移向量 dx,dy, 目标函数值, 备注列表, 原位基准值)。

    ⚠️ 基准值由本函数直接返回，**不要在调用方另算**——重算一次
    DP 对齐可能得到不同的匹配（DP 有并列解），导致改善率失真
    （实测踩过：这里 2059、调用方重算 3534，改善率算出来是假的）。

    ⚠️ **只平移，不改尺度**：尺度由锚点配准给出（那是几何上更可靠的
    约束）；路名约束只用来修正**绝对位置**。
    """
    notes: list[str] = []
    usable = [c for c in constraints if c.ok and c.segments]
    if not usable:
        return np.zeros(2), float("inf"), ["没有可用的路名约束"], 0.0

    if len(traj_merc) < 2:
        return np.zeros(2), float("inf"), ["轨迹点太少"], 0.0

    if weights is None:
        # 唯一性越高、段数越少的路名，权重越大
        weights = [c.uniqueness for c in usable]
    tot_w = sum(weights) or 1.0

    # --- 分段：用 DP 对齐路名序列到轨迹分段 ---
    #
    # ⚠️ 不能等分：用户给的路名是骑行顺序，但每条路占的弧长比例
    # **不是 1/N**。实测 10 个路名等分后，「小木桥路」被分到的段
    # 离它 2855 米（而整条轨迹上离它只有 27 米）。
    #
    # DP 对齐在**初始位置**跑一次，得到每条路名对应的弧长区间；
    # 搜索时固定这个对应关系——平移量只有几公里，路名与轨迹段的
    # 对应关系不会变。
    seg_len = np.linalg.norm(np.diff(traj_merc, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = float(cum[-1])
    n_seg = len(usable)

    try:
        spans = _align_segments(traj_merc, usable, weights)
    except Exception:
        spans = [(total_len * i / n_seg, total_len * (i + 1) / n_seg)
                 for i in range(n_seg)]

    seg_samples: list[np.ndarray] = []
    for (lo, hi) in spans:
        m = (cum >= lo) & (cum <= hi)
        pts = traj_merc[m]
        if len(pts) < 2:
            j = int(np.argmin(np.abs(cum - lo)))
            pts = traj_merc[max(0, j - 1):j + 2]
        if len(pts) > 30:
            idx = np.linspace(0, len(pts) - 1, 30).astype(int)
            pts = pts[idx]
        seg_samples.append(pts)

    def objective(dx: float, dy: float) -> float:
        shift = np.array([dx, dy], dtype=float)
        val = 0.0
        for pts, c, w in zip(seg_samples, usable, weights):
            moved = pts + shift
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
        f"路名分段约束：{len(usable)} 条路（DP 对齐到轨迹的 {n_seg} 段），"
        f"目标值 {base:.0f} → {val:.0f} 米，平移 东{bx:+.0f} 北{by:+.0f} 米"
    )
    for c, w in zip(usable, weights):
        notes.append(
            f"  「{c.name}」{c.n_segments} 段 / {c.total_len_m/1000:.1f} km"
            f"，权重 {w:.2f}"
        )
    return np.array([bx, by], dtype=float), val, notes, base
