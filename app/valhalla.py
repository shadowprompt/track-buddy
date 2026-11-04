"""Valhalla 路由客户端（OSM 生态）。

## 为什么需要它

OSRM 官方 Demo（`router.project-osrm.org`）**只有 driving profile**——
实测传 `bike` / `foot` / `cycling` / `bicycle` 等任意 profile 名，
返回结果**完全一样**（11064 米、同样的路名序列），profile 参数被直接忽略。
原因是 Demo 服务器只 `.osrm` 编译了一份 driving 的图。

Valhalla 则**原生支持多种 costing**（bicycle / pedestrian / auto），
而且公共实例 `valhalla1.openstreetmap.de` 免费、无需 key。

## 实测对比（同一张城市级截图，同样的起点/终点/8 个途经点）

| 方案 | 里程 | 命中用户路名 | 顺序一致 | lat 反转 | 几何点 |
|---|---|---|---|---|---|
| **Valhalla pedestrian** | **12.63 km** | **7/10** | **100%** | 27 | 269 |
| Valhalla auto | 16.46 km | 7/10 | 100% | 23 | 306 |
| OSRM driving | 16.72 km | 7/10 | 100% | - | - |
| Valhalla bicycle | 17.26 km | 4/10 | 100% | 19 | 336 |
| 高德 walking | 9.68 km | 1/14 | - | 0 | - |

真实骑行约 11.2 公里。**Valhalla pedestrian 的里程最接近**（12.63 km，
偏差 13%），而 OSRM driving 偏了 49%。

命中顺序 100% 一致尤其重要：Valhalla **完全不知道用户填了哪些路名**，
它自己算出的路线依次经过「中山南二路 → 东安路 → 肇嘉浜路 → 虹桥路 →
水城路 → 虹古路 → 安龙路」——与用户记忆的顺序完全吻合。这是一份
**独立于本项目算法**的第三方印证。

## 注意

* `bicycle` costing 反而不如 `pedestrian`：上海的路网里自行车常被引到
  高架/快速路下面的机动车道，或因 `use_roads` 权重绕远。实测命中从 7/10
  掉到 4/10、里程从 12.63 涨到 17.26 公里。
* 公共实例是**共享资源**，必须落盘缓存 + 超时 + 命中率门槛。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

# Valhalla 公共实例（按顺序探测，第一个可用的就用）
DEFAULT_HOSTS: tuple[str, ...] = (
    "valhalla1.openstreetmap.de",
    "valhalla1.openstreetmap.de",
)

# 支持的 costing（Valhalla 原生）
COSTINGS: tuple[str, ...] = ("bicycle", "pedestrian", "auto")

# 公共实例的 location 数量上限。
# ⚠️ 超过就 HTTP 400 `error_code:150 Exceeded max locations: 10`。
MAX_LOCATIONS = 10


def _thin_locations(
    points: list[tuple[float, float]], limit: int,
) -> list[tuple[float, float]]:
    """把 location 列表均匀采样到 `limit` 个（首尾一定保留）。

    首尾是起点与终点，必须保留；中间均匀采样，尽量保住空间分布。
    """
    if len(points) <= limit:
        return list(points)
    n_mid = limit - 2
    if n_mid <= 0:
        return [points[0], points[-1]]
    idx = np.linspace(1, len(points) - 2, n_mid)
    out = [points[0]]
    for i in idx:
        out.append(points[int(round(i))])
    out.append(points[-1])
    # 去重（采样后可能出现重复）
    dedup: list[tuple[float, float]] = []
    for p in out:
        if not dedup or abs(p[0] - dedup[-1][0]) > 1e-7 \
                or abs(p[1] - dedup[-1][1]) > 1e-7:
            dedup.append(p)
    return dedup


def decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
    """解码 Valhalla 的 encoded polyline（precision 6）。

    Valhalla 的 `leg.shape` 默认是这种编码格式，不是 GeoJSON。
    编码规则与 Google 相同：ASCII 63 基准、5 位一组、低位在高位，
    每组最高位表示符号。
    """
    if not encoded:
        return []
    coords: list[tuple[float, float]] = []
    index = 0
    lat = 0.0
    lon = 0.0
    factor = float(10 ** precision)
    n = len(encoded)
    while index < n:
        for axis in range(2):
            shift = 0
            result = 0
            while True:
                if index >= n:
                    return coords
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if axis == 0:
                lat += delta
            else:
                lon += delta
        coords.append((lon / factor, lat / factor))
    return coords


@dataclass
class ValhallaRoute:
    """一次 Valhalla 路线查询的结果。"""

    ok: bool = False
    error: str = ""
    costing: str = ""
    distance_m: float = 0.0
    duration_s: float = 0.0
    # 路名序列（按机动顺序，即实际经过顺序）
    road_names: list[str] = field(default_factory=list)
    # 完整几何（WGS-84 经纬度）
    geometry: list[tuple[float, float]] = field(default_factory=list)
    # 各 leg 的几何
    legs: list[list[tuple[float, float]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "costing": self.costing,
            "distance_m": round(self.distance_m, 1),
            "duration_s": round(self.duration_s, 1),
            "road_names": self.road_names,
            "n_geometry": len(self.geometry),
            "n_legs": len(self.legs),
        }


class ValhallaClient:
    """Valhalla 公共实例客户端。

    ⚠️ 公共实例是共享资源：
      * 所有请求落盘缓存（键用 md5，不用内置 hash——后者受
        PYTHONHASHSEED 随机化影响，每次进程启动路径都不同，
        缓存永远不命中）；
      * 短超时（公共实例偶尔会慢）；
      * 调用方应设置命中率门槛，不达标就放弃结果。
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        timeout: float = 30.0,
        hosts: tuple[str, ...] = DEFAULT_HOSTS,
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.hosts = hosts
        self.enabled = enabled
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "meituan-ride/0.1 (+local)"})
        self._last_call: dict[str, float] = {}

    # ---------- 缓存 ----------

    def _cache_key(self, costing: str, points: list[tuple[float, float]]) -> str:
        raw = costing + "|" + ";".join("%.6f,%.6f" % p for p in points)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / ("valhalla_%s.json" % key)

    def _load_cache(self, key: str) -> ValhallaRoute | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        r = ValhallaRoute()
        r.ok = bool(d.get("ok"))
        r.error = str(d.get("error") or "")
        r.costing = str(d.get("costing") or "")
        r.distance_m = float(d.get("distance_m") or 0.0)
        r.duration_s = float(d.get("duration_s") or 0.0)
        r.road_names = list(d.get("road_names") or [])
        r.geometry = [(float(a), float(b)) for a, b in (d.get("geometry") or [])]
        r.legs = [
            [(float(a), float(b)) for a, b in leg]
            for leg in (d.get("legs") or [])
        ]
        return r

    def _store_cache(self, key: str, r: ValhallaRoute) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(json.dumps({
                "ok": r.ok,
                "error": r.error,
                "costing": r.costing,
                "distance_m": r.distance_m,
                "duration_s": r.duration_s,
                "road_names": r.road_names,
                "geometry": [list(p) for p in r.geometry],
                "legs": [[list(p) for p in leg] for leg in r.legs],
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # ---------- 请求 ----------

    def _throttle(self, key: str, min_interval: float = 0.35) -> None:
        last = self._last_call.get(key, 0.0)
        wait = min_interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_call[key] = time.time()

    def route(
        self,
        points: list[tuple[float, float]],
        costing: str = "pedestrian",
        language: str = "zh-CN",
    ) -> ValhallaRoute:
        """查询路线。`points` 为 WGS-84 经纬度列表。

        途经点**真的生效**（与高德 walking 不同——高德会静默丢弃
        偏离最优走廊的途经点）。

        ⚠️ **location 数量上限**：公共实例 `error_code:150
        Exceeded max locations: 10`——超过 10 个直接 HTTP 400。
        这里自动截断到 10 个（保留首尾，均匀采样中间的）。
        """
        if not self.enabled:
            return ValhallaRoute(ok=False, error="客户端未启用")
        if len(points) < 2:
            return ValhallaRoute(ok=False, error="至少需要 2 个点")
        if costing not in COSTINGS:
            return ValhallaRoute(ok=False, error=f"不支持的 costing: {costing}")
        if len(points) > MAX_LOCATIONS:
            points = _thin_locations(points, MAX_LOCATIONS)

        key = self._cache_key(costing, points)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        payload = {
            "locations": [{"lon": lo, "lat": la} for lo, la in points],
            "costing": costing,
            "directions_options": {
                "units": "kilometers",
                "language": language,
            },
        }
        last_err = ""
        for host in self.hosts:
            # ⚠️ **必须自己拼 URL 并 quote**，不能用 `requests` 的 `params=`。
            #
            # `params={"json": json.dumps(payload)}` 会被 requests 编码成
            # `%7B%22locations%22...`，Valhalla 直接返回
            # `400 Failed to parse json request`（实测）。
            # 手动 `urllib.parse.quote` 则是 HTTP 200。
            url = (
                f"https://{host}/route?json="
                + urllib.parse.quote(json.dumps(payload, ensure_ascii=False))
            )
            self._throttle(host)
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    last_err = f"HTTP {resp.status_code}"
                    continue
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_err = f"{type(exc).__name__}"
                continue

            if data.get("error"):
                last_err = json.dumps(data["error"], ensure_ascii=False)[:160]
                continue

            trip = data.get("trip") or {}
            summary = trip.get("summary") or {}
            r = ValhallaRoute(
                ok=True,
                costing=costing,
                distance_m=float(summary.get("length") or 0.0) * 1000.0,
                duration_s=float(summary.get("time") or 0.0),
            )
            for leg in trip.get("legs") or []:
                for m in leg.get("maneuvers") or []:
                    for nm in m.get("street_names") or []:
                        if nm:
                            r.road_names.append(nm)
                leg_pts = decode_polyline(leg.get("shape") or "")
                r.legs.append(leg_pts)
                r.geometry.extend(leg_pts)
            self._store_cache(key, r)
            return r

        r = ValhallaRoute(ok=False, error=last_err or "全部实例都失败")
        self._store_cache(key, r)
        return r


def order_consistency(
    road_names: list[str],
    user_names: list[str],
) -> tuple[float, list[str]]:
    """用户填的路名在路线里的**相对顺序**是否一致。

    返回 (一致率, 命中的路名按路线出现顺序排列)。

    ⚠️ 这比「命中数」更有价值：命中的路名**顺序也对**，说明用户
    记忆里的路线与真实路网结构吻合，而不只是「这几个名字都存在」。

    判定用「子串双向匹配」：`"中山南二路"` 能匹配 `"中山南二路"`，
    `"虹桥路"` 也能匹配 `"虹桥路1829弄"`。
    """
    seq: list[tuple[str, int]] = []
    for u in user_names:
        best_idx: int | None = None
        for i, rn in enumerate(road_names):
            if u in rn or rn in u:
                best_idx = i
                break
        if best_idx is not None:
            seq.append((u, best_idx))
    if len(seq) < 2:
        return 0.0, [u for u, _ in seq]
    ok = sum(1 for i in range(1, len(seq)) if seq[i][1] > seq[i - 1][1])
    return ok / (len(seq) - 1), [u for u, _ in seq]


def match_ratio(road_names: list[str], user_names: list[str]) -> float:
    """命中用户路名的比例。"""
    if not user_names:
        return 0.0
    hits = sum(
        1 for u in user_names
        if any(u in rn or rn in u for rn in road_names)
    )
    return hits / len(user_names)


def polyline_length(points: list[tuple[float, float]]) -> float:
    """折线长度（米），按 WGS-84 经纬度近似。"""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        dx = (points[i][0] - points[i - 1][0]) * 95160.0
        dy = (points[i][1] - points[i - 1][1]) * 111320.0
        total += math.hypot(dx, dy)
    return total
