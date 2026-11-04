"""OSRM 路由客户端（OpenStreetMap 数据）。

为什么是 OSRM 而不是高德路径规划
--------------------------------
实测同一张城市级截图、同样的起点/终点/途经点：

| | 高德 walking | OSRM driving |
|---|---|---|
| 命中用户路名 | **1 / 14 步** | **15 / 51 步** |
| 途经点 | 静默丢弃（结果与不传完全一样）| **生效** |
| 命中的路 | 只有肇嘉浜路辅路 | 中山南二路、东安路、肇嘉浜路、虹桥路、水城路、虹古路 |

高德给的是「步行最短路」，走的是枫林路/建国西路/华山路/天山路——
与用户实际骑的路线几乎无关。OSRM 基于 OSM 图，路网更完整
（OSM 里有 `bicycle=yes` 的小区内部道路），而且途经点真的生效。

OSRM 与几何重建是**互补**的
----------------------------
* 几何重建（`app/shapefit.py`）：**逐段精确**——把每段贴到用户说的那条路上，
  逐段残差 7021 米；
* OSRM：**整体真实**——有正确的路口连接与绕行，形状更像真实骑行，
  但每一段的具体位置可能偏离用户说的那条路（逐段残差 31285 米）。

所以最优是**融合**：段内用路名几何，段间用 OSRM 的连接路径。
OSRM 返回的 `legs` 数组天然对应「起点→途经点1→…→终点」，
正好一一对应「路名 i 与路名 i+1 之间的连接」。

已知限制
--------
* 官方 Demo（`router.project-osrm.org`）**只有 driving profile**
  （bike/foot 会退化成 car），且是公共资源、有速率限制——
  所以结果必须**落盘缓存**，同一组坐标不重复请求；
* 本地自建需要 Docker + 区域 PBF，本项目环境不具备；
* OSRM 用 **WGS-84**，本项目内部用 **GCJ-02**，调用前后都要转换。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


@dataclass
class OsrmRoute:
    """一条 OSRM 路线。"""

    ok: bool = False
    distance_m: float = 0.0
    duration_s: float = 0.0
    # 各 leg 的几何（WGS-84 经纬度列表），leg[0] = 起点→途经点1
    legs: list[list[tuple[float, float]]] = field(default_factory=list)
    # 每条 leg 上的路名（来自 steps）
    leg_names: list[list[str]] = field(default_factory=list)
    # **顶层完整几何**（WGS-84 经纬度）——这才是可靠的：
    # OSRM 的 leg 级 geometry 字段可能缺失（实测缓存里是空的），
    # 而顶层 `route.geometry` 一直都在。
    geometry: list[tuple[float, float]] = field(default_factory=list)
    error: str = ""
    elapsed_ms: int = 0


class OsrmClient:
    """OSRM HTTP 客户端（带磁盘缓存）。"""

    def __init__(
        self,
        base_url: str = "https://router.project-osrm.org",
        profile: str = "driving",
        cache_dir: str = "data/cache",
        timeout: float = 25.0,
        enabled: bool = True,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.profile = profile or "driving"
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.enabled = enabled and bool(self.base_url)
        self.last_error = ""
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "meituan-ride/0.1 (+local)"})

    # ------------------------------------------------------------------
    def route(
        self,
        points: list[tuple[float, float]],
        via_names: list[str] | None = None,
    ) -> OsrmRoute:
        """规划路线。`points` 是 WGS-84 的 (lon, lat) 列表。"""
        res = OsrmRoute()
        if not self.enabled or len(points) < 2:
            res.error = "OSRM 未启用"
            return res
        key = self._cache_key(points)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        coord = ";".join("%.6f,%.6f" % (float(p[0]), float(p[1])) for p in points)
        url = (f"{self.base_url}/route/v1/{self.profile}/{coord}"
               "?overview=full&geometries=geojson&steps=true")
        t0 = time.time()
        try:
            r = self._session.get(url, timeout=self.timeout)
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}"
                self.last_error = res.error
                return res
            body = r.json()
        except Exception as exc:  # noqa: BLE001
            res.error = f"{type(exc).__name__}: {exc}"
            self.last_error = res.error
            return res
        if body.get("code") != "Ok" or not body.get("routes"):
            res.error = str(body.get("code") or "no route")
            self.last_error = res.error
            return res

        route = body["routes"][0]
        res.ok = True
        res.distance_m = float(route.get("distance") or 0.0)
        res.duration_s = float(route.get("duration") or 0.0)
        res.elapsed_ms = int((time.time() - t0) * 1000)
        # 顶层几何（可靠）
        top_geom = (route.get("geometry") or {}).get("coordinates") or []
        res.geometry = [(float(c[0]), float(c[1])) for c in top_geom]
        for leg in route.get("legs") or []:
            # leg 级几何**可能缺失**（实测如此），缺失就留空——
            # 需要整体几何时用 `res.geometry`（顶层，一直都在）。
            # ⚠️ 千万别在这里「兜底切分」：切分必须带累积偏移，
            # 否则每个 leg 都从几何开头切一段，累加后长度会暴涨
            # （实测 17553 米 -> 156377 米）。
            geom = (leg.get("geometry") or {}).get("coordinates") or []
            res.legs.append([(float(c[0]), float(c[1])) for c in geom])
            names: list[str] = []
            for st in leg.get("steps") or []:
                nm = st.get("name") or ""
                if nm and (not names or names[-1] != nm):
                    names.append(nm)
            res.leg_names.append(names)
        self._write_cache(key, res)
        return res

    # ------------------------------------------------------------------
    def _cache_key(self, points: list[tuple[float, float]]) -> str:
        """稳定的缓存键。

        ⚠️ **不能用内置 hash()**：Python 对 str 的 hash 受 PYTHONHASHSEED
        随机化影响，**每次进程启动都不一样**——缓存路径每次都变，
        缓存永远不命中，于是每次识别都重新请求公共 OSRM 服务器
        （实测端到端耗时从 150 秒涨到 528 秒）。
        """
        raw = ";".join("%.5f,%.5f" % (p[0], p[1]) for p in points)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / ("osrm_%s.json" % key)

    def _read_cache(self, key: str) -> OsrmRoute | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - float(d.get("ts", 0)) > 7 * 86400:
            return None
        r = OsrmRoute(ok=True)
        r.distance_m = float(d.get("distance", 0))
        r.duration_s = float(d.get("duration", 0))
        r.legs = [[(float(a), float(b)) for a, b in leg]
                  for leg in d.get("legs", [])]
        r.leg_names = [list(x) for x in d.get("leg_names", [])]
        r.geometry = [(float(a), float(b)) for a, b in d.get("geometry", [])]
        return r

    def _write_cache(self, key: str, r: OsrmRoute) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(json.dumps({
                "ts": time.time(),
                "distance": r.distance_m,
                "duration": r.duration_s,
                "legs": r.legs,
                "leg_names": r.leg_names,
                "geometry": r.geometry,
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def match_ratio(leg_names: list[str], user_names: list[str]) -> float:
    """路线上的路名与用户路名的命中率（按去重后的路名集合算）。"""
    if not leg_names or not user_names:
        return 0.0
    on_path = {n for n in leg_names if n}
    hit = 0
    for u in user_names:
        if any(u in n or n in u for n in on_path):
            hit += 1
    return hit / max(1, len(user_names))
