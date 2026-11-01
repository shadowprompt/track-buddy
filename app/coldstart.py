"""冷启动地理编码：用 OSM/Nominatim 为「词典里没有的地名」找坐标。

为什么需要它
------------
识别能力的第一道门槛是「知道这些地名在哪」。此前这一环只有两条路：

1. 本地词典 ``data/lexicon.json``——只覆盖已核验过的 38 个上海地名；
2. 腾讯/高德/百度/天地图的在线地理编码——需要用户自己申请密钥。

于是换一个区域（比如长宁剑河路一带）就完全无法识别：
OCR 明明读出了「安龙小区」「长宁实验小学」这些真实地名，
却因为「本地没这个词、也没配密钥」而拿不到任何坐标，
配准直接失败（实测新截图 A：0 个锚点命中）。

本模块补上第三条路：**按城市限定范围去 OSM 检索地名**。
它与项目既有做法完全一致——``app/seed_lexicon.py`` 的初始词典种子
本来就是「viewbox 限定的地理编码查询」得到的（并转 GCJ-02）。

合规说明
--------
本模块**只用于底图与地理编码的坐标参考**，不用于底图渲染，
与 ``app/roadfit.py`` 用 OSM 路网做几何精修是同一性质：
提供客观地理几何/坐标，不提供地图画面。
页面底图仍是腾讯地图；在线地理编码优先级仍高于本模块。

精度说明
--------
OSM 的 POI 点位与国内底图（美团/高德/腾讯）同源 POI 可能有 100-300 米偏差。
这正是 ``app/roadfit.py``「路网约束精修」存在的原因：
它用道路几何（各底图必须一致）纠正 POI 锚点带来的整体偏移。
所以本模块的产出被标记为 ``osm`` 源，质量权重低于用户确认与在线服务，
并最终由路网精修兜住绝对位置。
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .geo.coords import wgs84_to_gcj02

NOMINATIM = "https://nominatim.openstreetmap.org/search"

# Nominatim 使用政策要求最多 1 次/秒。必须遵守，
# 否则会被封 IP——这个项目是单机自用工具，封了就彻底没法冷启动了。
_MIN_INTERVAL = 1.05

# 中文行政区的常见后缀，用于从自由文本里识别城市名
_CITY_SUFFIX = re.compile(r"(北京|上海|天津|重庆|[一-龥]{2,4}?)(市|自治州|地区|盟)$")


@dataclass
class OsmPlace:
    """OSM 检索到的一个地点。"""

    name: str
    lon: float  # GCJ-02
    lat: float  # GCJ-02
    kind: str  # poi / transit / other
    osm_type: str = ""
    osm_id: int = 0

    @property
    def is_poi(self) -> bool:
        return self.kind == "poi"


def normalize_city(city: str) -> str:
    """把用户输入的城市规整成检索用的名字。

    「上海」与「上海市」在 Nominatim 里都���得到，但补全后更稳；
    常见的「北京/上海/天津/重庆」四个直辖市尤其需要补「市」。
    """
    c = (city or "").strip()
    if not c:
        return ""
    if c.endswith(("市", "区", "县")):
        return c
    if c in ("北京", "上海", "天津", "重庆"):
        return c + "市"
    return c


class ColdStartGeocoder:
    """按城市限定范围检索 OSM 地名，返回候选坐标。

    带进程内缓存与磁盘缓存，避免同一地名反复打网络。
    """

    def __init__(self, cache_path: str | None = None):
        self.cache_path = cache_path
        self._mem: dict[str, list[OsmPlace]] = {}
        self._lock = threading.RLock()
        self._last_call = 0.0
        self._bbox_cache: dict[str, tuple[float, float, float, float]] = {}
        if cache_path:
            self._load_disk()

    # ---------- 缓存 ----------

    def _load_disk(self) -> None:
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        for key, items in raw.items():
            self._mem[key] = [
                OsmPlace(
                    name=p["name"], lon=p["lon"], lat=p["lat"],
                    kind=p.get("kind", "poi"),
                    osm_type=p.get("osm_type", ""), osm_id=p.get("osm_id", 0),
                )
                for p in items
            ]

    def _save_disk(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        k: [
                            {"name": p.name, "lon": p.lon, "lat": p.lat,
                             "kind": p.kind, "osm_type": p.osm_type,
                             "osm_id": p.osm_id}
                            for p in v
                        ]
                        for k, v in self._mem.items()
                    },
                    f, ensure_ascii=False,
                )
        except OSError:
            pass

    # ---------- 网络 ----------

    def _throttle(self) -> None:
        with self._lock:
            wait = _MIN_INTERVAL - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _request(self, params: dict[str, Any]) -> list[dict]:
        url = NOMINATIM + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": "meituan-ride/0.1 (self-hosted local tool)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — 网络失败不应让整次识别崩掉
            return []

    def city_bbox(self, city: str) -> tuple[float, float, float, float] | None:
        """取城市的经纬度范围（WGS-84），用于限定检索、排除同名异地。

        Nominatim 返回的 boundingbox 顺序是 [south, north, west, east]。
        """
        cname = normalize_city(city)
        if not cname:
            return None
        with self._lock:
            if cname in self._bbox_cache:
                return self._bbox_cache[cname]
        self._throttle()
        data = self._request({
            "q": cname, "format": "jsonv2", "limit": 1,
            "countrycodes": "cn", "accept-language": "zh-CN",
        })
        if not data:
            return None
        b = data[0].get("boundingbox")
        if not b or len(b) != 4:
            return None
        try:
            south, north, west, east = (float(x) for x in b)
        except (TypeError, ValueError):
            return None
        bbox = (west, south, east, north)
        with self._lock:
            self._bbox_cache[cname] = bbox
        return bbox

    def lookup(
        self, name: str, city: str = "", kinds: tuple[str, ...] = ("poi",)
    ) -> list[OsmPlace]:
        """检索地名，返回候选（GCJ-02 坐标）。

        只接受**完全同名**的结果——地名被截断（「上海市职业教育开放实训中心」
        在图上只标了「上海市职业」）时宁可无结果，也不要给一个错的名字。
        宁可少一个锚点，也不要一个错锚点把整条轨迹带偏。
        """
        key = f"{normalize_city(city)}|{name}"
        with self._lock:
            hit = self._mem.get(key)
        if hit is not None:
            return [p for p in hit if p.kind in kinds]

        bbox = self.city_bbox(city)
        params: dict[str, Any] = {
            "q": f"{name} {normalize_city(city)}".strip(),
            "format": "jsonv2",
            "limit": 8,
            "countrycodes": "cn",
            "accept-language": "zh-CN",
        }
        if bbox:
            # viewbox 限定在城市范围内，避免匹配到同名异地地点
            params["viewbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        self._throttle()
        data = self._request(params)

        out: list[OsmPlace] = []
        for d in data:
            raw_name = (d.get("name") or "").strip()
            if raw_name != name:
                continue
            try:
                lon = float(d["lon"])
                lat = float(d["lat"])
            except (KeyError, TypeError, ValueError):
                continue
            osm_type = d.get("osm_type") or ""
            cat = (d.get("category") or "").strip()
            osm_class = (d.get("type") or "").strip()
            kind = "poi"
            if cat in ("highway", "waterway") or osm_class in ("road",):
                # 道路/水系：文字沿走向排布，中心不对应固定点位，不可作锚点
                kind = "other"
            elif cat in ("public_transport", "railway") or "地铁" in raw_name \
                    or "号线" in raw_name or "站" in raw_name:
                kind = "transit"
            glon, glat = wgs84_to_gcj02(lon, lat)
            out.append(
                OsmPlace(
                    name=raw_name, lon=glon, lat=glat, kind=kind,
                    osm_type=osm_type,
                    osm_id=int(d.get("osm_id") or 0),
                )
            )

        with self._lock:
            self._mem[key] = out
        self._save_disk()
        return [p for p in out if p.kind in kinds]


# 进程级单例：磁盘缓存路径由调用方注入
_COLD: ColdStartGeocoder | None = None
_COLD_LOCK = threading.Lock()


def get_cold_start(cache_path: str | None = None) -> ColdStartGeocoder:
    global _COLD
    with _COLD_LOCK:
        if _COLD is None:
            _COLD = ColdStartGeocoder(cache_path)
        return _COLD
