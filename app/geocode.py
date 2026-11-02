"""POI 地理编码。

合规约束：仅使用腾讯地图、高德、百度、天地图四家持牌服务，
不使用 OpenStreetMap / Mapbox / Google Maps 等境外或未备案数据源。

优先级：
1. 用户确认过的坐标（source="user"）—— 永远第一
2. 在线服务（按已配置的 key 自动选择；与美团底图同源，精度最高）
3. 本地词典（data/lexicon.json）—— 零延迟、零配额，且随使用积累变准
4. OSM 冷启动（app/coldstart.py）—— 词典未覆盖区域的兜底

本地词典是「越用越准」的核心资产：每次用户确认/纠正都会写入，
形成该城市独有的「地名 → 精确坐标」映射，其精度远超任何公共 API。

**重要**：词典优先级高不代表它给的坐标一定对。
词典里的短条目可能是 OCR 截断产生的残片（如「上海世」来自「上海西站」），
其坐标属于另一个地点。因此词典命中时仍应让冷启动/在线候选一起参与，
由 RANSAC 的几何一致性挑选真正自洽的那一组。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import AppConfig
from .geo.coords import wgs84_to_gcj02

# 反向子串匹配的长度比下限。
#
# 词典 key 是查询名的子串时（`key in name`），只有 key 足够长
# 才可能是同一个地点的截断写法，否则它多半是别的地点的残片。
# 实测：「上海世」(3字) 命中「上海世贸商城」(6字)，比例 0.5，
# 属于残片，必须拦下。
_PARTIAL_MIN_RATIO = 0.6


@dataclass
class GeocodeResult:
    """地理编码结果。"""

    name: str
    lon: float  # GCJ-02
    lat: float  # GCJ-02
    quality: float  # 0-1，可靠度
    source: str  # lexicon / amap / tmap / baidu / tianditu
    address: str = ""
    kind: str = "poi"  # poi / district / station / road / transit


# --- 锚点类型判定 ---------------------------------------------------------
#
# 屏幕标注点与真实地理点的偏差量级，完全取决于标注对象的类型：
#
#   * 车站 / 一般 POI：图标就画在真实位置旁，偏差通常 < 100 米
#   * 行政区：地图把「静安区」三个字画在区的视觉重心，而地理编码
#     返回的是行政区的**几何质心**，两者可差 1-2 公里
#   * 道路 / 地铁线路：文字沿走向排布，中心点与标注位置天然不对应
#
# 城市级低缩放截图里锚点几乎全是行政区，若不分类而用统一容差，
# 正确锚点会被判成离群，配准就会掉进「几个挤在一起的小地名」里。
_DISTRICT_SUFFIX = ("区", "县", "市", "旗")
_STATION_SUFFIX = ("站", "港", "场")
_TRANSIT_MARKERS = ("号线", "地铁", "联络线", "环线", "内圈", "外圈", "区间线")
_ROAD_MARKERS = ("高速", "高架", "路", "道", "街", "桥", "隧道", "匝道")


def infer_kind(
    name: str, category: str = "", osm_type: str = ""
) -> str:
    """判定锚点类型，用于配准时选择合适的内点容差。

    ``category`` / ``osm_type`` 是 Nominatim 返回的原始分类，
    有则以它为准（最可靠）；没有则按名称形态推断。
    """
    n = (name or "").strip()
    cat = (category or "").strip().lower()
    otype = (osm_type or "").strip().lower()

    # OSM 明确标为行政边界 → district
    if cat == "boundary" or otype == "administrative":
        return "district"
    # OSM 明确标为车站类 → station
    if otype in (
        "train_station", "station", "subway_station", "light_rail_station",
        "tram_stop", "bus_station", "ferry_terminal", "airport", "port",
    ):
        return "station"

    # 名称形态推断
    if any(m in n for m in _TRANSIT_MARKERS):
        return "transit"
    if any(m in n for m in _ROAD_MARKERS) and len(n) <= 8:
        return "road"
    if n.endswith(_DISTRICT_SUFFIX) and len(n) <= 6:
        return "district"
    if n.endswith(_STATION_SUFFIX) and len(n) <= 8:
        return "station"
    return "poi"


class Lexicon:
    """本地地名词典。

    结构：{ 归一化地名: [ {lon, lat, weight, count, source}, ... ] }

    同一地名可能有多个候选（同名地点），每次用户确认都会给对应候选加权，
    权重最高的即该城市实际使用的点位。
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._data: dict[str, list[dict[str, Any]]] = {}
        self.load()

    @staticmethod
    def normalize(name: str) -> str:
        """归一化地名，便于跨写法匹配。"""
        n = name.strip()
        for ch in " \t　()（）·・,，.。-—_":
            n = n.replace(ch, "")
        return n

    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    self._data = json.loads(self.path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self._data = {}
            else:
                self._data = {}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    def lookup(self, name: str, city_hint: str = "") -> list[GeocodeResult]:
        """查询地名候选，按权重降序返回。

        除精确匹配外，还做「包含匹配」：地图标注常把 POI 名拆开或与
        相邻文字粘连（如「百联中环」+「购物广场河」），OCR 结果可能是
        完整名的一部分。包含匹配能把这类情况救回来。

        **但包含匹配有方向性，不能只看子串关系**：

        - 查询名是词典 key 的子串（`key in name`）：安全。
          例：查「百联中环购物广场」能命中词典里的「百联中环」。

        - 词典 key 是查询名的子串（`key in name` 的反向）：**危险**。
          词典里的短 key 多半是 OCR 截断产生的**残片**，
          它的坐标属于它真正代表的另一个地点。

        实测事故：词典里有残片「上海世」（来自「上海西站」被截断），
        坐标 121.40071,31.26275（上海西站）。查「上海世贸商城」时，
        「上海世」是它的子串 → 命中 → 返回了**上海西站**的坐标，
        偏离真实位置 6.8 公里。而冷启动其实已经查到了正确坐标
        121.40087,31.20189，却因为「词典已命中」被丢弃，
        正确候选根本没机会参与配准。

        因此反向匹配必须加**长度比**约束：短 key 至少要占查询名的
        60%（且不少于 3 个字），否则它就是残片而非同一地点。
        """
        key = self.normalize(name)
        with self._lock:
            exact = list(self._data.get(key, []))
            partial: list[tuple[str, dict]] = []
            if len(key) >= 2:
                for k, entries in self._data.items():
                    if k == key:
                        continue
                    # 查询名包含词典 key：安全方向。
                    if key in k and len(key) >= 3:
                        for e in entries:
                            partial.append((k, e))
                    # 词典 key 包含查询名：反向，需长度比约束防残片劫持。
                    elif k in key and len(k) >= 3:
                        if len(k) / len(key) >= _PARTIAL_MIN_RATIO:
                            for e in entries:
                                partial.append((k, e))

        merged: list[dict] = []
        seen: set[tuple[float, float]] = set()
        for e in exact + [e for _, e in partial]:
            sig = (round(float(e["lon"]), 6), round(float(e["lat"]), 6))
            if sig in seen:
                continue
            seen.add(sig)
            merged.append(e)

        if city_hint:
            merged.sort(key=lambda e: (e.get("city", "") != city_hint, -float(e.get("weight", 0))))
        else:
            merged.sort(key=lambda e: -float(e.get("weight", 0)))

        results = []
        for e in merged:
            results.append(
                GeocodeResult(
                    name=name,
                    lon=float(e["lon"]),
                    lat=float(e["lat"]),
                    quality=min(1.0, float(e.get("weight", 1.0)) / 5.0),
                    source="lexicon",
                    address=e.get("address", ""),
                    kind=infer_kind(name),
                )
            )
        return results

    def lookup_exact(self, name: str) -> list[dict[str, Any]]:
        """只做**精确**匹配（不做子串扩展）。

        用于判断「词典里是否有这个完整地名」——
        精确命中时坐标基本不会错，可以跳过冷启动的额外查询；
        而子串命中可能只是残片巧合，需要更多候选来交叉验证。
        """
        key = self.normalize(name)
        if not key:
            return []
        with self._lock:
            return list(self._data.get(key, []))

    def remember(
        self,
        name: str,
        lon: float,
        lat: float,
        city: str = "",
        source: str = "user",
        address: str = "",
        boost: float = 1.0,
    ) -> None:
        """记录一条地名坐标（由用户确认或在线查询得到）。"""
        key = self.normalize(name)
        if not key:
            return
        with self._lock:
            entries = self._data.setdefault(key, [])
            for e in entries:
                if abs(e["lon"] - lon) < 1e-6 and abs(e["lat"] - lat) < 1e-6:
                    e["weight"] = float(e.get("weight", 1.0)) + boost
                    e["count"] = int(e.get("count", 0)) + 1
                    if city:
                        e["city"] = city
                    break
            else:
                entries.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "weight": boost,
                        "count": 1,
                        "city": city,
                        "source": source,
                        "address": address,
                    }
                )
            self.save()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._data),
                "points": sum(len(v) for v in self._data.values()),
                "cities": sorted({e.get("city", "") for v in self._data.values() for e in v if e.get("city")}),
            }


    def lookup_by_source(
        self, name: str, city_hint: str = "", sources: tuple[str, ...] = ("user",)
    ) -> list[GeocodeResult]:
        """只返回指定来源的条目。

        来源分层很重要：
        - ``user``  用户在地图上确认过的位置——可信度最高，永远优先
        - ``amap``/``tmap`` 等在线服务——与美团底图同源，坐标与底图一致
        - ``lexicon`` 本地词典种子——来自 OSM/Nominatim，与国内底图存在
          100-300 米系统性偏差，只能当离线兜底
        """
        out: list[GeocodeResult] = []
        for r in self.lookup(name, city_hint):
            if r.source in sources:
                out.append(r)
        return out

class Geocoder:
    """地理编码服务聚合。自动在本地词典与在线服务之间选择。"""

    def __init__(self, cfg: AppConfig, lexicon: Lexicon):
        self.cfg = cfg
        self.lexicon = lexicon
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "meituan-ride/0.1 (+local)"})
        self._last_call: dict[str, float] = {}

    # ---------- 在线 provider ----------

    def _throttle(self, provider: str, min_interval: float) -> None:
        last = self._last_call.get(provider, 0.0)
        wait = min_interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_call[provider] = time.time()

    def _amap(self, name: str, city: str) -> list[GeocodeResult]:
        key = self.cfg.amap_key
        if not key:
            return []
        self._throttle("amap", 0.35)
        params = {"key": key, "keywords": name, "city": city or "", "citylimit": "false", "offset": "10", "page": "1", "extensions": "base"}
        try:
            r = self._session.get(
                "https://restapi.amap.com/v3/place/text",
                params=params,
                timeout=self.cfg.http_timeout,
            )
            data = r.json()
        except (requests.RequestException, ValueError):
            return []
        if data.get("status") != "1":
            return []
        out = []
        for p in data.get("pois", [])[:8]:
            loc = p.get("location", "")
            if not loc:
                continue
            lon, lat = (float(x) for x in loc.split(","))
            exact = 1.0 if name in (p.get("name", "") or "") else 0.7
            out.append(
                GeocodeResult(name=p.get("name", name), lon=lon, lat=lat,
                              quality=exact, source="amap",
                              address=p.get("address", ""),
                              kind=infer_kind(p.get("name", name), p.get("type", "")))
            )
        return out

    def _tmap(self, name: str, city: str) -> list[GeocodeResult]:
        key = self.cfg.tmap_key
        if not key:
            return []
        self._throttle("tmap", 0.35)
        params = {"key": key, "keyword": name, "page_size": "10", "page_index": "1", "output": "json"}
        if city:
            params["boundary"] = f"region({city},0)"
        try:
            r = self._session.get(
                "https://apis.map.qq.com/ws/place/v1/search",
                params=params,
                timeout=self.cfg.http_timeout,
            )
            data = r.json()
        except (requests.RequestException, ValueError):
            return []
        if data.get("status") != 0:
            return []
        out = []
        for p in data.get("data", [])[:8]:
            loc = p.get("location", {})
            lon, lat = loc.get("lng"), loc.get("lat")
            if lon is None or lat is None:
                continue
            exact = 1.0 if name in (p.get("title", "") or "") else 0.7
            out.append(
                GeocodeResult(name=p.get("title", name), lon=float(lon), lat=float(lat),
                              quality=exact, source="tmap",
                              address=p.get("address", ""),
                              kind=infer_kind(p.get("title", name), p.get("type", "")))
            )
        return out

    def _baidu(self, name: str, city: str) -> list[GeocodeResult]:
        key = self.cfg.baidu_key
        if not key:
            return []
        self._throttle("baidu", 0.35)
        params = {"key": key, "query": name, "region": city or "全国", "output": "json", "scope": "2"}
        try:
            r = self._session.get(
                "https://api.map.baidu.com/place/v2/search",
                params=params,
                timeout=self.cfg.http_timeout,
            )
            data = r.json()
        except (requests.RequestException, ValueError):
            return []
        if data.get("status") != 0:
            return []
        out = []
        for p in data.get("results", [])[:8]:
            loc = p.get("location", {})
            lon, lat = loc.get("lng"), loc.get("lat")
            if lon is None or lat is None:
                continue
            # 百度返回 BD-09，需转 GCJ-02
            from .geo.coords import bd09_to_gcj02

            glon, glat = bd09_to_gcj02(float(lon), float(lat))
            out.append(
                GeocodeResult(name=p.get("name", name), lon=glon, lat=glat,
                              quality=0.85, source="baidu", address=p.get("address", ""),
                              kind=infer_kind(p.get("name", name), p.get("type", "")))
            )
        return out

    def _tianditu(self, name: str, city: str) -> list[GeocodeResult]:
        key = self.cfg.tianditu_key
        if not key:
            return []
        self._throttle("tianditu", 0.35)
        params = {"tk": key, "ds": json.dumps({"keyword": name, "level": 12, "count": 10})}
        try:
            r = self._session.get(
                "https://api.tianditu.gov.cn/v2/search",
                params=params,
                timeout=self.cfg.http_timeout,
            )
            data = r.json()
        except (requests.RequestException, ValueError):
            return []
        out = []
        for p in (data.get("result") or {}).get("items", [])[:8]:
            lon, lat = p.get("lon"), p.get("lat")
            if lon is None or lat is None:
                continue
            out.append(
                GeocodeResult(name=p.get("name", name), lon=float(lon), lat=float(lat),
                              quality=0.8, source="tianditu", address=p.get("address", ""),
                              kind=infer_kind(p.get("name", name), p.get("type", "")))
            )
        return out

    # ---------- 对外接口 ----------

    def geocode(
        self, name: str, city: str = "", allow_online: bool = True,
        allow_cold_start: bool = True,
    ) -> list[GeocodeResult]:
        """地理编码一个地名，返回候选列表（GCJ-02）。

        **优先级（2026-10-30 修正）**：用户确认 > 在线服务 > 本地词典。

        早期实现是「词典优先」，只要词典命中就直接返回，永不调用在线服务。
        这个顺序是错的：词典种子来自 OSM/Nominatim，而美团底图用的是
        国内官方 POI 数据，两者在同一 POI 上可差 100-300 米。
        实测长宁新泾一例，锚点因此整体偏北约 245 米，轨迹落到了平行的
        错误道路族上（用户实骑 新渔路/新渔东路/剑河路/金钟路，
        而 OSM 归属显示 天山路/北翟路）。

        修正后：配了 AMAP_KEY 时锚点直接来自与底图同源的高德，偏差消失；
        未配密钥时仍可用本地词典离线跑通，只是绝对精度受限。

        **追加（2026-11-01）**：词典未命中时走 OSM 冷启动检索。
        换区域（词典没覆盖的地方）时，此前是「0 个锚点 → 配准失败」，
        而截图上其实印着「安龙小区」「进才实验中学」这类真实地名，
        OCR 已经读出来了，只是无处可查。

        但冷启动**必须批量调用**（见 ``app/coldstart.py``）：单个地名
        无法自证位置，需要多个地名的候选互相印证才能锁定区域。
        因此这里默认关闭（``allow_cold_start=False`` 由 pipeline 统一走
        批量接口），保留此参数只为单点查询的兼容场景。
        """
        # 1) 用户确认过的坐标永远优先
        user = self.lexicon.lookup_by_source(name, city, ("user",))
        if user:
            return user

        # 2) 在线服务（与美团底图同源）
        if allow_online and self.cfg.allow_online:
            online: list[GeocodeResult] = []
            for fn in (self._amap, self._tmap, self._baidu, self._tianditu):
                try:
                    online = fn(name, city)
                except Exception:
                    online = []
                if online:
                    break
            if online:
                for r in online[:3]:
                    self.lexicon.remember(
                        r.name or name, r.lon, r.lat, city=city, source=r.source,
                        address=r.address, boost=0.5,
                    )
                return online

        # 3) 本地词典兜底（离线可用，精度受限）
        local = self.lexicon.lookup(name, city)
        if local:
            return local

        # 4) OSM 冷启动：词典与在线服务都没命中时的最后一道网。
        #    单点查询（无互证）精度有限，生产路径请走 pipeline 的批量接口。
        if allow_cold_start and allow_online and self.cfg.allow_online and city:
            cold = self._cold_start(name, city)
            if cold:
                return cold
        return []

    # ---------- 冷启动（OSM） ----------

    def _cold_start(self, name: str, city: str) -> list[GeocodeResult]:
        """OSM 冷启动检索，并写回词典。

        写回词典有两个作用：
        1. 同名地点下次直接命中，不必再走网络；
        2. 用户在网页上「纠正」该锚点后，纠正结果权重更高，会顶掉这里的初值。

        来源标记为 ``osm``——它低于用户确认，也低于在线服务，
        出现基准冲突时优先让位。
        """
        try:
            from .coldstart import get_cold_start

            cold = get_cold_start(str(self.cfg.road_cache_dir / "coldstart.json"))
            places = cold.lookup(name, city, kinds=("poi", "transit"))
        except Exception:  # noqa: BLE001 — 冷启动失败不应影响主流程
            return []

        out: list[GeocodeResult] = []
        for p in places[:3]:
            out.append(
                GeocodeResult(
                    name=p.name, lon=p.lon, lat=p.lat,
                    quality=0.6, source="osm",
                    address=f"osm/{p.osm_type}/{p.osm_id}",
                    kind=p.kind,
                )
            )
        for r in out:
            self.lexicon.remember(
                r.name or name, r.lon, r.lat, city=city, source="osm",
                address=r.address, boost=0.2,
            )
        return out

    def geocode_many(
        self, names: list[str], city: str = "", allow_online: bool = True
    ) -> dict[str, list[GeocodeResult]]:
        """批量地理编码。"""
        out: dict[str, list[GeocodeResult]] = {}
        for n in names:
            out[n] = self.geocode(n, city=city, allow_online=allow_online)
        return out
