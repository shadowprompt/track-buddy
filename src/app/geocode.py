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
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import AppConfig
from .geo.coords import lonlat_to_mercator
from .lexicon_guard import WriteDecision, evaluate_write
from .maps.contracts import Capability
from .maps.geocoding import build_geocoding_registry, convert_candidates

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
    source: str  # amap / tmap / baidu / tianditu / osm / user / lexicon
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
        # 识别会话期间的「待裁决」自动写入；None 表示不在会话中。
        # 见 app/lexicon_guard.py 与 pipeline.run() 的外层包裹。
        self._capture: list[dict[str, Any]] | None = None
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
                    # 词典条目自带真实来源（amap / tianditu / osm / user ...）。
                    # 早期统一写成 "lexicon"，把「这些坐标其实来自不同基准」
                    # 这件事抹掉了，配准无法按源做一致性诊断。
                    source=str(e.get("source") or "lexicon"),
                    address=e.get("address", ""),
                    kind=infer_kind(name),
                )
            )
        return results

    def city_bounds(self, city: str, pad_deg: float = 0.35) -> str | None:
        """从词典中该城市已核验坐标派生搜索视野 minx,miny,maxx,maxy。

        不内置任何城市边界：换城市、换区域只要词典里有该 city 的条目
        就能自动收窄；条目不足 3 个时返回 None，交给调用方决定兜底。
        用 2%/98% 分位而不是 min/max，避免个别错条目把视野撑到外省。
        """
        if not city:
            return None
        pts: list[tuple[float, float]] = []
        with self._lock:
            for entries in self._data.values():
                for e in entries:
                    if (e.get("city") or "") != city:
                        continue
                    try:
                        pts.append((float(e["lon"]), float(e["lat"])))
                    except (KeyError, TypeError, ValueError):
                        continue
        if len(pts) < 3:
            return None
        # 以中位数为中心剔除跨省错条目（词典里可能个别点位标错城市）。
        # 3° 约 300 km，覆盖大城市群，又不会把北京的点算进上海。
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
        keep = [p for p in pts if abs(p[0] - cx) <= 3.0 and abs(p[1] - cy) <= 3.0]
        if len(keep) < 3:
            keep = pts
        return (f"{min(p[0] for p in keep) - pad_deg:.4f},"
                f"{min(p[1] for p in keep) - pad_deg:.4f},"
                f"{max(p[0] for p in keep) + pad_deg:.4f},"
                f"{max(p[1] for p in keep) + pad_deg:.4f}")

    def in_city_core(
        self, lon: float, lat: float, city: str,
        quantile: float = 0.95, factor: float = 1.6,
        min_radius_m: float = 20000.0,
    ) -> bool | None:
        """判断一个点是否落在该城市已核验点的**核心范围**。

        与 ``city_bounds`` 的区别：后者用 min/max 经纬度、外加较宽的
        padding，适合当「搜索视野」（宁可宽一点）；但个别标错城市的
        条目会把 min/max 撑到几百公里外，拿它做写回校验等于没校验。

        这里改用「每个已核验点到中位数中心的距离」的分位数派生半径：
        尾部错条目不影响 95% 分位，连锁品牌的异地同名门店（实测距中心
        88~130 km）会被挡住，而城市边缘的正常点位仍能沉淀。

        ``quantile`` / ``factor`` 是无量纲统计量，``min_radius_m`` 是物理
        下限，三者都与城市、截图、厂商无关。返回 ``None`` 表示词典数据
        不足、无法判断，调用方应放行以免误杀新城市冷启动。
        """
        if not city:
            return None
        pts: list[tuple[float, float]] = []
        with self._lock:
            for entries in self._data.values():
                for e in entries:
                    if (e.get("city") or "") != city:
                        continue
                    try:
                        pts.append((float(e["lon"]), float(e["lat"])))
                    except (KeyError, TypeError, ValueError):
                        continue
        if len(pts) < 8:
            return None
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
        keep = [p for p in pts if abs(p[0] - cx) <= 2.0 and abs(p[1] - cy) <= 2.0]
        if len(keep) < 8:
            return None
        x0, y0 = lonlat_to_mercator(cx, cy)
        dists: list[float] = []
        for lo, la in keep:
            x, y = lonlat_to_mercator(lo, la)
            dists.append(math.hypot(x - x0, y - y0))
        dists.sort()
        radius = max(min_radius_m,
                     dists[min(len(dists) - 1, int(len(dists) * quantile))] * factor)
        x, y = lonlat_to_mercator(lon, lat)
        return math.hypot(x - x0, y - y0) <= radius

    def core_radius_m(
        self, city: str, quantile: float = 0.90, factor: float = 1.5,
        min_radius_m: float = 15000.0,
    ) -> float | None:
        """该城市已核验点的**核心半径**（米），供派生搜索视野。

        与 ``in_city_core`` 同源：先用中位数中心剔除跨省错条目，
        再取「到中心距离」的分位数作为半径基准。

        为什么需要它：``city_bounds`` 用 min/max 派生视野，而词典里
        偶尔会有标错城市的条目——实测上海词典里混进了经度 113.1
        （山西）、纬度 22.5（广东）的点，min/max 被撑到
        119.63~122.27°E / 30.47~32.19°N，苏州、杭州全在视野内，
        于是查「外滩」会返回**昆山**的「外滩8号」、
        查「静安寺」会返回**常熟**的静安寺。

        这里返回 ``None`` 表示词典数据不足，调用方应放宽或放弃收窄。
        """
        if not city:
            return None
        pts: list[tuple[float, float]] = []
        with self._lock:
            for entries in self._data.values():
                for e in entries:
                    if (e.get("city") or "") != city:
                        continue
                    try:
                        pts.append((float(e["lon"]), float(e["lat"])))
                    except (KeyError, TypeError, ValueError):
                        continue
        if len(pts) < 8:
            return None
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
        keep = [p for p in pts if abs(p[0] - cx) <= 2.0 and abs(p[1] - cy) <= 2.0]
        if len(keep) < 8:
            return None
        x0, y0 = lonlat_to_mercator(cx, cy)
        dists: list[float] = []
        for lo, la in keep:
            x, y = lonlat_to_mercator(lo, la)
            dists.append(math.hypot(x - x0, y - y0))
        dists.sort()
        base = dists[min(len(dists) - 1, int(len(dists) * quantile))]
        return max(min_radius_m, base * factor)

    def core_center(self, city: str) -> tuple[float, float] | None:
        """该城市已核验点的**中位数中心**（经度, 纬度）。

        与 ``core_radius_m`` 同源，用于「以城市中心为准」的场景
        （例如天地图视野以中心 + 半径派生，而不是靠 min/max 撑开）。
        """
        if not city:
            return None
        pts: list[tuple[float, float]] = []
        with self._lock:
            for entries in self._data.values():
                for e in entries:
                    if (e.get("city") or "") != city:
                        continue
                    try:
                        pts.append((float(e["lon"]), float(e["lat"])))
                    except (KeyError, TypeError, ValueError):
                        continue
        if len(pts) < 8:
            return None
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
        keep = [p for p in pts if abs(p[0] - cx) <= 2.0 and abs(p[1] - cy) <= 2.0]
        if len(keep) < 8:
            return None
        return cx, cy

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

    # ---------- 词典写入防毒 ----------

    def begin_capture(self) -> None:
        """开启一次识别会话：自动写入先排队，等结果出来统一裁决。"""
        with self._lock:
            self._capture = []

    def discard_capture(self) -> int:
        """异常/中断时丢弃待写入项，返回丢弃数量。"""
        with self._lock:
            pending = len(self._capture or [])
            self._capture = None
        return pending

    def flush_capture(
        self,
        confidence: float | None,
        self_consistent: bool = True,
        input_ok: bool = True,
    ) -> dict[str, Any]:
        """用本轮最终结果裁决排队的自动写入；返回可审计的统计。"""
        with self._lock:
            pending = list(self._capture or [])
            self._capture = None
        report: dict[str, Any] = {
            "pending": len(pending),
            "written": 0,
            "rejected": 0,
            "reasons": {},
            "confidence": (round(float(confidence), 3)
                           if confidence is not None else None),
            "self_consistent": bool(self_consistent),
            "input_ok": bool(input_ok),
        }
        written = 0
        for item in pending:
            decision = self._commit(
                **item, confidence=confidence,
                self_consistent=self_consistent, input_ok=input_ok,
            )
            if decision.allowed:
                written += 1
            else:
                key = decision.reason
                report["reasons"][key] = report["reasons"].get(key, 0) + 1
        report["written"] = written
        report["rejected"] = len(pending) - written
        if written:
            self.save()
        return report

    def remember(
        self,
        name: str,
        lon: float,
        lat: float,
        city: str = "",
        source: str = "user",
        address: str = "",
        boost: float = 1.0,
        confidence: float | None = None,
        self_consistent: bool = True,
        input_ok: bool = True,
        verified: bool = False,
    ) -> WriteDecision:
        """记录一条地名坐标（由用户确认或在线查询得到）。

        自动来源（在线地理编码 / OSM 冷启动）在识别会话里只排队，
        由 ``flush_capture()`` 拿到最终置信度后统一裁决；用户显式
        纠正 ``source="user"`` 立即写入，不受自动门槛约束。
        """
        key = self.normalize(name)
        if not key:
            return WriteDecision(False, 0.0, "empty_name")
        item = {
            "key": key, "lon": lon, "lat": lat, "city": city,
            "source": source, "address": address, "boost": boost,
            "verified": verified,
        }
        with self._lock:
            capturing = self._capture is not None
            if capturing and not verified and source != "user":
                self._capture.append(item)
                return WriteDecision(True, float(boost), "queued")
        decision = self._commit(
            **item, confidence=confidence,
            self_consistent=self_consistent, input_ok=input_ok,
        )
        if decision.allowed:
            self.save()
        return decision

    def _commit(
        self,
        *,
        key: str,
        lon: float,
        lat: float,
        city: str,
        source: str,
        address: str,
        boost: float,
        verified: bool,
        confidence: float | None,
        self_consistent: bool,
        input_ok: bool,
    ) -> WriteDecision:
        """真正改内存中的词典（不落盘，交给调用方批量 save）。"""
        with self._lock:
            entries = self._data.setdefault(key, [])
            decision = evaluate_write(
                key=key, lon=lon, lat=lat, source=source, boost=boost,
                entries=entries, confidence=confidence,
                self_consistent=self_consistent, input_ok=input_ok,
                verified=verified,
            )
            if not decision.allowed:
                return decision
            is_user = verified or source == "user"
            for e in entries:
                try:
                    same = (abs(float(e["lon"]) - float(lon)) < 1e-6
                            and abs(float(e["lat"]) - float(lat)) < 1e-6)
                except (KeyError, TypeError, ValueError):
                    continue
                if not same:
                    continue
                old = float(e.get("weight", 1.0))
                if is_user:
                    weight = old + decision.boost
                elif e.get("verified"):
                    # 已核验点位不能被自动写入继续加权（也不能被封顶拉低）
                    weight = old
                else:
                    weight = old + decision.boost
                    if decision.weight_cap is not None:
                        weight = min(weight, decision.weight_cap)
                e["weight"] = weight
                e["count"] = int(e.get("count", 0)) + 1
                if city:
                    e["city"] = city
                if address:
                    e["address"] = address
                if is_user:
                    # 用户纠正把未核验条目升级为已核验，后续不再封顶
                    e["source"] = "user"
                    e["verified"] = True
                if confidence is not None:
                    e["last_confidence"] = round(float(confidence), 3)
                break
            else:
                e = {
                    "lon": float(lon),
                    "lat": float(lat),
                    "weight": (min(decision.boost, decision.weight_cap)
                               if decision.weight_cap is not None
                               else decision.boost),
                    "count": 1,
                    "city": city,
                    "source": source,
                    "address": address,
                    "verified": bool(is_user),
                }
                if confidence is not None:
                    e["last_confidence"] = round(float(confidence), 3)
                entries.append(e)
            return decision

    def stats(self) -> dict[str, Any]:
        with self._lock:
            points = [e for v in self._data.values() for e in v]
            return {
                "entries": len(self._data),
                "points": len(points),
                "verified_points": sum(1 for e in points if e.get("verified")),
                "auto_points": sum(1 for e in points if not e.get("verified")),
                "cities": sorted({e.get("city", "") for e in points if e.get("city")}),
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
        # 最近一次在线地理编码失败的**可读原因**（如 key 平台类型不匹配）。
        # 用于 /api/health 暴露配置问题——静默失败会让人误以为
        # 「在线服务不可用」，而实际是 key 用错了地方。
        self._last_error: str = ""
        # 本次识别周期内**发生过**的在线源失败（成功不清空）。
        # 与 ``_last_error`` 的区别：后者是「最近一次错误」，
        # 这个是「这一轮有没有失败过」——pipeline 要用它判断
        # 「结果是不是降级来的」。
        self._online_failures: list[str] = []
        # 厂商实现与调用顺序全部交给能力层；Geocoder 只保留薄适配方法。
        self.registry, self._impl = build_geocoding_registry(
            self.cfg, lambda: self._session, self.lexicon)

    # ---------- 在线 provider ----------

    def _throttle(self, provider: str, min_interval: float) -> None:
        last = self._last_call.get(provider, 0.0)
        wait = min_interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_call[provider] = time.time()

    def probe_online(self) -> dict[str, Any]:
        """主动探测在线地理编码是否真的可用。

        「有 key」不等于「key 能用」：高德 key 绑定平台类型，
        ``USERKEY_PLAT_NOMATCH``(10009) 表示 Web端(JS API) 的 key 被拿去
        调 Web服务接口；此外配额超限、签名未关、tk 无效也都会让接口失败。
        这里遍历能力层注册表真打一次，把结果原样报出来，避免「配错了」
        看起来像「服务正常」。
        """
        return self.registry.probe(Capability.GEOCODING)

    # ---------- 在线 provider 薄适配层 ----------
    #
    # 真正的 HTTP 请求、解析与 CRS 声明都在
    # ``app/maps/geocoding/providers.py``。这里保留同名方法有两个作用：
    #   1. 兼容既有调用与测试注入（替换 g._amap 即可 mock 某个源）；
    #   2. 把 provider 记录的可读错误同步到 Geocoder._last_error，
    #      供 /api/health 暴露配置问题。
    # 调用顺序本身由 ProviderRegistry 决定，见 geocode()。

    def _invoke(self, name: str, query: str, city: str) -> list[GeocodeResult]:
        impl = self._impl.get(name)
        if impl is None:
            return []
        try:
            items = impl.search(query, city)
        except Exception:  # noqa: BLE001 - 单个源失败不能打断回退链
            return []
        return convert_candidates(items, GeocodeResult)

    def _amap(self, name: str, city: str) -> list[GeocodeResult]:
        out = self._invoke("amap", name, city)
        # 高德的 key 平台类型 / 配额错误必须留下可读原因，不能静默吞掉。
        if self._impl["amap"].last_error:
            self._last_error = self._impl["amap"].last_error
        return out

    def _tmap(self, name: str, city: str) -> list[GeocodeResult]:
        return self._invoke("tmap", name, city)

    def _baidu(self, name: str, city: str) -> list[GeocodeResult]:
        return self._invoke("baidu", name, city)

    def _tianditu(self, name: str, city: str) -> list[GeocodeResult]:
        out = self._invoke("tianditu", name, city)
        if self._impl["tianditu"].last_error:
            self._last_error = self._impl["tianditu"].last_error
        return out

    # ---------- 对外接口 ----------

    def geocode(
        self, name: str, city: str = "", allow_online: bool = True,
        allow_cold_start: bool = True,
    ) -> list[GeocodeResult]:
        """地理编码一个地名，返回候选列表（GCJ-02）。

        优先级：用户确认 > 在线服务 > 本地词典 > OSM 冷启动。
        在线源顺序由 ``MAP_GEOCODING_ORDER`` 配置驱动，任一源返回空就落到下一源。

        词典种子来自 OSM/Nominatim，与美团底图同源的国内 POI 数据可差
        100-300 米，所以在线优先；未配密钥时仍可离线跑通，只是绝对精度
        受限。冷启动必须批量调用才可靠（见 app/coldstart.py），单点查询
        仅作兼容保留。
        """
        # 1) 用户确认过的坐标永远优先
        user = self.lexicon.lookup_by_source(name, city, ("user",))
        if user:
            return user

        # 2) 在线服务：**单源严格模式**
        #
        # ⚠️ **不再「高德失败就静默落天地图」**。
        #
        # 旧行为是按注册表顺序逐个试，任一源返回空就落下一个。用户看到的
        # 现象是「配了高德却在用天地图」却毫无提示——实测高德当日配额用尽
        # （USER_DAILY_QUERY_OVER_LIMIT），每个地名都先打高德再打天地图，
        # 既慢一倍（每个地名付两次请求）又让人误判实际用的是哪家。
        #
        # 现在：**只用一个源**——由 ``MAP_GEOCODING_ORDER`` 指定的第一个
        # 可用源（默认 amap）。它失败就把原因记进 ``_last_error``，
        # 交给上层明确提示，**不自动换源**。
        #
        # 想用多家兜底仍可显式声明：``MAP_GEOCODING_ORDER=amap,tianditu``，
        # 这时才是「依次尝试」语义（且失败原因同样会被记录）。
        if allow_online and self.cfg.allow_online:
            online = self._online_search(name, city)
            if online:
                # 数据飞轮只沉淀「同名且坐标落在本城已核验范围」的候选。
                for r in online[:3]:
                    if Lexicon.normalize(r.name or "") != Lexicon.normalize(name):
                        continue
                    if not self._in_city_bounds(r.lon, r.lat, city):
                        continue
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
        if allow_cold_start and allow_online and self.cfg.allow_online and city:
            cold = self._cold_start(name, city)
            if cold:
                return cold
        return []

    def _online_search(self, name: str, city: str) -> list[GeocodeResult]:
        """按配置调用在线地理编码源。

        **单源严格模式**：
        - 用户显式配了 ``MAP_GEOCODING_ORDER=tianditu`` → 只用它，
          失败就记录原因并返回空，**不自动换源**；
        - 用户显式配了多个（``amap,tianditu``）→ 才启用依次回退；
        - **没配** → 用默认表里**第一个可用**的源（当前是 amap），
          只用它一个；失败不换源。

        为什么不能默认四家依次回退：实测高德当日配额用尽
        （USER_DAILY_QUERY_OVER_LIMIT），每个地名都先打高德再打天地图，
        既慢一倍（每个地名付两次请求），又让人误判实际用的是哪家——
        现象是「配了高德却在用天地图」却毫无提示。

        这样「配了哪家就用哪家」成为可预期行为：
        用户看到高德配额提示，就知道结果是降级来的，
        而不是误以为天地图本来就是主源。
        """
        configured = self.registry.configured_order(Capability.GEOCODING)
        if configured:
            order = configured
            strict = len(configured) == 1
        else:
            # 未配置：取默认顺序里第一个「有 key」的，只用它
            order = self.registry.order(Capability.GEOCODING)
            usable = [
                n for n in order
                if (p := self.registry.get(n, Capability.GEOCODING)) is not None
                and p.is_available()
            ]
            if not usable:
                self._last_error = (
                    "未配置可用的在线地理编码源（需要 AMAP_KEY / "
                    "TMAP_KEY / BAIDU_MAP_KEY / TIANDITU_API_KEY 之一）"
                )
                return []
            order = (usable[0],)
            strict = True

        errors: list[str] = []
        misses: list[str] = []
        for pname in order:
            fn = getattr(self, "_" + pname, None)
            if fn is None:
                continue
            try:
                online = fn(name, city)
            except Exception as exc:  # noqa: BLE001 - 记原因后按模式决定是否继续
                online = []
                errors.append(f"{pname}: {type(exc).__name__}: {exc}")
            if online:
                return online
            # 记录这个源的失败原因
            impl = self._impl.get(pname)
            reason = getattr(impl, "last_error", "") if impl else ""
            if reason:
                errors.append(f"{pname}: {reason}")
            elif not any(e.startswith(pname + ":") for e in errors):
                # ⚠️ 「无匹配结果」**不是故障**，只是这个地名没查到。
                # OCR 残片（如「支工」「丰河」）本来就查不到，
                # 把它报成失败会让人以为服务商坏了。
                # 所以单独归类，不与真正的错误（配额/鉴权）混在一起。
                misses.append(pname)

        if errors:
            self._last_error = "；".join(errors[:3])
            # 记录「这一轮失败过」，供 pipeline 判断结果是否降级而来
            for e in errors:
                if e not in self._online_failures:
                    self._online_failures.append(e)
        elif misses:
            # 只是「这个地名没查到」（OCR 残片常见），**不是故障**。
            # 必须清掉上一次的错误，否则会把陈旧的配额提示带到现在——
            # 实测天地图模式下明明查得到东方明珠，却仍显示上一次的
            # 「高德配额超限」。
            self._last_error = ""
        # 单源模式：失败就返回空，让上层提示，不再换源
        return []

    def _in_city_bounds(self, lon: float, lat: float, city: str) -> bool:
        """写回词典前校验坐标落在该城市已核验点的核心范围。

        连锁品牌 / 高速的异地同名候选（查「红星美凯龙」返回苏州门店）
        与查询名完全相同，同名护栏挡不住；写进词典后会撑大搜索视野、
        并可能进入 RANSAC。判据用距离分位数而不是经纬度 min/max，否则
        个别标错城市的条目会让护栏形同虚设。词典数据不足时放行，
        避免新城市冷启动被误杀。
        """
        verdict = self.lexicon.in_city_core(lon, lat, city)
        return verdict is not False

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
