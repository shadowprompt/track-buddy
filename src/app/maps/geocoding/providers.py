"""在线地理编码的厂商实现（能力①）。

每个 provider 只做三件事：

1. 声明 ``meta.output_crs`` —— 自己返回的是哪套坐标系；
2. 发一次请求；
3. 把返回项翻译成 ``PlaceCandidate``。

**坐标系换算不在 provider 里做**：``PlaceCandidate`` 的 ``lon/lat``
由 ``contracts.to_gcj02`` 按声明统一换算，调用方永远拿到 GCJ-02。
新增厂商 = 写一个类 + 注册，不需要改 Geocoder 的换算逻辑。

之所以保留 ``session_getter`` 回调而不是在构造时固定 session，是为了让
测试与运行时都能替换底层会话（``Geocoder._session`` 是可注入的）。
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Callable

import requests

from ...config import (
    AMAP_PLACE_SEARCH,
    TIANDITU_MAP_BOUND_DEFAULT,
    TIANDITU_SEARCH_URL,
    TMAP_PLACE_SEARCH,
)
from ..contracts import (
    CRS,
    Capability,
    GeoPoint,
    PlaceCandidate,
    ProbeResult,
    ProviderMeta,
)

BAIDU_PLACE_SEARCH = "https://api.map.baidu.com/place/v2/search"

# 归一化时会去掉的分隔符。
_NAME_SEPARATORS = " \t　()（）·・,，.。-—_/\\"


def _name_matches(query: str, title: str) -> bool:
    """查询词与 POI 标题是否「同名」（已归一化）。

    两级判据：

    1. **连续子串**：``query in title``。
       「东方明珠」⊂「东方明珠广播电视塔」✓
    2. **分段包含**：把标题按分隔符切开，查询词完整命中其中一段。
       「东方明珠-公交站」归一化成「东方明珠公交站」，
       它不是「东方明珠」的连续子串，但按原始分隔符切开后
       第一段就是「东方明珠」✓

    为什么需要第 2 条：地图上最常见的标注恰恰带分隔符
    （「XX-公交站」「XX公园东南门」「XX(地铁站)」），
    只用连续子串会把它们全漏掉。
    """
    if not query or not title:
        return False
    if query in title:
        return True
    # 用**原始**标题按分隔符切段，段内再归一化
    raw = title
    for ch in _NAME_SEPARATORS:
        raw = raw.replace(ch, "\x00")
    for seg in raw.split("\x00"):
        if seg and query in _strip_sep(seg):
            return True
    return False


def _strip_sep(s: str) -> str:
    for ch in _NAME_SEPARATORS:
        s = s.replace(ch, "")
    return s


class _BaseGeocodingProvider:
    """四个在线源共用的骨架：凭证、限流、会话、错误原因。"""

    name = ""
    output_crs: CRS = CRS.GCJ02
    credential_aliases: tuple[str, ...] = ()
    min_interval = 0.35

    def __init__(
        self,
        cfg: Any,
        credentials: Any,
        session_getter: Callable[[], Any],
        lexicon: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.credentials = credentials
        self._session_getter = session_getter
        self.lexicon = lexicon
        self.last_error = ""
        self._last_call = 0.0
        self.meta = ProviderMeta(
            name=self.name,
            capability=Capability.GEOCODING,
            output_crs=self.output_crs,
            credential_aliases=self.credential_aliases,
        )

    @property
    def session(self) -> Any:
        return self._session_getter()

    @property
    def key(self) -> str | None:
        return self.credentials.resolve(*self.credential_aliases)

    def is_available(self) -> bool:
        return bool(self.key)

    def _reset_error(self) -> None:
        """清掉**上一次**调用的错误。

        ⚠️ 必须每次查询前都清：`last_error` 是「最近一次错误」，
        而成功路径不会写它（只写失败路径）。不清的话会出现
        「上一次配额超限、这次成功」却仍报故障——
        实测天地图模式下明明查得到「东方明珠」，
        health 里却仍显示上一次的「高德配额超限」。
        """
        self.last_error = ""

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _get(self, url: str, params: dict[str, Any]) -> Any:
        r = self.session.get(url, params=params, timeout=self.cfg.http_timeout)
        return r.json()

    def _kind(self, title: str, category: str) -> str:
        # 锚点类型判定属于管线侧语义，延迟导入避免与 app.geocode 形成环。
        from ...geocode import infer_kind

        return infer_kind(title, category)

    def _candidate(
        self, *, name: str, lon: float, lat: float, quality: float,
        address: str, category: str,
    ) -> PlaceCandidate:
        return PlaceCandidate(
            name=name,
            point=GeoPoint(lon, lat, self.output_crs),
            quality=quality,
            source=self.name,
            address=address or "",
            kind=self._kind(name, category),
        )

    def search(self, name: str, city: str) -> list[PlaceCandidate]:
        raise NotImplementedError

    def probe(self) -> ProbeResult:
        return ProbeResult(True)


class AmapProvider(_BaseGeocodingProvider):
    """高德 POI 搜索：与美团底图同源，输出 GCJ-02。"""

    name = "amap"
    output_crs = CRS.GCJ02
    credential_aliases = ("AMAP_KEY",)

    def search(self, name: str, city: str) -> list[PlaceCandidate]:
        key = self.key
        if not key:
            return []
        self._reset_error()
        self._throttle()
        params = {
            "key": key, "keywords": name, "city": city or "",
            "citylimit": "false", "offset": "10", "page": "1",
            "extensions": "base",
        }
        try:
            data = self._get(AMAP_PLACE_SEARCH, params)
        except (requests.RequestException, ValueError):
            return []
        if data.get("status") != "1":
            info = str(data.get("info") or "")
            # 不要静默吞掉 key 的配置错误：USERKEY_PLAT_NOMATCH(10009)
            # 表示 key 是「Web端(JS API)」类型，本项目需要「Web服务」类型。
            if info:
                self.last_error = f"高德 {info}"
                if "USERKEY_PLAT_NOMATCH" in info:
                    self.last_error = (
                        "高德 key 平台类型不匹配（10009）：该 key 是"
                        "「Web端(JS API)」类型，本项目需要「Web服务」类型。"
                        "请到高德控制台为应用添加一个 Web服务 key，"
                        "并写进 .env 的 AMAP_KEY"
                    )
            return []
        out: list[PlaceCandidate] = []
        for p in (data.get("pois") or [])[:8]:
            loc = p.get("location", "")
            if not loc:
                continue
            lon, lat = (float(x) for x in loc.split(","))
            title = p.get("name", name)
            exact = 1.0 if name in (title or "") else 0.7
            out.append(self._candidate(
                name=title, lon=lon, lat=lat, quality=exact,
                address=p.get("address", ""), category=p.get("type", ""),
            ))
        return out

    def probe(self) -> ProbeResult:
        params = {"key": self.key or "", "keywords": "上海站",
                  "city": "上海", "citylimit": "true", "extensions": "base"}
        try:
            data = self._get(AMAP_PLACE_SEARCH, params)
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(False, error=f"{type(exc).__name__}: {exc}")
        if str(data.get("status")) == "1":
            pois = data.get("pois") or []
            return ProbeResult(True, hits=len(pois),
                               sample=pois[0].get("name") if pois else "")
        info = str(data.get("info") or "")
        hint = ""
        if "USERKEY_PLAT_NOMATCH" in info:
            hint = ("key 平台类型不匹配：该 key 是「Web端(JS API)」类型，"
                    "本项目需要「Web服务」类型——请到高德控制台为同一应用"
                    "添加一个 Web服务 key")
        elif "INVALID_USER_KEY" in info:
            hint = "key 无效或已被删除"
        elif "INVALID_USER_SCODE" in info or "INVALID_USER_SIGNATURE" in info:
            hint = "key 开启了数字签名，需要按官方算法生成 signature 参数"
        elif "DAILY_QUERY_OVER_LIMIT" in info or "OVER_LIMIT" in info:
            hint = "配额已用尽（当天调用次数超限）"
        self.last_error = f"高德 {info}" if info else ""
        return ProbeResult(False, error=info or "未知错误", hint=hint)


class TmapProvider(_BaseGeocodingProvider):
    """腾讯地图 POI 搜索：输出 GCJ-02。"""

    name = "tmap"
    output_crs = CRS.GCJ02
    credential_aliases = ("TMAP_KEY",)

    def search(self, name: str, city: str) -> list[PlaceCandidate]:
        key = self.key
        if not key:
            return []
        self._reset_error()
        self._throttle()
        params = {"key": key, "keyword": name, "page_size": "10",
                  "page_index": "1", "output": "json"}
        if city:
            params["boundary"] = f"region({city},0)"
        try:
            data = self._get(TMAP_PLACE_SEARCH, params)
        except (requests.RequestException, ValueError):
            return []
        if data.get("status") != 0:
            return []
        out: list[PlaceCandidate] = []
        for p in (data.get("data") or [])[:8]:
            loc = p.get("location") or {}
            lon, lat = loc.get("lng"), loc.get("lat")
            if lon is None or lat is None:
                continue
            title = p.get("title", name)
            exact = 1.0 if name in (title or "") else 0.7
            out.append(self._candidate(
                name=title, lon=float(lon), lat=float(lat), quality=exact,
                address=p.get("address", ""), category=p.get("type", ""),
            ))
        return out

    def probe(self) -> ProbeResult:
        params = {"key": self.key or "", "keyword": "上海站",
                  "page_size": "5", "page_index": "1", "output": "json"}
        try:
            data = self._get(TMAP_PLACE_SEARCH, params)
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(False, error=f"{type(exc).__name__}: {exc}")
        if int(data.get("status", -1)) == 0:
            return ProbeResult(True, hits=len(data.get("data") or []))
        return ProbeResult(False, error=str(data.get("message") or "未知错误"))


class BaiduProvider(_BaseGeocodingProvider):
    """百度 POI 搜索：输出 BD-09，由契约层换算到 GCJ-02。"""

    name = "baidu"
    output_crs = CRS.BD09
    credential_aliases = ("BAIDU_MAP_KEY",)

    def search(self, name: str, city: str) -> list[PlaceCandidate]:
        key = self.key
        if not key:
            return []
        self._reset_error()
        self._throttle()
        params = {"key": key, "query": name, "region": city or "全国",
                  "output": "json", "scope": "2"}
        try:
            data = self._get(BAIDU_PLACE_SEARCH, params)
        except (requests.RequestException, ValueError):
            return []
        if data.get("status") != 0:
            return []
        out: list[PlaceCandidate] = []
        for p in (data.get("results") or [])[:8]:
            loc = p.get("location") or {}
            lon, lat = loc.get("lng"), loc.get("lat")
            if lon is None or lat is None:
                continue
            out.append(self._candidate(
                name=p.get("name", name), lon=float(lon), lat=float(lat),
                quality=0.85, address=p.get("address", ""),
                category=p.get("type", ""),
            ))
        return out

    def probe(self) -> ProbeResult:
        params = {"key": self.key or "", "query": "上海站", "region": "全国",
                  "output": "json", "scope": "2"}
        try:
            data = self._get(BAIDU_PLACE_SEARCH, params)
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(False, error=f"{type(exc).__name__}: {exc}")
        if int(data.get("status", -1)) == 0:
            return ProbeResult(True, hits=len(data.get("results") or []))
        return ProbeResult(False, error=str(data.get("message") or "未知错误"))


def _anchor_spread_m(cands: list) -> float:
    """候选锚点（GCJ-02）之间的粗略最大跨度（米）。

    只用前 6 个（排序后最贴近查询词的），足够判断「是否泛指 / 残片」。
    """
    pts = [c.point.gcj02 for c in cands[:6] if c.point is not None]
    if len(pts) < 2:
        return 0.0
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    mlat = sum(lats) / len(lats)
    dlat = (max(lats) - min(lats)) * 111320.0
    dlon = (max(lons) - min(lons)) * 111320.0 * math.cos(math.radians(mlat))
    return math.hypot(dlat, dlon)


def _drop_spatially_ambiguous(
    cands: list, max_span_m: float = 5000.0
) -> list:
    """同名候选地理跨度过大（> ``max_span_m``）则整组弃用。

    跨度大说明该名字是泛指或 OCR 残片，多个候选落在相距很远的位置，
    无法确定截图上到底是哪一个。弃用后 RANSAC 仍可依靠其它无歧义锚点
    配准，避免单个错配把尺度撑爆。阈值 5km 远大于同址子建筑间距（<1km）。
    """
    if len(cands) < 2:
        return cands
    if _anchor_spread_m(cands) > max_span_m:
        return []
    return cands


class TiandituProvider(_BaseGeocodingProvider):
    """天地图「地名搜索 V2.0」：输出 CGCS2000，由契约层换算到 GCJ-02。

    官方格式是 ``/v2/search?postStr={...}&type=query&tk=...``，返回
    ``resultType=1`` 时 ``pois[].lonlat`` 是 "x,y" 字符串。早期实现误用
    地理编码接口的 ``ds={"keyword":...}``、按 ``result.items`` 解析，
    实测 HTTP 400，等于该源从未可用。

    CGCS2000 与 WGS-84 在工程精度上等价，与项目内部 GCJ-02 差约 570 m，
    必须换算一次；这里只声明坐标系，换算交给 ``to_gcj02``。
    """

    name = "tianditu"
    output_crs = CRS.CGCS2000
    credential_aliases = ("TIANDITU_API_KEY",)

    def _bounds(self, city: str) -> tuple[str, bool]:
        """派生搜索视野（minx,miny,maxx,maxy）与是否「已收窄」。

        ⚠️ **不能用 ``Lexicon.city_bounds``**（min/max + 0.35° padding）：
        词典里偶尔有标错城市的条目，实测上海词典混进了经度 113.1
        （山西）、纬度 22.5（广东）的点，min/max 被撑到
        119.63~122.27°E / 30.47~32.19°N——苏州、昆山、杭州全在视野内，
        于是查「外滩」返回**昆山**的「外滩8号」、
        查「静安寺」返回**常熟**的静安寺。

        改成「中位数中心 + 核心半径（距离分位数派生）」：
        尾部错条目不影响分位数，城市边缘的正常点位仍能命中。
        """
        custom = self.cfg.tianditu_map_bound_custom
        if custom:
            return custom, True
        if self.lexicon is not None and city:
            center = self.lexicon.core_center(city)
            radius = self.lexicon.core_radius_m(city)
            if center and radius:
                cx, cy = center
                # 半径换算成经纬度增量：纬度方向直接除，方向按 cos 修正
                dlat = radius / 111320.0
                dlon = radius / max(1.0, 111320.0 * math.cos(math.radians(cy)))
                bound = (f"{cx - dlon:.4f},{cy - dlat:.4f},"
                         f"{cx + dlon:.4f},{cy + dlat:.4f}")
                return bound, True
        derived = self.lexicon.city_bounds(city) if self.lexicon else None
        if derived:
            return derived, True
        return TIANDITU_MAP_BOUND_DEFAULT, False

    def search(self, name: str, city: str) -> list[PlaceCandidate]:
        from ...geocode import Lexicon  # 延迟导入，避免包初始化环

        key = self.key
        if not key:
            return []
        self._reset_error()
        self._throttle()
        bound_str, narrow = self._bounds(city)
        try:
            bx = [float(x) for x in bound_str.split(",")]
        except ValueError:
            bx = []
        post = {
            "keyWord": name, "queryType": 1, "level": 12,
            "mapBound": bound_str, "start": 0, "count": 10, "show": 2,
        }
        params = {"postStr": json.dumps(post, ensure_ascii=False),
                  "type": "query", "tk": key}
        try:
            data = self._get(TIANDITU_SEARCH_URL, params)
        except (requests.RequestException, ValueError) as exc:
            self.last_error = f"天地图 {type(exc).__name__}"
            return []
        status = data.get("status") or {}
        code = status.get("infocode") if isinstance(status, dict) else None
        if code is not None and str(code) != "1000":
            self.last_error = f"天地图 {status.get('cndesc') or '未知错误'}"
            return []
        if str(data.get("resultType")) != "1":
            return []
        out: list[PlaceCandidate] = []
        for p in (data.get("pois") or [])[:8]:
            ll = str(p.get("lonlat") or "").replace("，", ",").split(",")
            if len(ll) < 2:
                continue
            try:
                wlon, wlat = float(ll[0]), float(ll[1])
            except ValueError:
                continue
            # 全国视野会返回异地同名 POI；服务端 mapBound 未必严格过滤，
            # 客户端按坐标再校一次（换算后的 GCJ-02 坐标）。
            glon, glat = GeoPoint(wlon, wlat, self.output_crs).gcj02
            address = p.get("address") or ""
            title = p.get("name") or name
            if narrow and len(bx) == 4 and not (
                bx[0] <= glon <= bx[2] and bx[1] <= glat <= bx[3]
            ):
                continue
            if not narrow and city and address and city not in address:
                # ⚠️ 地址字段**不一定含城市名**：公交站、地铁站、门、入口
                # 这类标注的 address 常常就是 POI 名本身
                # （「东方明珠-公交站」的 address 是「东方明珠」）。
                # 旧实现一刀切按「地址含城市名」过滤，把这些全丢了。
                #
                # 改成：地址含城市名 → 放行；地址不含城市名时，
                # 只要**标题本身就是我们要找的那个名字**，也放行
                # （同名候选由后续的 RANSAC 几何一致性挑）。
                #
                # 注意判据是「标题 vs 查询词」，不是「地址 vs 城市名」。
                if not _name_matches(Lexicon.normalize(name), title):
                    continue
            # ⚠️ **同名判据必须是「包含关系」，不能要求完全相等**。
            #
            # 天地图的搜索是「包含关键词」语义，查「东方明珠」返回的是
            # 「东方明珠广播电视塔」「东方明珠-公交站」「东方明珠公园东南门」
            # 等**更长的名字**，一条标题恰好等于查询词的都没有。
            # 早期实现要求 `normalize(title) == normalize(name)`，
            # 结果这些全部被丢掉 —— 接口明明返回了 10 个坐标正确的 POI，
            # 解析后剩 0 个，这个源等于从未可用。
            #
            # 判据分两级：
            #   1) 归一化后**连续子串**（「东方明珠广播电视塔」含「东方明珠」）；
            #   2) 归一化后**分段包含**（「东方明珠-公交站」归一化成
            #      「东方明珠公交站」，「东方明珠」不是它的连续子串，
            #      但它是其中一个**词段**）。
            # 只用第 1 条会漏掉带分隔符的 POI（公交站、门、入口等），
            # 而这些恰恰是地图上最常见的标注。
            #
            # OCR 残片（名字被截断）会命中几十公里外的同名泛类 POI，
            # 但那些候选最终还要过 RANSAC 的几何一致性校验，由它来挑。
            q = Lexicon.normalize(name)
            t = Lexicon.normalize(title)
            if not q or not t:
                continue
            # ⚠️ 分段判据要用**原始标题**（分隔符还在），
            # 归一化后的标题已经把分隔符去掉了，段就切不出来了。
            if t != q and not _name_matches(q, title):
                continue
            # 贴进度：完全相等 > 前缀包含 > 任意包含。
            # 反映在 quality 上（越小越贴近），最终按 quality 排序。
            if t == q:
                quality = 0.9
            elif t.startswith(q):
                quality = 0.8
            else:
                quality = 0.7
            # ⚠️ 「旧址 / 旧址 / 原址 / 老院区」是**不同物理位置**的候选：
            # 当前地图截图只会标注现址，若把它当锚点会把尺度/配准撑爆
            # （同仁医院旧址长治路 vs 新址仙霞路相差 ~10km，正是 18km
            # 拉伸的元凶之一）。直接剔除，不参与配准；东门/西门/住院部
            # 等「同址子建筑」不受影响。
            if re.search(r"旧址|旧址|原址|老院区", title):
                continue
            out.append(self._candidate(
                name=title, lon=wlon, lat=wlat, quality=quality,
                address=address,
                category=p.get("typeName") or p.get("typeCode") or "",
            ))
        # 贴近查询词的排前面；同一贴进度下按标题长度短优先
        # （「东方明珠」比「东方明珠广播电视塔」更可能是用户要找的）
        out.sort(key=lambda c: (-c.quality, len(c.name)))
        if city and out:
            out.sort(key=lambda r: (0 if city in r.address else 1, -r.quality))
        # ⚠️ 同名候选地理分散 → 名字泛指 / OCR 残片，无法定位到具体哪一个，
        # 若当作锚点，错配会把尺度 / 配准撑爆
        # （「医学院附属」匹配到仁济 / 瑞金 / 新华等一堆相距 ~10km 的分院，
        # 取第一个就错 10km，正是 18km 拉伸的元凶）。跨度 >5km（远大于
        # 同址子建筑 <1km 间距）视为歧义，整组弃用——缺失一个锚点远好过
        # 一个错 10km 的锚点毁掉整次配准。
        return _drop_spatially_ambiguous(out)

    def probe(self) -> ProbeResult:
        post = {"keyWord": "北京大学", "queryType": 1, "level": 12,
                "mapBound": TIANDITU_MAP_BOUND_DEFAULT, "start": 0, "count": 3}
        params = {"postStr": json.dumps(post, ensure_ascii=False),
                  "type": "query", "tk": self.key or ""}
        try:
            data = self._get(TIANDITU_SEARCH_URL, params)
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(False, error=f"{type(exc).__name__}: {exc}")
        status = data.get("status") or {}
        code = status.get("infocode") if isinstance(status, dict) else None
        pois = data.get("pois") or []
        if pois and (code is None or str(code) == "1000"):
            return ProbeResult(True, hits=len(pois),
                               sample=pois[0].get("name") or "")
        info = (status.get("cndesc") if isinstance(status, dict)
                else str(status)) or "未知错误"
        self.last_error = f"天地图 {info}"
        hint = ""
        if str(code) in ("2001", "2002", "2003", "2004"):
            hint = "postStr 参数格式有误"
        elif str(code) in ("1001", "1002") or "key" in str(info).lower():
            hint = "检查 tk 是否有效、控制台是否放行了当前 IP / 域名"
        return ProbeResult(False, error=str(info), hint=hint)


PROVIDER_CLASSES = (
    AmapProvider, TmapProvider, BaiduProvider, TiandituProvider,
)
