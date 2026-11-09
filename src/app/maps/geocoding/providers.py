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
    credential_aliases = ("TIANDITU_KEY", "TIANDITU_API_KEY", "TIANDITU_TK")

    def _bounds(self, city: str) -> tuple[str, bool]:
        custom = self.cfg.tianditu_map_bound_custom
        if custom:
            return custom, True
        derived = self.lexicon.city_bounds(city) if self.lexicon else None
        if derived:
            return derived, True
        return TIANDITU_MAP_BOUND_DEFAULT, False

    def search(self, name: str, city: str) -> list[PlaceCandidate]:
        from ...geocode import Lexicon  # 延迟导入，避免包初始化环

        key = self.key
        if not key:
            return []
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
            if narrow and len(bx) == 4 and not (
                bx[0] <= glon <= bx[2] and bx[1] <= glat <= bx[3]
            ):
                continue
            if not narrow and city and address and city not in address:
                continue
            title = p.get("name") or name
            # 普通搜索是「包含关键词」语义；OCR 残片会命中几十公里外的
            # 同名泛类 POI，只保留同名候选，查不到就交给下游兜底。
            if Lexicon.normalize(title) != Lexicon.normalize(name):
                continue
            out.append(self._candidate(
                name=title, lon=wlon, lat=wlat, quality=0.9,
                address=address,
                category=p.get("typeName") or p.get("typeCode") or "",
            ))
        if city and out:
            out.sort(key=lambda r: (0 if city in r.address else 1, -r.quality))
        return out

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
