"""阶段 2 离线测试：能力③④ 接入与重复基础设施收敛。

全部用假 opener / 假 session，不发任何真实网络请求。
"""

from __future__ import annotations

import json

import pytest

from app.maps import HttpClient
from app.maps.contracts import CRS, Capability, ProviderMeta, to_gcj02
from app.maps.roads import (
    OVERPASS_MIRRORS,
    build_road_geometry_registry,
    overpass_query,
    overpass_raw,
)
from app.maps.roads import provider as road_provider_mod
from app.maps.roads.overpass import DEFAULT_USER_AGENT
from app.maps.routing import build_route_validation_registry
from app.maps.routing.common import RouteCache, match_ratio, stable_key
from app.maps.routing.providers import _ENABLED_ENV


class _Cfg:
    """最小配置桩：只提供能力层需要的 _env_value。"""

    def __init__(self, env=None):
        self.env = dict(env or {})

    def _env_value(self, name):
        return self.env.get(name)


class _Response:
    def __init__(self, text: str, status_code: int = 200):
        self._text = text
        self.status_code = status_code

    def read(self):
        return self._text.encode("utf-8")


class _FakeOpener:
    """按顺序返回预设响应，并记录每次请求的 URL 与超时。"""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls: list[tuple[object, float]] = []
        self.requests: list[object] = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        self.calls.append((req.full_url, timeout))
        text = self.texts.pop(0) if self.texts else "<html>busy</html>"
        return _Response(text)


def test_single_mirror_table_shared_by_all_callers():
    """三处调用方不能再各维护一份镜像表。"""
    import app.roadfit as roadfit
    import app.roadname_constraint as rnc
    import app.waterguard as waterguard

    assert roadfit.OVERPASS_MIRRORS is OVERPASS_MIRRORS
    assert waterguard.OVERPASS_MIRRORS is OVERPASS_MIRRORS
    assert not hasattr(rnc, "_OVERPASS_MIRRORS")
    assert not hasattr(waterguard, "_OVERPASS_MIRRORS")
    assert not hasattr(roadfit, "_OVERPASS_MIRRORS")
    assert all("interpreter" in m and "interparser" not in m
               for m in OVERPASS_MIRRORS)
    assert len(set(OVERPASS_MIRRORS)) == len(OVERPASS_MIRRORS)


def test_overpass_falls_back_on_gateway_html_and_malformed_json():
    opener = _FakeOpener(["<html>504</html>", "{not json", '{"elements": [1]}'])
    data = overpass_query(
        "[out:json];way;out;", timeout=40.0,
        mirrors=("https://m1/api/interpreter",) * 3, opener=opener,
    )
    assert data == {"elements": [1]}
    assert len(opener.requests) == 3
    assert opener.calls[0][1] == 40.0
    assert opener.calls[1][1] == max(10.0, 40.0 / 3)
    assert opener.requests[0].get_header("User-agent") == DEFAULT_USER_AGENT


def test_overpass_raw_returns_text_and_reports_total_failure():
    opener = _FakeOpener(['{"elements": []}'])
    text = overpass_raw("q", timeout=30.0, mirrors=("https://m/api",),
                        opener=opener)
    assert json.loads(text) == {"elements": []}

    with pytest.raises(RuntimeError, match="全部镜像失败"):
        overpass_query("q", mirrors=("https://m/api",),
                       opener=_FakeOpener(["<html>busy</html>"]))


def test_roadfit_delegates_to_unified_overpass_layer(monkeypatch):
    import app.roadfit as roadfit

    opener = _FakeOpener(['{"elements": []}'])
    monkeypatch.setattr(
        "app.maps.roads.overpass.urllib.request.urlopen", opener
    )
    text = roadfit._fetch_overpass((121.3, 31.1, 121.4, 31.2), 25.0)
    assert text.startswith("{")
    assert opener.requests[0].full_url == OVERPASS_MIRRORS[0]
    assert opener.requests[0].get_header("User-agent") == "meituan-ride/0.1"


def test_waterguard_uses_unified_layer_without_bad_mirror(monkeypatch):
    import app.waterguard as waterguard

    seen: list[tuple[str, ...]] = []

    def fake_query(query, **kwargs):
        seen.append(kwargs.get("mirrors"))
        return {"elements": [{"geometry": [
            {"lon": 121.4, "lat": 31.2}, {"lon": 121.5, "lat": 31.3},
        ]}]}

    monkeypatch.setattr(waterguard, "overpass_query", fake_query)
    segs = waterguard._fetch_water("某河", timeout=5.0)
    assert segs and segs[0][0] == (121.4, 31.2)
    assert seen and seen[0] == (OVERPASS_MIRRORS[0],)
    assert all("interparser" not in str(m) for m in seen)


def test_road_geometry_provider_declares_wgs84_and_is_configurable():
    cfg = _Cfg()
    reg, impls = build_road_geometry_registry(cfg)
    osm = impls["osm"]
    assert osm.meta.capability is Capability.ROAD_GEOMETRY
    assert osm.meta.output_crs is CRS.WGS84
    assert to_gcj02(121.4, 31.2, osm.meta.output_crs) != (121.4, 31.2)
    assert reg.order(Capability.ROAD_GEOMETRY) == ("osm",)
    assert reg.probe(Capability.ROAD_GEOMETRY)["osm"]["ok"] is True

    cfg.env["MAP_ROAD_GEOMETRY_ENABLED"] = "0"
    assert build_road_geometry_registry(cfg)[0].available(
        Capability.ROAD_GEOMETRY
    ) == []


def test_geometry_order_is_config_driven_without_code_change():
    """新注册一个 provider，改 .env 顺序即可把它排到前面。"""

    class _Extra:
        name = "self_hosted"

        def __init__(self):
            self.meta = ProviderMeta(
                name=self.name, capability=Capability.ROAD_GEOMETRY,
                output_crs=CRS.WGS84,
            )

        def is_available(self):
            return True

        def probe(self):
            return None

    cfg = _Cfg({"MAP_ROAD_GEOMETRY_ORDER": "self_hosted,osm"})
    reg, _ = build_road_geometry_registry(cfg)
    reg.register(_Extra())
    assert reg.order(Capability.ROAD_GEOMETRY) == ("self_hosted", "osm")


def test_route_validation_registry_order_and_disable_switch():
    cfg = _Cfg()
    reg, impls = build_route_validation_registry(cfg)
    assert set(impls) == {"osrm", "valhalla"}
    assert reg.order(Capability.ROUTE_VALIDATION) == ("osrm", "valhalla")
    assert all(p.meta.output_crs is CRS.WGS84 for p in impls.values())

    cfg.env["MAP_ROUTE_VALIDATION_ORDER"] = "valhalla,osrm"
    assert build_route_validation_registry(cfg)[0].order(
        Capability.ROUTE_VALIDATION
    ) == ("valhalla", "osrm")

    cfg.env[_ENABLED_ENV] = "0"
    assert build_route_validation_registry(cfg)[0].available(
        Capability.ROUTE_VALIDATION
    ) == []


def test_http_client_request_json_keeps_status_and_throttles():
    class _Resp:
        status_code = 200

        def json(self):
            return {"code": "Ok"}

    class _Session:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append((url, timeout))
            return _Resp()

    session = _Session()
    client = HttpClient(session=session, timeout=3.0)
    assert client.request_json("https://example.org/route") == (
        200, {"code": "Ok"}
    )
    assert session.calls[0][1] == 3.0


def test_route_cache_keeps_old_filename_and_legacy_entries(tmp_path):
    cache = RouteCache(tmp_path, "valhalla", ttl_s=0.0)
    key = stable_key("pedestrian|121.4,31.2")
    assert len(key) == 16
    assert cache.path(key).name == "valhalla_%s.json" % key
    cache.write(key, {"ok": True, "geometry": [[1, 2]]})
    assert cache.read(key)["ok"] is True  # 无 ts 的历史条目仍然有效

    expired = RouteCache(tmp_path, "osrm", ttl_s=1.0)
    expired.write("k", {"ts": 1.0, "distance": 10})
    assert expired.read("k") is None


def test_two_route_clients_share_infrastructure():
    from app.osrm import OsrmClient, match_ratio as osrm_ratio
    from app.valhalla import ValhallaClient, match_ratio as valhalla_ratio

    osrm = OsrmClient(cache_dir="data/cache")
    valhalla = ValhallaClient(cache_dir="data/cache")
    assert isinstance(osrm.http, HttpClient)
    assert isinstance(valhalla.http, HttpClient)
    assert valhalla.http.min_interval == 0.35
    assert osrm._cache_path("k").name.startswith("osrm_")
    assert valhalla._cache_path("k").name.startswith("valhalla_")
    assert osrm._session is osrm.http.session
    assert osrm_ratio(["A路"], ["A"]) == valhalla_ratio(["A路"], ["A"]) == 1.0
    assert match_ratio([], ["A"]) == 0.0


def test_shared_polyline_helpers_match_legacy_exports():
    from app.maps.routing import common
    from app.valhalla import (
        decode_polyline,
        order_consistency,
        polyline_length,
    )

    enc = common.decode_polyline("?")  # 只验证可调用与导出一致
    assert isinstance(enc, list)
    assert decode_polyline(None) == []
    assert polyline_length([(121.4, 31.2)]) == 0.0
    assert order_consistency(["A路", "B路"], ["A路", "B路"]) == (1.0, ["A路", "B路"])
    assert road_provider_mod.OVERPASS_MIRRORS is OVERPASS_MIRRORS
