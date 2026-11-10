"""地图能力层骨架（阶段 1a）的离线回归测试。

骨架不接现有调用点，因此这里只验证契约本身：
CRS 换算不会漏、凭证别名只有一份实现、注册表顺序可配置且能回退、
HTTP 基座可重试可缓存。全部不依赖网络、不消耗任何地图服务配额。
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import requests

from app.geo.coords import (
    gcj02_to_bd09,
    lonlat_to_mercator,
    wgs84_to_gcj02,
)
from app.maps import (
    CRS,
    Capability,
    CredentialStore,
    GeoPoint,
    HttpClient,
    PlaceCandidate,
    ProbeResult,
    ProviderMeta,
    ProviderRegistry,
    to_gcj02,
)


class _Cfg:
    """最小配置桩，避免测试读到真实 .env。"""

    def __init__(self, **values: str) -> None:
        self.values = values

    def _env_value(self, name: str) -> str | None:
        return self.values.get(name)


class _StubProvider:
    def __init__(
        self,
        name: str,
        capability: Capability = Capability.GEOCODING,
        crs: CRS = CRS.GCJ02,
        available: bool = True,
        probe: ProbeResult | None = None,
    ) -> None:
        self.meta = ProviderMeta(
            name=name, capability=capability, output_crs=crs,
            credential_aliases=("STUB_KEY",),
        )
        self._available = available
        self._probe = probe or ProbeResult(True, hits=1)

    def is_available(self) -> bool:
        return self._available

    def probe(self) -> ProbeResult:
        return self._probe


def _dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    x1, y1 = lonlat_to_mercator(*a)
    x2, y2 = lonlat_to_mercator(*b)
    return math.hypot(x2 - x1, y2 - y1)


# ---------------------------------------------------------------- CRS

def test_wgs84_and_cgcs2000_share_one_conversion():
    """天地图 CGCS2000 必须走和 WGS-84 同一条加偏路径，不能漏转。"""
    lon, lat = 121.40, 31.20
    a = to_gcj02(lon, lat, CRS.WGS84)
    b = to_gcj02(lon, lat, CRS.CGCS2000)
    assert a == b == wgs84_to_gcj02(lon, lat)
    assert 400 < _dist_m((lon, lat), a) < 800, "上海实测约 570 m 系统偏移"
    assert to_gcj02(lon, lat, "CGCS2000") == b, "字符串形式也应可解析"


def test_gcj02_is_identity_and_bd09_is_reversed():
    lon, lat = 121.40, 31.20
    assert to_gcj02(lon, lat, CRS.GCJ02) == (lon, lat)
    bd = gcj02_to_bd09(lon, lat)
    back = to_gcj02(*bd, CRS.BD09)
    assert _dist_m((lon, lat), back) < 5.0


def test_unknown_crs_is_rejected_instead_of_silently_passed_through():
    with pytest.raises(ValueError):
        to_gcj02(121.4, 31.2, "MARS-2000")


def test_place_candidate_exposes_only_gcj02_coordinates():
    """provider 声明 CGCS2000 后，调用方读到的坐标已经是项目基准。"""
    lon, lat = 121.40, 31.20
    p = GeoPoint(lon, lat, CRS.CGCS2000)
    c = PlaceCandidate(name="上海动物园", point=p, source="tianditu")
    assert c.point.crs == CRS.CGCS2000
    assert (c.lon, c.lat) == wgs84_to_gcj02(lon, lat)
    assert p.as_dict() == {"lon": c.lon, "lat": c.lat, "crs": "CGCS2000"}


# ---------------------------------------------------------------- 凭证

def test_credentials_resolve_single_env_name():
    store = CredentialStore(_Cfg(TIANDITU_API_KEY="tk-a"))
    assert store.resolve("TIANDITU_API_KEY") == "tk-a"
    assert store.has("TIANDITU_API_KEY")
    assert not store.has("MISSING_KEY")


def test_credentials_fall_back_to_process_env(monkeypatch):
    monkeypatch.setenv("MAP_TEST_KEY", "from-env")
    assert CredentialStore().resolve("MAP_TEST_KEY") == "from-env"
    meta = ProviderMeta("x", Capability.GEOCODING,
                        credential_aliases=("MAP_TEST_KEY",))
    assert CredentialStore().resolve_for(meta) == "from-env"


# ---------------------------------------------------------------- 注册表

def test_registry_order_honours_configured_env():
    cfg = _Cfg(MAP_GEOCODING_ORDER="tianditu,amap")
    reg = ProviderRegistry(cfg)
    for n in ("amap", "tmap", "baidu", "tianditu"):
        reg.register(_StubProvider(n))
    assert reg.order(Capability.GEOCODING)[:2] == ("tianditu", "amap")
    assert reg.order(Capability.GEOCODING)[2:] == ("tmap", "baidu")


def test_registry_env_unknown_names_and_duplicates_ignored():
    cfg = _Cfg(MAP_GEOCODING_ORDER="tianditu,typo,tianditu,amap")
    reg = ProviderRegistry(cfg)
    for n in ("amap", "tianditu"):
        reg.register(_StubProvider(n))
    assert reg.order(Capability.GEOCODING) == ("tianditu", "amap")


def test_registry_default_order_and_unlisted_provider_appended():
    reg = ProviderRegistry(_Cfg())
    for n in ("amap", "tmap", "baidu", "tianditu", "new_vendor"):
        reg.register(_StubProvider(n))
    order = reg.order(Capability.GEOCODING)
    assert order[:4] == ("amap", "tmap", "baidu", "tianditu")
    assert order[-1] == "new_vendor", "新注册实现不能因为漏改默认表而永不生效"


def test_registry_available_skips_unconfigured_and_probe_is_structured():
    reg = ProviderRegistry(_Cfg())
    reg.register(_StubProvider("amap", available=False))
    reg.register(_StubProvider(
        "tianditu", probe=ProbeResult(True, hits=3, sample="北京大学")))
    assert [p.meta.name for p in reg.available(Capability.GEOCODING)] == ["tianditu"]
    out = reg.probe(Capability.GEOCODING)
    assert out == {"tianditu": {"ok": True, "hits": 3, "sample": "北京大学"}}


def test_registry_probe_exception_does_not_break_health():
    class _Boom(_StubProvider):
        def probe(self):
            raise RuntimeError("upstream down")

    reg = ProviderRegistry(_Cfg())
    reg.register(_Boom("amap"))
    out = reg.probe(Capability.GEOCODING)
    assert out["amap"]["ok"] is False
    assert "RuntimeError" in out["amap"]["error"]


# ---------------------------------------------------------------- HTTP

class _FakeResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


class _FakeSession:
    def __init__(self, data, fail_times=0):
        self.headers = {}
        self.data = data
        self.fail_times = fail_times
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        if len(self.calls) <= self.fail_times:
            raise requests.ConnectionError("boom")
        return _FakeResponse(self.data)


def test_http_client_cache_avoids_second_request(tmp_path):
    s = _FakeSession({"pois": []})
    c = HttpClient(session=s, cache_dir=tmp_path / "cache", timeout=3.0)
    assert c.session.headers["User-Agent"].startswith("meituan-ride/")
    assert c.get_json("https://example.test/search", params={"q": "a"}, cache=True) == {"pois": []}
    assert c.get_json("https://example.test/search", params={"q": "a"}, cache=True) == {"pois": []}
    assert len(s.calls) == 1


def test_http_client_retries_and_lenient_variant(tmp_path):
    s = _FakeSession({"ok": 1}, fail_times=1)
    c = HttpClient(session=s, retries=1)
    assert c.get_json("https://example.test/ping") == {"ok": 1}
    assert len(s.calls) == 2

    dead = _FakeSession({}, fail_times=99)
    c2 = HttpClient(session=dead, retries=1)
    assert c2.try_get_json("https://example.test/ping") is None


# ---------------------------------------------------------------- 分层纯度

@pytest.mark.parametrize("name", ["contracts.py", "credentials.py", "registry.py"])
def test_contract_layer_does_not_depend_on_http_client(name):
    """契约/凭证/注册表不该各自再 import requests，否则统一基座形同虚设。"""
    path = Path(__file__).resolve().parents[1] / "src" / "app" / "maps" / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "requests" not in imported
