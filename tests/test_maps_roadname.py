"""能力②（路名）与注册表多维索引、health 装配的回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppConfig
from app.maps.contracts import CRS, Capability, ProbeResult, ProviderMeta
from app.maps.registry import ProviderRegistry

ROOT = Path(__file__).resolve().parent.parent

_ENV_NAMES = ("AMAP_KEY", "TMAP_KEY", "BAIDU_MAP_KEY", "MAP_ROADNAME_ORDER")


class _Stub:
    def __init__(self, name, capability, available=True):
        self.meta = ProviderMeta(name, capability, CRS.GCJ02, ("AMAP_KEY",))
        self._available = available

    def is_available(self):
        return self._available

    def probe(self):
        return ProbeResult(self._available)


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _cfg(tmp_path, monkeypatch, lines=()):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "roadname.env"
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return AppConfig(env_path=env)


def test_registry_indexes_by_capability_not_just_name():
    """高德同时提供地名与路名，同名 provider 不能互相覆盖。"""
    reg = ProviderRegistry(_cfg_stub())
    reg.register(_Stub("amap", Capability.GEOCODING))
    reg.register(_Stub("amap", Capability.ROADNAME))
    assert reg.names(Capability.GEOCODING) == ("amap",)
    assert reg.names(Capability.ROADNAME) == ("amap",)
    assert reg.get("amap", Capability.GEOCODING).meta.capability == Capability.GEOCODING
    assert reg.get("amap", Capability.ROADNAME).meta.capability == Capability.ROADNAME
    assert len(reg.available(Capability.ROADNAME)) == 1


def _cfg_stub():
    class _C:
        @staticmethod
        def _env_value(name):
            return None

    return _C()


def test_amap_key_goes_through_credential_store(tmp_path, monkeypatch):
    """_amap_key 不再自己读 os.environ / 拼 .env 路径。"""
    import app.config as config_mod
    import app.roadname as roadname_mod

    cfg = _cfg(tmp_path, monkeypatch, ["AMAP_KEY=from-dotenv"])
    monkeypatch.setattr(config_mod, "CONFIG", cfg)
    assert roadname_mod._amap_key() == "from-dotenv"
    src = (ROOT / "src" / "app" / "roadname.py").read_text(encoding="utf-8")
    assert "os.environ.get" not in src, "凭证读取必须只有能力层一处实现"


def test_attribute_by_online_uses_registry(tmp_path, monkeypatch):
    import app.maps.roadname.amap as amap_impl
    import app.roadname as roadname_mod

    payload = {"status": "1", "route": {"paths": [{"steps": [
        {"road": "安龙路", "distance": "500",
         "polyline": "121.30,31.20;121.31,31.21"},
        {"road": "剑河路", "distance": "300",
         "polyline": "121.31,31.21;121.32,31.22"},
    ]}]}}
    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append(url)
        return _Resp(payload)

    monkeypatch.setattr(amap_impl.requests, "get", _fake_get)
    cfg = _cfg(tmp_path, monkeypatch, ["AMAP_KEY=ak"])
    lonlat = [(121.30, 31.20), (121.32, 31.22)]
    attr = roadname_mod.attribute_by_online(lonlat, cfg=cfg, timeout=3.0)
    assert attr.source == "amap"
    assert attr.roads == ["安龙路", "剑河路"]
    assert calls and "bicycling" in calls[0]
    assert "attribute_by_amap" not in attr.note, "上层结果不绑定厂商名"


def test_attribute_by_online_degrades_without_key(tmp_path, monkeypatch):
    import app.roadname as roadname_mod

    cfg = _cfg(tmp_path, monkeypatch, [])
    attr = roadname_mod.attribute_by_online(
        [(121.30, 31.20), (121.32, 31.22)], cfg=cfg, timeout=3.0)
    assert attr.source == "none"
    assert attr.segments == []


def test_health_reads_registry_instead_of_hardcoded_keys():
    src = (ROOT / "src" / "app" / "server.py").read_text(encoding="utf-8")
    assert "bool(CONFIG.amap_key)" not in src
    assert "reg.names(Capability.GEOCODING)" in src
    assert "probe_online()" in src, "health 仍必须主动探测"
