"""能力①接入注册表后的端到端回归（全部离线 mock，不消耗配额）。

核心完成判据：**只改 .env 的顺序配置、不改一行 Python 代码**，
就能切换在线地理编码服务商并正确回退；CRS 由 provider 声明、
由契约层统一换算；跨市同名候选不得写回词典。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import AppConfig
from app.geocode import Geocoder, Lexicon
from app.maps.contracts import CRS, Capability

_ENV_NAMES = (
    "AMAP_KEY", "TMAP_KEY", "BAIDU_MAP_KEY",
    "TIANDITU_API_KEY", "MAP_GEOCODING_ORDER", "TIANDITU_MAP_BOUND",
    # 中转配置也要隔离：config.py 导入时 load_dotenv() 把项目 .env 灌进
    # os.environ，relay 开着会把 provider 的可用性与 URL 改写带进测试
    # （实测开发者本地开中转后 3 个用例误报）。
    "AMAP_RELAY_URL", "AMAP_RELAY_TOKEN",
)

_AMAP_URL = "restapi.amap.com/v3/place/text"
_TDT_URL = "api.tianditu.gov.cn/v2/search"


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _RoutingSession:
    def __init__(self, **by_fragment):
        self.by_fragment = by_fragment
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        for fragment, payload in self.by_fragment.items():
            if fragment in url:
                return _Resp(payload)
        return _Resp({"status": "0", "info": "no stub", "pois": []})

    @property
    def urls(self) -> list[str]:
        return [u for u, _ in self.calls]


def _amap_ok(name: str = "上海动物园", lon: float = 121.36, lat: float = 31.19):
    return {"status": "1", "pois": [
        {"name": name, "location": f"{lon},{lat}",
         "address": "上海市长宁区", "type": "风景名胜"},
    ]}


def _amap_empty():
    return {"status": "0", "info": "USER_DAILY_QUERY_OVER_LIMIT", "pois": []}


def _tdt_ok(name: str = "上海动物园"):
    return {"status": {"cndesc": "服务正常", "infocode": 1000},
            "resultType": 1, "pois": [
                {"name": name, "lonlat": "121.35808,31.19487",
                 "address": "上海市长宁区虹桥路2381号", "typeName": "公园"},
            ]}


def _mk(tmp_path, monkeypatch, lines, session):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "order.env"
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = AppConfig(env_path=env)
    g = Geocoder(cfg, Lexicon(tmp_path / "lexicon.json"))
    g._session = session
    return cfg, g


def test_env_order_switches_provider_without_code_change(tmp_path, monkeypatch):
    """tianditu 优先时完全不打高德；改成 amap 优先时按顺序回退。"""
    s1 = _RoutingSession(**{_TDT_URL: _tdt_ok(), _AMAP_URL: _amap_ok()})
    _, g1 = _mk(tmp_path, monkeypatch, [
        "TIANDITU_API_KEY=tk", "AMAP_KEY=ak",
        "MAP_GEOCODING_ORDER=tianditu,amap",
    ], s1)
    out1 = g1.geocode("上海动物园", city="上海", allow_online=True,
                      allow_cold_start=False)
    assert out1 and out1[0].source == "tianditu"
    assert not any(_AMAP_URL in u for u in s1.urls), "天地图优先时不应打高德"
    assert g1.registry.order(Capability.GEOCODING)[:2] == ("tianditu", "amap")

    s2 = _RoutingSession(**{_TDT_URL: _tdt_ok(), _AMAP_URL: _amap_empty()})
    _, g2 = _mk(tmp_path, monkeypatch, [
        "TIANDITU_API_KEY=tk", "AMAP_KEY=ak",
        "MAP_GEOCODING_ORDER=amap,tianditu",
    ], s2)
    out2 = g2.geocode("上海动物园", city="上海", allow_online=True,
                      allow_cold_start=False)
    assert out2 and out2[0].source == "tianditu", "上游返回空应自动落天地图"
    assert any(_AMAP_URL in u for u in s2.urls)
    assert any(_TDT_URL in u for u in s2.urls)


def test_cross_city_same_name_is_not_written_back(tmp_path, monkeypatch):
    """同名但坐标在外地的候选仍返回给 RANSAC，但不得沉淀进词典。"""
    s = _RoutingSession(**{_AMAP_URL: _amap_ok(
        "红星美凯龙", lon=120.2694, lat=31.0962)})
    _, g = _mk(tmp_path, monkeypatch, [
        "AMAP_KEY=ak", "MAP_GEOCODING_ORDER=amap",
    ], s)
    data = {
        f"地点{i}": [{"lon": 121.35 + i * 0.02, "lat": 31.18 + i * 0.01,
                      "city": "上海", "weight": 1, "count": 1}]
        for i in range(12)
    }
    # 一条标错城市的尾部条目：分位判据不该被它撑大
    data["错条目"] = [{"lon": 113.15, "lat": 22.50,
                       "city": "上海", "weight": 1, "count": 1}]
    g.lexicon.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    g.lexicon.load()
    assert g.lexicon.city_bounds("上海") is not None

    out = g.geocode("红星美凯龙", city="上海", allow_online=True,
                    allow_cold_start=False)
    assert out and out[0].source == "amap", "在线结果本身仍交给 RANSAC 判断"
    saved = json.loads(g.lexicon.path.read_text(encoding="utf-8"))
    assert "红星美凯龙" not in saved, "跨市同名候选不应污染城市视野"


def test_city_core_uses_distance_quantile_not_minmax(tmp_path, monkeypatch):
    """写回护栏用分位数半径：尾部错条目撑不宽，异地同名仍被挡。"""
    s = _RoutingSession(**{})
    _, g = _mk(tmp_path, monkeypatch, ["TIANDITU_API_KEY=tk"], s)
    data = {
        f"地点{i}": [{"lon": 121.35 + i * 0.01, "lat": 31.18 + i * 0.005,
                      "city": "上海", "weight": 1, "count": 1}]
        for i in range(12)
    }
    # 跨省错条目只占尾部，不能影响 95% 分位半径
    data["错条目"] = [{"lon": 113.15, "lat": 22.50,
                       "city": "上海", "weight": 1, "count": 1}]
    g.lexicon.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    g.lexicon.load()
    assert g.lexicon.in_city_core(121.40, 31.21, "上海") is True
    assert g.lexicon.in_city_core(120.2694, 31.0962, "上海") is False
    assert g.lexicon.in_city_core(120.6467, 31.1083, "上海") is False
    assert g._in_city_bounds(120.2694, 31.0962, "上海") is False
    assert g._in_city_bounds(121.40, 31.21, "上海") is True
    # 词典为空 / 新城市：无法判断时放行，不能误杀冷启动
    empty = Lexicon(tmp_path / "empty.json")
    assert empty.in_city_core(120.0, 30.0, "新城市") is None
    assert g._in_city_bounds(120.0, 30.0, "新城市") is True


def test_every_provider_declares_its_crs(tmp_path, monkeypatch):
    """四家源都声明 output_crs，换算只能走契约层一处。"""
    s = _RoutingSession(**{})
    _, g = _mk(tmp_path, monkeypatch, ["TIANDITU_API_KEY=tk"], s)
    expected = {
        "amap": CRS.GCJ02,
        "tmap": CRS.GCJ02,
        "baidu": CRS.BD09,
        "tianditu": CRS.CGCS2000,
    }
    for name, crs in expected.items():
        impl = g._impl[name]
        assert impl.meta.output_crs == crs
        assert impl.meta.capability == Capability.GEOCODING
    assert g.registry.names(Capability.GEOCODING) == tuple(expected)


def test_probe_online_shape_matches_health_contract(tmp_path, monkeypatch):
    s = _RoutingSession(**{_TDT_URL: _tdt_ok()})
    _, g = _mk(tmp_path, monkeypatch, ["TIANDITU_API_KEY=tk"], s)
    probe = g.probe_online()
    assert probe["tianditu"]["ok"] is True
    assert probe["tianditu"]["hits"] == 1
    assert "amap" not in probe, "未配置的源不应出现在健康探测里"
