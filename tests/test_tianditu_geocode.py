"""天地图在线地理编码回归测试（全部离线 mock，不消耗配额）。

守住四个实测踩过的坑：

1. `.env` 里用户写的是 ``TIANDITU_API_KEY``，旧代码只读 ``TIANDITU_KEY``；
2. `/v2/search` 必须用 ``postStr + type=query``，旧 ``ds={"keyword":...}``
   是地理编码接口的参数，实测直接 HTTP 400；
3. 天地图返回 CGCS2000（≈WGS-84），项目内部锚点按 GCJ-02，漏转会凭空
   引入约 570 米系统性偏移；
4. POI 在 ``pois[].lonlat`` 字符串里，不是 ``result.items[].lon/lat``。
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (AppConfig, ONLINE_PROVIDERS,
                        TIANDITU_MAP_BOUND_DEFAULT)
from app.geo.coords import lonlat_to_mercator, wgs84_to_gcj02
from app.geocode import Geocoder, Lexicon

_ENV_NAMES = (
    "AMAP_KEY", "TMAP_KEY", "BAIDU_MAP_KEY",
    "TIANDITU_KEY", "TIANDITU_API_KEY", "TIANDITU_TK",
    "GEOCODE_PROVIDER_ORDER", "MAP_GEOCODING_ORDER",
    "TIANDITU_MAP_BOUND",
)


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        return _Resp(self.payload)


def _payload():
    return {
        "status": {"cndesc": "服务正常", "infocode": 1000},
        "resultType": 1,
        "count": 3,
        "pois": [
            {"name": "异地同名公园", "lonlat": "116.310003,39.991957",
             "address": "北京市海淀区"},
            {"name": "上海动物园", "lonlat": "121.35808,31.19487",
             "address": "上海市长宁区虹桥路2381号", "typeName": "公园"},
            {"name": "上海动物园-公交站", "lonlat": "121.36357,31.191952",
             "address": "上海动物园", "typeName": "公交站"},
        ],
    }


def _mk(tmp_path, monkeypatch, payload=None, extra_env=None, env_name=".env"):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    lines = ["TIANDITU_API_KEY=test_tianditu_tk"]
    for k, v in (extra_env or {}).items():
        lines.append(f"{k}={v}")
    env = tmp_path / env_name
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = AppConfig(env_path=env)
    g = Geocoder(cfg, Lexicon(tmp_path / "lexicon.json"))
    g._session = _Session(payload if payload is not None else _payload())
    return cfg, g


def test_alias_and_defaults(tmp_path, monkeypatch):
    """TIANDITU_API_KEY 能被读到，且默认顺序/视野不含城市硬编码。"""
    cfg, _ = _mk(tmp_path, monkeypatch)
    assert cfg.tianditu_key == "test_tianditu_tk"
    assert cfg.online_provider_order == ONLINE_PROVIDERS
    assert cfg.tianditu_map_bound == TIANDITU_MAP_BOUND_DEFAULT

    legacy = tmp_path / "legacy.env"
    legacy.write_text("TIANDITU_KEY=legacy_tk\n", encoding="utf-8")
    cfg2 = AppConfig(env_path=legacy)
    assert cfg2.tianditu_key == "legacy_tk"


def test_v2_request_and_gcj02_conversion(tmp_path, monkeypatch):
    """官方 V2 格式 + lonlat 解析 + CGCS2000→GCJ-02 一次转换。"""
    _, g = _mk(tmp_path, monkeypatch)
    out = g._tianditu("上海动物园", "上海")
    # 空词典 ⇒ 全国视野；异地同名按地址过滤，泛匹配按同名规则过滤
    assert len(out) == 1
    url, params = g._session.calls[0]
    assert url.endswith("/v2/search")
    assert params["type"] == "query"
    assert params["tk"] == "test_tianditu_tk"
    assert "ds" not in params
    post = json.loads(params["postStr"])
    assert post["keyWord"] == "上海动物园"
    assert str(post["queryType"]) == "1"
    assert post["mapBound"] == TIANDITU_MAP_BOUND_DEFAULT
    assert post["start"] == 0 and post["count"] == 10

    assert out[0].name == "上海动物园"
    assert out[0].source == "tianditu"
    assert all("北京" not in r.address for r in out)
    assert all("公交站" not in r.name for r in out), "泛匹配候选会毒化 RANSAC"

    want = wgs84_to_gcj02(121.35808, 31.19487)
    assert abs(out[0].lon - want[0]) < 1e-9
    assert abs(out[0].lat - want[1]) < 1e-9
    x1, y1 = lonlat_to_mercator(121.35808, 31.19487)
    x2, y2 = lonlat_to_mercator(out[0].lon, out[0].lat)
    assert math.hypot(x2 - x1, y2 - y1) > 300, "坐标系没有转换会整体偏约 570 米"


def test_provider_order_and_fallback(tmp_path, monkeypatch):
    """配置可把天地图提到首位；默认顺序下上游返回空会自动落到天地图。"""
    _, g = _mk(tmp_path, monkeypatch,
               extra_env={"GEOCODE_PROVIDER_ORDER": "tianditu,amap"})
    calls = []
    g._amap = lambda name, city: calls.append("amap") or []
    res = g.geocode("上海动物园", city="上海", allow_online=True,
                    allow_cold_start=False)
    assert calls == [], "天地图优先时不应再打高德"
    assert res and res[0].source == "tianditu"

    _, g2 = _mk(tmp_path, monkeypatch, env_name="fallback.env")
    g2._amap = lambda name, city: []
    g2._tmap = lambda name, city: []
    g2._baidu = lambda name, city: []
    res2 = g2.geocode("上海动物园", city="上海", allow_online=True,
                      allow_cold_start=False)
    assert res2 and res2[0].source == "tianditu", "上游配额耗尽时应自动接管"


def test_city_bounds_and_narrow_filter(tmp_path, monkeypatch):
    """词典派生城市视野；异地同名不能把配准拉到外省。"""
    _, g = _mk(tmp_path, monkeypatch)
    data = {}
    for i in range(6):
        data[f"地点{i}"] = [{"lon": 121.35 + i * 0.02, "lat": 31.18 + i * 0.01,
                            "city": "上海", "weight": 1, "count": 1}]
    # 一条标错城市的跨省条目，不能把视野撑到华北
    data["错条目"] = [{"lon": 116.30, "lat": 39.90,
                     "city": "上海", "weight": 1, "count": 1}]
    g.lexicon.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    g.lexicon.load()
    bounds = g.lexicon.city_bounds("上海")
    assert bounds is not None
    bx = [float(x) for x in bounds.split(",")]
    assert bx[0] > 120 and bx[2] < 123, f"视野被离群条目污染: {bounds}"

    out = g._tianditu("上海动物园", "上海")
    url, params = g._session.calls[-1]
    assert json.loads(params["postStr"])["mapBound"] == bounds
    assert all("北京" not in r.address for r in out)
    assert len(out) == 1, "收窄视野 + 同名过滤后只剩截图上的那个点"


def test_probe_and_error_status(tmp_path, monkeypatch):
    """health 探测覆盖天地图，服务错误码要留下可读原因。"""
    _, g = _mk(tmp_path, monkeypatch)
    probe = g.probe_online()
    assert probe["tianditu"]["ok"] is True
    assert probe["tianditu"]["hits"] == 3

    bad = {"status": {"infocode": 2001, "cndesc": "请求参数错误"},
           "resultType": None, "pois": []}
    _, g2 = _mk(tmp_path, monkeypatch, payload=bad, env_name="bad.env")
    assert g2._tianditu("任意地名", "上海") == []
    assert "天地图" in g2._last_error


def test_fuzzy_online_hit_not_written_back(tmp_path, monkeypatch):
    """泛匹配的在线结果不能沉淀进词典（否则永久污染锚点与城市视野）。"""
    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []})
    from app.geocode import GeocodeResult
    g._amap = lambda name, city: [GeocodeResult(
        name="某生活广场", lon=121.4, lat=31.2, quality=0.7, source="amap",
        address="上海市某区")]
    g.lexicon.save()
    res = g.geocode("生活", city="上海", allow_online=True,
                    allow_cold_start=False)
    assert res and res[0].source == "amap", "在线结果本身仍可返回给 RANSAC 判断"
    saved = json.loads(g.lexicon.path.read_text(encoding="utf-8"))
    assert all("某生活广场" not in k for k in saved), "泛匹配不应写回词典"
    assert g.lexicon.city_bounds("上海") is None


def test_map_bound_config(tmp_path, monkeypatch):
    """视野范围可配置且只做格式校验，不内置城市坐标。"""
    cfg, _ = _mk(tmp_path, monkeypatch,
                 extra_env={"TIANDITU_MAP_BOUND": "120.8,30.6,122.2,31.9"})
    assert cfg.tianditu_map_bound == "120.8,30.6,122.2,31.9"
    cfg2, _ = _mk(tmp_path, monkeypatch,
                  extra_env={"TIANDITU_MAP_BOUND": "不是坐标"},
                  env_name="invalid.env")
    assert cfg2.tianditu_map_bound == TIANDITU_MAP_BOUND_DEFAULT


def test_old_ds_format_not_reintroduced():
    """静态护栏：不要把地理编码接口的 ds/keyword 参数再搬回 V2 搜索。

    厂商实现已迁入 ``app/maps/geocoding/providers.py``，护栏随之迁移，
    并额外要求「声明输出坐标系 + 走契约层换算」——漏转会整体偏约 570 m。
    """
    src = (ROOT / "app" / "maps" / "geocoding" / "providers.py").read_text(
        encoding="utf-8")
    body = src.split("class TiandituProvider", 1)[1].split("\nclass ", 1)[0]
    assert "postStr" in body and '"ds"' not in body
    assert '.get("items"' not in body
    assert "CRS.CGCS2000" in body, "必须声明输出坐标系"
    assert ".gcj02" in body, "坐标必须经契约层统一换算"
    contracts = (ROOT / "app" / "maps" / "contracts.py").read_text(
        encoding="utf-8")
    assert "def to_gcj02" in contracts and "wgs84_to_gcj02" in contracts
