"""在线地理编码「单源严格模式」与天地图同名过滤的回归测试。

两件事，本组测试各守一半：

**1. 天地图同名过滤放宽**
早期实现要求返回标题与查询词**完全相等**，于是查「东方明珠」返回的
「东方明珠广播电视塔」「东方明珠-公交站」「东方明珠公园东南门」
全被丢掉——接口明明返回了 10 个坐标正确的 POI，解析后剩 0 个，
这个源等于从未可用。现在判据是「包含关系」，并按贴进度排序。

**2. 单源严格模式**
旧行为是按注册表顺序逐个试，任一源返回空就落下一个。实测高德当日
配额用尽（USER_DAILY_QUERY_OVER_LIMIT），每个地名都先打高德再打
天地图——既慢一倍，又让人误判实际用的是哪家（现象是「配了高德却
在用天地图」却毫无提示）。

现在：**配了哪家就用哪家，失败直接提示，不自动换源**。

全部离线 mock，不消耗任何配额。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.config import AppConfig
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


def _mk(tmp_path, monkeypatch, payload=None, extra_env=None, env_name=".env"):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    lines = ["AMAP_KEY=test_amap_key", "TIANDITU_API_KEY=test_tianditu_tk"]
    for k, v in (extra_env or {}).items():
        lines.append(f"{k}={v}")
    env = tmp_path / env_name
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = AppConfig(env_path=env)
    g = Geocoder(cfg, Lexicon(tmp_path / "lexicon.json"))
    g._session = _Session(payload if payload is not None else {})
    return cfg, g


# ==================================================================
# 1) 天地图同名过滤：包含关系 + 贴进度排序
# ==================================================================


def _td_payload():
    """模拟天地图对「东方明珠」的返回：标题都比查询词长。"""
    return {
        "status": {"infocode": 1000, "cndesc": "服务正常"},
        "resultType": 1,
        "pois": [
            {"name": "东方明珠广播电视塔", "lonlat": "121.49519,31.24176",
             "address": "上海市浦东新区世纪大道1号", "typeName": "标志性建筑物"},
            {"name": "东方明珠-公交站", "lonlat": "121.496021,31.238826",
             "address": "东方明珠", "typeName": "公交站"},
            {"name": "东方明珠公园东南门", "lonlat": "121.497,31.239",
             "address": "上海市浦东新区陆家嘴街道", "typeName": "门"},
            {"name": "完全无关的景点", "lonlat": "121.5,31.24",
             "address": "上海市某处", "typeName": "景点"},
        ],
    }


def test_tianditu_fuzzy_name_is_accepted(tmp_path, monkeypatch):
    """标题比查询词长的候选必须被接受（旧实现全部丢弃）。"""
    _, g = _mk(tmp_path, monkeypatch, payload=_td_payload())
    out = g._tianditu("东方明珠", "上海")
    names = [r.name for r in out]
    assert "东方明珠广播电视塔" in names, \
        f"包含关系的候选被丢弃了: {names}"
    assert "东方明珠-公交站" in names, \
        f"包含关系的候选被丢弃了: {names}"
    # 完全无关的必须仍然被排除
    assert "完全无关的景点" not in names, "无关候选不应进入结果"


def test_tianditu_exact_match_ranks_first(tmp_path, monkeypatch):
    """完全相等的候选要排在前面。"""
    payload = _td_payload()
    payload["pois"].insert(0, {
        "name": "东方明珠", "lonlat": "121.495,31.2417",
        "address": "上海市浦东新区世纪大道1号", "typeName": "景点",
    })
    _, g = _mk(tmp_path, monkeypatch, payload=payload)
    out = g._tianditu("东方明珠", "上海")
    assert out[0].name == "东方明珠", f"精确匹配应排最前: {out[0].name}"
    assert out[0].quality >= 0.9, "精确匹配的质量分应最高"


def test_tianditu_prefix_ranks_before_loose_contains(tmp_path, monkeypatch):
    """「东方明珠XX」（前缀）应排在「XX东方明珠」（任意包含）之前。"""
    payload = {
        "status": {"infocode": 1000, "cndesc": "服务正常"},
        "resultType": 1,
        "pois": [
            {"name": "东方明珠广播电视塔", "lonlat": "121.49519,31.24176",
             "address": "上海市浦东新区", "typeName": "建筑"},
            {"name": "老东方明珠", "lonlat": "121.494,31.241",
             "address": "上海市浦东新区", "typeName": "景点"},
        ],
    }
    _, g = _mk(tmp_path, monkeypatch, payload=payload)
    out = g._tianditu("东方明珠", "上海")
    assert out[0].name == "东方明珠广播电视塔", \
        f"前缀包含应优先于任意包含: {[r.name for r in out]}"
    assert out[0].quality > out[1].quality


def test_tianditu_unrelated_candidate_rejected(tmp_path, monkeypatch):
    """不含查询词的名字必须被排除（防止泛类 POI 污染）。"""
    _, g = _mk(tmp_path, monkeypatch, payload=_td_payload())
    out = g._tianditu("东方明珠", "上海")
    for r in out:
        assert "东方明珠" in r.name, f"混入了无关候选: {r.name}"


# ==================================================================
# 2) 视野：改用「中位数中心 + 核心半径」，不被离群条目撑大
# ==================================================================


def test_bounds_use_core_center_not_min_max(tmp_path, monkeypatch):
    """词典里有跨省错条目时，视野必须以中位数中心派生。"""
    _, g = _mk(tmp_path, monkeypatch, payload=_td_payload())
    data = {}
    # 12 个上海本地点
    for i in range(12):
        data[f"地点{i}"] = [{"lon": 121.35 + i * 0.02, "lat": 31.18 + i * 0.01,
                            "city": "上海", "weight": 1, "count": 1}]
    # 一个离群点：山西
    data["错条目"] = [{"lon": 116.30, "lat": 39.90,
                     "city": "上海", "weight": 1, "count": 1}]
    g.lexicon.path.write_text(json.dumps(data, ensure_ascii=False),
                              encoding="utf-8")
    g.lexicon.load()

    g._tianditu("东方明珠", "上海")
    post = json.loads(g._session.calls[-1][1]["postStr"])
    pb = [float(x) for x in post["mapBound"].split(",")]
    # 中心应在 121.4 附近，半径约 30 km ⇒ 视野不含山西
    assert 121.2 < pb[0] and pb[2] < 121.7, f"视野异常: {post['mapBound']}"
    assert pb[3] < 32.0, f"纵向被离群条目污染: {post['mapBound']}"


def test_core_radius_and_center_agree_with_in_city_core(tmp_path, monkeypatch):
    """core_radius_m / core_center 与 in_city_core 口径一致。"""
    _, g = _mk(tmp_path, monkeypatch, payload=_td_payload())
    data = {}
    for i in range(12):
        data[f"地点{i}"] = [{"lon": 121.35 + i * 0.02, "lat": 31.18 + i * 0.01,
                            "city": "上海", "weight": 1, "count": 1}]
    data["错条目"] = [{"lon": 116.30, "lat": 39.90,
                     "city": "上海", "weight": 1, "count": 1}]
    g.lexicon.path.write_text(json.dumps(data, ensure_ascii=False),
                              encoding="utf-8")
    g.lexicon.load()

    center = g.lexicon.core_center("上海")
    radius = g.lexicon.core_radius_m("上海")
    assert center is not None and radius is not None
    assert 121.3 < center[0] < 121.5, f"中心偏移: {center}"
    assert 10000 < radius < 80000, f"半径异常: {radius} m"
    # 离群点必须在核心范围之外
    assert g.lexicon.in_city_core(116.30, 39.90, "上海") is False
    # 本地点在核心范围之内
    assert g.lexicon.in_city_core(121.40, 31.20, "上海") is True


def test_core_helpers_return_none_on_thin_data(tmp_path, monkeypatch):
    """数据不足时返回 None，让调用方走宽松路径（不能瞎猜）。"""
    _, g = _mk(tmp_path, monkeypatch, payload=_td_payload())
    assert g.lexicon.core_center("上海") is None
    assert g.lexicon.core_radius_m("上海") is None
    assert g.lexicon.core_center("") is None


# ==================================================================
# 3) 单源严格模式
# ==================================================================


def test_default_uses_only_first_available_source(tmp_path, monkeypatch):
    """未配置时只用**一个**源（第一个有 key 的），失败不换源。"""
    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []})
    calls = []
    g._amap = lambda name, city: calls.append("amap") or []
    g._tianditu = lambda name, city: calls.append("tianditu") or []
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert calls == ["amap"], f"不应回退到第二个源: {calls}"


def test_single_configured_source_never_falls_back(tmp_path, monkeypatch):
    """显式指定天地图时，高德失败也不能被调用。"""
    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []},
               extra_env={"MAP_GEOCODING_ORDER": "tianditu"})
    calls = []
    g._amap = lambda name, city: calls.append("amap") or []
    g._tianditu = lambda name, city: calls.append("tianditu") or []
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert "amap" not in calls, f"单源模式不应调用其他源: {calls}"
    assert calls == ["tianditu"]


def test_multiple_configured_sources_do_fall_back(tmp_path, monkeypatch):
    """显式配多个时才启用回退语义。"""
    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []},
               extra_env={"MAP_GEOCODING_ORDER": "amap,tianditu"})
    calls = []
    g._amap = lambda name, city: calls.append("amap") or []
    g._tianditu = lambda name, city: calls.append("tianditu") or []
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert calls == ["amap", "tianditu"], f"多源应依次尝试: {calls}"


def test_failure_is_recorded_and_surfaced(tmp_path, monkeypatch):
    """失败原因必须被记录（供 pipeline 生成提示）。

    模拟真实故障：provider 抛异常（配额超限、鉴权失败等）。
    """
    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []})

    def boom(name, city):
        raise RuntimeError("USER_DAILY_QUERY_OVER_LIMIT")

    g._amap = boom
    g._online_failures.clear()
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert g._online_failures, "失败必须被记录"
    assert any("amap" in e for e in g._online_failures)
    assert any("QUOTA" in e or "OVER_LIMIT" in e
               for e in g._online_failures), g._online_failures


def test_provider_last_error_is_recorded(tmp_path, monkeypatch):
    """provider 自己记录的错误（如配额超限）也要被汇总。"""
    # 高德 WebService 的错误格式：status="0" + info
    payload = {"status": "0", "info": "USERKEY_PLAT_NOMATCH", "pois": []}
    _, g = _mk(tmp_path, monkeypatch, payload=payload)
    g._online_failures.clear()
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert g._online_failures, "provider 的错误必须被记录"
    assert any("amap" in e for e in g._online_failures)


def test_no_match_is_not_a_failure(tmp_path, monkeypatch):
    """「查不到」不是故障——不能报成服务商坏了。"""
    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []})
    g._amap = lambda name, city: []
    g._tianditu = lambda name, city: []
    g._online_failures.clear()
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert not g._online_failures, \
        f"仅「查不到」不应记为故障: {g._online_failures}"


def test_last_error_cleared_after_success(tmp_path, monkeypatch):
    """上一次失败、这次成功 ⇒ 不得残留错误提示。"""
    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []})

    # 先制造一次失败（provider 报错）
    def boom(name, city):
        raise RuntimeError("USER_DAILY_QUERY_OVER_LIMIT")

    g._amap = boom
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert g._last_error, "第一次失败应记录原因"

    # 再成功一次
    from app.geocode import GeocodeResult
    g._amap = lambda name, city: [GeocodeResult(
        name="某地", lon=121.4, lat=31.2, quality=1.0, source="amap",
        address="上海市")]
    g._online_failures.clear()
    g.geocode("某地名", city="上海", allow_online=True, allow_cold_start=False)
    assert not g._online_failures, "成功一次后不应再有失败记录"


def test_provider_resets_its_own_error(tmp_path, monkeypatch):
    """provider 的 last_error 必须每次查询前清空。"""
    from app.maps.contracts import Capability

    _, g = _mk(tmp_path, monkeypatch, payload={"resultType": 1, "pois": []})
    p = g.registry.get("tianditu", Capability.GEOCODING)
    # 制造一次错误
    g._session.payload = {"status": {"infocode": 2001, "cndesc": "参数错误"},
                          "resultType": None, "pois": []}
    p.search("某地名", "上海")
    assert p.last_error, "错误应被记录"
    # 换成成功响应
    g._session.payload = _td_payload()
    p.search("东方明珠", "上海")
    assert p.last_error == "", f"成功后应清空错误: {p.last_error}"


def test_configured_order_distinguishes_from_default(tmp_path, monkeypatch):
    """configured_order 只反映显式配置，不返回默认表。"""
    from app.maps.contracts import Capability

    _, g = _mk(tmp_path, monkeypatch)
    assert g.registry.configured_order(Capability.GEOCODING) == ()

    _, g2 = _mk(tmp_path, monkeypatch, env_name="b.env",
                extra_env={"MAP_GEOCODING_ORDER": "tianditu,amap"})
    assert g2.registry.configured_order(Capability.GEOCODING) == ("tianditu", "amap")
