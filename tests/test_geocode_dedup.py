"""地理编码输入质量回归：词库同址多源去重 + 高德错字纠错拦截。

2026-11-21 行者宁波长截图调查暴露两处确定性缺陷：

1. Lexicon.lookup 只做 round(6) 精确去重，跨源几十米的差异漏过去
   （「宁波站」三源彼此 30-70m），一个地名占 3 个候选名额。
2. 高德 provider 把返回的 POI 照单全收，OCR 错字「董州区政府」被
   纠错成「鄞州区政府」，坐标偏离 1370m 拉偏整个配准模型。

两处修法都必须保住「有益的截断还原」：残片补全（杭甬高→杭甬高速、
波站→宁波站）不能被误杀——它们是配准主力锚点。
"""

from __future__ import annotations

import json

from app.config import AppConfig
from app.geocode import Lexicon
from app.maps.geocoding.providers import AmapProvider


class _Creds:
    def resolve(self, *names: str) -> str:
        return "fake-key"


def _write_lexicon(tmp_path, entries: dict) -> Lexicon:
    p = tmp_path / "lex.json"
    p.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return Lexicon(p)


def _amap_with(*names: str) -> AmapProvider:
    """构造一个返回固定 POI 名列表的高德 provider（不发真实请求）。"""
    p = AmapProvider(AppConfig(), _Creds(), lambda: None)
    pois = [
        {
            "name": name,
            "location": "121.5366,29.8619",
            "address": "宁波市海曙区",
            "type": "交通设施服务",
        }
        for name in names
    ]
    p._get = lambda url, params: {"status": "1", "pois": pois}
    return p


# --- 词库同址多源去重 -----------------------------------------------------


def test_lexicon_merges_nearby_sources_and_keeps_heaviest(tmp_path) -> None:
    """同址三源（彼此 <100m）聚成一个候选，权重最高的 amap 胜出。"""
    lex = _write_lexicon(tmp_path, {
        "测试站": [
            {"lon": 121.5370, "lat": 29.8621, "weight": 1.1,
             "source": "baidu", "city": "宁波"},
            {"lon": 121.5366, "lat": 29.8619, "weight": 3.0,
             "source": "amap", "city": "宁波"},
            {"lon": 121.5368, "lat": 29.8626, "weight": 0.2,
             "source": "tianditu", "city": "宁波"},
        ]
    })
    out = lex.lookup("测试站", "宁波")
    assert len(out) == 1, f"同址三源未合并: {[c.source for c in out]}"
    assert out[0].source == "amap", "应保留权重最高的来源"


def test_lexicon_keeps_distant_same_name(tmp_path) -> None:
    """相距数公里的同名地点必须保留为独立候选，不能被误合并。"""
    lex = _write_lexicon(tmp_path, {
        "测试站": [
            {"lon": 121.5300, "lat": 29.8600, "weight": 3.0,
             "source": "amap", "city": "宁波"},
            {"lon": 121.6000, "lat": 29.8600, "weight": 3.0,
             "source": "amap", "city": "宁波"},
        ]
    })
    out = lex.lookup("测试站", "宁波")
    assert len(out) == 2, "异地同名 POI 被误合并"


def test_real_lexicon_ningbo_station_deduped() -> None:
    """真实词库：「宁波站」三源沉淀应收敛为 1 个候选。"""
    from pathlib import Path

    lex = Lexicon(Path("data/lexicon.json"))
    out = lex.lookup("宁波站", "宁波")
    assert len(out) == 1, f"真实词库宁波站仍有 {len(out)} 个候选"
    assert out[0].source == "amap"


# --- 高德错字纠错拦截 -----------------------------------------------------


def test_amap_drops_wrong_word_correction() -> None:
    """高德把「董州区政府」纠错成「鄞州区政府」——必须剔除。"""
    out = _amap_with("董州区政府", "鄞州区政府").search("董州区政府", "宁波")
    assert [c.name for c in out] == ["董州区政府"]


def test_amap_keeps_fragment_completion() -> None:
    """残片补全是有益纠错，必须保留并给出合理质量分。"""
    out = _amap_with("杭甬高速").search("杭甬高", "宁波")
    assert len(out) == 1 and out[0].name == "杭甬高速"
    assert out[0].quality >= 0.8

    out = _amap_with("宁波站").search("波站", "宁波")
    assert len(out) == 1 and out[0].name == "宁波站"


def test_amap_keeps_suffixed_poi() -> None:
    """带后缀的真实 POI（公交站/门）走分段包含，不能被误杀。"""
    out = _amap_with("宁波站-公交站", "宁波站").search("宁波站", "宁波")
    assert len(out) == 2
