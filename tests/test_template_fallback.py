"""P1：宽泛概览图 → 轨迹模板兜底。

不启动服务、不读本地截图、不触发 OCR / 在线地理编码：用合成折线 +
用户提供的两张真实高缩放截图的 **OCR 文本清单** 做 fixture，所有验证
都是进程内纯函数或桩件。运行期词典/模板库一律指向 tmp_path。

重点覆盖三件事：
1. 宽泛锚点判据只在概览图上触发（街道级细节图 / 手填路名必须不触发）；
2. 形状命中后弧长投影正确（正向、反向、子段、夹持），细长直线的
   假阳性模板必须被拒识；
3. 命中后返回的是「参考级」结果——封顶置信度、明确 warnings、不写词典、
   不精修、不再沉淀；未命中则回落到原流程，未注入模板库时行为完全不变。
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import app.pipeline as P
from app.config import CONFIG
from app.pipeline import PipelineResult
from app.route_template import (
    MIN_LOCATE_COVERAGE_B,
    TEMPLATE_CONFIDENCE_CAP,
    ShapeMatch,
    TemplateStore,
    project_to_geo,
    resample_by_arc,
    shape_has_enough_info,
)
from app.server import _capture_template, _need_background_refine
from app.template_fallback import (
    assess_broad_overview,
    is_broad_name,
    is_street_name,
    reference_confidence,
)
from app.track import TrackResult

# 真实高缩放概览图（1260×2720，同一条路线的往返）离线 OCR 结果。
OCR_OVERVIEW_8 = [
    "环亮速", "南北高架路", "逸仙高架路", "中环路", "桃浦中央绿地",
    "大宁公园", "虹口", "上海西站", "普陀区", "3号线", "上海站",
    "12号线", "黄浦区", "静安区", "14号线", "长宁区", "终", "8号线",
    "4号线内圈", "徐汇区", "起", "中环路", "7号线", "上海世博园", "金",
    "西岸艺术中心", "外环高速", "上海南站", "龙吴路", "锦江乐园",
    "1号线", "闵行区", "西岸自然", "梅陇生态公园", "艺术公园",
    "机场联络线", "黄浦", "5号线", "江", "虹梅高架路", "美团", "美团地图",
]
OCR_DETAIL = [
    "北新泾", "同仁医院", "天山西路", "剑河路", "平塘路", "淞虹路",
    "金钟路", "福泉北路", "通协路", "协和路", "北翟路", "哈密路",
    "可乐路", "新渔路", "起", "终", "美团地图",
]


def _bent_polyline(n: int = 120) -> list[list[float]]:
    """几字形折线：有明显的走向信息，不是起终点连线。"""
    pts: list[list[float]] = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.4:
            x, y = t / 0.4 * 300, 0.0
        elif t < 0.7:
            x, y = 300.0, (t - 0.4) / 0.3 * 250
        else:
            x, y = 300 - (t - 0.7) / 0.3 * 300, 250.0
        pts.append([x + 50, y + 50])
    return pts


def _flat_polyline(n: int = 120) -> list[list[float]]:
    """起终点连线：形状信息量不足，实测会造成跨路线假阳性。"""
    return [[float(50 + i * 5), 100.0] for i in range(n)]


def _geo_for(px: list[list[float]]) -> list[list[float]]:
    """把像素折线线性映射成一段合法的 GCJ-02 轨迹（约 20-30km）。"""
    a = np.asarray(px, dtype=float)
    x0, y0 = a.min(axis=0)
    w, h = np.ptp(a, axis=0)
    return [[121.20 + (x - x0) / max(w, 1) * 0.20,
             31.30 - (y - y0) / max(h, 1) * 0.10] for x, y in a]


def _scaled(px, k=1.7, dx=40, dy=-20):
    return [[x * k + dx, y * k + dy] for x, y in px]


def _store(tmp_path: Path, px, geo=None, *, city="上海", confidence=0.8):
    s = TemplateStore(tmp_path / "route_templates.json")
    s.remember(px_polyline=px, geo_polyline=geo or _geo_for(px),
               confidence=confidence, city=city)
    return s


# --- 判据 ---------------------------------------------------------------

def test_broad_overview_triggered_on_real_ocr():
    r = assess_broad_overview(OCR_OVERVIEW_8)
    assert r["broad_overview"] is True
    assert r["n_broad"] >= 20
    assert r["n_street"] <= 2
    assert r["street_examples"] == ["龙吴路"]
    assert r["broad_ratio"] > 0.5


def test_detail_screenshot_not_triggered():
    r = assess_broad_overview(OCR_DETAIL)
    assert r["broad_overview"] is False
    assert r["n_street"] >= 10
    assert r["broad_ratio"] < 0.2


def test_user_road_names_disable_fallback():
    assert assess_broad_overview(
        OCR_OVERVIEW_8, ["中环路"])["broad_overview"] is False


def test_too_few_texts_disable_fallback():
    assert assess_broad_overview(OCR_OVERVIEW_8[:6])["broad_overview"] is False


def test_name_classification():
    assert is_broad_name("静安区")
    assert is_broad_name("4号线内圈")
    assert is_broad_name("中环路")
    assert is_broad_name("上海南站")
    assert is_street_name("龙吴路")
    assert is_street_name("世纪大道")   # 「大道」不能算大范围对象
    assert not is_street_name("中环路")
    assert not is_broad_name("天山西路")


def test_confidence_capped():
    assert reference_confidence({"coverage": 1.0, "residual": 0.0}) <= TEMPLATE_CONFIDENCE_CAP
    assert reference_confidence({"coverage": 0.9844, "residual": 0.06695}) < 0.45
    assert reference_confidence({"coverage": 0.0, "residual": 1.0}) >= 0.10


# --- 形状信息量与投影 ---------------------------------------------------

def test_shape_info_rejects_flat_line():
    assert shape_has_enough_info(_bent_polyline()) is True
    assert shape_has_enough_info(_flat_polyline()) is False
    assert shape_has_enough_info([[0, 0], [1, 1]]) is False


def _project_template(px):
    px_rs = resample_by_arc(px)
    geo_rs = resample_by_arc(_geo_for(px))
    return {"px_points": px_rs.tolist(), "geo_points": geo_rs.tolist()}, geo_rs


def test_project_full_match_forward():
    px = _bent_polyline()
    tpl, geo = _project_template(px)
    m = ShapeMatch(True, 0.0, 1.0, False, (0, 64), "ok", 0.0,
                   "query_in_template")
    out = project_to_geo(px, tpl, m)
    assert out is not None and len(out) == 64
    assert np.allclose(out[0], geo[0], atol=1e-4)
    assert np.allclose(out[-1], geo[-1], atol=1e-4)


def test_project_full_match_reversed():
    px = _bent_polyline()
    tpl, geo = _project_template(px)
    m = ShapeMatch(True, 0.0, 1.0, True, (0, 64), "ok", 0.0,
                   "query_in_template")
    out = project_to_geo(px, tpl, m)
    assert out is not None
    assert np.allclose(out[0], geo[-1], atol=1e-4)
    assert np.allclose(out[-1], geo[0], atol=1e-4)


def test_project_subspan_stays_inside_template():
    px = _bent_polyline()
    tpl, geo = _project_template(px)
    m = ShapeMatch(True, 0.0, 0.75, False, (8, 56), "ok", 0.0,
                   "query_in_template")
    out = project_to_geo(px, tpl, m)
    assert out is not None
    # 子段两端不能落到模板首尾，否则说明 span 被忽略、里程会翻倍。
    assert not np.allclose(out[0], geo[0], atol=1e-4)
    assert not np.allclose(out[-1], geo[-1], atol=1e-4)
    assert np.allclose(out[0], geo[8], atol=1e-4)
    assert np.allclose(out[-1], geo[55], atol=1e-4)


def test_project_template_in_query_clamps_outer_points():
    px = _bent_polyline()
    tpl, geo = _project_template(px)
    m = ShapeMatch(True, 0.0, 0.9, False, (4, 60), "ok", 0.0,
                   "template_in_query")
    out = project_to_geo(px, tpl, m)
    assert out is not None
    assert np.allclose(out[0], out[4], atol=1e-4)   # 绝不外推
    assert np.allclose(out[-1], out[59], atol=1e-4)


# --- locate -------------------------------------------------------------

def test_locate_hits_similar_and_reversed(tmp_path):
    px = _bent_polyline()
    s = _store(tmp_path, px)
    hit = s.locate(_scaled(px), city_hint="上海")
    assert hit is not None
    assert len(hit["coords_gcj"]) == 64
    assert hit["match"]["reversed"] is False
    rev = s.locate(_scaled(px[::-1]), city_hint="上海")
    assert rev is not None
    assert rev["match"]["reversed"] is True


def test_locate_rejects_flat_line_template(tmp_path):
    """实测③的假阳性：扁长直线模板不能用来给概览图定位。"""
    flat = _flat_polyline()
    s = _store(tmp_path, flat)
    assert s.locate(_bent_polyline(), city_hint="上海") is None


def test_locate_rejects_city_mismatch(tmp_path):
    s = _store(tmp_path, _bent_polyline(), city="上海")
    assert s.locate(_bent_polyline(), city_hint="北京") is None


def test_locate_empty_store_returns_none(tmp_path):
    assert TemplateStore(tmp_path / "empty.json").locate(_bent_polyline()) is None


def test_locate_template_in_query_needs_high_coverage():
    assert MIN_LOCATE_COVERAGE_B >= 0.85  # 守住「不凭空外推大段坐标」


# --- 流水线接入 ---------------------------------------------------------

@pytest.fixture
def patched(monkeypatch, tmp_path):
    """把 OCR / 轨迹提取 / 地理编码换成桩件，词典指向 tmp。"""
    px = _bent_polyline()
    track = TrackResult(
        polyline=np.asarray(px, dtype=float),
        start_marker=(350, 300), end_marker=(50, 50),
        mask=np.zeros((400, 500), dtype=np.uint8),
        bbox=(50, 50, 300, 250), confidence=0.95, notes=[],
    )
    calls: list[tuple] = []

    monkeypatch.setattr(P, "extract_track", lambda img, cfg: track)
    monkeypatch.setattr(P, "ocr_texts",
                        lambda img, cfg: [SimpleNamespace(text=t)
                                         for t in OCR_OVERVIEW_8])
    monkeypatch.setattr(P, "build_anchors", lambda img, items, cfg: [])
    monkeypatch.setattr(
        P.RidePipeline, "_resolve_anchors",
        lambda self, anchors, city, allow_online, warnings: (
            calls.append((city, allow_online)), [])[1],
    )
    cfg = replace(CONFIG, lexicon_path=tmp_path / "lexicon.json")
    return cfg, calls, px


def _img():
    return np.zeros((400, 500, 3), dtype=np.uint8)


def test_pipeline_hit_returns_reference_result(patched, tmp_path):
    cfg, calls, px = patched
    store = _store(tmp_path, px)
    pipe = P.RidePipeline(cfg, templates=store)
    res = pipe.run(_img(), city="上海", allow_online=False,
                   output_crs="WGS-84", refine=True)

    assert isinstance(res, PipelineResult)
    assert 0 < res.confidence <= TEMPLATE_CONFIDENCE_CAP
    assert res.diagnostics["source"] == "template"
    assert res.diagnostics["template_derived"] is True
    assert res.track_points == 64
    assert calls == [], "命中时必须跳过在线地理编码"
    assert res.refine_context is None
    assert any("参考级定位" in w for w in res.warnings)
    gj = res.geojson["features"][0]
    assert gj["properties"]["quality"] == "reference_only_unverified"
    assert gj["properties"]["crs"] == "WGS-84"
    assert len(gj["geometry"]["coordinates"]) == 64
    assert res.diagnostics["input_quality"]["usable"] is False
    assert res.diagnostics["lexicon_write_guard"]["written"] == 0
    assert _need_background_refine(res, True, False) is False
    assert _capture_template(res, {"geojson_gcj02": res.geojson_gcj02,
                                   "track_polyline_px": res.track_polyline_px},
                             city="上海", shot_id="t")["reason"] == \
        "template_derived_result"


def test_pipeline_miss_falls_back_to_original_flow(patched, tmp_path):
    cfg, calls, px = patched
    empty = TemplateStore(tmp_path / "empty2.json")
    pipe = P.RidePipeline(cfg, templates=empty)
    res = pipe.run(_img(), city="上海", allow_online=False, refine=False)
    assert calls, "未命中必须继续走原来的地理编码流程"
    assert res.diagnostics.get("source") != "template"
    assert any("模板库里没有" in w for w in res.warnings)


def test_pipeline_without_store_is_unchanged(patched, tmp_path):
    cfg, calls, px = patched
    pipe = P.RidePipeline(cfg)  # 不注入 → 整条兜底通道关闭
    pipe.run(_img(), city="上海", allow_online=False, refine=False)
    assert calls, "未注入模板库时必须完全保持原有行为"
