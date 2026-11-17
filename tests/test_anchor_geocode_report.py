"""锚点地理编码诊断明细（diagnostics.anchor_geocode）。

背景（2026-11-17）：「20 个锚点只有 1 个命中」这类问题光看 warnings
的汇总数字无从排查——是哪些名字没命中？配额耗尽还是 OSM 缺收录？
_resolve_anchors 现在为每个候选名产出一条结局记录（matched / missed /
skipped），经 diagnostics 透传到前端，由 console.table 展示。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.ocr_anchor import Anchor
from app.pipeline import RidePipeline


def _pipe() -> RidePipeline:
    """裸管线：词典/地理编码全部打桩（本测试不联网）。"""
    p = RidePipeline.__new__(RidePipeline)
    p.lexicon = SimpleNamespace(
        lookup=lambda name, city: False,
        lookup_exact=lambda name: False,
    )
    p.geocoder = SimpleNamespace(
        geocode=lambda name, **kw: [],
        _online_failures=["高德 USER_DAILY_QUERY_OVER_LIMIT"],
        _last_error="",
    )
    p.cfg = SimpleNamespace(allow_online=False)
    return p


def _anchor(name: str) -> Anchor:
    return Anchor(name=name, px=100.0, py=100.0, kind="poi", weight=0.8)


def test_report_records_miss_and_online_reason() -> None:
    p = _pipe()
    warnings: list[str] = []
    pairs = p._resolve_anchors(
        [_anchor("诺富特"), _anchor("上海荟聚Livat")], "上海",
        allow_online=False, warnings=warnings)
    assert pairs == []
    report = p._last_geocode_report
    assert {r["name"] for r in report} == {"诺富特", "上海荟聚Livat"}
    for r in report:
        assert r["status"] == "missed"
        assert "USER_DAILY_QUERY_OVER_LIMIT" in r["note"]


def test_report_marks_matched_and_over_quota_targets() -> None:
    p = _pipe()

    def fake_geocode(name, **kw):
        if name == "巴奴毛肚火锅":
            return [SimpleNamespace(
                lon=121.41, lat=31.23, quality=0.8, source="tianditu",
                kind="poi", address="")]
        return []

    p.geocoder.geocode = fake_geocode
    # 21 个候选：1 个可命中 + 20 个 MISS 名额占满 → 第 21 个被截断
    names = ["巴奴毛肚火锅"] + [f"占位锚点{i:02d}" for i in range(20)]
    p._resolve_anchors([_anchor(n) for n in names], "上海",
                       allow_online=False, warnings=[])
    report = p._last_geocode_report
    assert len(report) == 21
    hit = next(r for r in report if r["name"] == "巴奴毛肚火锅")
    assert hit["status"] == "matched"
    assert hit["source"] == "tianditu"
    assert hit["n_candidates"] == 1
    skipped = [r for r in report if r["status"] == "skipped"]
    assert len(skipped) == 1 and skipped[0]["name"] == "占位锚点19"
    assert "名额" in skipped[0]["note"]


def test_diagnostics_carry_report_in_success_path() -> None:
    """成功路径的 diagnostics 必须包含 anchor_geocode 字段（前端依赖）。"""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "app" / "pipeline.py").read_text(encoding="utf-8")
    # 两处 _base_diagnostics（成功 + 配准失败）都要透传明细
    assert src.count("anchor_geocode=") >= 2
    assert "_last_geocode_report" in src


def test_frontend_prints_geocode_table() -> None:
    """前端 console.table 打印锚点明细 + 精修轮询也复用同一打印。"""
    html = (Path(__file__).resolve().parents[1]
            / "src" / "web" / "index.html").read_text(encoding="utf-8")
    assert "function logDiag(" in html
    assert "anchor_geocode" in html
    assert "console.table" in html
    assert html.count("logDiag(") >= 3  # 定义 + 识别响应 + 精修轮询
