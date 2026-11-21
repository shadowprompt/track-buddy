"""CLI 轨迹/概要解耦：无概要出纯轨迹 GPX，有概要出时间轴 GPX + TCX。

不跑完整 Pipeline（慢且依赖网络）：只验证 --summary 的解析接线、
分层导出调用与摘要字段，Pipeline 本身由 e2e 覆盖。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.cli import _summary as cli_summary
from app.exporters import build_exports, build_track_exports, has_time_axis
from app.pipeline import PipelineResult

COORDS = [[121.5000, 31.2000], [121.5010, 31.2010], [121.5020, 31.2020]]
SUMMARY = {"started_at": "2026-11-11T20:20:26", "duration_s": 600,
           "distance_km": 0.5}


def _make_res() -> PipelineResult:
    return PipelineResult(
        geojson={"type": "FeatureCollection", "features": [{
            "geometry": {"type": "LineString", "coordinates": COORDS}}]},
        confidence=0.9,
        geojson_gcj02={"type": "FeatureCollection", "features": []},
        track_points=3,
        distance_m=500.0,
        warnings=[],
    )


def test_summary_exports_include_tcx() -> None:
    out = build_exports(COORDS, ride_summary=SUMMARY)
    assert set(out) == {"gpx", "tcx"}
    assert "<time>" in out["gpx"]
    assert "<Time>" in out["tcx"]


def test_track_exports_are_pure_and_never_include_tcx() -> None:
    out = build_track_exports(COORDS)
    assert set(out) == {"gpx"}
    assert "<time>" not in out["gpx"]
    # 纯轨迹不读概要：即使外部有完整概要也拿不到 TCX
    assert "tcx" not in out


def test_cli_summary_reports_export_tiers() -> None:
    files = {"geojson_wgs84": "x.geojson", "track_gpx": "x.gpx",
             "track_tcx": "x.tcx"}
    s = cli_summary(_make_res(), ".", files, 0.1, SUMMARY)
    assert s["ride_summary"] == SUMMARY
    assert set(s["exports"]) == {"track_gpx", "track_tcx"}

    files_track_only = {"geojson_wgs84": "x.geojson", "track_gpx": "x.gpx"}
    s2 = cli_summary(_make_res(), ".", files_track_only, 0.1, None)
    assert s2["ride_summary"] is None
    assert set(s2["exports"]) == {"track_gpx"}


def test_cli_summary_backward_compatible_without_summary_arg() -> None:
    # e2e.py 的旧调用只有 4 个参数，新增第 5 个参数必须有默认值
    s = cli_summary(_make_res(), ".", {}, 0.1)
    assert s["ride_summary"] is None
    assert s["exports"] == []


def test_summary_image_parser_rejects_garbage(tmp_path: Path) -> None:
    from app.ride_summary import parse_summary_image_bytes

    assert parse_summary_image_bytes(b"") is None
    assert parse_summary_image_bytes(b"not an image") is None
