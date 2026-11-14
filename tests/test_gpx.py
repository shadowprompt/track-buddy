"""GPX 合成契约：轨迹坐标 + 运动概要 → 可导入 Strava/Garmin 的 GPX 1.1。

关键点：
- 坐标必须是 WGS-84（``payload["geojson"]``），GCJ-02 进 GPX 会整体偏 ~290m；
- ``<trkpt><time>`` 按累计里程比例插值（运动平台靠它算时长/配速）；
- 订单页时间是东八区本地时间，GPX ``<time>`` 规范要求 UTC（Z 后缀）；
- 概要缺失时仍输出纯轨迹 GPX，绝不拒绝导出。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from app.gpx import _local_to_utc, build_gpx

# 简单 3 点轨迹（[lon, lat]，WGS-84，点间距 ~500m 量级）
COORDS = [[121.0, 31.0], [121.01, 31.0], [121.02, 31.0]]

SUMMARY = {
    "started_at": "2026-11-11T20:20:26",
    "duration_s": 1235,  # 20分35秒
    "distance_km": 3.5,
    "calories_kcal": 203,
    "carbon_g": 172,
    "available": True,
}


def _parse(text: str) -> ET.Element:
    return ET.fromstring(text)


def _trkpts(root: ET.Element) -> list[ET.Element]:
    return root.findall("./{*}trk/{*}trkseg/{*}trkpt")


def test_basic_structure_and_point_count() -> None:
    root = _parse(build_gpx(COORDS))
    assert root.tag.endswith("gpx")
    pts = _trkpts(root)
    assert len(pts) == 3
    # lat/lon 属性顺序：GeoJSON 是 [lon, lat]，GPX 是 lat/lon，必须换位
    assert pts[0].get("lon").startswith("121.")
    assert pts[0].get("lat").startswith("31.")


def test_track_times_span_start_to_end() -> None:
    """首点=开始时间、末点=开始+时长，且单调递增。"""
    root = _parse(build_gpx(COORDS, ride_summary=SUMMARY))
    times = [p.find("{*}time").text for p in _trkpts(root)]
    assert len(times) == 3
    assert times[0] == "2026-11-11T12:20:26Z"   # 东八区 → UTC
    assert times[-1] == "2026-11-11T12:41:01Z"  # +20分35秒
    assert times == sorted(times)


def test_time_interp_follows_distance_not_point_index() -> None:
    """中间点在起点之后 1/2 里程处，时刻也应约为一半（间距均匀时）。"""
    coords = [[121.0, 31.0], [121.01, 31.0], [121.02, 31.0]]
    root = _parse(build_gpx(coords, ride_summary=SUMMARY))
    times = [p.find("{*}time").text for p in _trkpts(root)]
    assert times[1] == "2026-11-11T12:30:43Z"  # 12:20:26 + 1235/2 = 617.5s


def test_local_to_utc_is_fixed_east_eight() -> None:
    from datetime import datetime
    assert _local_to_utc(datetime(2026, 1, 1, 0, 30)) == \
        datetime(2025, 12, 31, 16, 30)
    # 中国无夏令时：7 月偏移不变
    assert _local_to_utc(datetime(2026, 7, 1, 8, 0)) == \
        datetime(2026, 7, 1, 0, 0)


def test_no_summary_still_exports_valid_gpx() -> None:
    root = _parse(build_gpx(COORDS))
    assert len(_trkpts(root)) == 3
    # 没有概要就没有任何 <time>
    assert not root.findall(".//{*}time")
    assert not root.findall(".//{*}extensions")


def test_duration_without_start_has_no_abs_times() -> None:
    """只有时长没有开始时间：给不出绝对时刻序列，宁缺勿错。"""
    root = _parse(build_gpx(
        COORDS, ride_summary={"started_at": None, "duration_s": 600}))
    assert not root.findall(".//{*}trkpt/{*}time")


def test_summary_extensions_written() -> None:
    root = _parse(build_gpx(COORDS, ride_summary=SUMMARY))
    ext = root.find("./{*}trk/{*}extensions")
    assert ext is not None
    vals = {re.sub(r"\{.*\}", "", c.tag): c.text for c in ext}
    assert vals == {"distanceKm": "3.5", "durationS": "1235",
                    "caloriesKcal": "203", "carbonG": "172"}


def test_partial_summary_only_writes_present_fields() -> None:
    root = _parse(build_gpx(
        COORDS,
        ride_summary={"started_at": "2026-11-11T20:20:26",
                      "duration_s": None, "carbon_g": None}))
    ext = root.find("./{*}trk/{*}extensions")
    tags = {re.sub(r"\{.*\}", "", c.tag) for c in (ext or [])}
    assert tags == set()
    # metadata 里有开始时间（UTC），但没有 trkpt 时间
    assert root.find("./{*}metadata/{*}time") is not None


def test_name_is_localized_and_escaped() -> None:
    xml = build_gpx(
        COORDS,
        ride_summary={"started_at": "2026-11-11T20:20:26"},
        source_title="美团骑行",
    )
    assert "<name>美团骑行 2026-11-11 20:20</name>" in xml
    # 特殊字符必须转义，否则整份 XML 解析失败
    bad = build_gpx(COORDS, source_title='a<b>&"c')
    ET.fromstring(bad)  # 不抛异常即通过


def test_zero_distance_uniform_time_fallback() -> None:
    """所有点重合（里程 0）时按序号均匀分布，不许除零。"""
    coords = [[121.0, 31.0]] * 3
    root = _parse(build_gpx(coords, ride_summary=SUMMARY))
    times = [p.find("{*}time").text for p in _trkpts(root)]
    assert times[0] == "2026-11-11T12:20:26Z"
    assert times[-1] == "2026-11-11T12:41:01Z"


def test_bad_started_at_is_ignored() -> None:
    root = _parse(build_gpx(
        COORDS, ride_summary={"started_at": "不是时间", "duration_s": 600}))
    assert not root.findall(".//{*}time")


def test_empty_coords_produces_empty_track() -> None:
    root = _parse(build_gpx([], ride_summary=SUMMARY))
    assert _trkpts(root) == []


@pytest.mark.parametrize("lat,lon", [(95.0, 121.0), (31.0, 190.0)])
def test_out_of_range_coords_still_render(lat: float, lon: float) -> None:
    """异常坐标不做静默修正（识别层负责），但 XML 必须合法。"""
    xml = build_gpx([[lon, lat]])
    assert f'lat="{lat:.6f}"' in xml


# --- server 接线 -----------------------------------------------------------

def _payload_with_coords(coords) -> dict:
    return {
        "geojson": {"features": [{
            "geometry": {"type": "LineString", "coordinates": coords},
        }]},
        "ride_summary": dict(SUMMARY),
    }


def test_attach_gpx_sets_payload_field() -> None:
    from app.server import _attach_gpx
    payload = _payload_with_coords(COORDS)
    _attach_gpx(payload)
    root = _parse(payload["gpx"])
    assert len(_trkpts(root)) == 3
    # 概要扩展也在：一条 payload 同时携带轨迹与运动元数据
    assert root.find("./{*}trk/{*}extensions") is not None


def test_attach_gpx_skips_without_coords() -> None:
    from app.server import _attach_gpx
    payload = _payload_with_coords([])
    _attach_gpx(payload)
    assert "gpx" not in payload
    payload2 = {}
    _attach_gpx(payload2)
    assert "gpx" not in payload2


# --- 前端契约（静态源码断言，与 test_frontend_ride_summary.py 同法） -------

def test_frontend_gpx_download_link_wired() -> None:
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1]
            / "src" / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="dlGPX"' in html
    assert "d.gpx" in html
    # 无 GPX 时隐藏链接，不残留上一轮的 href
    assert "$('dlGPX').style.display = 'none';" in html
