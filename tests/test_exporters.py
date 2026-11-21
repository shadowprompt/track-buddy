"""运动文件导出层契约：轨迹坐标 + 运动概要 → GPX 1.1 / TCX v2。

关键点：
- 坐标必须是 WGS-84（``payload["geojson"]``），GCJ-02 进 GPS 类文件会整体偏 ~290m；
- ``<trkpt><time>`` / ``<Trackpoint><Time>`` 按累计里程比例插值（运动平台靠它算时长/配速）；
- 订单页时间是东八区本地时间，导出 ``<time>`` 规范要求 UTC（Z 后缀）；
- GPX 概要缺失仍输出纯轨迹；TCX 无时间轴必须返回 None（规范要求逐点时间，宁缺勿错）。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from app.exporters import build_exports, export_one, supported_formats
from app.exporters.base import local_to_utc
from app.exporters.gpx import build_gpx
from app.exporters.tcx import build_tcx

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
    assert local_to_utc(datetime(2026, 1, 1, 0, 30)) == \
        datetime(2025, 12, 31, 16, 30)
    # 中国无夏令时：7 月偏移不变
    assert local_to_utc(datetime(2026, 7, 1, 8, 0)) == \
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


def test_attach_exports_sets_gpx_and_tcx() -> None:
    from app.server import _attach_exports
    payload = _payload_with_coords(COORDS)
    _attach_exports(payload)
    gpx_root = _parse(payload["gpx"])
    assert len(_trkpts(gpx_root)) == 3
    # 概要扩展也在：一条 payload 同时携带轨迹与运动元数据
    assert gpx_root.find("./{*}trk/{*}extensions") is not None
    # 概要齐全时 TCX 同步产出（模块化注册表一次跑全部格式）
    assert "tcx" in payload


def test_attach_exports_tcx_missing_without_time_axis() -> None:
    from app.server import _attach_exports
    payload = _payload_with_coords(COORDS)
    payload["ride_summary"] = {"started_at": None, "duration_s": None}
    _attach_exports(payload)
    # GPX 照常产出，TCX 缺席（键不存在 → 前端不展示入口），不给多余告警
    assert "gpx" in payload
    assert "tcx" not in payload
    assert not any("TCX" in w for w in payload.get("warnings") or [])


def test_attach_exports_no_summary_is_track_only_no_warning() -> None:
    """只有轨迹截图（无概要）是一等公民路径：出 GPX，不给残缺告警。"""
    from app.server import _attach_exports
    payload = _payload_with_coords(COORDS)
    payload.pop("ride_summary", None)  # 没传概要时 _attach_ride_summary 不写这个键
    _attach_exports(payload)
    assert "gpx" in payload
    assert "tcx" not in payload
    assert not any("TCX" in w for w in payload.get("warnings") or [])


def test_has_time_axis_and_track_only_formats() -> None:
    from app.exporters import has_time_axis, track_only_formats

    assert track_only_formats() == ("gpx",)
    assert has_time_axis({"started_at": "2026-11-11T20:20:26",
                          "duration_s": 600})
    assert not has_time_axis({"started_at": None, "duration_s": 600})
    assert not has_time_axis({"started_at": "2026-11-11T20:20:26",
                              "duration_s": None})
    assert not has_time_axis(None)


def test_attach_exports_skips_without_coords() -> None:
    from app.server import _attach_exports
    payload = _payload_with_coords([])
    _attach_exports(payload)
    assert "gpx" not in payload and "tcx" not in payload
    payload2 = {}
    _attach_exports(payload2)
    assert "gpx" not in payload2 and "tcx" not in payload2


# --- 导出层注册表 -----------------------------------------------------------

def test_registry_and_dispatch() -> None:
    assert "gpx" in supported_formats()
    assert "tcx" in supported_formats()
    out = build_exports(COORDS, ride_summary=dict(SUMMARY),
                        source_title="美团骑行")
    assert set(out) == {"gpx", "tcx"}
    # 单格式导出 + 未知格式报错（FIT 尚未实现，错误信息要说清）
    assert export_one("gpx", COORDS).startswith("<?xml")
    with pytest.raises(ValueError, match="FIT"):
        export_one("fit", COORDS)


def test_attach_exports_uses_source_profile_title() -> None:
    from app.server import _attach_exports
    payload = _payload_with_coords(COORDS)
    _attach_exports(payload)
    # 输入源画像标题（meituan → 美团骑行）进入文件名
    assert "美团骑行" in payload["gpx"]
    assert "美团骑行" in payload["tcx"]


# --- TCX 导出器 -------------------------------------------------------------

def _tcx_pts(root: ET.Element) -> list[ET.Element]:
    return root.findall("./{*}Activities/{*}Activity/{*}Lap/{*}Track/"
                        "{*}Trackpoint")


def test_tcx_structure_and_lap_fields() -> None:
    xml = build_tcx(COORDS, ride_summary=dict(SUMMARY),
                    source_title="美团骑行")
    root = _parse(xml)
    act = root.find("./{*}Activities/{*}Activity")
    assert act is not None and act.get("Sport") == "Biking"
    lap = act.find("{*}Lap")
    assert lap.get("StartTime") == "2026-11-11T12:20:26Z"
    assert lap.find("{*}TotalTimeSeconds").text == "1235"
    assert lap.find("{*}DistanceMeters").text == "3500"
    assert lap.find("{*}Calories").text == "203"
    assert lap.find("{*}TriggerMethod").text == "Manual"
    assert len(_tcx_pts(root)) == 3


def test_tcx_point_order_and_times() -> None:
    """GeoJSON 是 [lon, lat]，TCX 是 Latitude/LongitudeDegrees，必须换位。"""
    root = _parse(build_tcx(COORDS, ride_summary=dict(SUMMARY)))
    pts = _tcx_pts(root)
    assert pts[0].find("{*}Position/{*}LatitudeDegrees").text.startswith("31.")
    assert pts[0].find("{*}Position/{*}LongitudeDegrees").text.startswith("121.")
    times = [p.find("{*}Time").text for p in pts]
    assert times[0] == "2026-11-11T12:20:26Z"
    assert times[-1] == "2026-11-11T12:41:01Z"


def test_tcx_requires_time_axis() -> None:
    """无开始时间/时长 → None（TCX 规范要求逐点 Time，宁缺勿错）。"""
    assert build_tcx(COORDS, ride_summary={"started_at": None,
                                           "duration_s": 600}) is None
    assert build_tcx(COORDS, ride_summary={"started_at":
                                           "2026-11-11T20:20:26",
                                           "duration_s": None}) is None
    assert build_tcx(COORDS, ride_summary=None) is None


def test_tcx_distance_prefers_order_value_falls_back_to_geom() -> None:
    # 订单里程优先（平台记账）
    root = _parse(build_tcx(COORDS, ride_summary=dict(SUMMARY),
                            geom_distance_m=3300))
    assert root.find(".//{*}DistanceMeters").text == "3500"
    # 无订单里程退化为几何里程
    root2 = _parse(build_tcx(
        COORDS, ride_summary={"started_at": "2026-11-11T20:20:26",
                              "duration_s": 600},
        geom_distance_m=3300))
    assert root2.find(".//{*}DistanceMeters").text == "3300"


def test_tcx_calories_zero_when_missing() -> None:
    """Calories 是 schema 必填字段：缺失填 0（合法且诚实），不编造。"""
    root = _parse(build_tcx(
        COORDS, ride_summary={"started_at": "2026-11-11T20:20:26",
                              "duration_s": 600}))
    assert root.find(".//{*}Calories").text == "0"


# --- 前端契约（静态源码断言，与 test_frontend_ride_summary.py 同法） -------

def test_frontend_download_links_wired() -> None:
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1]
            / "src" / "web" / "index.html").read_text(encoding="utf-8")
    # GeoJSON 只导出 WGS-84（GCJ-02 下载已收敛）
    assert 'id="dlG"' in html
    assert "trajectory_wgs84.geojson" in html
    assert 'id="dl"' not in html.replace('id="dlG"', "").replace(
        'id="dlGPX"', "").replace('id="dlTCX"', "")
    assert "trajectory_gcj02.geojson" not in html
    # GPX / TCX：无产出时隐藏链接，不残留上一轮的 href
    assert 'id="dlGPX"' in html and "d.gpx" in html
    assert "$('dlGPX').style.display = 'none';" in html
    assert 'id="dlTCX"' in html and "d.tcx" in html
    assert "$('dlTCX').style.display = 'none';" in html
