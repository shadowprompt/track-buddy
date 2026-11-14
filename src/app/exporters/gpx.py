"""GPX 1.1 导出器：轨迹 + 概要 → 可导入 Strava / Garmin / 两步路的 GPX。

GPX 对时间的要求最宽松：``<time>`` 全部可选，没有概要也能输出纯轨迹
GPX（只是导入运动平台后没有时长/配速）。概要字段写进 ``<trk>
<extensions>`` 自定义命名空间 ``mr:``，平台忽略未知扩展不影响导入。
"""
from __future__ import annotations

from typing import Any, Sequence

from .base import (
    iso_z,
    local_to_utc,
    parse_started_at,
    track_times,
    xml_escape,
)

# GPX 1.1 固定命名空间；mr 是本项目自定义扩展前缀（meituan-ride）
_GPX_NS = (
    'xmlns="http://www.topografix.com/GPX/1/1" '
    'xmlns:mr="http://github.com/meituan-ride/gpx-ext" '
    'creator="meituan-ride" version="1.1"'
)


def build_gpx(
    coords: Sequence[Sequence[float]],
    *,
    ride_summary: dict[str, Any] | None = None,
    source_title: str = "骑行",
    geom_distance_m: float | None = None,
) -> str:
    """轨迹坐标（``[lon, lat]``，**WGS-84**）+ 概要 → GPX 1.1 文本。

    ``geom_distance_m`` 与 TCX 保持注册表统一签名，GPX 不用它
    （几何里程不写入文件，避免与订单里程两套数字打架）。
    概要缺失时仍输出纯轨迹 GPX（无 ``<time>``），绝不因概要失败
    拒绝导出——轨迹本身就是有效的运动记录。
    """
    s = ride_summary or {}
    start = parse_started_at(s.get("started_at"))
    duration_s = s.get("duration_s")

    name = f"{source_title}轨迹"
    if start is not None:
        name = f"{source_title} {start.strftime('%Y-%m-%d %H:%M')}"

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<gpx {_GPX_NS}>",
        "<metadata>",
        f"  <name>{xml_escape(name)}</name>",
    ]
    if start is not None:
        lines.append(f"  <time>{iso_z(local_to_utc(start))}</time>")
    lines.append("</metadata>")

    lines.append("<trk>")
    lines.append(f"  <name>{xml_escape(name)}</name>")

    # 概要扩展：平台忽略未知命名空间，不影响导入
    ext: list[str] = []
    if s.get("distance_km") is not None:
        ext.append(f"    <mr:distanceKm>{s['distance_km']}</mr:distanceKm>")
    if duration_s is not None:
        ext.append(f"    <mr:durationS>{duration_s}</mr:durationS>")
    if s.get("calories_kcal") is not None:
        ext.append(f"    <mr:caloriesKcal>{s['calories_kcal']}"
                   "</mr:caloriesKcal>")
    if s.get("carbon_g") is not None:
        ext.append(f"    <mr:carbonG>{s['carbon_g']}</mr:carbonG>")
    if ext:
        lines.append("  <extensions>")
        lines.extend(ext)
        lines.append("  </extensions>")

    # trkpt 时间：需要同时有开始时间与时长才能给出绝对时刻序列
    times: list[str] = []
    if start is not None and duration_s:
        times = track_times(coords, start, float(duration_s))

    lines.append("  <trkseg>")
    for i, (lon, lat) in enumerate(coords):
        lines.append(f'    <trkpt lat="{lat:.6f}" lon="{lon:.6f}">')
        if i < len(times):
            lines.append(f"      <time>{times[i]}</time>")
        lines.append("    </trkpt>")
    lines.append("  </trkseg>")
    lines.append("</trk>")
    lines.append("</gpx>")
    return "\n".join(lines) + "\n"
