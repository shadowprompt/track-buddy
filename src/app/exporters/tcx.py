"""TCX（TrainingCenterDatabase v2）导出器：Garmin Connect / Strava 的
另一种通用导入格式。

**与 GPX 的关键差异**：TCX 规范要求每个 ``<Trackpoint>`` 必带
``<Time>``，``<Lap>`` 必带 ``TotalTimeSeconds / DistanceMeters /
Calories``——没有时间轴（缺开始时间或时长）就**产不出合规 TCX**，
此时本导出器返回 ``None``（宁缺勿错，不给半合规文件），前端隐藏
下载链接。这是「后续加 FIT 等格式」要复刻的契约样板：数据不足就
缺席，绝不输出坏文件。

**数据来源优先级**：
- ``DistanceMeters``：订单里程（平台记账，骑行的官方距离）优先，
  退化用轨迹几何里程 ``geom_distance_m``；
- ``Calories``：订单卡路里，缺失填 0（schema 必填字段，0 合法且
  诚实——比编造一个数好）。
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

_TCX_NS = (
    'xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'creator="meituan-ride"'
)


def build_tcx(
    coords: Sequence[Sequence[float]],
    *,
    ride_summary: dict[str, Any] | None = None,
    source_title: str = "骑行",
    geom_distance_m: float | None = None,
) -> str | None:
    """轨迹（WGS-84）+ 概要 → TCX v2 文本；无时间轴返回 ``None``。"""
    s = ride_summary or {}
    start = parse_started_at(s.get("started_at"))
    duration_s = s.get("duration_s")
    # TCX 规范：Trackpoint 必带 <Time>，Lap 必带 TotalTimeSeconds——
    # 时间轴两端点缺一就给不出合规文件。
    if start is None or not duration_s:
        return None

    times = track_times(coords, start, float(duration_s))
    start_utc = local_to_utc(start)

    distance_m: float | None = None
    if s.get("distance_km") is not None:
        distance_m = float(s["distance_km"]) * 1000.0
    elif geom_distance_m:
        distance_m = float(geom_distance_m)
    # Lap schema 必填字段：TotalTimeSeconds / DistanceMeters / Calories /
    # TriggerMethod。Calories 缺失填 0（合法且诚实）。
    total_time = int(round(float(duration_s)))
    distance_out = int(round(distance_m)) if distance_m else 0
    calories = int(round(float(s["calories_kcal"]))) \
        if s.get("calories_kcal") is not None else 0

    name = f"{source_title} {start.strftime('%Y-%m-%d %H:%M')}"
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<TrainingCenterDatabase {_TCX_NS}>",
        "<Activities>",
        '  <Activity Sport="Biking">',
        f"    <Id>{iso_z(start_utc)}</Id>",
        f'    <Lap StartTime="{iso_z(start_utc)}">',
        f"      <TotalTimeSeconds>{total_time}</TotalTimeSeconds>",
        f"      <DistanceMeters>{distance_out}</DistanceMeters>",
        f"      <Calories>{calories}</Calories>",
        "      <TriggerMethod>Manual</TriggerMethod>",
        "      <Track>",
    ]
    for i, (lon, lat) in enumerate(coords):
        lines.append("        <Trackpoint>")
        if i < len(times):
            lines.append(f"          <Time>{times[i]}</Time>")
        lines.append("          <Position>")
        lines.append(f"            <LatitudeDegrees>{lat:.6f}"
                     "</LatitudeDegrees>")
        lines.append(f"            <LongitudeDegrees>{lon:.6f}"
                     "</LongitudeDegrees>")
        lines.append("          </Position>")
        lines.append("        </Trackpoint>")
    lines.append("      </Track>")
    lines.append("      <Notes>" + xml_escape(name) + "</Notes>")
    lines.append("    </Lap>")
    lines.append("  </Activity>")
    lines.append("</Activities>")
    lines.append("</TrainingCenterDatabase>")
    return "\n".join(lines) + "\n"
