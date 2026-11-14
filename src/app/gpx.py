"""GPX 1.1 导出：把轨迹坐标与运动概要合成一份可导入 Strava / Garmin /
佳明 Connect / 两步路等平台的 GPX 文件。

**为什么要合成 GPX**：识别结果本来就是「轨迹几何 + 运动元数据」的完整
运动记录，但 GeoJSON 只是地理数据格式——导入运动平台后没有时间轴，
Strava 会把这条轨迹当成「无时间的 GPS 记录」，时长、配速、平均速度
全部缺失。GPX 的 ``<trkpt><time>`` 是运动平台唯一通用的时间证据。

**时间插值策略**：截图识别拿不到每个点的真实时刻，只有「开始时间 +
总时长」两个端点。按**累计里程比例**分配中间时刻（匀速假设）：

    t_i = start + (cumdist_i / total) * duration_s

按里程而不是按点序号分配，是因为识别点间距严重不均（直线段密、
弯道稀），按序号分配会产生忽快忽慢的假速度曲线。

**时区**：订单页时间戳是本地时间（东八区、无夏令时）。GPX 规范要求
``<time>`` 为 UTC（带 Z），Strava 等平台按 UTC 解析再按设备时区显示，
写死东八区换算即可，不引入 zoneinfo/tzdata 依赖（Windows 环境缺
IANA 时区库，得不偿失）。

**extensions**：概要字段（里程/时长/卡路里/减碳）写进 ``<trk>
<extensions>`` 自定义命名空间。平台会忽略未知扩展，不影响导入；
留在文件里是给人工核对和数据回流用的。
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Sequence

# 订单页时间戳所在时区：东八区（中国无夏令时，偏移恒定）
_TZ_OFFSET = timedelta(hours=8)

# GPX 1.1 固定命名空间；mr 是本项目自定义扩展前缀（meituan-ride）
_GPX_NS = (
    'xmlns="http://www.topografix.com/GPX/1/1" '
    'xmlns:mr="http://github.com/meituan-ride/gpx-ext" '
    'creator="meituan-ride" version="1.1"'
)


def _utc_z(dt_local: datetime) -> str:
    """本地时间 → GPX 规范要求的 UTC ISO8601（带 Z 后缀，毫秒截断）。"""
    return dt_local.replace(microsecond=0).isoformat() + "Z"


def _local_to_utc(dt_local: datetime) -> datetime:
    return dt_local - _TZ_OFFSET


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _parse_started_at(value: Any) -> datetime | None:
    """解析 ``ride_summary["started_at"]``（"YYYY-MM-DDTHH:MM:SS"）。"""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def _cumulative_distances(coords: Sequence[Sequence[float]]) -> list[float]:
    """相邻点球面距离的累计值（米），首点为 0。"""
    if not coords:
        return []
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + _haversine_m(coords[i - 1], coords[i]))
    return cum


def _haversine_m(a: Sequence[float], b: Sequence[float]) -> float:
    """两点球面距离（米）。a/b 为 [lon, lat]。"""
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(math.sqrt(min(1.0, h)))


def _track_times(
    coords: Sequence[Sequence[float]],
    start: datetime,
    duration_s: float,
) -> list[str]:
    """按累计里程比例给每个轨迹点分配时刻（UTC ISO 字符串）。"""
    cum = _cumulative_distances(coords)
    total = cum[-1] if cum else 0.0
    n = len(coords)
    if n == 0:
        return []
    if total <= 0 or duration_s <= 0:
        # 退化：里程为 0 或时长非法，按点序号均匀分布
        ratios = [i / max(1, n - 1) for i in range(n)]
    else:
        ratios = [c / total for c in cum]
    # 全程在 UTC 域内插值：start/end 先换算成 UTC，再按比例取中间时刻。
    # 若在本地时间域插值完再换算，`_utc_z` 只做格式化，会导致整条时间轴
    # 偏 8 小时（首点时刻 = 本地时间冒充 UTC）。
    start_utc = _local_to_utc(start)
    end_utc = start_utc + timedelta(seconds=float(duration_s))
    return [
        _utc_z(start_utc + (end_utc - start_utc) * r) for r in ratios
    ]


def build_gpx(
    coords: Sequence[Sequence[float]],
    *,
    ride_summary: dict[str, Any] | None = None,
    source_title: str = "骑行",
) -> str:
    """轨迹坐标（``[lon, lat]``，**WGS-84**）+ 概要 → GPX 1.1 文本。

    ``coords`` 必须是 WGS-84（``payload["geojson"]``），GCJ-02 坐标
    写进 GPX 会让整条轨迹在 Strava/Garmin 上偏移约 290 米。
    概要缺失时仍输出纯轨迹 GPX（无 ``<time>``），绝不因概要失败
    拒绝导出——轨迹本身就是有效的运动记录。
    """
    s = ride_summary or {}
    start = _parse_started_at(s.get("started_at"))
    duration_s = s.get("duration_s")

    name = f"{source_title}轨迹"
    if start is not None:
        name = f"{source_title} {start.strftime('%Y-%m-%d %H:%M')}"

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<gpx {_GPX_NS}>",
        "<metadata>",
        f"  <name>{_xml_escape(name)}</name>",
    ]
    if start is not None:
        lines.append(f"  <time>{_utc_z(_local_to_utc(start))}</time>")
    lines.append("</metadata>")

    lines.append("<trk>")
    lines.append(f"  <name>{_xml_escape(name)}</name>")

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
        times = _track_times(coords, start, float(duration_s))

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
