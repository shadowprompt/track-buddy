"""导出器公共基座：坐标系、时间插值与 XML 工具，所有文件格式共用。

**共同输入约定**：

- ``coords``：``[lon, lat]`` 序列，**WGS-84**（``payload["geojson"]``）。
  中国境内偏移坐标系（GCJ-02）进 GPS 类文件会让整条轨迹在
  Strava / Garmin 上偏约 290 米——导出层不收 GCJ-02。
- ``ride_summary``：``ride_summary.parse_ride_summary`` 的输出，字段
  缺失为 ``None``；导出层绝不猜测填充。
- ``source_title``：输入源画像标题（美团骑行 / 未来高德…），进文件名
  与 ``<name>``。

**时间插值**：截图识别拿不到逐点时刻，只有「开始时间 + 总时长」两个
端点，按累计里程比例分配（匀速假设），按里程而不是点序号分配——
识别点间距不均，按序号会产生忽快忽慢的假速度曲线。

**时区**：订单页时间戳是东八区本地时间（中国无夏令时，偏移恒定），
导出格式的 ``<time>`` 均要求 UTC；固定 +8h 换算，不引入 zoneinfo
依赖（Windows 缺 IANA 时区库）。
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Sequence

# 订单页时间戳所在时区：东八区
TZ_OFFSET = timedelta(hours=8)


def local_to_utc(dt_local: datetime) -> datetime:
    """东八区本地时间 → UTC。中国无夏令时，偏移恒定。"""
    return dt_local - TZ_OFFSET


def iso_z(dt_utc: datetime) -> str:
    """UTC datetime → ISO8601 带 Z 后缀（GPX/TCX 规范格式，毫秒截断）。"""
    return dt_utc.replace(microsecond=0).isoformat() + "Z"


def xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_started_at(value: Any) -> datetime | None:
    """解析 ``ride_summary["started_at"]``（"YYYY-MM-DDTHH:MM:SS"）。"""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def haversine_m(a: Sequence[float], b: Sequence[float]) -> float:
    """两点球面距离（米）。a/b 为 [lon, lat]。"""
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(math.sqrt(min(1.0, h)))


def cumulative_distances(coords: Sequence[Sequence[float]]) -> list[float]:
    """相邻点球面距离的累计值（米），首点为 0。"""
    if not coords:
        return []
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + haversine_m(coords[i - 1], coords[i]))
    return cum


def track_times(
    coords: Sequence[Sequence[float]],
    start: datetime,
    duration_s: float,
) -> list[str]:
    """按累计里程比例给每个轨迹点分配时刻（UTC ISO 字符串）。

    里程为 0 或时长非法时退化为按点序号均匀分布。全程在 UTC 域内
    插值——先换算再取中间时刻，顺序反了会整条时间轴冒充 UTC 偏 8 小时
    （2026-11-14 踩过，测试锁定）。
    """
    cum = cumulative_distances(coords)
    total = cum[-1] if cum else 0.0
    n = len(coords)
    if n == 0:
        return []
    if total <= 0 or duration_s <= 0:
        ratios = [i / max(1, n - 1) for i in range(n)]
    else:
        ratios = [c / total for c in cum]
    start_utc = local_to_utc(start)
    end_utc = start_utc + timedelta(seconds=float(duration_s))
    return [iso_z(start_utc + (end_utc - start_utc) * r) for r in ratios]
