"""路名归属：回答「这条轨迹骑的是哪几条路」。

为什么需要这一层
----------------
配准只能给出经纬度，但「经纬度对应哪条路」需要另一个数据源。
实测踩过的坑：

1. **第三方路网（OSM）的命名与国内官方命名不一致。** 上海很多道路改过名，
   OSM 往往保留旧名。而美团/腾讯/高德共享同一套官方路网数据。
   实测案例：轨迹实际所在的路，OSM 记作「北翟路」，而用户记忆中的官方名是
   「新渔东路」——两者指的是同一条物理道路，**几何一致、命名不一致**。

2. **第三方路网覆盖残缺。** 实测长宁新泾一带，OSM 有「枣阳路」「泉口路」
   「真北路」等路名但每个只有 1 个节点，主干路段大面积缺失。
   缺失路段会让「离最近道路距离」这类指标失真。

因此路名归属采用两级来源，并明确标注来源可信度：

- **高德路径规划 API**（首选）：与美团底图同源，返回官方路名。
  调用方式是拿轨迹的起终点请求路径，取每一步的 `road` 字段。
- **截图自带标签**（离线兜底）：直接把美团地图上印着的路名按位置归到
  轨迹分段上。这是「地图自己说的」，不依赖任何外部服务。

注意：这一步**不改变经纬度**，只做命名，不引入位置误差。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import requests

# 真正的道路名后缀。注意与 ocr_anchor.ROAD_SUFFIXES 区分：那里把
# 「公园/广场/中心」也算作 road 是为了**排除**它们当锚点；
# 而路名归属要的是真的马路，公园/购物中心不是路。
TRUE_ROAD_SUFFIXES = (
    "路", "街", "大道", "公路", "高架路", "高架", "隧道", "桥", "弄", "巷",
    "匝道", "立交", "环路", "环道", "里弄", "通道",
)


def is_road_name(text: str) -> bool:
    """判断一个标注是否是道路名（而不是公园、广场、购物中心等）。"""
    t = text.strip()
    if len(t) < 2 or len(t) > 10:
        return False
    return t.endswith(TRUE_ROAD_SUFFIXES)


AMAP_DIRECTION = "https://restapi.amap.com/v4/direction/bicycling"
AMAP_DIRECTION_V3 = "https://restapi.amap.com/v3/direction/driving"


@dataclass
class RoadSegment:
    """轨迹上的一段，对应一条道路。"""

    road: str
    start_index: int
    end_index: int
    length_m: float
    source: str = "amap"
    # 与官方路名的置信度：高德=high，截图标签=medium
    confidence: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "road": self.road,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "length_m": round(self.length_m, 1),
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class RoadAttribution:
    """整条轨迹的路名归属结果。"""

    segments: list[RoadSegment] = field(default_factory=list)
    source: str = "none"
    note: str = ""

    @property
    def roads(self) -> list[str]:
        """按出现顺序去重的路名列表。"""
        out: list[str] = []
        for s in self.segments:
            if not out or out[-1] != s.road:
                out.append(s.road)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "note": self.note,
            "roads": self.roads,
            "segments": [s.to_dict() for s in self.segments],
        }


# ---------------------------------------------------------------- 高德

def _amap_key() -> str:
    for k in ("AMAP_KEY",):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    # .env
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("AMAP_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _haversine(a, b) -> float:
    r = 6371008.8
    p1, p2 = math.radians(a[1]), math.radians(b[1])
    dp = math.radians(b[1] - a[1])
    dl = math.radians(b[0] - a[0])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


def attribute_by_amap(
    lonlat: list[tuple[float, float]],
    key: str | None = None,
    timeout: float = 12.0,
) -> RoadAttribution:
    """用高德路径规划 API 获取路名序列。

    高德与美团底图同源，其返回的 `road` 字段是国内官方路名——这正是
    用户在地图上看到、也会记住的名字。
    """
    key = key or _amap_key()
    if not key or len(lonlat) < 2:
        return RoadAttribution(
            source="none",
            note="未配置 AMAP_KEY，无法获取官方路名",
        )
    origin = f"{lonlat[0][0]:.6f},{lonlat[0][1]:.6f}"
    dest = f"{lonlat[-1][0]:.6f},{lonlat[-1][1]:.6f}"
    last_err = ""
    for url in (AMAP_DIRECTION, AMAP_DIRECTION_V3):
        try:
            resp = requests.get(
                url,
                params={"origin": origin, "destination": dest, "key": key,
                        "extensions": "base", "strategy": "0"},
                timeout=timeout,
            )
            data = resp.json()
        except Exception as exc:  # 网络/解析异常都降级，不影响主流程
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        if str(data.get("status")) not in ("1", "0"):
            last_err = str(data.get("info") or data.get("infocode") or "unknown")
            continue
        payload = data.get("route") or {}
        paths = payload.get("paths") or []
        if not paths:
            last_err = "route.paths 为空"
            continue
        steps = paths[0].get("steps") or []
        if not steps:
            last_err = "steps 为空"
            continue

        segs: list[RoadSegment] = []
        idx = 0
        for st in steps:
            road = (st.get("road") or "").strip()
            if not road:
                road = _road_from_instruction(st.get("instruction") or "")
            dist = float(st.get("distance") or 0.0)
            pts = _decode_polyline(st.get("polyline") or "")
            n = max(1, len(pts) or 1)
            segs.append(
                RoadSegment(
                    road=road or "（无名道路）",
                    start_index=idx,
                    end_index=idx + n,
                    length_m=dist,
                    source="amap",
                    confidence=0.92 if road else 0.5,
                )
            )
            idx += n
        return RoadAttribution(
            segments=segs,
            source="amap",
            note=f"路名来自高德路径规划（与美团底图同源），共 {len(segs)} 段",
        )
    return RoadAttribution(source="none", note=f"高德路径规划调用失败：{last_err}")


def _road_from_instruction(instr: str) -> str:
    """从「沿XX道路向XX方向行驶」里抽出路名。"""
    for kw in ("沿", "进入"):
        if kw in instr:
            tail = instr.split(kw, 1)[1]
            for sep in ("道路", "公路", "大街", "路向", "路"):
                if sep in tail:
                    name = tail.split(sep, 1)[0]
                    if 1 < len(name) <= 12:
                        return name
    return ""


def _decode_polyline(poly: str) -> list[tuple[float, float]]:
    """解码高德 polyline（分号分隔的 lon,lat）。"""
    out: list[tuple[float, float]] = []
    for seg in poly.split(";"):
        if not seg:
            continue
        parts = seg.split(",")
        if len(parts) < 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


# ------------------------------------------------- 截图自带标签（离线）

def attribute_by_shot_labels(
    lonlat: list[tuple[float, float]],
    pixel_points: list[tuple[float, float]],
    labels: list[dict[str, Any]],
    max_radius_px: float = 150.0,
) -> RoadAttribution:
    """用截图上印着的路名给轨迹分段命名。

    labels 形如 [{"text": "淞虹路", "px": 457, "py": 1536, "vertical": False}]
    这些标签来自美团地图本身，所以命名与用户看到的完全一致。

    关键约束：**路名是沿道路走向排印的**。所以只接受「排印方向与该段
    轨迹走向大致一致」的标签——否则起点旁那条南北向的「水城路」会被
    误贴到东西向的起始段上（实测踩过）。

    另加迟滞：当前标签在距离上没有被明显甩开时保持不变，避免标签在
    边界处来回跳导致分段碎片化。
    """
    if not labels or len(lonlat) < 2 or len(pixel_points) != len(lonlat):
        return RoadAttribution(source="none", note="截图未提供路名标签")

    px = [p[0] for p in pixel_points]
    py = [p[1] for p in pixel_points]

    # 每个轨迹点的局部走向（度）
    angles: list[float] = []
    for i in range(len(px)):
        a = max(0, i - 3)
        b = min(len(px) - 1, i + 3)
        dx = px[b] - px[a]
        dy = py[b] - py[a]
        angles.append(math.degrees(math.atan2(dy, dx)))

    def dir_ok(lb: dict[str, Any], ang: float) -> bool:
        """标签排印轴与路段走向的夹角是否在容差内。

        轴角：横排=0°，竖排=90°。图像 y 轴向下，所以走向角直接用
        atan2(dy, dx) 即可。折叠到 [-90,90) 后取绝对值即锐角。
        """
        axis = 90.0 if lb.get("vertical") else 0.0
        diff = abs((ang - axis + 90.0) % 180.0 - 90.0)
        return diff <= 50.0

    names: list[str] = []
    cur: str | None = None
    cur_d = 1e9
    for i in range(len(px)):
        best_name, best_d = "", 1e9
        for lb in labels:
            d = math.hypot(px[i] - lb["px"], py[i] - lb["py"])
            if d > max_radius_px:
                continue
            if not dir_ok(lb, angles[i]):
                continue
            if d < best_d:
                best_d, best_name = d, lb["text"]
        if best_name and cur is not None and best_name != cur:
            # 迟滞：新标签要明显更近才替换
            if best_d < cur_d * 0.7:
                cur, cur_d = best_name, best_d
        elif best_name:
            cur, cur_d = best_name, best_d
        else:
            cur, cur_d = None, 1e9
        names.append(cur or "")

    # 合并同名的连续段
    segs: list[RoadSegment] = []
    start = 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[start]:
            if names[start]:
                length = sum(
                    _haversine(lonlat[k - 1], lonlat[k]) for k in range(start + 1, i + 1)
                )
                segs.append(RoadSegment(
                    road=names[start], start_index=start, end_index=i - 1,
                    length_m=length, source="shot_label", confidence=0.55,
                ))
            start = i
    return RoadAttribution(
        segments=segs,
        source="shot_label",
        note="路名来自截图上美团地图自己印出的标注（按走向匹配，可能与官方路网存在改名差异）",
    )
