"""路网约束精修：用「轨迹必须走在真实道路上」这一硬约束纠正配准。

为什么需要这一步
----------------
配准的锚点是 **POI 文字标注**，而标注位置由地图排版决定：

- 美团把 POI 的文字标在建筑/场地的**北侧**以避开道路，文字中心因此
  偏离真实点位几十米；
- 更严重的是 **POI 坐标本身来自不同数据库**：美团底图用高德/腾讯的
  国内 POI 数据，而词典种子来自 OSM/Nominatim。同一 POI 在两套数据里
  可以差 100-300 米。

实测长宁新泾一例：锚点全部来自 OSM POI 点，配准因此整体偏北约 245 米，
轨迹落到了平行的错误道路族上（用户实骑 新渔路/新渔东路/剑河路/金钟路，
OSM 归属却显示 天山路/北翟路）。这不是算法 bug，是数据源基准不一致。

而**道路的几何是客观的**——路网线段的位置在所有地图上必须一致。
骑行轨迹天然贴着道路走，这是比任何 POI 都强的约束。

于是把配准拆成两步：
  1. 锚点拟合  给出粗略的尺度与位置（比例尺很准，位置可能偏）
  2. 路网精修  在「保持尺度与朝向」的前提下平移，使轨迹贴路距离最小

实测效果：轨迹到真实道路的中位距离 32.8 m → 4.7 m，路名归属也随即
正确输出为用户实骑的那几条路。

数据源说明
----------
路网几何默认取自 OpenStreetMap（OSM API / Overpass）。
**OSM 仅用于几何精修与路名参考，不作为底图渲染**，页面底图仍是腾讯地图。
高德/腾讯的路径规划 API 可作为替代数据源接入（`provider` 参数），
它同时能返回官方路名。
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .geo.coords import lonlat_to_mercator, wgs84_to_gcj02

# --- 数据获取 -------------------------------------------------------------

OSM_API = "https://api.openstreetmap.org/api/0.6/map"
OVERPASS = "https://overpass-api.de/api/interpreter"


def _fetch_osm_api(bbox: tuple[float, float, float, float], timeout: float) -> str:
    url = OSM_API + "?bbox=%.6f,%.6f,%.6f,%.6f" % bbox
    req = urllib.request.Request(url, headers={"User-Agent": "meituan-ride/0.1"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")


def _fetch_overpass(bbox: tuple[float, float, float, float], timeout: float) -> str:
    q = ('[out:json][timeout:90];way["highway"]["name"](%.6f,%.6f,%.6f,%.6f);out geom;'
         % (bbox[1], bbox[0], bbox[3], bbox[2]))
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(OVERPASS, data=data,
                                 headers={"User-Agent": "meituan-ride/0.1"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")


def fetch_roads(
    bbox: tuple[float, float, float, float],
    timeout: float = 25.0,
    cache_path: str | None = None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """取回 bbox 内的具名道路几何。

    返回 [(路名, [(lon, lat), ...]), ...]
    优先用 Overpass（一次拿到全部），失败则退回 OSM API（分块小范围）。

    缓存带 bbox 校验：bbox 变了必须重新拉，否则会把「查错区域」的结果
    一直用下去（这个坑踩过——早期 bbox 少算了一个角，缓存里存的就是
    偏移 4.4 公里的路网，之后每次识别都错）。
    """
    key = [round(v, 4) for v in bbox]
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("bbox") == key:
                return [(nm, [(p[0], p[1]) for p in pts])
                        for nm, pts in raw.get("roads", [])]
        except Exception:
            pass

    roads: list[tuple[str, list[tuple[float, float]]]] = []
    errors = []
    try:
        data = json.loads(_fetch_overpass(bbox, timeout))
        for e in data.get("elements", []):
            nm = (e.get("tags") or {}).get("name")
            g = e.get("geometry") or []
            if nm and len(g) >= 2:
                roads.append((nm, [(p["lon"], p["lat"]) for p in g]))
    except Exception as exc:
        errors.append(f"overpass: {type(exc).__name__}")

    if not roads:
        try:
            xml = _fetch_osm_api(bbox, timeout)
            root = ET.fromstring(xml)
            nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat")))
                     for n in root.iter("node")}
            for w in root.iter("way"):
                tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
                nm = tags.get("name")
                if not nm or tags.get("highway") is None:
                    continue
                pts = [nodes[nd.get("ref")] for nd in w.findall("nd")
                       if nd.get("ref") in nodes]
                if len(pts) >= 2:
                    roads.append((nm, pts))
        except Exception as exc:
            errors.append(f"osmapi: {type(exc).__name__}")

    if not roads:
        raise RuntimeError("路网获取失败：" + "，".join(errors))

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"bbox": key,
                           "roads": [[nm, pts] for nm, pts in roads]},
                          f, ensure_ascii=False)
        except OSError:
            pass
    return roads


# --- 精修 -----------------------------------------------------------------

@dataclass
class RoadFit:
    """路网约束精修的结果。"""

    ok: bool = False
    shift_e_m: float = 0.0
    shift_n_m: float = 0.0
    median_before_m: float = 0.0
    median_after_m: float = 0.0
    roads: list[str] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    source: str = "none"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "shift_e_m": round(self.shift_e_m, 1),
            "shift_n_m": round(self.shift_n_m, 1),
            "median_before_m": round(self.median_before_m, 1),
            "median_after_m": round(self.median_after_m, 1),
            "roads": self.roads,
            "segments": self.segments,
            "source": self.source,
            "notes": self.notes,
        }


class RoadNetwork:
    """路网索引：把道路几何采样成点云，支持「给任意折线逐点归属路名」。

    基准说明：传入的道路几何来自 OSM，是 **WGS-84**；而配准结果
    ``reg.pixel_to_mercator()`` 给出的是 **GCJ-02** 墨卡托米（与截图底图
    同一基准）。两者直接比较会引入约 470 米的伪偏移——轨迹会被当成
    "偏离道路"，然后被"路网精修"当成真实位置误差去纠正，结果整体跑偏。

    因此这里在采样前统一把 WGS-84 转成 GCJ-02。
    """

    def __init__(self, roads: list[tuple[str, list[tuple[float, float]]]],
                 sample_m: float = 6.0,
                 src_crs: str = "WGS-84"):
        self.samples: list[tuple[float, float]] = []
        self.owner: list[str] = []
        for nm, pts in roads:
            if len(pts) < 2:
                continue
            conv = pts
            if src_crs.upper() == "WGS-84":
                conv = [wgs84_to_gcj02(lo, la) for lo, la in pts]
            mpts = [lonlat_to_mercator(lo, la) for lo, la in conv]
            for i in range(len(mpts) - 1):
                (ax, ay), (bx, by) = mpts[i], mpts[i + 1]
                seg = math.hypot(bx - ax, by - ay)
                n = max(1, int(seg / sample_m))
                for k in range(n + 1):
                    t = k / n
                    self.samples.append((ax + (bx - ax) * t, ay + (by - ay) * t))
                    self.owner.append(nm)
        self.tree = cKDTree(np.array(self.samples)) if self.samples else None

    def __len__(self) -> int:
        return len(self.samples)

    def query(self, reg: Any, polyline_px: np.ndarray,
              shift: tuple[float, float] = (0.0, 0.0)):
        """返回 (每点最近道路名, 到该道路的距离米)。"""
        if self.tree is None or len(polyline_px) == 0:
            return [], np.array([])
        traj = np.array([reg.pixel_to_mercator(float(px), float(py))
                         for px, py in polyline_px])
        d, idx = self.tree.query(traj + np.array(shift))
        return [self.owner[i] for i in idx], d

    def segments(self, reg: Any, polyline_px: np.ndarray,
                 shift: tuple[float, float] = (0.0, 0.0),
                 min_len_m: float = 0.0):
        """把逐点路名合并成连续路段。"""
        names, d = self.query(reg, polyline_px, shift)
        if not names:
            return []
        traj = np.array([reg.pixel_to_mercator(float(px), float(py))
                         for px, py in polyline_px])
        out: list[dict[str, Any]] = []
        start = 0
        for i in range(1, len(names) + 1):
            if i == len(names) or names[i] != names[start]:
                length = 0.0
                if i > start:
                    length = float(np.sum(np.linalg.norm(
                        np.diff(traj[start:i + 1], axis=0), axis=1)))
                if length >= min_len_m:
                    out.append({
                        "road": names[start],
                        "start_index": start,
                        "end_index": i - 1,
                        "length_m": round(length, 1),
                        "dist_m": round(float(np.median(d[start:i])), 1),
                    })
                start = i
        return out


def refine_with_roads(
    reg: Any,
    polyline_px: np.ndarray,
    roads: list[tuple[str, list[tuple[float, float]]]],
    max_shift_m: float = 800.0,
    coarse_step_m: float = 25.0,
    fine_step_m: float = 5.0,
    sample_m: float = 6.0,
) -> RoadFit:
    """在保持尺度与朝向的前提下平移轨迹，使其贴路距离最小。

    reg           配准结果（提供 pixel→墨卡托 的换算）
    polyline_px   像素坐标折线
    roads         [(路名, 经纬点串), ...]

    返回的 shift_n_m 为**正表示向北**（几何意义上）。
    施加方式：``reg.translation += [shift_e_m, shift_n_m]``。
    """
    result = RoadFit()
    if len(polyline_px) < 3 or not roads:
        result.notes.append("轨迹或路网数据不足，跳过路网精修")
        return result

    net = RoadNetwork(roads, sample_m=sample_m)
    if len(net) < 50:
        result.notes.append("路网采样点过少，跳过精修")
        return result
    tree = net.tree
    # 轨迹像素 → 绝对墨卡托米（与路网同一坐标系）
    traj = np.array([reg.pixel_to_mercator(float(px), float(py))
                      for px, py in polyline_px])

    def median_dist(dx: float, dy: float) -> float:
        d, _ = tree.query(traj + np.array([dx, dy]))
        return float(np.median(d))

    before = median_dist(0.0, 0.0)
    result.median_before_m = before

    # 粗搜
    best = (before, 0.0, 0.0)
    steps = int(max_shift_m / coarse_step_m)
    for i in range(-steps, steps + 1):
        dx = i * coarse_step_m
        for j in range(-steps, steps + 1):
            dy = j * coarse_step_m
            m = median_dist(dx, dy)
            if m < best[0]:
                best = (m, dx, dy)

    # 局部精修
    bx, by = best[1], best[2]
    for i in range(-6, 7):
        for j in range(-6, 7):
            dx = bx + i * fine_step_m
            dy = by + j * fine_step_m
            m = median_dist(dx, dy)
            if m < best[0]:
                best = (m, dx, dy)

    after, dx, dy = best
    result.shift_e_m = dx
    result.shift_n_m = dy
    result.median_after_m = after
    result.source = "osm"

    # 只有明显改善才采纳；否则说明路网覆盖不足，不动配准
    if after > before * 0.9 or after > 60.0:
        result.notes.append(
            f"路网精修未采纳（贴路中位 {before:.0f}m → {after:.0f}m，"
            f"改善不足或路网覆盖不足）"
        )
        result.shift_e_m = result.shift_n_m = 0.0
        result.median_after_m = before
        return result

    # 归属路名（对原始骨架折线）
    result.segments = net.segments(reg, polyline_px, (dx, dy))
    ordered: list[str] = []
    for s in result.segments:
        if not ordered or ordered[-1] != s["road"]:
            ordered.append(s["road"])
    result.roads = ordered
    result.ok = True
    result.notes.append(
        f"路网精修：平移 东{dx:+.0f}m 北{dy:+.0f}m，"
        f"贴路中位 {before:.0f}m → {after:.0f}m"
    )
    return result
