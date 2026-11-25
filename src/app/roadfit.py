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
from .maps.roads import OVERPASS_MIRRORS, overpass_raw

# --- 数据获取 -------------------------------------------------------------

OSM_API = "https://api.openstreetmap.org/api/0.6/map"
# Overpass 镜像表与轮换逻辑统一维护在 app/maps/roads/overpass.py。
# 这里保留同名导出（OVERPASS_MIRRORS），兼容既有调用与测试；要加镜像、
# 改超时策略只需改能力层一处，不会再出现三份列表各改各的情况。


def _fetch_osm_api(bbox: tuple[float, float, float, float], timeout: float) -> str:
    url = OSM_API + "?bbox=%.6f,%.6f,%.6f,%.6f" % bbox
    req = urllib.request.Request(url, headers={"User-Agent": "meituan-ride/0.1"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")


def _fetch_overpass(
    bbox: tuple[float, float, float, float], timeout: float,
    highway_types: list[str] | None = None,
) -> str:
    if highway_types:
        filt = '~"^(%s)$"' % "|".join(highway_types)
    else:
        filt = ""
    q = ('[out:json][timeout:180];way["highway"%s]["name"](%.6f,%.6f,%.6f,%.6f);out geom;'
         % (filt, bbox[1], bbox[0], bbox[3], bbox[2]))
    # 镜像轮换、超时分配、网关错误页拦截都在能力层，行为与原先一致：
    # 主镜像拿完整 timeout，备用镜像拿 max(10, timeout/4)，失败不 sleep。
    return overpass_raw(q, timeout=timeout, user_agent="meituan-ride/0.1")


def fetch_roads(
    bbox: tuple[float, float, float, float],
    timeout: float = 25.0,
    cache_path: str | None = None,
    highway_types: list[str] | None = None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """取回 bbox 内的具名道路几何。

    返回 [(路名, [(lon, lat), ...]), ...]
    优先用 Overpass（一次拿到全部），失败则退回 OSM API（分块小范围）。

    缓存带 bbox 校验：bbox 变了必须重新拉，否则会把「查错区域」的结果
    一直用下去（这个坑踩过——早期 bbox 少算了一个角，缓存里存的就是
    偏移 4.4 公里的路网，之后每次识别都错）。

    缓存格式支持**多块**存储：``{"blocks": [{"bbox": [...], "roads": [...]}]}``。
    为什么需要多块：Overpass 拉一次路网要 20-40 秒（实测浦东一带），
    而每次识别的 bbox 都略有不同——单块缓存会导致每次都重新拉。
    多块缓存让「bbox 已被某个块完全包含」时直接复用。

    **复用条件是「完全包含」而非「重叠」**：重叠时复用会把区域外的
    道路混进结果。这正是早期踩过的坑——缓存里存着偏移 4.4 公里的路网，
    之后每次识别都错在错的地方，且因为「有数据」而看不出异常。
    """
    key = [round(v, 4) for v in bbox]
    blocks = _load_cache_blocks(cache_path)

    # 找完全包含目标 bbox 的缓存块
    for blk in blocks:
        bb = blk.get("bbox")
        if bb and _bbox_contains(bb, key):
            return [(nm, [(p[0], p[1]) for p in pts])
                    for nm, pts in blk.get("roads", [])]

    roads = _fetch_roads_missing(bbox, timeout, highway_types)
    _save_cache_block(cache_path, key, roads)
    return roads


def _load_cache_blocks(cache_path: str | None) -> list[dict]:
    """读缓存，兼容单块与多块两种格式。"""
    if not cache_path or not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if "blocks" in raw:
        return [b for b in raw.get("blocks", []) if isinstance(b, dict)]
    # 旧格式：{"bbox": [...], "roads": [...]}
    if "bbox" in raw:
        return [{"bbox": raw["bbox"], "roads": raw.get("roads", [])}]
    return []


def _save_cache_block(
    cache_path: str | None, bbox: list[float], roads: list
) -> None:
    """把一次查询结果并入缓存（保留其他块）。"""
    if not cache_path:
        return
    blocks = _load_cache_blocks(cache_path)
    blocks = [b for b in blocks if b.get("bbox") != bbox]
    blocks.append({"bbox": bbox, "roads": [[nm, pts] for nm, pts in roads]})
    # 只保留最近 8 块，避免缓存无限膨胀
    blocks = blocks[-8:]
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"blocks": blocks}, f, ensure_ascii=False)
    except OSError:
        pass


def _bbox_contains(outer: list[float], inner: list[float]) -> bool:
    """outer 是否完全包含 inner。"""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def _fetch_roads_missing(
    bbox: tuple[float, float, float, float], timeout: float,
    highway_types: list[str] | None = None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """实际走网络拉取指定 bbox 的道路。

    ``highway_types`` 非空时只取这些等级的道路。城市级低缩放截图一张图
    覆盖十几公里，全量路网的数据量大到查询要 40-90 秒；而**主干道 +
    次干道**既是骑行轨迹最常走的路，数量又只有全量的 1/3。

    但不能只留主干道——轨迹附近没有主干道时，贴路距离会被「最近的主干道
    有多远」主导（实测 432m vs 56m）。
    """
    roads: list[tuple[str, list[tuple[float, float]]]] = []
    errors = []
    # 按面积选数据源。⚠️ 阈值不能按 OSM API 的「面积上限 0.25 平方度」
    # 来定：它还有**节点数上限**（5 万个），上海这种高密度区域
    # 0.0054 平方度就已经超了，请求直接返回 HTTP 400（实测踩过：
    # bbox 面积 0.0054 → osmapi 失败，而同一范围 Overpass 亚秒级返回）。
    area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    order = ("osmapi", "overpass") if area < 0.0015 else ("overpass", "osmapi")
    for which in order:
        if roads:
            break
        if which == "overpass":
            # Overpass 公共实例不稳定：繁忙时返回 504 网关页、超时，
            # 甚至对某些 bbox 返回空结果。重试 + 扩大范围各来一次。
            for attempt in range(3):
                try:
                    data = json.loads(
                        _fetch_overpass(bbox, timeout, highway_types)
                    )
                    for e in data.get("elements", []):
                        nm = (e.get("tags") or {}).get("name")
                        g = e.get("geometry") or []
                        if nm and len(g) >= 2:
                            roads.append((nm, [(p["lon"], p["lat"]) for p in g]))
                    if roads:
                        break
                    errors.append("overpass: 返回空结果")
                except Exception as exc:
                    errors.append(f"overpass: {type(exc).__name__}")
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
            if not roads:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                hw = (bbox[2] - bbox[0])
                hh = (bbox[3] - bbox[1])
                big = (cx - hw, cy - hh, cx + hw, cy + hh)
                try:
                    data = json.loads(
                        _fetch_overpass(big, timeout, highway_types)
                    )
                    for e in data.get("elements", []):
                        nm = (e.get("tags") or {}).get("name")
                        g = e.get("geometry") or []
                        if nm and len(g) >= 2:
                            roads.append((nm, [(p["lon"], p["lat"]) for p in g]))
                except Exception as exc:
                    errors.append(f"overpass(扩大): {type(exc).__name__}")
        else:
            try:
                roads.extend(
                    _roads_via_osm_api(bbox, timeout, highway_types)
                )
            except Exception as exc:
                errors.append(f"osmapi: {type(exc).__name__}")

    if not roads:
        raise RuntimeError("路网获取失败：" + "，".join(errors))
    return roads


def _roads_via_osm_api(
    bbox: tuple[float, float, float, float],
    timeout: float,
    highway_types: list[str] | None = None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """走 OSM API（api.openstreetmap.org/api/0.6/map）取具名道路。

    适合很小的 bbox：实测 3.5km×2.8km 只要 3.5 秒，而 Overpass 主站
    同一时刻连测三次全超时。OSM API 的面积上限 0.25 平方度、节点数上限
    5 万个，分块后远低于此。
    """
    xml = _fetch_osm_api(bbox, timeout)
    root = ET.fromstring(xml)
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat")))
             for n in root.iter("node")}
    want = set(highway_types) if highway_types else None
    out: list[tuple[str, list[tuple[float, float]]]] = []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        nm = tags.get("name")
        hw = tags.get("highway")
        if not nm or hw is None:
            continue
        if want is not None and hw not in want:
            continue
        pts = [nodes[nd.get("ref")] for nd in w.findall("nd")
               if nd.get("ref") in nodes]
        if len(pts) >= 2:
            out.append((nm, pts))
    return out


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
    # 可选的锚点一致性证据（米）；用于解释为什么采纳/拒绝平移。
    anchor_before_m: float | None = None
    anchor_after_m: float | None = None
    # 被护栏否决时，**保留归零前搜索出的最优平移**，交给前端用另一种
    # 颜色画出「未采纳的精修路线」：用户可以自己对照截图判断护栏拒得
    # 对不对，而不是只看到一句「未采纳」。采纳成功时保持 None。
    candidate_shift_e_m: float | None = None
    candidate_shift_n_m: float | None = None
    candidate_median_after_m: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "ok": self.ok,
            "shift_e_m": round(self.shift_e_m, 1),
            "shift_n_m": round(self.shift_n_m, 1),
            "median_before_m": round(self.median_before_m, 1),
            "median_after_m": round(self.median_after_m, 1),
            "roads": self.roads,
            "segments": self.segments,
            "source": self.source,
            "anchor_before_m": (round(self.anchor_before_m, 1)
                                 if self.anchor_before_m is not None else None),
            "anchor_after_m": (round(self.anchor_after_m, 1)
                                if self.anchor_after_m is not None else None),
            "notes": self.notes,
        }
        # 只在否决时输出：旧调用方对 to_dict() 的键集有静态契约，
        # 采纳成功的结果必须保持原样。
        if self.candidate_shift_e_m is not None:
            out.update({
                "candidate_shift_e_m": round(self.candidate_shift_e_m, 1),
                "candidate_shift_n_m": round(self.candidate_shift_n_m or 0.0, 1),
                "candidate_median_after_m": (
                    round(self.candidate_median_after_m, 1)
                    if self.candidate_median_after_m is not None else None
                ),
            })
        return out


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
        # 按道路几何分组的采样点（GCJ-02 墨卡托米）：路网吸附需要沿路
        # 走（链式边），光有 name→sample 的平面映射不够——同名道路有
        # 多段不相接的几何，必须保留 way 身份与内部顺序。
        self.ways: list[tuple[str, list[tuple[float, float]]]] = []
        for nm, pts in roads:
            if len(pts) < 2:
                continue
            conv = pts
            if src_crs.upper() == "WGS-84":
                conv = [wgs84_to_gcj02(lo, la) for lo, la in pts]
            mpts = [lonlat_to_mercator(lo, la) for lo, la in conv]
            way_pts: list[tuple[float, float]] = []
            for i in range(len(mpts) - 1):
                (ax, ay), (bx, by) = mpts[i], mpts[i + 1]
                seg = math.hypot(bx - ax, by - ay)
                n = max(1, int(seg / sample_m))
                for k in range(n + 1):
                    t = k / n
                    p = (ax + (bx - ax) * t, ay + (by - ay) * t)
                    self.samples.append(p)
                    self.owner.append(nm)
                    if k < n:  # 段末点由下一段的起点补（避免重复）
                        way_pts.append(p)
            if len(mpts) >= 2:
                way_pts.append(mpts[-1])
            self.ways.append((nm, way_pts))
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
    shift_span_ratio: float = 0.15,
    anchor_points_m: np.ndarray | None = None,
    anchors_reliable: bool = True,
    anchor_guard_m: float = 100.0,
    anchor_worsen_m: float = 80.0,
    anchor_worsen_ratio: float = 2.5,
    eval_budget: int = 2000,
) -> RoadFit:
    """在保持尺度与朝向的前提下平移轨迹，使其贴路距离最小。

    reg               配准结果（提供 pixel→墨卡托 的换算）
    polyline_px       像素坐标折线
    roads             [(路名, 经纬点串), ...]
    shift_span_ratio  允许位移 / 轨迹总长度 的上限（无量纲）

    为什么要按轨迹尺度限幅
    ----------------------
    早期只用一个绝对上限（±800 米）+「贴路中位变小」作为采纳判据。
    这在长轨迹上是合理的，但对短轨迹会过拟合：路网是稀疏的平行线族，
    把一条几百米的轨迹整体搬几百米，一样能压在**另一条**平行道路上，
    贴路距离从 100 米降到 5 米，判据全部满足，位置却彻底错了。
    位移超过轨迹自身长度的一定比例时，这已经不是「微调」而是「搬家」。

    所以搜索半径与采纳上限都取
    ``min(max_shift_m, shift_span_ratio × 轨迹米制长度)``，
    与截图分辨率、城市、具体样本都无关。

    ``eval_budget`` 是估计贴路距离时最多用多少个轨迹点。位移候选数
    与轨迹点数是相乘关系：大画布上不设上限会做几千万次 KD 树查询。
    轨迹点本来就是等距采样，等距抽样不改变「中位距离」这个统计量，
    只降低估计精度到可接受范围；**搜索空间与护栏完全不变**，
    轨迹长度（位移限幅依据）始终按完整折线计算。设 0 关闭抽样。

    返回的 shift_n_m 为**正表示向北**（几何意义上）。
    施加方式：``reg.translation += [shift_e_m, shift_n_m]``。
    """
    result = RoadFit()
    if len(polyline_px) < 3 or not roads:
        result.notes.append("轨迹或路网数据不足，跳过路网精修")
        return result

    # 轨迹直线度（弦长/总长）：直线轨迹对「平移贴路」没有判别力——
    # 直线与任何平行路都能贴合（2026-11-17 假阳性的根源）；弯折轨迹
    # 的每个转弯都是形状证据，错误平移不可能让所有转弯同时贴合。
    # 该值只用于下面的「决定性贴路改善」判据，不影响任何搜索行为。
    _pl = np.asarray(polyline_px, dtype=float)
    _len = float(np.linalg.norm(np.diff(_pl, axis=0), axis=1).sum())
    _straightness = (
        min(1.0, float(np.linalg.norm(_pl[-1] - _pl[0])) / _len)
        if _len > 1e-6 else 1.0
    )

    net = RoadNetwork(roads, sample_m=sample_m)
    if len(net) < 50:
        result.notes.append("路网采样点过少，跳过精修")
        return result
    tree = net.tree
    # 轨迹像素 → 绝对墨卡托米（与路网同一坐标系）
    traj_all = np.array([reg.pixel_to_mercator(float(px), float(py))
                         for px, py in polyline_px])

    # 等距抽样只用于「估计中位贴路距离」，不改变网格搜索空间。
    if eval_budget > 0 and len(traj_all) > eval_budget:
        idx = np.linspace(0, len(traj_all) - 1, eval_budget).astype(int)
        traj = traj_all[idx]
    else:
        traj = traj_all

    def median_dist(dx: float, dy: float) -> float:
        d, _ = tree.query(traj + np.array([dx, dy]))
        return float(np.median(d))

    before = median_dist(0.0, 0.0)
    result.median_before_m = before

    # 轨迹自身的米制长度 → 本次允许的位移上限（见函数说明）。
    # ⚠️ 必须用**完整折线**：抽样后相邻点被拉直，弯曲轨迹的长度会
    # 被低估，位移限幅跟着变小，可能误拒本来合理的平移。
    span_m = float(np.linalg.norm(np.diff(traj_all, axis=0), axis=1).sum())
    shift_limit_m = min(max_shift_m, shift_span_ratio * span_m)
    shift_limit_m = max(shift_limit_m, coarse_step_m)

    # 只在允许的位移范围内收集候选，避免「搬到下一条平行路」进入解集
    candidates: list[tuple[float, float, float]] = [(before, 0.0, 0.0)]

    def consider(dx: float, dy: float) -> None:
        if float(np.hypot(dx, dy)) > shift_limit_m:
            return
        candidates.append((median_dist(dx, dy), dx, dy))

    # 粗搜
    steps = max(1, int(shift_limit_m / coarse_step_m))
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            consider(i * coarse_step_m, j * coarse_step_m)

    # ⚠️ 贴路距离并列时必须选**位移最小**的解。
    # 道路常有长直段：沿道路方向平移不改变贴路距离，若用「严格小于
    # 才更新最优」，网格边界上任意大的 dx 会和真正需要的 dy 一起入选，
    # 凭空造出几百米的横向位移。这里用「距离在最优值 + 容差内，
    # 取欧氏位移最小者」消除这种自由度。
    tol = max(1.0, before * 0.01)

    def pick_min_shift():
        min_m = min(c[0] for c in candidates)
        near = [c for c in candidates if c[0] <= min_m + tol]
        return min(near, key=lambda c: float(np.hypot(c[1], c[2])))

    _, bx, by = pick_min_shift()

    # 局部精修（同样不许越出限幅）
    for i in range(-6, 7):
        for j in range(-6, 7):
            consider(bx + i * fine_step_m, by + j * fine_step_m)

    after, dx, dy = pick_min_shift()
    result.shift_e_m = dx
    result.shift_n_m = dy
    result.median_after_m = after
    result.source = "osm"

    def _reject(note: str) -> RoadFit:
        """护栏否决：输出位移归零，但把搜索到的候选解保留下来。

        三道护栏（锚点一致性 / 位移限幅 / 改善不足）都要走这里，
        否则「未采纳的路线」只在一种失败原因下才画得出来。
        """
        result.notes.append(note)
        result.candidate_shift_e_m = dx
        result.candidate_shift_n_m = dy
        result.candidate_median_after_m = after
        result.shift_e_m = result.shift_n_m = 0.0
        result.median_after_m = before
        result.ok = False
        return result

    # 锚点是截图与地理世界之间的直接对应证据，优先级高于「最近 OSM
    # 道路距离」。如果当前配准已经能解释可靠锚点，单纯贴路距离下降
    # 不足以证明整体平移正确；平行道路会让这种优化产生假阳性。
    if anchor_points_m is not None and len(anchor_points_m):
        pts = np.asarray(anchor_points_m, dtype=float)
        before_anchor = float(np.median(np.linalg.norm(pts, axis=1)))
        after_anchor = float(
            np.median(np.linalg.norm(pts - np.array([dx, dy]), axis=1))
        )
        result.anchor_before_m = before_anchor
        result.anchor_after_m = after_anchor
        # 只有锚点**本身可信**时，锚点残差才有否决权。
        #
        # 截图上的 POI 文字是排版标注，不是真实地理点位：中低缩放图常有
        # 100~300m 的系统性标注偏移，此时 before_anchor 本来就大，锚点
        # 自己都没法证明绝对位置正确，「残差被放大」也就不能证明平移错了。
        # 反过来，这类图恰恰最需要路网几何来纠正整体位置。
        #
        # 相对倍数只在基线可信时才有意义；不加这个前提会出现反直觉的
        # 结果：同一段合理平移，锚点基线越小，相对放大倍数越大，反而
        # 越容易被判成「破坏锚点一致性」而误拒。锚点不可信时仍由位移
        # 比例护栏与贴路改善判据兜底，不会失去防止「搬到平行路」的能力。
        anchor_reliable = anchors_reliable and before_anchor <= anchor_guard_m
        if not anchors_reliable:
            # 锚点证据本身不可信（跨源基准混用 / 系统性偏移）时，
            # 「残差被平移放大」不构成否决理由：残差大主要反映的是
            # 基准噪声而非配准错误，此时由位移限幅与贴路改善判据兜底。
            result.notes.append(
                "锚点存在跨源基准混用或系统性偏移，"
                "锚点一致性护栏本次降级不参与否决"
            )
        # 决定性贴路改善：弯折轨迹（直线度 < 0.75，起终点重合的闭环
        # 轨迹弦长≈0，天然满足）经单次刚体平移把全线压到路网极近处
        # （改善 ≥85% 且绝对值 ≤15m）时，形状信息已经完成判别——
        # 错误的平行路平移不可能让一条弯折轨迹的所有转弯同时贴合，
        # 而直线轨迹没有这种判别力（保持原契约，见下方否决分支）。
        # 此时锚点残差被放大只说明 POI 文字标注点位有排版偏移
        # （常见 100m+），不足以推翻路网几何证据。
        # 2026-11-25 余姚实测：单源重配后锚点残差 74m（可信），但
        # 候选平移 东240m/北70m 把贴路中位 87m → 6m，被旧护栏误拒，
        # 轨迹整体停在楼块上 200m+。
        decisive_snap = (
            _straightness < 0.75
            and before >= 60.0
            and after <= max(15.0, before * 0.15)
        )
        if decisive_snap:
            result.notes.append(
                f"贴路改善具有决定性（{before:.0f}m → {after:.0f}m、"
                f"直线度 {_straightness:.2f}）：形状判别力已排除平行路"
                "假阳性，锚点残差放大不构成否决理由"
            )
        anchor_bad = (
            anchor_reliable
            and not decisive_snap
            and (
                after_anchor > before_anchor + anchor_worsen_m
                or (
                    before_anchor > 0
                    and after_anchor > before_anchor * anchor_worsen_ratio
                    and after_anchor > anchor_guard_m
                )
            )
        )
        if anchor_bad:
            return _reject(
                f"路网精修未采纳（可靠锚点残差 {before_anchor:.0f}m → "
                f"{after_anchor:.0f}m，平移会破坏截图锚点一致性）"
            )

    # 位移超过轨迹尺度允许的范围：贴路距离变小也不能采纳。
    # 这类解几乎一定是把轨迹搬到了另一条平行道路上。
    if float(np.hypot(dx, dy)) > shift_limit_m + fine_step_m:
        return _reject(
            f"路网精修未采纳（需平移 {float(np.hypot(dx, dy)):.0f}m，"
            f"超过该轨迹尺度允许的 {shift_limit_m:.0f}m，疑似贴到别的路上）"
        )

    # 只有明显改善才采纳；否则说明路网覆盖不足，不动配准
    if after > before * 0.9 or after > 60.0:
        return _reject(
            f"路网精修未采纳（贴路中位 {before:.0f}m → {after:.0f}m，"
            f"改善不足或路网覆盖不足）"
        )

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
