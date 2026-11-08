"""水系护栏：检测骑行轨迹是否横穿大江大河。

为什么需要
----------
低缩放截图里，远距离地标的文字标注会让锚点配准把尺度算大，
轨迹被拉长后可能越过河流、落到对岸街区；这类结果的置信度看起来
并不低，但物理上不成立。护栏是**兜底**：即使前面的环节有偏差，
也不该输出一条横穿通航大河的骑行轨迹。

几何依据（重要）
----------------
OSM 的 `way["waterway"="river"]` 几何常是**岸线**而非河心线，
河道也不一定有水面多边形。因此判定必须基于岸线的**局部法向**：
先确定「哪一侧算过江」，再看轨迹点是否落进那侧的河宽范围内；
不能简单地拿轨迹到中心线的距离当判据。

河流清单、侧别与半宽都是外部地理数据（`data/waters.json`），
本模块不认识任何具体河名。

实现
----
1. 拉取大河岸线（OSM way 几何）
2. 每个轨迹点找最近岸线点，用该点局部法向判定在河的哪一侧
3. 法向符号统一：以「正东」为参考，避免随河道弯曲翻转 180 度
4. 连续 N 个点落在「过江侧」即判定横穿

坐标
----
轨迹是 GCJ-02（与美团底图同源）；OSM 是 WGS-84。
两者差 300-600 米，**必须转换**后再比较，否则判定会整体偏移。
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

from .geo.coords import gcj02_to_wgs84, wgs84_to_gcj02
from .waterdata import fallback_polylines, search_bbox, water_specs

_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interparser",
    "https://overpass.private.coffee/api/interpreter",
]

# 河流清单、侧别参数与兜底岸线都来自外部地理数据（data/waters.json）。
# 代码里不内置任何河名、坐标或半宽：换城市只需替换数据文件，
# 护栏的几何判定逻辑一行都不用改。
#
# 参数含义：
#   bank_half_width_m: 岸线向对侧延伸多少米算「在江里」
#   east_side:         True = 河东/南侧算过江；False = 另一侧
WATER_SPECS: dict[str, tuple[str, float, bool]] = water_specs()
_FALLBACK: dict[str, list[tuple[float, float]]] = fallback_polylines()


def _cache_path(cache_dir: str, city: str) -> str:
    safe = "".join(ch for ch in city if ch.isalnum()) or "default"
    return os.path.join(cache_dir, f"waters_{safe}.json")


def _load_cached(cache_dir: str, city: str) -> dict[str, Any]:
    try:
        with open(_cache_path(cache_dir, city), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _store_cached(cache_dir: str, city: str, data: dict[str, Any]) -> None:
    try:
        os.makedirs(cache_dir or ".", exist_ok=True)
        with open(_cache_path(cache_dir, city), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def _fetch_water(name: str, timeout: float = 60.0) -> list[list[tuple[float, float]]]:
    """从 Overpass 拉一条河的岸线（WGS-84 经纬度段列表）。

    查询范围来自水系数据文件，不写死某个城市的经纬度框。
    """
    s, w, n, e = search_bbox()
    q = (f'[out:json][timeout:{int(timeout)}];'
         f'way["waterway"="river"]["name"="{name}"]({s},{w},{n},{e});'
         f'out geom;')
    for m in _OVERPASS_MIRRORS:
        try:
            req = urllib.request.Request(
                m, data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "track-buddy/0.1"})
            txt = urllib.request.urlopen(req, timeout=timeout).read().decode()
            if txt.lstrip()[:1] != "{":
                continue
            data = json.loads(txt)
            segs: list[list[tuple[float, float]]] = []
            for e in data.get("elements", []):
                g = e.get("geometry") or []
                if len(g) >= 2:
                    segs.append([(p["lon"], p["lat"]) for p in g])
            if segs:
                return segs
        except Exception:
            time.sleep(0.8)
    return []


def get_waters(
    city: str = "上海",
    cache_dir: str = "data/cache",
    allow_network: bool = True,
    timeout: float = 60.0,
) -> dict[str, list[list[tuple[float, float]]]]:
    """取主要大河的岸线（WGS-84）。结果按城市缓存 24 小时。

    ⚠️ **护栏不能依赖网络可用性**：`allow_network=False` 时若缓存也没有，
    就返回空 dict —— 此时跨江检测**完全无法执行**，
    等于护栏静默失效（这正是本次实测发现的问题：
    服务端 `allow_online=False` 时水系检查返回空，护栏形同虚设）。

    所以这里的行为是：
      * 缓存命中 -> 直接用；
      * 缓存未命中且允许联网 -> 拉取并缓存；
      * 缓存未命中且不允许联网 -> 用**内置兜底折线**
        （精度有限，但足以判断「是否跨江」）。
    """
    cached = _load_cached(cache_dir, city)
    if cached and (time.time() - cached.get("ts", 0) < 86400):
        return {k: [[(float(a), float(b)) for a, b in seg] for seg in v]
                for k, v in cached.get("waters", {}).items()}

    out: dict[str, list[list[tuple[float, float]]]] = {}
    if allow_network:
        for name, (query, _, _) in WATER_SPECS.items():
            segs = _fetch_water(query, timeout=timeout)
            if segs:
                out[name] = segs
    if out:
        _store_cached(cache_dir, city, {"ts": time.time(), "waters": out})
        return out

    # 兜底：数据文件里的岸线折线（WGS-84）。
    return {k: [[(float(lo), float(la)) for lo, la in pl]]
            for k, pl in _FALLBACK.items()}


def _signed_side(
    pts_wgs: np.ndarray,
    bank_wgs: np.ndarray,
    half_width_m: float,
    east_side: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """判定每个轨迹点相对岸线的位置。

    返回 (在河里的掩码, 到岸线的距离米)。

    做法：
      1. 找每个轨迹点最近的岸线点
      2. 用岸线在该点的**局部切向**算法向
      3. 法向统一符号：让法向始终指向「与正东夹角更小」的一侧，
         这样河道弯曲时法向不会翻转 180 度
      4. 点在法向正方向且距离 < half_width -> 在河里
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(bank_wgs)
    d, idx = tree.query(pts_wgs)
    near = bank_wgs[idx]

    # 局部切向（用相邻岸线点的差分）
    j0 = np.clip(idx - 1, 0, len(bank_wgs) - 1)
    j1 = np.clip(idx + 1, 0, len(bank_wgs) - 1)
    tang = bank_wgs[j1] - bank_wgs[j0]
    # 切向转成米
    lat0 = float(np.mean(pts_wgs[:, 1]))
    m_lat = 111320.0
    m_lon = 111320.0 * max(math.cos(math.radians(lat0)), 1e-6)
    tx = tang[:, 0] * m_lon
    ty = tang[:, 1] * m_lat
    tn = np.hypot(tx, ty)
    tn[tn < 1e-9] = 1.0
    tx, ty = tx / tn, ty / tn
    # 法向 = 切向逆时针转 90°
    nx, ny = -ty, tx
    # ⚠️ **统一符号**：法向随河道弯曲会翻转 180 度。
    # 用「法向的东向分量」决定符号：让 nx 尽量为正（朝东）。
    # 这样同一条河的两段不会因为弯道而被判成相反侧别。
    flip = np.where(nx < 0, -1.0, 1.0)
    nx, ny = nx * flip, ny * flip

    # 点的偏移（米）
    vx = (pts_wgs[:, 0] - near[:, 0]) * m_lon
    vy = (pts_wgs[:, 1] - near[:, 1]) * m_lat
    dist_m = np.hypot(vx, vy)
    side = vx * nx + vy * ny          # >0 表示在法向正方向（东侧）

    if not east_side:
        side = -side
    in_water = (side > 0) & (dist_m < half_width_m)
    return in_water, dist_m


def check_crossing(
    lonlat_gcj: list[tuple[float, float]],
    waters: dict[str, list[list[tuple[float, float]]]] | None = None,
    city: str = "上海",
    cache_dir: str = "data/cache",
    allow_network: bool = True,
    min_run: int = 3,
) -> tuple[bool, str, float]:
    """检查轨迹是否横穿大河。

    返回 (是否横穿, 说明, 横穿长度占比)。

    **占比** = 落在河里的轨迹点数 / 总点数。
    判定要求**连续 min_run 个点**在河里——单点擦过江边不算
    （贴着滨江骑是合法的，只有「骑在江里」才可疑）。
    """
    if len(lonlat_gcj) < 2:
        return False, "", 0.0
    if waters is None:
        waters = get_waters(city=city, cache_dir=cache_dir,
                            allow_network=allow_network)
    if not waters:
        return False, "", 0.0

    pts_wgs = np.array([gcj02_to_wgs84(lo, la) for lo, la in lonlat_gcj])

    crossed: list[str] = []
    worst = 0.0
    for name, segs in waters.items():
        spec = WATER_SPECS.get(name)
        if spec is None:
            continue
        _, half_w, east = spec
        # 岸线点云（WGS-84）
        bank: list[tuple[float, float]] = []
        for seg in segs:
            bank.extend(seg)
        if len(bank) < 3:
            continue
        bank_arr = np.array(bank)
        in_water, dist_m = _signed_side(
            pts_wgs, bank_arr, half_w, east)
        # 连续段
        best_run = 0
        run = 0
        hit = False
        for v in in_water:
            if v:
                run += 1
                best_run = max(best_run, run)
                if run >= min_run:
                    hit = True
            else:
                run = 0
        ratio = float(best_run) / len(in_water)
        worst = max(worst, ratio)
        if hit:
            crossed.append(name)

    if crossed:
        return True, "、".join(sorted(set(crossed))), worst
    return False, "", worst
