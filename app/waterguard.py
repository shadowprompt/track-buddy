"""水系护栏：检测骑行轨迹是否横穿大江大河。

为什么需要
----------
骑行轨迹横穿黄浦江几乎一定是错的。实测案例（上海城市级截图）：
锚点配准把尺度算大 26%，轨迹被拉长后东端甩过黄浦江，
起点落在浦东侧（59/417 个点在浦东），路名归属报出
「打浦路隧道 / 时光印记大道 / 世博大道 / 东岸绿道」——
用户反馈「我根本没去浦东」。

这个护栏的价值在于**兜底**：即使前面的算法再出 bug，
也不该输出一条横穿长江的骑行轨迹。

几何依据（重要）
----------------
OSM 的 `way["waterway"="river"]["name"="黄浦江"]` 几何是**岸线**，
不是河心线——实测该线距西岸艺术中心仅 313 米，而真正的河宽约 2600 米。
OSM 里也**没有**黄浦江的水面多边形（实测 `natural=water` 查询返回 0 个）。

所以判定必须基于「岸线」而非「中心线」：
用岸线的**局部法向**算出「河东侧」区域，轨迹落进去才算过江。

实现
----
1. 拉取大河岸线（OSM way 几何）
2. 对每个轨迹点，找最近的岸线点，用该点的**局部法向**判定它在河的哪一侧
3. 法向符号统一：以「正东」为参考方向，避免法向随河道弯曲翻转 180 度
4. 连续 N 个点落在「河东侧」即判定横穿

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

_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interparser",
    "https://overpass.private.coffee/api/interpreter",
]

# 河名 -> (查询名, 「过江」侧的判定参数)
#   half_width: 岸线向对侧延伸多少米算「在江里」
#   east_side:  True = 该河的「浦东/河东」侧算过江
#                False = 该河的另一侧算过江（如苏州河北侧）
#
# ⚠️ **半宽的标定依据**（实测）：
# 黄浦江在市中心段（lat 31.186 附近）西岸到东岸约 2600 米，
# OSM 的黄浦江 way 是**西岸线**（距西岸艺术中心约 390 米）。
# 取 1500 米：既能覆盖「骑在江里」，又不会把
# 「贴着浦西滨江骑」（合法的骑行场景，��离岸线 200-800 米）误判。
WATER_SPECS: dict[str, tuple[str, float, bool]] = {
    "黄浦江": ("黄浦江", 1500.0, True),
    "长江": ("长江", 2500.0, True),
    "苏州河": ("苏州河", 250.0, False),
}


# 兜底折线（WGS-84）。精度有限，但够判断「是否横穿」。
#
# ⚠️ 这些点是从 OSM 真实岸线里挑的**关键控制点**（不是随手编的）：
# 黄浦江的西岸线在市中心段依次经过
#   (121.4604, 31.1799) -> (121.4651, 31.1868) -> (121.4719, 31.1899)
#   -> (121.4733, 31.1905) -> (121.4876, 31.1935) -> (121.4930, 31.1964)
# 兜底折线必须穿过这些点，否则侧别判定会整体偏掉
# （实测兜底折线偏离真实岸线 300 多米时，浦西轨迹会被误判成横跨）。
_FALLBACK: dict[str, list[tuple[float, float]]] = {
    "黄浦江": [
        (121.1000, 31.2100), (121.2200, 31.1500), (121.3300, 31.1600),
        (121.3800, 31.1700), (121.4300, 31.1760), (121.4604, 31.1799),
        (121.4651, 31.1868), (121.4719, 31.1899), (121.4733, 31.1905),
        (121.4876, 31.1935), (121.4930, 31.1964), (121.5053, 31.2148),
        (121.5300, 31.2400), (121.5700, 31.2600), (121.6200, 31.2800),
        (121.7500, 31.2900), (121.9000, 31.3200),
    ],
    "长江": [
        (121.0000, 31.4000), (121.2000, 31.4000), (121.5000, 31.4000),
        (121.7000, 31.4000), (121.9000, 31.4000),
    ],
    "苏州河": [
        (121.1900, 31.2500), (121.2400, 31.2400), (121.2800, 31.2300),
        (121.3200, 31.2400), (121.3600, 31.2450), (121.4000, 31.2450),
        (121.4400, 31.2450),
    ],
}


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
    """从 Overpass 拉一条河的岸线（WGS-84 经纬度段列表）。"""
    q = (f'[out:json][timeout:{int(timeout)}];'
         f'way["waterway"="river"]["name"="{name}"](30.5,120.5,32.0,122.0);'
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
        for name in WATER_SPECS:
            segs = _fetch_water(name, timeout=timeout)
            if segs:
                out[name] = segs
    if out:
        _store_cached(cache_dir, city, {"ts": time.time(), "waters": out})
        return out

    # 兜底：内置折线（WGS-84）。精度有限，但足以判断是否跨江。
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
