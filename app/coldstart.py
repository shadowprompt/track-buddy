"""冷启动地理编码：用 OSM 为「词典里没有的地名」找坐标。

为什么需要它
------------
识别能力的第一道门槛是「知道这些地名在哪」。此前只有两条路：

1. 本地词典 ``data/lexicon.json``——只覆盖已核验过的少量地名；
2. 腾讯/高德/百度/天地图的在线地理编码——需要用户自己申请密钥。

换一个区域就完全无法识别：OCR 明明读出了「安龙小区」「进才实验中学」
这些真实地名，却因为「本地没这个词、也没配密钥」而拿不到任何坐标，
配准直接失败（实测新截图：0 个锚点命中，配准失败）。

本模块补上第三条路：**按城市限定范围去 OSM 检索地名**。
它与项目既有做法完全一致——``app/seed_lexicon.py`` 的初始词典种子
本来就是「viewbox 限定的地理编码查询」得到的。

三阶段策略
----------
**阶段一：多查询变体收集候选**

Nominatim 的索引对前缀/后缀敏感，同一个 POI 换个写法结果就不同：

    进才实验中学   → 0 条
    上海市进才实验中学 → 1 条  ✓

所以每个地名用多个变体检索：``{名} {市}``、``{市}{名}``、``{名}``。

**阶段二：名称相似度打分**

OCR 会截断地名（「进才实验中学」实际是「上海市进才实验中学」，
「水清木华」实际是「水清木华公寓」），因此不能要求完全同名。
用 Jaccard 字符相似度 + 子串加权打分，低于阈值直接丢弃。

**阶段三：多锚点互证定位**

单个地名无法自证位置——「花园2期」能匹配到朗香/星宇/尼德兰/三湘盛世
四个不同小区的「2期」，取哪个都可能错。

但**多个独立检索的地名若彼此都落在 1 公里内**，那这片区域就是真的。
实测浦东一例：华丽家族花园、天安花园、当代清水园、水清木华公寓
四个地名各自独立检索，命中点互相都在 300-500 米内，
由此锁定区域中心 (121.5537, 31.2282)，再用它过滤其余候选。

这个阶段的必要性由一次真实失败推动：早期版本用「完全同名 + 城市级
viewbox」，浦东那张截图 6 个 POI 全部 0 命中——viewbox 对上海市这样
的大范围 bbox 几乎没有过滤作用，同名异地地点混了进来。

合规说明
--------
本模块**只用于底图与地理编码的坐标参考**，不用于底图渲染，
与 ``app/roadfit.py`` 用 OSM 路网做几何精修是同一性质：
提供客观地理几何/坐标，不提供地图画面。
页面底图仍是腾讯地图；在线地理编码优先级仍高于本模块。

精度说明
--------
OSM 的 POI 点位与国内底图（美团/高德/腾讯）同源 POI 可能有 100-300 米偏差。
这正是 ``app/roadfit.py``「路网约束精修」存在的原因：
它用道路几何（各底图必须一致）纠正 POI 锚点带来的整体偏移。
所以本模块的产出被标记为 ``osm`` 源，质量权重低于用户确认与在线服务，
并最终由路网精修兜住绝对位置。
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .geo.coords import wgs84_to_gcj02

NOMINATIM = "https://nominatim.openstreetmap.org/search"

# Nominatim 使用政策要求最多 1 次/秒。必须遵守，
# 否则会被封 IP——这个项目是单机自用工具，封了就彻底没法冷启动了。
_MIN_INTERVAL = 1.05

# 名称相似度阈值：低于此值直接丢弃。
# 定这个值是因为「何文田上海水配水庫」与「乡土田园区」的相似度只有 0.08，
# 而「水清木华公寓」与「水清木华」是 0.92。0.55 落在中间，且不误伤。
_MIN_NAME_SCORE = 0.55

# 互证半径（米）：候选点彼此的距离小于它，才算互相印证。
# 取 800 是因为「同一小区群内不同楼栋」的距离就在这个量级，
# 而不同街区的同名地点通常隔 1 公里以上。
_CORROBORATE_RADIUS_M = 800.0

# 锚点与区域中心的最大允许距离（米）。
# 超过说明该候选不属于这个区域——比如「花园2期」在别的区也有。
_MAX_CENTER_DIST_M = 1500.0


@dataclass
class OsmPlace:
    """OSM 检索到的一个地点。"""

    name: str
    lon: float  # GCJ-02
    lat: float  # GCJ-02
    kind: str  # poi / transit / other
    score: float = 1.0  # 与 OCR 名的相似度
    osm_type: str = ""
    osm_id: int = 0

    @property
    def is_poi(self) -> bool:
        return self.kind == "poi"


def normalize_city(city: str) -> str:
    """把用户输入的城市规整成检索用的名字。

    「北京」「上海」「天津」「重庆」四个直辖市补「市」后更稳。
    """
    c = (city or "").strip()
    if not c:
        return ""
    if c.endswith(("市", "区", "县")):
        return c
    if c in ("北京", "上海", "天津", "重庆"):
        return c + "市"
    return c


def _city_stem(city: str) -> str:
    """去掉「市」后缀，得到「上海」这样的短名。"""
    c = normalize_city(city)
    return c[:-1] if c.endswith("市") else c


def name_score(ocr: str, cand: str) -> float:
    """OCR 名与 OSM 候选名的相似度（0~1）。

    真实案例说明为什么不能要求完全同名：
      OCR「进才实验中学」 → OSM「上海市进才实验中学」（前缀被截）
      OCR「水清木华」     → OSM「水清木华公寓」（后缀被截）
      OCR「花园2期」     → OSM「朗香花园2期」（中间被截）
    """
    if not ocr or not cand:
        return 0.0
    if ocr == cand:
        return 1.0
    if ocr in cand:
        # OCR 名是候选名的子串：按覆盖比例给分，完整覆盖给 1.0
        return 0.75 + 0.25 * len(ocr) / len(cand)
    if cand in ocr:
        return 0.70
    sa, sb = set(ocr), set(cand)
    return len(sa & sb) / max(len(sa | sb), 1) * 0.9


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """两点距离（米）。"""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class ColdStartGeocoder:
    """按城市检索 OSM 地名，返回候选坐标。

    带进程内缓存与磁盘缓存，避免同一地名反复打网络。
    """

    def __init__(self, cache_path: str | None = None):
        self.cache_path = cache_path
        self._mem: dict[str, list[OsmPlace]] = {}
        self._lock = threading.RLock()
        self._last_call = 0.0
        self._bbox_cache: dict[str, tuple[float, float, float, float]] = {}
        if cache_path:
            self._load_disk()

    # ---------- 缓存 ----------

    def _load_disk(self) -> None:
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        for key, items in raw.items():
            self._mem[key] = [
                OsmPlace(
                    name=p["name"], lon=p["lon"], lat=p["lat"],
                    kind=p.get("kind", "poi"), score=p.get("score", 1.0),
                    osm_type=p.get("osm_type", ""), osm_id=p.get("osm_id", 0),
                )
                for p in items
            ]

    def _save_disk(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        k: [
                            {"name": p.name, "lon": p.lon, "lat": p.lat,
                             "kind": p.kind, "score": p.score,
                             "osm_type": p.osm_type, "osm_id": p.osm_id}
                            for p in v
                        ]
                        for k, v in self._mem.items()
                    },
                    f, ensure_ascii=False,
                )
        except OSError:
            pass

    # ---------- 网络 ----------

    def _throttle(self) -> None:
        with self._lock:
            wait = _MIN_INTERVAL - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _request(self, params: dict[str, Any]) -> list[dict]:
        url = NOMINATIM + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": "meituan-ride/0.1 (self-hosted local tool)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — 网络失败不应让整次识别崩掉
            return []

    def city_bbox(self, city: str) -> tuple[float, float, float, float] | None:
        """取城市的经纬度范围（WGS-84）。

        Nominatim 返回的 boundingbox 顺序是 [south, north, west, east]。
        """
        cname = normalize_city(city)
        if not cname:
            return None
        with self._lock:
            if cname in self._bbox_cache:
                return self._bbox_cache[cname]
        self._throttle()
        data = self._request({
            "q": cname, "format": "jsonv2", "limit": 1,
            "countrycodes": "cn", "accept-language": "zh-CN",
        })
        if not data:
            return None
        b = data[0].get("boundingbox")
        if not b or len(b) != 4:
            return None
        try:
            south, north, west, east = (float(x) for x in b)
        except (TypeError, ValueError):
            return None
        bbox = (west, south, east, north)
        with self._lock:
            self._bbox_cache[cname] = bbox
        return bbox

    def _variants(self, name: str, city: str) -> list[str]:
        """构造查询变体。

        Nominatim 的索引对前缀敏感：「进才实验中学」查不到，
        「上海市进才实验中学」能查到。所以要多试几种写法。

        但每个变体都要付 1 秒限速的代价，因此按「最可能命中」排序，
        并在前两个变体就查到足够候选时提前返回。
        """
        cname = normalize_city(city)
        stem = _city_stem(city)
        out = [
            f"{name} {cname}",
            f"{cname}{name}",
            f"{stem}{name}",
            name,
        ]
        seen: set[str] = set()
        res: list[str] = []
        for v in out:
            if v and v not in seen:
                seen.add(v)
                res.append(v)
        return res

    def _classify(self, raw_name: str, d: dict) -> str:
        """判断 OSM 要素能否作配准锚点。"""
        cat = (d.get("category") or "").strip()
        osm_class = (d.get("type") or "").strip()
        if cat in ("highway", "waterway") or osm_class in ("road",):
            # 道路/水系：文字沿走向排布，中心不对应固定点位，不可作锚点
            return "other"
        if cat in ("public_transport", "railway") or "地铁" in raw_name \
                or "号线" in raw_name or raw_name.endswith("站"):
            return "transit"
        return "poi"

    def _collect_candidates(
        self, name: str, city: str, min_hits: int = 3
    ) -> list[OsmPlace]:
        """阶段一 + 二：多变体检索 + 相似度打分。

        ``min_hits``：查到这么多候选就停止试后续变体。
        每个变体要付 1 秒限速的代价，而 3 个候选已足够 RANSAC 挑选。
        """
        bbox = self.city_bbox(city)
        best: dict[tuple[float, float], OsmPlace] = {}
        for q in self._variants(name, city):
            params: dict[str, Any] = {
                "q": q,
                "format": "jsonv2",
                "limit": 20,
                "countrycodes": "cn",
                "accept-language": "zh-CN",
            }
            if bbox:
                # viewbox 只是排序提示，**不做强制过滤**——实测它对
                # 上海市这种大范围 bbox 几乎不起作用，同名异地地点照样返回。
                # 真正的过滤由阶段三的互证负责。
                params["viewbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            self._throttle()
            for d in self._request(params):
                raw_name = (d.get("name") or "").strip()
                if not raw_name:
                    continue
                s = name_score(name, raw_name)
                if s < _MIN_NAME_SCORE:
                    continue
                try:
                    lon = float(d["lon"])
                    lat = float(d["lat"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (round(lon, 4), round(lat, 4))
                prev = best.get(key)
                if prev is None or s > prev.score:
                    glon, glat = wgs84_to_gcj02(lon, lat)
                    best[key] = OsmPlace(
                        name=raw_name, lon=glon, lat=glat,
                        kind=self._classify(raw_name, d), score=s,
                        osm_type=d.get("osm_type") or "",
                        osm_id=int(d.get("osm_id") or 0),
                    )
            # 已经拿到足够候选就不用再试其他写法了
            if len(best) >= min_hits:
                break
        return sorted(best.values(), key=lambda p: -p.score)

    @staticmethod
    def _find_region_center(
        cands: dict[str, list[OsmPlace]],
    ) -> tuple[float, float, set[str]] | None:
        """阶段三：多锚点互证，找出互相印证的区域中心。

        判据：候选点周边有 ≥1 个**其他地名**的候选在 800 米内。
        取印证数量最多、总距离最小的那个点作为区域中心。

        返回的第三个元素是「被印证的地名集合」——**包含簇心自身的地名**。
        簇心选在某个候选点上，那个地名本身就是互证的一环：
        「进才实验中学」与「水清木华公寓」相距 500 米，
        以其中一个为簇心时，另一个在 800 米内 → 互证成立。

        为什么 1 个就够：实测浦东一例只有两个地名能查到（其余 4 个
        在 OSM 未收录），但「进才实验中学」与「水清木华公寓」互为印证
        且相距仅 500 米——两个独立检索的地名落在同一片区域，
        这本身就是很强的证据。要求 ≥2 个反而会让这类区域定位失败。

        注意互证是**双向**的：单看「进才实验中学」周围有「水清木华」，
        而「水清木华」周围也有「进才实验中学」，两者都成立。
        而「花园2期」的四个候选（星宇/朗香/尼德兰/三湘盛世花园2期）
        彼此相距 2-5 公里，谁也不印证谁——正是这个差别把它们区分开了。
        """
        flat: list[tuple[float, float, str]] = []
        for ocr, lst in cands.items():
            for p in lst:
                flat.append((p.lon, p.lat, ocr))
        if len(flat) < 2:
            return None

        best: tuple[Any, float, float, set[str]] | None = None
        for lon, lat, ocr in flat:
            near = [
                (o2, d) for lo2, la2, o2 in flat
                if o2 != ocr
                and (d := _haversine_m(lon, lat, lo2, la2)) <= _CORROBORATE_RADIUS_M
            ]
            distinct = {o2 for o2, _ in near}
            if not distinct:
                continue
            # 印证的地名集合包含簇心自身的地名——它本身也是互证的一环
            corroborated = distinct | {ocr}
            total_d = sum(d for _, d in near)
            key = (len(corroborated), -total_d)
            if best is None or key > (best[0], best[1]):
                best = (len(corroborated), -total_d, lon, lat, corroborated)
        if best is None:
            return None
        return best[2], best[3], best[4]

    def lookup_many(
        self, names: list[str], city: str = ""
    ) -> dict[str, list[OsmPlace]]:
        """批量检索多个地名（含互证定位）。

        这是对外的主入口：单锚点无法自证位置，必须成批处理。

        互证需要至少 3 个不同地名才有意义。待查地名不足 3 个时，
        直接用各自的最佳候选——由 RANSAC 的几何一致性兜底。
        """
        if not names or not city:
            return {}

        cands: dict[str, list[OsmPlace]] = {}
        for n in names:
            key = f"{normalize_city(city)}|{n}"
            with self._lock:
                hit = self._mem.get(key)
            if hit is not None:
                cands[n] = hit
                continue
            got = self._collect_candidates(n, city)
            cands[n] = got
            with self._lock:
                self._mem[key] = got
        self._save_disk()

        # 待查地名太少 → 互证无从谈起，直接给最佳候选。
        # 1 个地名时确实无从互证；2 个是可以的（双向印证，见 _find_region_center）。
        if len([n for n, lst in cands.items() if lst]) < 2:
            return {n: lst[:3] for n, lst in cands.items() if lst}

        center = self._find_region_center(cands)
        if center is None:
            # 互证失败：退回「每个地名各自的最佳候选」。
            # RANSAC 会用几何一致性挑出自洽的那一组。
            return {n: lst[:3] for n, lst in cands.items() if lst}

        clon, clat, corroborated = center
        out: dict[str, list[OsmPlace]] = {}
        for n, lst in cands.items():
            keep = [
                p for p in lst
                if _haversine_m(clon, clat, p.lon, p.lat) <= _MAX_CENTER_DIST_M
            ]
            if keep:
                out[n] = keep[:3]
        return out

    def lookup(
        self, name: str, city: str = "", kinds: tuple[str, ...] = ("poi",)
    ) -> list[OsmPlace]:
        """检索单个地名（不保证互证，仅作兼容入口）。"""
        res = self.lookup_many([name], city)
        return [p for p in res.get(name, []) if p.kind in kinds]


# 进程级单例：磁盘缓存路径由调用方注入
_COLD: ColdStartGeocoder | None = None
_COLD_LOCK = threading.Lock()


def get_cold_start(cache_path: str | None = None) -> ColdStartGeocoder:
    global _COLD
    with _COLD_LOCK:
        if _COLD is None:
            _COLD = ColdStartGeocoder(cache_path)
        return _COLD
