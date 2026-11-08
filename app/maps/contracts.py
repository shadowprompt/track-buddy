"""地图能力的公共契约：数据形状、坐标系与 provider 元信息。

这一层不认识任何厂商，也不发任何网络请求。它只回答三个问题：

1. 一份地图数据是什么形状（GeoPoint / PlaceCandidate / ...）；
2. 它用的是哪个坐标系（CRS），由谁负责换算；
3. 一个 provider 必须声明什么（ProviderMeta）。

**为什么 CRS 必须进契约**：项目内部一律使用 GCJ-02（美团 / 高德 /
腾讯底图），而外部源可能是 WGS-84（OSM / Overpass）、CGCS2000
（天地图）、BD-09（百度）。过去靠注释和记忆在各 provider 里手动转换，
漏一次就是数百米的系统性偏移（天地图实测约 570 m）。现在 provider
只声明 ``output_crs``，框架在契约层统一换算，新增厂商不可能漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..geo.coords import bd09_to_gcj02, wgs84_to_gcj02


class CRS(str, Enum):
    """外部地图源可能使用的坐标系。"""

    WGS84 = "WGS-84"
    CGCS2000 = "CGCS2000"
    GCJ02 = "GCJ-02"
    BD09 = "BD-09"


# CGCS2000 与 WGS-84 采用同一椭球、工程精度上等价（厘米级差异），
# 两者都需要做一次 GCJ-02 加偏修正。
_WGS84_LIKE = frozenset({CRS.WGS84, CRS.CGCS2000})


def to_gcj02(lon: float, lat: float, crs: CRS | str) -> tuple[float, float]:
    """把任意「已声明坐标系」的点换算到项目内部基准 GCJ-02。

    这是所有外部坐标进入管线的唯一换算入口。provider 不得自行换算后
    再把结果标成别的坐标系，否则会出现重复加偏。
    """
    if isinstance(crs, str):
        crs = CRS(crs)
    if crs in _WGS84_LIKE:
        return wgs84_to_gcj02(float(lon), float(lat))
    if crs is CRS.BD09:
        return bd09_to_gcj02(float(lon), float(lat))
    if crs is CRS.GCJ02:
        return (float(lon), float(lat))
    raise ValueError(f"未支持的坐标系: {crs!r}")


@dataclass(frozen=True)
class GeoPoint:
    """一个带坐标系声明的经纬度点。"""

    lon: float
    lat: float
    crs: CRS = CRS.GCJ02

    @property
    def gcj02(self) -> tuple[float, float]:
        return to_gcj02(self.lon, self.lat, self.crs)

    def as_dict(self) -> dict[str, Any]:
        lo, la = self.gcj02
        return {"lon": lo, "lat": la, "crs": self.crs.value}


@dataclass(frozen=True)
class PlaceCandidate:
    """地名检索的一个候选点，对外一律暴露 GCJ-02 坐标。"""

    name: str
    point: GeoPoint
    quality: float = 1.0
    source: str = ""
    address: str = ""
    kind: str = ""

    @property
    def lon(self) -> float:
        return self.point.gcj02[0]

    @property
    def lat(self) -> float:
        return self.point.gcj02[1]


class Capability(str, Enum):
    """项目用到的地图能力。替换服务商以「能力」为单位，不以文件为单位。"""

    GEOCODING = "geocoding"              # 地名 -> 坐标
    ROADNAME = "roadname"                # 坐标/轨迹 -> 路名
    ROAD_GEOMETRY = "road_geometry"      # 道路几何（OSM/Overpass 等）
    ROUTE_VALIDATION = "route_validation"  # 路线几何交叉验证
    BASEMAP = "basemap"                  # 网页底图（JS SDK，阶段 3）
    STATIC_MAP = "static_map"            # 静态地图素材（阶段 4）


@dataclass(frozen=True)
class ProviderMeta:
    """每个 provider 必须声明的静态信息。

    ``output_crs`` 是硬要求：任何产出坐标的 provider 都必须说明它给的
    是哪套坐标系，由框架换算，而不是让调用方去记。
    """

    name: str
    capability: Capability
    output_crs: CRS = CRS.GCJ02
    credential_aliases: tuple[str, ...] = ()
    needs_network: bool = True


@dataclass
class ProbeResult:
    """健康探测结果，供 /api/health 直接序列化。"""

    ok: bool
    error: str = ""
    hint: str = ""
    hits: int = 0
    sample: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            out["hits"] = self.hits
            if self.sample:
                out["sample"] = self.sample
        else:
            if self.error:
                out["error"] = self.error
            if self.hint:
                out["hint"] = self.hint
        return out


class MapProvider(Protocol):
    """provider 的最小接口；具体实现可自由增加方法。"""

    meta: ProviderMeta

    def is_available(self) -> bool: ...

    def probe(self) -> ProbeResult: ...
