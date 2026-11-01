"""坐标系转换与投影。

关键点：中国境内合规地图服务（高德/腾讯/百度）返回的是 GCJ-02 火星坐标，
而 WGS-84 才是 GeoJSON / GPX 的通用基准。两者在中国境内存在 100-700 米的
系统性偏移，若混用会导致轨迹整体错位数百米。

因此本项目内部统一以 GCJ-02 工作（与截图底图一致），输出时按需转换。
"""

from __future__ import annotations

import math

# WGS-84 相关常数
_A = 6378245.0
_EE = 0.00669342162296594323

# GCJ-02 加密偏移参数
_X_PI = math.pi * 3000.0 / 180.0


def _out_of_china(lon: float, lat: float) -> bool:
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(lon: float, lat: float) -> float:
    """纬度方向的加密偏移量。

    参数是 (lon-105, lat-35)，与公开算法一致。
    """
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat
    ret += 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(lon: float, lat: float) -> float:
    """经度方向的加密偏移量。

    参数是 (lon-105, lat-35)，与公开算法一致。
    """
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat
    ret += 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * math.pi) + 40.0 * math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 * math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    """WGS-84 → GCJ-02 火星坐标。境外坐标原样返回。

    注意：dlon 必须来自 ``_transform_lon``，dlat 来自 ``_transform_lat``。
    早期版本两个方向都用了 ``_transform_lat``（经度偏差约 375 米），
    这会让整个配准整体错位数百米。
    """
    if _out_of_china(lon, lat):
        return lon, lat
    dlon, dlat = _delta(lon, lat)
    return lon + dlon, lat + dlat


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    """GCJ-02 → WGS-84。

    实现要点：GCJ-02 的偏移量 dlon/dlat 依赖于**原始 WGS-84 坐标**，
    而我们手上只有 GCJ-02 坐标，因此这是一个隐式方程：
        lon_wgs = lon_gcj - dlon(lon_wgs, lat_wgs)
    早期实现用「以 GCJ 坐标为基准算偏移再减去」的一步近似，
    结果 GCJ→WGS→GCJ 往返会把点平移约 347 米——这会让轨迹整体偏移，
    且里程凭空缩短 15%。

    现在用不动点迭代求解：以当前估计为初值，反复用「当前估计的偏移量」
    修正，直到收敛（实测 4 次迭代即收敛到亚米级）。
    """
    if _out_of_china(lon, lat):
        return lon, lat
    # 初值：一步近似
    dlon0, dlat0 = _delta(lon, lat)
    glon, glat = lon - dlon0, lat - dlat0
    # 不动点迭代：offset 由 WGS 坐标决定，反过来修正 WGS 坐标
    for _ in range(8):
        dlon, dlat = _delta(glon, glat)
        nlon, nlat = lon - dlon, lat - dlat
        if abs(nlon - glon) < 1e-11 and abs(nlat - glat) < 1e-11:
            glon, glat = nlon, nlat
            break
        glon, glat = nlon, nlat
    return glon, glat


def _delta(lon: float, lat: float) -> tuple[float, float]:
    """返回 (dlon, dlat)：该点从 WGS-84 到 GCJ-02 的加密偏移。

    两个方向必须用各自的变换函数——这是本模块最关键的一行。
    """
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * math.pi)
    return dlon, dlat


def gcj02_to_bd09(lon: float, lat: float) -> tuple[float, float]:
    """GCJ-02 → 百度坐标。"""
    x = lon
    y = lat
    z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * _X_PI)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * _X_PI)
    return z * math.cos(theta) + 0.0065, z * math.sin(theta) + 0.006


def bd09_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    """百度坐标 → GCJ-02。"""
    x, y = lon - 0.0065, lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * _X_PI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * _X_PI)
    return z * math.cos(theta), z * math.sin(theta)


# --- Web 墨卡托投影（EPSG:3857）---

_MERC_R = 6378137.0


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """经纬度 → 墨卡托米。用于像素↔地理的线性配准。"""
    x = math.radians(lon) * _MERC_R
    y = _MERC_R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """墨卡托米 → 经纬度。"""
    lon = math.degrees(x / _MERC_R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / _MERC_R)) - math.pi / 2.0)
    return lon, lat


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """两点球面距离（米）。"""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
