"""挑选一组**完全落在底图内**的合成路线参数。

手工调经纬度太容易出错（墨卡托换算 + 底图只覆盖 523 米见方），
这里用程序保证：起点/终点/所有中间点都在画布内，且离边缘留出余量。
"""
from __future__ import annotations

import math

M_PER_PX = 0.5108
SIZE = 1024
MARGIN = 90          # 距边缘至少 90 像素（给徽章半径留空间）


def lonlat_to_px(lon: float, lat: float, lon0: float, lat0: float):
    dx = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0)) / M_PER_PX
    dy = -(lat - lat0) * 111320.0 / M_PER_PX
    return SIZE / 2 + dx, SIZE / 2 + dy


def pick(lon0: float, lat0: float, offsets_m: list[tuple[float, float]]):
    """offsets_m 是相对中心的**米**偏移 (东, 北)。返回经纬度序列与像素序列。"""
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    lls = []
    pxs = []
    for de, dn in offsets_m:
        lon = lon0 + de / m_per_deg_lon
        lat = lat0 + dn / m_per_deg_lat
        lls.append((lon, lat))
        pxs.append(lonlat_to_px(lon, lat, lon0, lat0))
    return lls, pxs


def check(pxs) -> bool:
    for x, y in pxs:
        if not (MARGIN <= x <= SIZE - MARGIN and MARGIN <= y <= SIZE - MARGIN):
            return False
    return True


def main() -> None:
    # 一条 Z 字形，总长约 430 米，最大偏移约 ±130 米
    offsets = [
        (130, 40), (60, 20), (-10, -10), (-80, 10), (-120, 50), (-140, 30),
    ]
    lon0, lat0 = 121.4579, 31.1917
    lls, pxs = pick(lon0, lat0, offsets)
    ok = check(pxs)
    print(f"在界内: {ok}")
    for (lo, la), (x, y) in zip(lls, pxs):
        print(f"  ({lo:.6f}, {la:.6f}) -> px({x:.0f}, {y:.0f})")
    xs = [p[0] for p in pxs]
    ys = [p[1] for p in pxs]
    print(f"像素范围 x[{min(xs):.0f},{max(xs):.0f}] y[{min(ys):.0f},{max(ys):.0f}]")
    # 路线长度
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pxs, pxs[1:]):
        total += math.hypot(x2 - x1, y2 - y1) * M_PER_PX
    print(f"路线长度 = {total:.0f} 米")
    # 切片建议：覆盖 x 范围，前后各留余量
    print(f"\n建议切片（tile 宽 700）: x0 = {[int(min(xs) - 40), int(min(xs) + 150), int(min(xs) + 340)]}")


if __name__ == "__main__":
    main()
