"""端到端验证多图接口 `/api/recognize_multi`。

用高德静态地图合成「美团风格的街道级截图」：
底图 + 深色轨迹线 + 起终点徽章 + 静态 UI，
再切成多张有重叠的图，走接口看能不能拼出完整路线。

这些图的地理真值是已知的（静态图中心点 + 墨卡托换算），
所以能对结果做**客观的像素级/米级校验**，而不是靠肉眼判断。
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import load_dotenv  # noqa: E402
from app.maps.staticmap import amap_key, fetch_static_map  # noqa: E402

load_dotenv()
# 密钥只从环境变量 / .env 读取，绝不写进仓库
AMAP_KEY = amap_key() or ""
BASE = "http://127.0.0.1:8848"

# 高德静态图 zoom=17、size=1024px 宽时的实测分辨率：
# 经度差 0.005° = 476 米 = 932 像素 → 0.5108 米/像素
# （相当于标准切片 zoom=18；1024px 只覆盖 523 米见方）
M_PER_PX = 0.5108
COVER_M = 1024 * M_PER_PX          # 底图覆盖 523 米


def fetch_static(lon: float, lat: float, zoom: int = 17, size: int = 1024):
    """取静态底图。实现收敛到 app/maps/staticmap/。"""
    res = fetch_static_map(lon, lat, zoom, size, provider="amap", key=AMAP_KEY)
    if res is None:
        raise RuntimeError("静态地图获取失败")
    buf = np.frombuffer(res.data, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def lonlat_to_px(lon: float, lat: float, lon0: float, lat0: float, size: int):
    """把经纬度转成「以 (lon0,lat0) 为中心、size 像素宽」的像素坐标。"""
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    dx = (lon - lon0) * m_per_deg_lon / M_PER_PX
    dy = -(lat - lat0) * m_per_deg_lat / M_PER_PX
    return size / 2 + dx, size / 2 + dy


def make_shot(base, route_px, with_ui=True):
    """在底图上画轨迹线与起终点徽章，模拟美团截图。

    配色必须与真实截图一致，否则测的是合成图的缺陷而不是流程的缺陷：
      - 真实「起」徽章是**纯绿**（HSV H≈60，落在 start_hsv_range 35-85 内），
        用 BGR (60,200,90) 是错的——那是 H≈105 的蓝绿，落在范围外。
      - 真实「终」徽章是**正红**（H≈0 / 170-179）。
      - 白色描边要画在徽章**下面**（真实截图是图标自带白边，
        不是在色块上再挖一圈）。
    """
    img = base.copy()
    h, w = img.shape[:2]
    pts = np.array(route_px, dtype=np.int32)
    # 美团风格：深色描边 + 浅灰填充
    cv2.polylines(img, [pts], False, (60, 60, 60), 16, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, (200, 200, 200), 9, cv2.LINE_AA)
    # 起点绿 / 终点红（先画白底，再填色）
    for pt, color in ((pts[0], (60, 200, 60)), (pts[-1], (60, 60, 235))):
        cv2.circle(img, tuple(pt), 30, (255, 255, 255), -1)
        cv2.circle(img, tuple(pt), 24, color, -1)
    if with_ui:
        cv2.rectangle(img, (0, 0), (w, 90), (245, 245, 245), -1)
        cv2.putText(img, "08:17", (40, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (30, 30, 30), 3)
        cv2.circle(img, (w - 120, 45), 32, (220, 220, 220), -1)
        cv2.circle(img, (w - 90, 150), 34, (60, 130, 240), -1)
        cv2.circle(img, (w - 90, h - 140), 42, (255, 255, 255), -1)
    return img


def crop_shot(img, x0, y0, w, h, drop_badges=False):
    """裁出一块（模拟用户在 App 里平移后截到的区域）。

    ``drop_badges=True`` 会把落在裁剪区内的起终点徽章涂掉 ——
    模拟真实场景：**滑动截图中途的那张通常没有起终点徽章**，
    只有第一张有「起」、最后一张有「终」。
    """
    crop = img[y0:y0 + h, x0:x0 + w].copy()
    if drop_badges:
        ch, cw = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 起点绿
        m = cv2.inRange(hsv, (35, 90, 90), (85, 255, 255))
        # 终点红
        m |= cv2.inRange(hsv, (0, 90, 90), (12, 255, 255))
        m |= cv2.inRange(hsv, (170, 90, 90), (179, 255, 255))
        crop[m > 0] = (255, 255, 255)
    return crop


def post_multishot(paths, city="上海", road_names=""):
    """走多图接口。"""
    boundary = "----wbtest"
    body = b""
    for p in paths:
        with open(p, "rb") as f:
            data = f.read()
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{p.split("/")[-1]}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + data + b"\r\n"
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="city"\r\n\r\n{city}\r\n'
    ).encode()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="road_names"\r\n\r\n{road_names}\r\n'
    ).encode()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="refine"\r\n\r\nfalse\r\n'
    ).encode()
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/api/recognize_multi", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    if not AMAP_KEY:
        print("跳过：未配置 AMAP_KEY（写入 .env 或设置环境变量）")
        return 0
    outdir = "tmp_debug/e2e"
    import os
    os.makedirs(outdir, exist_ok=True)

    # 路线：一条 Z 字形，走在「枫林路 / 小木桥路」一带
    #
    # ⚠️ 路线必须**完全落在 1024px 底图范围内**（底图只覆盖 523 米见方），
    # 否则起点/终点会被裁到画布外，徽章自然检不出来——
    # 那是测试数据的缺陷，不是流程的缺陷。
    # 参数由 tools/pick_synth_route.py 程序化挑选，保证在界内。
    lon0, lat0 = 121.4579, 31.1917
    route_lonlat = [
        (121.459265, 31.192059), (121.458530, 31.191880),
        (121.457795, 31.191610), (121.457060, 31.191790),
        (121.456640, 31.192149), (121.456430, 31.191969),
    ]
    base = fetch_static(lon0, lat0)
    route_px = [lonlat_to_px(lo, la, lon0, lat0, base.shape[1]) for lo, la in route_lonlat]
    full = make_shot(base, route_px)
    cv2.imwrite(f"{outdir}/full_street.png", full)

    # 真值：路线首尾的经纬度
    print(f"真值起点 {route_lonlat[0]} 终点 {route_lonlat[-1]}")

    # 切成 3 张有重叠的图（模拟沿路线滑动截图）
    #
    # 真实场景：只有**第一张**有「起」徽章、**最后一张**有「终」徽章，
    # 中间那张两个都没有。这跟「拼图后再识别」完全相反——
    # 拼图会把三张的徽章都保留，所以中间那张反而没有徽章这件事
    # 必须在**拼接前**就处理掉，否则拼出来的图上会出现
    # 「路线中间有一个假的起点」，轨迹提取会被带偏。
    H, W = full.shape[:2]
    tile_w = 700
    xs = [197, 387, 577]
    paths = []
    for i, x0 in enumerate(xs):
        crop = crop_shot(full, x0, 0, tile_w, H, drop_badges=(i != 0))
        p = f"{outdir}/tile_{i}.jpg"
        cv2.imwrite(p, crop)
        paths.append(p)
        print(f"  tile{i}: x={x0}..{x0 + tile_w}"
              f"{'（含起点徽章）' if i == 0 else '（无徽章）'}")

    # --- 单图识别（对照）---
    print("\n=== 单图（只截到路线中间一段）===")
    r1 = post_multishot([paths[1]])
    print(f"  ok={r1.get('ok')} 置信度={r1.get('confidence'):.3f} "
          f"点数={r1.get('track_points')} 里程={r1.get('distance_m', 0):.0f}m")
    for w in r1.get("warnings", [])[:4]:
        print(f"    · {w}")

    # --- 多图识别 ---
    print("\n=== 三图拼接 ===")
    r3 = post_multishot(paths)
    st = r3.get("stitch", {})
    print(f"  拼接 ok={st.get('ok')} 用上 {st.get('used')} 排除 {st.get('unused')}")
    for l in st.get("links", []):
        print(f"    link {l['i']}->{l['j']} 内点{l['inliers']} "
              f"相似度{l['ncc']} 位移({l['tx']},{l['ty']})")
    print(f"  画布 {st.get('canvas_size')}")
    print(f"  识别 ok={r3.get('ok')} 置信度={r3.get('confidence'):.3f} "
          f"点数={r3.get('track_points')} 里程={r3.get('distance_m', 0):.0f}m")
    for w in r3.get("warnings", [])[:6]:
        print(f"    · {w}")

    # --- 校验里程 ---
    # 真值路线长度（按平面近似）
    from app.geo.coords import gcj02_to_wgs84  # noqa: F401
    total = 0.0
    for (lo1, la1), (lo2, la2) in zip(route_lonlat, route_lonlat[1:]):
        mlon = 111320.0 * math.cos(math.radians((la1 + la2) / 2))
        total += math.hypot((lo2 - lo1) * mlon, (la2 - la1) * 111320.0)
    print(f"\n  真值路线长度 = {total:.0f} 米")
    print(f"  单图识别里程 = {r1.get('distance_m', 0):.0f} 米")
    print(f"  多图识别里程 = {r3.get('distance_m', 0):.0f} 米")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ".")
    raise SystemExit(main())
