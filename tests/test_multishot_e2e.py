"""多图拼接的端到端验证（需要服务在跑）。

这个测试回答的是一个具体问题：
**街道级多图拼接，是不是真的比城市级单图更准？**

验证方式不看肉眼，而是拿已知真值的合成数据算误差：
  * 合成一张「街道级美团截图」（底图 + 轨迹线 + 起终点徽章 + 静态 UI）；
  * 沿路线切成 3 张有重叠的图，模拟用户在 App 里滑动截图；
  * 走 `/api/recognize_multi` 拼成大图后识别；
  * 用**米级误差**对比单图识别与多图识别。

真值是构造出来的（路线首尾的经纬度已知），
所以误差数字是客观的，不依赖「看起来对不对」。
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from app.config import load_dotenv  # noqa: E402

load_dotenv()
BASE = os.environ.get("RIDE_BASE", "http://127.0.0.1:8848")
# 密钥只从环境变量 / .env 读取，绝不写进仓库
AMAP_KEY = os.environ.get("AMAP_KEY", "")
TMP = "tmp_debug/e2e"

# 高德静态图 zoom=17、1024px 宽：实测 0.5108 米/像素（底图覆盖 523 米见方）
M_PER_PX = 0.5108
SIZE = 1024

LON0, LAT0 = 121.4579, 31.1917
# 由 tools/pick_synth_route.py 程序化挑选，保证完全落在底图内
ROUTE = [
    (121.459265, 31.192059), (121.458530, 31.191880),
    (121.457795, 31.191610), (121.457060, 31.191790),
    (121.456640, 31.192149), (121.456430, 31.191969),
]
TILE_XS = [197, 387, 577]
TILE_W = 700


def lonlat_to_px(lon: float, lat: float) -> tuple[float, float]:
    dx = (lon - LON0) * 111320.0 * math.cos(math.radians(LAT0)) / M_PER_PX
    dy = -(lat - LAT0) * 111320.0 / M_PER_PX
    return SIZE / 2 + dx, SIZE / 2 + dy


def fetch_static(lon: float, lat: float) -> np.ndarray | None:
    url = (
        f"https://restapi.amap.com/v3/staticmap?location={lon},{lat}"
        f"&zoom=17&size={SIZE}*{SIZE}&key={AMAP_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            buf = np.frombuffer(r.read(), np.uint8)
    except Exception:  # noqa: BLE001
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_shot(base: np.ndarray, route_px) -> np.ndarray:
    """合成一张美团风格的街道级截图。"""
    img = base.copy()
    h, w = img.shape[:2]
    pts = np.array(route_px, dtype=np.int32)
    # 深色描边 + 浅灰填充（美团轨迹线样式）
    cv2.polylines(img, [pts], False, (60, 60, 60), 16, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, (200, 200, 200), 9, cv2.LINE_AA)
    # 起点纯绿（H≈60，落在 start_hsv_range 35-85 内）/ 终点正红
    for pt, color in ((pts[0], (60, 200, 60)), (pts[-1], (60, 60, 235))):
        cv2.circle(img, tuple(pt), 30, (255, 255, 255), -1)
        cv2.circle(img, tuple(pt), 24, color, -1)
    # 静态 UI（每张截图位置相同）
    cv2.rectangle(img, (0, 0), (w, 90), (245, 245, 245), -1)
    cv2.putText(img, "08:17", (40, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (30, 30, 30), 3)
    cv2.circle(img, (w - 120, 45), 32, (220, 220, 220), -1)
    cv2.circle(img, (w - 90, 150), 34, (60, 130, 240), -1)
    cv2.circle(img, (w - 90, h - 140), 42, (255, 255, 255), -1)
    return img


def crop_shot(img: np.ndarray, x0: int, drop_badges: bool) -> np.ndarray:
    """裁出一块，模拟滑动截图。

    真实场景：只有**第一张**有「起」徽章，中间和最后一张都没有
    （最后一张有「终」，但本例中终点在第一张的范围内）。
    """
    crop = img[0:img.shape[0], x0:x0 + TILE_W].copy()
    if drop_badges:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (35, 90, 90), (85, 255, 255))
        m |= cv2.inRange(hsv, (0, 90, 90), (12, 255, 255))
        m |= cv2.inRange(hsv, (170, 90, 90), (179, 255, 255))
        crop[m > 0] = (255, 255, 255)
    return crop


def post_multishot(paths) -> dict:
    boundary = "----wbtest"
    body = b""
    for p in paths:
        with open(p, "rb") as f:
            data = f.read()
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{os.path.basename(p)}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + data + b"\r\n"
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="city"\r\n\r\n上海\r\n'
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


def dist_m(a, b) -> float:
    mlon = 111320.0 * math.cos(math.radians(a[1]))
    return math.hypot((a[0] - b[0]) * mlon, (a[1] - b[1]) * 111320.0)


def path_len_m(coords) -> float:
    return sum(dist_m(p, q) for p, q in zip(coords, coords[1:]))


def main() -> int:
    if not AMAP_KEY:
        print("跳过：未配置 AMAP_KEY（写入 .env 或设置环境变量）")
        return 0
    if not _service_up():
        print("跳过：服务未启动（先跑 python -m app.server）")
        return 0

    os.makedirs(TMP, exist_ok=True)
    base = fetch_static(LON0, LAT0)
    if base is None:
        print("跳过：静态地图拉取失败")
        return 0

    route_px = [lonlat_to_px(lo, la) for lo, la in ROUTE]
    full = make_shot(base, route_px)
    cv2.imwrite(f"{TMP}/full_street.png", full)

    paths = []
    for i, x0 in enumerate(TILE_XS):
        crop = crop_shot(full, x0, drop_badges=(i != 0))
        p = f"{TMP}/tile_{i}.jpg"
        cv2.imwrite(p, crop)
        paths.append(p)

    truth_start = list(ROUTE[0])
    truth_end = list(ROUTE[-1])
    truth_len = 0.0
    for a, b in zip(ROUTE, ROUTE[1:]):
        truth_len += dist_m(a, b)
    print(f"真值：起点 {truth_start} 终点 {truth_end} 长度 {truth_len:.0f} 米")

    # --- 多图 ---
    r3 = post_multishot(paths)
    st = r3.get("stitch", {})
    print(f"\n拼接：ok={st.get('ok')} 用上 {st.get('used')} "
          f"画布 {st.get('canvas_size')}")
    for l in st.get("links", []):
        print(f"  第{l['i'] + 1}↔第{l['j'] + 1}张：内点 {l['inliers']} "
              f"相似度 {l['ncc']} 位移 ({l['tx']}, {l['ty']})")

    coords3 = _coords(r3)
    if not coords3:
        print("多图识别没出坐标，无法校验")
        return 1
    err_s3 = dist_m(coords3[0], truth_start)
    err_e3 = dist_m(coords3[-1], truth_end)
    len3 = path_len_m(coords3)
    print(f"多图：起误差 {err_s3:.0f} 米 终误差 {err_e3:.0f} 米 "
          f"里程 {len3:.0f} 米（真值 {truth_len:.0f}，偏差 "
          f"{abs(len3 - truth_len) / truth_len * 100:.1f}%）")

    # --- 单图（对照：只截到中间那段）---
    r1 = post_multishot([paths[1]])
    coords1 = _coords(r1)
    if coords1:
        len1 = path_len_m(coords1)
        print(f"单图：里程 {len1:.0f} 米（真值 {truth_len:.0f}，偏差 "
              f"{abs(len1 - truth_len) / truth_len * 100:.0f}%）")

    # --- 判定 ---
    ok = True
    if not st.get("ok"):
        print("\n失败：拼接没成功")
        ok = False
    if len(st.get("used") or []) < 2:
        print("\n失败：没有用上 2 张以上的图")
        ok = False
    if err_s3 > 60 or err_e3 > 60:
        print(f"\n失败：起终点误差过大（{err_s3:.0f} / {err_e3:.0f} 米）")
        ok = False
    if abs(len3 - truth_len) / truth_len > 0.25:
        print(f"\n失败：里程偏差过大（{len3:.0f} vs {truth_len:.0f}）")
        ok = False

    print("\n" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


def _coords(resp: dict) -> list[list[float]]:
    gj = resp.get("geojson_gcj02") or resp.get("geojson") or {}
    feats = gj.get("features") or []
    if not feats:
        return []
    return [list(c) for c in feats[0]["geometry"]["coordinates"]]


def _service_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
