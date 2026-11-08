"""实验：美团 App 截图的底图 能否与高德静态地图做像素级特征匹配。

如果能，则街道级截图的绝对定位可以**完全绕开 OCR 路名锚点**
（路名文字中心与真实路口差几百米，街道级下更不可靠），
直接用底图纹理匹配求出 像素→经纬度 的全局单应/相似变换。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import load_dotenv  # noqa: E402

load_dotenv()
# 密钥只从环境变量 / .env 读取，绝不写进仓库
AMAP_KEY = os.environ.get("AMAP_KEY", "")


def fetch_static(lon: float, lat: float, zoom: int, size: int = 1024) -> np.ndarray | None:
    import urllib.request
    url = (
        f"https://restapi.amap.com/v3/staticmap?location={lon},{lat}"
        f"&zoom={zoom}&size={size}*{size}&key={AMAP_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            buf = np.frombuffer(r.read(), np.uint8)
    except Exception as exc:  # noqa: BLE001
        print("fetch fail", exc)
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def sift_match(a: np.ndarray, b: np.ndarray, ratio: float = 0.75):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(ga, None)
    k2, d2 = sift.detectAndCompute(gb, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    ms = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in ms if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0
    )
    if M is None or inl is None:
        return None
    return M, int(inl.sum()), len(good), k1, k2, good


def main() -> int:
    if not AMAP_KEY:
        print("跳过：未配置 AMAP_KEY（写入 .env 或设置环境变量）")
        return 0
    shot_path = sys.argv[1] if len(sys.argv) > 1 else "samples/meituan_ride_shanghai.jpg"
    shot = cv2.imread(shot_path)
    if shot is None:
        print("shot not found")
        return 1
    print(f"截图 {shot_path} shape={shot.shape}")

    # 盲搜：拉若干中心的静态图，看哪张能匹配上
    # 中心取上次识别的起终点中点附近
    cands = []
    base_lon, base_lat = 121.4170, 31.2026
    for zoom in (12, 13, 14):
        for dlon in (-0.05, -0.025, 0.0, 0.025, 0.05):
            for dlat in (-0.03, 0.0, 0.03):
                cands.append((base_lon + dlon, base_lat + dlat, zoom))

    best = None
    for lon, lat, zoom in cands:
        img = fetch_static(lon, lat, zoom)
        if img is None:
            continue
        r = sift_match(shot, img)
        if r is None:
            continue
        M, inl, ng, *_ = r
        print(f"  z={zoom} loc=({lon:.4f},{lat:.4f}) inliers={inl}/{ng}")
        if best is None or inl > best[0]:
            best = (inl, ng, M, lon, lat, zoom, img)

    if best is None:
        print("没有任何候选匹配上")
        return 1
    inl, ng, M, lon, lat, zoom, img = best
    print(f"\n最佳：z={zoom} loc=({lon},{lat}) inliers={inl}/{ng}")
    print("仿射矩阵：\n", np.round(M, 4))
    print("尺度 =", round(float(np.hypot(M[0, 0], M[1, 0])), 5))
    print("旋转 =", round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 4), "度")
    cv2.imwrite("tmp_debug/basemap_match_best.png", img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
