"""实测：多图拼接到底需要多少重叠？（街道级底图，独立渲染）

## 用户的问题

> 对截图合并的要求高吗，如果几张截图的并没有保证相邻的两张之前有三分之一的重合

## 实验设计（两次踩坑后的正确做法）

### 第一次踩坑：用「同一张图切片」当两张截图 —— ❌ 错
切片出来的两张图重叠区**像素级完全相同**，SIFT 反而分不清
「同一片」和「相似片」：匹配对只有 225 个，位移分布弥散（标准差 153px），
完全配不出真值。

### 第二次踩坑：用真实美团城市级截图 —— ❌ 也不合适
城市级底图只有行政区/地铁线/大 POI，**结构高度重复**，
实测任意两个视角的匹配位移中位数都是 0、离散度 200-500px —— 全是误匹配。
城市级底图本来就不适合做特征匹配（这与 §21.4「城市级底图匹配不可行」一致）。

### 正确做法：街道级底图 + **独立渲染**的不同中心
模拟用户「在 App 里滑动地图连续截图」：
每次请求都是独立渲染（带轻微的渲染差异，模拟真实截图），
中心按经度递增，重叠比例可控。

真值完全已知：经度差 → 地面距离 → 像素位移。
"""

from __future__ import annotations

import glob
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_dotenv  # noqa: E402
from app.multishot import _describe, _match_pair  # noqa: E402

load_dotenv()

# 街道级实测分辨率：zoom=17、1024px 宽 → 0.5108 米/像素
M_PER_PX = 0.5108
LAT0 = 31.1920


def px_per_deg_lon(lat: float = LAT0) -> float:
    """1 经度对应多少像素（zoom=17、1024px 宽）。"""
    m_per_deg = 111320.0 * math.cos(math.radians(lat))
    return m_per_deg / M_PER_PX


def load_slides() -> list[tuple[str, float]]:
    """加载滑动序列，返回 (文件, 经度)。"""
    out = []
    for f in sorted(glob.glob("tmp_debug/zoomtest/sl_*.png")):
        # 文件名里带经纬度
        base = os.path.basename(f)[3:-4]
        lon = float(base.split("_")[0])
        out.append((f, lon))
    out.sort(key=lambda x: -x[1])   # 经度递减 = 从东往西滑
    return out


def make_pair(dlon: float) -> tuple[np.ndarray, np.ndarray, float]:
    """渲染两张中心相差 dlon 经度的图，返回 (a, b, 真值位移)。

    真值：b 的中心在 a 的西边 dlon 度 ⇒ b 的像素 (x,y) 对应 a 的 (x - dx, y)
    ⇒ 「a→b」的位移是 +dx。
    """
    import urllib.request

    key = os.environ.get("AMAP_KEY", "")
    lon0 = 121.4600
    lat = LAT0

    def fetch(lon: float) -> np.ndarray:
        url = (f"https://restapi.amap.com/v3/staticmap?location={lon},{lat}"
               f"&zoom=17&size=1024*1024&key={key}")
        with urllib.request.urlopen(url, timeout=30) as r:
            buf = np.frombuffer(r.read(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    a = fetch(lon0)
    b = fetch(lon0 - dlon)
    true_dx = dlon * px_per_deg_lon(lat)
    return a, b, true_dx


def main() -> int:
    if not os.environ.get("AMAP_KEY"):
        print("跳过：未配置 AMAP_KEY（写入 .env 或设置环境变量）")
        return 0
    os.makedirs("tmp_debug/overlap", exist_ok=True)
    print(f"街道级分辨率 {M_PER_PX} 米/像素；"
          f"1 经度 = {px_per_deg_lon():.1f} 像素\n")

    print(f"{'重叠':>6} {'经度差':>9} {'真值dx':>8} {'内点':>6} {'NCC':>8} "
          f"{'误差':>8}  结论")
    print("-" * 66)

    results = []
    # 重叠比例 → 需要的经度差
    for pct in (75, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3):
        overlap_px = 1024 * pct / 100
        dlon = overlap_px / px_per_deg_lon()
        a, b, true_dx = make_pair(dlon)
        ka, kb = _describe(a), _describe(b)
        link = _match_pair(a, b, ka, kb)
        if link is None:
            print(f"{pct:>5}% {dlon:>8.4f}° {true_dx:>7.0f} {'—':>6} {'—':>8} "
                  f"{'—':>8}  ❌ 配不上")
            results.append((pct, dlon, true_dx, None))
            continue
        tx = float(link.matrix[0, 2])
        err = abs(tx - true_dx)
        ok = link.inliers >= 15 and link.ncc >= 0.2 and err <= 4
        print(f"{pct:>5}% {dlon:>8.4f}° {true_dx:>7.0f} {link.inliers:>6} "
              f"{link.ncc:>8.4f} {err:>7.1f}px  {'✅ 可用' if ok else '⚠️ 不达标'}")
        results.append((pct, dlon, true_dx, (link.inliers, link.ncc, err, ok)))

    good = [r for r in results if r[3] and r[3][3]]
    print()
    if good:
        lo = min(good, key=lambda r: r[0])
        print(f"✅ 街道级底图最低可用重叠 = {lo[0]}%"
              f"（经度差 {lo[1]:.4f}° ≈ {lo[2]:.0f} 像素）")
        print(f"   对应真实距离：相邻两张至少重叠 "
              f"{1024 * lo[0] / 100 * M_PER_PX:.0f} 米")
    else:
        print("❌ 全部不达标")

    # 反过来：固定一张「街道级截图」，另一张是它的裁剪（模拟部分重叠）
    print("\n--- 另一种情况：两张图**有重叠但不是同一中心渲染** ---")
    print("（真实场景：用户在 App 里滑动，两张图内容部分相同）")
    base = cv2.imread("tmp_debug/zoomtest/sl_121.4600_31.1920.png")
    if base is not None:
        band = base[80:900].copy()
        bw = band.shape[1]
        for pct in (60, 40, 30, 20, 15, 10, 8, 5):
            dx = int(bw * (1 - pct / 100))
            a = band[:, 0:bw - dx].copy()
            b = band[:, dx:bw].copy()
            ka, kb = _describe(a), _describe(b)
            link = _match_pair(a, b, ka, kb)
            if link is None:
                print(f"  重叠 {pct:>2}%：❌ 配不上")
                continue
            tx = float(link.matrix[0, 2])
            err = abs(tx - (-dx))
            ok = link.inliers >= 15 and link.ncc >= 0.2 and err <= 4
            print(f"  重叠 {pct:>2}%：内点 {link.inliers:>4} NCC {link.ncc:.4f} "
                  f"误差 {err:6.1f}px  {'✅' if ok else '⚠️'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
