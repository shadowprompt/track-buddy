"""绝对精度校验工具（不依赖 POI、不依赖路名）。

原理：美团底图自己渲染了路网，道路比背景更蓝（b - r ≥ 28）。
把这个「渲染路网」当作地面真相：
  1. 提取渲染路网掩膜
  2. 取 OSM 该区域具名道路几何（WGS-84 → GCJ-02 → 墨卡托）
  3. 把 OSM 几何投影回截图，统计落在渲染道路上的比例
  4. 同时统计识别轨迹的像素点落在渲染道路上的比例

第 4 项与地理假设无关——它衡量的是像素级提取精度。
第 3 项衡量的是「像素 → 地理」映射的绝对精度。

用法：python tools/verify_roads.py [图片路径]
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import CONFIG  # noqa: E402
from app.cv_utils import imread_unicode, imwrite_unicode  # noqa: E402
from app.geo.coords import lonlat_to_mercator, wgs84_to_gcj02  # noqa: E402
from app.ocr_anchor import build_anchors, ocr_texts  # noqa: E402
from app.pipeline import RidePipeline  # noqa: E402
from app.register import register  # noqa: E402
from app.track import extract_track  # noqa: E402


def road_mask(img: np.ndarray, track_mask: np.ndarray) -> np.ndarray:
    """美团渲染的道路：比背景更蓝。"""
    im = img.astype(np.int16)
    cool = im[:, :, 0] - im[:, :, 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((cool >= 28) & (im.max(axis=2) >= 200)).astype(np.uint8)
    m[track_mask > 0] = 0
    sat = hsv[:, :, 1]
    m[sat > 50] = 0
    m[hsv[:, :, 2] < 185] = 0
    m[((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168)) & (sat > 70)] = 0
    m[(hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (sat > 90)] = 0
    m[0:190, :] = 0
    return cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def main(sample: str) -> None:
    img = imread_unicode(sample)
    tr = extract_track(img, CONFIG.vision)
    mask = road_mask(img, tr.mask)
    ys, xs = np.where(mask > 0)
    tree = cKDTree(np.column_stack([xs, ys]).astype(float))
    print("渲染道路掩膜: %d px (%.2f%%)" % (mask.sum(), mask.sum() / mask.size * 100))

    pipe = RidePipeline(CONFIG)
    items = ocr_texts(img, CONFIG.vision)
    anchors = build_anchors(img, items, CONFIG.vision)
    warn: list[str] = []
    pairs = pipe._resolve_anchors(anchors, "上海", False, warn)
    reg = register(pairs, inlier_m=CONFIG.vision.ransac_inlier_m,
                   min_inliers=CONFIG.vision.ransac_min_inliers)
    res = pipe.run(img, city="上海", allow_online=False, output_crs="WGS-84")
    rf = res.road_fit or {}
    s = reg.scale
    base_tx, base_ty = float(reg.translation[0]), float(reg.translation[1])
    cur_tx = base_tx + float(rf.get("shift_e_m", 0))
    cur_ty = base_ty + float(rf.get("shift_n_m", 0))
    print("锚点配准 scale=%.4f model=%s inliers=%d" %
          (s, reg.model, len(reg.inliers)))
    print("路网精修 E%+.0f N%+.0f | 贴路 %.0f→%.0f m" %
          (rf.get("shift_e_m", 0), rf.get("shift_n_m", 0),
           rf.get("median_before_m", 0), rf.get("median_after_m", 0)))

    # OSM 路网（WGS-84 → GCJ-02）
    from app.roadfit import fetch_roads
    gcj = res.geojson_gcj02["features"][0]["geometry"]["coordinates"]
    lons = [c[0] for c in gcj]
    lats = [c[1] for c in gcj]
    bbox = (min(lons) - 0.008, min(lats) - 0.008,
            max(lons) + 0.008, max(lats) + 0.008)
    try:
        roads = fetch_roads(bbox, timeout=90)
    except Exception as exc:
        print("路网获取失败：%s" % exc)
        roads = []
    per: dict[str, list] = {}
    for nm, pts in roads:
        arr = []
        for lo, la in pts:
            glo, gla = wgs84_to_gcj02(lo, la)
            arr.append(lonlat_to_mercator(glo, gla))
        if arr:
            per.setdefault(nm, []).extend(arr)
    if not per:
        print("无 OSM 路网可比对")
        return
    all_pts = np.vstack([np.array(v) for v in per.values()])

    def to_px(merc, tx, ty):
        return np.column_stack([(merc[:, 0] - tx) / s, -(merc[:, 1] - ty) / s])

    def hit(pts_px, radius=14.0):
        d, _ = tree.query(pts_px)
        return float(np.median(d)), float((d <= radius).mean() * 100)

    traj_px = tr.polyline.astype(float)
    m_t, p_t = hit(traj_px)
    print("\n=== 像素级：轨迹点是否落在美团渲染道路上 ===")
    print("  中位 %.1f px   命中(<14px) %.1f%%   [与地理假设无关]" % (m_t, p_t))

    print("\n=== 几何级：OSM 道路投影后落在渲染道路上的比例 ===")
    for label, tx, ty in (("仅锚点配准", base_tx, base_ty),
                          ("当前输出", cur_tx, cur_ty)):
        m_r, p_r = hit(to_px(all_pts, tx, ty))
        print("  %-12s 命中 %5.1f%%   中位 %5.1f px" % (label, p_r, m_r))

    # 平移搜索绝对最优
    best = (-1.0, 0.0, 0.0)
    for dy in range(-100, 101, 5):
        for dx in range(-100, 101, 5):
            q = to_px(all_pts, base_tx, base_ty) + np.array([dx, dy])
            _m, p = hit(q)
            if p > best[0]:
                best = (p, float(dx), float(dy))
    h2_tx = base_tx - best[1] * s
    h2_ty = base_ty + best[2] * s
    m_r, p_r = hit(to_px(all_pts, h2_tx, h2_ty))
    print("  %-12s 命中 %5.1f%%   中位 %5.1f px   (dx=%+.0f dy=%+.0f px)"
          % ("绝对最优", p_r, m_r, best[1], best[2]))
    d_cur = math.hypot(cur_tx - h2_tx, cur_ty - h2_ty)
    d_h0 = math.hypot(base_tx - h2_tx, base_ty - h2_ty)
    print("\n  当前输出距绝对最优 %.0f m   (仅锚点配准 %.0f m)" % (d_cur, d_h0))

    print("\n=== 归属路名 ===")
    print("  %s" % " → ".join(res.road_attribution.get("roads", [])))

    # 可视化
    vis = img.copy()
    vis[mask > 0] = (255, 0, 255)
    q = to_px(all_pts, cur_tx, cur_ty).astype(int)
    for i in range(len(q) - 1):
        if abs(q[i + 1][0] - q[i][0]) < 400 and abs(q[i + 1][1] - q[i][1]) < 400:
            cv2.line(vis, tuple(q[i]), tuple(q[i + 1]), (255, 180, 0), 2, cv2.LINE_AA)
    for i in range(len(tr.polyline) - 1):
        cv2.line(vis, tuple(tr.polyline[i].astype(int)),
                 tuple(tr.polyline[i + 1].astype(int)), (0, 240, 255), 6, cv2.LINE_AA)
    out = os.path.join(ROOT, "docs", "verify_roads.png")
    imwrite_unicode(out, vis)
    print("\nsaved", out)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "samples", "meituan_ride_shanghai.jpg")
    main(target)
