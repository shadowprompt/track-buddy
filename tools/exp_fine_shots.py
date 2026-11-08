"""诊断：同一条路线拆成 12 张细截图后识别失败（2026-11-08 12:00）。

失败现象：
  * 起终点徽章已定位: 起(125,1064) 终(334,73)   ← 终点疑似 UI 假徽章
  * 未找到轨迹连通域 / 轨迹提取失败：掩膜为空
  * 大量「位移近 0 但内容不同，判为错配并断开」
  * stitch.used = [5..11]，前 5 张（含起点）被排除

按文件 mtime 还原的上传顺序（= 前端文件选择顺序）。
"""
from __future__ import annotations

import itertools
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG  # noqa: E402
from app.cv_utils import bbox_distance  # noqa: E402
from app.multishot import _describe, _match_pair, stitch  # noqa: E402
from app.track import (  # noqa: E402
    build_track_mask, detect_markers, disk_kernel, extract_track,
)

U = "data/uploads"
NAMES = [
    "1794110270_3f7842e6", "1794110270_2b7a3086", "1794110270_dd87c35f",
    "1794110270_dcfee266", "1794110270_0aa85ee7", "1794110270_ace458b1",
    "1794110270_eaf0f0a3", "1794110270_b9471107", "1794110270_f9899cf0",
    "1794110270_e6d3d15f", "1794110270_320845a4", "1794110270_0d980f41",
]
CANVAS = "data/uploads/1794110270_3f7842e6_stitched.png"


def list_markers(hsv: np.ndarray, ranges, lo: int, hi: int, label: str) -> None:
    """列出所有面积合格的色块候选（不只是 find_marker 选中的那个）。"""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, r[:3], r[3:])
    cnt, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    for i in range(1, cnt):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if not (lo <= area <= hi):
            continue
        cands.append((
            area, (x, y, w, h), (int(cent[i][0]), int(cent[i][1])),
            round(w / max(h, 1), 2), round(area / max(w * h, 1), 2),
        ))
    cands.sort(reverse=True)
    print(f"  {label} 候选 {len(cands)} 个（面积, 框, 中心, 宽高比, 填充率）:")
    for c in cands[:12]:
        print("   ", c)


def comp_stats(mask: np.ndarray, label: str, s, e) -> None:
    nz = int((mask > 0).sum())
    cnt, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, cnt):
        x, y, w, h, area = [int(v) for v in stats[i]]
        comps.append((area, (x, y, w, h)))
    comps.sort(reverse=True)
    print(f"  {label}: 非零像素={nz} 连通域={len(comps)}")
    for area, bb in comps[:6]:
        print(f"    area={area} box={bb} d起={round(bbox_distance(bb, s), 0) if s else '-'} "
              f"d终={round(bbox_distance(bb, e), 0) if e else '-'}")


def main() -> None:
    imgs = [cv2.imread(os.path.join(U, n + ".jpg")) for n in NAMES]
    if any(im is None for im in imgs):
        print("样本缺失")
        return

    print("=== A. 逐对配准（只看相邻对 / 位移近 0 的对）===")
    feats = [_describe(im) for im in imgs]
    near_zero = []
    strong = []
    for i, j in itertools.combinations(range(12), 2):
        link = _match_pair(imgs[i], imgs[j], feats[i], feats[j])
        if link is None:
            continue
        tx, ty = float(link.matrix[0, 2]), float(link.matrix[1, 2])
        rec = (i + 1, j + 1, link.inliers, round(link.ncc, 3), round(tx, 1), round(ty, 1))
        if abs(tx) < 6 and abs(ty) < 6:
            near_zero.append(rec)
        if link.ncc >= 0.50:
            strong.append(rec)
        if j == i + 1:
            print(f"  相邻 图{i + 1}↔图{j + 1}: {rec}")
    print(f"  -- 位移近 0 的对 {len(near_zero)} 个（前 8）--")
    for r in near_zero[:8]:
        print("   ", r)
    print(f"  -- ncc>=0.50 的对 {len(strong)} 个 --")
    for r in strong:
        print("   ", r)

    print("\n=== B. 当前 stitch 结果 ===")
    res = stitch(imgs)
    print("  ok", res.ok, "used", res.used, "unused", res.unused)
    if res.canvas is not None:
        print("  canvas", res.canvas.shape[1], res.canvas.shape[0])
    for n in res.notes[-6:]:
        print("   ·", n)

    print("\n=== C. 已存画布的徽章与掩膜 ===")
    img = res.canvas if res.canvas is not None else cv2.imread(CANVAS)
    if img is None:
        print("  画布缺失")
        return
    print("  canvas shape", img.shape)
    small = cv2.resize(img, (1226, int(1226 * img.shape[0] / img.shape[1])))
    os.makedirs("tmp_debug", exist_ok=True)
    cv2.imwrite("tmp_debug/fine_canvas_small.png", small)

    cfg = CONFIG.vision
    s, e, _ = detect_markers(img, cfg)
    print("  detect_markers:", s, e)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    list_markers(hsv, [cfg.start_hsv_range], cfg.badge_min_area, cfg.badge_max_area, "绿色「起」")
    list_markers(hsv, [cfg.end_hsv_range, cfg.end_hsv_range2],
                 cfg.badge_min_area, cfg.badge_max_area, "红色「终」")

    mask = build_track_mask(img, cfg)
    comp_stats(mask, "严格掩膜", s, e)
    v, sat = hsv[:, :, 2], hsv[:, :, 1]
    relaxed = ((v < 215) & (sat < 120)).astype(np.uint8)
    relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, disk_kernel(13))
    relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, disk_kernel(5))
    comp_stats(relaxed, "放宽掩膜", s, e)

    print("\n  -- 深蓝色粗线候选（B 显著高于 R/G）--")
    b, g, r = cv2.split(img)
    blue = ((b.astype(np.int16) > r.astype(np.int16) + 25) &
            (b.astype(np.int16) > g.astype(np.int16) + 10) & (b > 100)).astype(np.uint8)
    comp_stats(blue, "蓝色像素", s, e)

    print("\n  -- 完整 extract_track --")
    tr = extract_track(img, cfg)
    print("  轨迹点数", len(tr.polyline), "置信度", round(tr.confidence, 3),
          "起", tr.start_marker, "终", tr.end_marker, "bbox", tr.bbox)
    for n in tr.notes:
        print("   ·", n)


if __name__ == "__main__":
    main()
