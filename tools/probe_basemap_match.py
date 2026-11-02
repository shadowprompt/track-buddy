"""底图匹配可行性探针（可复现实验）。

结论（2026-11 实测，见 docs/HANDOVER.md 第 21 章）：
  * **底图确实同源**：高德瓦片的「上海动物园」位置与美团截图精确对齐。
  * **街道级可行**：骨架 Chamfer 距离 3.99 米 vs 随机基线 79.2 米（20 倍）。
  * **城市级不可行**：最优解只比原位好 1.22 倍，且前 12 个候选的分数差
    只有 0.1%（3.52 vs 3.52 米），但地理位置分散在 900-1600 米范围内——
    信号不足以定出唯一位置。

所以：低缩放下**没有任何单一信息源**能给出精确绝对位置。
底图匹配在街道级是主力手段，在城市级只能当「合理性校验」。

用法：
    python tools/probe_basemap_match.py [图片路径]
"""
from __future__ import annotations

import io
import math
import sys
import time

import cv2
import numpy as np
import requests
from PIL import Image

sys.path.insert(0, ".")
from app.config import CONFIG
from app.cv_utils import imread_unicode
from app.ocr_anchor import build_anchors, ocr_texts
from app.pipeline import RidePipeline
from app.register import register
from app.track import extract_track

R = 6378137.0
ORIGIN = 20037508.34
TILE = 256

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def tile(tx: int, ty: int, z: int) -> np.ndarray:
    u = (f"https://webrd0{1 + (tx + ty) % 4}.is.autonavi.com/appmaptile"
         f"?lang=zh_cn&size=1&scale=1&style=8&x={tx}&y={ty}&z={z}")
    for _ in range(4):
        try:
            r = session.get(u, timeout=20)
            if r.status_code == 200 and len(r.content) > 1000:
                return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception:
            pass
        time.sleep(0.6)
    return np.full((TILE, TILE, 3), 245, dtype=np.uint8)


def roads_of(bgr: np.ndarray) -> np.ndarray:
    """提取底图里的「道路/建筑轮廓」结构（二值图），与配色无关。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    if th.max() > 0:
        th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.GaussianBlur(th, (3, 3), 0)
    e = cv2.Canny(th, 30, 90)
    return (e > 0).astype(np.uint8)


def main() -> None:
    path = (sys.argv[1] if len(sys.argv) > 1
            else "data/uploads/1793589815_20da8357.jpg")
    shot = imread_unicode(path)
    h, w = shot.shape[:2]
    print(f"城市级截图 {w}x{h}")

    cfg = CONFIG.vision
    pipe = RidePipeline(CONFIG)
    tr = extract_track(shot, cfg)
    texts = ocr_texts(shot, cfg)
    anchors = build_anchors(shot, texts, cfg)
    pairs = pipe._resolve_anchors(anchors, "上海", True, [])
    reg = register(pairs, inlier_m=cfg.ransac_inlier_m,
                   min_inliers=cfg.ransac_min_inliers)
    scale = float(reg.scale)
    print(f"配准尺度 {scale:.3f} 墨卡托米/px，内点 {len(reg.inliers)}")
    mat = np.asarray(reg.matrix, dtype=float)
    trn = np.asarray(reg.translation, dtype=float)

    def px2merc(px, py):
        v = mat @ np.array([px, py], dtype=float) + trn
        return float(v[0]), float(v[1])

    # 取一块「干净底图」区域（远离轨迹线与起终点徽章）
    bx, by, bw, bh = tr.bbox
    cx = int(bx + bw / 2)
    cy = int(min(h - 260, by + bh * 0.15))
    half = 220
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    crop = shot[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    print(f"采样区域 ({x0},{y0})-({x1},{y1}) {cw}x{ch}")

    # 瓦片层级：截图尺度 ≈ 15.7 墨卡托米/px，z=13 是 16.35 → 1:1
    z = 13
    n = 2 ** z

    def merc2abs(mx, my):
        return ((mx + ORIGIN) / (2 * ORIGIN) * n * TILE,
                (ORIGIN - my) / (2 * ORIGIN) * n * TILE)

    # 搜索半径：±1500 米
    search_m = 1500.0
    search_px = search_m / scale
    mxa, mya = px2merc(x0, y0)
    mxb, myb = px2merc(x1, y1)
    # 加上搜索余量
    dxm = search_m
    dym = search_m
    ax0, ay0 = merc2abs(mxa - dxm, mya + dym)
    ax1, ay1 = merc2abs(mxb + dxm, myb - dym)
    tx0 = int(ax0 // TILE)
    ty0 = int(ay0 // TILE)
    tx1 = int(ax1 // TILE)
    ty1 = int(ay1 // TILE)
    ntx, nty = tx1 - tx0 + 1, ty1 - ty0 + 1
    print(f"瓦片 {ntx}x{nty} = {ntx*nty} 片 (z={z})，搜索 ±{search_m:.0f} 米")

    mos = np.zeros((nty * TILE, ntx * TILE, 3), dtype=np.uint8)
    for i in range(nty):
        for j in range(ntx):
            mos[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE] = tile(
                tx0 + j, ty0 + i, z)
        time.sleep(0.15)

    # 瓦片拼图 → 截图像素 仿射（GCJ-02 墨卡托，高德瓦片本身就是 GCJ 渲染）
    src, dst = [], []
    for px, py in [(x0, y0), (x1, y0), (x0, y1)]:
        mx, my = px2merc(px, py)
        qx, qy = merc2abs(mx, my)
        src.append([qx - tx0 * TILE, qy - ty0 * TILE])
        dst.append([px, py])
    A = np.array([[q[0], q[1], 1.0] for q in src])
    B = np.array(dst)
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    M = np.ascontiguousarray(coef.T)
    aligned_full = cv2.warpAffine(mos, M, (w, h), flags=cv2.INTER_AREA)

    sk_a = roads_of(crop)
    sk_b = roads_of(aligned_full)
    print(f"结构像素：美团 {sk_a.sum()}，高德 {sk_b.sum()}")
    da = cv2.distanceTransform(1 - sk_b, cv2.DIST_L2, 3)
    ys, xs = np.nonzero(sk_a)
    print(f"美团结构点 {len(ys)} 个")

    step = 16
    span = int(search_px)
    results = []
    for dy in range(-span, span + 1, step):
        for dx in range(-span, span + 1, step):
            tx_, ty_ = x0 + dx, y0 + dy
            if tx_ < 0 or ty_ < 0 or tx_ + cw > w or ty_ + ch > h:
                continue
            sub = da[ty_:ty_ + ch, tx_:tx_ + cw]
            if sub.size == 0:
                continue
            v = float(sub[ys, xs].mean())
            results.append((v, dx, dy))
    results.sort()
    print(f"搜索 {len(results)} 个位置")
    print("最优 6 个（Chamfer 米，越小越好）：")
    for v, dx, dy in results[:6]:
        print(f"   dx={dx:+5d}px dy={dy:+5d}px  位移={np.hypot(dx,dy)*scale:7.0f} 米"
              f"  Chamfer={v:6.2f} 米")
    # 基线：配准位置本身
    at_reg = None
    for v, dx, dy in results:
        if dx == 0 and dy == 0:
            at_reg = v
            break
    if at_reg is None:
        # 找最接近原点的
        near = min(results, key=lambda r: abs(r[1]) + abs(r[2]))
        at_reg = near[0]
        print(f"（原点不在搜索网格上，用最近点 Chamfer={at_reg:.2f} 米）")
    else:
        print(f"配准原位 Chamfer = {at_reg:.2f} 米")
    print(f"最优 {results[0][0]:.2f} 米 vs 原位 {at_reg:.2f} 米 → "
          f"改善 {at_reg/max(results[0][0],1e-6):.2f} 倍")

    # 最优位置 vs 随机位置
    vals = np.array([r[0] for r in results])
    print(f"位置分布：最小 {vals.min():.2f}  中位 {np.median(vals):.2f}  "
          f"最大 {vals.max():.2f} 米")

    v, dx, dy = results[0]
    best = aligned_full[y0 + dy:y0 + dy + ch, x0 + dx:x0 + dx + cw]
    vis = np.zeros((ch, cw * 2 + 8, 3), dtype=np.uint8)
    vis[:, :cw] = crop
    vis[:, cw + 8:] = best
    cv2.imwrite("tmp_debug/city_skel_best.png", vis)
    print("saved tmp_debug/city_skel_best.png")

    # 关键检验：最优解是否唯一？把前若干个候选都算出来，
    # 看它们之间的地理距离——若多个候选相隔公里级且分数几乎相同，
    # 说明这个信号**无法定出唯一位置**。
    print("\n=== 候选唯一性检验 ===")
    top = results[:12]
    print("前 12 候选的地理位置（相对配准原点的偏移）：")
    for i, (vv, ddx, ddy) in enumerate(top):
        mx, my = px2merc(x0 + cw / 2 + ddx, y0 + ch / 2 + ddy)
        lo, la = __import__("app.geo.coords", fromlist=["x"]).mercator_to_lonlat(mx, my)
        print(f"  #{i+1}: Chamfer={vv:.2f}m  偏移 {ddx*scale:+7.0f}m 东 "
              f"{ddy*scale:+7.0f}m 南  → ({lo:.5f}, {la:.5f})")

    # 最优 vs 第二优的分数差
    print(f"\n最优与次优的 Chamfer 差：{top[0][0]-top[1][0]:.3f} 米 "
          f"（相对 {top[0][0]:.2f} 的 {abs(top[0][0]-top[1][0])/max(top[0][0],1e-9)*100:.1f}%）")
    print(f"最优与中位的差：{np.median(vals)-vals.min():.2f} 米")


if __name__ == "__main__":
    main()
