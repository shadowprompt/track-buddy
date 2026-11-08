"""诊断：用户倒序上传同一批 4 张截图时，去重误丢含起点的图。

复现数据：data/uploads/1794108885_*.jpg 与 1794096457_*.jpg 的 sha1 完全一致，
只是上传顺序变成 3-4/3-3/3-2/3-1（正序的第 4、3、2、1 张）。
"""
from __future__ import annotations

import itertools
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG  # noqa: E402
from app.multishot import (  # noqa: E402
    _describe, _gray, _is_near_duplicate, _match_pair, stitch,
)
from app.track import detect_markers, extract_track  # noqa: E402

NAMES = [
    "1794096457_2463f8d7",  # 含起点
    "1794096457_5abde3aa",
    "1794096457_63ce03bd",
    "1794096457_d2051989",  # 含终点
]
LABELS = ["起点图1", "图2", "图3", "终点图4"]
U = "data/uploads"


def gray_ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = _gray(a).astype(np.float32).ravel()
    y = _gray(b).astype(np.float32).ravel()
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return float((x * y).mean())


def main() -> None:
    imgs = [cv2.imread(os.path.join(U, n + ".jpg")) for n in NAMES]
    if any(im is None for im in imgs):
        print("样本缺失")
        return

    print("=== 单张徽章 ===")
    for lab, im in zip(LABELS, imgs):
        s, e, _ = detect_markers(im, CONFIG.vision)
        print(f"  {lab}: {im.shape[1]}x{im.shape[0]}  起={s} 终={e}")

    print("\n=== 两两差异（关键：图1 与 图4 是不是同一视角）===")
    feats = [_describe(im) for im in imgs]
    for i, j in itertools.combinations(range(4), 2):
        a, b = imgs[i], imgs[j]
        diff = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
        ncc = gray_ncc(a, b)
        link = _match_pair(a, b, feats[i], feats[j])
        tx = ty = None
        if link is not None:
            tx, ty = float(link.matrix[0, 2]), float(link.matrix[1, 2])
        print(
            f"  图{i + 1}↔图{j + 1}: 平均差={diff:.2f} 灰度NCC={ncc:.4f} "
            f"near_dup={_is_near_duplicate(a, b)} "
            f"SIFT={'None' if link is None else f'(inl={link.inliers}, ncc={link.ncc:.2f}, t=({tx:.1f},{ty:.1f}))'}"
        )

    print("\n=== 不同上传顺序的拼接结果 ===")
    for name, order in (("正序 1-2-3-4", [0, 1, 2, 3]), ("倒序 4-3-2-1", [3, 2, 1, 0])):
        res = stitch([imgs[k] for k in order])
        print(f"  {name}: ok={res.ok} used={res.used} canvas="
              f"{None if res.canvas is None else (res.canvas.shape[1], res.canvas.shape[0])}")
        for n in res.notes:
            print(f"    · {n}")
        if res.ok and res.canvas is not None:
            tr = extract_track(res.canvas, CONFIG.vision)
            print(f"    → 起={tr.start_marker} 终={tr.end_marker} "
                  f"轨迹={len(tr.polyline)}点 conf={tr.confidence:.3f}")


if __name__ == "__main__":
    main()
