"""画像视觉参数标定 CLI：对真实截图统计 HSV 分布，输出 VisionConfig 建议。

用法（仓库根执行）::

    # 对一组截图做全景统计：高饱和像素的 H/S/V 分布 + 蓝色系细分
    python src/tools/calibrate_profile.py samples/amap_1.jpg samples/amap_2.jpg

    # 深挖某个色相带（OpenCV H 域 0-179），只统计 H∈[95,120] 的像素
    python src/tools/calibrate_profile.py samples/amap_1.jpg --hue 95 120

    # 徽章排查：列出绿色/红色连通域候选（面积排序，前 10 个）
    python src/tools/calibrate_profile.py samples/amap_1.jpg --badges

    # 与某个画像的当前参数对比：掩膜各档位能圈进多少像素
    python src/tools/calibrate_profile.py samples/amap_1.jpg --profile amap

给谁用：``app/source_profile.py`` 里新增 / 精标画像的人。流程——

1. 先跑全景统计，看轨迹线落在哪个色相带（高德预期 H≈95-120）；
2. 用 ``--hue`` 收窄，读出 S / V 的分位数（p5/p50/p95），据此填
   ``track_hsv_range``（下限取 p5 再留余量，上限一般拉满）；
3. 用 ``--badges`` 找起/终徽章的色相与面积，填 ``start_hsv_range`` /
   ``end_hsv_range(2)`` / ``badge_min_area``；
4. 改完画像后用 ``--profile`` 复核覆盖率，最后以真实识别的
   ``overlay_result.png`` / ``road_verify.png`` 人工核对收尾。

**统计只读截图，不调用在线服务**；本脚本不改任何配置文件。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.config import VisionConfig  # noqa: E402
from app.cv_utils import imread_unicode  # noqa: E402
from app.source_profile import PROFILES  # noqa: E402
from app.track import build_track_mask  # noqa: E402


def _pct(values: np.ndarray, ps=(5, 25, 50, 75, 95)) -> dict[int, float]:
    if values.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(values, p)) for p in ps}


def _fmt_pct(tag: str, pct: dict[int, float]) -> str:
    return (f"  {tag:<6} p5={pct[5]:6.1f}  p25={pct[25]:6.1f}  "
            f"p50={pct[50]:6.1f}  p75={pct[75]:6.1f}  p95={pct[95]:6.1f}")


def analyze_image(path: Path, hue_band: tuple[int, int] | None) -> None:
    img = imread_unicode(str(path))
    if img is None:
        print(f"[跳过] 读不到图片: {path}")
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = h.size
    print(f"\n=== {path.name}  ({img.shape[1]}x{img.shape[0]}) ===")

    # ① 高饱和像素（S≥120）的色相直方图：找轨迹线的色相带
    sat = s >= 120
    n_sat = int(sat.sum())
    print(f"高饱和像素 (S>=120): {n_sat} ({n_sat / total:.1%})")
    if n_sat:
        hs = h[sat]
        hist = np.bincount(hs, minlength=180)
        top = np.argsort(hist)[::-1][:8]
        bands = [(int(x), int(hist[x])) for x in top if hist[x] > 0]
        print("  色相 Top8 (H: 像素数): " + ", ".join(f"{a}:{b}" for a, b in bands))

    # ② 色相带内的 S/V 分布：给 track_hsv_range 填数用
    if hue_band is not None and n_sat:
        lo, hi = hue_band
        inband = sat & (h >= lo) & (h <= hi)
        n_band = int(inband.sum())
        print(f"色相带 H∈[{lo},{hi}]: {n_band} 像素 ({n_band / total:.2%})")
        if n_band:
            print(_fmt_pct("S", _pct(s[inband])))
            print(_fmt_pct("V", _pct(v[inband])))
            print(f"  → 建议 track_hsv_range=({lo}, <S下限>, <V下限>, "
                  f"{hi}, 255, 255)，S/V 下限取 p5 再留 10-20 余量")

    # ③ 全图灰阶概况（对照上限式掩膜的适用性）
    grayish = (s < 90) & (v > 150)
    print(f"低饱和高亮像素 (S<90,V>150, 美团式): {int(grayish.sum())} "
          f"({grayish.sum() / total:.1%}) —— 若轨迹不在其中，上限式掩膜不适用")


def analyze_badges(path: Path) -> None:
    """列出绿/红连通域候选（面积排序），辅助定徽章 HSV 与面积阈值。"""
    img = imread_unicode(str(path))
    if img is None:
        print(f"[跳过] 读不到图片: {path}")
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    named = [("绿", (35, 90, 90, 85, 255, 255)), ("红1", (0, 90, 90, 12, 255, 255)),
             ("红2", (170, 90, 90, 179, 255, 255))]
    for name, rng in named:
        m = cv2.inRange(hsv, np.array(rng[:3], np.uint8), np.array(rng[3:], np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cands = sorted(
            ((int(stats[i, cv2.CC_STAT_AREA]),
              (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])),
              (int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])))
             for i in range(1, n)), reverse=True)[:10]
        if not cands:
            continue
        print(f"\n{name} 候选 (面积, 左上角, 宽高) 前10 —— 徽章通常 4000+ 且近方形:")
        for area, pos, wh in cands:
            print(f"  area={area:6d}  pos={pos}  wh={wh}")


def compare_profile(path: Path, profile_id: str) -> None:
    """用注册画像的当前参数跑掩膜，报告命中率，验证参数方向。"""
    prof = PROFILES[profile_id]
    img = imread_unicode(str(path))
    if img is None:
        print(f"[跳过] 读不到图片: {path}")
        return
    mask = build_track_mask(img, prof.vision)
    total = mask.size
    # 掩膜像素值可能是 0/1（上限式）或 0/255（inRange 区间式），统一归一
    hit = int(np.count_nonzero(mask))
    print(f"\n=== {path.name} × 画像 {profile_id} ===")
    print(f"掩膜命中 {hit} 像素 ({hit / total:.2%})；轨迹线一般占 0.5%-8%")
    print("  过低(≤0.1%)→阈值太紧漏轨迹；过高(≥15%)→太松混入底图，"
          "用 --hue/--badges 收窄后回改画像")


def main() -> None:
    ap = argparse.ArgumentParser(description="画像视觉参数标定（只读统计）")
    ap.add_argument("images", nargs="+", help="截图路径（支持中文路径）")
    ap.add_argument("--hue", nargs=2, type=int, metavar=("LO", "HI"),
                    help="额外统计该 OpenCV H 区间内高饱和像素的 S/V 分布")
    ap.add_argument("--badges", action="store_true", help="列出绿/红连通域候选")
    ap.add_argument("--profile", help="用注册画像的掩膜参数做覆盖率复核")
    args = ap.parse_args()

    band = (args.hue[0], args.hue[1]) if args.hue else None
    for raw in args.images:
        path = Path(raw)
        analyze_image(path, band)
        if args.badges:
            analyze_badges(path)
        if args.profile:
            if args.profile not in PROFILES:
                print(f"[错误] 未知画像 {args.profile!r}；可用："
                      f"{', '.join(sorted(PROFILES))}")
                sys.exit(2)
            compare_profile(path, args.profile)


if __name__ == "__main__":
    main()
