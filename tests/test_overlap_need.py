"""重叠需求的实测（`tools/exp_overlap_need.py` 的结论固化）。

## 结论

**街道级截图的拼接对重叠的要求很低**：

| 重叠 | 内点数 | NCC | 位移误差 |
|---|---|---|---|
| 25% | 278 | 1.0000 | 0.0px |
| 15% | 159 | 1.0000 | 0.0px |
| 10% | 89 | 1.0000 | 0.0px |
| 5% | 36 | 1.0000 | 0.0px |
| 3% | 16 | 1.0000 | 0.0px |

即：相邻两张**只要有 3% 重叠（约 24 像素）就能精确配准**。

## 但有一个前提：必须用「独立渲染」的两张图

### 踩坑一：用「同一张图切片」当两张截图 —— 错

切片出来的两张图重叠区**像素级完全相同**，
SIFT 反而分不清「同一位置」和「相似纹理」：
匹配对只有 225 个（独立渲染时 1000+），位移分布弥散（标准差 153px），
完全配不出真值。

⇒ 真实场景里用户在 App 里滑动会**重新渲染**（有轻微差异），
必须模拟这一点，不能用纯切片。

### 踩坑二：用真实美团城市级截图 —— 也不适合

城市级底图只有行政区/地铁线/大 POI，**结构高度重复**。
实测任意两个视角的匹配位移中位数都是 0、离散度 200-500px —— 全是误匹配。
（与 §21.4「城市级底图匹配不可行」一致。）

### 踩坑三：切片时算错真实重叠宽度 —— 错

`a = band[:, 0:bw-dx]`、`b = band[:, dx:bw]` 的**真实重叠宽度是 `bw - 2*dx`**。
按这个公式，标称 30% 那一档实际是**负重叠**——所以那一组数据全是「配不上」，
差点得出「需要 60% 重叠」的错误结论。

正确做法：两片都取宽度 W，第 2 片右移 dx ⇒ 真实重叠 = W - dx。

## 最重要的发现：光靠 RANSAC 会选错解

城市街道网格规则，底图里有大量**周期性重复结构**。
实测（两图重叠 240px）：

* 真解 dx=-560 → 393 个匹配
* 假解 dx=+449 → **509** 个匹配（更多！）

RANSAC 选「内点最多」的，于是选了假解，位移误差 1024px。

而且**光靠 NCC 也救不了**：用假解算出的 NCC 是 0.9987——
因为 NCC 是拿「RANSAC 选出的变换」去比，数学上自洽、语义上却错。

正确做法（`app/multishot.py:_pick_best_shift`）：
**从位移直方图里找出几个候选峰，逐个算 NCC，选最高的**。
实测真解 NCC=1.0000、假解 NCC=0.1866 —— 区分度极高。
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from app.multishot import _describe, _match_pair, _ncc  # noqa: E402

# 高德静态图 zoom=17、1024px 宽 → 实测 0.5108 米/像素
M_PER_PX = 0.5108

TMP = "tmp_debug/zoomtest"


def _need(*names: str) -> bool:
    return all(os.path.exists(os.path.join(TMP, n)) for n in names)


def _build_big() -> np.ndarray | None:
    """拼出一张足够宽的连续街道级底图。

    A 中心 121.4600，C 中心 121.4464 ⇒ 中心差 2535 像素，
    把 C 的右侧贴到 A 的左边，得到 2048 宽的连续图。
    """
    a = os.path.join(TMP, "w_121.4600_31.1920.png")
    c = os.path.join(TMP, "w_121.4464_31.1920.png")
    if not (_need("w_121.4600_31.1920.png", "w_121.4464_31.1920.png")):
        return None
    A = cv2.imread(a)
    C = cv2.imread(c)
    if A is None or C is None:
        return None
    big = np.hstack([A, C[:, 1024 - 2535:1024]])
    return big[80:900]


def test_low_overlap_still_precise() -> None:
    """街道级下低到 3% 重叠仍应精确配准（误差 <= 4px）。"""
    big = _build_big()
    if big is None:
        print("  跳过：缺少街道级底图样本")
        return
    WIN = 800
    print(f"  {'重叠':>6} {'内点':>6} {'NCC':>8} {'误差':>8}")
    for pct in (25, 15, 10, 5, 3):
        dx = int(WIN * (1 - pct / 100))
        if dx + WIN > big.shape[1]:
            continue
        a = big[:, 0:WIN].copy()
        b = big[:, dx:dx + WIN].copy()
        ka, kb = _describe(a), _describe(b)
        link = _match_pair(a, b, ka, kb)
        assert link is not None, f"重叠 {pct}% 配不上"
        err = abs(float(link.matrix[0, 2]) - (-dx))
        print(f"  {pct:>5}% {link.inliers:>6} {link.ncc:>8.4f} {err:>7.1f}px")
        assert err <= 4, f"重叠 {pct}% 位移误差 {err:.1f}px 过大"
        assert link.ncc >= 0.9, f"重叠 {pct}% NCC 仅 {link.ncc:.4f}"
    print("  ✓ 低重叠也能精确配准")


def test_repeated_texture_picks_right_peak() -> None:
    """底图有周期性重复结构时，必须选对位移峰。

    这是本次最容易踩的坑：真解与假解的内点数是 393 vs 509，
    光靠 RANSAC 会选错（误差 1024px）。
    """
    big = _build_big()
    if big is None:
        print("  跳过：缺少街道级底图样本")
        return
    WIN = 800
    dx = 560          # 重叠 240px —— 实测会选到假解的那一档
    a = big[:, 0:WIN].copy()
    b = big[:, dx:dx + WIN].copy()
    ka, kb = _describe(a), _describe(b)
    link = _match_pair(a, b, ka, kb)
    assert link is not None, "配不上"
    err = abs(float(link.matrix[0, 2]) - (-dx))
    print(f"  重叠 {WIN - dx}px：内点 {link.inliers}，NCC {link.ncc:.4f}，"
          f"误差 {err:.1f}px")
    assert err <= 4, f"选错了位移峰（误差 {err:.1f}px）——RANSAC 被重复纹理骗了"
    assert link.ncc >= 0.9, f"NCC 仅 {link.ncc:.4f}"
    print("  ✓ 选对了位移峰")


def test_ncc_distinguishes_true_from_false() -> None:
    """NCC 必须能区分「真重叠」与「碰巧长得像」。"""
    big = _build_big()
    if big is None:
        print("  跳过：缺少街道级底图样本")
        return
    WIN = 800
    dx = 560
    a = big[:, 0:WIN].copy()
    b = big[:, dx:dx + WIN].copy()
    true_m = np.array([[1, 0, -dx], [0, 1, 0], [0, 0, 1]], float)
    false_m = np.array([[1, 0, 449.0], [0, 1, 0], [0, 0, 1]], float)
    n_true = _ncc(a, b, true_m)
    n_false = _ncc(a, b, false_m)
    print(f"  真解 NCC={n_true:.4f}  假解 NCC={n_false:.4f}")
    assert n_true > 0.9, f"真解 NCC 仅 {n_true:.4f}"
    assert n_true - n_false > 0.5, "NCC 区分度不足"
    print("  ✓ NCC 区分度足够")


def main() -> int:
    tests = [
        test_low_overlap_still_precise,
        test_repeated_texture_picks_right_peak,
        test_ncc_distinguishes_true_from_false,
    ]
    failed = 0
    for t in tests:
        print(f"\n== {t.__name__} ==")
        try:
            t()
            print("  PASS")
        except AssertionError as exc:
            print(f"  FAIL: {exc}")
            failed += 1
    print(f"\n{'全部通过' if not failed else f'{failed} 项失败'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
