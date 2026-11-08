"""轨迹提取的形状合理性检查（`app/track.py`）的测试。

## 为什么要有这个检查

用户实测多图拼接时看到「中间一大块绿色三角形」，以为拼接错了。
实际是**腾讯淡色底图里的绿地（公园）被当成了骑行轨迹**。

实测数据（用户上传的 4 张截图）：
* 绿连通域：色相 68、饱和度高、面积 1692 像素
* 轨迹掩膜的取值区间是「低饱和 + 中等亮度」——绿地正好落在里面
* 结果：骨架化后变成一个实心三角形，还被当成轨迹画到了地图上

## 判据

骑行轨迹线的本质特征：
1. **不占满整幅图的宽度**——它是屏幕上的一条线；
   误判样本的连通域宽 1150~1260 像素，而图宽就是 1260。
2. **细长而非团块**——长宽比大、骨架化后填充率低。

真实样本（`samples/`）必须零误伤——这是硬约束。
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG  # noqa: E402
from app.track import extract_track  # noqa: E402

UPLOADS = "data/uploads"
# 用户 2026-11-06 上传的 4 张腾讯淡色底图截图（没有骑行轨迹线）
USER_SHOTS = [
    "1793930041_1d6027d4",
    "1793930041_152f92f5",
    "1793930041_cedd9acf",
    "1793930041_70036afc",
]


def _exists(name: str) -> bool:
    return os.path.exists(os.path.join(UPLOADS, f"{name}.jpg"))


def test_real_samples_not_broken() -> None:
    """真实轨迹样本必须正常提取（不能被形状检查误伤）。"""
    samples = sorted(glob.glob("samples/*.jpg"))
    if not samples:
        print("  跳过：没有样本")
        return
    for f in samples:
        im = cv2.imread(f)
        if im is None:
            continue
        tr = extract_track(im, CONFIG.vision)
        hurt = [n for n in tr.notes if "形状不合理" in n]
        assert not hurt, f"{os.path.basename(f)} 被形状检查误伤：{hurt[0][:80]}"
        assert len(tr.polyline) > 50, (
            f"{os.path.basename(f)} 轨迹点数异常少：{len(tr.polyline)}"
        )
        print(f"  {os.path.basename(f)}: {len(tr.polyline)} 点，正常")


def test_user_tencent_shots_rejected() -> None:
    """腾讯淡色底图（无轨迹线）的截图必须被拦住，不产出假轨迹。

    ⚠️ 拦住的方式可以不同：
    * 有的图被「形状不合理」拦住（误判出整幅图宽的色块）；
    * 有的图是「掩膜为空」（压根没提取到任何东西）。
    两种都算正确——只要**不产出轨迹**即可。
    """
    avail = [n for n in USER_SHOTS if _exists(n)]
    if not avail:
        print("  跳过：没有用户样本")
        return
    for name in avail:
        im = cv2.imread(os.path.join(UPLOADS, f"{name}.jpg"))
        tr = extract_track(im, CONFIG.vision)
        assert len(tr.polyline) == 0, (
            f"{name} 仍产出了 {len(tr.polyline)} 个点（应是底图色块误判）"
        )
        assert tr.confidence == 0.0, f"{name} 置信度应为 0，实际 {tr.confidence}"
        why = next(
            (n for n in tr.notes if "形状不合理" in n or "提取失败" in n),
            None,
        )
        assert why, f"{name} 没有说明为什么没提取到轨迹：{tr.notes}"
        print(f"  {name}: 已拦住 —— {why[:52]}…")


def test_blob_shape_rejected() -> None:
    """合成一个「团块」色块，必须被拦住（不产出轨迹）。"""
    im = np.full((600, 800, 3), 245, np.uint8)
    # 画一个低饱和、中等亮度的方块（模拟绿地被掩膜捕获）
    cv2.rectangle(im, (200, 150), (600, 450), (150, 190, 170), -1)
    tr = extract_track(im, CONFIG.vision)
    assert len(tr.polyline) == 0, f"团块被当成了轨迹：{len(tr.polyline)} 点"
    assert tr.confidence == 0.0, f"置信度应为 0，实际 {tr.confidence}"
    print("  团块色块已被拦住")


def main() -> int:
    tests = [
        test_real_samples_not_broken,
        test_user_tencent_shots_rejected,
        test_blob_shape_rejected,
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
