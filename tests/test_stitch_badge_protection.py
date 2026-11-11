"""拼接画布必须保住「起/终」徽章。

真实事故（2026-11-11 21:19，7 张细节图）：
含绿色「起」徽章的那张图在贴图时排在重叠区后面，4270 个绿色像素
被一张「同区域但没有徽章」的截图 100% 覆盖，画布上最大绿色连通域
只剩 5 px。`extract_track()` 因此找不到拓扑锚，整次识别直接失败
（"未检测到起点徽章" → "未能提取到有效轨迹线"）。

这里用自包含的合成图复现「含徽章的图 depth 更大、必然被先贴的图盖住」
的最坏情形：修复前起点徽章会丢，修复后必须原位保住。
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from app.config import VisionConfig  # noqa: E402
from app.cv_utils import find_marker  # noqa: E402
from app.multishot import stitch  # noqa: E402


def _textured_base(h: int = 560, w: int = 1100, seed: int = 7) -> np.ndarray:
    """给 SIFT 足够多可配准特征的浅色底图（不依赖网络/外部样本）。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 238, np.uint8)
    for y in range(0, h, 37):
        cv2.line(img, (0, y), (w, y), (205, 205, 205), 1)
    for x in range(0, w, 41):
        cv2.line(img, (x, 0), (x, h), (210, 210, 210), 1)
    for _ in range(220):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = tuple(int(v) for v in rng.integers(70, 190, 3))
        cv2.circle(img, (x, y), int(rng.integers(2, 8)), color, -1)
    for _ in range(45):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 40))
        cv2.rectangle(
            img, (x, y),
            (x + int(rng.integers(15, 60)), y + int(rng.integers(10, 35))),
            (170, 170, 170), 2,
        )
    return img


def _shots() -> tuple[np.ndarray, np.ndarray]:
    base = _textured_base()
    # A 覆盖底图 x=0..700，B 覆盖 x=300..1000（重叠 400px）。
    # 贴图顺序按 (depth, index)：A 是 root（depth 0）先贴，B（depth 1）后贴。
    A = base[:, 0:700].copy()
    B = base[:, 300:1000].copy()
    # 「起」放在 B 的 x=100 → 底图 x=400，正好落在 A 的覆盖范围内，
    # 旧逻辑下必然被 A 的内容覆盖掉。
    cv2.circle(B, (100, 280), 34, (255, 255, 255), -1)
    cv2.circle(B, (100, 280), 28, (60, 200, 60), -1)
    # 「终」放在 B 的 x=560 → 底图 x=860，A 覆盖不到，作为对照。
    cv2.circle(B, (560, 280), 34, (255, 255, 255), -1)
    cv2.circle(B, (560, 280), 28, (60, 60, 235), -1)
    return A, B


def test_badge_survives_when_overlapped_by_earlier_shot() -> None:
    A, B = _shots()
    cfg = VisionConfig()

    # 输入本身确实各有一枚徽章
    for img, ranges in ((B, [cfg.start_hsv_range]),
                        (B, [cfg.end_hsv_range, cfg.end_hsv_range2])):
        assert find_marker(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), ranges,
                           cfg.badge_min_area, cfg.badge_max_area) is not None

    res = stitch([A, B])
    assert res.ok, res.notes
    assert len(res.used) == 2, res.used
    assert res.canvas is not None

    hsv = cv2.cvtColor(res.canvas, cv2.COLOR_BGR2HSV)
    start = find_marker(hsv, [cfg.start_hsv_range],
                        cfg.badge_min_area, cfg.badge_max_area)
    end = find_marker(hsv, [cfg.end_hsv_range, cfg.end_hsv_range2],
                      cfg.badge_min_area, cfg.badge_max_area)
    assert start is not None, (
        "起点徽章被先贴的图盖掉了——extract_track() 会失去拓扑锚"
    )
    assert end is not None, "终点徽章不应受影响"
    # 写回必须落在原位：两枚徽章在 B 上相距 460px，变换是纯平移
    dx = abs(start[0][0] - end[0][0])
    assert abs(dx - 460) <= 6, f"徽章写回位置错了吗？两枚横向间距 {dx}px"
    assert any("徽章" in n for n in res.notes), f"缺少保护说明：{res.notes}"


def test_no_badge_images_are_untouched() -> None:
    """没有徽章时不应产生保护动作，也不该改变拼接结果。"""
    A, B = _shots()
    # 把 B 上的两枚徽章抹掉
    hsv = cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (35, 90, 90), (85, 255, 255))
    m |= cv2.inRange(hsv, (0, 90, 90), (12, 255, 255))
    B[m > 0] = (255, 255, 255)

    res = stitch([A, B])
    assert res.ok, res.notes
    assert not any("徽章" in n for n in res.notes), res.notes
