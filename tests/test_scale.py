"""尺度自适应结构的回归测试。

核心命题：阈值必须由「标尺 × 无量纲系数」产生，且**两类尺度互不串扰**。

* 内容尺度（徽章直径 → 线宽、贴合容差、形态学核）不随画布尺寸变化；
* 画布尺度（对角线 → 拓扑距离、画布预算）不随徽章尺寸变化。

旧代码把二者混用且写死绝对像素，于是换一批截图 / 换一个分辨率
就要重新调一次数字。这里的测试就是防止那种写法回流。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.scale import (
    BADGE_D_MAX,
    BADGE_D_MIN,
    BYTES_PER_PIXEL,
    CANVAS_INPUT_FACTOR,
    DEFAULT_BADGE_D,
    MAX_CANVAS_BYTES,
    Scale,
    canvas_pixel_budget,
    odd_kernel,
)

CONTENT_ATTRS = (
    "line_w",
    "touch_dist",
    "coarse_dist",
    "min_area",
    "close_kernel",
    "open_kernel",
    "skeleton_kernel",
    "bridge_width",
    "conf_touch",
    "marker_lead",
)


def _img(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def test_odd_kernel_is_odd_and_bounded() -> None:
    for v in (0.1, 1.0, 2.4, 3.7, 11.2, 40.0):
        k = odd_kernel(v)
        assert k % 2 == 1
        assert k >= 3
    assert odd_kernel(1.0, minimum=7) == 7


def test_badge_diameter_is_clamped() -> None:
    assert Scale.from_image(_img(100, 100), 1).badge_d == BADGE_D_MIN
    assert Scale.from_image(_img(100, 100), 99999).badge_d == BADGE_D_MAX
    assert Scale.from_image(_img(100, 100)).badge_d == DEFAULT_BADGE_D


def test_content_scale_ignores_canvas_size() -> None:
    """画布再大，UI 徽章和轨迹线宽仍是同一个尺寸。"""
    small = Scale.from_image(_img(1000, 600), 60)
    big = Scale.from_image(_img(10000, 7000), 60)
    for attr in CONTENT_ATTRS:
        assert getattr(small, attr) == getattr(big, attr), attr


def test_content_scale_tracks_badge_size() -> None:
    """换缩放级别（徽章变大）时，内容阈值同比放大。"""
    a = Scale.from_image(_img(1000, 600), 60)
    b = Scale.from_image(_img(1000, 600), 120)
    assert math.isclose(b.line_w / a.line_w, 2.0)
    assert math.isclose(b.touch_dist / a.touch_dist, 2.0)
    assert b.min_area > a.min_area
    assert b.close_kernel > a.close_kernel


def test_canvas_scale_ignores_badge_size() -> None:
    a = Scale.from_image(_img(1000, 600), 60)
    b = Scale.from_image(_img(1000, 600), 120)
    assert b.topo_dist == a.topo_dist


def test_canvas_scale_tracks_diagonal() -> None:
    a = Scale.from_image(_img(1000, 600), 60)
    b = Scale.from_image(_img(4000, 2400), 60)
    assert b.diag / a.diag == pytest.approx(4.0)
    assert b.topo_dist / a.topo_dist == pytest.approx(4.0)


def test_bridge_gap_is_adaptive_and_bounded() -> None:
    """补缝容差随路线跨度增长，但始终受内容尺度上下界约束。"""
    s = Scale.from_image(_img(4000, 4000), 64)
    short = s.bridge_gap((0, 0), (100, 0))
    long = s.bridge_gap((0, 0), (5000, 3000))
    assert short < long
    assert short >= 64 * Scale.BRIDGE_MIN_PER_BADGE
    assert long <= 64 * Scale.BRIDGE_PER_BADGE


def test_zero_shift_tol_scales_with_short_side() -> None:
    a = Scale.from_image(_img(1000, 600), 60)
    b = Scale.from_image(_img(2000, 1200), 60)
    assert b.zero_shift_tol > a.zero_shift_tol
    assert Scale.from_image(_img(10, 10), 60).zero_shift_tol == Scale.ZERO_MIN


def test_canvas_budget_adapts_to_input_and_is_capped() -> None:
    """上限由输入规模与内存预算共同决定，不是固定像素数。"""
    one = canvas_pixel_budget(3_000_000)
    many = canvas_pixel_budget(36_000_000)
    assert one < many
    assert one == int(3_000_000 * CANVAS_INPUT_FACTOR)
    assert canvas_pixel_budget(10**12) == MAX_CANVAS_BYTES // BYTES_PER_PIXEL
    assert canvas_pixel_budget(0) > 0


def test_all_derived_values_positive() -> None:
    s = Scale.from_image(_img(1234, 987), 73)
    for attr in CONTENT_ATTRS + ("topo_dist", "zero_shift_tol", "diag", "short"):
        assert getattr(s, attr) > 0, attr


# --- 集成：换分辨率不能失效 -------------------------------------------------

FOUR_SHOTS = (
    "1794096457_2463f8d7",
    "1794096457_5abde3aa",
    "1794096457_63ce03bd",
    "1794096457_d2051989",
)


def test_scale_invariance_on_real_shots() -> None:
    """同一批截图整体缩小后仍须识别出完整长轨迹。

    这条测试直接防止「为某个分辨率调死阈值」的写法回流：
    缩放后徽章、线宽、画布、配准残差全都变了，
    若阈值是绝对像素，链路必然在某一步断掉。
    """
    paths = [Path("data/uploads") / f"{name}.jpg" for name in FOUR_SHOTS]
    if not all(p.exists() for p in paths):
        pytest.skip("真实截图未随仓库分发")

    from app.config import CONFIG
    from app.multishot import stitch
    from app.track import extract_track

    base = [cv2.imread(str(p)) for p in paths]
    assert all(im is not None for im in base)
    factor = 0.85
    imgs = [
        cv2.resize(im, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        for im in base
    ]
    res = stitch(imgs)
    assert res.ok and res.canvas is not None, res.notes
    tr = extract_track(res.canvas, CONFIG.vision)
    assert len(tr.polyline) > 100
    assert tr.start_marker and tr.end_marker
    span = math.dist(tr.start_marker, tr.end_marker)
    diag = math.hypot(res.canvas.shape[1], res.canvas.shape[0])
    # 起终点跨度必须占画布显著比例，避免「只抓到一小段」也算通过
    assert span > 0.1 * diag
