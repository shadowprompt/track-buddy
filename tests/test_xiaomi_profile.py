"""小米运动健康来源画像回归测试。

画像基于 2026-11-23 用户提供的 864×1920 户外跑步概览截图建立：
上部地图 + 红橙黄绿速度渐变轨迹 + 下部运动统计卡片。

当前测试锁定的是来源接入和视觉前处理契约；绝对位置精度仍需更多真实样本，
因此不在这里伪造地理坐标或把单张截图当成完整标定真值。
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.config import VisionConfig
from app.cv_utils import crop_map_region
from app.input_quality import extract_distance_km
from app.source_profile import PROFILES, XIAOMI, resolve_source_profile
from app.track import build_track_mask


def _items(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=t) for t in texts]


def test_xiaomi_profile_registered() -> None:
    assert "xiaomi" in PROFILES
    src = resolve_source_profile("xiaomi")
    assert src is XIAOMI
    assert src.title == "小米运动健康"
    assert XIAOMI.vision.map_crop_v == (0.0, 0.54)
    assert XIAOMI.vision.loop_course is True
    assert XIAOMI.vision.loop_bridge_fragments is True
    # 起终点相距 133px（走廊 ~1.2km），成对相邻约束会把走廊上的红色
    # 轨迹块配成终点（2026-11-23 实测）——必须关闭。
    assert XIAOMI.vision.pair_loop_badges is False
    # 棒棒糖交付排序 + 徽章团块抹除（UI 晕圈半径 ~42px）。
    assert XIAOMI.vision.split_loop_delivery is True
    assert XIAOMI.vision.erase_marker_blobs_r == 45
    assert XIAOMI.vision.track_hsv_range == (
        (0, 140, 120, 40, 255, 255),
        (40, 140, 120, 85, 255, 255),
    )


def test_xiaomi_map_crop_removes_summary_card() -> None:
    img = np.zeros((1920, 864, 3), dtype=np.uint8)
    out = crop_map_region(img, XIAOMI.vision)
    assert out.shape[:2] == (int(round(1920 * 0.54)), 864)


def test_xiaomi_gradient_track_mask_keeps_warm_and_green_segments() -> None:
    img = np.full((120, 300, 3), 255, dtype=np.uint8)
    # BGR colors chosen to fall into the calibrated red/orange and green bands.
    cv2.line(img, (10, 35), (130, 35), (20, 120, 255), 8)   # red-orange
    cv2.line(img, (160, 35), (285, 35), (40, 230, 80), 8)   # green
    # Low-saturation pale green background line must not be treated as track.
    cv2.line(img, (10, 90), (285, 90), (170, 220, 190), 8)

    mask = build_track_mask(img, XIAOMI.vision)
    assert int(mask[25:46, 10:130].sum()) > 0
    assert int(mask[25:46, 160:285].sum()) > 0
    assert int(mask[80:101, 10:285].sum()) == 0


def test_lollipop_order_reorders_walk() -> None:
    """棒棒糖排序：walk[尖端→起点→结点→环→结点] 重排为
    起点→走廊→环→走廊→起点→终点，弧长恰好多出走廊+底段一遍。"""
    from app.track import _lollipop_order

    # 结构：尖端(0..9) → 起点(10) → 走廊(10..40，止于结点) →
    # 环(41..69) → 回到结点(70=尾端)
    tip = [(0.0, 100.0 - i) for i in range(10)]
    corridor = [tuple(v) for v in np.linspace(
        (10.0, 90.0), (40.0, 120.0), 31)]            # 31 点止于结点
    loop = [(40.0 + 10 * np.sin(t), 120.0 + 10 * (1 - np.cos(t)))
            for t in np.linspace(0, 2 * np.pi, 30)][1:]   # 圆环回结点
    walk = np.array(tip + corridor + loop, dtype=float)
    out = _lollipop_order(walk, start_pt=(10, 90), end_pt=(0, 100))
    assert out is not None
    n, S, j = len(walk), 10, 40
    assert len(out) == n + (j - S) + 2  # 走廊多一遍 + 两处 1 点衔接
    assert tuple(out[0]) == (10.0, 90.0)       # 起点在头部
    assert tuple(out[-1]) == (0.0, 100.0)      # 终点（尖端）在尾部
    # 走廊被计两遍：弧长 = walk 弧长 + 走廊弧长
    def _arc(a):
        return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
    assert _arc(out) == pytest.approx(
        _arc(walk) + _arc(walk[S:j + 1]), rel=1e-6)


def test_lollipop_order_rejects_non_lollipop_walk() -> None:
    """尾端不是途中 revisit 点 / 起终点不在 walk 上时回退 None。"""
    from app.track import _lollipop_order

    walk = np.array(
        [(float(i), 50.0) for i in range(30)], dtype=float)
    # 尾端 (29,50) 只出现一次 → 结构不符
    assert _lollipop_order(walk, start_pt=(0, 50), end_pt=(29, 50)) is None
    # 起点不在 walk 附近
    pts: list[tuple[float, float]] = [(0.0, 100.0 - i) for i in range(10)]
    pts += [(10.0 + i, 90.0) for i in range(30)]
    pts += [(40.0 + i, 120.0 + i) for i in range(15)]
    pts += [(55.0 - i, 135.0) for i in range(15)]
    pts.append((40.0, 120.0))
    assert _lollipop_order(
        np.array(pts, dtype=float), start_pt=(500, 500), end_pt=(0, 100),
    ) is None


def test_xiaomi_distance_keywords_prefer_activity_distance() -> None:
    # The speed/pace/card text must not become a distance candidate.
    texts = _items(
        "13.09 公里",
        "4'34\"",
        "831kcal",
        "平均配速",
        "累计 99 公里",
    )
    assert extract_distance_km(texts) == pytest.approx(13.09)


def test_xiaomi_loop_course_enabled() -> None:
    """小米实测路线是闭环（起点→主干→顶部环→右下行→回到起点附近）。

    开环模式只描出上行半程（余姚样本 805px），右侧下行段被 POI 文字
    打断后整段丢失；闭环邮路追踪描出全程 1700px，隐含尺度与概要里程
    13.09km 自洽。成对徽章定位（`endpoint_fragment_detect`）沿用行者
    逻辑，在小米样本上同样正确定位了绿起/红终。
    """
    assert XIAOMI.vision.loop_course
    assert XIAOMI.vision.loop_bridge_fragments


def test_xiaomi_config_remains_separate_from_default() -> None:
    default = VisionConfig()
    assert XIAOMI.vision is not default
    assert XIAOMI.vision.track_hsv_range != default.track_hsv_range
    assert XIAOMI.vision.map_crop_v != default.map_crop_v


def test_prepare_input_image_crops_and_normalizes_for_xiaomi() -> None:
    """CLI/直跑链路的输入准备：小米裁地图区 + 归一化到 864 短边。"""
    from app.pipeline import prepare_input_image

    # 1.25× 标定分辨率的输入（1080×2400）：裁剪后归一化回 864 宽
    img = np.zeros((2400, 1080, 3), dtype=np.uint8)
    out, meta = prepare_input_image(img, XIAOMI.vision)
    assert meta["crop_v"] == (0.0, 0.54)
    # 裁剪到 1080×1296 后归一化到 864 短边（1037 = round(1296×0.8)）
    assert out.shape[:2] == (1037, 864)
    assert meta["prepared_hw"] == [1037, 864]

    # 标定分辨率输入：裁剪但不缩放。prepared_hw 始终记录实际尺寸
    # （2026-11-25 契约变更：None 曾导致 API 层跳过像素反投影，
    #  image_size 留在裁剪空间，前端叠绘纵向拉伸 1.85×）。
    img2 = np.zeros((1920, 864, 3), dtype=np.uint8)
    out2, meta2 = prepare_input_image(img2, XIAOMI.vision)
    assert meta2["prepared_hw"] == [1037, 864]
    assert meta2["original_hw"] == [1920, 864]
    assert out2.shape[:2] == (1037, 864)

    from app.source_profile import MEITUAN, XINGZHE

    # 美团无裁剪，标定分辨率恰为 1260 短边：不缩放，prepared_hw =
    # 实际尺寸（与 original 相同，反投影为恒等变换）
    big = np.zeros((2720, 1260, 3), dtype=np.uint8)
    same, m3 = prepare_input_image(big, MEITUAN.vision)
    assert m3["crop_v"] is None and m3["prepared_hw"] == [2720, 1260]
    assert same.shape[:2] == (2720, 1260)
    bigger = np.zeros((5440, 2520, 3), dtype=np.uint8)
    out3, m4 = prepare_input_image(bigger, MEITUAN.vision)
    assert m4["prepared_hw"] == [2720, 1260]

    # 行者归一化到 640
    up = np.zeros((1420, 800, 3), dtype=np.uint8)
    out4, m5 = prepare_input_image(up, XINGZHE.vision)
    assert m5["prepared_hw"] == [1136, 640]


def test_xiaomi_ui_texts_rejected_real_pois_kept() -> None:
    """小米统计卡片/图例/状态栏文案不得占用地理编码名额。"""
    from app.anchor_filter import is_ui_noise

    ui_texts = [
        "公里", "更慢", "更快", "运动时长", "活动卡路里",
        "平均配速", "平均海拔", "平均步频", "平均步幅",
        "小米运动健康一户外跑步",          # OCR 粘连块
        "中国联通21:50385",                # 状态栏粘连块
        "动态轨迹",
    ]
    for t in ui_texts:
        assert is_ui_noise(t), f"小米 UI 文案未被过滤: {t!r}"

    places = ["牛山", "姜庵", "湖闸", "竹轩亭", "梁湖头村",
              "达环路", "冶山路", "热情世界杯杨梅采摘山庄"]
    for t in places:
        assert not is_ui_noise(t), f"真实地名被误杀: {t!r}"

