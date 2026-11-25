"""分辨率归一化回归：视觉提取行为不得随输入分辨率漂移。

2026-11-24 实测教训：小米 1.25× 等比截图（1080×2400 vs 标定 864×1920）
上，按标定分辨率实测的绝对像素参数（抹除半径/容差/面积窗口）全部失准，
弧长少 25%、配准尺度爆炸。修复方案：``vision_ref_short_side`` 预归一化
（视觉提取前等比缩放到画像标定样本短边），配合终点碎片检测的内容尺度
相对形式 ``endpoint_fragment_detect_rel``。

本文件的验证策略：把**标定样本本身**合成 1.25× / 0.5× 缩放图喂进同一条
提取链，断言「归一化后行为与原始分辨率一致」。样本缺失（CI / 他人
clone）时跳过，本地标定样本放 samples/ 目录。
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import pytest

from app.cv_utils import imread_unicode
from app.pipeline import prepare_input_image
from app.source_profile import AMAP, MEITUAN, XINGZHE, XIAOMI

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _sample(name: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(SAMPLES, name)))
    return hits[0] if hits else None


def _extract(path: str, profile) -> dict:
    """跑完整视觉提取，返回行为摘要（标记位置 / 弧长 / 点数）。"""
    from app.track import extract_track

    img = imread_unicode(path)
    prepared, _meta = prepare_input_image(img, profile.vision)
    r = extract_track(prepared, profile.vision)
    p = r.polyline.astype(float)
    arc = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return {
        "prepared_hw": (int(prepared.shape[0]), int(prepared.shape[1])),
        "start": r.start_marker,
        "end": r.end_marker,
        "n_points": len(p),
        "arc": arc,
    }


@pytest.mark.parametrize("profile", [MEITUAN, XINGZHE, XIAOMI, AMAP])
def test_prepared_size_independent_of_input_resolution(profile) -> None:
    """归一化后尺寸只由画像决定：同内容不同分辨率 → 同一 prepared 尺寸。

    画布尺寸取各画像标定短边的 0.5× / 1× / 1.6×（同一纵横比）。
    """
    ref = profile.vision.vision_ref_short_side
    assert ref, f"{profile.id} 必须配置 vision_ref_short_side"
    sizes = []
    for k in (0.5, 1.0, 1.6):
        w, h = int(ref * k), int(ref * 2.2 * k)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        out, _meta = prepare_input_image(img, profile.vision)
        sizes.append((out.shape[0], out.shape[1]))
    # 裁剪比例 × 缩放系数在整数像素上各有舍入，±2px 属同一行为
    hs = [s[0] for s in sizes]
    ws = [s[1] for s in sizes]
    assert max(hs) - min(hs) <= 2, f"prepared 高度随分辨率漂移: {sizes}"
    assert max(ws) - min(ws) <= 2, f"prepared 宽度随分辨率漂移: {sizes}"


def test_xingzhe_extraction_invariant_to_resolution() -> None:
    """行者标定样本 1.25× 放大后，提取行为必须与原分辨率一致。"""
    path = _sample("xingzhe_ningbo.jpg")
    if not path:
        pytest.skip("需要 samples/xingzhe_ningbo.jpg（本地标定样本）")
    base = _extract(path, XINGZHE)

    img = imread_unicode(path)
    big = cv2.resize(img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    big_path = path.replace(".jpg", "_x125.jpg")
    cv2.imwrite(big_path, big)
    try:
        scaled = _extract(big_path, XINGZHE)
    finally:
        os.remove(big_path)

    # 归一化把 1.25× 输入拉回同一 prepared 尺寸
    assert scaled["prepared_hw"] == base["prepared_hw"]
    # 徽章定位一致（重采样边缘误差给 8px 容差）
    for key in ("start", "end"):
        if base[key] is None or scaled[key] is None:
            continue
        d = float(np.hypot(base[key][0] - scaled[key][0],
                           base[key][1] - scaled[key][1]))
        assert d <= 8.0, f"{key} 漂移 {d:.1f}px"
    # 弧长/点数一致（闭环描迹对重采样敏感，给 8% 容差）
    assert scaled["arc"] == pytest.approx(base["arc"], rel=0.08)
    assert scaled["n_points"] == pytest.approx(base["n_points"], rel=0.10)


def test_meituan_extraction_invariant_to_resolution() -> None:
    """美团标定样本 0.5× 缩小后，提取行为必须与原分辨率一致。"""
    path = _sample("meituan_ride_shanghai.jpg")
    if not path:
        pytest.skip("需要 samples/meituan_ride_shanghai.jpg")
    base = _extract(path, MEITUAN)

    img = imread_unicode(path)
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small_path = path.replace(".jpg", "_x05.jpg")
    cv2.imwrite(small_path, small)
    try:
        scaled = _extract(small_path, MEITUAN)
    finally:
        os.remove(small_path)

    assert scaled["prepared_hw"] == base["prepared_hw"]
    for key in ("start", "end"):
        if base[key] is None or scaled[key] is None:
            continue
        d = float(np.hypot(base[key][0] - scaled[key][0],
                           base[key][1] - scaled[key][1]))
        assert d <= 8.0, f"{key} 漂移 {d:.1f}px"
    if base["arc"] > 0:
        assert scaled["arc"] == pytest.approx(base["arc"], rel=0.10)


def test_xiaomi_extraction_invariant_to_resolution() -> None:
    """小米标定样本 1.25× 放大后，闭环提取行为必须与原分辨率一致。

    这正是 2026-11-24 线上事故的形态（1080×2400 vs 864×1920）。
    """
    path = _sample("xiaomi_yuyao_864.jpg")
    if not path:
        pytest.skip("需要 samples/xiaomi_yuyao_864.jpg（本地标定样本）")
    base = _extract(path, XIAOMI)

    img = imread_unicode(path)
    big = cv2.resize(img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    big_path = path.replace(".jpg", "_x125.jpg")
    cv2.imwrite(big_path, big)
    try:
        scaled = _extract(big_path, XIAOMI)
    finally:
        os.remove(big_path)

    assert scaled["prepared_hw"] == base["prepared_hw"]
    for key in ("start", "end"):
        if base[key] is None or scaled[key] is None:
            continue
        d = float(np.hypot(base[key][0] - scaled[key][0],
                           base[key][1] - scaled[key][1]))
        assert d <= 8.0, f"{key} 漂移 {d:.1f}px"
    assert scaled["arc"] == pytest.approx(base["arc"], rel=0.08)


def test_amap_extraction_invariant_to_resolution() -> None:
    """高德标定样本 1.25× 放大后，提取行为必须与原分辨率一致。

    2026-11-25 补测：标定样本 samples/amap_ride_shanghai.jpg（2026-11-16
    上海 3.68km 骑行，从 uploads 归档）为 1260×2720，按
    map_crop_v=(0.065,0.44) 裁剪后短边 1020 → vision_ref_short_side=1020。
    """
    path = _sample("amap_ride_shanghai.jpg")
    if not path:
        pytest.skip("需要 samples/amap_ride_shanghai.jpg（高德标定样本）")
    base = _extract(path, AMAP)

    img = imread_unicode(path)
    big = cv2.resize(img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    big_path = path.replace(".jpg", "_x125.jpg")
    cv2.imwrite(big_path, big)
    try:
        scaled = _extract(big_path, AMAP)
    finally:
        os.remove(big_path)

    assert scaled["prepared_hw"] == base["prepared_hw"]
    for key in ("start", "end"):
        if base[key] is None or scaled[key] is None:
            continue
        d = float(np.hypot(base[key][0] - scaled[key][0],
                           base[key][1] - scaled[key][1]))
        assert d <= 8.0, f"{key} 漂移 {d:.1f}px"
    assert scaled["arc"] == pytest.approx(base["arc"], rel=0.08)


def test_endpoint_fragment_rel_scales_with_badge_d() -> None:
    """终点碎片检测的相对形式随 badge_d 缩放：badge_d 翻倍 → 半径翻倍。"""
    from app.config import VisionConfig
    from app.track import _endpoint_fragment_mask

    # 160×160 掩膜：主体在左侧，碎片 (130, 80) 处 4×4 团块
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[70:90, 10:60] = 1          # 主轨迹
    mask[78:82, 128:132] = 1        # 碎片
    end_pt = (40, 80)

    cfg = VisionConfig(endpoint_fragment_detect_rel=(1.0, 0.0005))
    # 半径维度：badge_d=80 → radius=80，碎片 (130,80) 距终点 90px 超出
    n_far, m_far = _endpoint_fragment_mask(mask, end_pt, cfg, badge_d=80.0)
    assert n_far == 0 and m_far.sum() == 0
    # badge_d=120 → radius=120 覆盖碎片、min_area=7.2 ≤ 16 → 检出
    n_ok, m_ok = _endpoint_fragment_mask(mask, end_pt, cfg, badge_d=120.0)
    assert n_ok == 16 and int(m_ok.sum()) == 16
    # min_area 维度：同 badge_d、更严格的面积系数（0.005×14400=72 > 16）→ 拦下
    cfg_strict = VisionConfig(endpoint_fragment_detect_rel=(1.0, 0.005))
    n_blocked, _ = _endpoint_fragment_mask(mask, end_pt, cfg_strict, badge_d=120.0)
    assert n_blocked == 0


def test_xingzhe_uses_rel_fragment_spec() -> None:
    """行者画像必须走内容尺度相对形式（绝对像素形式已退役）。"""
    v = XINGZHE.vision
    assert v.endpoint_fragment_detect is None
    assert v.endpoint_fragment_detect_rel == pytest.approx((2.55, 0.052))
