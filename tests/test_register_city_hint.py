"""配准的城市提示先验与物理尺度护栏回归。

2026-11-25 余姚 13.09km 跑步样本事故：截图是局部放大视图（可见弧长仅
~4km），概要÷弧长先验虚胖 3 倍；同时「姜庵/梁湖头村/牛山」等村名在
宁波江北有同名 POI 簇，该簇像素自洽（RMSE 261m），把轨迹拉到 30km 外。
真簇（余姚凤山街道，腾讯地图+地名志双确认）被先验窗挡在门外。

三层防线：
1. 物理尺度护栏（register 内）：同名异地候选可拼出任意大的伪尺度解
   （实测 102 m/px 拿到 2 个「自洽」内点），超出真实截图可能范围的
   解一律拒绝；
2. 城市提示先验（register_with_hint）：城市内候选簇优先于城外同名簇；
3. 投票降级重试：锚点尺度投票被垃圾候选带偏时，丢弃先验做纯几何枚举。
"""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from app.register import AnchorPair, register, register_with_hint  # noqa: E402


def _pair(name: str, px: float, py: float, lon: float, lat: float,
          kind: str = "poi") -> AnchorPair:
    return AnchorPair(name=name, px=px, py=py, lon=lon, lat=lat,
                      kind=kind, weight=0.85, source="baidu")


# 余姚凤山街道（真簇，腾讯地图确认坐标）
_YUYAO_BBOX = (120.95, 29.90, 121.35, 30.20)


def test_scale_gate_rejects_pseudo_scale() -> None:
    """两个错候选相距 15km / 像素 330px → 47 m/px 伪解必须被拒绝。"""
    pairs = [
        _pair("甲", 0.0, 0.0, 121.00, 30.00),
        _pair("乙", 330.0, 0.0, 121.35, 30.00),   # ~33.7km / 330px ≈ 102 m/px
    ]
    reg = register(pairs, inlier_m=200.0, min_inliers=2,
                   external_scale_prior=None, _skip_vote_prior=True)
    assert not reg.ok, "物理尺度护栏必须拦下伪尺度解"

    # 对照：正常城市级尺度（~10 m/px）不受影响
    pairs_ok = [
        _pair("甲", 0.0, 0.0, 121.00, 30.00),
        _pair("乙", 400.0, 0.0, 121.04, 30.00),   # ~3.85km / 400px ≈ 9.6
    ]
    reg_ok = register(pairs_ok, inlier_m=200.0, min_inliers=2,
                      external_scale_prior=None, _skip_vote_prior=True)
    assert reg_ok.ok


def test_city_hint_rescues_correct_cluster() -> None:
    """同名异地两簇都像素自洽时，城市提示必须选城内真簇。

    复刻余姚事故形态：城外簇（宁波江北同名姜庵）像素自洽且尺度合理，
    城内簇（余姚凤山街道）也是真候选——没有提示时几何分不出对错，
    城市提示必须裁决。
    """
    pairs = [
        # 城外簇：像素自洽（scale ~7 m/px），宁波江北同名 POI，
        # 3 个锚点使其成为无提示时的全量最优解
        _pair("姜庵", 548.5, 289.5, 121.5572, 29.8813),
        _pair("热情世界杯杨", 579.0, 542.5, 121.5565, 29.8658),
        _pair("湖闸", 443.0, 771.0, 121.5467, 29.8514),
        # 城内簇：余姚凤山街道（腾讯确认）
        _pair("梁湖头村", 522.5, 403.5, 121.2004, 30.0731),
        _pair("牛山", 330.0, 114.0, 121.1926, 30.0804),
    ]
    reg = register_with_hint(
        list(pairs), inlier_m=200.0, min_inliers=2,
        external_scale_prior=None,
        city_bbox=_YUYAO_BBOX, city_name="余姚",
    )
    assert reg.ok
    inlier_names = {p.name for p in reg.inliers}
    assert "梁湖头村" in inlier_names and "牛山" in inlier_names
    # 交付解的内点必须落在城市 bbox 内
    for p in reg.inliers:
        assert _YUYAO_BBOX[0] <= p.lon <= _YUYAO_BBOX[2]
        assert _YUYAO_BBOX[1] <= p.lat <= _YUYAO_BBOX[3]
    # 落选的全量解保留为「换个位置」备选
    assert any("姜庵" in {q.name for q in a.inliers}
               for a in reg.alternatives if a.ok)


def test_no_city_hint_keeps_geometry_choice() -> None:
    """没有城市提示时不强行纠偏：几何自洽簇照常交付，不崩不拒。"""
    pairs = [
        _pair("姜庵", 548.5, 289.5, 121.5572, 29.8813),
        _pair("热情世界杯杨", 579.0, 542.5, 121.5565, 29.8658),
        _pair("梁湖头村", 522.5, 403.5, 121.2004, 30.0731),
        _pair("牛山", 330.0, 114.0, 121.1926, 30.0804),
    ]
    reg = register_with_hint(
        list(pairs), inlier_m=200.0, min_inliers=2,
        external_scale_prior=None,
    )
    assert reg.ok
    assert len({p.name for p in reg.inliers}) >= 2


def test_wrong_prior_fails_honestly_without_city_hint() -> None:
    """错误先验 + 无城市提示：必须诚实失败（2026-11-17 契约——
    外部先验是权威，矛盾时不许背叛先验去救锚点簇）。

    局部视图的正确解法在管线层：截断检测（track_clipped）不把先验
    传进 register；有城市提示时走城市内簇路径。两者都缺失时宁可失败。
    """
    pairs = [
        _pair("梁湖头村", 522.5, 403.5, 121.2004, 30.0731),
        _pair("牛山", 330.0, 114.0, 121.1926, 30.0804),
        _pair("湖闸", 443.0, 771.0, 121.1964, 30.0622),
    ]
    reg = register_with_hint(
        list(pairs), inlier_m=200.0, min_inliers=2,
        external_scale_prior=9.92,   # 局部视图虚胖 3 倍的先验
    )
    assert not reg.ok, "外部先验与锚点簇矛盾且无城市提示时必须诚实失败"
    assert reg.notes


def test_city_hint_with_wrong_prior_still_recovers() -> None:
    """错误先验 + 城市提示：城市内簇按几何优先配准，先验不掺和。"""
    pairs = [
        _pair("梁湖头村", 522.5, 403.5, 121.2004, 30.0731),
        _pair("牛山", 330.0, 114.0, 121.1926, 30.0804),
        _pair("湖闸", 443.0, 771.0, 121.1964, 30.0622),
    ]
    reg = register_with_hint(
        list(pairs), inlier_m=200.0, min_inliers=2,
        external_scale_prior=9.92,
        city_bbox=_YUYAO_BBOX, city_name="余姚",
    )
    assert reg.ok
    inlier_names = {p.name for p in reg.inliers}
    assert {"梁湖头村", "牛山"} & inlier_names
    assert reg.rmse_reliable_m < 300.0
    # 交付尺度必须是几何真值（~3 m/px），不能被先验带跑
    import math
    lats = [p.lat for p in reg.inliers]
    ground = reg.scale * math.cos(math.radians(sum(lats) / len(lats)))
    assert ground < 6.0


def test_vote_degradation_recovers_when_vote_is_garbage() -> None:
    """尺度投票被垃圾候选带偏时，纯几何枚举降级必须救回正确解。"""
    # 垃圾候选（异地同名）会把两两距离比投票拉到 >40 m/px；
    # 正确簇在 ~3 m/px。投票先验窗内无物理合理解 → 降级重试。
    pairs = [
        # 垃圾：把投票拉爆
        _pair("梅采摘山庄", 564.0, 573.2, 121.0579, 30.0096),
        _pair("竹轩亭", 638.0, 848.0, 121.3097, 29.9103),
        # 正确簇（余姚凤山街道）
        _pair("梁湖头村", 522.5, 403.5, 121.2004, 30.0731),
        _pair("牛山", 330.0, 114.0, 121.1926, 30.0804),
        _pair("湖闸", 443.0, 771.0, 121.1964, 30.0622),
    ]
    reg = register(list(pairs), inlier_m=200.0, min_inliers=2,
                   external_scale_prior=None)
    assert reg.ok, "投票被带偏时必须降级重试而非直接失败"
    inlier_names = {p.name for p in reg.inliers}
    assert "梁湖头村" in inlier_names
    # 交付尺度必须物理合理（降级后仍受护栏约束）
    lats = [p.lat for p in reg.inliers]
    import math
    ground = reg.scale * math.cos(math.radians(sum(lats) / len(lats)))
    assert 0.05 <= ground <= 40.0
