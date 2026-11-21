"""退化配准矩阵的回归测试。

背景：锚点地理坐标几乎重合时，兜底模型 `north_up_2pt` 会估出
scale≈0、矩阵全零，却仍标记 ok=True。结果是所有像素映射到同一个
经纬度（里程 0），下游 `_self_check` 对矩阵求逆还会抛 LinAlgError
让 `/api/recognize` 直接 500。

修复分两层：
  1. `app/register.is_degenerate` —— 变换不可逆就不算配准成功；
  2. `app/pipeline._self_check`  —— 诊断逻辑对矩阵求逆做兜底，
     任何来源的退化矩阵都不会打断整次识别。
"""

from __future__ import annotations

import numpy as np

from app.config import CONFIG
from app.pipeline import RidePipeline
from app.register import AnchorPair, Registration, is_degenerate, register


def _pair(px: float, py: float, lon: float, lat: float) -> AnchorPair:
    return AnchorPair(
        name="测试锚点", px=px, py=py, weight=1.0, kind="poi",
        lon=lon, lat=lat,
    )


def test_zero_matrix_is_degenerate():
    reg = Registration(ok=True, matrix=np.zeros((2, 2)), translation=np.zeros(2))
    assert is_degenerate(reg)


def test_nan_matrix_is_degenerate():
    m = np.array([[np.nan, 0.0], [0.0, 1.0]])
    assert is_degenerate(Registration(ok=True, matrix=m))


def test_near_zero_scale_is_degenerate():
    # 1e-5 米/像素：数学上可逆，但物理上等于没有比例
    m = np.array([[1e-5, 0.0], [0.0, -1e-5]])
    assert is_degenerate(Registration(ok=True, matrix=m))


def test_normal_matrix_is_not_degenerate():
    # 城市级截图约 15 米/像素、街道级约 0.5 米/像素，都远大于下限
    for s in (0.5, 3.0, 15.0):
        m = np.array([[s, 0.0], [0.0, -s]])
        assert not is_degenerate(Registration(ok=True, matrix=m))


def test_self_check_never_raises_on_singular_matrix():
    """自校验是诊断，不能把识别流程打成 500。"""
    reg = Registration(
        ok=True,
        matrix=np.zeros((2, 2)),
        translation=np.zeros(2),
        inliers=[_pair(100.0, 100.0, 121.3, 31.2)],
    )
    med, notes = RidePipeline(CONFIG)._self_check(reg, [])
    assert med == 0.0
    assert notes and "跳过" in notes[0]


def _self_check_reg(offsets_px: list[float], kinds: list[str], scale: float = 15.0):
    """构造一个「每个内点反投影偏移 = offsets_px[i]」的可逆配准。

    令像素→墨卡托为 `m = matrix @ p`，其中 matrix = [[s,0],[0,-s]]，
    translation = 0。取锚点真值 mercator = matrix @ (文字px + 偏移)，
    则反投影 `inv @ mercator = 文字px + 偏移`，偏移可控。
    """
    from app.geo.coords import mercator_to_lonlat

    matrix = np.array([[scale, 0.0], [0.0, -scale]])
    inliers = []
    for i, (off, kind) in enumerate(zip(offsets_px, kinds)):
        px, py = 100.0 + i * 100, 100.0
        mx = scale * (px + off)
        my = -scale * py
        lon, lat = mercator_to_lonlat(mx, my)
        p = _pair(px, py, lon, lat)
        p.kind = kind
        inliers.append(p)
    return Registration(ok=True, matrix=matrix, translation=np.zeros(2),
                        inliers=inliers)


def test_self_check_no_false_alarm_for_few_large_poi_offsets():
    """少数 transit/POI 标注偏移大，不应误判为整体平移。

    行者宁波长截图（2026-11-21）：城市级 14.9m/px，内点偏移中位
    597m，但 3 个可靠锚点相对方位全吻合——是标注偏移而非平移。
    旧代码用 400m 一刀切误报，并把置信度打 0.45 折。
    """
    # 6 个内点：4 个正常(100-250m)，2 个 transit 偏移大(1300m)
    # 中位约 250m，但只有 2/6=33% 超过各自类型容差 → 不报警
    reg = _self_check_reg(
        offsets_px=[10, 15, 12, 16, 87, 90],   # ×15 = 150..1350m
        kinds=["station", "poi", "poi", "station", "transit", "transit"],
    )
    med, notes = RidePipeline(CONFIG)._self_check(reg, [])
    assert med > 0
    assert not notes, f"少数大偏移被误判为整体平移: {notes}"


def test_self_check_still_catches_systematic_translation():
    """多数内点同时大幅偏移（整体平移），必须仍然报警。"""
    reg = _self_check_reg(
        offsets_px=[90, 95, 100, 92, 88, 97],  # ×15 ≈ 1350m，POI 容差 200m
        kinds=["station", "poi", "poi", "station", "poi", "icon"],
    )
    med, notes = RidePipeline(CONFIG)._self_check(reg, [])
    assert med > 0
    assert notes and "整体平移" in notes[0], f"整体平移漏报: {notes}"


def test_register_rejects_coincident_geo_anchors():
    """像素相距很远、地理坐标几乎相同的锚点，不能产出 ok=True。"""
    pairs = [
        _pair(100.0, 100.0, 121.300000, 31.200000),
        _pair(900.0, 900.0, 121.300001, 31.200001),
    ]
    reg = register(pairs, min_inliers=2)
    assert not reg.ok
    assert reg.notes


def test_external_scale_prior_rejects_wrong_scale_model():
    """外部尺度先验（概要里程÷轨迹像素长度）救回被垃圾锚点拉爆的尺度。

    2026-11-17 实测：街区级特写（真值 ~0.4 地面米/像素）里，2 点退化
    模型被垃圾锚点拉到 31 m/px，352 米骑行爆成 26.5 公里。概要里程
    0.352km 作为外部先验后，错误尺度的模型组合直接被枚举丢弃，
    north_up 模型由真实的荟聚簇锚点撑起。
    """
    from app.geo.coords import mercator_to_lonlat

    # 4 个自洽锚点：墨卡托坐标 = 0.6 * 像素 + 原点（上海 lat≈31.2）
    origin_x, origin_y = 13_500_000.0, 3_650_000.0
    px_pts = [(100.0, 100.0), (500.0, 100.0),
              (100.0, 500.0), (500.0, 500.0)]
    true_scale = 0.6  # 墨卡托米/像素
    pairs = []
    for i, (px, py) in enumerate(px_pts):
        mx = origin_x + true_scale * px
        my = origin_y - true_scale * py
        lon, lat = mercator_to_lonlat(mx, my)
        pairs.append(AnchorPair(
            name=f"测试锚点{i}", px=px, py=py, weight=1.0, kind="poi",
            lon=lon, lat=lat,
        ))

    # 外部先验与真值一致（先验为地面 m/px = 0.6*cos(lat)），
    # 不应破坏正确配准
    mean_lat = 31.2
    prior_ground = true_scale * __import__("math").cos(__import__("math").radians(mean_lat))
    reg = register(pairs, min_inliers=3, external_scale_prior=prior_ground)
    assert reg.ok
    assert abs(reg.scale - true_scale) < 0.2

    # 外部先验与真值矛盾（先验 5.0 = 数倍偏离）时也不该把正确模型
    # 全灭——先验只约束枚举搜索（容差 30%），正确簇若在容差内仍可
    # 存活；若被拒则诚实失败，但绝不允许回到 2 点垃圾模型的错误尺度。
    reg2 = register(pairs, min_inliers=3, external_scale_prior=5.0)
    if reg2.ok:
        assert abs(reg2.scale - 5.0) < 5.0 * 0.3
