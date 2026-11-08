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


def test_register_rejects_coincident_geo_anchors():
    """像素相距很远、地理坐标几乎相同的锚点，不能产出 ok=True。"""
    pairs = [
        _pair(100.0, 100.0, 121.300000, 31.200000),
        _pair(900.0, 900.0, 121.300001, 31.200001),
    ]
    reg = register(pairs, min_inliers=2)
    assert not reg.ok
    assert reg.notes
