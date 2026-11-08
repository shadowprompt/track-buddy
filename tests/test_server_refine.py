"""后台路网精修触发条件的回归测试。

事故：服务端曾用「尺度必须大于模式分界值」当触发门槛，而
`pipeline._road_refine` 恰好以同一个值为界：
  小于分界 → 修正模式（平移轨迹贴路，能消除 POI 标注造成的道路偏移）
  大于分界 → 校验模式（不平移）
结果落在修正模式、最需要平移的那一类图反而永远拿不到精修。

修复后判据只看状态：允许精修、未精修过、已有真实坐标、有轨迹点。
测试取「分界值两侧的相对倍数」作为输入尺度，不绑定任何具体截图。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.config import CONFIG
from app.server import _need_background_refine

BOUNDARY = CONFIG.road_fit_low_zoom_m_per_px


def _result(
    scale: float | None = None,
    coords: list[list[float]] | None = None,
    road_fit: dict | None = None,
    track_points: int = 100,
) -> SimpleNamespace:
    if scale is None:
        scale = BOUNDARY * 0.5
    if coords is None:
        coords = [[0.0, 0.0], [0.001, 0.001]]
    return SimpleNamespace(
        road_fit=road_fit if road_fit is not None else {"ok": False},
        diagnostics={"scale_ground_m_per_px": scale},
        track_points=track_points,
        geojson={"features": [{"geometry": {"coordinates": coords}}]},
    )


def test_below_boundary_triggers_refine():
    """落在修正模式的尺度（旧逻辑漏掉的那一档）必须触发。"""
    assert _need_background_refine(
        _result(scale=BOUNDARY * 0.5), True, False
    ) is True


def test_street_level_also_triggers():
    assert _need_background_refine(
        _result(scale=BOUNDARY * 0.05), True, False
    ) is True


def test_above_boundary_still_triggers():
    """城市级低缩放原本就会触发，不能回归。"""
    assert _need_background_refine(
        _result(scale=BOUNDARY * 2.0), True, False
    ) is True


def test_already_refined_is_skipped():
    assert _need_background_refine(
        _result(road_fit={"ok": True}), True, False
    ) is False


def test_registration_failure_is_skipped():
    assert _need_background_refine(_result(coords=[]), True, False) is False
    assert _need_background_refine(_result(scale=0.0), True, False) is False
    assert _need_background_refine(_result(track_points=0), True, False) is False


def test_user_opt_out_or_sync_is_skipped():
    assert _need_background_refine(_result(), False, False) is False
    assert _need_background_refine(_result(), True, True) is False
