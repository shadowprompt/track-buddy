"""轨迹「直线度」检测：识别「截图里只有一条起终点连线」的输入。

这类输入下配准、路网精修再准也得不到沿路走向，必须明确告知用户，
而不是给一个置信度不高却看起来精确的坐标结果。判据只看折线形状，
与截图分辨率、城市、坐标系无关。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline import _track_lacks_shape_info, _track_straightness

PIPELINE_SRC = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "src" / "app" / "pipeline.py"
).read_text(encoding="utf-8")


def test_perfect_line_is_one() -> None:
    pts = np.array([[float(i * 10), 1300.0 + i * 0.3] for i in range(80)])
    assert _track_straightness(pts) == pytest.approx(1.0, abs=1e-6)


def test_real_curved_track_is_well_below_threshold() -> None:
    # 明显的之字形：折线总长远大于端点弦长
    pts = np.array(
        [[float(i * 10), 1300.0 + (120 if i % 4 in (1, 2) else 0)]
         for i in range(80)]
    )
    assert _track_straightness(pts) < 0.8


def test_curved_track_passes_the_gate() -> None:
    # 一个直角转弯的真实骑行线不该被误判成「起终点连线」
    pts = np.array(
        [[float(i * 10), 1300.0] for i in range(40)]
        + [[390.0, 1300.0 + float(j * 10)] for j in range(1, 40)]
    )
    assert _track_straightness(pts) < 0.8


def test_scale_and_translation_invariant() -> None:
    base = np.array([[float(i), float(i) * 0.05] for i in range(50)])
    scaled = base * 7.3 + np.array([120.0, -900.0])
    assert _track_straightness(scaled) == pytest.approx(
        _track_straightness(base), abs=1e-9
    )


def test_degenerate_inputs_return_zero() -> None:
    assert _track_straightness([]) == 0.0
    assert _track_straightness([[1.0, 2.0], [3.0, 4.0]]) == 0.0
    assert _track_straightness([[5.0, 5.0]] * 10) == 0.0


def test_never_exceeds_one() -> None:
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1000, size=(200, 2))
    assert 0.0 <= _track_straightness(pts) <= 1.0


def test_gate_needs_all_three_signals() -> None:
    """只看直线度会误判「沿直路骑」的真实轨迹，必须三条件同时成立。"""
    line = np.array([[float(i * 10), 1300.0 + 3.0 * np.sin(i / 9.0)]
                     for i in range(80)])
    flat_box = (100.0, 1300.0, 800.0, 100.0)   # 长宽比 8
    square_box = (100.0, 1300.0, 800.0, 500.0)  # 长宽比 1.6
    assert _track_lacks_shape_info(line, flat_box, False) is True
    # 有街道名锚点：交给缩放评估与路名约束，不在这里下结论
    assert _track_lacks_shape_info(line, flat_box, True) is False
    # 轨迹框不扁长：可能是沿直路骑的真实轨迹（实测 12 图拼接为 2.0）
    assert _track_lacks_shape_info(line, square_box, False) is False
    # 真正曲折的轨迹不该被判成连线
    bent = np.array([[float(i * 10), 1300.0 + 200.0 * np.sin(i / 6.0)]
                     for i in range(80)])
    assert _track_lacks_shape_info(bent, flat_box, False) is False
    # 点数太少 / bbox 异常时安全降级
    assert _track_lacks_shape_info(line[:5], flat_box, False) is False
    assert _track_lacks_shape_info(line, None, False) is False


def test_real_sample_triggers_the_gate() -> None:
    """这次实测的直线轨迹（原始 731 点、bbox 710x95）必须触发。"""
    line = np.array([[float(269 + i * 0.96), 1350.0 + 12.0 * np.sin(i / 12.0)]
                     for i in range(731)])
    # 原始折线带提取抖动，直线度约 0.83，仍在 0.80 门槛之上
    assert _track_straightness(line) >= 0.80
    assert _track_lacks_shape_info(line, (269.0, 1313.0, 710.0, 95.0), False)


def test_pipeline_wires_the_warning_and_diagnostic() -> None:
    """判据、文案与 diagnostics 字段必须真接进主流程。"""
    assert "_track_lacks_shape_info(" in PIPELINE_SRC
    assert 'bool(zoom.get("has_road_names"))' in PIPELINE_SRC
    assert "起终点连线而非沿道路的实际轨迹" in PIPELINE_SRC
    assert "请截取包含完整轨迹曲线的页面" in PIPELINE_SRC
    assert '"track_straightness",' in PIPELINE_SRC
    assert "track_straightness=round(straightness, 3)" in PIPELINE_SRC
    # 必须在配准之前算完，不能依赖 reg / geojson
    assert PIPELINE_SRC.index("straightness = _track_straightness") < \
        PIPELINE_SRC.index("reg = register_with_hint(")
