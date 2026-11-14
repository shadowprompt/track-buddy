"""前端运动概要契约：订单详情上传控件 + 结果卡「运动概要」行。

与 ``tests/test_frontend_stitch_backend.py`` 相同的静态源码断言方式：
前端是单文件 HTML，渲染逻辑用 Node 测试单独验证（如需要），这里锁定
数据流两端必须存在的字符串。
"""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1]
        / "src" / "web" / "index.html").read_text(encoding="utf-8")


def test_summary_upload_control_exists() -> None:
    assert 'id="dropSummary"' in HTML
    assert 'id="summary"' in HTML
    assert "订单详情" in HTML


def test_summary_follows_the_form_data_pipeline() -> None:
    assert "pickedSummary" in HTML
    assert "fd.append('summary', pickedSummary, pickedSummary.name);" in HTML
    # 清空按钮也要清掉概要选择
    assert "setSummary(null);" in HTML


def test_result_card_renders_ride_summary() -> None:
    assert "d.ride_summary && d.ride_summary.available" in HTML
    # 显示字段：里程 / 时长 / 卡路里 / 减碳 / 开始时间 / 交叉校验
    for needle in ("d.ride_summary.distance_km", "d.ride_summary.duration_s",
                   "d.ride_summary.calories_kcal", "d.ride_summary.carbon_g",
                   "d.ride_summary.started_at", "与轨迹里程一致"):
        assert needle in HTML
