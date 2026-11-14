"""前端运动概要契约：运动概要上传控件（第一位）+ 结果卡「运动概要」行。

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
    # 话术：不再叫「订单详情」，统一叫「运动概要」；且排在截图上传第一位
    assert "运动概要截图" in HTML
    assert "订单详情" not in HTML
    assert HTML.index('id="dropSummary"') < HTML.index('id="drop"') < HTML.index('id="dropDetails"')


def test_summary_follows_the_form_data_pipeline() -> None:
    assert "pickedSummary" in HTML
    assert "fd.append('summary', pickedSummary, pickedSummary.name);" in HTML
    # 清空按钮也要清掉概要选择
    assert "setSummary(null);" in HTML


def test_set_summary_is_defined_and_bound() -> None:
    # 回归锁定：setSummary 被引用就必须有定义，否则 bindDrop(dropSummary, ...)
    # 在页面加载时抛 ReferenceError，整个初始化脚本中断，上传框全部失效。
    assert "function setSummary(" in HTML
    # 三个上传框都以同样的方式绑定点击 / 拖拽。
    for binding in ("bindDrop(drop, fileInput, setOverview, false);",
                    "bindDrop(dropDetails, detailsInput, setDetails, true);",
                    "bindDrop(dropSummary, summaryInput, setSummary, false);"):
        assert binding in HTML


def test_result_card_renders_ride_summary() -> None:
    # 概要字段以独立 stat 卡片展示（与置信度/里程并列），不埋在诊断信息里
    for needle in ("rsm.started_at", "rsm.duration_s", "rsm.distance_km",
                   "rsm.calories_kcal", "rsm.carbon_g",
                   "['开始时间'", "['骑行耗时'", "['概要里程'", "['卡路里'"):
        assert needle in HTML
    # 交叉校验结论留在诊断区，一致性/不一致各有明确文案
    assert "rsm.cross_check" in HTML
    assert "✓ 概要里程与轨迹里程一致" in HTML
    assert "里程量级不符" in HTML
