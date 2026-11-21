"""OCR 文本有效性过滤层的单元测试（可插拔接口）。

守住的契约：
  * 默认规则实现拦得住 UI 文案/数值/OCR 碎片/泛词，且不误杀真实地名；
  * 接口可替换：注册自定义实现、环境变量切换、构造失败回落默认；
  * 旧的函数名（is_ui_noise/is_ocr_garbage）作为兼容别名仍可用。
"""
from __future__ import annotations

import pytest

from app.anchor_filter import (
    ANCHOR_FILTER_ENV,
    AnchorFilter,
    RuleFilter,
    get_filter,
    is_ocr_garbage,
    is_ui_noise,
    register_filter,
)
from app.config import AppConfig


class _AlwaysReject:
    """测试桩：丢弃一切文本。"""

    name = "always_reject"

    def reject(self, text: str) -> bool:
        return True


class _BrokenFilter:
    """测试桩：构造即失败，用于验证回落。"""

    def __init__(self) -> None:
        raise RuntimeError("boom")

    def reject(self, text: str) -> bool:
        return False


def test_default_is_rule_filter() -> None:
    f = get_filter("rule")
    assert isinstance(f, RuleFilter)
    assert isinstance(f, AnchorFilter)


@pytest.mark.parametrize("text", [
    "距离", "速度", "运动时间", "全程均速", "累计上升", "海拔范围",
    "导出删除分享", "中国移动", "34.63", "15.67km/h", "655米",
    "A.十", "y.十层", "34.67新=江18.29", "大厦", "某路出入口", "中",
    "",
])
def test_rule_filter_rejects_noise(text: str) -> None:
    assert get_filter("rule").reject(text), f"噪声未被拦截: {text!r}"


@pytest.mark.parametrize("text", [
    "宁波站", "鄞州区政府", "钟公庙村", "丽兹大酒店", "迪士尼", "东柳",
    "上海荟聚Livat", "7-Eleven", "G2", "天一阁",
])
def test_rule_filter_keeps_real_places(text: str) -> None:
    assert not get_filter("rule").reject(text), f"真实地名被误杀: {text!r}"


def test_compat_aliases_still_work() -> None:
    assert is_ui_noise("距离") is True
    assert is_ui_noise("宁波站") is False
    assert is_ocr_garbage("A.十") is True
    assert is_ocr_garbage("迪士尼") is False


def test_custom_filter_can_be_registered_and_selected(monkeypatch) -> None:
    register_filter("always_reject", _AlwaysReject)
    # 显式名字优先
    assert isinstance(get_filter("always_reject"), _AlwaysReject)
    assert get_filter("always_reject").reject("宁波站") is True
    # 环境变量也能选（monkeypatch 隔离，不污染其它测试）
    monkeypatch.setenv(ANCHOR_FILTER_ENV, "always_reject")
    assert isinstance(get_filter(), _AlwaysReject)


def test_broken_filter_falls_back_to_rule(monkeypatch) -> None:
    register_filter("broken", _BrokenFilter)
    assert isinstance(get_filter("broken"), RuleFilter)
    monkeypatch.setenv(ANCHOR_FILTER_ENV, "broken")
    assert isinstance(get_filter(), RuleFilter)


def test_unknown_filter_falls_back_to_rule() -> None:
    assert isinstance(get_filter("no_such_filter"), RuleFilter)


def test_config_has_filter_name_field() -> None:
    assert AppConfig().anchor_filter == "rule"
