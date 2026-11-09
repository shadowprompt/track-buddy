"""前端「贴路校正」开关与长任务提示的静态契约测试。

这些是纯前端行为：服务端日志全 200 也发现不了控件没接线。用静态断言
锁住三件事——开关只在多图生效、勾选才发 ``refine=on``、轮询能识别
后端返回的 ``elapsed_s`` 与 verify 档。
"""

from __future__ import annotations

from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "src" / "web" / "index.html"


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_road_fit_checkbox_exists() -> None:
    h = _html()
    assert 'id="roadFit"' in h
    assert 'type="checkbox" id="roadFit"' in h
    assert "多图拼接只做" in h and "不平移" in h


def test_switch_only_affects_multi_shot() -> None:
    h = _html()
    assert "const syncFit = !!(multi && $('roadFit') && $('roadFit').checked);" in h
    assert "if (syncFit) fd.append('refine', 'on');" in h


def test_poll_handles_verify_mode_and_long_tasks() -> None:
    h = _html()
    assert "rf.mode === 'verify'" in h
    assert "Number(j.elapsed_s || 0) > 180" in h
    assert "refineWarned" in h
    # verify 分支必须排在通用 rf.ok 分支之前，否则永远走不到
    assert h.index("rf.mode === 'verify'") < h.index("} else if (rf.ok) {")
