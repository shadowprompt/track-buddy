"""前端输入预检与独立校验展示的静态契约。

纯前端行为服务端日志发现不了：预检要在选图时触发、不能拦截上传、
识别结果要把三条与配准无关的外部证据显示出来。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parent.parent / "src" / "web" / "index.html"


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_precheck_element_and_hook():
    h = _html()
    assert 'id="precheck"' in h
    assert "runPrecheck(list);" in h
    assert "async function runPrecheck(list)" in h
    assert "readImageSize" in h
    assert "shortSide < 600" in h
    assert "20 * 1024 * 1024" in h
    assert "box.style.display = 'none'" in h


def test_precheck_is_advisory_only():
    """预检只提示，不能 disable 上传按钮或清空已选文件。"""
    h = _html()
    assert "go.disabled = !picked;" in h
    assert "不拦截上传" in h
    assert "建议使用未压缩的原图" in h


def test_result_shows_independent_checks():
    h = _html()
    assert "dg.scale_check && dg.scale_check.available" in h
    assert "dg.route_validation && dg.route_validation.available" in h
    assert "dg.input_quality && dg.input_quality.level !== 'ok'" in h
    assert "尺度独立校验" in h
    assert "外部路由校验" in h
    assert "输入质量" in h


def test_inline_script_parses():
    """整段内联脚本必须能被 JS 解析器接受（防止模板拼接漏括号）。"""
    if shutil.which("node") is None:
        pytest.skip("未安装 Node.js，跳过 JS 语法检查")
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", _html(), re.S)
    assert blocks
    code = "\n".join(blocks)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        cp = subprocess.run(["node", "--check", path],
                            capture_output=True, text=True, timeout=60)
        assert cp.returncode == 0, cp.stderr
    finally:
        Path(path).unlink(missing_ok=True)
