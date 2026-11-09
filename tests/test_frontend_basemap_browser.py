"""真实浏览器验证：阶段 3 的底图适配层在页面上确实工作。

检查四件事：
1. 页面加载后适配层已就绪（``RideBasemap`` 存在、provider 正确）；
2. 识别一次后地图上真的有轨迹图层（适配层建图 + 画线成功）；
3. 适配层的 provider 名与服务端注入的一致；
4. 无 JS 异常（尤其不是 "Unhandled error. (undefined)"）。

这是阶段 3 最关键的一条验证：前面的 Node 测试都跑在**桩**上，
只有真实浏览器能证明「腾讯 GL JS + 适配层」这条链路真的通。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().parent / "verify_basemap_browser.py"


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node.js")
def test_basemap_adapter_in_real_browser() -> None:
    """真实浏览器验证。

    ⚠️ 部分沙箱环境**不允许启动浏览器进程**（Playwright driver 连接被
    直接切断）。此时脚本会打印「浏览器无法启动，跳过」并以 0 退出——
    这是环境限制，不是代码问题。CI/本地有浏览器时才会真正执行断言。
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT),
    )
    out = proc.stdout + proc.stderr
    if "浏览器无法启动" in out:
        pytest.skip("当前环境不允许启动浏览器进程")
    assert proc.returncode == 0, (
        "真实浏览器验证失败:\n" + proc.stdout[-3000:]
        + "\n--- stderr ---\n" + proc.stderr[-1500:]
    )
