"""异步精修结果必须真正刷新地图（pytest 入口）。

`frontend_map_refresh_test.js` 用最小 TMap 桩跑页面里真实的
`drawMapInner()` 两次，断言第二次复用图层并换掉几何；
这里把它接进全量回归，避免只跑后端测试时漏掉纯前端回归。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "src" / "web" / "index.html"
SCRIPT = Path(__file__).resolve().parent / "frontend_map_refresh_test.js"


def test_async_refine_refreshes_map() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("未安装 Node.js，跳过前端逻辑测试")
    proc = subprocess.run(
        [node, str(SCRIPT), str(HTML)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, (
        "前端地图刷新测试失败:\n"
        + proc.stdout[-2500:]
        + "\n--- stderr ---\n"
        + proc.stderr[-1200:]
    )
