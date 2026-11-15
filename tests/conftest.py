"""pytest 全局配置：把 ``src/`` 加进 ``sys.path``。

目录布局（2026-11 起）::

    <repo>/            仓库根：data/ samples/ docs/ tests/ .env
      src/             源代码
        app/           Python 包（识别核心）
        web/           前端页面
        tools/         实验脚本

``app`` 包在 ``src/app/``，所以导入它需要把 ``src/`` 放进 ``sys.path``。
放在 conftest 里做，**所有测试文件都自动生效**，不必每个文件各写一遍
``sys.path.insert``——那种写法容易漏，也容易在目录调整时全网改一遍。

同时修掉了一个历史坏味道：``tests/test_endpointfit.py`` 等文件曾把
``r"D:\\Dev\\track-buddy"`` 硬编码进 ``sys.path``，换台机器就崩。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
