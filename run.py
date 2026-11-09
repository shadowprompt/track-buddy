"""启动入口。

用法：
    python run.py                      # 默认 127.0.0.1:8000
    python run.py --port 9000          # 指定端口
    python -m app.server --port 9000   # 等价写法（需先设 PYTHONPATH=src）
    python run.py --reload              # 开发模式（代码热重载）

⚠️ ``app`` 包在 ``src/app/``，导入它需要把 ``src/`` 放进 ``sys.path``。
这里在 import 之前就设好，因此**不需要**用户手动设 ``PYTHONPATH``——
直接 ``python run.py`` 就能跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from app.server import main as _main

    _main()


if __name__ == "__main__":
    main()
