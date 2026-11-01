"""启动入口。

用法：
    python run.py                      # 默认 127.0.0.1:8000
    python run.py --port 9000          # 指定端口
    python -m app.server --port 9000   # 等价写法
    python run.py --reload              # 开发模式（代码热重载）
"""
from __future__ import annotations


def main() -> None:
    from app.server import main as _main

    _main()


if __name__ == "__main__":
    main()
