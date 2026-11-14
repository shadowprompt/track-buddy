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
import threading
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _warmup_ocr_async() -> None:
    """后台预热 OCR，不阻塞 uvicorn 起监听。

    ⚠️ 只在真实启动入口做这件事，绝不放进 ``app.server`` 的模块导入或
    FastAPI startup 事件：pytest 里的 TestClient 会触发 startup，那样
    几百个单元测试每次都要加载一遍 onnxruntime 模型。
    """
    def _run() -> None:
        try:
            from app.ocr_anchor import warmup_ocr

            warmup_ocr()
        except Exception:  # noqa: BLE001 — 预热失败不影响服务
            pass

    threading.Thread(target=_run, name="ocr-warmup", daemon=True).start()


def main() -> None:
    from app.server import main as _main

    _warmup_ocr_async()
    _main()


if __name__ == "__main__":
    main()
