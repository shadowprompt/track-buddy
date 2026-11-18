"""高德中转服务的启动入口（部署在服务器上，与主服务相互独立）。

用法：
    python run_relay.py                      # 默认 127.0.0.1:8849
    python run_relay.py --host 0.0.0.0       # 无 nginx 时直接对外
    python run_relay.py --port 9000          # 指定端口

为什么独立于 ``run.py``：主服务加载 OCR 模型、占内存且监听在
127.0.0.1:8848 给 nginx 反代；中转只是十来行转发逻辑的轻量服务，
单独一个 pm2 进程，互不影响（识别重启不打断本地调试）。

⚠️ 默认监听 **0.0.0.0**（2026-11-18 定稿）：本服务的部署形态就是
「本地开发机直连 8849」，必须对外才可达。安全由 AMAP_RELAY_TOKEN
兜底（未配置时拒绝一切转发）。若改走 nginx 反代，用环境变量收回去：
    RIDE_RELAY_HOST=127.0.0.1 pm2 start ecosystem.relay.config.js --update-env
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    import uvicorn

    from app.amap_relay.server import create_app

    parser = argparse.ArgumentParser(description="高德 WebService 中转服务")
    # 默认 0.0.0.0：本地开发机直连的部署形态（见文件头说明）。
    parser.add_argument("--host", default=os.environ.get("RIDE_RELAY_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("RIDE_RELAY_PORT", "8849")))
    args = parser.parse_args()

    uvicorn.run(create_app(), host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
