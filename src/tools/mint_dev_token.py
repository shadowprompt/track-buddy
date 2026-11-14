"""本地验收用：签发一个运动记录测试 token（**不要用于生产**）。

小程序尚未接入 wx.login 时，用它在 curl / Postman 里验证真实 COS 链路：
它只读取服务端 .env 的 ``RECORDS_TOKEN_SECRET``，不接触 COS、不联网，
也不打印密钥。任何能读到 .env 的人本来就能签发 token，因此仅限本机使用。

用法（工具与其它脚本一样放在 src/tools/ 下）：
    D:\\soft\\Python311\\python.exe src\\tools\\mint_dev_token.py --label devA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/tools/ → 仓库根
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app import records  # noqa: E402
from app.config import CONFIG  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="签发运动记录测试 token")
    ap.add_argument("--label", default="dev-test-user",
                    help="测试身份标签（经同一 HMAC 派生为 uid）")
    ap.add_argument("--ttl", type=int, default=None,
                    help="有效期秒数，默认取 RECORDS_TOKEN_TTL")
    args = ap.parse_args()

    secret = CONFIG.records_token_secret
    if not secret:
        raise SystemExit("未配置 RECORDS_TOKEN_SECRET：请先在 .env 写入后再试")
    uid = records.uid_from_openid(args.label, secret)
    ttl = args.ttl or CONFIG.records_token_ttl
    print(records.issue_token(uid, secret=secret, ttl=ttl))


if __name__ == "__main__":
    main()
