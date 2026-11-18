"""高德 WebService 中转服务（部署在服务器上）。

为什么需要
----------
高额度的高德 Web 服务 key 只对**服务器出口 IP** 开放（IP 白名单）。
本地开发机直接调高德会被拒，而真实 key 也不应该发到本地。

做法：本服务跑在服务器上，本地只跟它说话::

    本地代码 → POST/GET /amap/v3/place/text?...&relay_token=xxx
             → 本服务注入 AMAP_KEY → restapi.amap.com

安全设计（改之前先想清楚）：
  1. **令牌必填**：``AMAP_RELAY_TOKEN`` 未配置时拒绝所有请求
     （fail closed）——这个服务暴露在公网上，没令牌等于把额度送人。
  2. **上游固定**：只转发 ``https://restapi.amap.com``，路径必须
     ``v<数字>/`` 开头，不给任何 SSRF 空间。
  3. **key 只在服务端**：丢弃客户端带来的 ``key`` 参数，注入服务器
     ``.env`` 里的 ``AMAP_KEY``（沿用 app.config 的热读，改 .env
     不用重启）。
  4. CORS 全开——这是给本地开发调试用的，浏览器直连时需要。

部署见仓库根 ``ecosystem.relay.config.js`` 与 ``docs/amap_relay.md``。
"""

from __future__ import annotations

import logging
import re
import secrets as _secrets
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("amap-relay")

# 上游固定为高德 WebService——不做成可配置项，防止误配成任意地址。
UPSTREAM = "https://restapi.amap.com"

# 只允许 v3/v4 等 WebService 版本路径，其余（探活路径、异常路径）拒绝。
_PATH_RE = re.compile(r"^v\d+(/|$)")

# 高德偶发慢（路径规划 5s+），给足余量；本服务是同步 requests + 线程池，
# 超时只是释放线程，不会拖垮 uvicorn。
HTTP_TIMEOUT = 20.0


def _json(info: str, status_code: int) -> JSONResponse:
    """按高德 WebService 的响应风格返回错误（status=0 + info）。"""
    return JSONResponse(
        {"status": "0", "info": info, "infocode": "RELAY"},
        status_code=status_code,
        headers={"Access-Control-Allow-Origin": "*"},
    )


def create_app() -> FastAPI:
    """构建中转应用。

    复用 ``app.config.Config`` 的 .env 热读（mtime+size+sha256 指纹），
    服务器上改 .env 换 key / 换令牌立即生效，无需重启。
    Config 本体是纯 stdlib dataclass，import 没有重依赖。
    """
    from ..config import AppConfig

    cfg = AppConfig()

    app = FastAPI(
        title="amap-relay",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,  # 不对外暴露接口文档
    )
    # 本地开发是浏览器 + 本地脚本混用，CORS 全开即可；
    # 鉴权靠令牌而不是源，跨域放开不放大风险。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 会话级连接池：中转是高频小请求，复用 TCP 连接省 TLS 握手。
    # 挂在 app.state 上而非闭包局部变量——测试里可以直接换替身。
    import requests as _requests

    app.state.session = _requests.Session()
    app.state.session.headers.update(
        {"User-Agent": "track-buddy-amap-relay/0.1"})

    def _authorized(request: Request) -> bool:
        """校验访问令牌：header ``X-Relay-Token`` 或 query ``relay_token``。

        令牌未配置时一律拒绝——fail closed，宁可不可用也不裸奔。
        """
        token = cfg.amap_relay_token
        if not token:
            return False
        supplied = (
            request.headers.get("x-relay-token")
            or request.query_params.get("relay_token")
            or ""
        )
        return bool(supplied) and _secrets.compare_digest(supplied, token)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """探活：只暴露「配没配」，绝不暴露值本身。"""
        return {
            "ok": True,
            "key_configured": bool(cfg.amap_key),
            "token_configured": bool(cfg.amap_relay_token),
        }

    @app.api_route("/amap/{path:path}", methods=["GET", "POST"])
    async def relay(path: str, request: Request) -> Response:
        if not _PATH_RE.match(path):
            return _json("INVALID_RELAY_PATH", 403)
        if not _authorized(request):
            return _json("INVALID_RELAY_TOKEN", 401)
        key = cfg.amap_key
        if not key:
            # 令牌对了但服务器没配 key：这是部署问题，要显式暴露。
            return _json("RELAY_KEY_NOT_CONFIGURED", 503)

        params = dict(request.query_params)
        params.pop("relay_token", None)  # 令牌不出机房，不转发给高德
        # 高德的 key 是服务器侧凭据：客户端就算带了也一律丢弃。
        params.pop("key", None)
        params["key"] = key

        body: bytes | None = None
        headers: dict[str, str] | None = None
        if request.method == "POST":
            body = await request.body()
            ctype = request.headers.get("content-type")
            if ctype:
                headers = {"Content-Type": ctype}

        url = f"{UPSTREAM}/{path}"
        try:
            resp = await run_in_threadpool(
                request.app.state.session.request,
                request.method,
                url,
                params=params,
                data=body,
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 — 上游故障以高德风格返回
            logger.warning("upstream failed %s %s: %s",
                           request.method, path, type(exc).__name__)
            return _json("UPSTREAM_ERROR", 502)

        logger.info("%s %s -> %s", request.method, path, resp.status_code)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers={"Access-Control-Allow-Origin": "*"},
        )

    return app
