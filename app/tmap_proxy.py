"""腾讯地图 GL JS 本地代理。

为什么需要
----------
腾讯地图 GL JS 在加载时必须完成**鉴权**：向 ``apikey.map.qq.com`` 发起
``checkKey`` 请求，拿到 ``subkey`` 后才能请求瓦片与服务。

浏览器直接请求这些接口会把 key 暴露在前端源码里。合规的做法是让浏览器
只跟本地服务说话，由本地服务代为转发并在转发时注入 key。

但这个代理**原本只有占位符、从未真正实现**::

    serviceHost: 'http://127.0.0.1:__WB_HTTP_PORT__/_TMapService/_wbt/__WB_TMAP_SECRET__'

占位符原样发给了浏览器，SDK 于是去请求一个非法地址
（``__WB_HTTP_PORT__`` 不是一个合法端口），鉴权必然失败。

SDK 内部的失败路径是::

    console.error("鉴权失败：网络请求失败")   →  AUTH_ERROR.netError
    → 触发 EventEmitter 的 'error' 事件，而该事件没有监听器
    → Node 风格的标准行为：throw new Error("Unhandled error. (undefined)")

于是页面报「识别失败：Unhandled error. (undefined)」。
注意这个错误**与识别本身无关**——服务端返回的是 200，异常发生在
识别完成之后的地图渲染阶段。

本模块实现真实的代理，并把端口与密钥在服务端渲染进页面，
使 key 不出现在浏览器可见的静态文件里。
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

# SDK 会请求这两个路径，全部转发到腾讯地图上游。
#
# 查 SDK 源码（map.qq.com/api/gljs）里 serviceHost 的用法可知：
#   - ``serviceHost + "/checkKey?"``  →  apikey.map.qq.com/mkey/index.php/mkey/check
#   - ``serviceHost + "/oversea?"``  →  overseactrl.map.qq.com/v4
#
# 底图瓦片**不走** serviceHost——它们来自 p{s}.map.gtimg.com 等 CDN，
# 是公开地址，浏览器直连即可，无需代理（也不需要 key）。
# 所以代理只需覆盖这两个鉴权/配置类请求。
PROXY_PATHS = {
    "checkKey": "https://apikey.map.qq.com/mkey/index.php/mkey/check",
    "oversea": "https://overseactrl.map.qq.com/v4",
    # SDK 的性能上报（pingd），走 serviceHost 且不带路径后缀。
    # 它只是埋点，失败无影响，但转发过去可以避免浏览器报 CORS 错误。
    "": "https://pr.map.qq.com/pingd?appid=jsapi_v3",
    "ping": "https://pr.map.qq.com/pingd?appid=jsapi_v3",
}

# 页面里的占位符 —— 服务端渲染首页时替换
PLACEHOLDER_PORT = "__WB_HTTP_PORT__"
PLACEHOLDER_SECRET = "__WB_TMAP_SECRET__"


def _inject_key(url: str, key: str | None) -> str:
    """把 key 注入到请求参数里。

    SDK 期望参数名叫 ``key``（或 ``key2``）。代理在此注入，
    浏览器永远看不到 key。
    """
    if not key:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}key={key}"


def create_proxy_router(
    tmap_key: str | None,
    http_timeout: float = 15.0,
) -> APIRouter:
    """创建代理路由。

    URL 形如 ``/_TMapService/_wbt/<secret>/checkKey?...``：
    ``_wbt/<secret>`` 之后的段才是真正的目标路径（checkKey / service / oversea）。
    用 ``requests`` 而非 ``httpx``——项目依赖里没有 httpx，
    没必要为这一个功能引入新依赖。同步请求放进线程池执行，不阻塞事件循环。
    """
    import requests
    from starlette.concurrency import run_in_threadpool

    router = APIRouter(prefix="/_TMapService", tags=["tmap-proxy"])
    session = requests.Session()
    session.headers.update({"User-Agent": "meituan-ride/0.1 (local proxy)"})

    def _fetch(url: str, params: dict):
        return session.get(url, params=params, timeout=http_timeout)

    @router.api_route("/_wbt/{secret}", methods=["GET", "POST"])
    @router.api_route("/_wbt/{secret}/", methods=["GET", "POST"])
    @router.api_route("/_wbt/{secret}/{path:path}", methods=["GET", "POST"])
    async def proxy(secret: str, request: Request, path: str = "") -> Response:
        # SDK 的 serviceHost 拼接方式有两种：
        #   serviceHost + "/checkKey?..."   → 带路径
        #   serviceHost + "?..."           → 不带路径（pingd 等）
        # 后者会被 FastAPI 重定向到带斜杠的版本，因此两个形式都要注册。
        upstream = PROXY_PATHS.get(path) if path else PROXY_PATHS.get("")
        if upstream is None:
            # 未识别的路径按 pingd 处理——它是 SDK 的性能上报，无害。
            upstream = PROXY_PATHS.get("ping")

        # 原样转发查询参数
        params = dict(request.query_params)
        # SDK 的 checkKey 会带上 key 与 key2（子 key），代理统一用配置里的 key
        params.pop("key", None)
        params.pop("key2", None)
        if tmap_key:
            params["key"] = tmap_key

        try:
            resp = await run_in_threadpool(_fetch, upstream, params)
        except Exception:  # noqa: BLE001 — 上游失败不应让前端崩掉
            # 返回 SDK 期望的失败结构，它会转成 AUTH_ERROR 并给出可读提示
            return Response(
                content='mkeycb&&mkeycb({"info":{"error":-1,"msg":"NETWORK_ERROR"},"detail":{}})',
                media_type="application/javascript",
                headers={"Access-Control-Allow-Origin": "*"},
            )

        ctype = resp.headers.get("content-type", "application/javascript")
        # checkKey 返回的是 JSONP（mkeycb&&mkeycb({...})），
        # 必须原样透传——解析成 JSON 会让 SDK 认不出来。
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=ctype,
            headers={"Access-Control-Allow-Origin": "*"},
        )

    return router


def render_page(html: str, port: int, secret: str) -> str:
    """把占位符替换成实际值。

    ``secret`` 是给代理路径用的标记——浏览器请求 ``/_TMapService/_wbt/<secret>/...``，
    服务端据此识别出这是代理流量。真正的腾讯地图 key 不出现在页面里。
    """
    html = html.replace(PLACEHOLDER_PORT, str(port))
    html = html.replace(PLACEHOLDER_SECRET, secret)
    return html


def make_secret() -> str:
    """生成一个随机的代理标记。"""
    import secrets as _secrets

    return _secrets.token_urlsafe(12)
