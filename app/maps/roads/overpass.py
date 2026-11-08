"""能力③的统一 Overpass 请求层。

历史问题：``roadfit.py`` / ``roadname_constraint.py`` / ``waterguard.py``
各维护一份镜像列表与轮换逻辑，三份列表还不一致（waterguard 里甚至有
``interparser`` 拼写错误的镜像）。任意一处修镜像或超时策略，另外两处
都不会跟着变。

现在只有这一份实现，调用方只传「查什么」和「自己的超时策略」：

* ``overpass_raw``  返回原始文本，给需要自己解析的调用方；
* ``overpass_query`` 返回解析好的 dict。

主站最可靠，因此第一个镜像拿完整超时，备用镜像只拿
``max(10, timeout / 镜像数)``，快速失败换下一个而不是死等；需要
每个镜像都拿完整超时的调用方（水系岸线）可显式传
``per_mirror_timeout``。
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

# 唯一一份公共镜像表。改这里，三个调用点同时生效。
OVERPASS_MIRRORS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
)

DEFAULT_USER_AGENT = "meituan-ride/0.1"


def _request(
    query: str,
    *,
    as_json: bool,
    timeout: float,
    user_agent: str,
    mirrors: tuple[str, ...] | list[str],
    per_mirror_timeout: float | None,
    retry_pause: float,
    opener: Callable[..., Any] | None,
) -> Any:
    data = urllib.parse.urlencode({"data": query}).encode()
    urlopen = opener or urllib.request.urlopen
    count = max(1, len(mirrors))
    last: Exception | None = None
    for i, mirror in enumerate(mirrors):
        if per_mirror_timeout is not None:
            per = float(per_mirror_timeout)
        elif i == 0:
            per = float(timeout)
        else:
            per = max(10.0, float(timeout) / count)
        req = urllib.request.Request(
            mirror, data=data, headers={"User-Agent": user_agent}
        )
        try:
            text = urlopen(req, timeout=per).read().decode("utf-8")
            # Overpass 繁忙时返回 HTML 网关错误页而不是 JSON；
            # 不先挡一道，后面只会看到一个莫名其妙的 JSONDecodeError。
            if text.lstrip()[:1] != "{":
                raise ValueError("响应不是 JSON（镜像返回了网关错误页）")
            # 解析也放在 try 内：畸形 JSON 同样应该换下一个镜像。
            return json.loads(text) if as_json else text
        except Exception as exc:  # noqa: BLE001 - 换下一个镜像
            last = exc
            if retry_pause > 0:
                time.sleep(retry_pause)
    raise RuntimeError(f"Overpass 全部镜像失败：{last}")


def overpass_raw(
    query: str,
    *,
    timeout: float = 90.0,
    user_agent: str = DEFAULT_USER_AGENT,
    mirrors: tuple[str, ...] | list[str] = OVERPASS_MIRRORS,
    per_mirror_timeout: float | None = None,
    retry_pause: float = 0.0,
    opener: Callable[..., Any] | None = None,
) -> str:
    """返回原始 JSON 文本（调用方自行 ``json.loads``）。"""
    return str(_request(
        query, as_json=False, timeout=timeout, user_agent=user_agent,
        mirrors=mirrors, per_mirror_timeout=per_mirror_timeout,
        retry_pause=retry_pause, opener=opener,
    ))


def overpass_query(
    query: str,
    *,
    timeout: float = 90.0,
    user_agent: str = DEFAULT_USER_AGENT,
    mirrors: tuple[str, ...] | list[str] = OVERPASS_MIRRORS,
    per_mirror_timeout: float | None = None,
    retry_pause: float = 0.0,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """返回解析后的 Overpass JSON；全镜像失败抛 ``RuntimeError``。"""
    data = _request(
        query, as_json=True, timeout=timeout, user_agent=user_agent,
        mirrors=mirrors, per_mirror_timeout=per_mirror_timeout,
        retry_pause=retry_pause, opener=opener,
    )
    return data if isinstance(data, dict) else {}
