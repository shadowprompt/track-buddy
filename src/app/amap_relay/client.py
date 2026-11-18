"""本地侧的高德请求改写：把 ``restapi.amap.com`` 改走服务器中转。

只做 **纯函数式的 URL/参数改写**，不发任何网络请求——调用方仍然用
自己原来的 ``requests.get(url, params=...)``，只是 url 与 params 先
过一遍这里的函数。

用法（调用方模式，全项目统一）::

    from ...amap_relay.client import amap_relay_enabled, rewrite_amap_request

    if not key and not amap_relay_enabled(cfg):
        return []                      # 没本地 key 也没中转 → provider 不可用
    url, params = rewrite_amap_request(url, params, cfg)
    resp = session.get(url, params=params, ...)

改写语义（配置了 ``AMAP_RELAY_URL`` 时）：
  - ``https://restapi.amap.com/v3/place/text`` → ``<relay>/amap/v3/place/text``
  - 丢弃 ``key`` 参数（本地的是占位符或没有，真实 key 由中转注入）；
  - 追加 ``relay_token``（服务器侧校验用，不转发给高德）。

未配置中转时原样返回——**线上服务器路径零改动**。
"""

from __future__ import annotations

import urllib.parse
from typing import Any

# 只有发往这个 host 的请求才改写；其它（腾讯/天地图/OSM）不受影响。
AMAP_HOST_MARK = "restapi.amap.com"

# 模块级惰性单例：Config() 本身很轻，但每次调用都新建会丢掉 .env
# 的 mtime 指纹缓存，这里缓存一个实例（指纹机制仍在实例内生效）。
_default_cfg: Any = None


def _resolve_cfg(cfg: Any | None) -> Any:
    """优先用调用方传入的 cfg；没传则用模块级默认实例。"""
    global _default_cfg
    if cfg is not None:
        return cfg
    if _default_cfg is None:
        from ..config import AppConfig

        _default_cfg = AppConfig()
    return _default_cfg


def amap_relay_enabled(cfg: Any | None = None) -> bool:
    """是否配置了中转（``AMAP_RELAY_URL`` 非空）。

    provider 的 ``is_available()`` 用它兜底：本地没有真实 AMAP_KEY、
    只有中转时，provider 仍应视为可用。
    """
    try:
        return bool(_resolve_cfg(cfg).amap_relay_url)
    except Exception:  # noqa: BLE001 — 配置读不到就当没配，不影响原逻辑
        return False


def rewrite_amap_request(
    url: str,
    params: dict[str, Any] | None = None,
    cfg: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """按需改写一条高德请求。

    ``url`` 可以是不带 query 的端点（配合 ``params`` 使用，项目内主流
    写法），也可以是已拼好 query 的完整 URL（staticmap 的写法）——
    两种都会被统一拆解、合并、改写。

    返回 ``(url, params)``：url 不含 query，query 全部进 params 字典，
    调用方用 ``requests.get(url, params=params)`` 即可。
    未配置中转或 url 不是高德时**原样返回**（params 为 None 时给空
    字典，调用方直接透传也无碍）。
    """
    relay = (_resolve_cfg(cfg).amap_relay_url or "").rstrip("/")
    if not relay or AMAP_HOST_MARK not in url:
        return url, params or {}

    parsed = urllib.parse.urlsplit(url)
    # URL 里已带的 query（staticmap 风格）与 params 字典合并，
    # params 优先——与「requests 里两者都传」的语义一致。
    merged: dict[str, Any] = dict(
        urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    if params:
        merged.update({k: v for k, v in params.items() if v is not None})
    # 本地 key 是占位符（甚至没有），绝不把它发出去；真实 key 由
    # 中转在服务器侧注入。
    merged.pop("key", None)
    token = _resolve_cfg(cfg).amap_relay_token
    if token:
        merged["relay_token"] = token

    relay_url = f"{relay}/amap/{parsed.path.lstrip('/')}"
    return relay_url, merged
