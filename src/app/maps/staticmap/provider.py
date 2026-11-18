"""能力⑥「静态地图素材」的 provider。

静态地图是**离线实验与合成测试**用的底图素材：把经纬度渲染成 PNG，
用来做像素级特征匹配、轨迹真值合成、多图拼接实验。

历史问题：``tools/exp_basemap_match.py`` / ``tools/exp_e2e_multishot.py`` /
``tools/exp_overlap_need.py`` 各自抄了一份「拼 URL + urllib 取图 + 解码」
的代码。三份的 URL 参数顺序、size 拼法、错误处理都不一样，
改一次瓦片参数要动三个文件。

现在只有这一份实现：``app/maps/staticmap/``。
调用方只说「我要哪个中心、哪个缩放级别、多大图」。

**注意**：静态地图属于**能力层之外的实验素材**，不是识别链路的依赖。
识别流程不依赖它，删掉整个包也不影响 `app/server.py` 的可用性。
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from ...amap_relay.client import rewrite_amap_request

# 腾讯地图静态地图 WebService 端点
TMAP_STATICMAP = "https://apis.map.qq.com/ws/staticmap/v2/"
# 高德静态地图 WebService 端点
AMAP_STATICMAP = "https://restapi.amap.com/v3/staticmap"

DEFAULT_USER_AGENT = "meituan-ride/0.1"


@dataclass
class StaticMap:
    """一张静态地图。

    ``data`` 是 PNG 字节；``provider`` 记录来源，便于实验记录溯源。
    """

    data: bytes
    provider: str
    url: str


def _amap_url(
    lon: float, lat: float, zoom: int, size: int, key: str,
    markers: str | None = None,
) -> str:
    params = {
        "location": f"{lon},{lat}",
        "zoom": str(zoom),
        "size": f"{size}*{size}",
        "key": key,
    }
    if markers:
        params["markers"] = markers
    return AMAP_STATICMAP + "?" + urllib.parse.urlencode(params)


def _tmap_url(
    lon: float, lat: float, zoom: int, size: int, key: str,
    markers: str | None = None,
) -> str:
    # 腾讯静态地图用 center + 像素级 zoom，
    # zoom 需要按 256px 瓦片换算成 level。
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "width": str(size),
        "height": str(size),
        "key": key,
    }
    if markers:
        params["markers"] = markers
    return TMAP_STATICMAP + "?" + urllib.parse.urlencode(params)


def fetch_static_map(
    lon: float,
    lat: float,
    zoom: int,
    size: int = 1024,
    *,
    provider: str = "amap",
    key: str | None = None,
    markers: str | None = None,
    timeout: float = 30.0,
    opener: Callable[..., Any] | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> StaticMap | None:
    """取一张静态地图；失败返回 ``None``（实验脚本据此跳过）。

    ``provider`` 目前支持 ``amap`` 与 ``tmap``。腾讯的 zoom 语义与高德
    不同（腾讯用 level，高德用 zoom），这里统一按「高德的 zoom 语义」
    处理：腾讯侧按 256px 瓦片换算成 level。
    """
    if not key:
        return None
    if provider == "tmap":
        url = _tmap_url(lon, lat, zoom, size, key, markers)
    else:
        # _amap_url 产出的是「带 query 的完整 URL」；rewrite 会拆开
        # query、丢弃占位 key 并改写到中转（未配中转时原样返回）。
        url, q = rewrite_amap_request(_amap_url(lon, lat, zoom, size, key, markers))
        if q:
            url = url + "?" + urllib.parse.urlencode(q)
    urlopen = opener or urllib.request.urlopen
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception:  # noqa: BLE001 - 实验脚本只需知道「拿不到」
        return None
    if not data:
        return None
    return StaticMap(data=data, provider=provider, url=url)


def amap_key(cfg: Any | None = None) -> str | None:
    """从配置里取高德 key（供实验脚本使用）。"""
    if cfg is not None:
        getter = getattr(cfg, "amap_key", None)
        if callable(getter):
            return getter()
    return os.environ.get("AMAP_KEY") or None
