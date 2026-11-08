"""能力⑥：静态地图素材（实验与合成测试用）。

对外只暴露 ``fetch_static_map`` 与 ``amap_key``；
``tools/`` 下的三个实验脚本共用这一份实现，不再各抄一份 URL 拼接。
"""

from __future__ import annotations

from .provider import (
    AMAP_STATICMAP,
    TMAP_STATICMAP,
    StaticMap,
    amap_key,
    fetch_static_map,
)

__all__ = [
    "AMAP_STATICMAP",
    "TMAP_STATICMAP",
    "StaticMap",
    "amap_key",
    "fetch_static_map",
]
