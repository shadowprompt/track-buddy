"""能力③：道路几何（OSM / Overpass）。

上层只使用 ``overpass_query`` / ``overpass_raw`` 与
``build_road_geometry_registry``；镜像表、轮换、超时、UA 只在这里维护。
配置顺序为 ``MAP_ROAD_GEOMETRY_ORDER``，整体开关为
``MAP_ROAD_GEOMETRY_ENABLED``。
"""

from __future__ import annotations

from typing import Any

from ..contracts import Capability
from ..credentials import CredentialStore
from ..registry import ProviderRegistry
from .overpass import (
    DEFAULT_USER_AGENT,
    OVERPASS_MIRRORS,
    overpass_query,
    overpass_raw,
)
from .provider import OsmRoadGeometryProvider

__all__ = [
    "OVERPASS_MIRRORS",
    "DEFAULT_USER_AGENT",
    "overpass_query",
    "overpass_raw",
    "OsmRoadGeometryProvider",
    "build_road_geometry_registry",
]


def build_road_geometry_registry(
    cfg: Any, credentials: CredentialStore | None = None,
    *, opener: Any | None = None,
) -> tuple[ProviderRegistry, dict[str, Any]]:
    """装配道路几何能力注册表（当前只有公共 OSM 源）。"""
    creds = credentials or CredentialStore(cfg)
    reg = ProviderRegistry(cfg, creds)
    impl = OsmRoadGeometryProvider(cfg, creds, opener=opener)
    reg.register(impl)
    return reg, {impl.name: impl}


def order(cfg: Any) -> tuple[str, ...]:
    """当前生效的调用顺序，供 health 直接上报。"""
    reg, _ = build_road_geometry_registry(cfg)
    return reg.order(Capability.ROAD_GEOMETRY)
