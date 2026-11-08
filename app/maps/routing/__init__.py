"""能力④：路线几何交叉校验（OSRM / Valhalla）。

对外提供：

* ``build_route_validation_registry`` —— 按 ``MAP_ROUTE_VALIDATION_ORDER``
  装配 provider，默认 ``osrm,valhalla``；
* ``common`` 里的共享工具 —— polyline 解码、命中率、顺序一致性、
  折线长度、稳定缓存键与磁盘缓存，两个客户端不再各写一份。

配置：``MAP_ROUTE_VALIDATION_ORDER``（顺序）、
``MAP_ROUTE_VALIDATION_ENABLED=0``（整体关闭）。
"""

from __future__ import annotations

from typing import Any

from ..contracts import Capability
from ..credentials import CredentialStore
from ..registry import ProviderRegistry
from .providers import OsrmRouteProvider, ValhallaRouteProvider

__all__ = [
    "OsrmRouteProvider",
    "ValhallaRouteProvider",
    "build_route_validation_registry",
    "order",
]

_PROVIDER_CLASSES = (OsrmRouteProvider, ValhallaRouteProvider)


def build_route_validation_registry(
    cfg: Any, credentials: CredentialStore | None = None,
) -> tuple[ProviderRegistry, dict[str, Any]]:
    """装配路线校验能力注册表。"""
    creds = credentials or CredentialStore(cfg)
    reg = ProviderRegistry(cfg, creds)
    impls: dict[str, Any] = {}
    for cls in _PROVIDER_CLASSES:
        provider = cls(cfg, creds)
        reg.register(provider)
        impls[provider.name] = provider
    return reg, impls


def order(cfg: Any) -> tuple[str, ...]:
    """当前生效的调用顺序，供 health 直接上报。"""
    reg, _ = build_route_validation_registry(cfg)
    return reg.order(Capability.ROUTE_VALIDATION)
