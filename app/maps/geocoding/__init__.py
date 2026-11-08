"""能力①：地名 -> 坐标。

上层只使用 ``build_geocoding_registry`` 与 ``convert_candidates``，
不再出现厂商名。厂商实现集中在 ``providers.py``，坐标系统一由
``contracts.to_gcj02`` 换算。
"""

from __future__ import annotations

from typing import Any, Callable

from ..contracts import PlaceCandidate
from ..credentials import CredentialStore
from ..registry import ProviderRegistry
from .providers import PROVIDER_CLASSES

__all__ = [
    "build_geocoding_registry",
    "convert_candidates",
    "PROVIDER_CLASSES",
]


def build_geocoding_registry(
    cfg: Any,
    session_getter: Callable[[], Any],
    lexicon: Any | None = None,
    credentials: CredentialStore | None = None,
) -> tuple[ProviderRegistry, dict[str, Any]]:
    """装配地理编码注册表，返回 (注册表, 名称 -> 实现)。

    顺序读 ``MAP_GEOCODING_ORDER``（兼容 ``GEOCODE_PROVIDER_ORDER``），
    未知名称在注册表里直接忽略，不会打断回退链。
    """
    creds = credentials or CredentialStore(cfg)
    reg = ProviderRegistry(cfg, creds)
    impls: dict[str, Any] = {}
    for cls in PROVIDER_CLASSES:
        provider = cls(cfg, creds, session_getter, lexicon)
        impls[provider.name] = provider
        reg.register(provider)
    return reg, impls


def convert_candidates(items: list[PlaceCandidate], result_cls: Any) -> list[Any]:
    """把契约层的 PlaceCandidate 转成管线使用的 GeocodeResult（GCJ-02）。

    ``result_cls`` 由调用方传入，避免本包反向依赖 ``app.geocode`` 的
    数据类定义。
    """
    return [
        result_cls(
            name=c.name, lon=c.lon, lat=c.lat, quality=c.quality,
            source=c.source, address=c.address, kind=c.kind,
        )
        for c in items
    ]
