"""能力③「道路几何」的 OSM/Overpass provider。

这个 provider 没有密钥：公共 Overpass 实例是开放服务。它仍然要走注册表，
是为了让 ``MAP_ROAD_GEOMETRY_ORDER`` 与将来可能的自建 Overpass / 商业路网
实现共用同一套装配与回退语义，而不是在业务代码里写死 ``urllib``。

``output_crs = WGS84`` 是硬声明：OSM 的经纬度不是 GCJ-02，任何消费方
都必须经 ``contracts.to_gcj02`` 换算，避免漏一次就整体偏约 500 m。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ..contracts import (
    CRS,
    Capability,
    ProbeResult,
    ProviderMeta,
)
from .overpass import DEFAULT_USER_AGENT, OVERPASS_MIRRORS, overpass_query

_ENABLED_ENV = "MAP_ROAD_GEOMETRY_ENABLED"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})


class OsmRoadGeometryProvider:
    name = "osm"

    def __init__(
        self,
        cfg: Any | None = None,
        credentials: Any | None = None,
        *,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.credentials = credentials
        self._opener = opener
        self.meta = ProviderMeta(
            name=self.name,
            capability=Capability.ROAD_GEOMETRY,
            output_crs=CRS.WGS84,
            needs_network=True,
        )

    def _env(self, name: str) -> str | None:
        getter = getattr(self.cfg, "_env_value", None)
        if callable(getter):
            return getter(name)
        return os.environ.get(name)

    def is_available(self) -> bool:
        """公共源无需凭证；可用 ``MAP_ROAD_GEOMETRY_ENABLED=0`` 整体关闭。"""
        raw = self._env(_ENABLED_ENV)
        return raw is None or str(raw).strip().lower() not in _DISABLED_VALUES

    def probe(self) -> ProbeResult:
        """无凭证可验，因此只做静态可达性声明。

        真正的可用性在首次查询时由镜像轮换体现；这里刻意不发网络请求，
        避免 ``/api/health`` 因为公共 Overpass 繁忙而卡住几十秒。
        """
        if not self.is_available():
            return ProbeResult(False, error="已在配置中禁用")
        return ProbeResult(
            True, hint="公共 Overpass，无需凭证；实际可用性在查询时探测"
        )

    def query(self, ql: str, *, timeout: float = 90.0,
              user_agent: str = DEFAULT_USER_AGENT,
              retry_pause: float = 0.0) -> dict[str, Any]:
        return overpass_query(
            ql, timeout=timeout, user_agent=user_agent,
            mirrors=OVERPASS_MIRRORS, retry_pause=retry_pause,
            opener=self._opener,
        )
