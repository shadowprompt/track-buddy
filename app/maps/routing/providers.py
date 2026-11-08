"""能力④「路线校验」的两个 provider。

OSRM 与 Valhalla 都是开放公共服务（无密钥），但它们是**两套独立实现**：
OSRM Demo 只有 driving 图，Valhalla 原生支持 bicycle/pedestrian/auto。
放进注册表后，``MAP_ROUTE_VALIDATION_ORDER`` 可以决定先试谁、或干脆
只留一个；将来接入自建服务也只是再加一个 provider。

两个客户端类仍留在 ``app/osrm.py`` / ``app/valhalla.py``（领域模型与
业务调用点在那里），这里用延迟导入做委托，避免包初始化环。
"""

from __future__ import annotations

import os
from typing import Any

from ..contracts import CRS, Capability, ProbeResult, ProviderMeta

_ENABLED_ENV = "MAP_ROUTE_VALIDATION_ENABLED"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})


class _BaseRouteProvider:
    name = ""

    def __init__(
        self, cfg: Any | None = None, credentials: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.credentials = credentials
        self.meta = ProviderMeta(
            name=self.name,
            capability=Capability.ROUTE_VALIDATION,
            output_crs=CRS.WGS84,
            needs_network=True,
        )

    def _env(self, name: str) -> str | None:
        getter = getattr(self.cfg, "_env_value", None)
        if callable(getter):
            return getter(name)
        return os.environ.get(name)

    def is_available(self) -> bool:
        raw = self._env(_ENABLED_ENV)
        return raw is None or str(raw).strip().lower() not in _DISABLED_VALUES

    def probe(self) -> ProbeResult:
        """无凭证可验：只声明静态可用性，不在 health 里打公共实例。"""
        if not self.is_available():
            return ProbeResult(False, error="已在配置中禁用")
        return ProbeResult(
            True, hint="公共路由服务，无需凭证；实际可用性在查询时探测"
        )

    def client(self, cache_dir: str = "data/cache") -> Any:
        raise NotImplementedError

    def route(self, points: list[tuple[float, float]], **kwargs: Any) -> Any:
        return self.client().route(points, **kwargs)


class OsrmRouteProvider(_BaseRouteProvider):
    name = "osrm"

    def client(self, cache_dir: str = "data/cache") -> Any:
        from ...osrm import OsrmClient

        return OsrmClient(cache_dir=cache_dir)


class ValhallaRouteProvider(_BaseRouteProvider):
    name = "valhalla"

    def client(self, cache_dir: str = "data/cache") -> Any:
        from ...valhalla import ValhallaClient

        return ValhallaClient(cache_dir=cache_dir)
