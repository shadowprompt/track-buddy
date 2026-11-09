"""按能力装配 provider：配置驱动顺序、可用性过滤、统一健康探测。

调用方只说「我要 geocoding」，不写厂商名。加厂商 = 注册一个实现 +
（可选）在 .env 的顺序里加一个名字，调用方零改动。

顺序环境变量统一为 ``MAP_<CAPABILITY>_ORDER``；为兼容已部署配置，
地理编码仍读取旧的 ``GEOCODE_PROVIDER_ORDER``（新名优先）。
未知名称直接忽略，不会让一个拼错的 provider 打断整条链路。
"""

from __future__ import annotations

import os
from typing import Any

from .contracts import Capability, MapProvider, ProbeResult
from .credentials import CredentialStore

DEFAULT_ORDER: dict[Capability, tuple[str, ...]] = {
    Capability.GEOCODING: ("amap", "tmap", "baidu", "tianditu"),
    Capability.ROADNAME: ("amap", "shot_labels"),
    Capability.ROAD_GEOMETRY: ("osm",),
    Capability.ROUTE_VALIDATION: ("osrm", "valhalla"),
}

ORDER_ENV: dict[Capability, tuple[str, ...]] = {
    Capability.GEOCODING: ("MAP_GEOCODING_ORDER", "GEOCODE_PROVIDER_ORDER"),
    Capability.ROADNAME: ("MAP_ROADNAME_ORDER",),
    Capability.ROAD_GEOMETRY: ("MAP_ROAD_GEOMETRY_ORDER",),
    Capability.ROUTE_VALIDATION: ("MAP_ROUTE_VALIDATION_ORDER",),
}


class ProviderRegistry:
    """一个进程内可共享的 provider 注册表。"""

    def __init__(self, cfg: Any | None = None,
                 credentials: CredentialStore | None = None):
        self.cfg = cfg
        self.credentials = credentials or CredentialStore(cfg)
        # 不同能力可以有同名 provider（例如高德同时提供地名与路名），
        # 因此索引必须带能力维度，否则后注册的会覆盖先注册的。
        self._providers: dict[tuple[Capability, str], MapProvider] = {}

    def register(self, provider: MapProvider) -> MapProvider:
        """注册实现；同一能力内同名后来者覆盖前者，便于测试替换。"""
        key = (provider.meta.capability, provider.meta.name)
        self._providers[key] = provider
        return provider

    def get(self, name: str,
            capability: Capability | None = None) -> MapProvider | None:
        if capability is not None:
            return self._providers.get((capability, name))
        for (cap, pname), provider in self._providers.items():
            if pname == name:
                return provider
        return None

    def names(self, capability: Capability | None = None) -> tuple[str, ...]:
        return tuple(
            pname for (cap, pname) in self._providers
            if capability is None or cap == capability
        )

    def _env(self, name: str) -> str | None:
        getter = getattr(self.cfg, "_env_value", None)
        if callable(getter):
            return getter(name)
        return os.environ.get(name)

    def order(self, capability: Capability) -> tuple[str, ...]:
        """该能力的调用顺序。

        已注册但未写进默认顺序、也没写进配置的实现会追加到末尾，
        保证「新注册一个 provider」不会因为漏改默认表而永远不被调用。
        """
        known = set(self.names(capability))
        for env_name in ORDER_ENV.get(capability, ()):
            raw = self._env(env_name) or ""
            picked = tuple(dict.fromkeys(
                x.strip() for x in raw.split(",")
                if x.strip() and x.strip() in known
            ))
            if picked:
                tail = [n for n in self.names(capability) if n not in picked]
                return picked + tuple(tail)
        base = [n for n in DEFAULT_ORDER.get(capability, ()) if n in known]
        base += [n for n in self.names(capability) if n not in base]
        return tuple(base)

    def configured_order(self, capability: Capability) -> tuple[str, ...]:
        """**显式配置**的调用顺序（不读默认值）。

        与 ``order()`` 的区别：``order()`` 在没配置时返回内置默认表
        （地理编码是 amap/tmap/baidu/tianditu 四家），而调用方往往需要
        区分「用户指定了谁」与「系统默认是什么」——
        比如单源严格模式：用户只写了一个名字就只用它；
        没写则按默认表里**第一个可用**的来，不该被当成「四家依次回退」。

        没配置时返回空元组。
        """
        known = set(self.names(capability))
        for env_name in ORDER_ENV.get(capability, ()):
            raw = self._env(env_name) or ""
            picked = tuple(dict.fromkeys(
                x.strip() for x in raw.split(",")
                if x.strip() and x.strip() in known
            ))
            if picked:
                return picked
        return ()

    def available(self, capability: Capability) -> list[MapProvider]:
        """按序返回「已注册且当前可用」的实现。

        回退语义由调用方执行：依次尝试，返回空结果就落到下一个。
        注册表不替调用方决定什么叫「成功」，因为不同能力的成功判据不同。
        """
        out: list[MapProvider] = []
        for name in self.order(capability):
            p = self._providers.get((capability, name))
            if p is None or p.meta.capability != capability:
                continue
            if p.is_available():
                out.append(p)
        return out

    def probe(self, capability: Capability | None = None) -> dict[str, dict[str, Any]]:
        """逐个探测可用 provider，异常转成结构化结果，不让 health 500。"""
        caps = ([capability] if capability is not None
                else [c for c in Capability if c in DEFAULT_ORDER])
        single = len(caps) == 1
        out: dict[str, dict[str, Any]] = {}
        for cap in caps:
            for name in self.names(cap):
                p = self._providers[(cap, name)]
                if not p.is_available():
                    continue
                try:
                    result = p.probe()
                except Exception as exc:  # noqa: BLE001 - health 不能因单点失败而崩
                    result = ProbeResult(False, error=f"{type(exc).__name__}: {exc}")
                # 多能力一起探测时加前缀，避免同名 provider 互相覆盖。
                key = name if single else f"{cap.value}/{name}"
                out[key] = result.to_dict()
        return out
