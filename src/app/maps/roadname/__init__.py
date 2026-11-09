"""能力②：坐标/轨迹 -> 路名。

上层通过 ``attribute_by_online``（见 app/roadname.py）按配置顺序调用，
不出现厂商名。当前注册高德；截图自带标签是离线兜底，由 pipeline 在
在线源都不可用 / 未返回结果时调用，属于同一能力的另一实现。
"""

from __future__ import annotations

from typing import Any

from ..credentials import CredentialStore
from ..registry import ProviderRegistry
from .amap import AmapRoadNameProvider

__all__ = ["build_roadname_registry", "AmapRoadNameProvider"]


def build_roadname_registry(
    cfg: Any, credentials: CredentialStore | None = None
) -> tuple[ProviderRegistry, dict[str, Any]]:
    """装配路名能力注册表，返回 (注册表, 名称 -> 实现)。

    顺序读 ``MAP_ROADNAME_ORDER``。默认 ``amap`` 优先；把配置改成
    不含 amap 的值即可完全关闭在线路名，落到截图标签兜底，无需改代码。
    """
    creds = credentials or CredentialStore(cfg)
    reg = ProviderRegistry(cfg, creds)
    impl = AmapRoadNameProvider(cfg, creds)
    reg.register(impl)
    return reg, {impl.name: impl}
