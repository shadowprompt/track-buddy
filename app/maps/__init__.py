"""可插拔地图能力层。

分层约定（上层只依赖本包的接口，不出现厂商名）::

    app/maps/
      contracts.py    数据契约 + CRS 声明 + 统一坐标系换算
      credentials.py  统一 env/.env 与别名解析
      registry.py     按能力装配、配置驱动顺序、回退与健康探测
      http.py         Session / UA / 限流 / 重试 / 磁盘缓存
      geocoding/      阶段 1b：地名 -> 坐标
      roadname/       阶段 1b：坐标 -> 路名
      roads/          阶段 2：道路几何
      routing/        阶段 2：路线校验

阶段 1a 当前只落地骨架四个模块，**不接任何现有调用点**，因此不改变
已验收的识别行为。接入从阶段 1b 开始，每刀单独跑全量测试与验收。
"""

from __future__ import annotations

from .contracts import (
    CRS,
    Capability,
    GeoPoint,
    MapProvider,
    PlaceCandidate,
    ProbeResult,
    ProviderMeta,
    to_gcj02,
)
from .credentials import CredentialStore
from .http import HttpClient
from .registry import ProviderRegistry

__all__ = [
    "CRS",
    "Capability",
    "GeoPoint",
    "MapProvider",
    "PlaceCandidate",
    "ProbeResult",
    "ProviderMeta",
    "to_gcj02",
    "CredentialStore",
    "HttpClient",
    "ProviderRegistry",
]
