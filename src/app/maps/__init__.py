"""可插拔地图能力层。

分层约定（上层只依赖本包的接口，不出现厂商名）::

    app/maps/
      contracts.py    数据契约 + CRS 声明 + 统一坐标系换算
      credentials.py  统一 env/.env 与别名解析
      registry.py     按能力装配、配置驱动顺序、回退与健康探测
      http.py         Session / UA / 限流 / 重试 / 磁盘缓存
      geocoding/      能力①：地名 -> 坐标
      roadname/       能力②：坐标 -> 路名
      roads/          能力③：道路几何（统一 Overpass 镜像与轮换）
      routing/        能力④：路线校验（共享 HTTP / 缓存 / polyline 工具）
      staticmap/      能力⑥：静态地图素材（实验与合成测试用）

阶段 1（骨架 + ①② 接入）、阶段 2（③④ 接入 + 收敛重复基础设施）与
阶段 4（⑥ 静态地图收敛）已完成，每阶段单独全量测试、单独提交。

能力⑤（前端底图）在另一条通道：适配层是前端文件 ``web/js/basemap.js``，
不由本包装配——JS SDK 的鉴权模型与后端 HTTP 服务不同
（腾讯走本地代理、高德用前端 key + securityJsCode），
详见 docs/HANDOVER.md §22.32。
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
