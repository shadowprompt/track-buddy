"""能力④「路线校验」的共享工具。

``app/osrm.py`` 与 ``app/valhalla.py`` 过去各自实现了一遍
polyline 解码、路名命中率、折线长度、md5 缓存键、磁盘缓存与限流。
两份实现语义相同但细节有出入（例如 ``match_ratio`` 对空名单的处理），
改一处忘一处就会让两种路线的评分标准漂移。

这里只放**与厂商无关**的纯函数和缓存容器；HTTP 会话/UA/限流统一由
``app/maps/http.py`` 提供，厂商特有的请求与解析仍留在各自客户端。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def stable_key(*parts: str) -> str:
    """稳定的缓存键。

    ⚠️ 不能用内置 ``hash()``：Python 对 str 的 hash 受 PYTHONHASHSEED
    随机化影响，每次进程启动缓存路径都不同，缓存永远不命中
    （实测端到端耗时从 150 秒涨到 528 秒）。
    """
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def decode_polyline(
    encoded: str | None, precision: int = 6,
) -> list[tuple[float, float]]:
    """解码 Valhalla / Google 风格的 encoded polyline。

    编码规则：ASCII 63 基准、5 位一组、低位在高位，每组最高位表示符号，
    lat 在前 lon 在后，都是相对上一点的增量。
    """
    if not encoded:
        return []
    coords: list[tuple[float, float]] = []
    index = 0
    lat = 0.0
    lon = 0.0
    factor = float(10 ** precision)
    n = len(encoded)
    while index < n:
        for axis in range(2):
            shift = 0
            result = 0
            while True:
                if index >= n:
                    return coords
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if axis == 0:
                lat += delta
            else:
                lon += delta
        coords.append((lon / factor, lat / factor))
    return coords


def match_ratio(names: list[str], user_names: list[str]) -> float:
    """路线上的路名命中用户路名的比例（子串双向匹配、按集合去重）。

    OSRM 与 Valhalla 的旧实现都等价于此；统一后两种路线的评分口径一致。
    """
    if not names or not user_names:
        return 0.0
    on_path = {n for n in names if n}
    hits = sum(
        1 for u in user_names
        if any(u in n or n in u for n in on_path)
    )
    return hits / len(user_names)


def polyline_length(points: list[tuple[float, float]]) -> float:
    """折线长度（米），按 WGS-84 经纬度近似。"""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        dx = (points[i][0] - points[i - 1][0]) * 95160.0
        dy = (points[i][1] - points[i - 1][1]) * 111320.0
        total += math.hypot(dx, dy)
    return total


def order_consistency(
    road_names: list[str], user_names: list[str],
) -> tuple[float, list[str]]:
    """用户填的路名在路线里的相对顺序是否一致。

    返回 (一致率, 命中的路名按路线出现顺序排列)。顺序正确比「名字都存在」
    更有价值：它说明用户记忆的路线与真实路网结构吻合。
    """
    seq: list[tuple[str, int]] = []
    for u in user_names:
        best_idx: int | None = None
        for i, rn in enumerate(road_names):
            if u in rn or rn in u:
                best_idx = i
                break
        if best_idx is not None:
            seq.append((u, best_idx))
    if len(seq) < 2:
        return 0.0, [u for u, _ in seq]
    ok = sum(1 for i in range(1, len(seq)) if seq[i][1] > seq[i - 1][1])
    return ok / (len(seq) - 1), [u for u, _ in seq]


class RouteCache:
    """路由结果的磁盘缓存（按厂商前缀隔离，保持旧文件名与字段格式）。

    ``ttl_s`` 只对带 ``ts`` 字段的条目生效：OSRM 缓存一直写 ts（7 天
    过期），Valhalla 历史缓存没有 ts，读时视为仍然有效，避免重构后
    所有既有缓存一夜失效、把公共实例打爆。
    """

    def __init__(
        self, cache_dir: str | Path, prefix: str, ttl_s: float = 7 * 86400.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.prefix = prefix
        self.ttl_s = float(ttl_s)

    def path(self, key: str) -> Path:
        return self.cache_dir / ("%s_%s.json" % (self.prefix, key))

    def read(self, key: str) -> dict[str, Any] | None:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(d, dict):
            return None
        ts = d.get("ts")
        if ts is not None and self.ttl_s > 0:
            try:
                if time.time() - float(ts) > self.ttl_s:
                    return None
            except (TypeError, ValueError):
                pass
        return d

    def write(self, key: str, data: dict[str, Any]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.path(key).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass  # 缓存写失败不能影响本次结果
