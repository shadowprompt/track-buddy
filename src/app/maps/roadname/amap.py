"""能力②的路名 provider：高德路径规划（与美团底图同源）。

只取 ``steps[].road`` 拿国内官方路名，**不使用返回的 polyline 几何**——
实测高德 walking 会静默丢弃途经点，几何不可信，路线形状始终来自
截图提取 + OSM 路网。领域模型（RoadAttribution / RoadSegment）留在
``app/roadname.py``，这里延迟导入以避免包初始化环。
"""

from __future__ import annotations

from typing import Any

import requests

from ..contracts import CRS, Capability, ProbeResult, ProviderMeta

AMAP_DIRECTION = "https://restapi.amap.com/v4/direction/bicycling"
AMAP_DIRECTION_V3 = "https://restapi.amap.com/v3/direction/driving"


class AmapRoadNameProvider:
    name = "amap"

    def __init__(self, cfg: Any, credentials: Any,
                 session: Any | None = None) -> None:
        self.cfg = cfg
        self.credentials = credentials
        self._session = session
        self.last_error = ""
        self.meta = ProviderMeta(
            name=self.name,
            capability=Capability.ROADNAME,
            output_crs=CRS.GCJ02,
            credential_aliases=("AMAP_KEY",),
        )

    @property
    def key(self) -> str | None:
        return self.credentials.resolve("AMAP_KEY")

    def is_available(self) -> bool:
        return bool(self.key)

    def probe(self) -> ProbeResult:
        """打一次 bicycling，确认 key 与接口都可用。"""
        params = {"origin": "121.360000,31.190000",
                  "destination": "121.400000,31.220000",
                  "key": self.key or "", "extensions": "base", "strategy": "0"}
        try:
            r = requests.get(AMAP_DIRECTION, params=params,
                             timeout=self.cfg.http_timeout)
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(False, error=f"{type(exc).__name__}: {exc}")
        if str(data.get("status")) in ("1", "0"):
            paths = (data.get("route") or {}).get("paths") or []
            return ProbeResult(True, hits=len(paths))
        info = str(data.get("info") or data.get("infocode") or "未知错误")
        return ProbeResult(False, error=info)

    def attribute(self, lonlat: list[tuple[float, float]],
                  key: str | None = None,
                  timeout: float = 12.0) -> Any:
        """按起终点请求路径，把每一步映射到一条 RoadSegment。"""
        from ...roadname import (  # 延迟导入避免循环
            RoadAttribution,
            RoadSegment,
            _decode_polyline,
            _road_from_instruction,
        )

        key = key or self.key
        if not key or len(lonlat) < 2:
            return RoadAttribution(
                source="none", note="未配置在线路名服务密钥，无法获取官方路名")
        origin = f"{lonlat[0][0]:.6f},{lonlat[0][1]:.6f}"
        dest = f"{lonlat[-1][0]:.6f},{lonlat[-1][1]:.6f}"
        last_err = ""
        for url in (AMAP_DIRECTION, AMAP_DIRECTION_V3):
            try:
                resp = requests.get(
                    url,
                    params={"origin": origin, "destination": dest, "key": key,
                            "extensions": "base", "strategy": "0"},
                    timeout=timeout,
                )
                data = resp.json()
            except Exception as exc:  # 网络/解析异常都降级，不影响主流程
                last_err = f"{type(exc).__name__}: {exc}"
                continue
            if str(data.get("status")) not in ("1", "0"):
                last_err = str(data.get("info") or data.get("infocode") or "unknown")
                continue
            paths = (data.get("route") or {}).get("paths") or []
            if not paths:
                last_err = "route.paths 为空"
                continue
            steps = paths[0].get("steps") or []
            if not steps:
                last_err = "steps 为空"
                continue

            segs: list[Any] = []
            idx = 0
            for st in steps:
                road = (st.get("road") or "").strip()
                if not road:
                    road = _road_from_instruction(st.get("instruction") or "")
                dist = float(st.get("distance") or 0.0)
                pts = _decode_polyline(st.get("polyline") or "")
                n = max(1, len(pts) or 1)
                segs.append(RoadSegment(
                    road=road or "（无名道路）",
                    start_index=idx, end_index=idx + n,
                    length_m=dist, source="amap",
                    confidence=0.92 if road else 0.5,
                ))
                idx += n
            return RoadAttribution(
                segments=segs, source="amap",
                note=f"路名来自在线规划（与美团底图同源），共 {len(segs)} 段",
            )
        self.last_error = last_err
        return RoadAttribution(
            source="none", note=f"在线路名服务调用失败：{last_err}")
