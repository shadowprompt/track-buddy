"""识别流水线：从截图到 GeoJSON 轨迹。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import AppConfig
from .cv_utils import resample_polyline, simplify_polyline
from .geocode import Geocoder, Lexicon
from .geo.coords import gcj02_to_wgs84, lonlat_to_mercator
from .ocr_anchor import Anchor, build_anchors, classify_text, ocr_texts
from .register import AnchorPair, Registration, apply_to_polyline, register
from .track import TrackResult, extract_track


def _mean_lat(points: list[tuple[float, float]]) -> float:
    """轨迹点的平均纬度（用于墨卡托尺度换算）。"""
    if not points:
        return 0.0
    return sum(p[1] for p in points) / len(points)


@dataclass
class PipelineResult:
    """流水线结果。"""

    geojson: dict[str, Any]
    confidence: float
    # GCJ-02 版本，供腾讯/高德等国内地图底图直接使用
    geojson_gcj02: dict[str, Any] = field(default_factory=dict)
    crs: str = "GCJ-02"
    track_points: int = 0
    distance_m: float = 0.0
    duration_note: str = ""
    anchors_used: list[dict[str, Any]] = field(default_factory=list)
    anchors_rejected: list[dict[str, Any]] = field(default_factory=list)
    road_attribution: dict[str, Any] = field(default_factory=dict)
    road_fit: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    # 中间产物，供前端叠加显示
    debug: dict[str, Any] = field(default_factory=dict)
    # 像素空间的轨迹折线，供前端在原图上叠加
    track_polyline_px: list[list[int]] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)


class RidePipeline:
    """完整的识别流水线。"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.lexicon = Lexicon(cfg.lexicon_path)
        self.geocoder = Geocoder(cfg, self.lexicon)

    def run(
        self,
        img: np.ndarray,
        city: str = "",
        allow_online: bool = True,
        output_crs: str = "WGS-84",
    ) -> PipelineResult:
        t0 = time.time()
        warnings: list[str] = []
        cfg = self.cfg.vision

        # --- 阶段 1-2：轨迹提取 ---
        track: TrackResult = extract_track(img, cfg)
        warnings.extend(track.notes)
        if len(track.polyline) < 10:
            return PipelineResult(
                geojson=_empty_geojson(),
                confidence=0.0,
                warnings=warnings + ["未能提取到有效轨迹线"],
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        # --- 阶段 2：OCR ---
        texts = ocr_texts(img, cfg)
        anchors = build_anchors(img, texts, cfg)
        if not anchors:
            warnings.append("OCR 未识别到可用地名锚点")

        # --- 阶段 3：地理编码 ---
        pairs = self._resolve_anchors(anchors, city, allow_online, warnings)

        # --- 阶段 4：配准 ---
        reg = register(
            pairs,
            inlier_m=cfg.ransac_inlier_m,
            min_inliers=cfg.ransac_min_inliers,
        )
        warnings.extend(reg.notes)
        if not reg.ok:
            return PipelineResult(
                geojson=_empty_geojson(),
                confidence=0.0,
                warnings=warnings + ["地理配准失败，无法得到坐标"],
                diagnostics={"track_points": len(track.polyline)},
                elapsed_ms=int((time.time() - t0) * 1000),
                debug={"track_mask": track.mask},
            )

        # 绝对位置自校验：抓「坐标对错同名地点」这类整体平移错误
        med_px, check_notes = self._self_check(reg, pairs)
        warnings.extend(check_notes)

        # --- 阶段 4.5：路网约束精修 ---
        # 用「轨迹必须走在真实道路上」纠正绝对位置。锚点是 POI 文字标注，
        # 标注点位与真实 POI 存在 100-300m 系统偏差（不同 POI 数据库之间），
        # 会让轨迹整体落到平行的错误道路族上。
        road_fit, road_net = self._road_refine(reg, track)
        if road_fit.get("ok"):
            warnings.append(
                f"路网精修：平移 东{road_fit['shift_e_m']:+.0f}m "
                f"北{road_fit['shift_n_m']:+.0f}m，贴路中位 "
                f"{road_fit['median_before_m']:.0f}m → {road_fit['median_after_m']:.0f}m"
            )
        else:
            warnings.extend(road_fit.get("notes", []))

        # --- 阶段 5：折线投影 ---
        # 输出密度按「米」而非「像素」控制：目标每点约 10 米。
        # 1 像素 ≈ 5.5 米，故步长取 2 像素。简化容差同样以米计（2 米），
        # 否则会把轨迹的弯道拉直（早期用 1 像素容差，轨迹被削掉 19% 长度）。
        #
        # 注意：reg.scale 是「墨卡托米/像素」。墨卡托投影在纬度 φ 处会把
        # 地面距离放大 1/cos(φ)，因此真实地面距离 = 墨卡托距离 × cos(φ)。
        # 里程用 haversine 计算（已含该修正），不要用 像素长 × scale 估算。
        meters_per_px = max(reg.scale, 1e-6)
        step_px = max(1.0, 10.0 / meters_per_px)
        sampled = resample_polyline(track.polyline, step_px=step_px)
        simplified = simplify_polyline(sampled, eps=2.0 / meters_per_px)
        lonlat_gcj = apply_to_polyline(reg, simplified)

        # --- 输出坐标转换 ---
        if output_crs.upper() in ("WGS-84", "WGS84", "EPSG:4326"):
            lonlat = [gcj02_to_wgs84(lo, la) for lo, la in lonlat_gcj]
            crs = "WGS-84"
        else:
            lonlat = lonlat_gcj
            crs = "GCJ-02"

        distance = 0.0
        for i in range(1, len(lonlat)):
            from .geo.coords import haversine

            distance += haversine(
                lonlat[i - 1][0], lonlat[i - 1][1], lonlat[i][0], lonlat[i][1]
            )

        geojson = _build_geojson(lonlat, reg, track)
        confidence = self._score_confidence(reg, track, anchors, pairs, road_fit)

        # --- 阶段 6：路名归属 ---
        # 只做命名，不改坐标。高德与美团底图同源，其官方路名才是用户
        # 在地图上看到、也会记住的名字；OSM 等第三方源常保留旧名。
        road_attr = self._attribute_roads(
            lonlat_gcj, simplified, texts, allow_online, road_fit, road_net, reg
        )
        if road_attr.segments:
            warnings.append(f"路名归属（{road_attr.source}）："
                            + "、".join(road_attr.roads[:8]))

        # 额外提供 GCJ-02 版本：国内地图服务（腾讯/高德）底图用的是
        # GCJ-02，消费者若把 WGS-84 坐标直接画上去会整体偏移约 350 米。
        geojson_gcj = _build_geojson(lonlat_gcj, reg, track)
        geojson_gcj["features"][0]["properties"]["crs"] = "GCJ-02"

        return PipelineResult(
            geojson=geojson,
            geojson_gcj02=geojson_gcj,
            confidence=confidence,
            crs=crs,
            track_points=len(lonlat),
            distance_m=round(distance, 1),
            anchors_used=[_anchor_dict(a) for a in reg.inliers],
            anchors_rejected=[_anchor_dict(a) for a in reg.outliers],
            road_attribution=road_attr.to_dict(),
            road_fit=road_fit,
            diagnostics={
                # 注意：下面是「墨卡托米/像素」，不是地面米/像素。
                # 地面距离 = 墨卡托距离 × cos(纬度)，在本例纬度约差 15%。
                "scale_mercator_m_per_px": round(reg.scale, 4),
                "scale_ground_m_per_px": round(
                    reg.scale * math.cos(math.radians(_mean_lat(lonlat_gcj))), 4
                ),
                "rotation_deg": round(reg.rotation_deg, 3),
                "model": reg.model,
                "rmse_m": round(reg.rmse_m, 2),
                "anchor_backproj_px": round(med_px, 1),
                "inliers": len(reg.inliers),
                "outliers": len(reg.outliers),
                "ocr_texts": len(texts),
                "anchor_candidates": len(anchors),
                "track_points_px": len(track.polyline),
                "bbox_px": list(track.bbox),
                "start_marker": list(track.start_marker) if track.start_marker else None,
                "end_marker": list(track.end_marker) if track.end_marker else None,
                "online_geocoding": bool(self.cfg.has_online_key and allow_online),
                "lexicon": self.lexicon.stats(),
            },
            warnings=warnings,
            elapsed_ms=int((time.time() - t0) * 1000),
            debug={"track_mask": track.mask},
            track_polyline_px=[[int(px), int(py)] for px, py in simplified],
            image_size=(int(img.shape[1]), int(img.shape[0])),
        )

    def _resolve_anchors(
        self,
        anchors: list[Anchor],
        city: str,
        allow_online: bool,
        warnings: list[str],
    ) -> list[AnchorPair]:
        """为锚点匹配候选经纬度。

        只有 kind 为 poi / icon 的锚点参与配准；road 类仅用于区域校验
        （实测道路名锚点会让配准整体偏移 3 公里）。

        每个锚点可能匹配多个候选（如同名地点、词典的部分匹配）。
        这里把候选全量交给 RANSAC，由几何一致性来挑选——这比
        「取第一个候选」稳健得多。
        """
        pairs: list[AnchorPair] = []
        # 只对权重较高、名字较完整的锚点做地理编码，减少请求量。
        # 排序策略：先按「本地词典能否命中」排序（命中的优先），
        # 再按锚点权重、名称长度排序。
        # 早期实现按权重排序后截断，因 OCR 垃圾文本（如「支工」「丰河」）
        # 与正常 POI 权重相同、且排序稳定，导致真正的 POI 锚点被截掉。
        targets = [
            a for a in anchors
            if a.kind in ("icon", "poi") and len(a.name) >= 2
        ]
        targets.sort(
            key=lambda a: (
                0 if self.lexicon.lookup(a.name, city) else 1,
                -a.weight,
                -len(a.name),
            )
        )
        targets = targets[:14]

        matched = 0
        for a in targets:
            results = self.geocoder.geocode(a.name, city=city, allow_online=allow_online)
            if not results:
                continue
            matched += 1
            # 词典里权重最高的候选优先，其余作为备选（最多 3 个）
            for r in results[:3]:
                pairs.append(
                    AnchorPair(
                        name=a.name,
                        px=a.px,
                        py=a.py,
                        weight=a.weight,
                        kind=a.kind,
                        lon=r.lon,
                        lat=r.lat,
                        match_quality=r.quality,
                    )
                )
        if not pairs:
            warnings.append(
                "所有锚点均未匹配到坐标：可能是冷启动城市，或未配置地图服务密钥"
                "（设置 AMAP_KEY / TMAP_KEY 环境变量后可启用在线地理编码）"
            )
        else:
            warnings.append(f"{matched}/{len(targets)} 个 POI 锚点匹配到坐标")
        return pairs

    def _road_refine(self, reg: Registration, track: TrackResult):
        """路网约束精修：用「轨迹必须走在真实道路上」纠正配准的绝对位置。

        锚点来自 POI 文字标注，标注位置与真实 POI 点位存在系统性偏差
        （实测同一 POI 在 OSM 与国内底图之间可差 100-300 米），
        会让整条轨迹落到平行的错误道路族上。

        道路几何不存在这种问题——路网位置在所有地图上必须一致。
        这里在保持尺度与朝向的前提下平移轨迹，使贴路距离最小，
        并同时给出每一段所属的道路名。
        """
        from .roadfit import RoadNetwork, fetch_roads, refine_with_roads

        if not self.cfg.road_fit_enabled:
            return {"ok": False, "notes": ["路网精修已关闭"]}, None

        # 轨迹像素 bbox → 经纬度 bbox（留 margin 余量覆盖周边路网）
        # 注意 track.bbox 的语义是 (x, y, w, h)，不是 (x0, y0, x1, y1)
        try:
            bx, by, bw, bh = track.bbox
            x0, y0 = float(bx), float(by)
            x1, y1 = float(bx + bw), float(by + bh)
            corners = [
                reg.pixel_to_lonlat(x0, y0),
                reg.pixel_to_lonlat(x1, y0),
                reg.pixel_to_lonlat(x1, y1),
                reg.pixel_to_lonlat(x0, y1),
            ]
            lons = [c[0] for c in corners]
            lats = [c[1] for c in corners]
            margin = 0.006
            bbox = (min(lons) - margin, min(lats) - margin,
                    max(lons) + margin, max(lats) + margin)
        except Exception as exc:
            return {"ok": False,
                    "notes": [f"bbox 计算失败：{type(exc).__name__}"]}, None

        try:
            roads = fetch_roads(
                bbox, timeout=self.cfg.http_timeout * 2,
                cache_path=str(self.cfg.road_cache_dir / "roads.json"),
            )
        except Exception as exc:
            return {"ok": False,
                    "notes": [f"路网获取失败，跳过精修：{type(exc).__name__}"]}, None

        net = RoadNetwork(roads)
        fit = refine_with_roads(reg, track.polyline, roads)
        if fit.ok:
            # 把平移量并入配准矩阵（墨卡托 y 轴向上，北为正）
            reg.translation = reg.translation + np.array(
                [fit.shift_e_m, fit.shift_n_m], dtype=float
            )
        return fit.to_dict(), net

    def _attribute_roads(
        self,
        lonlat_gcj: list[tuple[float, float]],
        simplified: np.ndarray,
        texts: list[Any],
        allow_online: bool = True,
        road_fit: dict[str, Any] | None = None,
        road_net: Any = None,
        reg: Registration | None = None,
    ):
        """给轨迹分段标注道路名。

        优先用高德路径规划 API——它与美团底图同源，返回的是国内官方路名。
        没有密钥时降级为「截图自带标签」：直接把美团地图上印着的路名
        按像素距离归到各段上。这是地图自己说的，命名与用户看到的完全一致。
        """
        from .roadname import (RoadAttribution, RoadSegment, attribute_by_amap,
                                attribute_by_shot_labels, is_road_name)

        # 路网精修已经给出每段所属道路——这是最准的来源（几何直接判定）
        if road_fit and road_fit.get("ok") and road_net is not None and reg is not None:
            # 索引必须与输出折线一一对应，所以对「简化后的折线」重新归属。
            # 此时 reg 已并入平移量，坐标已是精修后的位置。
            raw = road_net.segments(reg, simplified, min_len_m=1.0)
            segs = [
                RoadSegment(
                    road=s["road"], start_index=s["start_index"],
                    end_index=s["end_index"], length_m=s["length_m"],
                    source="road_network", confidence=0.85,
                )
                for s in raw
            ]
            return RoadAttribution(
                segments=segs,
                source="road_network",
                note="路名来自轨迹与真实路网的贴合并集（几何判定，最可靠）",
            )

        if allow_online and self.cfg.allow_online:
            attr = attribute_by_amap(lonlat_gcj, timeout=self.cfg.http_timeout)
            if attr.segments:
                return attr

        # 离线兜底：截图上印着的路名
        labels = []
        for t in texts:
            if getattr(t, "text", "") in {"美团", "美团地图"}:
                continue
            if is_road_name(t.text):
                labels.append({"text": t.text, "px": t.cx, "py": t.cy,
                               "vertical": bool(getattr(t, "is_vertical", False))})
        if not labels:
            return RoadAttribution(source="none", note="无路名标签可用")
        pts = [(float(px), float(py)) for px, py in simplified]
        return attribute_by_shot_labels(lonlat_gcj, pts, labels)

    def _self_check(
        self, reg: Registration, pairs: list[AnchorPair]
    ) -> tuple[float, list[str]]:
        """绝对位置自校验。

        做法：把每个内点锚点的「真值经纬度」反投影回像素，看是否落回
        它自己的文字位置附近。地图标注的文字中心与真实点位通常有
        几十到几百米的偏移（POI 类实测中位 170 米），因此允许一定容差。

        若反投影偏移普遍很大，说明词典里的坐标对错了同名地点，
        配准结果整体平移——这类错误靠 RMSE 看不出来（因为 RMSE 只是
        相对锚点集合的自洽性）。

        返回 (偏移中位数像素, 说明列表)
        """
        notes: list[str] = []
        if not reg.ok or not reg.inliers:
            return 0.0, notes

        inv = np.linalg.inv(reg.matrix)
        offsets: list[float] = []
        for a in reg.inliers:
            if a.lon is None or a.lat is None:
                continue
            mx, my = lonlat_to_mercator(a.lon, a.lat)
            v = inv @ (np.array([mx, my]) - reg.translation)
            offsets.append(float(np.hypot(v[0] - a.px, v[1] - a.py)))
        if not offsets:
            return 0.0, notes

        med = float(np.median(offsets))
        # POI 类锚点的合理上限：文字中心偏移一般不超过 ~300 米
        tol_m = 400.0
        tol_px = tol_m / max(reg.scale, 1e-6)
        if med > tol_px:
            notes.append(
                f"绝对位置自校验未通过：锚点真值反投影偏移中位 {med:.0f}px"
                f"（≈{med * reg.scale:.0f} 米），超过容差 {tol_m:.0f} 米。"
                f"通常是词典中该地名的坐标指向了同名地点，结果会整体平移。"
            )
        return med, notes

    def _score_confidence(
        self,
        reg: Registration,
        track: TrackResult,
        anchors: list[Anchor],
        pairs: list[AnchorPair],
        road_fit: dict[str, Any] | None = None,
    ) -> float:
        """综合置信度。

        构成：几何拟合质量（主导）+ 锚点数量 + 锚点类型质量 + 轨迹提取质量
             + 绝对位置自校验。
        """
        # 拟合质量：RMSE 越小越好，30m 以内视为优秀
        if reg.rmse_m <= 15:
            fit = 1.0
        elif reg.rmse_m <= 40:
            fit = 0.85
        elif reg.rmse_m <= 80:
            fit = 0.6
        else:
            fit = 0.35

        # 内点占比
        total = len(reg.inliers) + len(reg.outliers)
        ratio = len(reg.inliers) / total if total else 0.0

        # 锚点数量（越多越稳，但收益递减）
        n = len(reg.inliers)
        count = min(1.0, 0.35 + 0.11 * n) if n else 0.0

        # 锚点质量：图标锚点最可靠
        kinds = [a.kind for a in reg.inliers]
        kind_q = 1.0 if "icon" in kinds else (0.9 if "poi" in kinds else 0.7)

        conf = 0.40 * fit + 0.18 * ratio + 0.18 * count + 0.14 * kind_q
        conf *= track.confidence if track.confidence < 1 else 1.0

        # 绝对位置自校验：偏移过大则大幅降权
        med_px, _ = self._self_check(reg, pairs)
        if med_px > 0:
            tol_px = 400.0 / max(reg.scale, 1e-6)
            if med_px > tol_px:
                conf *= 0.45
            elif med_px > tol_px * 0.6:
                conf *= 0.8

        # 路网约束精修成功时，绝对位置由「轨迹贴在真实道路上」来担保，
        # 这比锚点残差可靠得多——锚点残差大恰恰说明锚点数据源与底图
        # 基准不一致，而这正是路网精修要纠正的东西。此时置信度改由
        # 路网贴合度主导。
        if road_fit and road_fit.get("ok"):
            after = float(road_fit.get("median_after_m", 999))
            before = float(road_fit.get("median_before_m", 999))
            if after <= 8:
                road_q = 1.0
            elif after <= 20:
                road_q = 0.9
            elif after <= 40:
                road_q = 0.75
            else:
                road_q = 0.55
            gain = 1.0 if before <= 0 else max(
                0.0, min(1.0, (before - after) / max(before, 1)))
            conf = 0.60 * road_q + 0.18 * ratio + 0.12 * count + 0.10 * gain
            conf *= track.confidence if track.confidence < 1 else 1.0
        return round(min(0.99, max(0.0, conf)), 3)


def _anchor_dict(a: AnchorPair) -> dict[str, Any]:
    return {
        "name": a.name,
        "px": round(a.px, 1),
        "py": round(a.py, 1),
        "kind": a.kind,
        "lon": round(a.lon, 7) if a.lon is not None else None,
        "lat": round(a.lat, 7) if a.lat is not None else None,
        "quality": round(a.match_quality, 2),
    }


def _build_geojson(
    lonlat: list[tuple[float, float]], reg: Registration, track: TrackResult
) -> dict[str, Any]:
    """构建 GeoJSON。

    坐标顺序严格遵循「骑行方向」：LineString 的第一个点是**起点**，
    最后一点是**终点**。这一点很重要——很多第三方工具会默认
    「首点=起点」来标注起终点，若顺序相反，地图上就会把起终点画反。

    另在 properties 中显式携带 crs 字段，避免消费方把 WGS-84 坐标
    直接画在 GCJ-02 的底图上（会造成数百米偏移）。
    """
    coords = [[round(lo, 7), round(la, 7)] for lo, la in lonlat]
    props = {
        "source": "meituan-ride-screenshot",
        "crs": "WGS-84",
        "coordinate_order": "lon,lat",
        "direction": "first_point_is_start",
        "scale_m_per_px": round(reg.scale, 4),
        "rmse_m": round(reg.rmse_m, 2),
        "anchor_count": len(reg.inliers),
    }
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "LineString", "coordinates": coords},
            },
            {
                "type": "Feature",
                "properties": {"kind": "start", "name": "起点"},
                "geometry": {
                    "type": "Point",
                    "coordinates": coords[0] if coords else [0, 0],
                },
            },
            {
                "type": "Feature",
                "properties": {"kind": "end", "name": "终点"},
                "geometry": {
                    "type": "Point",
                    "coordinates": coords[-1] if coords else [0, 0],
                },
            },
        ],
    }


def _empty_geojson() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"status": "failed"},
                "geometry": {"type": "LineString", "coordinates": []},
            }
        ],
    }
