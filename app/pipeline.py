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

# 路网精修的适用尺度上限（米/像素）。超过这个尺度说明截图是城市级
# 概览图，路网既拉不动（数据量过大）也约束不了（像素精度不足）。
_ROAD_FIT_MAX_M_PER_PX = 8.0

# 能作为「强约束」的锚点类型。地铁线号（transit）与高架路（road）
# 天然指向一段线或一条路，与标注点的对应关系很弱；它们不匹配不代表
# 定位失败，不该计入置信度的分母。
_STRONG_ANCHOR_KINDS = ("poi", "station", "icon", "district")


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
    # 用户提供的路名约束及其效果
    road_name_constraints: list[dict[str, Any]] = field(default_factory=list)
    road_name_fit: dict[str, Any] = field(default_factory=dict)
    # 备选位置（低置信度时给用户「换个位置」）
    #
    # 低缩放截图的定位信号本质上是**多峰**的：底图匹配实测前 12 个候选
    # 的分数差只有 0.1%，但地理位置分散在 900-1600 米范围内。
    # 只给一个答案等于替用户乱猜，不如把几个候选摆出来。
    alternatives: list[dict[str, Any]] = field(default_factory=list)


# diagnostics 的完整键集。
#
# 早期实现只在成功路径填充这些字段，失败路径只给 {"track_points": N}，
# 前端于是读到 undefined 再调 .toFixed()，整页崩掉，
# 后端 warnings 里真正的失败原因反而被吞掉——用户只看到
# "Cannot read properties of undefined (reading 'toFixed')"，
# 完全不知道发生了什么。这里用一份默认值把结构固定下来。
DIAG_KEYS: tuple[str, ...] = (
    "scale_mercator_m_per_px",
    "scale_ground_m_per_px",
    "rotation_deg",
    "model",
    "rmse_m",
    "anchor_backproj_px",
    "inliers",
    "outliers",
    "ocr_texts",
    "anchor_candidates",
    "track_points_px",
    "bbox_px",
    "start_marker",
    "end_marker",
    "online_geocoding",
    "lexicon",
    "alternatives",
)


def _base_diagnostics(**over: Any) -> dict[str, Any]:
    """返回结构完整的 diagnostics，缺省值用 None（前端显示为「—」）。"""
    dg: dict[str, Any] = {k: None for k in DIAG_KEYS}
    dg["lexicon"] = {}
    dg.update(over)
    return dg


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
        refine: bool = True,
        road_names: list[str] | None = None,
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
                diagnostics=_base_diagnostics(
                    track_points_px=len(track.polyline),
                    bbox_px=list(track.bbox),
                    start_marker=list(track.start_marker) if track.start_marker else None,
                    end_marker=list(track.end_marker) if track.end_marker else None,
                    lexicon=self.lexicon.stats(),
                ),
                image_size=(int(img.shape[1]), int(img.shape[0])),
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
                diagnostics=_base_diagnostics(
                    track_points_px=len(track.polyline),
                    bbox_px=list(track.bbox),
                    start_marker=list(track.start_marker) if track.start_marker else None,
                    end_marker=list(track.end_marker) if track.end_marker else None,
                    ocr_texts=len(texts),
                    anchor_candidates=len(anchors),
                    online_geocoding=bool(self.cfg.has_online_key and allow_online),
                    lexicon=self.lexicon.stats(),
                ),
                elapsed_ms=int((time.time() - t0) * 1000),
                debug={"track_mask": track.mask},
                track_polyline_px=[[int(px), int(py)] for px, py in track.polyline],
                image_size=(int(img.shape[1]), int(img.shape[0])),
            )

        # 绝对位置自校验：抓「坐标对错同名地点」这类整体平移错误
        med_px, check_notes = self._self_check(reg, pairs)
        warnings.extend(check_notes)

        # --- 备选位置 ---
        # 收集与最优解地理上分离（>300 米）的其他自洽解。
        # 低置信度时前端会把它们列出来，让用户一键切换。
        alternatives = self._collect_alternatives(reg, track, output_crs)

        # --- 阶段 4.5：路网约束精修 ---
        # 用「轨迹必须走在真实道路上」纠正绝对位置。锚点是 POI 文字标注，
        # 标注点位与真实 POI 存在 100-300m 系统偏差（不同 POI 数据库之间），
        # 会让轨迹整体落到平行的错误道路族上。
        road_fit, road_net = (
            self._road_refine(reg, track) if refine
            else ({"ok": False, "notes": ["本次跳过路网精修（异步模式下由后台任务执行）"]}, None)
        )
        if road_fit.get("ok"):
            warnings.append(
                f"路网精修：平移 东{road_fit['shift_e_m']:+.0f}m "
                f"北{road_fit['shift_n_m']:+.0f}m，贴路中位 "
                f"{road_fit['median_before_m']:.0f}m → {road_fit['median_after_m']:.0f}m"
            )
        else:
            warnings.extend(road_fit.get("notes", []))

        # --- 阶段 4.6：用户路名约束 ---
        #
        # 用户提供的路名是**线约束**（不是点约束）——让整条轨迹贴着
        # 用户说的那条路的几何走。实测效果：城市级样本上，轨迹到全路网的
        # 中位距离从 191 米降到 98 米（详见 docs/HANDOVER.md 21.9）。
        #
        # 为什么放在路网精修之后：路网精修是「自由优化」，路名约束是
        # 「带用户意图的强约束」，后者应该压过前者。
        road_name_constraints: list[dict[str, Any]] = []
        road_name_fit: dict[str, Any] = {}
        if road_names:
            road_name_constraints, road_name_fit = self._apply_road_names(
                reg, track, road_names, city, allow_online
            )
            for n in road_name_fit.get("notes", []):
                warnings.append(n)
            # 路名约束改变了轨迹位置 → 重新做一次路网校验，
            # 否则 road_fit 里还是**平移前**的贴路距离，
            # 置信度会用过期数据算（实测偏低 0.18）。
            if road_name_fit.get("ok") and road_net is not None:
                try:
                    road_fit = self._verify_roads(reg, track, road_net)
                    warnings.append(
                        f"路名约束后复核：贴路中位 "
                        f"{road_fit.get('median_after_m', 0):.0f} 米"
                    )
                except Exception:
                    pass

        # --- 阶段 5：折线投影 ---
        # 输出密度按「米」而非「像素」控制：目标每点约 10 米。
        # 1 像素 ≈ 5.5 米，故步长取 2 像素。简化容差同样以米计（2 米），
        # 否则会把轨迹的弯道拉直（早期用 1 像素容差，轨迹被削掉 19% 长度）。
        #
        # 注意：reg.scale 是「墨卡托米/像素」。墨卡托投影在纬度 φ 处会把
        # 地面距离放大 1/cos(φ)，因此真实地面距离 = 墨卡托距离 × cos(φ)。
        # 里程用 haversine 计算（已含该修正），不要用 像素长 × scale 估算。
        #
        # 尺度自适应有个陷阱：高倍截图（尺度小）时，2 米的容差换算成像素
        # 会变得很大。实测长宁剑河路截图 scale=0.26 m/px，容差变成 7.7 像素，
        # 把所有弯道都削平了——1990 像素的轨迹最后只剩 6 个点。
        # 因此容差必须同时受「像素上限」约束：不超过 2 像素，
        # 否则再密的采样也保不住形状。
        #
        # 采样步长取 5 米（而不是 10 米）：10 米下长直线段会被压得很稀
        # （实测对角线样本只留下 77 个点，低于验收要求的 100 个），
        # 而轨迹点在 GeoJSON 里没有任何成本压力，多留一些更稳妥。
        meters_per_px = max(reg.scale, 1e-6)
        step_px = min(max(1.0, 5.0 / meters_per_px), 4.0)
        eps_px = min(1.0 / meters_per_px, 1.0)
        sampled = resample_polyline(track.polyline, step_px=step_px)
        simplified = simplify_polyline(sampled, eps=eps_px)
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
        confidence = self._score_confidence(
            reg, track, anchors, pairs, road_fit, road_name_fit
        )

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
            road_name_constraints=road_name_constraints,
            road_name_fit=road_name_fit,
            alternatives=alternatives,
            diagnostics=_base_diagnostics(
                # 注意：下面是「墨卡托米/像素」，不是地面米/像素。
                # 地面距离 = 墨卡托距离 × cos(纬度)，在本例纬度约差 15%。
                scale_mercator_m_per_px=round(reg.scale, 4),
                scale_ground_m_per_px=round(
                    reg.scale * math.cos(math.radians(_mean_lat(lonlat_gcj))), 4
                ),
                rotation_deg=round(reg.rotation_deg, 3),
                model=reg.model,
                rmse_m=round(reg.rmse_m, 2),
                anchor_backproj_px=round(med_px, 1),
                inliers=len(reg.inliers),
                outliers=len(reg.outliers),
                ocr_texts=len(texts),
                anchor_candidates=len(anchors),
                track_points_px=len(track.polyline),
                bbox_px=list(track.bbox),
                start_marker=list(track.start_marker) if track.start_marker else None,
                end_marker=list(track.end_marker) if track.end_marker else None,
                online_geocoding=bool(self.cfg.has_online_key and allow_online),
                lexicon=self.lexicon.stats(),
                alternatives=len(alternatives),
            ),
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
        # 上限从 14 提到 20：城市级低缩放截图里能用的锚点本来就稀疏
        # （整张图只有十几个标注），截掉一半会直接让配准失去约束。
        # 名额给到 20 是权衡 Nominatim 1 次/秒的限速——命中缓存后
        # 二次识别几乎不耗时间。
        targets = targets[:20]

        matched = 0
        matched_names: set[str] = set()
        # 未命中词典/在线服务的地名，交给冷启动批量处理
        pending: list[Anchor] = []
        # 词典已命中、但仍要额外查冷启动的地名。
        #
        # 为什么不能「词典命中就直接返回」：词典里的短条目可能是
        # OCR 截断产生的残片（实测「上海世」来自「上海西站」），
        # 其坐标属于另一个地点，偏离可达 6.8 公里。
        # 冷启动往往能查到正确坐标（上海世贸商城 → 121.40087,31.20189），
        # 但被「词典已命中」挡在门外。
        #
        # 正确做法：两边的候选**都**交给 RANSAC，由几何一致性挑选。
        # RANSAC 本来就是干这个的——它能识别出哪一组候选在像素上自洽。
        pending_extra: list[Anchor] = []
        for a in targets:
            # ⚠️ 这里**不能**「够用即停」。
            #
            # 早期实现有一个 break：一旦匹配到 5 个名字就停止处理剩余锚点。
            # 但不同名字提供的是**不同的空间约束**——5 个内点不一定比
            # 3 个内点更可靠（可能是 3 个恰好共线的小地名）。
            #
            # 实测城市级样本：本地词典里有 43 个条目，其中 10 个
            # （红星美凯龙、花卉市场、长风公、阿拉城、上海会聚…）都能
            # 提供有效约束，却被这个 break 挡在门外，配准只剩 3 个内点、
            # RMSE 160 米；放开后内点涨到 6 个、路名归属也恢复成
            # 安龙路/剑河路/金钟路等正确结果。
            #
            # 词典命中是**纯本地查询、零网络成本**，全部保留没有代价；
            # 真正需要控制的是冷启动的请求量，那部分由 pending 控制。
            results = self.geocoder.geocode(
                a.name, city=city, allow_online=allow_online,
                allow_cold_start=False,
            )
            if not results:
                pending.append(a)
                continue
            matched += 1
            matched_names.add(a.name)
            # 词典里权重最高的候选优先，其余作为备选（最多 3 个）
            for r in results[:3]:
                pairs.append(
                    AnchorPair(
                        name=a.name,
                        px=a.px,
                        py=a.py,
                        weight=a.weight,
                        # 锚点类型以地理编码结果的判定为准（行政区 / 车站 /
                        # 道路 / POI），配准时按类型选内点容差。
                        kind=r.kind if r.kind else a.kind,
                        lon=r.lon,
                        lat=r.lat,
                        match_quality=r.quality,
                    )
                )
            # 词典命中不代表坐标一定对——也让冷启动查一遍，两边候选一起
            # 交给 RANSAC。只有「精确命中」才跳过（那基本不会错）。
            if len(a.name) >= 4 and self.lexicon.lookup_exact(a.name):
                continue
            pending_extra.append(a)

        # --- 冷启动：批量检索未命中的地名 ---
        #
        # 必须**批量**而不是逐个：单个地名无法自证位置——
        # 「花园2期」能匹配到朗香/星宇/尼德兰/三湘盛世四个不同小区的「2期」。
        # 但多个独立检索的地名若彼此都落在 1 公里内，那片区域就是真的，
        # 再用它过滤其余候选。实测浦东一例正是靠这个定位成功的。
        if pending and allow_online and self.cfg.allow_online and city:
            cold_pairs = self._cold_start_pairs(pending, city, warnings)
            pairs.extend(cold_pairs)
            matched_names.update({p.name for p in cold_pairs})
            matched += len({p.name for p in cold_pairs})

        # --- 冷启动（补充）：为词典已命中的锚点也提供候选 ---
        #
        # 词典命中时冷启动查到的可能是**正确**坐标，而词典给的是残片坐标。
        # 两边一起交给 RANSAC，几何一致性会挑出自洽的那一组。
        if pending_extra and allow_online and self.cfg.allow_online and city:
            extra_pairs = self._cold_start_pairs(pending_extra, city, warnings)
            pairs.extend(extra_pairs)

        if not pairs:
            warnings.append(
                "所有锚点均未匹配到坐标：可能是冷启动城市，或未配置地图服务密钥"
                "（设置 AMAP_KEY / TMAP_KEY 环境变量后可启用在线地理编码）"
            )
        else:
            warnings.append(f"{matched}/{len(targets)} 个 POI 锚点匹配到坐标")
        return pairs

    def _cold_start_pairs(
        self, anchors: list[Anchor], city: str, warnings: list[str]
    ) -> list[AnchorPair]:
        """用 OSM 冷启动为未命中的锚点找坐标。"""
        try:
            from .coldstart import get_cold_start

            cold = get_cold_start(str(self.cfg.road_cache_dir / "coldstart.json"))
            # 把像素位置一起传过去：冷启动会用它做「位置一致性校验」，
            # 剔除地理上装得下、但图上装不下的候选组合。
            # 实测上海古北一例：仅按地理距离互证会选错簇心，
            # 把两个正确候选过滤掉；加入像素方位校验后即可纠正。
            pixels = {a.name: (a.px, a.py) for a in anchors}
            found = cold.lookup_many([a.name for a in anchors], city, pixels=pixels)
        except Exception as exc:  # noqa: BLE001 — 冷启动失败不应影响主流程
            warnings.append(f"冷启动地理编码失败：{type(exc).__name__}")
            return []

        if not found:
            warnings.append(
                f"冷启动未找到可用坐标（{len(anchors)} 个地名在 OSM 中无同名或近名要素）。"
                "截图里的地名可能尚未被 OSM 收录——可在地图上点选纠正，或配置 "
                "AMAP_KEY / TMAP_KEY 使用与美团底图同源的在线地理编码"
            )
            return []

        pairs: list[AnchorPair] = []
        for a in anchors:
            places = found.get(a.name) or []
            # 候选多留一些（早期只取 3 个）：正确性由 RANSAC 的几何一致性
            # 决定，候选越全，RANSAC 越有机会挑出自洽的那一组。
            # 实测上海古北一例，截断过早会让唯一正确的两个候选进不来。
            for p in places[:5]:
                pairs.append(
                    AnchorPair(
                        name=a.name,
                        px=a.px,
                        py=a.py,
                        weight=a.weight,
                        # 用 OSM 判定出的真实类型（行政区 / 车站 / 道路 / POI），
                        # 而不是 OCR 的 icon/poi——配准的内点容差按类型区分，
                        # 行政区标注点与真实质心可差 1-2 公里。
                        kind=p.kind if p.kind != "other" else a.kind,
                        lon=p.lon,
                        lat=p.lat,
                        match_quality=0.6 * p.score,
                    )
                )
        return pairs

    def _apply_road_names(
        self,
        reg: Registration,
        track: TrackResult,
        road_names: list[str],
        city: str = "",
        allow_online: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """应用用户提供的路名约束（线约束）。

        两级策略（顺序很重要）：
          1. **起终点锚定**（`app/endpointfit.py`）——解出相似变换，
             修正锚点配准给错的**尺度**。
          2. **平移精修**（`app/roadname_constraint.py`）——在 1 的基础上
             做最后的像素级微调。

        为什么必须先修尺度
        ------------------
        低缩放截图下锚点配准的尺度会偏大 26%（实测 15.73 vs 真值 12.5-13.4），
        轨迹被拉长后东端甩过黄浦江——起点落在浦东，路名归属报出
        「打浦路隧道 / 时光印记大道 / 世博大道 / 东岸绿道」。
        而只允许平移的约束**无法修复尺度错误**，怎么挪都救不回来。

        ⚠️ 平移精修阶段仍然**只平移、不改尺度**：尺度已由起终点锚定
        确定（那是与地标标注无关的精确信号），平移阶段只微调绝对位置。
        """
        from .roadname_constraint import fetch_road_name, refine_by_road_names

        names = [n.strip() for n in (road_names or []) if n and n.strip()]
        if not names:
            return [], {}

        # 检索范围：城市 bbox（拿不到就用当前落点周边）
        bbox = None
        if city:
            try:
                from .coldstart import get_cold_start
                cold = get_cold_start(
                    str(self.cfg.road_cache_dir / "coldstart.json")
                )
                bbox = cold.city_bbox(city)
            except Exception:
                bbox = None
        if not bbox:
            # 用当前落点周边 ±6 公里兜底
            try:
                bx, by, bw, bh = track.bbox
                corners = [
                    reg.pixel_to_lonlat(x, y)
                    for x, y in [(bx, by), (bx + bw, by + bh)]
                ]
                lons = [c[0] for c in corners]
                lats = [c[1] for c in corners]
                from .geo.coords import gcj02_to_wgs84
                w_lo, w_la = gcj02_to_wgs84(min(lons) - 0.06, min(lats) - 0.06)
                w_hi, w_ha = gcj02_to_wgs84(max(lons) + 0.06, max(lats) + 0.06)
                bbox = (w_lo, w_la, w_hi, w_ha)
            except Exception:
                return [], {"ok": False, "notes": ["无法确定路名检索范围"]}

        constraints = []
        info = []
        cache_path = str(self.cfg.road_cache_dir / "roadnames.json")
        roads_cache = str(self.cfg.road_cache_dir / "roads.json")
        for nm in names:
            c = fetch_road_name(nm, bbox=bbox, timeout=60.0,
                                cache_path=cache_path,
                                roads_cache_path=roads_cache)
            constraints.append(c)
            info.append({
                "name": nm,
                "ok": c.ok,
                "n_segments": c.n_segments,
                "total_len_km": round(c.total_len_m / 1000.0, 2),
                "uniqueness": round(c.uniqueness, 3),
                "notes": c.notes,
            })

        usable = [c for c in constraints if c.ok and c.segments]
        if not usable:
            return info, {
                "ok": False,
                "notes": ["用户提供的路名在 OSM 里查不到几何，无法作为约束"],
            }

        notes: list[str] = []
        anchor_fit: dict[str, Any] = {}

        # --- 第一级：起终点锚定（修尺度） ---
        if track.start_marker and track.end_marker and len(usable) >= 3:
            try:
                from .endpointfit import fit_by_endpoints
                px = np.array(track.polyline, dtype=float)
                arc = np.concatenate([[0.0], np.cumsum(
                    np.linalg.norm(np.diff(px, axis=0), axis=1))])
                fit = fit_by_endpoints(
                    px, arc,
                    (float(track.start_marker[0]), float(track.start_marker[1])),
                    (float(track.end_marker[0]), float(track.end_marker[1])),
                    [(c.name, c.segments) for c in usable],
                    weights=[c.uniqueness for c in usable],
                )
                if fit.ok:
                    old_scale = reg.scale
                    # 锚定结果直接替换配准矩阵（尺度 + 旋转 + 平移一起换）
                    reg.matrix = fit.matrix
                    reg.translation = fit.translation
                    notes.extend(fit.notes)
                    anchor_fit = {
                        "ok": True,
                        "scale_before": round(float(old_scale), 4),
                        "scale_after": round(float(fit.scale), 4),
                        "scale_ratio": round(
                            float(fit.scale / max(old_scale, 1e-9)), 4),
                        "rotation_deg": round(float(fit.rotation_deg), 3),
                        "start_road": fit.start_road,
                        "end_road": fit.end_road,
                        "chain_residual_m": round(float(fit.value), 1),
                    }
            except Exception as exc:  # noqa: BLE001
                notes.append(
                    f"起终点锚定失败（{type(exc).__name__}），退回平移约束")

        traj = np.array([
            reg.pixel_to_mercator(float(px), float(py))
            for px, py in track.polyline
        ])
        shift, val, notes2, base_val = refine_by_road_names(traj, usable)
        notes.extend(notes2)
        if not np.isfinite(val):
            return info, {"ok": False, "notes": notes, "anchor_fit": anchor_fit}

        # 目标值改善程度 → 采纳判定
        base_note = notes[0] if notes else ""
        # 重新算一次原位目标值做对比
        from .roadname_constraint import _dist_to_segments
        step_idx = max(1, len(traj) // 150)
        sample = traj[::step_idx]
        base_val = float(np.mean([
            float(np.median(_dist_to_segments(sample, c.segments)))
            for c in usable
        ]))
        # 采纳门槛：目标值必须改善到原来的 0.85 倍以下。
        #
        # ⚠️ 为什么不是 0.7：路名约束的天然上限受限于「用户说的路」
        # 与「真实骑行轨迹」不完全重合——骑行会穿过小路、辅路，
        # 轨迹不会精确压在路名对应的车道几何上。实测城市级样本
        # 3085 → 1586 米（0.51 倍）已经是很好的结果，但若门槛定 0.7
        # 就会被拒掉，约束形同虚设。
        #
        # 0.85 的含义：至少改善 15% 才认为「这条约束确实在起作用」。
        # 若用户给错了路名，目标值不会改善，会被拒掉。
        improved = base_val > 0 and val < base_val * 0.85

        if improved:
            reg.translation = reg.translation + shift
            notes.append(
                f"已采纳路名约束：轨迹整体平移 东{shift[0]:+.0f} "
                f"北{shift[1]:+.0f} 米"
            )
        else:
            notes.append(
                f"未采纳路名约束（目标值 {base_val:.0f} → {val:.0f} 米，"
                f"改善不足，可能是路名与轨迹不匹配）"
            )

        # --- 跨江护栏 ---
        #
        # 骑行轨迹横穿黄浦江几乎一定是错的。本例中锚点配准把尺度算大 26%，
        # 轨迹被拉长后东端甩过江，起点落在浦东侧。护栏的作用是**兜底**：
        # 即使上面的算法再出 bug，也不该输出一条横穿长江的骑行轨迹。
        water_check: dict[str, Any] = {}
        try:
            from .waterguard import check_crossing
            lonlat_now = [
                reg.pixel_to_lonlat(float(px), float(py))
                for px, py in track.polyline
            ]
            crossed, which, ratio = check_crossing(
                lonlat_now,
                city=city or "上海",
                cache_dir=str(self.cfg.road_cache_dir),
                allow_network=allow_online,
            )
            water_check = {
                "crossed": bool(crossed),
                "waters": which,
                "ratio": round(float(ratio), 4),
            }
            if crossed:
                notes.append(
                    f"⚠️ 轨迹疑似横跨{which}（{ratio * 100:.0f}% 的点在河面上），"
                    f"结果可能不可靠——请核对起终点是否在江的同一侧"
                )
        except Exception as exc:  # noqa: BLE001
            # 护栏失败不应该让整个识别挂掉，但必须留下痕迹——
            # 否则「护栏没生效」和「护栏判定未横跨」无法区分。
            # ⚠️ 这里**不能**用 `except Exception: pass`：
            # 实测踩过——参数名写错（allow_network vs allow_online）
            # 触发 NameError，被静默吞掉，护栏形同虚设却毫无迹象。
            notes.append(f"跨江检查未能完成（{type(exc).__name__}）")

        return info, {
            "ok": bool(improved),
            "shift_e_m": round(float(shift[0]), 1),
            "shift_n_m": round(float(shift[1]), 1),
            "before_m": round(base_val, 1),
            "after_m": round(val, 1),
            "n_constraints": len(usable),
            "anchor_fit": anchor_fit,
            "water_check": water_check,
            "notes": notes,
        }

    def _collect_alternatives(
        self, reg: Registration, track: TrackResult,
        output_crs: str = "WGS-84",
    ) -> list[dict[str, Any]]:
        """收集备选位置（低置信度时给用户「换个位置」）。

        为什么要有这个：低缩放截图（一张图覆盖 20 公里）下，几何信号
        本质上是多峰的——底图匹配实测前 12 个候选的 Chamfer 分数差
        只有 0.1%（3.52 vs 3.52 米），但地理位置分散在 900-1600 米。
        只输出最优解等于**替用户乱猜**。

        产出格式与 anchors_used 一致，方便前端复用渲染逻辑。
        """
        out: list[dict[str, Any]] = []
        for i, alt in enumerate(getattr(reg, "alternatives", []) or []):
            try:
                pts = apply_to_polyline(alt, track.polyline)
                if not pts:
                    continue
                # ⚠️ 备选坐标用 **GCJ-02**：前端地图底图是腾讯地图（GCJ-02），
                # 主绘制路径也优先用 geojson_gcj02。若这里输出 WGS-84，
                # 切换候选后轨迹会整体偏移约 290 米。
                start = pts[0]
                end = pts[-1]
                out.append({
                    "index": i + 1,
                    "lon": round(float(start[0]), 6),
                    "lat": round(float(start[1]), 6),
                    "end_lon": round(float(end[0]), 6),
                    "end_lat": round(float(end[1]), 6),
                    "model": alt.model,
                    "inliers": len(alt.inliers),
                    "rmse_m": round(float(alt.rmse_m), 1),
                    "scale_m_per_px": round(float(alt.scale), 3),
                    "anchor_names": [a.name for a in alt.inliers][:8],
                    # 完整轨迹（GCJ-02）：前端切换时直接用它重画，
                    # 不用回服务端重算
                    "coords": [[round(float(x), 6), round(float(y), 6)]
                               for x, y in pts],
                    "crs": "GCJ-02",
                })
            except Exception:
                continue
        return out

    def _road_refine(self, reg: Registration, track: TrackResult):
        """路网约束精修：用「轨迹必须走在真实道路上」纠正配准的绝对位置。

        两种模式，按截图尺度自动切换：

        * **街道级**（尺度 < 5 米/像素）——**修正**模式。
          在保持尺度与朝向的前提下平移轨迹，使贴路距离最小，
          并给出每一段所属的道路名。

        * **城市级低缩放**（尺度 > 8 米/像素）——**校验 + 命名**模式。
          只算贴路距离与路名归属，**不做平移/尺度微调**。

          为什么不做微调：OSM 路网与高德底图本身有 30-100 米的数据差异，
          低缩放下贴路能到 60 米已经是数据源的上限；此时再「优化」只会
          过拟合到路网的局部结构——实测尺度被推到 ×1.08、锚点残差从
          271 米涨到 1692 米，而贴路只从 58 米变到 54 米，**净亏**。
          低缩放下路网真正的价值是①验证定位是否落在合理街区
          ②给出路名归属。

        ⚠️ 低缩放下「贴路距离小」**不能**证明位置正确：主干道网稀疏，
        轨迹整体平移几公里照样能压在另一条平行的主干道上。绝对位置
        仍然只能靠锚点，路网只能验证形状与尺度是否自洽。
        """
        from .roadfit import RoadNetwork, fetch_roads, refine_with_roads

        if not self.cfg.road_fit_enabled:
            return {"ok": False, "notes": ["路网精修已关闭"]}, None

        low_zoom = reg.scale > _ROAD_FIT_MAX_M_PER_PX
        # 路网等级：**主干道 + 次干道 + 三级路 + 居住路**，
        # 但排除 service / footway / path 之类的细碎小路。
        #
        # 两个方向都要试过才知道边界在哪：
        #   * 只留主次干道 → 轨迹附近没有主次干道时，贴路距离被
        #     「最近的主干道有多远」主导（实测 221m）；
        #   * 留全量（含 service/辅路）→ 轨迹「就近吸附」到旁边的
        #     辅路上，贴路卡在 30 多米（实测 33m）；
        #   * 排除细碎小路、保留到居住路 → 贴路 56m（城市级）/ 13m（街道级）。
        # 另外全量路网的数据量大到查询要 40-90 秒，按等级过滤能降到 1/3。
        highway_types = [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "residential", "living_street", "pedestrian",
            "motorway_link", "trunk_link", "primary_link", "secondary_link",
        ]

        # 轨迹像素 bbox → 经纬度 bbox（留 margin 余量覆盖周边路网）
        # 注意 track.bbox 的语义是 (x, y, w, h)，不是 (x0, y0, x1, y1)
        #
        # 坐标系：reg.pixel_to_lonlat() 返回的是 **GCJ-02**（与美团底图同源），
        # 而 OSM 的 bbox 参数期望 **WGS-84**。直接传 GCJ 会让查询中心
        # 偏移约 450 米（东）/200 米（南）——拉回来的路网覆盖的是别的街区。
        # 这里必须先转成 WGS-84 再交给 fetch_roads。
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
            # margin 要够大：轨迹附近的路网覆盖不全时，贴路距离会被
            # 「最近的路有多远」主导，而不是「轨迹有没有压在道路上」。
            # 实测城市级 600m → 贴路 432m，2000m → 贴路 56m。
            margin = 0.02 if low_zoom else 0.006
            gcj_bbox = (min(lons) - margin, min(lats) - margin,
                        max(lons) + margin, max(lats) + margin)
            # GCJ-02 → WGS-84（fetch_roads / OSM API 期望的基准）
            w_lo, w_la = gcj02_to_wgs84(gcj_bbox[0], gcj_bbox[1])
            w_hi, w_ha = gcj02_to_wgs84(gcj_bbox[2], gcj_bbox[3])
            bbox = (w_lo, w_la, w_hi, w_ha)
            if low_zoom:
                # 对齐到固定网格：分块边界随轨迹采样与配准的亚像素变化
                # 而微动，不对齐的话缓存块的「完全包含」判定永远不成立，
                # 每次识别都重新拉一遍路网（实测每次 220 秒）。
                g = 0.005
                bbox = (math.floor(w_lo / g) * g, math.floor(w_la / g) * g,
                        math.ceil(w_hi / g) * g, math.ceil(w_ha / g) * g)
        except Exception as exc:
            return {"ok": False,
                    "notes": [f"bbox 计算失败：{type(exc).__name__}"]}, None

        try:
            roads = fetch_roads(
                bbox, timeout=90.0 if low_zoom else self.cfg.http_timeout * 2,
                cache_path=str(self.cfg.road_cache_dir / "roads.json"),
                highway_types=highway_types,
            )
        except Exception as exc:
            return {"ok": False,
                    "notes": [f"路网获取失败，跳过精修：{type(exc).__name__}: {exc}"]}, None

        net = RoadNetwork(roads)
        if low_zoom:
            fit = self._road_verify(reg, track, net)
            return fit.to_dict(), net

        fit = refine_with_roads(reg, track.polyline, roads)
        if fit.ok:
            # 把平移量并入配准矩阵（墨卡托 y 轴向上，北为正）
            reg.translation = reg.translation + np.array(
                [fit.shift_e_m, fit.shift_n_m], dtype=float
            )
        return fit.to_dict(), net

    def _verify_roads(
        self, reg: Registration, track: TrackResult, net: Any
    ) -> dict[str, Any]:
        """只算贴路距离 + 路名归属，不做位置修正。

        路名约束改变轨迹位置后用它复核——此时轨迹已经落在用户指定的
        路上，不应该再被「路网精修」二次拉动。
        """
        from .roadfit import RoadFit
        fit = RoadFit(source="osm")
        names, d = net.query(reg, track.polyline)
        if len(d) == 0:
            fit.notes.append("路网校验失败：轨迹没有采样到路网点")
            return fit.to_dict()
        median = float(np.median(d))
        fit.median_before_m = median
        fit.median_after_m = median
        fit.segments = net.segments(reg, track.polyline)
        ordered: list[str] = []
        for sg in fit.segments:
            if not ordered or ordered[-1] != sg["road"]:
                ordered.append(sg["road"])
        fit.roads = ordered
        fit.ok = True
        fit.notes.append(f"路网校验：贴路中位 {median:.0f} 米")
        return fit.to_dict()

    def _road_verify(self, reg: Registration, track: TrackResult, net: Any):
        """城市级低缩放：只校验贴路距离 + 给出路名归属，不做位置修正。

        见 `_road_refine` 的说明——低缩放下路网是「校验器 + 命名器」，
        不是「修正器」。
        """
        from .roadfit import RoadFit

        fit = RoadFit(source="osm")
        names, d = net.query(reg, track.polyline)
        if len(d) == 0:
            fit.notes.append("路网校验失败：轨迹没有采样到路网点")
            return fit
        median = float(np.median(d))
        fit.median_before_m = median
        fit.median_after_m = median
        fit.segments = net.segments(reg, track.polyline)
        ordered: list[str] = []
        for sg in fit.segments:
            if not ordered or ordered[-1] != sg["road"]:
                ordered.append(sg["road"])
        fit.roads = ordered
        fit.ok = True
        fit.notes.append(
            f"路网校验：贴路中位 {median:.0f} 米"
            f"（低缩放下只校验不修正——OSM 与高德底图本身差 30-100 米）"
        )
        return fit

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
        road_name_fit: dict[str, Any] | None = None,
    ) -> float:
        """综合置信度。

        构成：几何拟合质量（主导）+ 锚点数量 + 锚点类型质量 + 轨迹提取质量
             + 绝对位置自校验。
        """
        # 拟合质量：残差换算成**像素**再分档。
        #
        # 不能直接用米：同一个 200 米残差，在街道级截图（0.5 米/像素）
        # 是 400 像素、完全不可信；在城市级截图（15 米/像素）只有
        # 13 像素、其实相当准。用米做绝对阈值会让低缩放截图的
        # 置信度被无端压低。
        rmse_m = reg.rmse_reliable_m if reg.n_reliable >= 2 else reg.rmse_m
        med_px_fit = rmse_m / max(reg.scale, 1e-6)
        if med_px_fit <= 3:
            fit = 1.0
        elif med_px_fit <= 8:
            fit = 0.85
        elif med_px_fit <= 20:
            fit = 0.6
        else:
            fit = 0.35

        # 内点占比
        #
        # ⚠️ 分母不能简单用「内点 + 离群」：
        # 离群候选是**我们主动提供给 RANSAC 的备选项**（同名的多个地点、
        # 冷启动查到的多个候选），RANSAC 挑出自洽的那一组后，
        # 剩下的自然都是离群。备选越多，占比越低，置信度越低——
        # 但结果完全正确。
        #
        # 实测浦东：候选放宽后内点仍是正确的 2 个、RMSE 0.6m、贴路 3.3m，
        # 但离群候选从 0 涨到 3，内点占比从 1.0 掉到 0.4，
        # 置信度被无端压低了 0.11。
        #
        # 正确做法：按**每个地名的候选组**算占比——
        # 一个地名若至少有一个候选进了内点，这个地名就算「命中」。
        # 这样离群的是候选，而不是「配错的地名」。
        hit_names = {a.name for a in reg.inliers}
        # 分母只算「能作强约束」的地名。城市级截图上 OCR 会读出
        # 「3号线」「8号线」「外环高速」这类名字，它们指向的是线/路而非
        # 点位，本来就只是弱参考——把它们算进分母会把命中率
        # 从 5/12 压到 5/23，置信度凭空掉一截。
        all_pairs = list(reg.inliers) + list(reg.outliers)
        strong_names = {
            a.name for a in all_pairs if a.kind in _STRONG_ANCHOR_KINDS
        }
        ratio = len(hit_names & strong_names) / len(strong_names) if strong_names else 0.0

        # 锚点数量（越多越稳，但收益递减）
        n = len(reg.inliers)
        count = min(1.0, 0.35 + 0.11 * n) if n else 0.0

        # 锚点质量：按锚点类型加权。
        # 车站（图标锚定真实点位）最可靠；行政区标注点会被地图引擎
        # 挪到视野空白处，质量要打折。
        kinds = [a.kind for a in reg.inliers]
        if "station" in kinds:
            kind_q = 1.0
        elif "poi" in kinds or "icon" in kinds:
            kind_q = 0.9
        elif "district" in kinds:
            kind_q = 0.6
        else:
            kind_q = 0.7

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

        # 用户路名约束生效时，置信度由「路名贴合度」主导。
        #
        # ⚠️ 这一段必须放在路网精修之后：路名约束会把轨迹整体平移
        # （实测 1.5 公里），此时路网精修算出的「贴路中位距离」已经过时，
        # 直接沿用会把置信度算低（实测 0.404 → 0.227，明显不合理——
        # 轨迹明明被用户拉到了正确的路上）。
        if road_name_fit and road_name_fit.get("ok"):
            rn_after = float(road_name_fit.get("after_m", 999))
            rn_before = float(road_name_fit.get("before_m", 999))
            # ⚠️ 分段约束下**不能用绝对值分档**。
            #
            # 「分段目标值」的量纲是「轨迹到各路名对应段的加权中位距离」，
            # 一条 16 公里的轨迹经过 10 条路，每条路上只占一小段，
            # 目标值天然在几百米量级——它衡量的是「整体贴合程度」，
            # 不是「位置误差」。用绝对值分档会把「明明改善了 42%」
            # 的结果判成低质量（实测 1767 米 → rn_q=0.5，置信度反降）。
            #
            # 正确的判据是**改善率**：约束把轨迹拉近了多远。
            rn_gain = 1.0 if rn_before <= 0 else max(
                0.0, min(1.0, (rn_before - rn_after) / max(rn_before, 1)))
            if rn_gain >= 0.5:
                rn_q = 1.0
            elif rn_gain >= 0.3:
                rn_q = 0.92
            elif rn_gain >= 0.15:
                rn_q = 0.8
            else:
                rn_q = 0.65
            # 用户明确给了约束 → 这是**主动信息**，置信度应当更高
            conf = (0.55 * rn_q + 0.20 * ratio + 0.12 * count +
                    0.13 * rn_gain)
            conf *= track.confidence if track.confidence < 1 else 1.0

        # --- 跨江惩罚 ---
        #
        # 骑行轨迹横穿黄浦江/长江几乎一定是错的。
        # 本例中锚点配准把尺度算大 26%，轨迹被拉长后东端甩过江，
        # 起点落在浦东——若不做惩罚，置信度仍有 0.83（看起来很可信），
        # 用户会以为结果是准的。
        #
        # 惩罚力度：命中则**减半**。因为过桥是合法的（外白渡桥等），
        # 不能直接归零；但横跨大河确实大幅降低了结果的可信度。
        water = (road_name_fit or {}).get("water_check") or {}
        if water.get("crossed"):
            conf *= 0.5

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
