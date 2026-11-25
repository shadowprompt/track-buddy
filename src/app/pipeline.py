"""识别流水线：从截图到 GeoJSON 轨迹。"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from .calibration import apply_calibration
from .config import AppConfig
from .cv_utils import crop_map_region, resample_polyline, simplify_polyline
from .geocode import Geocoder, Lexicon
from .geo.coords import gcj02_to_wgs84, lonlat_to_mercator
from .input_quality import (
    assess_input_quality,
    extract_distance_column,
    extract_distance_km,
    route_validation_factor,
    scale_consistency,
)
from .anchor_filter import ANCHOR_FILTER_ENV, get_filter
from .ocr_anchor import Anchor, build_anchors, classify_text, ocr_texts
from .register import (
    AnchorPair,
    Registration,
    apply_to_polyline,
    register,
    register_with_hint,
)
from .route_template import (
    SHAPE_DOMAIN_OVERVIEW,
    TEMPLATE_CONFIDENCE_CAP,
    TemplateStore,
    polyline_length_m,
)

# 非宽泛图的模板采信门槛：「近复刻」才直跳。实测同一张概览图重传时
# residual≈0.0116，而不同路线即便形状相近也**通常**在 0.05 以上；0.045
# 与 route_template.DEDUP_RESIDUAL 对齐，覆盖率 0.90 表示整条折线都参与
# 匹配。但 2026-11-14 18:18 实测证伪了「0.05 以上」的假设：上海一条
# 不同路线与历史模板**反向**匹配 residual 仅 0.0266、coverage 0.98——
# 两条不同的长路线在 64 点归一化形状空间里可以非常接近（尤其走向相近
# 的带弯长条形）。因此追加两道门：
NEAR_HIT_RESIDUAL = 0.045
NEAR_HIT_COVERAGE = 0.90
# 方向签名（direction_signature 距离）：实测同类（同路线重传）≈0.00、
# 上面那次假阳性 0.3632。0.30 仍给真实复刻留足余量（渲染差异）。
NEAR_HIT_SIGNATURE = 0.30
# 反向匹配（reversed=True）：同一条路线反向重骑是合法场景，但归一化
# 形状空间里反向匹配的假阳性率显著更高（本次事故即 reversed），门槛
# 收紧到正向的一半——真复刻（正/反向）实测 residual 都在 0.01 量级，
# 0.025 不会误伤真实反骑复用。
NEAR_HIT_REVERSED_RESIDUAL = 0.0225


def _near_template_hit(located: dict[str, Any]) -> bool:
    """非宽泛概览图是否达到「近复刻」采信门槛。

    宽泛概览图走 locate() 自身的门槛即可；普通截图（街道名较多、
    OCR 判不出宽泛概览）只有在形状与已注册概览模板**几乎一致**时才
    采信模板结果，否则一律回落原流程——普通截图本来就有机会拿到
    带当次地理核验的正式配准，比参考级插值更可信。

    门槛三道（全部通过才算近复刻）：

    * residual ≤ NEAR_HIT_RESIDUAL（反向匹配再收紧一半）；
    * coverage ≥ NEAR_HIT_COVERAGE；
    * 方向签名差 ≤ NEAR_HIT_SIGNATURE——形状距离度量不了「走法」，
      两条不同的路可以几何形状相近，但转向序列不会。
    """
    m = located.get("match") or {}
    try:
        residual = float(m.get("residual", 1.0))
        coverage = float(m.get("coverage", 0.0))
    except (TypeError, ValueError):
        return False
    if coverage < NEAR_HIT_COVERAGE or residual > NEAR_HIT_RESIDUAL:
        return False
    try:
        sig = float(m.get("signature", 1.0))
    except (TypeError, ValueError):
        sig = 1.0
    if sig > NEAR_HIT_SIGNATURE:
        return False
    if bool(m.get("reversed")) and residual > NEAR_HIT_REVERSED_RESIDUAL:
        return False
    return True
from .template_fallback import assess_broad_overview, reference_confidence
from .track import TrackResult, extract_track

# 能作为「强约束」的锚点类型。地铁线号（transit）与高架路（road）
# 天然指向一段线或一条路，与标注点的对应关系很弱；它们不匹配不代表
# 定位失败，不该计入置信度的分母。
_STRONG_ANCHOR_KINDS = ("poi", "station", "icon", "district")


def _mean_lat(points: list[tuple[float, float]]) -> float:
    """轨迹点的平均纬度（用于墨卡托尺度换算）。"""
    if not points:
        return 0.0
    return sum(p[1] for p in points) / len(points)


def _candidate_shift_px(reg: Registration,
                        road_fit: dict[str, Any]) -> dict[str, Any]:
    """把「被否决的精修平移量」换算成截图上的像素位移。

    配准是 ``f(p) = M·p + t``。精修想让同一条像素轨迹对应到
    ``f(p) + d`` 的地理位置；以截图底图为固定参照，这个候选位置应当
    画在 ``p + M⁻¹·d``。前端把主轨迹每个像素点加上该增量即可画出
    未采纳的路线，不必知道尺度、旋转或坐标系，也不必再传一份折线。

    没有候选位移或配准矩阵退化时返回空字典（前端就不画线）。
    """
    de = road_fit.get("candidate_shift_e_m")
    dn = road_fit.get("candidate_shift_n_m")
    matrix = getattr(reg, "matrix", None)
    if de is None or dn is None or matrix is None:
        return {}
    try:
        dpx, dpy = np.linalg.solve(
            np.asarray(matrix, dtype=float),
            np.array([float(de), float(dn)], dtype=float),
        )
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return {}
    if not np.all(np.isfinite([dpx, dpy])):
        return {}
    return {"candidate_shift_px": [round(float(dpx), 2),
                                   round(float(dpy), 2)]}


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
    # HMM 地图匹配的**备选显示层**（默认关闭）。主 GeoJSON / 置信度 /
    # 里程一律不改，这里只是多给一份「沿路折线」供人工对照。
    map_matched: dict[str, Any] = field(default_factory=dict)
    # 后台精修要复用的中间产物（只存在进程内，不进接口 payload）。
    # ``_build_payload*`` 是显式字段白名单，所以这里不会被序列化。
    refine_context: Any = None


@dataclass
class RefineContext:
    """主请求留下的中间产物，供后台精修复用。

    后台精修原先会重跑整条 ``run()``：大画布上 OCR、轨迹提取、锚点
    地理编码、配准全部再来一遍（12 图拼接实测多花 4 分钟），而且
    **在线配额波动会让后台拿到与主请求不同的锚点集合**，出现「主请求
    判该精修、后台却判不该精修」的不稳定结果。

    这里把「阶段 1-4 + 备选位置 + 缩放评估」的结果留下来，后台只补算
    平移量再重投影。字段与 ``run()`` 的局部变量一一对应；刻意不保存
    图片本身——拼接画布动辄几千万像素，常驻内存不划算。
    """

    reg: Any
    track: Any
    simplified: Any                       # np.ndarray，像素空间简化折线
    anchors: list[Any]
    pairs: list[Any]
    texts: list[Any]
    zoom: dict[str, Any]
    alternatives: list[Any]
    road_name_fit: dict[str, Any]
    med_px: float
    city: str = ""
    output_crs: str = "WGS-84"
    allow_online: bool = True
    road_names: list[str] | None = None
    image_size: tuple[int, int] = (0, 0)
    online_geocoding: bool = False


# diagnostics 的完整键集。
#
# 早期实现只在成功路径填充这些字段，失败路径只给 {"track_points": N}，
# 前端于是读到 undefined 再调 .toFixed()，整页崩掉，
# 后端 warnings 里真正的失败原因反而被吞掉——用户只看到
# "Cannot read properties of undefined (reading 'toFixed')"，
# 完全不知道发生了什么。这里用一份默认值把结构固定下来。
DIAG_KEYS: tuple[str, ...] = (
    "scale_mercator_m_per_px",    "scale_ground_m_per_px",
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
    "track_straightness",
    "anchor_fragments",
    "anchor_bias",
    "anchor_sources",
    "scale_check",
    "route_validation",
    "input_quality",
    "map_match",
    "start_marker",
    "end_marker",
    "online_geocoding",
    "lexicon",
    "alternatives",
    "zoom",
    # 置信度是否被真值校准过；以及本轮词典写入的防毒审计。
    # 失败路径也必须带这两个键（前端统一遍历 diagnostics）。
    "confidence_calibration",
    "lexicon_write_guard",
)


def _base_diagnostics(**over: Any) -> dict[str, Any]:
    """返回结构完整的 diagnostics，缺省值用 None（前端显示为「—」）。"""
    from .config import CONFIG  # 延迟 import：避免 config ↔ pipeline 循环

    dg: dict[str, Any] = {k: None for k in DIAG_KEYS}
    dg["lexicon"] = {}
    # 输入源画像（SOURCE_PROFILE）随诊断透传：多来源时代排查「参数用错
    # 画像」的第一线索，一眼可见本次识别走的是哪套视觉参数。
    src = CONFIG.source
    dg["source"] = {"id": src.id, "title": src.title}
    dg.update(over)
    return dg


def _result_is_self_consistent(diag: dict[str, Any]) -> bool:
    """本轮识别结果是否「自洽到可以反哺词典」。

    只看**独立于配准**的证据，不用贴路距离或锚点残差——它们在整体
    平移到平行道路时同样会变好（§22.38），拿来做门槛等于放行假解。

    四项一票否决：低信息量直线输入、截图里程与几何里程量级不符、
    锚点残差方向高度相干（系统性基准偏移）、低缩放且位置未被修正。
    缺省字段一律按「无证据反对」处理，避免因版本差异误杀冷启动。
    """
    iq = diag.get("input_quality") or {}
    scale_check = diag.get("scale_check") or {}
    bias = diag.get("anchor_bias") or {}
    zoom = diag.get("zoom") or {}
    if iq.get("level") == "reject":
        return False
    if not bool(scale_check.get("consistent", True)):
        return False
    if bool(bias.get("systematic", False)):
        return False
    if bool(zoom.get("zoom_too_low", False)) and not bool(
            zoom.get("position_ok", True)):
        return False
    return True


def _track_straightness(polyline_px: Any) -> float:
    """折线的「直线度」= 端点弦长 / 折线总长，1.0 表示完全笔直。

    美团在部分页面（路线概览、低缩放）会把骑行轨迹渲染成一条带箭头的
    **起终点连线**，而不是沿道路曲折的实际轨迹。这类输入无论配准多准、
    路网精修是否采纳，输出的经纬度都只能是一条横穿街区的直线。

    判据与截图分辨率、城市、坐标都无关，只看形状本身；比按
    「米/像素」判缩放级别更能抓住「轨迹本身没有信息量」这种情况
    （实测一例街道级 2.38 m/px，缩放评估判定正常，但折线直线度 0.97）。
    """
    pts = np.asarray(polyline_px, dtype=float)
    if len(pts) < 3:
        return 0.0
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if length < 1e-6:
        return 0.0
    return float(min(1.0, chord / length))


def _track_lacks_shape_info(
    polyline_px: Any, bbox: Any, has_road_names: bool
) -> bool:
    """是否属于「轨迹被渲染成起终点连线」的低信息量输入。

    单独用直线度不够：一段较长但确实沿直路骑的真实轨迹，弦长/总长
    也能到 0.87（实测 12 图拼接样本 0.869）。三个条件必须同时成立：

    * 折线整体贴近端连线（直线度 ≥ 0.80）；
    * 轨迹框极度扁长（长边 / 短边 ≥ 4，实测起终点连线 7.5、
      12 图拼接真实轨迹 2.0）；
    * 截图里**没有可辨认的街道名锚点**——这是「看不到路、无法核对走向」
      的直接证据。

    有街道名锚点时交由缩放评估与路名约束处理，不在这里下结论。
    """
    pts = np.asarray(polyline_px, dtype=float)
    if has_road_names or len(pts) < 20:
        return False
    try:
        _, _, bw, bh = bbox
        ratio = max(float(bw), float(bh)) / max(1.0, min(float(bw), float(bh)))
    except (TypeError, ValueError):
        return False
    return _track_straightness(pts) >= 0.80 and ratio >= 4.0


# 「大范围对象」类型：行政区、地铁线路。
# 这类标注的文字位置与真实点位可差 1-16 公里，无法支撑街区级定位。
_BROAD_KINDS = ("district", "transit")

# 高速公路 / 国道 / 高架——名字里含这些的「道路」同样是几十公里级的大对象，
# 不能算作街道级锚点（实测「外环高速」偏差 3087 米）。
# ⚠️ **不要排除「大道」**：世纪大道/龙东大道虽是长道路，但在城市里
# 仍是有效的定位参考（用户提供的路名里就有「中山南二路」这类）。
_EXPRESSWAY_MARKERS = ("高速", "国道", "省道", "高架", "环线")


def prepare_input_image(
    img: np.ndarray, vision: Any
) -> tuple[np.ndarray, dict]:
    """画像级输入准备：地图区裁剪 + 分辨率归一化。

    返回 ``(处理后的图, meta)``，meta 结构::

        {"crop_v": tuple | None,      # 生效的 map_crop_v（原图高度比例）
         "prepared_hw": [h, w] | None} # 归一化后的尺寸（未缩放则 None）

    meta 随 diagnostics 下发（``vision_prep``；裁剪比例另发兼容键
    ``map_crop_applied``），visuals 对回读原截图做同样变换以对齐像素
    坐标空间。

    两步变换的顺序约定：**先裁剪后缩放**——crop_v 比例相对整图高度，
    在原图坐标系里解释；缩放只改变内容尺度，两者互不干扰。

    分辨率归一化（``vision_ref_short_side``）：把输入等比缩放到画像
    标定样本的短边。画像里一批按标定分辨率实测的绝对像素参数（徽章
    面积窗口、行者 scale_badge_d 锁定值等）由此在任意分辨率下都回到
    标定状态。2026-11-24 实测：小米 1.25× 等比截图上抹除半径/容差
    全部失准，弧长少 25%、配准尺度爆炸。

    为什么放在 pipeline 入口而不是各调用方：server 已经在拼接前逐图
    准备（传 ``img_is_prepared=True`` + ``prep_meta`` 跳过），而
    CLI / 直跑链路此前**完全不裁**——小米/高德整屏截图下部的概要
    卡片与轨迹同色系，速度图例条（绿→红渐变）被当成起终点徽章、
    卡片文案被当地理编码锚点，配准只能靠 UI 文本支撑（2026-11-23
    小米实测）。
    """
    meta: dict = {
        "crop_v": None, "prepared_hw": None,
        # 原始输入尺寸：API 响应层把像素坐标反投影回原图时必需
        #（前端「原始截图」叠绘画在完整原图上，见 backproject_prep_px）。
        "original_hw": [int(img.shape[0]), int(img.shape[1])],
    }
    frac = getattr(vision, "map_crop_v", None)
    if frac:
        img = crop_map_region(img, vision)
        meta["crop_v"] = tuple(float(x) for x in frac)
    ref = getattr(vision, "vision_ref_short_side", None)
    if ref:
        short = min(img.shape[:2])
        if abs(short - int(ref)) > 2:  # 2px 内不缩，避免无谓重采样
            s = float(ref) / short
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
    # ⚠️ prepared_hw 必须始终记录实际尺寸（裁剪也算变换，不止缩放）。
    # None 会让 API 响应层跳过像素坐标反投影——864 标定分辨率输入
    # 恰好不触发缩放，但 image_size 仍需回到原图高度（2026-11-25
    # 用户实测：绿线纵向拉伸 1.85×，因为前端拿裁剪高 1037 当原图高）。
    meta["prepared_hw"] = [int(img.shape[0]), int(img.shape[1])]
    return img, meta


def backproject_prep_px(
    points: Any,
    crop_v: tuple[float, float] | None,
    prepared_hw: list[int] | None,
    original_hw: list[int] | None,
) -> Any:
    """把 prepared（裁剪+归一化）像素坐标反投影回**原图**坐标系。

    正向变换（prepare_input_image）：先按 ``crop_v`` 裁出 ``ch`` 行，
    再等比缩放 ``s`` 得到 prepared。反投影即：
    ``orig_x = px / s``，``orig_y = py / s + y0``。

    为什么必须有它：前端「原始截图」叠绘把像素坐标画在**完整原图**
    上，而管线内部的像素坐标全部在 prepared 空间。裁剪不切宽度时
    横向凑巧不偏，但纵向比例 = 原图高 / 裁剪高（小米 1920/1037 ≈
    1.85×）——绿线纵向被拉伸、环下坠、尾线冲出红标（2026-11-25
    用户实测两次复现）。模板/内部链路继续用 prepared 坐标，只有
    API 响应层做此换算。

    ``original_hw`` 缺失（旧结果/模板派生）时返回 ``None``，调用方
    保持原字段不动（兼容旧数据）。
    """
    if not original_hw or not prepared_hw:
        return None
    try:
        oh, ow = int(original_hw[0]), int(original_hw[1])
        ph, pw = int(prepared_hw[0]), int(prepared_hw[1])
    except (TypeError, ValueError, IndexError):
        return None
    if min(oh, ow, ph, pw) <= 0:
        return None
    s = ow / pw
    y0 = float(crop_v[0]) * oh if crop_v else 0.0

    def _bp(p: Any) -> Any:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                return [float(p[0]) / s, float(p[1]) / s + y0]
            except (TypeError, ValueError):
                return p
        return p

    def _walk(x: Any) -> Any:
        # 折线是 [[x, y], ...]；锚点是 {"px":..,"py":..}
        if isinstance(x, list):
            if x and isinstance(x[0], (list, tuple)):
                return [_bp(p) for p in x]
            return _bp(x)
        if isinstance(x, dict):
            out = dict(x)
            if "px" in out and "py" in out:
                bp = _bp([out["px"], out["py"]])
                out["px"], out["py"] = bp[0], bp[1]
            return out
        return x

    return _walk(points)


class RidePipeline:
    """完整的识别流水线。"""

    def __init__(self, cfg: AppConfig, *,
                 templates: TemplateStore | None = None):
        self.cfg = cfg
        # P1 的轨迹模板库由 server 注入；单元测试不注入时整条兜底通道
        # 完全关闭，保证既有行为一字不变。
        self.templates = templates
        self.lexicon = Lexicon(cfg.lexicon_path)
        self.geocoder = Geocoder(cfg, self.lexicon)
        # 文本有效性过滤器（可插拔）：整条流水线共用一个实例，未来换成
        # LLM 实现时可复用其缓存/会话。
        # 选择优先级由 get_filter 统一处理：显式环境变量 ANCHOR_FILTER >
        # cfg.anchor_filter > 默认 "rule"。不传 name 让它自己读环境变量。
        _env_filter = os.environ.get(ANCHOR_FILTER_ENV, "").strip()
        self.anchor_filter = get_filter(
            _env_filter or getattr(cfg, "anchor_filter", None)
        )
        # 最近一次 _score_confidence 的校准状态，写进 diagnostics。
        self._last_calibration: dict[str, Any] = {
            "enabled": False, "status": "uncalibrated", "n_samples": 0,
        }
        # 最近一次配准的内点地名（规范化后），供词典写入做锚点级证据裁决。
        self._last_anchor_keys: set[str] = set()
        # 最近一次 run() 实际生效的地图区裁剪比例（None = 未裁）。
        # server 调用方自行裁剪（img_is_prepared=True），CLI / 直跑由
        # run() 内部补裁；visuals 据此对齐像素坐标空间。
        self._last_prep: dict = {"crop_v": None, "prepared_hw": None}
        # 大模型是**可选增强**：没配 LLM_BASE_URL/LLM_API_KEY 就完全跳过，
        # 现有行为一字不变。
        self.llm = None
        try:
            from .llm import LLMClient
            if getattr(cfg, "llm_enabled", True):
                self.llm = LLMClient(cfg)
        except Exception:  # noqa: BLE001
            self.llm = None

    def run(
        self,
        img: np.ndarray,
        city: str = "",
        allow_online: bool = True,
        output_crs: str = "WGS-84",
        refine: bool = True,
        road_names: list[str] | None = None,
        skip_template_recall: bool = False,
        distance_hint_km: float | None = None,
        img_is_prepared: bool = False,
        prep_meta: dict | None = None,
    ) -> PipelineResult:
        """识别入口：外层只负责「词典写入会话」的生命周期。

        在线地理编码 / 冷启动发生在**配准之前**，那时还不知道这次识别
        准不准。若低置信、低信息量的结果也照单全收，错误坐标会一次次
        加权，形成「越用越偏」的词典中毒（交接文档 §22.40）。

        因此本轮所有**自动**写入先排队，等 ``_run_impl()`` 算出最终
        置信度与自洽诊断后再统一裁决落盘；用户显式纠正不排队、立即写。

        ``img_is_prepared``：调用方是否已经做过画像级的输入准备（地图区
        裁剪 + 分辨率归一化，见 ``prepare_input_image``）。server 在拼
        接前逐图准备，传 True 避免二次处理；CLI / 直跑 pipeline 的调用
        方不准备，这里统一补上——小米/高德这类整屏截图的概要卡片与轨迹
        同色系（速度图例条就是绿→红渐变），不裁会被当成起终点徽章与锚
        点（2026-11-23 小米实测：图例条两端被定位成起(213,1335)/终
        (612,1335)，配准仅靠「活动卡路里/运动时长」两个 UI 文本支撑，
        RMSE 260m）。

        ``prep_meta``：``img_is_prepared=True`` 时调用方准备时拿到的
        meta（``prepare_input_image`` 的返回值第二项），用于把生效的
        裁剪/归一化写进 diagnostics——server 路径此前不记录裁剪，
        overlay 反投影会整体画偏（裁掉多少偏多少）。
        """
        prep: dict = {"crop_v": None, "prepared_hw": None, "original_hw": None}
        summary_strip: np.ndarray | None = None
        if not img_is_prepared:
            full_img = img
            img, prep = prepare_input_image(img, self.cfg.vision)
            # 被裁掉的底部统计条保留：没传概要图时它是里程/时长先验的
            # 唯一来源（小米「13.09 公里 / 00:59:52」就印在主截图底部）。
            _frac = getattr(self.cfg.vision, "map_crop_v", None) or None
            if _frac:
                try:
                    _y0 = int(float(_frac[1]) * full_img.shape[0])
                    if full_img.shape[0] - _y0 >= 40:
                        summary_strip = full_img[_y0:]
                except Exception:  # noqa: BLE001 — 兜底失败不影响主流程
                    summary_strip = None
        elif prep_meta:
            prep = {
                "crop_v": prep_meta.get("crop_v"),
                "prepared_hw": prep_meta.get("prepared_hw"),
                "original_hw": prep_meta.get("original_hw"),
            }
        if distance_hint_km is None and summary_strip is not None:
            try:
                from .ride_summary import parse_summary_from_region

                _fs = parse_summary_from_region(summary_strip)
                if _fs and _fs.get("distance_km"):
                    distance_hint_km = float(_fs["distance_km"])
            except Exception:  # noqa: BLE001
                pass
        self._last_prep = prep
        self._last_calibration = {
            "enabled": False, "status": "uncalibrated", "n_samples": 0,
            "note": "本次未计算置信度（早退路径）",
        }
        # 每次识别重置：早退/异常路径没有内点证据，自动写入应全部拒绝。
        self._last_anchor_keys = set()
        # 锚点地理编码诊断明细（_resolve_anchors 填充，进 diagnostics）
        self._last_geocode_report = []
        self.lexicon.begin_capture()
        try:
            result = self._run_impl(
                img, city=city, allow_online=allow_online,
                output_crs=output_crs, refine=refine, road_names=road_names,
                skip_template_recall=skip_template_recall,
                distance_hint_km=distance_hint_km,
            )
        except BaseException:
            self.lexicon.discard_capture()
            raise

        diag = dict(result.diagnostics or {})
        if self._last_prep.get("crop_v"):
            # 记录生效的裁剪比例：visuals（overlay / road_verify / chart）
            # 据此对回读的原截图做同样的裁剪，保证像素坐标空间一致。
            diag["map_crop_applied"] = [
                float(x) for x in self._last_prep["crop_v"]
            ]
        if self._last_prep.get("prepared_hw"):
            # 分辨率归一化后的尺寸：visuals 把原图裁剪区缩放到该尺寸，
            # 反投影像素坐标才与配准模型对齐。
            diag["vision_prep"] = {
                "prepared_hw": [
                    int(self._last_prep["prepared_hw"][0]),
                    int(self._last_prep["prepared_hw"][1]),
                ],
                # 原始输入尺寸：API 响应层把像素坐标反投影回原图坐标系
                # 时必需（前端「原始截图」叠绘，见 backproject_prep_px）。
                "original_hw": (
                    [int(self._last_prep["original_hw"][0]),
                     int(self._last_prep["original_hw"][1])]
                    if self._last_prep.get("original_hw") else None
                ),
            }
        iq = diag.get("input_quality") or {}
        guard = self.lexicon.flush_capture(
            confidence=result.confidence,
            self_consistent=_result_is_self_consistent(diag),
            input_ok=bool(iq.get("usable", True)),
            anchor_keys=self._last_anchor_keys,
        )
        diag["lexicon_write_guard"] = guard
        diag["confidence_calibration"] = self._last_calibration
        diag["lexicon"] = self.lexicon.stats()
        result.diagnostics = diag
        return result

    def _template_result(
        self,
        located: dict[str, Any],
        *,
        img: np.ndarray,
        track: TrackResult,
        texts: list[Any],
        broad: dict[str, Any],
        output_crs: str,
        t0: float,
        warnings: list[str],
    ) -> PipelineResult:
        """构造「参考级」结果：坐标来自历史同类轨迹，没有当次地理核验。

        刻意把 ``input_quality.usable`` 设为 False：既让 ``flush_capture``
        拒绝本轮词典自动写入，也让 server 的 ``_capture_template`` 不会
        把模板推导出的坐标再沉淀成新模板（否则错误会自我强化）。
        不设 ``scale_ground_m_per_px``，后台精修判据据此直接跳过。
        """
        coords_gcj = [tuple(float(x) for x in p) for p in located["coords_gcj"]]
        crs_norm = str(output_crs or "").upper().replace("-", "")
        use_wgs = crs_norm in ("WGS84", "WGS")
        lonlat = ([gcj02_to_wgs84(lo, la) for lo, la in coords_gcj]
                  if use_wgs else coords_gcj)
        main_crs = "WGS-84" if use_wgs else "GCJ-02"
        match = located["match"]
        conf = reference_confidence(match, cap=TEMPLATE_CONFIDENCE_CAP)
        if broad.get("broad_overview"):
            why = ("本图地名多为行政区/地铁线/高架，空间精度只有公里级，"
                   "本次未做地理配准")
        else:
            why = ("本图轨迹与一条历史路线的形状高度一致（近复刻命中），"
                   "本次未做地理配准")
        ref_warns = list(warnings) + [
            f"⚠️ 参考级定位（未核验）：{why}；坐标是把轨迹形状对齐到"
            "一条历史路线后按弧长插值得到的，仅供轮廓参考，"
            "不能当作精确轨迹使用。",
            "如需街区级精度，请放大地图到能看到街道名后重新截图，"
            "或用人工核验反馈把这条路线标为已核验。",
        ]
        diag = _base_diagnostics(
            source="template",
            template_derived=True,
            template_match={
                **match,
                "template": located["template"],
                "template_length_m": located["length_m"],
                "broad_overview": broad,
            },
            input_quality={
                "level": "reject",
                "usable": False,
                "reasons": ["参考级定位：坐标来自历史记录，无当次地理核验"],
                "suggestions": ["放大地图后重新截图，或用人工反馈核验路线"],
            },
            ocr_texts=len(texts),
            track_points_px=len(track.polyline),
            bbox_px=list(track.bbox),
            start_marker=(list(track.start_marker)
                          if track.start_marker else None),
            end_marker=(list(track.end_marker) if track.end_marker else None),
            track_straightness=round(_track_straightness(track.polyline), 3),
            online_geocoding=False,
            lexicon=self.lexicon.stats(),
        )
        return PipelineResult(
            geojson=_template_geojson(lonlat, main_crs, located),
            geojson_gcj02=_template_geojson(coords_gcj, "GCJ-02", located),
            confidence=conf,
            crs=main_crs,
            track_points=len(lonlat),
            distance_m=round(polyline_length_m(coords_gcj), 1),
            diagnostics=diag,
            warnings=ref_warns,
            elapsed_ms=int((time.time() - t0) * 1000),
            debug={"track_mask": track.mask},
            track_polyline_px=[[int(px), int(py)] for px, py in track.polyline],
            image_size=(int(img.shape[1]), int(img.shape[0])),
        )

    def _run_impl(
        self,
        img: np.ndarray,
        city: str = "",
        allow_online: bool = True,
        output_crs: str = "WGS-84",
        refine: bool = True,
        road_names: list[str] | None = None,
        skip_template_recall: bool = False,
        distance_hint_km: float | None = None,
    ) -> PipelineResult:
        t0 = time.time()
        warnings: list[str] = []
        # 把本轮的过滤器实例挂到 vision 配置副本上，build_anchors 从 cfg
        # 读取；浅拷贝生成副本而非直接改共享对象，避免请求间污染。
        # 不用 dataclasses.replace：anchor_filter 是运行时注入属性，非字段。
        cfg = copy.copy(self.cfg.vision)
        # getattr 兜底：测试桩用 __new__ 裸构造时没走 __init__，无此属性。
        cfg.anchor_filter = getattr(self, "anchor_filter", None) or get_filter()

        # 清空「本轮在线源失败」记录：它是**每次识别**的，
        # 不是进程级的（否则上一张图的失败会污染这一张的提示）。
        self.geocoder._online_failures.clear()

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
        straightness = _track_straightness(track.polyline)

        # --- 阶段 2.5：宽泛概览图的轨迹模板兜底（P1）-----------------
        # 判据只用 OCR 文本，必须放在 _resolve_anchors() 的在线地理编码
        # **之前**：这类图上的行政区/地铁线/高架名会被编码成区中心或不
        # 相关 POI，标注位置与真实点位差 0.8-16km，继续配准只会得到
        # 尺度崩坏的假解（HANDOVER §21.4、§22.43）。
        # 只有命中模板才「直跳」；没命中就继续原流程，零回归。
        broad = assess_broad_overview(texts, road_names)
        # 触发模板探测的两种情形：
        #   1) 宽泛概览图（原判据，OCR 主导）；
        #   2) 库里已有 overview 域模板——「上次拼过的同一条路线又来了」。
        #      这类图 OCR 往往有街道名、判不出宽泛概览，但只要形状与已
        #      注册模板近乎复刻（近复刻门槛见 _near_template_hit），就
        #      可以免传细节图直接复用历史真值；不够像则静默回落原流程。
        stats_fn = (getattr(self.templates, "stats", None)
                    if self.templates is not None else None)
        n_overview_tpl = 0
        if callable(stats_fn):
            try:
                n_overview_tpl = int((stats_fn() or {}).get("overview", 0))
            except Exception:  # noqa: BLE001 — 统计失败不影响主流程
                n_overview_tpl = 0
        # skip_template_recall：调用方传了细节图（paired 模式）时必须置位。
        # 2026-11-14 18:18 事故：传了 5 张细节图、拼接成功拿真值在即，
        # 但阶段 2.5 在 pipeline 内部最先执行，画布形状被历史模板
        # 「近复刻」误命中（reversed、residual 0.0266）后直接 return——
        # 真值路径被短路，payload 里同时出现「参考级定位」与
        # 「本次结果为地理真值」两句自相矛盾的话。原则：用户愿意传
        # 细节图，就一定走当次地理配准；模板召回只服务「免传细节图」
        # 的复用场景，绝不与真值路径竞争。
        if (
            self.templates is not None
            and not skip_template_recall
            and (broad["broad_overview"] or n_overview_tpl > 0)
        ):
            located: dict[str, Any] | None = None
            try:
                # 这里进来的必然是概览图，只比对「概览图形状」注册的模板。
                # 旧的 detail 域模板（细节图/拼接画布）形状域不同，放进来
                # 会把局部小路线误判成整条概览（§22.45 实测假阳性）。
                # 画像隔离：只召回本次请求来源（入参/环境变量）的模板，
                # 防止美团像素形状命中其它 App 的截图。
                located = self.templates.locate(
                    track.polyline, city_hint=city.strip(),
                    shape_domain=SHAPE_DOMAIN_OVERVIEW,
                    profile=self.cfg.source_profile_id)
            except Exception as exc:  # noqa: BLE001 — 兜底失败必须回落原流程
                warnings.append(
                    f"历史记录匹配异常，已回落原流程：{type(exc).__name__}")
            if located is not None and (
                    broad["broad_overview"]
                    or _near_template_hit(located)):
                return self._template_result(
                    located, img=img, track=track, texts=texts, broad=broad,
                    output_crs=output_crs, t0=t0, warnings=warnings)
            if broad["broad_overview"]:
                warnings.append(
                    "这张图是城市级概览（地名多为行政区/地铁线/高架，公里级精度），"
                    "但历史记录里没有形状足够接近的同类路线；仍按原流程尝试，"
                    "结果可能偏差较大。"
                )
            # 非宽泛图探测未命中/不够像：静默回落原流程，不打扰用户。

        # --- 阶段 3：地理编码 ---
        pairs = self._resolve_anchors(anchors, city, allow_online, warnings)

        # --- 阶段 3.5：缩放级别评估 ---
        # 城市级截图里底图不显示街道名，能认出的只有行政区/地铁线/大 POI，
        # 它们的文字标注位置与真实点位差公里级。此时必须明确告诉用户
        # 「这张图的信息量不够」，而不是给一个看起来精确却错的结果。
        zoom = self._assess_zoom_level(pairs, road_names)
        if zoom["zoom_too_low"]:
            ex = "、".join(zoom["broad_examples"][:4])
            if zoom["position_ok"]:
                # 位置已靠路名修正，但形状修不了——必须说清楚。
                warnings.append(
                    f"地图缩放级别过低：底图未显示街道名，"
                    f"仅认出 {zoom['n_broad_anchors']} 个大范围地标"
                    f"（{ex} 等），它们的位置误差可达数公里。"
                    f"你填的路名已把**位置**修正到街区级"
                    f"（起点终点误差约 300-900 米），"
                    f"但**街道级走向无法从这张图恢复**——"
                    f"城市级缩放下截图里的轨迹线本身就是直线，"
                    f"锯齿在截图阶段就丢了。"
                    f"如需真实路线，请把地图放大到能看到街道名后重新截图。"
                )
            else:
                warnings.append(
                    f"地图缩放级别过低：底图未显示街道名，"
                    f"仅认出 {zoom['n_broad_anchors']} 个大范围地标"
                    f"（{ex} 等），"
                    f"它们的位置误差可达数公里，无法定位到街区。"
                    f"请填写你记得走过的路名，或截取放大到街道级的地图。"
                )

        # --- 阶段 2.6：截图里程 OCR（尺度先验 + scale_check 共用）-------
        #
        # 先试带单位的正则（公里/km）；行者等 App 的顶栏数字**无单位**
        # 且跨列粘连（实测「34.67新=江18.29」），正则漏掉时走按列切分
        # 的二次 OCR 兜底。概要页里程（服务端解析）比截图 OCR 更权威，
        # 两者都有时以 ``distance_hint_km`` 为准。
        ocr_distance_km = extract_distance_km(texts)
        if ocr_distance_km is None:
            try:
                ocr_distance_km = extract_distance_column(img, texts)
            except Exception:  # noqa: BLE001 —— 裁切/OCR 失败不影响主流程
                ocr_distance_km = None

        # --- 阶段 3.6：轨迹形状的信息量 ---
        # 缩放级别只看「米/像素」，抓不到「轨迹本身被渲染成直线」。
        # 这类输入下配准再准也得不到沿路的形状，必须明确说明，
        # 否则一个置信度不高却看起来精确的结果会误导用户。
        if _track_lacks_shape_info(
            track.polyline, track.bbox, bool(zoom.get("has_road_names"))
        ):
            warnings.append(
                "截图里的轨迹整体呈一条扁长的直线，且没有可辨认的街道名锚点，"
                "很可能是起终点连线而非沿道路的实际轨迹：本结果只能定位到"
                "街区级，无法还原沿路走向，画在底图上会横穿街区。"
                "请截取包含完整轨迹曲线的页面，或把地图放大到能看到道路名后"
                "重新截图；也可以在「你记得走过的路」填入路名来约束位置。"
            )

        # --- 阶段 4：配准 ---
        #
        # 外部尺度先验：概要里程（用户传的概要页解析出的真实里程）
        # ÷ 轨迹像素长度 = 米/像素。它完全独立于锚点质量——
        # 锚点尺度投票在「候选全是垃圾/同名异地」时会失效或投错
        # （2026-11-17 街区级特写实测：6 个自洽锚点在候选里，
        #  2 点退化模型仍被「支队亲」这类垃圾锚点拉爆到 31 m/px，
        #  真值约 0.5）。物理里程不会撒谎，投不过它时以它为准。
        #
        # ⚠️ 先验有一个隐含假设：**截图包含轨迹全程**。用户把详情页
        # 地图放大（局部视图）或轨迹被底部卡片截断时，可见弧长只是
        # 全程的一小段，先验会虚胖数倍——实测余姚 13.09km 跑步样本：
        # 可见弧长仅 ~4km，先验 9.9 m/px（真值 ~3.1），把正确的
        # 余姚锚点簇挤出尺度容差窗，同名异地簇（宁波江北）乘虚而入。
        # 检测：轨迹下缘贴近裁剪区底部（≥90% 高度）即视为局部视图，
        # 先验不参与定尺度（交由 register_with_hint 的回退逻辑）。
        track_clipped = False
        if len(track.polyline) >= 2:
            _bx, _by, _bw, _bh = track.bbox
            if _by + _bh >= 0.90 * float(img.shape[0]):
                track_clipped = True
        external_prior = None
        hint_km = distance_hint_km or ocr_distance_km
        if (
            hint_km and hint_km > 0 and len(track.polyline) >= 2
            and not track_clipped
        ):
            pl = np.asarray(track.polyline, dtype=float)
            track_px = float(np.linalg.norm(
                np.diff(pl, axis=0), axis=1).sum())
            if track_px >= 20.0:
                external_prior = hint_km * 1000.0 / track_px
        if track_clipped and hint_km:
            warnings.append(
                "轨迹延伸出截图可视范围（被屏幕边缘/信息卡片截断），"
                "可见部分只是全程的一段：概要里程不再用于推算比例尺，"
                "几何里程小于概要里程属预期现象。"
            )
        # 城市提示 bbox：城市内候选簇优先于同名异地簇（见
        # register_with_hint）。bbox 来自冷启动的行政区划数据。
        city_bbox = None
        if city and city.strip():
            try:
                from .coldstart import get_cold_start
                _cold = get_cold_start(
                    str(self.cfg.road_cache_dir / "coldstart.json")
                )
                city_bbox = _cold.city_bbox(city.strip())
            except Exception:  # noqa: BLE001 —— 拿不到 bbox 就退回纯几何
                city_bbox = None
        reg = register_with_hint(
            pairs,
            inlier_m=cfg.ransac_inlier_m,
            min_inliers=cfg.ransac_min_inliers,
            external_scale_prior=external_prior,
            city_bbox=city_bbox,
            city_name=city.strip(),
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
                    anchor_geocode=list(getattr(
                        self, "_last_geocode_report", []) or []),
                    online_geocoding=bool(
                        self.cfg.has_online_key and allow_online
                        and not getattr(self.geocoder, "_online_failures", [])),
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

        # 2 点退化模型告警：与 _score_confidence 的封顶配套，让用户知道
        # 为什么置信度被压低——相似变换被 2 点完全确定、无冗余校验。
        if len({p.name for p in reg.inliers}) <= 2:
            names = "、".join(dict.fromkeys(p.name for p in reg.inliers))
            warnings.append(
                f"地理配准仅由 2 个锚点（{names}）支撑：相似变换无冗余校验，"
                "尺度可能整体错误。建议补充放大的细节图重试，"
                "或以上传的运动概要里程为准")

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
        # 无路名时这两个变量也必须存在——阶段 5.5 会用到
        _road_cons: list[Any] = []
        _notes_collected = 0
        if road_names:
            road_name_constraints, road_name_fit, _road_cons = (
                self._apply_road_names(reg, track, road_names, city, allow_online)
            )
            for n in road_name_fit.get("notes", []):
                warnings.append(n)
            _notes_collected = len(road_name_fit.get("notes", []))
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

        # --- 阶段 5.5：按路名串出路线 ---
        #
        # 城市级截图里 11 公里的骑行只占 ~680 像素，美团画的那条线
        # **本身就是直线**——街道锯齿在截图阶段就丢了。
        # 路名能修正**位置**，但修不了**形状**；这里用用户的路名列表
        # （本身就是路线描述）**串出**完整路线（见 `app/routebuild.py`）。
        if _road_cons and road_name_fit.get("ok"):
            rebuilt = self._build_route(
                lonlat_gcj, _road_cons, road_name_fit
            )
            if rebuilt is not None:
                lonlat_gcj = rebuilt
                # ⚠️ **重建后必须重跑跨江检查**。
                # 串路会让轨迹变长、变弯（实测 13102 -> 14656 米），
                # 可能触到江边；而跨江护栏是**安全底线**，
                # 必须反映**最终**输出，而不是重建前的状态。
                try:
                    from .waterguard import check_crossing
                    crossed2, which2, ratio2 = check_crossing(
                        [(float(lo), float(la)) for lo, la in lonlat_gcj],
                        city=city or "上海",
                        cache_dir=str(self.cfg.road_cache_dir),
                        allow_network=allow_online,
                    )
                    road_name_fit["water_check"] = {
                        "crossed": bool(crossed2),
                        "waters": which2,
                        "ratio": round(float(ratio2), 4),
                        "after_rebuild": True,
                    }
                    if crossed2:
                        road_name_fit.setdefault("notes", []).append(
                            f"⚠️ 形状重建后的轨迹疑似横跨{which2}"
                            f"（{ratio2:.0%} 的点在江面上），请核对起终点")
                except Exception as exc:  # noqa: BLE001
                    road_name_fit.setdefault("notes", []).append(
                        f"形状重建后跨江检查未能完成（{type(exc).__name__}）")
        # ⚠️ **补收 notes**：形状重建在阶段 5.5 往 road_name_fit 里追加了
        # 说明，而上面的收集发生在阶段 4.5 结束时——不补收就静默丢失
        # （实测重建成功了但 warnings 里一个字都没有）。
        if road_name_fit.get("notes"):
            for n in road_name_fit["notes"][_notes_collected:]:
                warnings.append(n)

        # --- 阶段 5.8a：路网吸附准备（可选） ---
        #
        # **同时修里程**：交付链原用 RDP 简化（eps 0.075px），弧长判据
        # 把近直线路段连锁坍缩，实测把 34.9km 削到 30.4km（-12%）——
        # 这才是「识别里程偏低」的主因之一。吸附交付时跳过 RDP，改用
        # 2px 等距重采样（弧长保真，点数 ~1300 在 GeoJSON 无成本压力）。
        road_snap: dict[str, Any] | None = None
        if (getattr(cfg, "road_snap_deliver", False)
                and road_net is not None
                and getattr(road_net, "samples", None)):
            low_shape = _track_lacks_shape_info(
                track.polyline, track.bbox, bool(zoom.get("has_road_names")))
            if low_shape:
                road_snap = {"ok": False, "method": "gentle",
                             "reason": "直线输入不做路网吸附（会把直线强行贴成路）"}
            else:
                road_snap = {}  # 占位：5.8b 执行温和吸附
                simplified = resample_polyline(track.polyline, step_px=2.0)
                lonlat_gcj = apply_to_polyline(reg, simplified)

        # --- 阶段 5.7：里程尺度强制校准 ---
        #
        # OCR/概要页里程是物理真值（App 记录），比 7-12 个混源内点拟合
        # 出的尺度更可信。实测宁波闭环样本：内点拟合地面尺度 12.24 m/px，
        # 交付 30.37km，比 App 记录 34.67km 少 12.4%——一致性阈值 ±15%
        # 拦不住。偏差 >8% 时直接绕轨迹质心缩放模型，把交付里程拉回
        # 先验值（±3% 内）。绕质心缩放保证轨迹位置不动、只改尺度；
        # 反投影 overlay 用交付模型，天然一致。
        # 注：校准必须在吸附**之前**——吸附点 lying on roads 是相对
        # 固定路网的，先校准再吸附才能两全。
        scale_calibration: dict[str, Any] | None = None
        _prior_km = distance_hint_km or ocr_distance_km
        # ⚠️ 局部视图（track_clipped）下禁止校准：几何里程 < 概要里程
        # 是「可见弧长只是全程一段」的必然结果，不是提取丢段。绕质心
        # 放大只会把轨迹拉出锚点钉住的真实范围（实测余姚样本被 ×1.6
        # 拉到 8.8km，超出截图可视区域覆盖的实际走廊）。
        if (_prior_km and _prior_km > 0 and len(simplified) >= 50
                and lonlat_gcj and len(lonlat_gcj) >= 2
                and not track_clipped):
            from .geo.coords import haversine as _hav

            _geom_m = sum(
                _hav(lonlat_gcj[i - 1][0], lonlat_gcj[i - 1][1],
                     lonlat_gcj[i][0], lonlat_gcj[i][1])
                for i in range(1, len(lonlat_gcj))
            )
            _target_m = float(_prior_km) * 1000.0
            if _geom_m > 100.0 and abs(_geom_m - _target_m) / _target_m > 0.08:
                _f = max(0.6, min(1.6, _target_m / _geom_m))
                _s = (np.asarray(simplified, dtype=float)[:200:10]
                      if len(simplified) > 200
                      else np.asarray(simplified, dtype=float))
                _c = np.mean(
                    [reg.pixel_to_mercator(float(px), float(py))
                     for px, py in _s],
                    axis=0,
                )
                reg.matrix = reg.matrix * _f
                reg.translation = reg.translation * _f + (1.0 - _f) * _c
                # ⚠️ 不要给 reg.scale 赋值：scale 是从 matrix 实时计算的
                # 只读属性（register.py Registration.scale），上一行缩放
                # matrix 后 scale 已自动跟着变，赋值会 AttributeError
                # （实测切高德底图触发校准分支才暴露）。
                lonlat_gcj = apply_to_polyline(reg, simplified)
                scale_calibration = {
                    "factor": round(_f, 4),
                    "before_m": round(_geom_m, 1),
                    "after_m": round(_geom_m * _f, 1),
                    "target_m": round(_target_m, 1),
                    "source": "summary" if distance_hint_km else "ocr",
                }
                warnings.append(
                    f"里程尺度校准：几何里程 {_geom_m / 1000:.1f}km 与记录 "
                    f"{_prior_km:.2f}km 偏差超 8%，已按先验缩放模型 "
                    f"（×{_f:.3f}），交付里程以校准后为准"
                )

        # --- 阶段 5.8b：路网温和吸附执行 ---
        #
        # 像素描迹离 OSM 路网中位 ~52m（p90 ~190m），画在地图上会有
        # 明显的「不在马路上」观感。gentle_snap 把离路 ≤60m 的点投影到
        # 路上（窗口多数表决压抖动、跳变回退防跨街直线），其余保持
        # 原样——绝对位置偏差（~300-500m 结构性）不是吸附能修的，
        # 强行全局重匹配实测虚增 29-42%，已废弃。
        if road_snap is None or road_snap.get("ok") is False:
            pass  # 未启用或准备阶段已否决（直线输入等），保持原样
        else:
            try:
                from .snaproute import gentle_snap
                _snap = gentle_snap(reg, simplified, road_net)
                road_snap.update(_snap)
            except Exception as exc:  # noqa: BLE001 — 吸附失败不影响主结果
                road_snap = {"ok": False, "method": "gentle",
                             "reason": f"路网吸附失败：{type(exc).__name__}"}
            if road_snap.get("ok"):
                road_snap["coordinates_raw_gcj02"] = [
                    [float(lo), float(la)] for lo, la in lonlat_gcj]
                lonlat_gcj = [
                    (float(lo), float(la))
                    for lo, la in road_snap["coordinates_gcj02"]]
                warnings.append(
                    f"路网温和吸附：{road_snap['n_snapped']}/"
                    f"{road_snap['n_points']} 点已投影到道路上"
                    f"（吸附距离中位 {road_snap['median_snap_m']:.0f}m，"
                    f"长度 {road_snap['length_before_m']:.0f}→"
                    f"{road_snap['length_after_m']:.0f}m，回退跳变 "
                    f"{road_snap.get('n_reverted', 0)} 处）；"
                    "离路超容差的点保持原样——绝对位置偏差需统一锚点"
                    "源才能根治，吸附不伪造")
            else:
                warnings.append(
                    f"路网吸附未采纳：{road_snap.get('reason') or '未知原因'}"
                    "——交付保留原始像素描迹形状")

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

        geojson = _build_geojson(lonlat, reg, track, source_id=self.cfg.source_profile_id)

        # --- 可选：HMM 地图匹配（备选显示层，不改主结果） ---
        # 已走路网吸附交付（road_snap）时跳过重复的 HMM 诊断——同一件事
        # 做两遍浪费且口径混乱；吸附诊断在 road_snap 里已含。
        if road_snap is not None and road_snap.get("ok"):
            map_matched = {
                "ok": True,
                "mean_snap_m": road_snap.get("mean_snap_m"),
                "median_snap_m": road_snap.get("median_snap_m"),
                "n_points": road_snap.get("n_points"),
                "delivered": True,
            }
        else:
            map_matched = self._map_match_optional(reg, track, road_net, zoom)

        # --- 独立校验：与配准自洽指标无关的两条外部证据 ---
        # 截图里程来自 OCR（地图区内通常有里程牌/大卡）、外部路由来自
        # Valhalla/OSRM，二者都不吃配准残差。地图区内没有里程文字时
        # （amap 单图被裁掉下半屏概要卡），退回概要页解析出的里程。
        scale_check = scale_consistency(
            distance, ocr_distance_km or distance_hint_km)
        route_check = route_validation_factor(road_name_fit)
        input_quality = assess_input_quality(
            image_size=(int(img.shape[1]), int(img.shape[0])),
            straightness=straightness,
            bbox_px=tuple(track.bbox),
            has_road_names=bool(zoom.get("has_road_names")),
            n_ocr_texts=len(texts),
        )
        if not scale_check["consistent"]:
            warnings.append(
                f"截图里程约 {scale_check['ocr_km']} km，还原轨迹几何里程 "
                f"{scale_check['geom_distance_m'] / 1000:.1f} km"
                f"（比值 {scale_check['ratio']}）：两者量级不一致，"
                "配准尺度或轨迹形状可能有误，建议核对后重新截图"
            )
        confidence = self._score_confidence(
            reg, track, anchors, pairs, road_fit, road_name_fit, zoom,
            scale_check=scale_check, route_check=route_check,
        )

        # 记录本轮真正支撑配准的地名：整图置信度不足时，词典写入改用
        # 「锚点级证据」裁决（见 lexicon_guard.evaluate_write），避免
        # position_ok=False 的 0.2 封顶与 0.35 写入门槛形成冷启动死锁。
        self._last_anchor_keys = {
            self.lexicon.normalize(a.name) for a in reg.inliers
            if (a.name or "").strip()
        }

        # --- 退化配准的置信度封顶（诚实化）---
        #
        # 实测失败样本：只有 3 个可靠内点、RMSE 1800m、尺度被撑大到
        # 22 m/px，却因「比率/数量/类型」分项叠加给出了 0.46 的
        # 中等置信度，误导用户以为结果可信。可靠锚点 < 4 个时配准本质
        # 是退化的（城市级截图里很容易因个别错候选而只剩 3 点自洽），
        # 此时无论其它分项多高，置信度都封顶到 0.35，明确告知「仅供参考」。
        # 校正几何（Fix：跨源冲突剔除）后可靠内点通常会回到 4+，
        # 这条封顶就不触发，正确的结果不会被误伤。
        if getattr(reg, "n_reliable", 0) < 4:
            confidence = min(confidence, 0.35)

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
        geojson_gcj = _build_geojson(lonlat_gcj, reg, track, source_id=self.cfg.source_profile_id)
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
                # 真实配准模型（像素→墨卡托米）：overlay 反投影必须用它，
                # 不能重新拟合——重拟合的锚点解析条件（城市/在线开关）稍有
                # 不同就会拟出另一个模型，绿线整体偏离原图（2026-11-18）。
                model_matrix=[[float(v) for v in row] for row in reg.matrix],
                model_translation=[float(reg.translation[0]),
                                   float(reg.translation[1])],
                rmse_m=round(reg.rmse_m, 2),
                anchor_backproj_px=round(med_px, 1),
                inliers=len(reg.inliers),
                outliers=len(reg.outliers),
                ocr_texts=len(texts),
                anchor_candidates=len(anchors),
                anchor_geocode=list(getattr(
                    self, "_last_geocode_report", []) or []),
                track_points_px=len(track.polyline),
                bbox_px=list(track.bbox),
                track_straightness=round(straightness, 3),
                anchor_fragments=len(reg.fragment_rejected),
                anchor_bias=reg.anchor_bias,
                anchor_sources=reg.source_bias,
                scale_check=scale_check,
                scale_calibration=scale_calibration,
                road_snap=road_snap,
                route_validation=route_check,
                input_quality=input_quality,
                map_match=(
                    {k: map_matched.get(k) for k in
                     ("ok", "reason", "mean_snap_m", "median_snap_m", "n_points")}
                    if map_matched else None),
                start_marker=list(track.start_marker) if track.start_marker else None,
                end_marker=list(track.end_marker) if track.end_marker else None,
                # ⚠️ 这个字段的语义是「本次识别**真的用到**了在线源」，
                # 不是「配了 key」。单源严格模式下，配的那家可能配额用尽、
                # 每次都失败，结果其实是词典 + 冷启动兜出来的——
                # 报 True 会让人以为在线源工作正常。
                online_geocoding=bool(
                    self.cfg.has_online_key and allow_online
                    and not getattr(self.geocoder, "_online_failures", [])),
                lexicon=self.lexicon.stats(),
                alternatives=len(alternatives),
                zoom=zoom,
            ),
            warnings=warnings,
            map_matched=map_matched,
            elapsed_ms=int((time.time() - t0) * 1000),
            debug={"track_mask": track.mask},
            track_polyline_px=[[int(px), int(py)] for px, py in simplified],
            image_size=(int(img.shape[1]), int(img.shape[0])),
            # 只有**成功配准**的结果才留下上下文；早退分支没有 reg。
            # 后台精修据此复用，不必重新读图、OCR 和查在线锚点。
            refine_context=RefineContext(
                reg=reg,
                track=track,
                simplified=simplified,
                anchors=anchors,
                pairs=pairs,
                texts=texts,
                zoom=zoom,
                alternatives=alternatives,
                road_name_fit=road_name_fit,
                med_px=med_px,
                city=city,
                output_crs=output_crs,
                allow_online=allow_online,
                road_names=road_names,
                image_size=(int(img.shape[1]), int(img.shape[0])),
                online_geocoding=bool(
                    self.cfg.has_online_key and allow_online
                    and not getattr(self.geocoder, "_online_failures", [])),
            ),
        )

    def _assess_zoom_level(
        self,
        pairs: list[AnchorPair],
        road_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """判断截图的地图缩放级别是否过低（城市级视野）。

        为什么必须有这个判断：城市级缩放（一张图覆盖 20 公里）时，
        底图**根本不显示街道名**，OCR 只能认出行政区、地铁线、
        大型 POI 这类「大范围对象」。而它们的**文字标注位置**被
        地图引擎挪到视野空白处，与真实点位可差很远——

        实测上海城市级截图（1260x2720，覆盖整个上海）：

        | 锚点 | 标注位置→真实位置 偏差 |
        |---|---|
        | 14号线 / 3号线 / 上海西站 | 800–917 米 |
        | 上海站 / 12号线 / 上海南站 | 1638–2301 米 |
        | 静安区 / 徐汇区 / 4号线内圈 | 2681–3225 米 |
        | 1号线 / 西岸自然 | 6143–16696 米 |

        **每个锚点的偏移都是公里级，而整条轨迹只有 11 公里长**——
        锚点偏移比轨迹本身还大，原理上就无法定位到街区级别。
        这时无论怎么调配准算法都救不回来，必须靠用户补信息。

        判据：地理编码后是否存在**街道级锚点**（普通道路）。
        高速公路（外环高速）不算——它同样是几十公里级的大对象。
        """
        # 按名字去重：同名多候选是常态（「虹桥路」149 段），
        # 一个名字只算一次，且优先取「非大范围」的候选类型。
        kinds: dict[str, str] = {}
        for p in pairs:
            nm = (p.name or "").strip()
            if not nm:
                continue
            cur = kinds.get(nm)
            if cur is None or (cur in _BROAD_KINDS and p.kind not in _BROAD_KINDS):
                kinds[nm] = p.kind

        street = [
            nm for nm, k in kinds.items()
            if k == "road" and not any(x in nm for x in _EXPRESSWAY_MARKERS)
        ]
        broad = [
            nm for nm, k in kinds.items()
            if k in ("district", "transit") or nm.endswith(("区", "县", "市"))
        ]
        n = len(kinds)
        has_road_names = bool([x for x in (road_names or []) if x.strip()])
        broad_ratio = len(broad) / n if n else 0.0
        # ⚠️ **两个条件都要满足**，避免误判：
        #   1) 一个街道级锚点都没有；
        #   2) 大范围地标确实占多数。
        # 只看条件 1 的话，街道级截图上「路名没查到地理坐标」的情况
        # 会被误判成缩放过低（实际缺的是地名数据，不是分辨率）。
        no_street = bool(n >= 5 and not street and broad_ratio >= 0.45)

        # ⚠️ **「位置」和「形状」是两个独立维度，不能混为一谈**。
        #
        # 城市级截图下：
        #   * 位置 -> 可以靠用户提供的路名修正（起终点锚定实测准到 300-900 米）
        #   * 形状 -> **修不了**。截图里那条像素轨迹本身就是一条直线，
        #     街道级锯齿在截图阶段就丢了。
        #
        # 实测（同一张城市级截图）：
        #   真实路线是 Z 字形（小木桥路往南 -> 中山南二路往西 -> 肇嘉浜路往北
        #   -> 虹桥路往西 -> 虹古路往北），而还原出来是一条斜线，
        #   中段偏差 1-1.5 公里。
        #
        # 所以「填了路名就不提示」是错的——那会让用户以为拿到了完整结果。
        # 正确的说法是：**位置已修正到街区级，街道级走向无法从这张图恢复**。
        position_ok = bool(has_road_names or street)
        return {
            # 无街道级锚点 -> 底图没显示路名 -> 形状不可恢复（提示用）
            "zoom_too_low": no_street,
            # 位置是否可信（有路名约束或有街道级锚点）-> 影响置信度封顶
            "position_ok": position_ok,
            "has_road_names": has_road_names,
            "n_anchor_names": n,
            "n_street_anchors": len(street),
            "n_broad_anchors": len(broad),
            "broad_ratio": round(broad_ratio, 3),
            "broad_examples": broad[:6],
        }

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
        # 裸构造（__new__）的测试桩没有 anchor_filter，回退全局默认。
        _filter = getattr(self, "anchor_filter", None) or get_filter()
        # 只对权重较高、名字较完整的锚点做地理编码，减少请求量。
        # 排序策略：先按「本地词典能否命中」排序（命中的优先），
        # 再按锚点权重、名称长度排序。
        # 早期实现按权重排序后截断，因 OCR 垃圾文本（如「支工」「丰河」）
        # 与正常 POI 权重相同、且排序稳定，导致真正的 POI 锚点被截掉。
        eligible = [
            a for a in anchors
            if a.kind in ("icon", "poi") and len(a.name) >= 2
            # 双保险：即使上游（测试桩/vendor/另类 OCR 路径）漏过滤，
            # UI 文案也不能占用地理编码名额。
            and not _filter.reject(a.name)
        ]
        eligible.sort(
            key=lambda a: (
                0 if self.lexicon.lookup(a.name, city) else 1,
                -a.weight,
                -len(a.name),
            )
        )
        # 超出名额的候选也记录进诊断明细（status=skipped）——用户排查
        # 「为什么我的 POI 没参与配准」时需要看到它是被截掉还是没查到。
        # 名额默认 20（历史：按 Nominatim 1次/秒限速权衡），接高额度
        # 高德中转后可在 .env 用 GEOCODE_TARGET_LIMIT 调大（热读）。
        # getattr 兜底：测试桩/vendor 场景可能传入无此属性的轻量 cfg。
        _limit = getattr(self.cfg, "geocode_target_limit", None) or 20
        skipped_targets = eligible[_limit:]
        targets = eligible[:_limit]

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
            # 词典里权重最高的候选优先，其余作为备选（最多 5 个）。
            # 上限从 3 提到 5：商业源（百度）的相关门店按全市相关性排序，
            # 正确的「截图那家分店」常排在第 4 位（如巴奴：陆家嘴店在前、
            # 荟聚店在后），截到 3 会把它截掉，RANSAC 拼不出自洽簇。
            for r in results[:5]:
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
                        source=r.source,
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

        # ---- 在线地理编码源的失败原因必须让用户看到 ----
        #
        # 两种降级形态都要提示（2026-11-25 用户实测「配了高德却在用
        # 百度」毫无提示的教训）：
        # - 部分降级：高德配额超限 → 百度接单。结果仍是在线的，但
        #   已不是配置的首选源，用户必须知道；
        # - 完全降级：所有在线源都失败 → 本地词典 + OSM 冷启动兜底。
        online_err = getattr(self.geocoder, "_last_error", "")
        online_failed = list(getattr(self.geocoder, "_online_failures", []))
        if online_failed:
            # 去重后只留前 3 条，避免刷屏
            uniq = list(dict.fromkeys(online_failed))[:3]
            served_online = {
                p.source for p in pairs
                if p.source in ("amap", "tmap", "baidu", "tianditu")
            }
            if served_online:
                # 部分降级：仍有在线源在服务，但不是首选的那家
                fallback = "、".join(sorted(served_online))
                warnings.append(
                    f"首选地理编码源不可用（{'；'.join(uniq)}），"
                    f"本次编码由 {fallback} 兜底：同名 POI 的坐标口径"
                    "可能与首选源有 30-70m 系统偏差。"
                    "如需调整顺序，在 .env 设置 MAP_GEOCODING_ORDER。"
                )
            else:
                warnings.append(
                    "在线地理编码不可用，已降级到本地词典与 OSM 冷启动："
                    + "；".join(uniq)
                    + "。如需更换服务商，在 .env 设置 MAP_GEOCODING_ORDER"
                    "（如 tianditu 或 amap,tianditu）后无需重启即可生效。"
                )

        # --- 诊断明细：每个候选名的地理编码结局 ------------------------
        #
        # 「20 个锚点只有 1 个命中」这类问题光看 warnings 的汇总数字
        # 无从下手：是哪些名字没命中？是配额耗尽还是 OSM 收录缺失？
        # 明细进 diagnostics.anchor_geocode，前端 console.table 展示，
        # 用户自己就能排查（2026-11-17 用户需求）。
        #
        # 统计必须在冷启动之后：pending 的名字可能在冷启动里拿到坐标，
        # 只有最终 pairs 里也没有的才算真 MISS。
        final_names = {p.name for p in pairs}
        report: list[dict[str, Any]] = []
        fail_reason = ""
        if online_failed:
            fail_reason = "；".join(list(dict.fromkeys(online_failed)))[:120]
        for a in targets:
            if a.name in final_names:
                cand = [p for p in pairs if p.name == a.name]
                report.append({
                    "name": a.name,
                    "status": "matched",
                    "source": cand[0].source if cand else "",
                    "quality": round(max(p.match_quality for p in cand), 2),
                    "n_candidates": len(cand),
                })
            else:
                report.append({
                    "name": a.name,
                    "status": "missed",
                    "source": "",
                    "quality": None,
                    "note": (
                        f"词典/在线/冷启动均未命中（{fail_reason}）"
                        if fail_reason else
                        "词典/在线/冷启动均未命中（OSM 无同名或近名要素）"
                    ),
                })
        for a in skipped_targets:
            report.append({
                "name": a.name, "status": "skipped",
                "source": "", "quality": None,
                "note": "超出单次地理编码名额（按词典命中/权重/名称长度排序）",
            })
        # 允许早退分支安全读取（template 召回路径不会执行到这里）
        self._last_geocode_report = report
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
                        source="osm",
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
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[Any]]:
        """应用用户提供的路名约束（线约束）。

        第三个返回值是**可用的路名约束列表**（含 OSM 几何），
        供后续的形状重建复用——避免为同一批路名再查一次 OSM。

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
                return [], {"ok": False, "notes": ["无法确定路名检索范围"]}, []

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
            }, []

        notes: list[str] = []
        anchor_fit: dict[str, Any] = {}
        llm_review: dict[str, Any] = {}

        # --- 可选：大模型语义层校验路名序列 ---
        #
        # 大模型只做**语义层**判断（顺序对不对、哪条路可能记错），
        # 坐标层仍由 OSM 几何与 DP 对齐负责——大模型没有 OSM 级别的几何。
        #
        # ⚠️ **LLM 的建议必须再通过几何验证才被采纳**：
        # 新顺序的「路名到轨迹距离」代价必须更低，否则保留原顺序。
        # 这样即使模型判断错，几何证据会把它挡下来。
        if self.llm is not None and getattr(self.llm, "available", False) \
                and len(usable) >= 3:
            llm_review = self._llm_review_road_names(
                names, usable, reg, track, city)
            if llm_review.get("applied"):
                names = [str(x) for x in llm_review.get("names", names)]
                by_name = {c.name: c for c in usable}
                usable = [by_name[n] for n in names if n in by_name]
                info = [
                    {"name": c.name, "ok": c.ok, "n_segments": c.n_segments,
                     "total_len_km": round(c.total_len_m / 1000.0, 2),
                     "uniqueness": round(c.uniqueness, 3), "notes": c.notes}
                    for c in usable
                ]
                notes.extend(llm_review.get("notes", []))
                for w in llm_review.get("warnings", []):
                    notes.append(f"路名可能有误：{w}")

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
            return info, {"ok": False, "notes": notes,
                          "anchor_fit": anchor_fit}, usable

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
            "llm_review": llm_review,
            "notes": notes,
        }, usable

    def _llm_review_road_names(
        self,
        names: list[str],
        usable: list[Any],
        reg: Registration,
        track: TrackResult,
        city: str = "",
    ) -> dict[str, Any]:
        """让大模型校验路名序列（可选增强，失败不影响主流程）。

        几何验证用的是「各路名到轨迹的最近距离之和」——
        便宜、够用：它能区分「顺序反了」这种**明显**的错误，
        但不参与精细优化（精细优化由 DP 对齐负责）。
        """
        try:
            import numpy as np

            from .llm_geo import check_road_sequence
            from .roadname_constraint import _dist_to_segments

            traj = np.array([
                reg.pixel_to_mercator(float(px), float(py))
                for px, py in track.polyline
            ])
            if len(traj) > 200:                      # 采样，控制计算量
                traj = traj[:: max(1, len(traj) // 200)]
            by_name = {c.name: c for c in usable}

            def cost_fn(order: list[str]) -> float:
                tot = 0.0
                for nm in order:
                    c = by_name.get(nm)
                    if c is None:
                        return float("inf")
                    d = _dist_to_segments(traj, c.segments)
                    tot += float(np.median(d))
                return tot

            return check_road_sequence(self.llm, names, city, cost_fn)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "applied": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _external_verify(
        self,
        fit: Any,
        constraints: list[Any],
        road_name_fit: dict[str, Any],
    ) -> None:
        """用**外部 OSM 路由引擎**做交叉验证：它自己算出的路线是否也经过用户说的路。

        为什么需要外部验证
        ------------------
        本项目的算法（链式 DP + 几何重建）用的是**同一批路名实例**，
        它给出的结果无法自证——「轨迹贴着用户说的路」本来就是设计目标，
        谈不上验证。

        外部引擎只拿到起终点（和途经点），**完全不知道用户填了哪些路名**。
        它自己算出的路线能命中这些路名、而且**顺序也对**，才说明
        用户的记忆与真实路网结构吻合——这是一份**独立于本项目算法**
        的第三方证据。

        引擎选择（实测，同一张城市级截图、同样的起点/终点/途经点）
        ----------------------------------------------------------
        | 引擎 | 里程 | 命中 | 顺序一致 | 说明 |
        |---|---|---|---|---|
        | **Valhalla pedestrian** | 10.5 km | 5/10 | **100%** | 公共实例免费，原生支持 bicycle/pedestrian/auto |
        | Valhalla auto | 15.0 km | 6/10 | 80% | |
        | OSRM driving | 16.7 km | 5/10 | 100% | Demo **只有 driving profile**（实测传 bike/foot 返回完全相同）|
        | BRouter trekking | 10.3 km | - | - | 不返回路名 |
        | 高德 walking | 9.7 km | 1/14 | - | 途经点被**静默丢弃** |

        所以优先用 Valhalla（顺序一致率更高、里程更接近真实），
        OSRM 作为降级。

        ⚠️ **只用它验证，不用它的几何当轨迹**：实测 Valhalla 里程
        10.5 公里，而用户实际骑行约 17.5 公里（10 条路名实例长度之和
        17308 米是独立标尺）——Valhalla **跳过了起点那一片**
        （小木桥路/龙华中路/船厂路/中山南二路共 8.7 公里）。
        形状仍由几何重建提供。
        """
        try:
            import numpy as np

            from .geo.coords import gcj02_to_wgs84, mercator_to_lonlat
            from .valhalla import (
                ValhallaClient, match_ratio, order_consistency,
            )

            if fit.points is None or len(fit.points) < 10:
                return
            user_names = [c.name for c in constraints]
            if not user_names:
                return

            def to_wgs(merc):
                lo, la = mercator_to_lonlat(float(merc[0]), float(merc[1]))
                return gcj02_to_wgs84(lo, la)

            # 途经点 = 各路名与轨迹的接触点（路名主数据源是 OSM）
            #
            # ⚠️ **必须过滤 NaN**：`contact_points` 在实例为空时写入
            # `(nan, nan)`。带着 NaN 去请求，Valhalla 会直接失败
            # （而失败时若无 note 就是静默失效）。
            import math as _math

            raw_wps = list(getattr(self, "_chain_contact_geo", []))
            wps: list[tuple[float, float]] = []
            for cg in raw_wps:
                if cg is None:
                    continue
                try:
                    lo, la = float(cg[0]), float(cg[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not (_math.isfinite(lo) and _math.isfinite(la)):
                    continue
                wps.append(to_wgs((lo, la)))
            if not wps:
                # 兜底：用串路结果里各路名的接触点（routebuild 的 segments）
                for seg in (getattr(fit, "segments", None) or []):
                    if not isinstance(seg, dict):
                        continue
                    frm = seg.get("from_m")
                    if frm is None:
                        continue
                    j = int(np.argmin(np.abs(np.cumsum(
                        np.linalg.norm(np.diff(fit.points, axis=0), axis=1))
                    ) - float(frm)))
                    wps.append(to_wgs(fit.points[j]))
            # 去重相邻重复点（Valhalla 对重复位置敏感）
            ded: list[tuple[float, float]] = []
            for p in wps:
                if not ded or abs(p[0] - ded[-1][0]) > 1e-7 \
                        or abs(p[1] - ded[-1][1]) > 1e-7:
                    ded.append(p)
            s0 = to_wgs(fit.points[0])
            s1 = to_wgs(fit.points[-1])
            pts_req = [s0] + ded + [s1]

            client = getattr(self, "valhalla", None)
            if client is None:
                client = ValhallaClient(cache_dir=str(self.cfg.road_cache_dir))
                self.valhalla = client
            if not client.enabled:
                return

            best: dict[str, Any] | None = None
            for costing in ("pedestrian", "auto"):
                r = client.route(pts_req, costing=costing)
                if not r.ok:
                    continue
                ratio = match_ratio(r.road_names, user_names)
                oscore, seq = order_consistency(r.road_names, user_names)
                hits = [n for n in user_names
                        if any(n in rn or rn in n for rn in r.road_names)]
                cand = {
                    "ok": True,
                    "engine": f"Valhalla {costing}",
                    "match_ratio": round(ratio, 3),
                    "order_consistency": round(oscore, 3),
                    "hits": hits,
                    "n_user_names": len(user_names),
                    "route_distance_m": round(r.distance_m, 1),
                    "hit_sequence": seq,
                    "note": "外部 OSM 路由引擎独立算出的路线命中的用户路名",
                }
                # 排序依据：先看命中数，再看顺序一致率
                key = (len(hits), oscore)
                if best is None or key > (len(best.get("hits") or []),
                                          best.get("order_consistency") or 0):
                    best = cand
                if len(hits) >= max(3, int(0.4 * len(user_names))) and oscore >= 0.8:
                    break

            if best is None:
                road_name_fit.setdefault("notes", []).append(
                    "外部路网校验未取得结果（Valhalla 公共实例不可用）")
                return
            road_name_fit["external_verify"] = best
            # 兼容旧字段名
            road_name_fit["osrm_verify"] = best
            hits = best.get("hits") or []
            if hits:
                oscore = best.get("order_consistency") or 0
                road_name_fit.setdefault("notes", []).append(
                    f"外部路网校验：{best['engine']} 独立规划出的路线也经过"
                    f"{'、'.join(hits[:6])}（{len(hits)}/{best['n_user_names']} "
                    f"条路名，顺序一致 {oscore:.0%}）")
        except Exception as exc:  # noqa: BLE001
            road_name_fit.setdefault("notes", []).append(
                f"外部路网校验未完成（{type(exc).__name__}）")

    def _insert_seam_paths(
        self,
        fit: Any,
        constraints: list[Any],
    ) -> None:
        """用 Valhalla 的逐对连接路径填充接缝。

        串路只解决了「每条路走哪一段」，相邻两条路之间仍是**直线跳跃**。
        实测（同一张城市级截图）：
          船厂路 → 中山南二路  直线 1054 米 | 真实路径 1478 米
                               经由 零陵路、枫林路、斜土路
          中山南二路 → 东安路  直线 1692 米 | 真实路径 2024 米
                               经由 斜土路、枫林路、肇嘉浜路、宛平路

        插入后整条路线才是一条**可骑行**的路线，而不是若干直线的拼接。

        ⚠️ 物理约束：Valhalla 路径长度必然 >= 直线距离，所以插入只会让
        路线**变长**。若反而变短，说明对齐错了——必须放弃。
        """
        try:
            import numpy as np

            from .geo.coords import (
                gcj02_to_wgs84, lonlat_to_mercator, mercator_to_lonlat,
                wgs84_to_gcj02,
            )
            from .seampath import insert_seams
            from .valhalla import ValhallaClient

            route = getattr(fit, "points", None)
            if route is None or len(route) < 10:
                return
            segs = getattr(fit, "segments", None) or []
            if len(segs) < 2:
                return

            client = getattr(self, "valhalla", None)
            if client is None:
                client = ValhallaClient(cache_dir=str(self.cfg.road_cache_dir))
                self.valhalla = client
            if not client.enabled:
                fit.notes.append("接缝插入：Valhalla 未启用，跳过")
                return

            def to_wgs(merc):
                lo, la = mercator_to_lonlat(float(merc[0]), float(merc[1]))
                return gcj02_to_wgs84(lo, la)

            def to_merc(wgs):
                clo, cla = wgs84_to_gcj02(wgs[0], wgs[1])
                return lonlat_to_mercator(clo, cla)

            # 接缝 i = 「路 i 的离开点」-> 「路 i+1 的进入点」。
            # 这两个点就是 routebuild 算出的锚点（各段的起止）。
            bounds = list(getattr(fit, "arc_bounds", None) or [])
            # ⚠️ arc_bounds 是**每条路名**的锚点（长度 = 路名数），
            # 而 segments 只含**命中**的路名——两者长度不同。
            # 所以判定用「路名数」，不是 segments 数。
            n_roads = len(getattr(fit, "segments", None) or []) or len(bounds)
            if len(bounds) < 2:
                fit.notes.append("接缝插入：缺少分段锚点，跳过")
                return

            paths: list[Any] = []
            for i in range(len(bounds) - 1):
                p_a = route[bounds[i][1]]
                p_b = route[bounds[i + 1][0]]
                straight = float(np.hypot(*(p_b - p_a)))
                r = client.route([to_wgs(p_a), to_wgs(p_b)],
                                 costing="pedestrian")
                if not r.ok or len(r.geometry) < 2:
                    paths.append(None)
                    continue
                # 路径长度与直线距离之比有上限（绕得太远说明选错了锚点）
                if straight > 50.0 and r.distance_m > straight * 3.0:
                    paths.append(None)
                    continue
                paths.append(np.array([to_merc(p) for p in r.geometry]))

            seam = insert_seams(route, paths, [b[0] for b in bounds],
                                [b[1] for b in bounds])
            for note in seam.notes:
                fit.notes.append("接缝插入：" + note)
            if seam.ok and seam.points is not None:
                fit.points = seam.points
                fit.length_m = seam.length_after_m
            fit.seam_insert = seam.to_dict()

        except Exception as exc:  # noqa: BLE001
            fit.notes.append(
                f"接缝插入未执行（{type(exc).__name__}: {exc}）")

    def _build_route(
        self,
        lonlat_gcj: list[tuple[float, float]],
        constraints: list[Any],
        road_name_fit: dict[str, Any],
    ) -> list[tuple[float, float]] | None:
        """用用户提供的路名**串出**路线（替换旧的「掰弯直线」）。

        背景
        ----
        城市级缩放（一张图覆盖 20 公里）时，截图里的骑行轨迹线
        **本身就是一条直线**——街道锯齿在截图阶段就丢了。

        旧做法（`app/shapefit.py`）是把这条直线**掰弯**到路名上。
        实测**这条路是错的**（用几何判定，不是肉眼看图）：

        | 路名 | 掰弯后接触点数（60 米内）|
        |---|---|
        | 小木桥路 | **0**（完全没经过）|
        | 肇嘉浜路 | **0**（完全没经过）|
        | 中山南二路 | 131 |

        中间有 5.8 公里完全没贴任何路——那还是原始直线。
        而且旧指标「逐段残差 9333 -> 7021」与「覆盖 99%」都是
        **假的**：DP 强制把弧长铺满给每条路，所以覆盖率必然接近 100%，
        它衡量的是「分配了多少弧长」，不是「真的贴在路上」。

        新做法
        ----
        用户的路名列表本身就是**路线描述**——「从小木桥路出发，经
        龙华中路、���厂路、中山南二路……到安龙路」。所以直接把它
        **串成几何**（`app/routebuild.py`）：

        1. 链式 DP 选空间连贯的实例
        2. 相邻两条路求最近点对 -> 路口
        3. 每条路的「进入点」= 与上一条路的路口，「离开点」= 与下一条
        4. 按主轴方向裁出 entry->exit 的子弧，串成完整路线
        5. 枚举每条路的正/反方向，取「接触顺序与用户顺序最一致」的

        实测（同一张城市级截图，10 条路名）：

        | | 掰弯直线 | 串路 |
        |---|---|---|
        | 路名经过数 | 8/10 | **10/10** |
        | 顺序一致率 | — | **100%** |
        | 里程 | 17553 m | **14656 m** |
        | 端点偏移 | 0 m | 236 m / 188 m |

        独立标尺：10 条路名的 OSM 实例主轴长度之和为 **17308 米**——
        真实骑行必然接近这个量级。14656 米比 17553 米更接近。

        ⚠️ 这是「基于用户路名推断出来的」路线，不是从截图直接读出的。
        原始轨迹保留在 `road_name_fit["route_build"]["original"]`，
        置信度**不上浮**（宁可保守）。
        """
        try:
            import numpy as np

            from .geo.coords import lonlat_to_mercator, mercator_to_lonlat
            from .roadchain import build_candidates, select_chain
            from .routebuild import build_route
            from .shapefit import main_axis

            traj = np.array(
                [lonlat_to_mercator(lo, la) for lo, la in lonlat_gcj])
            if len(traj) < 10:
                return None
            arc = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(traj, axis=0), axis=1))])

            cands = build_candidates(
                [(c.name, c.segments) for c in constraints])
            # ⚠️ **必须带长度惩罚**：同名路的最长实例未必是骑行的那段。
            # 实测「中山南二路」最长实例 4963 米（在西南 2.7 公里外），
            # 惩罚后正确选中 366 米那段（离起点 1302 米）。
            chain = select_chain(cands, traj, arc, length_penalty=0.5)
            if not chain.ok:
                road_name_fit.setdefault("notes", []).append(
                    "路线构建未执行：路名实例无法选出空间连贯的组合")
                return None

            # 每条路的**全部实例**（主轴）。build_route 内部会：
            #   合并首尾相接的片段 -> 链式 DP 选实例 -> 算路口 -> 裁子弧
            per_road = []
            for k in range(len(constraints)):
                per_road.append([main_axis(i.points) for i in cands[k]])
            # 每条路至少要有一个长度 >= 2 的实例
            if any(not any(len(q) >= 2 for q in cands_k)
                   for cands_k in per_road):
                road_name_fit.setdefault("notes", []).append(
                    "路线构建未执行：路名实例几何不足")
                return None

            fit = build_route(
                per_road,
                [c.name for c in constraints],
                [c.segments for c in constraints],
                traj[0], traj[-1],
            )
            # --- 接缝插入：让相邻路名之间走真实路网，而不是直线跳跃 ---
            if fit.ok and fit.points is not None:
                self._insert_seam_paths(fit, constraints)
            info = fit.to_dict()
            info["original"] = [
                [round(lo, 6), round(la, 6)] for lo, la in lonlat_gcj]
            road_name_fit["route_build"] = info
            # 兼容旧字段名（前端两处都读）
            road_name_fit["shape_fit"] = info
            for note in fit.notes:
                road_name_fit.setdefault("notes", []).append(
                    "路线构建：" + note)
            if not fit.ok or fit.points is None:
                return None

            # 外部路网交叉验证（Valhalla）：它只知道起终点，不知道
            # 用户填了哪些路名，能命中说明用户的记忆与真实路网吻合。
            self._external_verify(fit, constraints, road_name_fit)

            out = [tuple(mercator_to_lonlat(float(x), float(y)))
                   for x, y in fit.points]
            road_name_fit.setdefault("notes", []).append(
                f"路线构建：已按路名串出完整路线"
                f"（经过 {fit.n_hit}/{len(constraints)} 条路名，"
                f"顺序一致 {fit.order_consistency:.0%}，"
                f"里程 {fit.length_m:.0f} 米）")
            return out
        except Exception as exc:  # noqa: BLE001
            road_name_fit.setdefault("notes", []).append(
                f"路线构建未执行（{type(exc).__name__}）")
            return None

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

    def _road_refine(self, reg: Registration, track: TrackResult,
                     mode: str = "auto"):
        """路网约束精修：用「轨迹必须走在真实道路上」纠正配准的绝对位置。

        ``mode``：
          * ``auto``   —— 按截图尺度自动选修正 / 校验模式（同步识别默认）。
          * ``verify`` —— 强制「校验 + 命名」，只算贴路距离与路名、不平移。
            多图拼接画布只有一个刚体配准，链式拼接的累积漂移会被当成
            「配准误差」，用一个平移量去纠正容易压到平行主干道上；
            因此多图后台精修默认走这一档。

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

        low_zoom = reg.scale > self.cfg.road_fit_low_zoom_m_per_px
        if mode == "verify":
            low_zoom = True
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

        # 以可靠锚点的「地理真值 - 当前配准预测」作为额外证据。
        # 传入残差向量而不是图片/文件信息，规则因此与样本无关。
        anchor_residuals = []
        for a in reg.inliers:
            if a.kind not in ("poi", "station", "icon", "district"):
                continue
            if a.lon is None or a.lat is None:
                continue
            target = np.array(lonlat_to_mercator(a.lon, a.lat), dtype=float)
            pred = np.array(reg.pixel_to_mercator(a.px, a.py), dtype=float)
            anchor_residuals.append(target - pred)
        fit = refine_with_roads(
            reg, track.polyline, roads,
            max_shift_m=float(self.cfg.road_fit_max_shift_m),
            shift_span_ratio=float(self.cfg.road_fit_shift_span_ratio),
            anchor_points_m=np.asarray(anchor_residuals, dtype=float),
            # 锚点证据可信度降级：内点跨源混用（基准打架）或残差呈
            # 系统性方向（坐标源整体偏移）时，锚点残差主要反映基准
            # 噪声而非配准错误，锚点一致性护栏不再有否决权（2026-11-25
            # 余姚实测：混源解 RMSE 128m，路网想东移 200m 贴路被护栏
            # 误拒，轨迹停在楼块上）。
            anchors_reliable=not (
                (getattr(reg, "source_bias", {}) or {}).get("warning")
                or (getattr(reg, "anchor_bias", {}) or {}).get("systematic")
            ),
            anchor_guard_m=float(self.cfg.road_fit_anchor_guard_m),
            anchor_worsen_m=float(self.cfg.road_fit_anchor_worsen_m),
            anchor_worsen_ratio=float(self.cfg.road_fit_anchor_worsen_ratio),
            eval_budget=int(getattr(self.cfg, "road_fit_eval_budget", 2000)),
        )
        fit_dict = fit.to_dict()
        # 否决时把候选平移换算成截图像素位移，前端据此画第二种颜色。
        # 采纳时 fit_dict 没有 candidate 字段，这里自然返回空。
        fit_dict.update(_candidate_shift_px(reg, fit_dict))
        if fit.ok:
            # 把平移量并入配准矩阵（墨卡托 y 轴向上，北为正）
            reg.translation = reg.translation + np.array(
                [fit.shift_e_m, fit.shift_n_m], dtype=float
            )
        return fit_dict, net

    def refine_existing(
        self,
        base: PipelineResult,
        *,
        mode: str = "auto",
        allow_online_roadname: bool = False,
    ) -> PipelineResult:
        """在主请求结果上补算路网精修，**不重跑整条流水线**。

        只做三件事：补算平移量 → 把简化折线按新的 ``reg`` 重投影 →
        重算置信度、路名归属与 warnings。OCR、轨迹提取、锚点地理编码、
        配准、备选位置这些最贵的阶段直接复用 ``base.refine_context``。

        ``allow_online_roadname`` 默认关闭：后台任务不该再悄悄打一次
        商业在线接口（配额波动会让同一张图两次结果不一致）。精修成功时
        路名本来就直接由 OSM 几何判定，不依赖在线服务。
        """
        t0 = time.time()
        ctx: RefineContext | None = base.refine_context
        if ctx is None:
            raise ValueError("主请求未留下精修上下文，无法复用精修")

        # 主请求那两条说明在精修结果里已经过期：一条说「跳过精修」，
        # 一条是粗结果的在线路名归属，后台会重新归属。
        _kept_warnings = [
            w for w in base.warnings
            if "跳过路网精修" not in str(w)
            and "路名归属（" not in str(w)
        ]

        def _with(base_result: PipelineResult, road_fit: dict[str, Any],
                  notes: list[str]) -> PipelineResult:
            return replace(
                base_result,
                road_fit=road_fit,
                warnings=_kept_warnings + notes,
                elapsed_ms=base_result.elapsed_ms
                + int((time.time() - t0) * 1000),
            )

        # 用户路名约束是**强约束**：它会改尺度并整体平移。若在它之后
        # 再叠一个「贴路最小化」平移，等于把轨迹从用户指定的路上拉走。
        # 这条路保持主请求结果不变，只把原因写清楚。
        if (base.road_name_fit or {}).get("ok"):
            notes = ["已由用户路名约束定位，后台路网平移跳过"
                     "（避免覆盖用户指定的路线）"]
            return _with(base, {"ok": False, "notes": notes}, notes)

        # deepcopy：后台任务若被重试，不能在上一次的平移上再叠一次。
        reg = copy.deepcopy(ctx.reg)
        road_fit, road_net = self._road_refine(reg, ctx.track, mode=mode)
        if not road_fit.get("ok"):
            return _with(base, road_fit, list(road_fit.get("notes", [])))

        # ``_road_refine`` 已把平移并进 ``reg.translation``；形状、尺度、
        # 朝向都没变，因此对同一条简化折线重新投影一次即可。
        lonlat_gcj = apply_to_polyline(reg, ctx.simplified)
        road_attr = self._attribute_roads(
            lonlat_gcj, ctx.simplified, ctx.texts,
            allow_online=allow_online_roadname,
            road_fit=road_fit, road_net=road_net, reg=reg,
        )
        notes: list[str] = []
        if road_attr.segments:
            notes.append(f"路名归属（{road_attr.source}）："
                         + "、".join(road_attr.roads[:8]))

        diag = dict(base.diagnostics or {})
        diag["refine_reused"] = True

        if mode == "verify":
            # 只校验 + 命名：坐标、置信度、里程一律保持主请求的值。
            # 多图拼接画布只有一个刚体配准，链式累积漂移不该被「优化」
            # 成一个平移量；路网在这里只用来验证贴路程度并给出路名。
            road_fit = dict(road_fit)
            road_fit["mode"] = "verify"
            notes.extend(road_fit.get("notes", []))
            return replace(
                base,
                road_fit=road_fit,
                road_attribution=road_attr.to_dict(),
                diagnostics=diag,
                warnings=_kept_warnings + notes,
                elapsed_ms=base.elapsed_ms + int((time.time() - t0) * 1000),
            )

        if ctx.output_crs.upper() in ("WGS-84", "WGS84", "EPSG:4326"):
            lonlat = [gcj02_to_wgs84(lo, la) for lo, la in lonlat_gcj]
        else:
            lonlat = lonlat_gcj

        from .geo.coords import haversine

        distance = 0.0
        for i in range(1, len(lonlat)):
            distance += haversine(
                lonlat[i - 1][0], lonlat[i - 1][1],
                lonlat[i][0], lonlat[i][1],
            )
        # refine_existing 可能被绕过 __init__ 调用（测试/轻量复用），
        # 用 getattr 容错；拿不到画像时 source 退化为通用标签。
        _source_id = getattr(getattr(self, "cfg", None), "source_profile_id", "")
        geojson = _build_geojson(lonlat, reg, ctx.track, source_id=_source_id)
        geojson_gcj = _build_geojson(lonlat_gcj, reg, ctx.track, source_id=_source_id)
        geojson_gcj["features"][0]["properties"]["crs"] = "GCJ-02"

        _base_diag = dict(base.diagnostics or {})
        # 精修改变了轨迹长度，尺度独立校验要用新的几何里程重算，
        # 否则诊断里留的是主请求的旧值。
        _sc = _base_diag.get("scale_check") or {}
        if _sc.get("ocr_km"):
            diag["scale_check"] = scale_consistency(distance, _sc["ocr_km"])
        confidence = self._score_confidence(
            reg, ctx.track, ctx.anchors, ctx.pairs,
            road_fit, ctx.road_name_fit, ctx.zoom,
            scale_check=diag.get("scale_check"),
            route_check=_base_diag.get("route_validation"),
        )
        notes.append(
            f"路网精修：平移 东{road_fit['shift_e_m']:+.0f}m "
            f"北{road_fit['shift_n_m']:+.0f}m，贴路中位 "
            f"{road_fit['median_before_m']:.0f}m → "
            f"{road_fit['median_after_m']:.0f}m"
        )
        return replace(
            base,
            geojson=geojson,
            geojson_gcj02=geojson_gcj,
            confidence=confidence,
            track_points=len(lonlat),
            distance_m=round(distance, 1),
            road_attribution=road_attr.to_dict(),
            road_fit=road_fit,
            diagnostics=diag,
            warnings=_kept_warnings + notes,
            elapsed_ms=base.elapsed_ms + int((time.time() - t0) * 1000),
        )

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
        from .roadname import (RoadAttribution, RoadSegment,
                                attribute_by_online,
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
            attr = attribute_by_online(
                lonlat_gcj, cfg=self.cfg, timeout=self.cfg.http_timeout)
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

    def _map_match_optional(
        self, reg: Registration, track: TrackResult,
        road_net: Any, zoom: dict[str, Any],
    ) -> dict[str, Any]:
        """可选：输出 HMM 地图匹配的沿路折线（默认关闭）。

        三条护栏在 ``mapmatch.hmm_match`` 内部；这里额外把
        「起终点直线」这类低信息量输入显式传进去，避免把概览页的
        直连线强行贴成一条看似合理的路。
        """
        if not self.cfg.map_match_enabled:
            return {}
        if road_net is None or getattr(road_net, "tree", None) is None:
            return {"ok": False, "reason": "无可用路网几何（本次未取路网）"}
        low = _track_lacks_shape_info(
            track.polyline, track.bbox, bool(zoom.get("has_road_names")))
        try:
            from .mapmatch import hmm_match

            return hmm_match(
                reg, track.polyline, road_net, low_shape_info=low
            ).to_dict()
        except Exception as exc:  # noqa: BLE001 — 备选层失败不能影响主结果
            return {"ok": False, "reason": f"地图匹配失败：{type(exc).__name__}"}

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

        # 自校验只是**诊断**，绝不能因为矩阵退化把整次识别打成 500。
        # register 出口已经把退化解标成 ok=False；这里再兜一层，
        # 保证任何来源的 Registration 都不会让求逆抛异常。
        mat = np.asarray(reg.matrix, dtype=float)
        if mat.shape != (2, 2) or not np.all(np.isfinite(mat)):
            return 0.0, ["绝对位置自校验跳过：配准矩阵无效"]
        try:
            inv = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            return 0.0, ["绝对位置自校验跳过：配准矩阵不可逆"]
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
        # 容差必须与 register 的按类型内点容差（_KIND_TOLERANCE）自洽，
        # 不能用一刀切的 400m：项目文档实测 POI 标注偏移 35-150m、
        # 行政区/道路/交通线可到 800m-2km，register 给这些类型的内点
        # 容差下限本就是 200m（POI/station）到 800m（transit/road）。
        # 旧代码对城市级截图（14.9 m/px）用 400m=27px 卡中位偏移，
        # 把「几个 transit 标注偏移大」误判成「整体平移」（2026-11-21
        # 行者宁波：med=597m 触发误报，但 3 个可靠锚点相对方位全吻合）。
        #
        # 新判据：**逐点**与各自类型容差比较，看超容差的内点比例。
        # 整体平移错误 → 多数内点同时超容差；个别标注偏移大 → 仅少数超。
        from .register import _KIND_TOLERANCE, _DEFAULT_TOLERANCE

        n_over = 0
        n_cmp = 0
        limit_m_sum = 0.0
        for a in reg.inliers:
            if a.lon is None or a.lat is None:
                continue
            mx, my = lonlat_to_mercator(a.lon, a.lat)
            v = inv @ (np.array([mx, my]) - reg.translation)
            off_px = float(np.hypot(v[0] - a.px, v[1] - a.py))
            px_cap, m_floor = _KIND_TOLERANCE.get(a.kind, _DEFAULT_TOLERANCE)
            # 与 register._anchor_tolerance 同式：像素上限换算 + 米制下限
            limit_m = max(m_floor, px_cap * max(reg.scale, 1e-6))
            limit_m_sum += limit_m
            n_cmp += 1
            if off_px * reg.scale > limit_m:
                n_over += 1

        over_ratio = (n_over / n_cmp) if n_cmp else 0.0
        mean_limit_m = (limit_m_sum / n_cmp) if n_cmp else 400.0
        # 过半内点超过各自类型容差，才判为整体平移（单/双内点退化时
        # over_ratio 只能是 0 或 1，由 2 点退化告警另行提示，不在此重复）。
        if n_cmp >= 3 and over_ratio > 0.6:
            notes.append(
                f"绝对位置自校验未通过：{n_over}/{n_cmp} 个内点的真值反投影偏移"
                f"超过各自类型容差（偏移中位 {med * reg.scale:.0f} 米，"
                f"平均容差 {mean_limit_m:.0f} 米）。可能是同名地点坐标错误，"
                f"结果或整体平移。"
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
        zoom: dict[str, Any] | None = None,
        scale_check: dict[str, Any] | None = None,
        route_check: dict[str, Any] | None = None,
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

        # 绝对位置自校验：按「超各自类型容差的内点比例」降权，判据与
        # _self_check 的警告一致（见该函数：400m 一刀切会把 POI/transit
        # 标注偏移误判成整体平移，2026-11-21 行者宁波误报）。
        med_px, self_check_notes = self._self_check(reg, pairs)
        if self_check_notes:
            conf *= 0.45
        elif med_px > 0 and med_px * reg.scale > 250.0:
            # 未判定整体平移，但中位偏移仍偏大 → 温和降权
            conf *= 0.9

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

        # --- 缩放级别封顶 ---
        #
        # 城市级截图里底图不显示街道名，能用的锚点只有行政区/地铁线/大 POI，
        # 它们的文字标注位置与真实点位差公里级（实测中位 2.2 公里、
        # 最大 16.7 公里），而整条轨迹才 11 公里——**锚点偏移比轨迹还大**。
        # 这种情况下配准再"自洽"也是自洽在错误的位置上（实测 RMSE 856 米，
        # 起点被拽到 3 公里外的另一个区），置信度必须封顶，
        # 否则用户会以为结果可信。
        #
        # ⚠️ **只在「位置也不可信」时封顶**：用户填了路名时位置已经被
        # 起终点锚定修正到街区级（实测误差 300-900 米），这时封顶会
        # 把一个正确的结果说成不可信。缩放过低仍然要提示——那影响的是
        # **形状**（街道级走向无法从城市级截图恢复）。
        # --- 独立校验乘子 ---
        # 两条证据都不吃配准残差：截图里程来自 OCR，外部路由来自
        # Valhalla/OSRM。缺失结果一律不惩罚，只有明确不一致时才轻罚。
        if scale_check and not scale_check.get("consistent", True):
            conf *= 0.80
        if route_check and route_check.get("available"):
            conf *= float(route_check.get("factor", 1.0))

        if zoom and not zoom.get("position_ok", True):
            conf = min(conf, 0.2)

        # --- 2 点退化模型闸门 ---
        #
        # 相似变换恰好被 2 个点完全确定：残差恒 0、无任何冗余校验，
        # 「看起来完美」的拟合其实连一个多余观测都没有。实测（2026-11-17）
        # 高德详情页只有「诺富特」「大厦」两个泛词锚点命中，2 点模型把
        # 352 米骑行放大成 51.5 公里，置信度却仍有 0.5+、照常 ok:true 交付。
        #
        # 2 内点不必然错（浦东实测 2 内点 + 贴路 3.3m 的结果是正确的），
        # 但必须满足其一才可信：路网精修确认轨迹贴路（median_after 达标），
        # 或里程交叉校验自洽。这里只做置信度封顶——把「自信的错误」压成
        # 「明示的低可信」，最终拦截由概要里程硬闸门完成。
        n_anchor_names = len({p.name for p in reg.inliers})
        road_protected = bool(
            road_fit and road_fit.get("ok")
            and float(road_fit.get("median_after_m", 999)) <= 40)
        if n_anchor_names <= 2 and not road_protected:
            conf = min(conf, 0.3)

        # --- 真值校准（最后一步，默认不启用）---
        #
        # 上面所有乘子仍是「自洽分」。只有真的用 (截图, GPX) 做过评测、
        # 拟合出 data/confidence_calibration.json，才把分数映射到经验
        # 成功率；文件不存在/样本不足/损坏时原样返回（安全降级）。
        raw = round(min(0.99, max(0.0, conf)), 3)
        # 单元测试会用 RidePipeline.__new__ 造裸对象直接调本方法；
        # 取不到 cfg 时按全局默认路径处理（无模型则原样返回）。
        adjusted, status = apply_calibration(raw, getattr(self, "cfg", None))
        self._last_calibration = status
        return adjusted


def _anchor_dict(a: AnchorPair) -> dict[str, Any]:
    return {
        "name": a.name,
        "px": round(a.px, 1),
        "py": round(a.py, 1),
        "kind": a.kind,
        "lon": round(a.lon, 7) if a.lon is not None else None,
        "lat": round(a.lat, 7) if a.lat is not None else None,
        "quality": round(a.match_quality, 2),
        "source": a.source,
    }


def _template_geojson(
    lonlat: list[tuple[float, float]], crs: str, located: dict[str, Any]
) -> dict[str, Any]:
    """参考级结果的 GeoJSON；properties 里明确写出「未核验」与模板来源。"""
    coords = [[round(lo, 7), round(la, 7)] for lo, la in lonlat]
    match = located.get("match") or {}
    tpl = located.get("template") or {}
    props = {
        "source": "route-template",
        "crs": crs,
        "coordinate_order": "lon,lat",
        "direction": "first_point_is_start",
        "quality": "reference_only_unverified",
        "template_id": tpl.get("id"),
        "shape_residual": match.get("residual"),
        "shape_coverage": match.get("coverage"),
        "shape_reversed": match.get("reversed"),
    }
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props,
             "geometry": {"type": "LineString", "coordinates": coords}},
            {"type": "Feature", "properties": {"kind": "start", "name": "起点"},
             "geometry": {"type": "Point",
                          "coordinates": coords[0] if coords else [0, 0]}},
            {"type": "Feature", "properties": {"kind": "end", "name": "终点"},
             "geometry": {"type": "Point",
                          "coordinates": coords[-1] if coords else [0, 0]}},
        ],
    }


def _build_geojson(
    lonlat: list[tuple[float, float]], reg: Registration, track: TrackResult,
    source_id: str = "",
) -> dict[str, Any]:
    """构建 GeoJSON。

    坐标顺序严格遵循「骑行方向」：LineString 的第一个点是**起点**，
    最后一点是**终点**。这一点很重要——很多第三方工具会默认
    「首点=起点」来标注起终点，若顺序相反，地图上就会把起终点画反。

    另在 properties 中显式携带 crs 字段，避免消费方把 WGS-84 坐标
    直接画在 GCJ-02 的底图上（会造成数百米偏移）。
    """
    coords = [[round(lo, 7), round(la, 7)] for lo, la in lonlat]
    # source 跟随实际来源画像（meituan/amap/xingzhe/...），不再硬编码。
    # 空值时给通用标签，兼容未指定画像的调用。
    props = {
        "source": f"{source_id}-ride-screenshot" if source_id else "ride-screenshot",
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
