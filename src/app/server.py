"""FastAPI 服务：上传截图 → 返回 GeoJSON 轨迹。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import records
from .config import CONFIG, DATA_DIR, ROOT, SRC_ROOT, load_dotenv
from .cv_utils import imread_unicode
from .pipeline import prepare_input_image
# 保留这个模块级别名：既有回归测试通过 monkeypatch `server.stitch_images`
# 注入假拼接结果，默认 SIFT 路径必须继续走它。
from .multishot import stitch as stitch_images
from .stitch_backends import (
    DEFAULT_BACKEND,
    StitchBackendError,
    list_backends,
    run_stitch,
)
from .exporters import build_exports
from .pipeline import RidePipeline, _result_is_self_consistent, backproject_prep_px
from .route_template import TemplateStore
from .source_profile import (
    PROFILES,
    SourceProfile,
    activate_profile_globally,
    active_source_profile,
    resolve_source_profile,
)
from .tmap_proxy import create_proxy_router, make_secret, render_page
from .track import extract_track

load_dotenv()

app = FastAPI(
    title="美团骑行截图轨迹识别",
    description="从美团骑行截图反推 GPS 轨迹，输出 GeoJSON",
    version="0.1.0",
)

# 轨迹模板库：与词典并列的第二种资产，启动时加载一次（含真实位置，不入库）
# ⚠️ 必须先于 _pipeline 创建——P1 的概览图兜底要把它注入流水线。
_templates = TemplateStore(CONFIG.template_path)
_pipeline = RidePipeline(CONFIG, templates=_templates)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 腾讯地图 GL JS 的本地代理。
#
# SDK 加载时必须完成鉴权（checkKey），浏览器直接请求会把 key 暴露出去，
# 因此由本地服务代为转发。代理标记是随机串，浏览器拿不到真正的 key。
_TMAP_SECRET = make_secret()
app.include_router(
    create_proxy_router(CONFIG.tmap_gl_key, http_timeout=CONFIG.http_timeout)
)

# 实际监听端口（main() 启动时写入），供页面渲染代理地址用
_SERVER_PORT: int = 0

# 前端底图适配层（web/js/basemap.js）已注册的服务商。
# 与前端实现保持同步——新增服务商时两边都要改。
_BASEMAP_PROVIDERS = ["tmap", "amap"]


@app.get("/api/health")
def health() -> dict[str, Any]:
    """健康检查，同时报告当前可用的地理编码能力。

    ⚠️ **必须区分「有 key」和「key 能用」**（实测踩过）：
    高德 key 有平台类型之分——「Web端(JS API)」的 key 调 Web 服务接口会返回
    `USERKEY_PLAT_NOMATCH`（10009）。只检查 key 是否存在，
    会让「key 配错了」看起来像「在线服务正常」。

    所以这里对有 key 的服务做一次**主动探测**（用真实端点打一次），
    把失败原因原样报出来。探测很便宜（一次请求），只在 health 调用时发生。
    """
    from .maps.contracts import Capability
    from .maps.roadname import build_roadname_registry
    from .maps.roads import build_road_geometry_registry
    from .maps.routing import build_route_validation_registry

    # 配置状态与调用顺序全部由能力层注册表驱动：新增服务商时 health
    # 会自动多出一个键，不必再在这里补一行硬编码的 key 布尔值。
    reg = _pipeline.geocoder.registry
    names = reg.names(Capability.GEOCODING)
    geo: dict[str, Any] = {}
    for name in names:
        provider = reg.get(name, Capability.GEOCODING)
        geo[name] = bool(provider and provider.is_available())
    geo["order"] = list(reg.order(Capability.GEOCODING))
    geo["online_available"] = any(geo[n] for n in names)
    rn_reg, _ = build_roadname_registry(CONFIG)
    geo["roadname"] = {
        "available": bool(rn_reg.available(Capability.ROADNAME)),
        "order": list(rn_reg.order(Capability.ROADNAME)),
    }
    # 上一次在线调用失败的原因（若已发生过）
    if _pipeline.geocoder._last_error:
        geo["last_error"] = _pipeline.geocoder._last_error
    # 主动探测：有 key 的服务真的能用吗？
    # （key 平台类型不匹配时，key 存在但接口必然失败）
    if geo["online_available"]:
        try:
            geo["probe"] = _pipeline.geocoder.probe_online()
        except Exception:
            geo["probe"] = {"error": "探测失败"}

    # 后端四类能力统一上报：只查装配与顺序，不打公共实例，
    # 避免 health 被 Overpass / OSRM 的偶发超时拖住。
    road_reg, _ = build_road_geometry_registry(CONFIG)
    route_reg, _ = build_route_validation_registry(CONFIG)
    maps_status = {
        "road_geometry": {
            "order": list(road_reg.order(Capability.ROAD_GEOMETRY)),
            "available": bool(road_reg.available(Capability.ROAD_GEOMETRY)),
        },
        "route_validation": {
            "order": list(route_reg.order(Capability.ROUTE_VALIDATION)),
            "available": bool(route_reg.available(Capability.ROUTE_VALIDATION)),
        },
        # 前端底图（阶段 3）：适配层选中的服务商与可用实现。
        # 改 .env 的 BASEMAP_PROVIDER 后这里会立刻变，便于验收确认。
        "basemap": {
            "provider": CONFIG.basemap_provider,
            "available": _BASEMAP_PROVIDERS,
        },
    }
    return {
        "status": "ok",
        "version": app.version,
        "source_profile": {
            # 环境级默认画像（请求可用 source 入参临时切换）
            "id": CONFIG.source_profile_id,
            "title": CONFIG.source.title,
            # 全部已注册画像：前端据此渲染「输入源」选择器
            "available": [
                {"id": p.id, "title": p.title}
                for pid, p in sorted(PROFILES.items())
            ],
        },
        "geocoding": geo,
        "maps": maps_status,
        "lexicon": _pipeline.lexicon.stats(),
        "templates": _templates.stats(),
        # 零网络：只看配置齐全与 SDK 是否可导入，不真的连 COS。
        "records": {
            "enabled": records.records_enabled(),
            "sdk_available": records.store.cos_sdk_available(),
        },
    }


@app.get("/api/stitch-backends")
def stitch_backends() -> dict[str, Any]:
    """可选的拼接后端清单，供前端渲染下拉与禁用不可用项。

    只查配置、不发网络请求：大模型是否可用以 LLM_BASE_URL / LLM_API_KEY
    是否配置为依据，真正能不能调通要等实际请求时才知道。
    """
    return {"default": DEFAULT_BACKEND, "backends": list_backends()}


def _canvas_preview_id(shot_id: str, really_stitched: bool) -> str:
    """拼接画布预览的文件名（不含扩展名）。

    只有真的拼出画布才叫 ``_stitched``；退回第 1 张原图时叫
    ``_fallback``。2026-11-13 多次出现「文件名 stitched、像素却与原图
    完全一致」的误导，两个识别入口统一走这里，避免各写一套判据。
    """
    return f"{shot_id}_stitched" if really_stitched else f"{shot_id}_fallback"


def _stitch_with_backend(
    imgs: list[np.ndarray], backend: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """两个识别入口共用的拼接调用。

    ⚠️ 后端失败时**绝不改用 SIFT 出一张看起来正常的画布**：那会让
    A/B 对比里「大模型这一栏」偷偷变成 SIFT 的结果，比直接报错更糟。
    失败时返回 ``ok=False`` 且 ``backend_error=True``，由调用方决定
    是否继续用第 1 张原图做单图识别，并把原因如实写进结果。
    """
    name = (backend or DEFAULT_BACKEND).strip().lower()
    try:
        if name == DEFAULT_BACKEND:
            # 默认路径保持原有调用形态（只传 imgs）：既有回归测试靠
            # monkeypatch `server.stitch_images` 控制拼接结果。
            # 计算上与 run_stitch(backend="sift") 等价——后者也是调同一个
            # multishot.stitch() 且不带任何后端参数。
            st = stitch_images(imgs)
            if st is None:
                return {
                    "ok": False, "backend": name,
                    "notes": ["拼接未成功（拼接函数未返回结果）"],
                }, None
            info = st.to_dict()
            info["backend"] = name
            info["backend_diagnostics"] = {
                "strategy": "全组合逐对 SIFT + RANSAC + 候选峰 NCC",
                "pairs_attempted": len(imgs) * (len(imgs) - 1) // 2,
            }
            canvas = st.canvas if (st.ok and st.canvas is not None) else None
            return info, canvas
        st = run_stitch(imgs, backend=name)
    except StitchBackendError as exc:
        # 失败也要把模型诊断带出去：否则页面上「模型诊断」整块是空的，
        # 用户只能翻 last_result.json 的 warnings 才知道是超时还是参数被拒。
        diag = dict(getattr(exc, "diagnostics", None) or {})
        diag["error"] = str(exc)
        return {
            "ok": False,
            "backend": name,
            "backend_error": True,
            "backend_diagnostics": diag,
            "notes": [
                f"拼接后端失败：{exc}",
                "未回退到其它拼接后端；识别将改用第 1 张原图继续，"
                "本次结果不代表该后端的拼接效果",
            ],
        }, None
    except Exception as exc:  # noqa: BLE001 — 拼接失败不该让整个请求 500
        return {
            "ok": False,
            "backend": name,
            "notes": [f"拼接失败: {exc}"],
        }, None
    info = st.to_dict()
    canvas = st.canvas if (st.ok and st.canvas is not None) else None
    return info, canvas


def _parse_road_names(raw: str) -> list[str]:
    """把用户输入的路名拆成去重后的列表。

    支持逗号、顿号、换行分隔。
    """
    names: list[str] = []
    for part in raw.replace("\n", ",").replace("、", ",").split(","):
        p = part.strip()
        if p and p not in names:
            names.append(p)
    return names


def _save_upload(data: bytes, filename: str | None) -> tuple[str, Path]:
    """保存上传的图片，返回 (shot_id, 路径)。"""
    ext = Path(filename or "shot.jpg").suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    shot_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    shot_path = UPLOAD_DIR / f"{shot_id}{ext}"
    shot_path.write_bytes(data)
    return shot_id, shot_path


_BADGE_LABELS = {"start": "起点", "end": "终点"}

# 模板注册失败的常见原因 → 用户能看懂的解释。
# 见 route_template.register_pair() / TemplateStore.remember()。
_PAIRED_FAILURE_REASONS: dict[str, str] = {
    "not_enough_points": "本次轨迹提取未成功（见上方告警），没有可供注册的地理真值",
    "low_confidence": "本次识别置信度不足，未达到历史记录写入门槛",
    "self_inconsistent": "本次识别结果自洽性校验未通过",
    "low_information_input": "本次输入信息量不足",
    "partial_route": "细节图只覆盖了路线的一部分，形状配对未通过",
    "degenerate_shape": "轨迹形状退化，无法形成有效历史记录",
    "projection_failed": "概览图形状投影到地理坐标失败",
    "invalid_value": "输入的起终点数值不合法",
    "invalid_coordinate": "地理坐标不合法",
    "deduped": "轨迹形状与已有历史记录重复，本次跳过",
    "template_derived_result": "本次结果参考了历史记录，不能再注册成新的历史记录",
    "stitch_failed": "细节图拼接未成功，本次没有可用的多图地理真值",
    "recognition_failed": "本次多图识别未产出可用几何（见上方告警），没有可注册的真值",
    "geo_length_out_of_range": "地理真值里程不在合理区间，疑似错误解，不能固化为历史记录",
}


def _explain_badge_loss(
    warnings: list[str], conservation: dict[str, Any] | None
) -> list[str]:
    """把「未检测到徽章」改回真正的失败层。

    `track.py` 只看最终画布，它报「未检测到起点徽章」时，输入原图里
    可能明明有——徽章是在多图贴图阶段被覆盖掉的。有了拼接层的守恒
    诊断才能区分这两种情况，不再让用户去反复检查自己的截图。
    """
    if not conservation or conservation.get("ok"):
        return warnings
    missing = [k for k in (conservation.get("missing") or []) if k in _BADGE_LABELS]
    if not missing:
        return warnings
    stale = tuple(f"未检测到{_BADGE_LABELS[k]}徽章" for k in missing)
    kept = [w for w in warnings if not any(s in w for s in stale)]

    parts = []
    for kind in missing:
        shots = (conservation.get("input") or {}).get(kind) or []
        where = (
            "输入第 " + "、".join(str(i + 1) for i in shots) + " 张图有"
            if shots else "输入图里有"
        )
        parts.append(
            f"{_BADGE_LABELS[kind]}徽章（{where}，拼接画布上未检出）"
        )
    lead = (
        "拼接阶段丢失了" + "、".join(parts) +
        "：这是多图贴图覆盖造成的信息丢失，不是截图没截到；"
        "下方「轨迹形状不合理 / 未能提取到有效轨迹线」等告警都是它的连锁结果。"
    )
    return [lead] + kept


def _paired_failure_note(reason: Any) -> str:
    """模板注册失败按真实原因分流，别一律甩给「细节图没覆盖整条路线」。"""
    r = str(reason or "unknown")
    why = _PAIRED_FAILURE_REASONS.get(r)
    if why is None:
        if r.startswith("shape_"):
            why = f"概览图与多图真值的形状配对未通过（{r}）"
        elif r.startswith("error:"):
            why = f"历史记录注册过程出错（{r}）"
        else:
            why = f"历史记录注册条件未满足（{r}）"
    if r == "deduped":
        tail = "下次同类图可只传单图，不必重复注册"
    elif r == "template_derived_result":
        tail = "参考级结果不会二次沉淀，下次同类图仍需补传"
    else:
        tail = "下次同类图仍需补传"
    return f"概览图未能注册（{r}）：{why}；{tail}"


def _img_px_transform(result: Any) -> tuple[float, float, int, int] | None:
    """从诊断元数据解出 prepared → 原图坐标的换算参数 ``(s, y0, ow, oh)``。

    ``original_hw`` 缺失（旧结果 / 模板派生 / 未裁剪画像）时返回 None，
    调用方保持像素字段原样——旧数据兼容。
    """
    dg = getattr(result, "diagnostics", None) or {}
    vp = dg.get("vision_prep") or {}
    orig_hw = vp.get("original_hw")
    prep_hw = vp.get("prepared_hw")
    if not orig_hw or not prep_hw:
        return None
    try:
        oh, ow = int(orig_hw[0]), int(orig_hw[1])
        pw = int(prep_hw[1])
    except (TypeError, ValueError, IndexError):
        return None
    if min(oh, ow, pw) <= 0:
        return None
    s = ow / pw
    cf = dg.get("map_crop_applied")
    y0 = float(cf[0]) * oh if cf else 0.0
    return s, y0, ow, oh


def _to_original_img_px(payload: dict[str, Any]) -> dict[str, Any]:
    """把响应里的像素字段从 prepared 坐标系换算到**原图**坐标系。

    覆盖前端「原始截图」叠绘用到的全部字段：
    - ``image_size`` → 真实原图尺寸（前端据此算显示比例）；
    - ``track_polyline_img_px`` → 新增字段，原图坐标轨迹折线
      （``track_polyline_px`` 保持 prepared 语义不动——模板真值
      沉淀与比对在用，改动会污染飞轮）；
    - ``anchors_used`` / ``anchors_rejected`` 的 ``px/py``；
    - ``road_fit.candidate_shift_px`` → 换算成原图像素增量
      （位移是刚性平移的像素表达，按缩放比换算，无 y0 偏移）。
    """
    t = _img_px_transform(payload.get("_result_ref"))
    # _result_ref 由调用方临时塞入，换算完移除
    payload.pop("_result_ref", None)
    if t is None:
        return payload
    s, y0, ow, oh = t
    payload["image_size"] = [ow, oh]
    pl = payload.get("track_polyline_px")
    if pl:
        payload["track_polyline_img_px"] = [
            [round(x * s, 1), round(y * s + y0, 1)] for x, y in pl
        ]
    for key in ("anchors_used", "anchors_rejected"):
        items = payload.get(key) or []
        out = []
        for a in items:
            if isinstance(a, dict) and "px" in a and "py" in a:
                a = dict(a)
                a["px"] = round(float(a["px"]) * s, 1)
                a["py"] = round(float(a["py"]) * s + y0, 1)
            out.append(a)
        payload[key] = out
    rf = payload.get("road_fit") or {}
    cs = rf.get("candidate_shift_px")
    if cs and len(cs) == 2:
        rf = dict(rf)
        rf["candidate_shift_px"] = [round(float(cs[0]) * s, 1),
                                    round(float(cs[1]) * s, 1)]
        payload["road_fit"] = rf
    return payload


def _build_payload_from(result: Any, shot_id: str, shot_url: str) -> dict[str, Any]:
    """把 PipelineResult 打包成接口返回结构。

    像素字段的坐标契约：``track_polyline_px`` 等保持管线内部的
    prepared 坐标系（模板真值链路依赖），前端叠绘字段经
    ``_to_original_img_px`` 统一换算到原图坐标系。
    """
    payload = {
        "ok": result.confidence > 0 and result.track_points > 0,
        "shot_id": shot_id,
        "shot_url": shot_url,
        "geojson": result.geojson,
        "geojson_gcj02": result.geojson_gcj02,
        "confidence": result.confidence,
        "crs": result.crs,
        "track_points": result.track_points,
        "distance_m": result.distance_m,
        "anchors_used": result.anchors_used,
        "anchors_rejected": result.anchors_rejected,
        "road_attribution": result.road_attribution,
        "road_fit": result.road_fit,
        "road_name_constraints": result.road_name_constraints,
        "road_name_fit": result.road_name_fit,
        "alternatives": result.alternatives,
        "diagnostics": result.diagnostics,
        "warnings": result.warnings,
        "elapsed_ms": result.elapsed_ms,
        "track_polyline_px": result.track_polyline_px,
        "map_matched": result.map_matched,
        "image_size": list(result.image_size),
    }
    payload["_result_ref"] = result
    return _to_original_img_px(payload)


async def _preparse_ride_summary(
    summary_file: UploadFile | None,
) -> dict[str, Any] | None:
    """端点前置解析概要图：在 pipeline 配准**之前**拿到概要里程。

    概要里程是配准的**外部尺度先验**（轨迹像素长度 ÷ 真实里程），
    必须在 register 之前可用——事后校验只能拦住错误结果，
    先验才能救回本来会失败的识别（2026-11-17 街区级特写实测）。

    返回 None 表示没有概要图或解析不出字段；诊断落盘在此处做
    （pipeline 与 _attach_ride_summary 都不再重复读图）。
    """
    if summary_file is None or not (summary_file.filename or "").strip():
        return None
    try:
        data = await summary_file.read()
        await summary_file.seek(0)  # 后续 _attach_ride_summary / byte_sink 还要读
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    # 诊断落盘：概要图 OCR 读不到里程时（如 2026-11-17 352米案例），
    # 没有原图就无从排查「到底传了什么」。失败静默不影响主流程。
    try:
        _sum_dir = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
        _sum_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(_sum_dir / f"{int(time.time())}_summary.jpg"), img)
    except Exception:
        pass
    # 与 CLI 共用同一解析路径，避免两处 OCR 参数/词表漂移。
    from .ride_summary import parse_summary_image_bytes

    return parse_summary_image_bytes(data)


def _fallback_summary_from_main(
    full_img: Any,
    crop_frac: tuple[float, float],
) -> dict[str, Any] | None:
    """没传概要图时，从主截图被裁掉的底部统计卡片兜底解析概要。

    小米运动健康等 App 把「距离 / 时长 / 开始时间」直接印在主截图
    底部卡片上，``map_crop_v`` 裁剪后 pipeline 看不到它——若用户也
    没单独传概要图，配准的**外部尺度先验**与时长-里程物理闸门就全部
    失明。实测后果（2026-11-24 网页端）：同一张小米截图，CLI 传了
    --summary 落点正确，网页端没传概要，2 锚点模型的尺度爆炸无拦截，
    轨迹整体偏出约 40km。返回 None 表示条带里解析不出字段。
    """
    try:
        h = int(full_img.shape[0])
        y0 = int(float(crop_frac[1]) * h)
        if h - y0 < 40:  # 条带太薄，OCR 没有意义
            return None
        from .ride_summary import parse_summary_from_region

        out = parse_summary_from_region(full_img[y0:])
    except Exception:  # noqa: BLE001 — 兜底失败静默，不影响主流程
        return None
    if not out:
        return None
    notes = list(out.get("notes") or [])
    notes.append(
        "未单独传概要图：已从主截图底部统计卡片解析出运动概要"
        "（作为里程/时长先验参与校准）"
    )
    out["notes"] = notes
    return out


async def _attach_ride_summary(
    payload: dict[str, Any],
    summary_file: UploadFile | None,
    *,
    byte_sink: list[bytes] | None = None,
    pre_parsed: dict[str, Any] | None = None,
) -> None:
    """解析运动概要截图，把本次运动的开始时间/时长等并入识别结果（可选上传）。

    概要字段（里程/时长/卡路里/减碳/开始时间）进 ``payload["ride_summary"]``，
    非空字段同时并入 GeoJSON ``properties["ride_summary"]``——这样一次
    识别的输出就是「轨迹几何 + 运动元数据」的完整运动记录。解析失败
    只加告警，绝不影响轨迹识别本身。

    ``pre_parsed``：``_preparse_ride_summary`` 的结果。端点为了给
    pipeline 供尺度先验已经解析过一次，这里直接复用，省一次 OCR。
    注意 cross_check 仍要重算——它依赖 payload 的几何里程，
    前置解析时还不知道。

    ``pre_parsed`` 也可以在**没传概要文件**时单独出现（来自主截图
    底部卡片的兜底解析，见 ``_fallback_summary_from_main``）：此时
    照常并入 payload，让里程交叉校验与时长-里程物理闸门生效。
    """
    has_file = bool(
        summary_file is not None and (summary_file.filename or "").strip())
    if not has_file and pre_parsed is None:
        return
    if has_file:
        try:
            data = await summary_file.read()
            # 运动概要图不落盘（诊断落盘在 _preparse_ride_summary）；只有
            # 调用方要把记录存进 COS 时才收集字节。
            if byte_sink is not None:
                byte_sink.append(data)
        except Exception:
            data = b""
    if pre_parsed is not None:
        summary = dict(pre_parsed)
        summary.pop("cross_check", None)
    else:
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            payload.setdefault("warnings", []).append(
                "运动概要截图无法解码，已跳过")
            return
        from .ocr_anchor import ocr_texts
        from .ride_summary import parse_ride_summary, summary_properties

        texts = ocr_texts(img, CONFIG.vision)
        summary = parse_ride_summary(
            texts, geom_distance_m=payload.get("distance_m"),
            ride_labels=CONFIG.source.ride_labels)
    from .ride_summary import summary_cross_check, summary_properties

    summary["cross_check"] = summary_cross_check(
        summary, payload.get("distance_m"))
    payload["ride_summary"] = summary
    props = summary_properties(summary)
    if props:
        for gj_key in ("geojson_gcj02", "geojson"):
            gj = payload.get(gj_key) or {}
            feats = gj.get("features") or []
            if feats:
                feats[0].setdefault("properties", {})["ride_summary"] = dict(props)
    cc = summary.get("cross_check") or {}
    if cc.get("available") and not cc.get("consistent"):
        payload.setdefault("warnings", []).append(str(cc["note"]))
        _maybe_reject_by_summary_ratio(payload, summary, cc)
    # 时长-里程物理一致性闸门：不依赖概要里程，只要概要里有时长就能
    # 拦住尺度爆炸（概要页没传或没解析出里程时，里程交叉校验失效，
    # 这是最后一道防线）。
    _maybe_reject_by_physical_speed(payload, summary)


# 里程交叉校验硬闸门阈值。
#
# ride_summary 的 cross_check 在比值 0.6~1.7 之外只给 warning；
# 这里更严：<1/3 或 >3 意味着**数量级**错误——典型场景是锚点不足时
# 相似变换尺度爆炸（实测 352 米骑行被 2 个泛词锚点放大成 51.5 公里、
# 照常 ok:true 交付，2026-11-17）。数量级错误时拒绝输出轨迹，
# 把「自信的错误」变成「明示的失败」。
_SUMMARY_GATE_RATIO = (1.0 / 3.0, 3.0)


def _reject_bad_track(payload: dict[str, Any], msg: str) -> None:
    """统一的「拒绝交付错误轨迹」：ok=False + 清空轨迹与导出。"""
    payload["ok"] = False
    payload["message"] = msg
    payload["confidence"] = 0.0
    payload["geojson"] = {"type": "FeatureCollection", "features": []}
    payload["geojson_gcj02"] = {"type": "FeatureCollection", "features": []}
    payload["distance_m"] = 0
    payload["track_points"] = 0
    # 导出层读空 GeoJSON 后会自动跳过，这里把可能已挂上的内容摘掉
    payload.pop("gpx", None)
    payload.pop("tcx", None)
    payload.setdefault("warnings", []).append(msg)


def _maybe_reject_by_summary_ratio(
    payload: dict[str, Any],
    summary: dict[str, Any],
    cc: dict[str, Any],
) -> None:
    """概要里程与几何里程数量级严重不符时，拒绝交付错误轨迹。"""
    try:
        ratio = float(cc.get("ratio"))
    except (TypeError, ValueError):
        return
    lo, hi = _SUMMARY_GATE_RATIO
    if lo <= ratio <= hi:
        return
    dkm = summary.get("distance_km")
    gm = payload.get("distance_m")
    msg = (
        f"识别里程 {gm:.0f}m 与运动概要里程 {dkm}km 数量级严重不符"
        f"（比值 {ratio:.1f}），很可能是锚点不足导致的尺度爆炸——"
        "已拒绝输出该轨迹。建议：把地图放大到街道级、补充分段细节图"
        "（相邻截图重叠约 1/3）后重新识别；概要页里程本身可直接采信"
    )
    _reject_bad_track(payload, msg)


# 骑行**平均速度**物理上限（米/秒）。世界级冲刺瞬时能到 ~25 m/s，
# 但整段运动的平均速度不可能持续超过 ~18 m/s；取 20 m/s（72 km/h）
# 留足裕量。这个闸门只需要**时长**，不需要概要里程——概要页没传或
# 「里程没解析出来」时，里程交叉校验失效，而尺度爆炸必然打破
# 时长-里程物理关系（实测 2026-11-17：用户只传概览图，89 秒的骑行
# 被放大成 29.7 公里 = 平均时速 1202 km/h，ok:true 照常交付）。
_PHYSICAL_SPEED_MAX_MPS = 20.0


def _maybe_reject_by_physical_speed(
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """识别里程与概要时长违反物理一致性（平均速度超骑行上限）时拒绝。

    注意这是**平均速度**：红灯停车、推行都只会拉低平均值，
    所以超过上限只能是几何里程错了，不存在合理情形。
    """
    dur = summary.get("duration_s")
    dm = payload.get("distance_m")
    try:
        dur_f = float(dur)
        dm_f = float(dm)
    except (TypeError, ValueError):
        return
    if dur_f <= 0 or dm_f <= 0:
        return
    speed = dm_f / dur_f
    if speed <= _PHYSICAL_SPEED_MAX_MPS:
        return
    kmh = speed * 3.6
    msg = (
        f"识别里程 {dm_f:.0f}m 与运动时长 {dur_f:.0f}s 违反物理一致性"
        f"（平均速度 {speed:.1f} m/s = {kmh:.0f} km/h，超出骑行极限）"
        "——几何里程存在尺度爆炸，已拒绝输出该轨迹。建议补传概要页"
        "（含「骑行 XXX 千米」大卡）或分段细节图后重新识别"
    )
    _reject_bad_track(payload, msg)


def _attach_exports(payload: dict[str, Any]) -> None:
    """把 WGS-84 轨迹 + 可选运动概要交给导出层，产出能产出的运动文件。

    轨迹与概要可分离：只有轨迹就出纯轨迹 GPX，概要到位才多出 TCX、
    GPX 补时间轴。导出层对数据不足的格式直接缺席（键不出现在 payload
    里），前端据此不展示对应入口，无需额外文案。导出失败只告警，
    不影响轨迹识别。
    """
    gj = payload.get("geojson") or {}
    feats = gj.get("features") or []
    coords: list[list[float]] | None = None
    if feats:
        geom = feats[0].get("geometry") or {}
        cs = geom.get("coordinates") or []
        if cs:
            coords = [[float(lo), float(la)] for lo, la in cs]
    if not coords:
        return

    rs = payload.get("ride_summary") or {}
    # CONFIG.source_profile_id 读的是请求级画像上下文（source 入参 >
    # SOURCE_PROFILE 环境变量），单请求内导出层与识别链路自动同源。
    title = CONFIG.source.title
    try:
        exports = build_exports(
            coords, ride_summary=rs, source_title=title,
            geom_distance_m=payload.get("distance_m"))
    except Exception:  # noqa: BLE001 —— 导出失败不影响轨迹识别本身
        payload.setdefault("warnings", []).append("运动文件导出失败，已跳过")
        return
    for fmt, content in exports.items():
        payload[fmt] = content


def _persist_last_result(payload: dict[str, Any]) -> None:
    """落盘给「轨迹地图对照页」用。

    低缩放截图的定位结果光看坐标看不出对错，
    必须落到真实底图上才知道落在哪个街区。
    """
    try:
        p = dict(payload)
        gj = p.get("geojson_gcj02") or p.get("geojson") or {}
        feats = gj.get("features") or []
        coords = (
            [[round(float(lo), 6), round(float(la), 6)]
             for lo, la in feats[0]["geometry"]["coordinates"]]
            if feats else []
        )
        p["coords"] = coords
        p["coords_a"] = list(coords)
        p["coords_b"] = list(coords)
        (DATA_DIR / "last_result.json").write_text(
            json.dumps(p, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _capture_template(result: Any, payload: dict[str, Any], *,
                      city: str, shot_id: str,
                      profile_id: str = "") -> dict[str, Any]:
    """把一次成功识别沉淀成轨迹模板（失败绝不影响接口返回）。

    判据与词典写入保持同源：复用 ``_result_is_self_consistent`` 的四项
    一票否决，再由 ``TemplateStore`` 裁决置信度、坐标合法性、信息量与
    形状去重。P0 只沉淀，不拿模板参与识别，因此这一步对结果是只读的。
    ``profile_id`` 是本次请求的输入源画像（模板按画像隔离召回）。
    """
    try:
        diag = result.diagnostics or {}
        # P1：模板反推出的坐标不能再沉淀成新模板，否则错误会自我强化。
        if diag.get("template_derived"):
            return {"written": False, "reason": "template_derived_result"}
        gj = payload.get("geojson_gcj02") or payload.get("geojson") or {}
        feats = gj.get("features") or []
        coords = feats[0]["geometry"]["coordinates"] if feats else []
        iq = diag.get("input_quality") or {}
        input_ok = bool(iq.get("usable", True)) if isinstance(iq, dict) else True
        return _templates.remember(
            px_polyline=payload.get("track_polyline_px") or [],
            geo_polyline=coords,
            confidence=float(result.confidence or 0.0),
            city=city,
            shot_id=shot_id,
            self_consistent=_result_is_self_consistent(diag),
            input_ok=input_ok,
            profile=profile_id,
        )
    except Exception as exc:  # noqa: BLE001 — 沉淀失败不能拖垮识别请求
        return {"written": False, "reason": f"error:{type(exc).__name__}"}


def _capture_paired_template(
    *,
    overview_img: np.ndarray,
    truth_payload: dict[str, Any],
    truth_result: Any,
    city: str,
    shot_id: str,
    stitch_info: dict[str, Any] | None = None,
    profile_id: str = "",
) -> dict[str, Any]:
    """把「概览单图形状 ↔ 多图配准真值」注册成 ``overview`` 域模板。

    这是 P2 飞轮的入口：多图只需首次（或换路线时）补传一次，之后形状
    接近的概览单图上传时由 ``pipeline`` 阶段 2.5 直接命中模板，不必再带
    多图。``register_pair()`` 会校验多图覆盖整条路线并把真值按弧长投影
    到单图形状上；失败只写进 payload，绝不影响本次识别结果。

    证据门与单图自动沉淀不同（§22.56 死锁 A/C）：多图真值的可信度来自
    「拼接成功 + 逐像素复核 + 徽章守恒 + 识别产出几何」，画布 OCR 的
    input_quality 与单画布置信度封顶（无路名时 0.2）对它不适用——实测
    概览↔画布形状配对 residual 0.0116 / coverage 1.0，依然会被这两道
    单图门槛连坐拒写。
    """
    try:
        if not truth_payload.get("ok"):
            return {"written": False, "reason": "recognition_failed"}
        si = stitch_info or {}
        evidence_ok = True
        cons = si.get("badge_conservation")
        if isinstance(cons, dict) and not cons.get("ok"):
            evidence_ok = False
        verify = si.get("verify")
        if isinstance(verify, dict) and verify:
            try:
                if min(float(v) for v in verify.values()) < 0.5:
                    evidence_ok = False
            except (TypeError, ValueError):
                pass
        tr = extract_track(overview_img, CONFIG.vision)
        overview_px = tr.polyline.tolist()
        detail_px = truth_payload.get("track_polyline_px") or []
        gj = (truth_payload.get("geojson_gcj02")
              or truth_payload.get("geojson") or {})
        feats = gj.get("features") or []
        geo = feats[0]["geometry"]["coordinates"] if feats else []
        return _templates.register_pair(
            overview_px=overview_px,
            detail_px=detail_px,
            geo_polyline=geo,
            confidence=float(truth_result.confidence or 0.0),
            city=city,
            shot_id=shot_id,
            self_consistent=evidence_ok,
            # input_quality 是画布 OCR 的属性，与「概览↔真值」配对的可信
            # 度无关，配对路径恒按通过处理（死锁 C 的解除）。
            input_ok=True,
            # ⚠️ 画像必须透传：漏传会让空串被 _norm_profile 回落成默认
            # meituan——amap 真值注册的模板盖上美团的章，召回时被画像
            # 隔离过滤掉，「概览单图复用模板」永远打不中（2026-11-16 实测）。
            profile=profile_id,
        )
    except Exception as exc:  # noqa: BLE001 — 配对注册是附加能力
        return {"written": False, "reason": f"error:{type(exc).__name__}"}


async def _active_source(
    source: str = Form(""),
) -> SourceProfile:
    """请求级输入源画像依赖（yield 形式，自动还原）。

    ``source`` 表单入参：空 → 走 ``SOURCE_PROFILE`` 环境变量默认；
    未注册 id → 400（带可用值清单），绝不静默回退。yield 依赖与端点
    主体运行在同一个 asyncio task 里，``active_source_profile`` 设置的
    ContextVar 在整个请求处理期间生效——``CONFIG.source`` /
    ``CONFIG.vision`` / ``CONFIG.source_profile_id`` 的既有读取点
    （pipeline、visuals、multishot、input_quality、ride_summary、导出层）
    无需任何改动就自动跟随本次请求的画像。
    """
    try:
        profile = resolve_source_profile(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    with active_source_profile(profile):
        yield profile


TEMPLATE_MODES = ("auto", "prefer", "off")


def _template_mode_skip(mode: str, paired: bool) -> bool:
    """把 ``template`` 入参翻译成 pipeline 的 ``skip_template_recall``。

    * ``auto``：没传细节图时允许召回；带了细节图（paired）则跳过——
      用户愿意传细节图就一定走当次地理配准（§22.65 事故教训）；
    * ``prefer``：优先模板，paired 也不跳过（命中即省配准时间）；
    * ``off``：一律跳过。
    """
    m = (mode or "auto").strip().lower() or "auto"
    if m not in TEMPLATE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"template 取值 {mode!r} 无效；可用值："
                   f"{', '.join(TEMPLATE_MODES)}")
    if m == "off":
        return True
    if m == "prefer":
        return False
    return paired


# --- 运动记录云存储接线 ------------------------------------------------
# 本地 data/uploads 仍是调试缓存（/traj-map、/stitch-check、后台精修依赖），
# COS 才是运动记录的权威源。只有识别请求带合法 Bearer token 时才保存，
# 无 token 一次 COS 调用都不发 —— 保证旧客户端零回归。

def _local_shot_paths(payload: dict[str, Any]) -> list[Path]:
    """从 payload 里的 /api/shots/ 链接反查已落盘的图片（只读，不新存）。"""
    out: list[Path] = []
    for key in ("shot_url", "overview_url", "stitched_url"):
        url = payload.get(key)
        if not isinstance(url, str) or "/api/shots/" not in url:
            continue
        path = UPLOAD_DIR / url.rsplit("/", 1)[-1]
        if path.is_file():
            out.append(path)
    return out


def _record_image_paths(payload: dict[str, Any],
                        extra: list[Path] | None = None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in list(extra or []) + _local_shot_paths(payload):
        if path is not None and path.is_file() and path.name not in seen:
            seen.add(path.name)
            out.append(path)
    return out


def _maybe_save_record(*, uid: str | None, rec_id: str,
                       payload: dict[str, Any],
                       extra_images: list[Path] | None = None,
                       summary_bytes: bytes = b"",
                       city: str = "", union_uid: str = "") -> None:
    """识别返回前的钩子；任何云存储异常都由 service 层吞掉。"""
    if not uid:
        return
    records.service.try_save_record(
        uid=uid, rec_id=rec_id, payload=payload,
        image_paths=_record_image_paths(payload, extra_images),
        summary_bytes=summary_bytes, city=city, union_uid=union_uid,
        store=records.store.get_record_store(),
    )


@app.post("/api/recognize_multi")
async def recognize_multi(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    city: str = Form(""),
    output_crs: str = Form("WGS-84"),
    allow_online: bool = Form(True),
    refine: str = Form("auto"),
    road_names: str = Form(""),
    stitch_backend: str = Form(DEFAULT_BACKEND),
    template: str = Form("auto"),
    overview: UploadFile | None = File(None),
    summary: UploadFile | None = File(None),
    profile: SourceProfile = Depends(_active_source),
    record_uid: str | None = Depends(records.optional_record_uid),
    union_uid: str | None = Depends(records.optional_union_uid),
) -> JSONResponse:
    """上传**多张**街道级截图，拼成一张大图后识别。

    ``source`` 表单入参（可选）：截图来自哪个 App 的画像 id
    （可用值见 ``GET /api/health`` 的 ``source_profiles``）。不传走
    ``SOURCE_PROFILE`` 环境变量默认；未知 id 返回 400。模板召回/沉淀
    按该画像隔离。

    ``template`` 表单入参同 ``/api/recognize``：``auto``（默认，多图
    场景一律走当次配准）/ ``prefer``（优先模板，近复刻命中即返回
    参考级结果）/ ``off``（禁用模板）。

    ## 为什么需要这个接口

    城市级截图里，11 公里的骑行只占约 680 像素——底图上的轨迹线
    **本身就是直线**，街道锯齿在截图阶段就已丢失。这种图能定出
    「位置」（误差 300-900 米），但**恢复不了「形状」**。

    街道级截图能看清形状，但一张图盖不住整条路线。
    所以做法是：沿路线滑动截 4-6 张有重叠的图，服务端用特征匹配
    求出图间位移、拼成一张大图，再走单图的完整流程。

    拼成大图还顺带解决了**尺度**问题：
    单张街道级图的尺度会算成 16 米/像素（偏差 8 倍），
    而大图上的 POI 标注铺开在数公里范围内，尺度能被稳定定出来。

    ``overview`` 是可选的「原始概览单图」（缩放等级高、盖住整条路线、
    地名看不清）。它**不参与拼接**：与街道级细节图尺度差一个量级，
    混进 ``files`` 会让 SIFT 拼不上甚至触发「退回第 1 张」。P0 只单独
    保存并标记，P2 会用它给拼接画布做全局形状约束。

    ``stitch_backend`` 选择拼接方式（A/B 对比）：``sift`` / ``llm`` /
    ``llm_guided``，可用清单见 ``GET /api/stitch-backends``。后端不可用时
    **不会**静默改用 SIFT，结果里以 ``stitch.backend_error`` 明确报失败。
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有收到图片")
    max_bytes = CONFIG.vision.max_upload_mb * 1024 * 1024
    names = _parse_road_names(road_names)

    saved: list[tuple[str, Path]] = []
    imgs: list[np.ndarray] = []
    first_full_img: np.ndarray | None = None
    for f in files:
        data = await f.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{f.filename or '文件'} 超过 {CONFIG.vision.max_upload_mb}MB 限制",
            )
        if not data:
            continue
        shot_id, shot_path = _save_upload(data, f.filename)
        try:
            img = imread_unicode(shot_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=f"图片解析失败: {exc}") from exc
        # 画像级输入准备（裁剪 + 归一化）逐图执行：App UI 卡片与轨迹
        # 同色系（高德速度图例条/红心/缩略图），留着会污染掩膜并让
        # SIFT 拿相同的卡片特征乱配（实测 8 图画布徽章全部误检）；
        # 归一化让各图内容尺度一致、徽章面积窗口始终在标定状态。
        if first_full_img is None:
            # 第 1 张原图保留：没传概要图时，其底部统计卡片是尺度
            # 先验的唯一来源（见 _fallback_summary_from_main）。
            first_full_img = img
        img, _ = prepare_input_image(img, profile.vision)
        saved.append((shot_id, shot_path))
        imgs.append(img)

    if not imgs:
        raise HTTPException(status_code=400, detail="没有可用的图片")

    # --- 概览图（可选，独立通道）---
    # ⚠️ 绝不 append 到 imgs：概览图只用于保存/后续全局约束，不进拼接。
    overview_id: str | None = None
    overview_path: Path | None = None
    if overview is not None:
        ov = await overview.read()
        if ov and len(ov) <= max_bytes:
            overview_id, overview_path = _save_upload(ov, overview.filename)

    # 概要图前置解析：里程是配准的外部尺度先验，必须在 pipeline 之前
    # 就绪（见 _preparse_ride_summary）。结果同时供 _attach_ride_summary
    # 复用，整次请求只 OCR 概要图一次。
    summary_pre = await _preparse_ride_summary(summary)
    if summary_pre is None and first_full_img is not None:
        _mcf = getattr(profile.vision, "map_crop_v", None)
        if _mcf:
            summary_pre = _fallback_summary_from_main(first_full_img, tuple(_mcf))

    # --- 拼接 ---
    stitch_info: dict[str, Any] = {}
    canvas = imgs[0]
    stitched: np.ndarray | None = None
    if len(imgs) > 1:
        stitch_info, stitched = _stitch_with_backend(imgs, stitch_backend)
        if stitched is not None:
            canvas = stitched
        elif not stitch_info.get("backend_error"):
            # 拼接不上就退回第一张，但必须告诉用户。
            # 后端自身失败时 notes 已说明原因，这里别再塞 SIFT 的排查建议。
            stitch_info.setdefault("notes", []).append(
                "拼接未成功，已退回用第 1 张图识别——"
                "请确认各张截图有足够重叠（相邻两张至少约 1/3 重叠）"
            )
    else:
        stitch_info = {
            "ok": True,
            "backend": (stitch_backend or DEFAULT_BACKEND).strip().lower(),
            "notes": ["只有一张图，无需拼接"],
            "used": [0],
        }

    # 拼接阶段会整体覆盖 stitch_info，概览图标记必须在那之后补上。
    if overview_path is not None:
        stitch_info["overview_present"] = True
        stitch_info.setdefault("notes", []).append(
            "概览图已单独保存，不参与 SIFT 拼接（P2 将用作全局形状约束）")

    # --- 识别（复用单图流程）---
    want_refine = refine.strip().lower() in ("1", "true", "yes", "on", "auto")
    do_sync_refine = refine.strip().lower() in ("1", "true", "yes", "on")
    # 模板复用模式在重活之前校验（400 不能被 except Exception 包成 500）。
    tpl_skip = _template_mode_skip(template, True)
    try:
        result = _pipeline.run(
            canvas,
            city=city.strip(),
            allow_online=allow_online,
            output_crs=output_crs,
            refine=do_sync_refine,
            road_names=names or None,
            distance_hint_km=(summary_pre or {}).get("distance_km"),
            # 多图本来就有当次地理配准的真值路径，auto 模式绝不让模板
            # 召回短路它（2026-11-14 事故）；「优先模板」显式选择。
            skip_template_recall=tpl_skip,
            # 上传时已逐图 prepare_input_image，pipeline 内不得二次处理
            img_is_prepared=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"识别过程出错: {exc}") from exc

    shot_id, shot_path = saved[0]
    payload = _build_payload_from(result, shot_id, f"/api/shots/{shot_id}{shot_path.suffix}")
    payload["template_mode"] = (template or "auto").strip().lower() or "auto"
    payload["stitch"] = stitch_info
    payload["n_shots"] = len(imgs)
    if overview_path is not None:
        payload["overview_url"] = f"/api/shots/{overview_id}{overview_path.suffix}"
    payload["template_capture"] = _capture_template(
        result, payload, city=city.strip(), shot_id=shot_id,
        profile_id=profile.id)
    warns = list(result.warnings)
    # 多图 auto 现在会启动后台精修。pipeline 里那句通用提示没说清
    # 「只校验不平移」，留着会和后面的准确说明并存甚至矛盾，先去掉。
    if want_refine and not do_sync_refine:
        warns = [w for w in warns if "跳过路网精修" not in w]
    payload["warnings"] = _explain_badge_loss(
        warns + list(stitch_info.get("notes") or []),
        stitch_info.get("badge_conservation"),
    )

    # 拼接后的画布存一份，供前端叠加显示。
    #
    # ⚠️ **必须用 PNG，不能用 JPEG**：
    # 用户要靠这张图判断「拼得对不对」，而 JPEG 的有损压缩会在
    # 接缝处产生可见的块状噪声，还会让底图的街道注记糊掉，
    # 反而更难判断。PNG 无损，画布又只是白底+拼接内容，体积可接受。
    try:
        # 只有真正拼出画布才叫 _stitched。拼接失败/只有一张图时，这张
        # 「画布」其实就是第 1 张原图；叫 stitched 会让人误以为拼接成功
        # （2026-11-13 多次出现：文件名 stitched，像素却与原图完全一致）。
        really_stitched = (
            len(imgs) > 1
            and stitch_info.get("ok") is True
            and stitched is not None
        )
        canvas_shot_id = _canvas_preview_id(shot_id, really_stitched)
        cv2.imwrite(str(UPLOAD_DIR / f"{canvas_shot_id}.png"), canvas)
        payload["stitched_url"] = f"/api/shots/{canvas_shot_id}.png"
    except Exception:
        payload["stitched_url"] = f"/api/shots/{shot_id}{shot_path.suffix}"

    # --- 多图后台精修 ---
    # 阶段 1 之后后台不再重跑 pipeline（只补算路网校验/平移，大画布的
    # OCR 与配准一次都不重做），所以这里可以安全地把它接上。
    #
    # ⚠️ 默认 ``mode="verify"``：拼接画布只有**一个刚体配准**，depth=10
    # 的链式拼接存在累积漂移，用一个平移量去「贴路」很容易把整条轨迹
    # 压到平行的主干道上。多图后台只做校验 + 路名归属；要真正平移必须
    # 显式传 ``refine=on``（同步执行，用户自己承担耗时与风险）。
    refine_args: dict[str, Any] | None = None
    if want_refine and not do_sync_refine:
        multi_mode = len(imgs) > 1
        if _need_background_refine(result, True, False):
            refine_args = {
                "shot_id": shot_id,
                "shot_path": shot_path,
                "base_result": result,
                "mode": "verify" if multi_mode else "auto",
                "extra": {
                    "stitch": stitch_info,
                    "n_shots": len(imgs),
                    "stitched_url": payload.get("stitched_url"),
                },
            }
            payload["refine_pending"] = True
            payload["refine_state"] = "running"
            note = (
                "多图后台精修已启动：只做路网校验与路名归属，"
                "不平移配准（拼接画布存在链式累积漂移）；"
                "需要贴路校正请显式传 refine=on"
            ) if multi_mode else (
                "后台路网精修已启动，完成后会自动刷新地图"
            )
        else:
            note = "本次未启动后台精修（结果尚无可用坐标）"
        payload["warnings"] = list(payload["warnings"]) + [note]

    summary_sink: list[bytes] = []
    await _attach_ride_summary(
        payload, summary, byte_sink=summary_sink if record_uid else None,
        pre_parsed=summary_pre)
    _attach_exports(payload)
    _persist_last_result(payload)
    # 多图结果单独存一份——否则之后跑一次单图识别就会把它冲掉，
    # 核对页就看不到了。
    try:
        (DATA_DIR / "last_stitch.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    # ⚠️ 必须等上面两份结果落盘之后再启动线程：``_run_refine`` 会读
    # ``last_result.json`` 做 shot_id 保护，早启动会因为文件还没写
    # 而误判「用户已经换了新图」，精修结果就不落盘了。
    if refine_args is not None:
        _start_refine(**refine_args)
    # 云存储放到响应之后：COS 抖动/重试不该占用识别请求的等待时间，
    # 更不能让同步 I/O 卡在 async 端点里挡住整个事件循环。
    background_tasks.add_task(
        _maybe_save_record,
        uid=record_uid, rec_id=shot_id, payload=payload,
        extra_images=[p for _, p in saved]
        + ([overview_path] if overview_path is not None else []),
        summary_bytes=summary_sink[0] if summary_sink else b"",
        city=city.strip(),
        union_uid=union_uid or "",
    )
    return JSONResponse(payload)


@app.post("/api/recognize")
async def recognize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    city: str = Form(""),
    output_crs: str = Form("WGS-84"),
    allow_online: bool = Form(True),
    refine: str = Form("auto"),
    road_names: str = Form(""),
    stitch_backend: str = Form(DEFAULT_BACKEND),
    template: str = Form("auto"),
    details: list[UploadFile] | None = File(None),
    summary: UploadFile | None = File(None),
    profile: SourceProfile = Depends(_active_source),
    record_uid: str | None = Depends(records.optional_record_uid),
    union_uid: str | None = Depends(records.optional_union_uid),
) -> JSONResponse:
    """上传**一张概览单图**识别；缩放过高时可补传多张街道级细节图。

    ``source`` 表单入参（可选）：截图来自哪个 App 的画像 id
    （可用值见 ``GET /api/health`` 的 ``source_profiles``）。不传走
    ``SOURCE_PROFILE`` 环境变量默认；未知 id 返回 400。模板召回/沉淀
    按该画像隔离。

    ``template`` 表单入参（可选）控制「轨迹模板」复用策略：

    * ``auto``（默认）：没传细节图时允许模板召回（飞轮核心价值）；
      传了细节图则一律走当次地理配准真值，召回不与真值竞争
      （§22.65 事故教训）；
    * ``prefer``：优先模板——即使带了细节图也先探测召回，近复刻命中
      直接返回参考级结果（省配准时间），未命中回落完整流程；
    * ``off``：禁用模板，每次都做当次地理配准。
    未知取值返回 400。

    两个上传入口的语义
    ------------------
    * ``file``（必填）：整条路线的概览单图。平时只传它即可——若形状
      已通过配对注册沉淀过，由 ``pipeline`` 阶段 2.5 直接命中模板。
    * ``details``（可选）：沿路线滑动截的街道级细节图，仅在概览图
      缩放过高、模板库里还没有同类路线时补传。服务端拼成大图做完整
      地理配准，把**真值坐标**作为本次结果返回，同时把「概览图形状
      ↔ 真值」注册成 ``overview`` 域模板；之后形状接近的概览图就不必
      再带多图（P2 飞轮）。

    配对模式返回的叠加底图是拼接画布（与 ``track_polyline_px`` 同坐标系），
    单图本身放在 ``overview_url``；两者混用会让前端轨迹叠加错位。仅多图、
    没有概览图的场景仍走 ``/api/recognize_multi``。
    """
    data = await file.read()
    max_bytes = CONFIG.vision.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"文件超过 {CONFIG.vision.max_upload_mb}MB 限制")
    if not data:
        raise HTTPException(status_code=400, detail="空文件")

    # 保存原图，便于前端叠加显示与问题排查
    ext = Path(file.filename or "shot.jpg").suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    shot_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    shot_path = UPLOAD_DIR / f"{shot_id}{ext}"
    shot_path.write_bytes(data)

    try:
        img = imread_unicode(shot_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=f"图片解析失败: {exc}") from exc
    # 画像带 map_crop_v 时裁出地图区（概要页走独立通道，不经过这里）。
    # 保留原图与裁剪比例：没传概要图时，主图被裁掉的底部统计卡片
    # （小米「13.09 公里 / 00:59:52」等）是尺度先验的唯一来源，
    # 见 _fallback_summary_from_main。
    full_img = img
    crop_frac = getattr(profile.vision, "map_crop_v", None) or None
    # 画像级输入准备（裁剪 + 分辨率归一化，见 prepare_input_image）。
    # meta 传回 pipeline 写进 diagnostics：此前 server 裁剪不记录，
    # visuals 反投影会整体画偏（裁掉多少偏多少）。
    img, prep_meta = prepare_input_image(img, profile.vision)

    # 城市级低缩放截图的路网精修要联网拉路网（实测 40-80 秒），
    # 放在主流程里会让识别请求长时间挂起。这里改成：
    # 主流程先出粗结果并立即返回，后台线程继续做精修，
    # 前端轮询 /api/refine/{shot_id} 取精修后的结果。
    want_refine = refine.strip().lower() in ("1", "true", "yes", "on", "auto")
    do_sync_refine = want_refine and refine.strip().lower() in ("1", "true", "yes", "on")

    # 用户提供的路名（逗号/顿号/换行分隔）
    # 这些是**线约束**——让轨迹贴着用户说的那条路走。
    # 实测城市级样本上精度几乎翻倍（详见 docs/HANDOVER.md 21.9）。
    _names: list[str] = []
    for _part in road_names.replace("\n", ",").replace("、", ",").split(","):
        _p = _part.strip()
        if _p and _p not in _names:
            _names.append(_p)

    # --- 可选多图入口：拼接成画布做完整配准，拿到地理真值 -----------
    run_img = img
    paired_mode = False
    canvas_url: str | None = None
    n_details = 0
    detail_paths: list[Path] = []
    pair_notes: list[str] = []
    stitch_info: dict[str, Any] = {}
    stitch_failed = False
    if details:
        imgs: list[np.ndarray] = []
        for f in details:
            d = await f.read()
            if not d:
                continue
            if len(d) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{f.filename or '文件'} 超过 {CONFIG.vision.max_upload_mb}MB 限制",
                )
            _, sp = _save_upload(d, f.filename)
            detail_paths.append(sp)
            try:
                dimg = imread_unicode(sp)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"细节图解析失败: {exc}") from exc
            imgs.append(prepare_input_image(dimg, profile.vision)[0])
        if imgs:
            paired_mode = True
            n_details = len(imgs)
            run_img = imgs[0]
            if len(imgs) > 1:
                stitch_info, stitched = _stitch_with_backend(imgs, stitch_backend)
                if stitched is not None:
                    run_img = stitched
                else:
                    stitch_failed = True
                    pair_notes.extend(stitch_info.get("notes") or [])
                    if not stitch_info.get("backend_error"):
                        pair_notes.append(
                            "细节图拼接未成功，已退回用第 1 张识别——"
                            "请确认相邻两张至少约 1/3 重叠")
            # 画布必须存 PNG：JPEG 有损会在接缝产生块状噪声。
            # 拼接未成功时 run_img 只是第 1 张原图，别再叫 _stitched。
            canvas_name = _canvas_preview_id(
                shot_id, n_details > 1 and not stitch_failed,
            ) + ".png"
            try:
                cv2.imwrite(str(UPLOAD_DIR / canvas_name), run_img)
                canvas_url = f"/api/shots/{canvas_name}"
            except Exception:  # noqa: BLE001 — 预览图存不下不影响识别
                canvas_url = f"/api/shots/{shot_id}{ext}"
            # 「本次结果为地理真值」要等识别结束才能说，见下方 paired_mode 分支。

    # 模板复用模式必须在重活之前校验：HTTPException 在 _pipeline.run
    # 的参数求值里抛出会被 except Exception 包成 500，语义就错了。
    tpl_skip = _template_mode_skip(template, paired_mode)
    # 概要图前置解析（尺度先验），与 multi 端点相同（见其注释）。
    summary_pre = await _preparse_ride_summary(summary)
    if summary_pre is None and crop_frac:
        summary_pre = _fallback_summary_from_main(full_img, crop_frac)
    try:
        result = _pipeline.run(
            run_img,
            city=city.strip(),
            allow_online=allow_online,
            output_crs=output_crs,
            refine=do_sync_refine,
            road_names=_names or None,
            distance_hint_km=(summary_pre or {}).get("distance_km"),
            # paired 模式（带细节图）默认走当次地理配准真值；「优先模板」
            # 由 template 入参显式选择（2026-11-14 事故教训）。
            skip_template_recall=tpl_skip,
            # 上传时已 prepare_input_image（单图/细节图），pipeline 内
            # 不得二次处理；逐图 meta 只对单图直跑分支有效（canvas 是
            # 拼接空间，逐图裁剪/缩放信息不适用）。
            img_is_prepared=True,
            prep_meta=None if paired_mode else prep_meta,
        )
    except Exception as exc:  # noqa: BLE001 — 兜底，避免单次请求打挂服务
        raise HTTPException(status_code=500, detail=f"识别过程出错: {exc}") from exc

    shot_url = canvas_url or f"/api/shots/{shot_id}{ext}"
    payload = _build_payload_from(result, shot_id, shot_url)
    # 本次生效的模板复用策略与结果是否参考了模板：前端展示、排查
    # 「结果准不准」时的第一依据。
    payload["template_mode"] = (template or "auto").strip().lower() or "auto"
    if paired_mode:
        payload["overview_url"] = f"/api/shots/{shot_id}{ext}"
        payload["n_shots"] = n_details
        if stitch_info:
            payload["stitch"] = stitch_info
        if stitch_failed:
            # 拼接失败时底下只是第 1 张细节图的单图结果。拿它当「多图
            # 地理真值」注册模板，等于把一张不可信的图固化成以后所有
            # 同形状图的参考答案——错误会自我强化，必须在这里挡住。
            cap = {"written": False, "reason": "stitch_failed"}
        else:
            cap = _capture_paired_template(
                overview_img=img, truth_payload=payload, truth_result=result,
                city=city.strip(), shot_id=shot_id, stitch_info=stitch_info,
                profile_id=profile.id)
        payload["paired_registration"] = cap
        if cap.get("written"):
            pair_notes.append(
                "这张概览图已记入历史记录；以后形状接近的概览图"
                "只需单独上传，不必再带细节图")
        else:
            pair_notes.append(_paired_failure_note(cap.get("reason")))
        # 识别失败时不能宣称「结果为地理真值」——那会让人以为底下的
        # 轨迹/模板告警只是小问题。
        if stitch_failed:
            # 拼接没成功却说「地理真值」是最容易误导人的一句：
            # 底下其实只是第 1 张细节图的单图识别结果。
            pair_notes.insert(0, (
                f"已收到 {n_details} 张细节图，但本次拼接未成功，"
                "结果基于第 1 张图识别，不能作为地理真值"))
        elif (payload.get("diagnostics") or {}).get("template_derived"):
            # prefer 模式近复刻命中：结果是参考级而非本次配准真值。
            # 「已用细节图拼接 → 地理真值」在这里不成立，必须明说。
            pair_notes.insert(0, (
                "「优先参考」已命中历史路线：本次结果为参考级"
                "（由历史记录插值得出，未做当次地理配准）；"
                "如需本次独立配准，请把「参考历史记录」改为「自动」后重试"))
        else:
            pair_notes.insert(0, (
                f"已用 {n_details} 张细节图拼接配准，本次结果为地理真值"
                if payload["ok"] else
                f"已用 {n_details} 张细节图拼接，但本次识别未成功（原因见下方告警）"
            ))
        # 纯语义配准的位移只是粗比例：接缝没对齐时轨迹线会断成几段，
        # 用户只看到「未找到轨迹连通域 / 掩膜为空」会误以为是颜色或截图
        # 问题。2026-11-13 11:05 实测即 dy 全 0（详见 HANDOVER §22.55）。
        if (not payload["ok"] and stitch_info.get("ok")
                and stitch_info.get("backend") == "llm"):
            pair_notes.append(
                "本次用「纯 VLM」拼接：图间位移由大模型粗估计，接缝处"
                "轨迹线可能断开，导致「未找到轨迹连通域 / 掩膜为空」；"
                "这是位移精度问题，不是轨迹颜色问题。建议改选"
                "「大模型候选 + SIFT 精配准」(llm_guided) 或 SIFT，"
                "用同一批截图复测对比")
        payload["warnings"] = _explain_badge_loss(
            list(result.warnings) + pair_notes,
            stitch_info.get("badge_conservation"),
        )
    else:
        payload["template_capture"] = _capture_template(
            result, payload, city=city.strip(), shot_id=shot_id,
            profile_id=profile.id)
    summary_sink: list[bytes] = []
    await _attach_ride_summary(
        payload, summary, byte_sink=summary_sink if record_uid else None,
        pre_parsed=summary_pre)
    _attach_exports(payload)
    _persist_last_result(payload)

    # 需要后台精修吗？判据见 `_need_background_refine`：
    # 只看「配准有没有出坐标 + 是否还没精修过」，不再用尺度门槛。
    # ⚠️ 配对模式的 result 来自拼接画布，而 shot_path 是概览单图；拿单图
    # 路径去校验画布结果会串图，且多图画布有链式累积漂移，本来就不该平移。
    if paired_mode:
        if want_refine:
            payload["warnings"] = list(payload.get("warnings") or []) + [
                "配对模式不启动后台路网精修：结果来自拼接画布，"
                "平移会放大链式累积漂移"]
    elif _need_background_refine(result, want_refine, do_sync_refine):
        # ⚠️ 直接把主请求结果交给后台。路名约束、起终点锚定、尺度修正
        # 都已经体现在 result 里，后台只补算平移量再重投影，不会再重跑
        # OCR / 地理编码（旧实现重跑整条 pipeline，既不慢又会因在线配额
        # 波动拿到不同的锚点集合）。
        _start_refine(shot_id, shot_path, base_result=result, mode="auto")
        payload["refine_pending"] = True
        payload["refine_state"] = "running"
    # 与多图端点一致：记录写入不占用识别响应时间（见上）。
    background_tasks.add_task(
        _maybe_save_record,
        uid=record_uid, rec_id=shot_id, payload=payload,
        extra_images=detail_paths,
        summary_bytes=summary_sink[0] if summary_sink else b"",
        city=city.strip(),
        union_uid=union_uid or "",
    )
    return JSONResponse(payload)


# --- 异步路网精修 ---------------------------------------------------------

_refine_jobs: dict[str, dict[str, Any]] = {}
# shot_id -> {"base": PipelineResult, "mode": "auto" | "verify", "multi": bool}
#
# 主请求把已经算好的中间产物留在这里，后台线程直接复用。任务一结束
# （成功或失败）就从字典里删掉，避免大画布结果长期占内存。
_refine_ctx: dict[str, dict[str, Any]] = {}
_refine_lock = threading.Lock()


def _need_background_refine(
    result: Any, want_refine: bool, sync_refine: bool
) -> bool:
    """是否要在后台补跑一次路网精修。

    ⚠️ **不能用尺度当门槛**（这里踩过一次）：`pipeline._road_refine` 按
    配置项 `cfg.road_fit_low_zoom_m_per_px` 分两种模式——

      * 尺度小于分界 → **修正模式**：保持尺度与朝向，平移轨迹使
        贴路距离最小，用来消除 POI 文字标注带来的整体道路偏移。
      * 尺度大于分界 → **校验 + 命名模式**：只算贴路距离和路名，
        不做平移（低缩放下平移会过拟合到平行主干道）。

    旧代码把后台触发条件也写成「尺度必须大于同一个分界值」，把**分界值
    误当成触发门槛**，恰好把落在修正模式、最需要平移的那一类图全部排除，
    只给城市级图跑一个不会平移的校验。

    所以判据只看**状态**，不看尺度：用户允许精修、这次没跑过、
    已经配准出真实坐标、且 `road_fit` 尚未成功。采用哪种模式、
    耗时多长由 `_road_refine` 自己按配置决定，异步执行不阻塞主请求。
    验证用例覆盖分界两侧的通用尺度，见 `tests/test_server_refine.py`。
    """
    if not want_refine or sync_refine:
        return False
    if (result.road_fit or {}).get("ok"):
        return False
    diag = result.diagnostics or {}
    if float(diag.get("scale_ground_m_per_px") or 0.0) <= 0.0:
        return False
    if result.track_points <= 0:
        return False
    feats = (result.geojson or {}).get("features") or []
    coords = (feats[0].get("geometry") or {}).get("coordinates") if feats else None
    return bool(coords)


def _build_payload(result: Any, shot_id: str, shot_path: Path) -> dict[str, Any]:
    ext = shot_path.suffix.lower() or ".jpg"
    payload = {
        "ok": result.confidence > 0 and result.track_points > 0,
        "shot_id": shot_id,
        "shot_url": f"/api/shots/{shot_id}{ext}",
        "geojson": result.geojson,
        "geojson_gcj02": result.geojson_gcj02,
        "confidence": result.confidence,
        "crs": result.crs,
        "track_points": result.track_points,
        "distance_m": result.distance_m,
        "anchors_used": result.anchors_used,
        "anchors_rejected": result.anchors_rejected,
        "road_attribution": result.road_attribution,
        "road_fit": result.road_fit,
        "road_name_constraints": result.road_name_constraints,
        "road_name_fit": result.road_name_fit,
        "alternatives": result.alternatives,
        "diagnostics": result.diagnostics,
        "warnings": result.warnings,
        "elapsed_ms": result.elapsed_ms,
        "track_polyline_px": result.track_polyline_px,
        "map_matched": result.map_matched,
        "image_size": list(result.image_size),
    }
    payload["_result_ref"] = result
    return _to_original_img_px(payload)


def _inherit_refine_fields(payload: dict[str, Any], prev: dict[str, Any]) -> None:
    """精修落盘前，把主流程 payload 的沉淀字段与闸门裁决合并回来。

    精修 payload 是 ``_build_payload`` 重新组装的，天然没有主流程后来
    才加的沉淀字段；直接覆盖会把 paired_registration / template_capture
    冲掉，事后复盘就再也找不到「这次有没有注册模板」的证据。
    ride_summary 同理：订单页概要属于本次运动，精修不改轨迹的起止
    时间，丢了它 GPX 就没有时间轴了。

    里程闸门裁决也必须随 ride_summary 一起继承——数量级不符的错误
    轨迹不会因精修变得相符（精修只平移、不改尺度）。实测 2026-11-17：
    主流程闸门已判 ok=False，精修收尾覆盖 last_result.json 后
    ok=True、16.9km 假轨迹又「复活」回来了。
    """
    for k in ("paired_registration", "template_capture", "ride_summary"):
        if k not in payload and k in prev:
            payload[k] = prev[k]
    rs = payload.get("ride_summary") or {}
    cc = rs.get("cross_check") or {}
    if cc.get("available") and not cc.get("consistent"):
        _maybe_reject_by_summary_ratio(payload, rs, cc)
    # 物理一致性闸门同理：精修只平移不改尺度，尺度爆炸不会因精修消失。
    _maybe_reject_by_physical_speed(payload, rs)


def _run_refine(shot_id: str, shot_path: Path) -> None:
    """后台精修：**复用主请求上下文**，只补算平移量再重投影。

    旧实现是 ``imread_unicode`` + ``_pipeline.run(refine=True)``，等于把
    OCR、轨迹提取、在线锚点地理编码、配准全部重跑一遍——12 图拼接画布上
    这一遍要 4 分钟，而且后台会重新取在线锚点，配额一波动就和主请求
    的判定对不上。现在只调 ``Pipeline.refine_existing``，图上那几个最贵
    的阶段一次都不重跑。
    """
    # 后台线程不继承请求的 ContextVar：入口处按 ctx 里记录的画像 id
    # 重建上下文，精修收尾（导出层源标题等）才与主请求同源。
    with _refine_lock:
        _ctx0 = _refine_ctx.get(shot_id)
    _pid = (_ctx0 or {}).get("profile_id")
    if _pid:
        try:
            activate_profile_globally(resolve_source_profile(_pid))
        except ValueError:
            pass  # 画像等待期间被移除：回落默认，精修不该因此失败
    try:
        with _refine_lock:
            ctx = _refine_ctx.get(shot_id)
        if ctx is None:
            # 按「配置哪个就用哪个、失败直接提示」的原则，不静默降级
            # 回旧的重跑逻辑；明确报错，让用户重新识别即可。
            raise RuntimeError("精修上下文已失效，请重新识别该截图")
        result = _pipeline.refine_existing(ctx["base"], mode=ctx["mode"])
        payload = _build_payload(result, shot_id, shot_path)
        payload.update(ctx.get("extra") or {})
        # 精修结果同样要落盘，否则用户刷新页面 / 打开对照页会退回
        # 主流程那份未精修的粗结果。
        #
        # ⚠️ 早期只在上层判 ``road_fit.ok`` 时才落盘，理由是「未采纳就
        # 不必写诊断信息」。但候选路线可视化上线后这个前提不成立了：
        # 护栏否决时 ``candidate_shift_*``、拒因与形状提示**只存在于
        # 后台 payload**，不落盘的话刷新后橙色虚线和「在地图上查看
        # 未采纳的贴路位置」按钮会一起消失。否决时输出位移为 0、
        # 坐标一字未改，落盘只是补齐诊断信息，不会改动结果本身。
        #
        # 唯一保留的护栏是 shot_id：精修期间用户又上传新图时，
        # 旧任务不能覆盖更新的结果。
        try:
            prev = json.loads(
                (DATA_DIR / "last_result.json").read_text(encoding="utf-8")
            )
            if prev.get("shot_id") == shot_id:
                # 覆盖前把主流程的沉淀字段与闸门裁决合并回来（见函数注释）。
                _inherit_refine_fields(payload, prev)
                refine_valid = True
        except (OSError, ValueError, KeyError, TypeError):
            refine_valid = False
        # GPX 按精修后的坐标重新合成（轨迹可能整体平移，旧 GPX 已过期）；
        # 时间轴来自 ride_summary，与坐标无关，不受精修影响。
        _attach_exports(payload)
        if refine_valid:
            _persist_last_result(payload)
            if ctx.get("multi"):
                # 多图结果单独存一份，否则之后跑一次单图识别
                # 会把核对页上的拼接结果冲掉。
                (DATA_DIR / "last_stitch.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8")
        # ⚠️ 落盘完成后才对外宣布 done：前端轮询拿到结果就会立刻渲染，
        # 若此时核对页的 last_result / last_stitch 还没写，用户刷新页面
        # 又会退回未精修的粗结果。
        with _refine_lock:
            _refine_jobs[shot_id] = {"state": "done", "payload": payload}
    except Exception as exc:  # noqa: BLE001
        with _refine_lock:
            _refine_jobs[shot_id] = {"state": "error", "message": str(exc)}
    finally:
        # 上下文只在任务执行期间需要；无论成败都立刻释放。
        with _refine_lock:
            _refine_ctx.pop(shot_id, None)


def _start_refine(
    shot_id: str, shot_path: Path, *,
    base_result: Any,
    mode: str = "auto",
    extra: dict[str, Any] | None = None,
) -> None:
    """登记并启动一次后台精修。

    ``base_result`` 是主请求刚算出的 ``PipelineResult``（含
    ``refine_context``）。``mode`` 见 ``Pipeline._road_refine``。
    ``extra`` 是需要原样带回后台结果的多图专有字段
    （``stitch`` / ``n_shots`` / ``stitched_url``）——前端拿到精修结果
    后要重新渲染，缺了它们会丢掉拼接画布。
    """
    with _refine_lock:
        if shot_id in _refine_jobs:
            return
        _refine_jobs[shot_id] = {"state": "running",
                                 "started_at": time.time()}
        _refine_ctx[shot_id] = {
            "base": base_result, "mode": mode,
            "extra": dict(extra or {}), "multi": bool(extra),
            # 后台线程不继承请求的 ContextVar——把画像 id 带过去，
            # _run_refine 入口重建上下文，导出层（GPX/TCX 源标题）等
            # 画像敏感的收尾步骤才不会串源。
            "profile_id": CONFIG.source_profile_id,
        }
    th = threading.Thread(
        target=_run_refine,
        args=(shot_id, shot_path),
        daemon=True,
    )
    th.start()


@app.get("/api/refine/{shot_id}")
def get_refine(shot_id: str) -> JSONResponse:
    """查询后台路网精修的进度与结果。

    运行中额外返回 ``elapsed_s``：前端据此在长任务时给出「仍在计算」
    的提示，而不是让用户对着一个不动的状态条一直等。
    """
    with _refine_lock:
        job = _refine_jobs.get(shot_id)
    if job is None:
        return JSONResponse({"state": "unknown"})
    out = dict(job)
    if out.get("state") == "running" and out.get("started_at"):
        out["elapsed_s"] = round(time.time() - float(out["started_at"]), 1)
    return JSONResponse(out)


@app.get("/api/shots/{name}")
def get_shot(name: str) -> FileResponse:
    """回传原图，供前端叠加轨迹显示。"""
    path = UPLOAD_DIR / name
    if not path.exists() or path.parent != UPLOAD_DIR:
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(path)


# --- 运动记录（腾讯 COS，后端代理）-------------------------------------
# COS 私钥只在服务端：客户端拿到的只有短期 HMAC token 与 JSON/图片流。
# 所有 /api/records* 都必须带 Bearer token；uid 由 openid 派生，
# 对象键按 uid 分目录，跨用户访问一律 404（不泄露对象是否存在）。

def _records_store():
    store = records.store.get_record_store()
    if store is None:
        raise HTTPException(status_code=503, detail="运动记录存储未就绪")
    return store


@app.post("/api/auth/wxlogin")
async def wxlogin(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """``wx.login`` 的 code 换记录 token；绝不回显 openid/session_key。"""
    if not records.records_enabled():
        raise HTTPException(status_code=503, detail="运动记录功能未启用")
    code = str((payload or {}).get("code") or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="缺少 code")
    try:
        data = records.auth.code2session(code)
    except records.WeChatServiceError as exc:
        raise HTTPException(status_code=502, detail="微信登录服务暂不可用") from exc
    except records.TokenError as exc:
        raise HTTPException(status_code=401, detail="微信登录失败，请重试") from exc
    secret = CONFIG.records_token_secret or ""
    uid = records.uid_from_openid(str(data["openid"]), secret)
    # 只有绑定到同一微信开放平台账号的小程序才会返回 unionid；没有时
    # 优雅降级，uid 仍由 openid 派生，现有 COS 数据一行都不用迁移。
    unionid = str(data.get("unionid") or "").strip()
    union_uid = (records.union_uid_from_unionid(unionid, secret)
                 if unionid else "")
    token = records.issue_token(
        uid, secret=secret, ttl=CONFIG.records_token_ttl,
        union_uid=union_uid or None)
    # session_key / 明文 unionid 到此为止：不落库、不落 COS、不进响应。
    # 响应里只给脱敏锚点，小程序可据此对齐跨端身份。
    return {"token": token, "expires_in": CONFIG.records_token_ttl,
            "union_uid": union_uid}


@app.get("/api/records")
def list_records(
    limit: int = 20,
    cursor: str | None = None,
    uid: str = Depends(records.require_record_uid),
) -> dict[str, Any]:
    """当前用户的记录列表（keyset 分页，index.json 缺失会自动重建）。"""
    store = _records_store()
    try:
        items, next_cursor, total = store.list_records(
            uid, cursor=cursor, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="cursor 无效") from exc
    except Exception as exc:  # noqa: BLE001 — 云故障统一 502
        raise HTTPException(status_code=502, detail="运动记录读取失败") from exc
    return {"items": items, "next_cursor": next_cursor, "total": total}


@app.get("/api/records/identity")
def record_identity(
    ident: dict[str, Any] = Depends(records.require_record_identity),
) -> dict[str, Any]:
    """当前 token 的跨端身份：只返回本人 token 里的 union 关联，不接受
    客户端指定 union_uid，避免借此枚举其他用户的 uid。"""
    uid, union_uid = ident["uid"], ident["union_uid"]
    if not union_uid:
        return {"uid": uid, "union_uid": "", "linked_uids": [uid]}
    store = _records_store()
    try:
        linked = store.linked_uids(union_uid)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail="跨端身份读取失败") from exc
    if uid not in linked:
        linked = [uid] + linked
    return {"uid": uid, "union_uid": union_uid, "linked_uids": linked}


@app.get("/api/records/{rec_id}")
def get_record(
    rec_id: str,
    uid: str = Depends(records.require_record_uid),
) -> dict[str, Any]:
    """单条完整记录（含 geojson/gpx/tcx/diagnostics）。"""
    store = _records_store()
    try:
        found = store.get_record(uid, rec_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="运动记录读取失败") from exc
    if found is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    return records.service.record_to_response(found)


@app.get("/api/records/{rec_id}/images/{name}")
def get_record_image(
    rec_id: str,
    name: str,
    uid: str = Depends(records.require_record_uid),
) -> StreamingResponse:
    """图片走后端代理流：不签发可被转发的公网预签名 URL。"""
    store = _records_store()
    try:
        found = store.get_image(uid, rec_id, name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="运动记录读取失败") from exc
    if found is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    data, media_type = found
    return StreamingResponse(iter([data]), media_type=media_type)


@app.delete("/api/records/{rec_id}")
def delete_record(
    rec_id: str,
    uid: str = Depends(records.require_record_uid),
) -> dict[str, Any]:
    """删除我的整条记录（对象 + 列表卡片）。"""
    store = _records_store()
    try:
        ok = store.delete_record(uid, rec_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="运动记录删除失败") from exc
    if not ok:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True}


@app.post("/api/feedback")
async def feedback(payload: dict[str, Any]) -> dict[str, Any]:
    """用户确认/纠正锚点。

    这是「越用越准」的数据入口：每次纠正都会强化本地词典中
    该地名对应的坐标，下次同区域识别时直接命中，精度更高。
    """
    name = str(payload.get("name", "")).strip()
    lon = payload.get("lon")
    lat = payload.get("lat")
    if not name or lon is None or lat is None:
        raise HTTPException(status_code=400, detail="缺少 name/lon/lat")

    city = str(payload.get("city", "")).strip()
    # 词典内部统一存 GCJ-02
    if str(payload.get("crs", "GCJ-02")).upper().startswith("WGS"):
        from .geo.coords import wgs84_to_gcj02

        lon, lat = wgs84_to_gcj02(float(lon), float(lat))
    _pipeline.lexicon.remember(
        name, float(lon), float(lat), city=city, source="user", boost=2.0
    )
    return {"ok": True, "lexicon": _pipeline.lexicon.stats()}


@app.post("/api/feedback/reset")
def reset_feedback(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """清空**全部**人工纠正点位，保留自动写入的点位。

    这是「纠错了怎么办」的回滚入口：用户点位会短路掉在线地理编码，
    一条错纠正会长期生效，且当前没有单条删除接口。清空后该城市回到
    纯自动证据重新积累，内存与磁盘同步更新，**不需要重启服务**。

    要求显式 ``confirm=true``，避免误触把核验成果清掉；没有人工点位时
    返回计数 0（幂等），不会重写磁盘文件。
    """
    if not payload or payload.get("confirm") is not True:
        raise HTTPException(
            status_code=400, detail="需要 confirm=true 才会清空人工纠正")
    report = _pipeline.lexicon.clear_user_points()
    return {"ok": True, **report, "lexicon": _pipeline.lexicon.stats()}


@app.get("/api/lexicon")
def lexicon() -> dict[str, Any]:
    """查看本地词典内容。"""
    return {
        "stats": _pipeline.lexicon.stats(),
        "entries": _pipeline.lexicon._data,  # noqa: SLF001 — 只读展示
    }


@app.get("/api/lexicon/export")
def lexicon_export() -> FileResponse:
    """导出词典为 JSON 文件。"""
    path = DATA_DIR / "lexicon.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="词典尚未建立")
    return FileResponse(path, media_type="application/json", filename="lexicon.json")


# --- 静态前端 ---
WEB_DIR = SRC_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def _server_port(request: Request | None = None) -> int:
    """取本服务实际监听的端口。

    页面里的腾讯地图代理需要知道往哪个端口发请求。
    端口在 ``main()`` 启动时记录（命令行可能用 --port 改过）；
    读不到时从当前请求的 Host 头反推——这最可靠，
    因为它就是浏览器实际连上来的地址。
    """
    if _SERVER_PORT:
        return _SERVER_PORT
    if request is not None:
        host = request.headers.get("host", "")
        if ":" in host:
            try:
                return int(host.rsplit(":", 1)[1])
            except ValueError:
                pass
    try:
        for server in getattr(app, "servers", []) or []:
            for sock in getattr(server, "sockets", []) or []:
                try:
                    return int(sock.getsockname()[1])
                except (OSError, IndexError, TypeError):
                    continue
    except Exception:  # noqa: BLE001 — 读不到就用默认值
        pass
    return 8000


@app.get("/stitch-check")
async def stitch_check(request: Request, shot: str = "") -> Response:
    """多图拼接的核对页。

    把三样东西按**同一坐标系**叠在一张画布上：

    1. 拼接图（大图，网页缩放到能看全）；
    2. 每张原图在画布上的位置框（黄色虚线，编号）；
    3. 识别出的轨迹（绿色）与锚点（黄色点）。

    只给「拼接成功 + 复核分 0.95」这样的数字，用户没法判断拼得对不对；
    这个页面让他一眼看出：位置框是不是首尾相接、轨迹有没有贴住底图。

    页面还能**下载对照图**（PNG），方便留存或发给别人看。
    """
    from .stitchcheck import load_stitch_result, render_stitch_check

    result = load_stitch_result(shot or None)
    html = render_stitch_check(result)
    html = render_page(html, _server_port(request), _TMAP_SECRET)
    return Response(content=html, media_type="text/html")


@app.get("/traj-map")
async def traj_map(request: Request, shot: str = "") -> Response:
    """把识别结果画到真实腾讯地图上（可选对比「有/无路名约束」两条轨迹）。

    低缩放截图的定位结果光看坐标看不出对错——必须落到真实底图上
    才能一眼看出「落在哪个街区」。
    """
    from .trajmap import render_traj_map, load_result

    result = load_result(shot or None)
    html = render_traj_map(result)
    html = render_page(html, _server_port(request), _TMAP_SECRET)
    return Response(content=html, media_type="text/html")


@app.get("/")
def index(request: Request) -> Response:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="前端页面缺失")
    # 渲染占位符：腾讯地图 SDK 需要知道本地代理的端口与标记。
    # 这些值原本是硬编码在页面里的 __WB_HTTP_PORT__ / __WB_TMAP_SECRET__，
    # 从未被替换过——SDK 因此去请求一个非法地址，鉴权失败，
    # 最终抛出没有上下文的 "Unhandled error. (undefined)"。
    #
    # 另外注入底图服务商名与（切高德时的）高德 JS SDK 片段，
    # 前端适配层据此选择实现，业务代码与服务商解耦。
    html = page.read_text(encoding="utf-8")
    html = render_page(
        html, _server_port(request), _TMAP_SECRET,
        basemap_provider=CONFIG.basemap_provider,
        amap_js_key=CONFIG.amap_js_key,
        amap_js_security=CONFIG.amap_js_security,
    )
    return Response(content=html, media_type="text/html")


def main() -> None:
    """命令行入口：python -m app.server"""
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="美团骑行截图轨迹识别服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    global _SERVER_PORT
    _SERVER_PORT = args.port

    print(f"服务启动中… http://{args.host}:{args.port}")
    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
