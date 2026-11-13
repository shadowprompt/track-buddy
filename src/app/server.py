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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import CONFIG, DATA_DIR, ROOT, SRC_ROOT, load_dotenv
from .cv_utils import imread_unicode
# 保留这个模块级别名：既有回归测试通过 monkeypatch `server.stitch_images`
# 注入假拼接结果，默认 SIFT 路径必须继续走它。
from .multishot import stitch as stitch_images
from .stitch_backends import (
    DEFAULT_BACKEND,
    StitchBackendError,
    list_backends,
    run_stitch,
)
from .pipeline import RidePipeline, _result_is_self_consistent
from .route_template import TemplateStore
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
        "geocoding": geo,
        "maps": maps_status,
        "lexicon": _pipeline.lexicon.stats(),
        "templates": _templates.stats(),
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
    "low_confidence": "本次识别置信度不足，未达到模板写入门槛",
    "self_inconsistent": "本次识别结果自洽性校验未通过",
    "low_information_input": "本次输入信息量不足",
    "partial_route": "细节图只覆盖了路线的一部分，形状配对未通过",
    "degenerate_shape": "轨迹形状退化，无法形成有效模板",
    "projection_failed": "概览图形状投影到地理坐标失败",
    "invalid_value": "输入的起终点数值不合法",
    "invalid_coordinate": "地理坐标不合法",
    "deduped": "轨迹形状与已有模板重复，本次跳过",
    "template_derived_result": "本次是模板参考级定位，不能再拿去注册模板",
    "stitch_failed": "细节图拼接未成功，本次没有可用的多图地理真值",
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
            why = f"模板注册过程出错（{r}）"
        else:
            why = f"模板注册条件未满足（{r}）"
    if r == "deduped":
        tail = "下次同类图可只传单图，不必重复注册"
    elif r == "template_derived_result":
        tail = "参考级结果不会二次沉淀，下次同类图仍需补传"
    else:
        tail = "下次同类图仍需补传"
    return f"概览图未能注册（{r}）：{why}；{tail}"


def _build_payload_from(result: Any, shot_id: str, shot_url: str) -> dict[str, Any]:
    """把 PipelineResult 打包成接口返回结构。"""
    return {
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
                      city: str, shot_id: str) -> dict[str, Any]:
    """把一次成功识别沉淀成轨迹模板（失败绝不影响接口返回）。

    判据与词典写入保持同源：复用 ``_result_is_self_consistent`` 的四项
    一票否决，再由 ``TemplateStore`` 裁决置信度、坐标合法性、信息量与
    形状去重。P0 只沉淀，不拿模板参与识别，因此这一步对结果是只读的。
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
) -> dict[str, Any]:
    """把「概览单图形状 ↔ 多图配准真值」注册成 ``overview`` 域模板。

    这是 P2 飞轮的入口：多图只需首次（或换路线时）补传一次，之后形状
    接近的概览单图上传时由 ``pipeline`` 阶段 2.5 直接命中模板，不必再带
    多图。``register_pair()`` 会校验多图覆盖整条路线并把真值按弧长投影
    到单图形状上；失败只写进 payload，绝不影响本次识别结果。
    """
    try:
        tr = extract_track(overview_img, CONFIG.vision)
        overview_px = tr.polyline.tolist()
        detail_px = truth_payload.get("track_polyline_px") or []
        gj = (truth_payload.get("geojson_gcj02")
              or truth_payload.get("geojson") or {})
        feats = gj.get("features") or []
        geo = feats[0]["geometry"]["coordinates"] if feats else []
        diag = truth_result.diagnostics or {}
        iq = diag.get("input_quality") or {}
        input_ok = bool(iq.get("usable", True)) if isinstance(iq, dict) else True
        return _templates.register_pair(
            overview_px=overview_px,
            detail_px=detail_px,
            geo_polyline=geo,
            confidence=float(truth_result.confidence or 0.0),
            city=city,
            shot_id=shot_id,
            self_consistent=_result_is_self_consistent(diag),
            input_ok=input_ok,
        )
    except Exception as exc:  # noqa: BLE001 — 配对注册是附加能力
        return {"written": False, "reason": f"error:{type(exc).__name__}"}


@app.post("/api/recognize_multi")
async def recognize_multi(
    files: list[UploadFile] = File(...),
    city: str = Form(""),
    output_crs: str = Form("WGS-84"),
    allow_online: bool = Form(True),
    refine: str = Form("auto"),
    road_names: str = Form(""),
    stitch_backend: str = Form(DEFAULT_BACKEND),
    overview: UploadFile | None = File(None),
) -> JSONResponse:
    """上传**多张**街道级截图，拼成一张大图后识别。

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
    try:
        result = _pipeline.run(
            canvas,
            city=city.strip(),
            allow_online=allow_online,
            output_crs=output_crs,
            refine=do_sync_refine,
            road_names=names or None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"识别过程出错: {exc}") from exc

    shot_id, shot_path = saved[0]
    payload = _build_payload_from(result, shot_id, f"/api/shots/{shot_id}{shot_path.suffix}")
    payload["stitch"] = stitch_info
    payload["n_shots"] = len(imgs)
    if overview_path is not None:
        payload["overview_url"] = f"/api/shots/{overview_id}{overview_path.suffix}"
    payload["template_capture"] = _capture_template(
        result, payload, city=city.strip(), shot_id=shot_id)
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
    return JSONResponse(payload)


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    city: str = Form(""),
    output_crs: str = Form("WGS-84"),
    allow_online: bool = Form(True),
    refine: str = Form("auto"),
    road_names: str = Form(""),
    stitch_backend: str = Form(DEFAULT_BACKEND),
    details: list[UploadFile] | None = File(None),
) -> JSONResponse:
    """上传**一张概览单图**识别；缩放过高时可补传多张街道级细节图。

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
            try:
                imgs.append(imread_unicode(sp))
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"细节图解析失败: {exc}") from exc
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

    try:
        result = _pipeline.run(
            run_img,
            city=city.strip(),
            allow_online=allow_online,
            output_crs=output_crs,
            refine=do_sync_refine,
            road_names=_names or None,
        )
    except Exception as exc:  # noqa: BLE001 — 兜底，避免单次请求打挂服务
        raise HTTPException(status_code=500, detail=f"识别过程出错: {exc}") from exc

    shot_url = canvas_url or f"/api/shots/{shot_id}{ext}"
    payload = _build_payload_from(result, shot_id, shot_url)
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
                city=city.strip(), shot_id=shot_id)
        payload["paired_registration"] = cap
        if cap.get("written"):
            pair_notes.append(
                "这张概览图已注册进轨迹模板库；以后形状接近的概览图"
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
        else:
            pair_notes.insert(0, (
                f"已用 {n_details} 张细节图拼接配准，本次结果为地理真值"
                if payload["ok"] else
                f"已用 {n_details} 张细节图拼接，但本次识别未成功（原因见下方告警）"
            ))
        payload["warnings"] = _explain_badge_loss(
            list(result.warnings) + pair_notes,
            stitch_info.get("badge_conservation"),
        )
    else:
        payload["template_capture"] = _capture_template(
            result, payload, city=city.strip(), shot_id=shot_id)
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
    return {
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


def _run_refine(shot_id: str, shot_path: Path) -> None:
    """后台精修：**复用主请求上下文**，只补算平移量再重投影。

    旧实现是 ``imread_unicode`` + ``_pipeline.run(refine=True)``，等于把
    OCR、轨迹提取、在线锚点地理编码、配准全部重跑一遍——12 图拼接画布上
    这一遍要 4 分钟，而且后台会重新取在线锚点，配额一波动就和主请求
    的判定对不上。现在只调 ``Pipeline.refine_existing``，图上那几个最贵
    的阶段一次都不重跑。
    """
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
                _persist_last_result(payload)
                if ctx.get("multi"):
                    # 多图结果单独存一份，否则之后跑一次单图识别
                    # 会把核对页上的拼接结果冲掉。
                    (DATA_DIR / "last_stitch.json").write_text(
                        json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8")
        except (OSError, ValueError, KeyError, TypeError):
            pass
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
