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

from .config import CONFIG, DATA_DIR, ROOT, load_dotenv
from .cv_utils import imread_unicode
from .multishot import stitch as stitch_images
from .pipeline import RidePipeline
from .tmap_proxy import create_proxy_router, make_secret, render_page

load_dotenv()

app = FastAPI(
    title="美团骑行截图轨迹识别",
    description="从美团骑行截图反推 GPS 轨迹，输出 GeoJSON",
    version="0.1.0",
)

_pipeline = RidePipeline(CONFIG)
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
    geo: dict[str, Any] = {
        "amap": bool(CONFIG.amap_key),
        "tmap": bool(CONFIG.tmap_key),
        "baidu": bool(CONFIG.baidu_key),
        "tianditu": bool(CONFIG.tianditu_key),
        "online_available": CONFIG.has_online_key,
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
    return {
        "status": "ok",
        "version": app.version,
        "geocoding": geo,
        "lexicon": _pipeline.lexicon.stats(),
    }


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


@app.post("/api/recognize_multi")
async def recognize_multi(
    files: list[UploadFile] = File(...),
    city: str = Form(""),
    output_crs: str = Form("WGS-84"),
    allow_online: bool = Form(True),
    refine: str = Form("auto"),
    road_names: str = Form(""),
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

    # --- 拼接 ---
    stitch_info: dict[str, Any] = {}
    canvas = imgs[0]
    if len(imgs) > 1:
        try:
            st = stitch_images(imgs)
        except Exception as exc:  # noqa: BLE001 — 拼接失败不该让整个请求 500
            st = None
            stitch_info = {"ok": False, "notes": [f"拼接失败: {exc}"]}
        if st is not None:
            stitch_info = st.to_dict()
            if st.ok and st.canvas is not None:
                canvas = st.canvas
            elif not st.ok:
                # 拼接不上就退回第一张，但必须告诉用户
                stitch_info.setdefault("notes", []).append(
                    "拼接未成功，已退回用第 1 张图识别——"
                    "请确认各张截图有足够重叠（相邻两张至少约 1/3 重叠）"
                )
    else:
        stitch_info = {"ok": True, "notes": ["只有一张图，无需拼接"], "used": [0]}

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
    warns = list(result.warnings)
    # 多图接口不启动后台精修：拼接画布动辄几千万像素，重跑一遍
    # pipeline 的成本远高于收益。pipeline 里那条通用提示在这里会
    # 让用户以为「后台还在算」而一直等，必须换成准确说明。
    if len(imgs) > 1 and want_refine and not do_sync_refine:
        warns = [
            ("多图模式不执行后台路网精修（大画布重跑成本过高）；"
             "需要贴路校正请改用单图，或显式传 refine=on")
            if "跳过路网精修" in w else w
            for w in warns
        ]
    payload["warnings"] = warns + list(stitch_info.get("notes") or [])

    # 拼接后的画布存一份，供前端叠加显示。
    #
    # ⚠️ **必须用 PNG，不能用 JPEG**：
    # 用户要靠这张图判断「拼得对不对」，而 JPEG 的有损压缩会在
    # 接缝处产生可见的块状噪声，还会让底图的街道注记糊掉，
    # 反而更难判断。PNG 无损，画布又只是白底+拼接内容，体积可接受。
    try:
        canvas_shot_id = f"{shot_id}_stitched"
        cv2.imwrite(str(UPLOAD_DIR / f"{canvas_shot_id}.png"), canvas)
        payload["stitched_url"] = f"/api/shots/{canvas_shot_id}.png"
    except Exception:
        payload["stitched_url"] = f"/api/shots/{shot_id}{shot_path.suffix}"

    _persist_last_result(payload)
    # 多图结果单独存一份——否则之后跑一次单图识别就会把它冲掉，
    # 核对页就看不到了。
    try:
        (DATA_DIR / "last_stitch.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return JSONResponse(payload)


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    city: str = Form(""),
    output_crs: str = Form("WGS-84"),
    allow_online: bool = Form(True),
    refine: str = Form("auto"),
    road_names: str = Form(""),
) -> JSONResponse:
    """上传截图，识别轨迹并返回 GeoJSON。"""
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

    try:
        result = _pipeline.run(
            img,
            city=city.strip(),
            allow_online=allow_online,
            output_crs=output_crs,
            refine=do_sync_refine,
            road_names=_names or None,
        )
    except Exception as exc:  # noqa: BLE001 — 兜底，避免单次请求打挂服务
        raise HTTPException(status_code=500, detail=f"识别过程出错: {exc}") from exc

    payload = _build_payload_from(result, shot_id, f"/api/shots/{shot_id}{ext}")
    _persist_last_result(payload)

    # 需要后台精修吗？判据见 `_need_background_refine`：
    # 只看「配准有没有出坐标 + 是否还没精修过」，不再用尺度门槛。
    if _need_background_refine(result, want_refine, do_sync_refine):
        # ⚠️ 必须把 road_names 传进去：后台精修会**重跑整个 pipeline**，
        # 不传的话用户的路名约束（以及基于它的起终点锚定）全部丢失，
        # 精修后的结果反而比主流程更差（实测尺度会退回被污染的 15.73）。
        _start_refine(shot_id, shot_path, city.strip(), output_crs,
                      allow_online, _names or None)
        payload["refine_pending"] = True
        payload["refine_state"] = "running"
    return JSONResponse(payload)


# --- 异步路网精修 ---------------------------------------------------------

_refine_jobs: dict[str, dict[str, Any]] = {}
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
        "image_size": list(result.image_size),
    }


def _run_refine(
    shot_id: str, shot_path: Path, city: str,
    output_crs: str, allow_online: bool,
    road_names: list[str] | None = None,
) -> None:
    try:
        img = imread_unicode(shot_path)
        result = _pipeline.run(
            img, city=city, allow_online=allow_online,
            output_crs=output_crs, refine=True,
            road_names=road_names or None,
        )
        payload = _build_payload(result, shot_id, shot_path)
        with _refine_lock:
            _refine_jobs[shot_id] = {"state": "done", "payload": payload}
        # 精修结果同样要落盘，否则用户刷新页面 / 打开对照页会退回
        # 主流程那份未贴路的粗结果。两个条件都要满足：
        #   1) 精修确实被采纳（road_fit.ok）——若护栏判定「疑似贴到
        #      别的路上」而未采纳，落盘只会把未采纳的诊断信息写进
        #      用户当前看到的结果；
        #   2) 这张图仍是最近一次识别——避免精修期间用户又上传新图
        #      后被旧任务覆盖。
        if (payload.get("road_fit") or {}).get("ok"):
            try:
                prev = json.loads(
                    (DATA_DIR / "last_result.json").read_text(encoding="utf-8")
                )
                if prev.get("shot_id") == shot_id:
                    _persist_last_result(payload)
            except (OSError, ValueError, KeyError, TypeError):
                pass
    except Exception as exc:  # noqa: BLE001
        with _refine_lock:
            _refine_jobs[shot_id] = {"state": "error", "message": str(exc)}


def _start_refine(
    shot_id: str, shot_path: Path, city: str,
    output_crs: str, allow_online: bool,
    road_names: list[str] | None = None,
) -> None:
    with _refine_lock:
        if shot_id in _refine_jobs:
            return
        _refine_jobs[shot_id] = {"state": "running"}
    th = threading.Thread(
        target=_run_refine,
        args=(shot_id, shot_path, city, output_crs, allow_online, road_names),
        daemon=True,
    )
    th.start()


@app.get("/api/refine/{shot_id}")
def get_refine(shot_id: str) -> JSONResponse:
    """查询后台路网精修的进度与结果。"""
    with _refine_lock:
        job = _refine_jobs.get(shot_id)
    if job is None:
        return JSONResponse({"state": "unknown"})
    return JSONResponse(job)


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
WEB_DIR = ROOT / "web"
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
    html = page.read_text(encoding="utf-8")
    html = render_page(html, _server_port(request), _TMAP_SECRET)
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
