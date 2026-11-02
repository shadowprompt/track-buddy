"""FastAPI 服务：上传截图 → 返回 GeoJSON 轨迹。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import CONFIG, DATA_DIR, ROOT, load_dotenv
from .cv_utils import imread_unicode
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
    """健康检查，同时报告当前可用的地理编码能力。"""
    return {
        "status": "ok",
        "version": app.version,
        "geocoding": {
            "amap": bool(CONFIG.amap_key),
            "tmap": bool(CONFIG.tmap_key),
            "baidu": bool(CONFIG.baidu_key),
            "tianditu": bool(CONFIG.tianditu_key),
            "online_available": CONFIG.has_online_key,
        },
        "lexicon": _pipeline.lexicon.stats(),
    }


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    city: str = Form(""),
    output_crs: str = Form("WGS-84"),
    allow_online: bool = Form(True),
    refine: str = Form("auto"),
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

    try:
        result = _pipeline.run(
            img,
            city=city.strip(),
            allow_online=allow_online,
            output_crs=output_crs,
            refine=do_sync_refine,
        )
    except Exception as exc:  # noqa: BLE001 — 兜底，避免单次请求打挂服务
        raise HTTPException(status_code=500, detail=f"识别过程出错: {exc}") from exc

    payload: dict[str, Any] = {
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
        "alternatives": result.alternatives,
        "diagnostics": result.diagnostics,
        "warnings": result.warnings,
        "elapsed_ms": result.elapsed_ms,
        "track_polyline_px": result.track_polyline_px,
        "image_size": list(result.image_size),
    }

    # 需要后台精修吗？
    #
    # 判据：尺度大（低缩放）且路网精修还没跑过。低缩放下路网精修的价值
    # 最大（贴路中位 432m → 58m，落点修正近 1 公里），但耗时也最长。
    scale = float(result.diagnostics.get("scale_ground_m_per_px") or 0.0)
    need_bg = (
        want_refine
        and not do_sync_refine
        and scale > 8.0
        and not (result.road_fit or {}).get("ok")
    )
    if need_bg:
        _start_refine(shot_id, shot_path, city.strip(), output_crs, allow_online)
        payload["refine_pending"] = True
        payload["refine_state"] = "running"
    return JSONResponse(payload)


# --- 异步路网精修 ---------------------------------------------------------

_refine_jobs: dict[str, dict[str, Any]] = {}
_refine_lock = threading.Lock()


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
) -> None:
    try:
        img = imread_unicode(shot_path)
        result = _pipeline.run(
            img, city=city, allow_online=allow_online,
            output_crs=output_crs, refine=True,
        )
        payload = _build_payload(result, shot_id, shot_path)
        with _refine_lock:
            _refine_jobs[shot_id] = {"state": "done", "payload": payload}
    except Exception as exc:  # noqa: BLE001
        with _refine_lock:
            _refine_jobs[shot_id] = {"state": "error", "message": str(exc)}


def _start_refine(
    shot_id: str, shot_path: Path, city: str,
    output_crs: str, allow_online: bool,
) -> None:
    with _refine_lock:
        if shot_id in _refine_jobs:
            return
        _refine_jobs[shot_id] = {"state": "running"}
    th = threading.Thread(
        target=_run_refine,
        args=(shot_id, shot_path, city, output_crs, allow_online),
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
