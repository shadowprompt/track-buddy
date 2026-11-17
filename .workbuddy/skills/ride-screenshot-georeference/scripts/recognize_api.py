"""运动截图 → 完整运动记录：本地服务 API 调用封装。

与 recognize.py（便携单图 CLI）不同，本脚本走 HTTP 服务，提供完整能力：
多来源画像、多图拼接、运动概要识别、GPX / TCX 导出、云记录。

前置：本地服务需由用户自己启动（默认 http://127.0.0.1:8848）。
本脚本只检测并调用服务，绝不代为启动进程。

用法::

    # 单图 + 来源 + 运动概要
    python recognize_api.py <轨迹图> --source amap --summary <概要图> --out <目录>

    # 多图拼接
    python recognize_api.py <图1> <图2> <图3> --source meituan --out <目录>

只用 Python 标准库（urllib / json），不引入 requests 等额外依赖。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8848"


def _post_multipart(url: str, fields: dict[str, str],
                    files: list[tuple[str, str, str]]) -> dict:
    """手工构造 multipart/form-data（无第三方依赖）。

    files: [(表单字段名, 文件路径, 内容类型), ...]
    """
    boundary = "----wbRideBoundary7MA4YWxkTrZu0gW"
    chunks: list[bytes] = []

    for key, value in fields.items():
        if value is None or value == "":
            continue
        chunks.append(
            ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
             % (boundary, key, value)).encode("utf-8"))

    for field_name, path, content_type in files:
        with open(path, "rb") as fh:
            data = fh.read()
        filename = os.path.basename(path)
        chunks.append(
            ("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
             "Content-Type: %s\r\n\r\n"
             % (boundary, field_name, filename, content_type)).encode("utf-8"))
        chunks.append(data)
        chunks.append(b"\r\n")

    chunks.append(("--%s--\r\n" % boundary).encode("utf-8"))
    body = b"".join(chunks)
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    with urllib.request.urlopen(req, timeout=660) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _health(base: str) -> dict | None:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/health",
                                    timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _save_text(out_dir: str, name: str, text: str) -> str:
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="recognize_api.py",
        description="调用本地服务识别运动截图（多图/来源/概要/GPX/TCX）")
    ap.add_argument("images", nargs="+", help="轨迹截图路径（多张则拼接）")
    ap.add_argument("--base", default=os.environ.get("TRACK_BUDDY_BASE", DEFAULT_BASE),
                    help="服务根地址，默认 http://127.0.0.1:8848")
    ap.add_argument("--source", default=None, help="来源画像 id（meituan / amap）")
    ap.add_argument("--summary", default=None, help="运动概要截图路径（可选）")
    ap.add_argument("--city", default=None, help="城市提示")
    ap.add_argument("--road-names", default=None, help="途经道路（逗号分隔，强约束）")
    ap.add_argument("--refine", default=None, help="精修档 auto / on / off")
    ap.add_argument("--out", default=None, help="输出目录（默认当前目录）")
    args = ap.parse_args()

    for p in args.images + ([args.summary] if args.summary else []):
        if not os.path.exists(p):
            print(json.dumps({"ok": False, "error": "图片不存在：%s" % p},
                             ensure_ascii=False))
            return 2

    health = _health(args.base)
    if health is None:
        print(json.dumps({
            "ok": False,
            "error": "本地服务未就绪：%s。请先由用户启动服务（python run.py "
                     "--host 127.0.0.1 --port 8848），本脚本不会代为启动。" % args.base,
        }, ensure_ascii=False))
        return 2

    out_dir = os.path.abspath(args.out or os.getcwd())
    os.makedirs(out_dir, exist_ok=True)

    files: list[tuple[str, str, str]] = []
    if len(args.images) == 1:
        files.append(("file", args.images[0], "image/jpeg"))
        endpoint = "/api/recognize"
    else:
        files += [("files", p, "image/jpeg") for p in args.images]
        endpoint = "/api/recognize_multi"
    if args.summary:
        files.append(("summary", args.summary, "image/jpeg"))

    fields = {
        "source": args.source or "",
        "city": args.city or "",
        "road_names": args.road_names or "",
        "refine": args.refine or "",
    }

    try:
        payload = _post_multipart(args.base.rstrip("/") + endpoint, fields, files)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        print(json.dumps({"ok": False, "error": "服务返回 HTTP %s" % exc.code,
                          "detail": detail}, ensure_ascii=False))
        return 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "调用服务失败：%s: %s"
                          % (type(exc).__name__, exc)}, ensure_ascii=False))
        return 1

    # 写出交付文件（服务返回的文本字段）
    saved: dict[str, str] = {}
    geojson = payload.get("geojson") or payload.get("geojson_wgs84")
    if isinstance(geojson, str) and geojson.strip():
        saved["geojson"] = _save_text(out_dir, "trajectory_wgs84.geojson", geojson)
    elif isinstance(geojson, dict):
        saved["geojson"] = _save_text(out_dir, "trajectory_wgs84.geojson",
                                      json.dumps(geojson, ensure_ascii=False, indent=1))
    for fmt in ("gpx", "tcx"):
        text = payload.get(fmt)
        if isinstance(text, str) and text.strip():
            saved[fmt] = _save_text(out_dir, "trajectory.%s" % fmt, text)

    result = {
        "ok": True,
        "service": args.base,
        "endpoint": endpoint,
        "confidence": payload.get("confidence"),
        "distance_m": payload.get("distance_m"),
        "ride_summary": payload.get("ride_summary"),
        "warnings": payload.get("warnings", []),
        "diagnostics": payload.get("diagnostics", {}),
        "files": saved,
        "out_dir": out_dir,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
