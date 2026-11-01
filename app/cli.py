"""命令行入口：骑行截图 → GeoJSON 轨迹。

用法::

    python -m app.cli <图片路径> [--city 上海] [--out 输出目录]
                        [--online] [--no-chart] [--no-road-verify]

产出（写入 --out 目录）：
  trajectory_gcj02.geojson   GCJ-02 基准，腾讯/高德底图直接可用
  trajectory_wgs84.geojson   WGS-84 基准，GPS 设备 / Strava / GIS 工具
  overlay_result.png         原截图 + 识别轨迹叠加
  trajectory_chart.png       独立轨迹图（米制等比例，标注路名）
  road_verify.png            OSM 道路几何投影回截图 + 轨迹（绝对位置验证）

stdout 输出一段 JSON 摘要，便于上层程序解析。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .config import CONFIG
from .cv_utils import imread_unicode
from .pipeline import RidePipeline


def _summary(res, out_dir: str, files: dict[str, str], elapsed: float) -> dict:
    dg = res.diagnostics
    gcj = res.geojson_gcj02["features"][0]["geometry"]["coordinates"]
    return {
        "ok": res.track_points > 0,
        "confidence": res.confidence,
        "track_points": res.track_points,
        "distance_m": res.distance_m,
        "crs": res.crs,
        "start_gcj02": gcj[0] if gcj else None,
        "end_gcj02": gcj[-1] if gcj else None,
        "roads": res.road_attribution.get("roads", []),
        "road_fit": {k: v for k, v in (res.road_fit or {}).items()
                     if k != "segments"},
        "diagnostics": dg,
        "warnings": res.warnings,
        "elapsed_ms": int(elapsed * 1000),
        "files": files,
        "out_dir": os.path.abspath(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="从骑行 App 截图识别轨迹并输出 GeoJSON")
    ap.add_argument("image", help="截图路径")
    ap.add_argument("--city", default="上海", help="城市提示（用于地理编码过滤）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <项目>/out）")
    ap.add_argument("--online", action="store_true",
                    help="允许调用在线地理编码服务（需已配置 AMAP_KEY 等）")
    ap.add_argument("--no-chart", action="store_true", help="不生成轨迹图")
    ap.add_argument("--no-road-verify", action="store_true",
                    help="不生成路网验证图（省一次网络请求）")
    args = ap.parse_args(argv)

    if not os.path.exists(args.image):
        print(json.dumps({"ok": False, "error": "图片不存在：%s" % args.image},
                         ensure_ascii=False))
        return 2

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.abspath(args.out or os.path.join(root, "out"))
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    img = imread_unicode(args.image)
    if img is None:
        print(json.dumps({"ok": False, "error": "图片无法解码"}, ensure_ascii=False))
        return 2

    pipe = RidePipeline(CONFIG)
    res = pipe.run(img, city=args.city, allow_online=args.online,
                   output_crs="WGS-84")
    elapsed = time.time() - t0

    files: dict[str, str] = {}
    if res.track_points:
        p_gcj = os.path.join(out_dir, "trajectory_gcj02.geojson")
        p_wgs = os.path.join(out_dir, "trajectory_wgs84.geojson")
        with open(p_gcj, "w", encoding="utf-8") as f:
            json.dump(res.geojson_gcj02, f, ensure_ascii=False, indent=1)
        with open(p_wgs, "w", encoding="utf-8") as f:
            json.dump(res.geojson, f, ensure_ascii=False, indent=1)
        files["geojson_gcj02"] = p_gcj
        files["geojson_wgs84"] = p_wgs

    try:
        from .visuals import build_chart, build_overlay, build_road_verify

        files["overlay_png"] = build_overlay(res, args.image, out_dir)
        if not args.no_road_verify:
            try:
                files["road_verify_png"] = build_road_verify(
                    res, args.image, out_dir)
            except Exception as exc:          # 网络失败不应影响主产出
                res.warnings.append("路网验证图生成失败：%s" % exc)
        if not args.no_chart:
            files["chart_png"] = build_chart(res, args.image, out_dir)
    except Exception as exc:
        res.warnings.append("可视化生成失败：%s: %s" % (type(exc).__name__, exc))

    print(json.dumps(_summary(res, out_dir, files, elapsed), ensure_ascii=False))
    return 0 if res.track_points else 1


if __name__ == "__main__":
    sys.exit(main())
