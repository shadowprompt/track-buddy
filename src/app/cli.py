"""命令行入口：骑行截图 → GeoJSON / GPX（+ 可选概要 → TCX）。

用法::

    python -m app.cli <图片路径> [--city 上海] [--out 输出目录]
                        [--summary 概要截图] [--online]
                        [--no-chart] [--no-road-verify]

轨迹截图与运动概要截图是可分离的两类输入：

* 只有轨迹截图：产出 ``trajectory_wgs84.geojson`` + ``track.gpx``
  （纯轨迹、无时间轴）——这是完整的一等公民产物，不是降级；
* 额外传 ``--summary``：GPX 补上逐点 ``<time>``，并多产 ``track.tcx``。

产出（写入 --out 目录）：
  trajectory_wgs84.geojson   WGS-84 基准（交付统一坐标系），GPS 设备 /
                             Strava / GIS 工具；GCJ-02 是识别与底图
                             显示的内部坐标，不再作为 CLI 交付物
  track.gpx                  GPX 1.1：无概要=纯轨迹；有概要=含时间轴
  track.tcx                  TCX v2：仅概要含开始时间+时长时产出
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


def _summary(res, out_dir: str, files: dict[str, str], elapsed: float,
             ride_summary: dict | None = None) -> dict:
    dg = res.diagnostics
    # 识别失败时 geojson_gcj02 是空字典（PipelineResult 的默认值为 {}），
    # 没有 "features" 键。早期实现直接 res.geojson_gcj02["features"][0]，
    # 失败路径上会抛 KeyError: 'features' —— CLI 崩溃而不是输出 JSON 摘要，
    # 上层脚本拿不到任何可解析的结果。这里做防御性读取。
    gcj: list = []
    feats = (res.geojson_gcj02 or {}).get("features") or []
    if feats:
        coords = (feats[0].get("geometry") or {}).get("coordinates") or []
        gcj = coords
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
        # 概要与导出能力：调用方据此判断拿到的是纯轨迹还是完整运动记录
        "ride_summary": ride_summary,
        "exports": [k for k in files if k.startswith("track_")],
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
    ap.add_argument("--summary", default=None,
                    help="运动概要截图（可选）：提供开始时间/时长后 GPX "
                         "带时间轴并额外产出 TCX；不传则只导出纯轨迹 GPX")
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

    # 运动概要截图（可选）：与 server 共用解析路径，既给配准供尺度先验，
    # 又给导出层供时间轴。解析失败只加告警，不影响轨迹主流程。
    summary: dict | None = None
    if args.summary:
        if not os.path.exists(args.summary):
            print(json.dumps({"ok": False, "error":
                              "概要截图不存在：%s" % args.summary},
                             ensure_ascii=False))
            return 2
        try:
            with open(args.summary, "rb") as fh:
                from .ride_summary import parse_summary_image_bytes

                summary = parse_summary_image_bytes(fh.read())
        except Exception as exc:  # noqa: BLE001
            summary = None
            print("[warn] 概要截图解析失败：%s" % exc, file=sys.stderr)

    pipe = RidePipeline(CONFIG)
    res = pipe.run(img, city=args.city, allow_online=args.online,
                   output_crs="WGS-84",
                   distance_hint_km=(summary or {}).get("distance_km"))
    elapsed = time.time() - t0
    if args.summary and summary is None:
        res.warnings.append("运动概要截图解析失败，已按纯轨迹导出")

    files: dict[str, str] = {}
    if res.track_points:
        # 交付文件统一 WGS-84（GPS 标准）；GCJ-02 是识别/显示的内部坐标，
        # 不再作为 CLI 交付物。
        p_wgs = os.path.join(out_dir, "trajectory_wgs84.geojson")
        with open(p_wgs, "w", encoding="utf-8") as f:
            json.dump(res.geojson, f, ensure_ascii=False, indent=1)
        files["geojson_wgs84"] = p_wgs

        # 运动文件导出：与 server 的 _attach_exports 共用同一导出层。
        # 只有轨迹就出纯轨迹 GPX，概要到位才多出 TCX；数据不足的格式
        # 由导出层直接缺席，不额外解释。
        try:
            from .exporters import build_exports

            feats = (res.geojson or {}).get("features") or []
            coords = [
                [float(lo), float(la)]
                for lo, la in ((feats[0].get("geometry") or {})
                               .get("coordinates") or [])
            ] if feats else []
            if coords:
                exports = build_exports(
                    coords, ride_summary=summary,
                    source_title=CONFIG.source.title,
                    geom_distance_m=res.distance_m)
                for fmt, content in exports.items():
                    p_fmt = os.path.join(out_dir, f"track.{fmt}")
                    with open(p_fmt, "w", encoding="utf-8") as f:
                        f.write(content)
                    files[f"track_{fmt}"] = p_fmt
        except Exception as exc:  # 导出失败不影响主产出
            res.warnings.append("运动文件导出失败：%s: %s"
                                % (type(exc).__name__, exc))

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

    print(json.dumps(_summary(res, out_dir, files, elapsed, summary),
                     ensure_ascii=False))
    return 0 if res.track_points else 1


if __name__ == "__main__":
    sys.exit(main())
