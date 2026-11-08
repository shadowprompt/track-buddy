"""单张截图识别回归 A/B 诊断。

用来回答「为什么原来测的单图现在识别不对」：把「代码版本」和「词典状态」
两个变量拆开，其余输入完全一致。

用法：
    python tools/exp_single_regression.py <lexicon.json> <img> [old_track]

    old_track 传 1 时，把 app.pipeline 里的 extract_track 换成
    ca34a75（尺度自适应重构之前）的版本，其余模块一律用当前代码。

运行前请把第一个参数指向的词典复制到 data/lexicon.json（由调用方负责），
这样 A/B 之间不会互相污染。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_old_track():
    path = ROOT / "tmp_debug/ab_old/app/track.py"
    spec = importlib.util.spec_from_file_location("app._track_ab_old", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app._track_ab_old"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    lexicon_path = Path(sys.argv[1])
    img_path = Path(sys.argv[2])
    use_old = len(sys.argv) > 3 and sys.argv[3] == "1"

    shutil.copyfile(lexicon_path, ROOT / "data/lexicon.json")

    from app.config import CONFIG
    from app.pipeline import RidePipeline
    import app.pipeline as pipeline_mod

    if use_old:
        pipeline_mod.extract_track = _load_old_track().extract_track

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"image missing: {img_path}")
        return 2

    res = RidePipeline(CONFIG).run(img, city="上海", allow_online=True, refine=False)
    coords = res.geojson["features"][0]["geometry"]["coordinates"]
    dg = res.diagnostics

    summary = {
        "confidence": round(res.confidence, 4),
        "distance_m": round(res.distance_m, 1),
        "track_points": res.track_points,
        "polyline_px": len(res.track_polyline_px),
        "coords_first": [round(v, 6) for v in coords[0]] if coords else None,
        "coords_last": [round(v, 6) for v in coords[-1]] if coords else None,
        "anchors_used": len(res.anchors_used),
        "anchors_rejected": len(res.anchors_rejected),
        "rmse_m": dg.get("rmse_m"),
        "inliers": dg.get("inliers"),
        "outliers": dg.get("outliers"),
        "model": dg.get("model"),
        "scale_m_per_px": dg.get("scale_ground_m_per_px"),
        "alternatives": len(res.alternatives),
        "warnings": [w for w in res.warnings if any(
            k in w for k in ("冷启动", "内点", "备选", "路名", "徽章", "轨迹")
        )],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
