"""完整流水线 A/B：对比任意两个代码版本对同一张单图截图的识别结果。

与 exp_single_regression.py 的区别：这个脚本切换的是**整个 app 包**，
用来定位「多图功能改造 pipeline 之后，老单图样本是否退化」。

用法：
    python tools/exp_ab_full.py <pkg_root> <img> [lexicon.json]

pkg_root 传 `.` 表示当前工作区；传 `tmp_debug/ab_base` 表示
`git archive 5713484 app` 导出的基线版本。
旧包目录需要自备 data/lexicon.json、data/cache/ 和 .env
（因为 app/config.py 的 ROOT 由 __file__ 决定）。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parent.parent
pkg_root = (PROJECT / sys.argv[1]).resolve()
img_path = Path(sys.argv[2])
lexicon_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

sys.path.insert(0, str(pkg_root))

if lexicon_path:
    shutil.copyfile(lexicon_path, pkg_root / "data/lexicon.json")
    shutil.copyfile(lexicon_path, PROJECT / "data/lexicon.json")

from app.config import CONFIG  # noqa: E402
from app.pipeline import RidePipeline  # noqa: E402

img = cv2.imread(str(img_path))
if img is None:
    raise SystemExit(f"image missing: {img_path}")

res = RidePipeline(CONFIG).run(img, city="上海", allow_online=True, refine=False)
coords = res.geojson["features"][0]["geometry"]["coordinates"]
dg = res.diagnostics

summary = {
    "pkg": sys.argv[1],
    "confidence": round(res.confidence, 4),
    "distance_m": round(res.distance_m, 1),
    "track_points": res.track_points,
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
        k in w for k in ("冷启动", "内点", "备选", "徽章", "轨迹", "路名")
    )],
}
print(json.dumps(summary, ensure_ascii=False))
