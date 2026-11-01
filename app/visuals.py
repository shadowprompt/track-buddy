"""识别结果的可视化产出（包内模块，供 CLI 与技能脚本调用）。

三个产出：
  overlay_result.png   原截图 + 识别轨迹叠加
  road_verify.png      OSM 道路几何投影回截图 + 轨迹（验证绝对位置）
  trajectory_chart.png 独立轨迹图（米制等比例，标注路名）

所有函数都显式接收 sample 路径与输出目录，不依赖全局状态。
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

from .config import CONFIG
from .cv_utils import imread_unicode, imwrite_unicode
from .geo.coords import lonlat_to_mercator, wgs84_to_gcj02
from .ocr_anchor import build_anchors, ocr_texts
from .pipeline import RidePipeline
from .register import register

# 用户实骑道路（2026-10-30 由用户确认）
FIVE_ROADS = {
    "安龙路": (0, 140, 255),
    "新渔东路": (255, 130, 0),
    "新渔路": (255, 0, 255),
    "剑河路": (0, 255, 255),
    "金钟路": (0, 220, 0),
    "北翟路": (0, 0, 255),
}


def _normalize(res):
    """统一输入形态：接受 PipelineResult 对象或 API 返回的 dict。"""
    if isinstance(res, dict):
        return res
    return {
        "geojson": res.geojson,
        "geojson_gcj02": res.geojson_gcj02,
        "road_attribution": res.road_attribution,
        "road_fit": res.road_fit,
        "diagnostics": res.diagnostics,
        "track_points": res.track_points,
        "distance_m": res.distance_m,
        "confidence": res.confidence,
        "warnings": res.warnings,
    }


def _tag(sample: str) -> str:
    """输出文件名前缀：不同样本不互相覆盖。"""
    base = os.path.splitext(os.path.basename(sample))[0]
    return "" if base == "meituan_ride_shanghai" else base + "_"


def _registration(res: dict, sample: str):
    """复现流水线的配准（含路网精修平移量）。"""
    img = imread_unicode(sample)
    pipe = RidePipeline(CONFIG)
    items = ocr_texts(img, CONFIG.vision)
    anchors = build_anchors(img, items, CONFIG.vision)
    warn: list[str] = []
    pairs = pipe._resolve_anchors(anchors, "上海", False, warn)
    reg = register(pairs, inlier_m=CONFIG.vision.ransac_inlier_m,
                   min_inliers=CONFIG.vision.ransac_min_inliers)
    rf = res.get("road_fit") or {}
    if rf.get("ok"):
        reg.translation = reg.translation + np.array(
            [rf["shift_e_m"], rf["shift_n_m"]], dtype=float)
    return img, reg


def build_overlay(res, sample: str, out_dir: str) -> str:
    res = _normalize(res)
    """把识别出的经纬度轨迹反投影回原截图，叠加显示。"""
    img, reg = _registration(res, sample)
    gcj = res["geojson_gcj02"]["features"][0]["geometry"]["coordinates"]
    # 注意：geojson_gcj02 里的坐标**已经是 GCJ-02**，不能再做 wgs84_to_gcj02，
    # 否则会重复叠加一次基准偏移（约 500 米 / 100+ 像素），画出来必然偏离轨迹。
    rec = np.array([
        reg.mercator_to_pixel(*lonlat_to_mercator(lo, la))
        for lo, la in gcj
    ])

    vis = img.copy()
    for i in range(len(rec) - 1):
        cv2.line(vis, tuple(rec[i].astype(int)), tuple(rec[i + 1].astype(int)),
                 (255, 200, 0), 6, cv2.LINE_AA)
    for i in range(0, len(rec), max(1, len(rec) // 45)):
        cv2.circle(vis, tuple(rec[i].astype(int)), 7, (255, 255, 255), -1)
    cv2.circle(vis, tuple(rec[0].astype(int)), 27, (0, 200, 0), -1)
    cv2.circle(vis, tuple(rec[0].astype(int)), 28, (255, 255, 255), 3)
    cv2.circle(vis, tuple(rec[-1].astype(int)), 27, (0, 0, 235), -1)
    cv2.circle(vis, tuple(rec[-1].astype(int)), 28, (255, 255, 255), 3)
    cv2.putText(vis, "START", tuple((rec[0] + np.array([34, -22])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 130, 0), 2)
    cv2.putText(vis, "END", tuple((rec[-1] + np.array([34, -22])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 180), 2)
    for a in res.get("anchors_used", []):
        cv2.circle(vis, (int(a["px"]), int(a["py"])), 16, (0, 230, 230), 3)
        cv2.putText(vis, a["name"], (int(a["px"]) + 20, int(a["py"]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 150), 2)

    rf = res.get("road_fit") or {}
    dg = res["diagnostics"]
    roads = res.get("road_attribution", {}).get("roads", [])
    cv2.rectangle(vis, (26, 26), (900, 198), (255, 255, 255), -1)
    cv2.rectangle(vis, (26, 26), (900, 198), (70, 70, 70), 2)
    lines = [
        "Recognized trajectory (cyan) | START green | END red",
        "road-network corrected: E%+.0f N%+.0f m | road dist %.0f -> %.0f m"
        % (rf.get("shift_e_m", 0), rf.get("shift_n_m", 0),
           rf.get("median_before_m", 0), rf.get("median_after_m", 0)),
        "scale %.3f m/px | %d points | %.3f km | confidence %.3f"
        % (dg["scale_ground_m_per_px"], res["track_points"],
           res["distance_m"] / 1000, res["confidence"]),
        "roads: %s" % " / ".join(roads[:8]),
    ]
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (42, 58 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (25, 25, 25), 2, cv2.LINE_AA)

    out = os.path.join(out_dir, _tag(sample) + "overlay_result.png")
    imwrite_unicode(out, vis)
    return out


def build_road_verify(res, sample: str, out_dir: str) -> str:
    res = _normalize(res)
    """把 OSM 的用户实骑道路投影回截图，与识别轨迹叠放。

    这是「路名归属是否可信」的直接证据：轨迹应精确骑在这些道路的中心线上。
    """
    img, reg = _registration(res, sample)
    from app.roadfit import fetch_roads
    gcj_all = res["geojson_gcj02"]["features"][0]["geometry"]["coordinates"]
    lons = [c[0] for c in gcj_all]
    lats = [c[1] for c in gcj_all]
    bbox = (min(lons) - 0.008, min(lats) - 0.008,
            max(lons) + 0.008, max(lats) + 0.008)
    try:
        raw_roads = fetch_roads(bbox, timeout=90)
    except Exception as exc:
        print("路网获取失败：%s" % exc)
        return ""
    s = reg.scale
    tx, ty = float(reg.translation[0]), float(reg.translation[1])

    def to_px(merc):
        return np.column_stack([(merc[:, 0] - tx) / s, -(merc[:, 1] - ty) / s])

    per: dict[str, list] = {}
    for nm, pts in raw_roads:
        if nm not in FIVE_ROADS:
            continue
        arr = []
        for lo, la in pts:
            glo, gla = wgs84_to_gcj02(lo, la)
            arr.append(lonlat_to_mercator(glo, gla))
        if arr:
            per.setdefault(nm, []).extend(arr)

    vis = img.copy()
    for nm, col in FIVE_ROADS.items():
        if nm not in per:
            continue
        q = to_px(np.array(per[nm])).astype(int)
        for i in range(len(q) - 1):
            if abs(q[i + 1][0] - q[i][0]) < 400 and abs(q[i + 1][1] - q[i][1]) < 400:
                cv2.line(vis, tuple(q[i]), tuple(q[i + 1]), col, 4, cv2.LINE_AA)

    gcj = res["geojson_gcj02"]["features"][0]["geometry"]["coordinates"]
    rec = np.array([
        reg.mercator_to_pixel(*lonlat_to_mercator(lo, la))
        for lo, la in gcj
    ])
    # 轨迹用「白 + 黑边」：路网配色里已经有黄色（剑河路），若轨迹也用黄，
    # 用户无法分辨哪条是识别结果、哪条是道路。
    for i in range(len(rec) - 1):
        cv2.line(vis, tuple(rec[i].astype(int)), tuple(rec[i + 1].astype(int)),
                 (0, 0, 0), 11, cv2.LINE_AA)
    for i in range(len(rec) - 1):
        cv2.line(vis, tuple(rec[i].astype(int)), tuple(rec[i + 1].astype(int)),
                 (255, 255, 255), 7, cv2.LINE_AA)
    cv2.circle(vis, tuple(rec[0].astype(int)), 26, (0, 200, 0), -1)
    cv2.circle(vis, tuple(rec[0].astype(int)), 27, (255, 255, 255), 3)
    cv2.circle(vis, tuple(rec[-1].astype(int)), 26, (0, 0, 235), -1)
    cv2.circle(vis, tuple(rec[-1].astype(int)), 27, (255, 255, 255), 3)

    legend = [(nm, FIVE_ROADS[nm]) for nm in FIVE_ROADS if nm in per]
    rows = [("识别轨迹", (255, 255, 255), True)] + \
           [(nm, col, False) for nm, col in legend]
    _bh = 56 + 24 * len(rows)
    cv2.rectangle(vis, (26, 26), (320, _bh), (255, 255, 255), -1)
    cv2.rectangle(vis, (26, 26), (320, _bh), (70, 70, 70), 2)
    cv2.putText(vis, "OSM roads (thin) vs recognized track (white)",
                (42, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (25, 25, 25), 2, cv2.LINE_AA)
    for i, (nm, col, is_track) in enumerate(rows):
        y = 84 + i * 24
        if is_track:
            cv2.line(vis, (44, y - 5), (84, y - 5), (0, 0, 0), 9, cv2.LINE_AA)
            cv2.line(vis, (44, y - 5), (84, y - 5), (255, 255, 255), 5, cv2.LINE_AA)
        else:
            cv2.line(vis, (44, y - 5), (84, y - 5), col, 4, cv2.LINE_AA)
        cv2.putText(vis, nm, (94, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (25, 25, 25), 1, cv2.LINE_AA)

    out = os.path.join(out_dir, _tag(sample) + "road_verify.png")
    imwrite_unicode(out, vis)
    return out


def build_chart(res, sample: str, out_dir: str) -> str:
    res = _normalize(res)
    """独立轨迹图：米制等比例，标注路名与公里刻度。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyArrowPatch

    gcj = res["geojson_gcj02"]["features"][0]["geometry"]["coordinates"]
    lat0 = math.radians(np.mean([c[1] for c in gcj]))
    kx = 111320.0 * math.cos(lat0)
    ky = 110574.0
    x = np.array([c[0] * kx for c in gcj])
    y = np.array([c[1] * ky for c in gcj])
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    LINE = "#0f6e56"
    LINE2 = "#d54941"
    AMBER = "#9a6b1e"

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    cn = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")

    fig, ax = plt.subplots(figsize=(13.2, 8.4), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfaf7")

    ax.plot(x, y, color=LINE, lw=4.2, solid_capstyle="round", zorder=5)
    ax.plot([x[0]], [y[0]], "o", ms=15, color="#0f6e56", mec="white", mew=2.2, zorder=8)
    ax.plot([x[-1]], [y[-1]], "o", ms=15, color=LINE2, mec="white", mew=2.2, zorder=8)
    ax.annotate("起", (x[0], y[0]), textcoords="offset points", xytext=(34, -4),
                ha="center", fontsize=15, fontweight="bold", color=LINE, fontproperties=cn,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=LINE, lw=1.4))
    ax.annotate("终", (x[-1], y[-1]), textcoords="offset points", xytext=(-30, 16),
                ha="center", fontsize=15, fontweight="bold", color=LINE2, fontproperties=cn,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=LINE2, lw=1.4))

    # 公里刻度
    for d in range(1000, int(total), 1000):
        i = int(np.searchsorted(cum, d))
        ax.plot([x[i]], [y[i]], "o", ms=9, mfc="white", mec=LINE, mew=2.2, zorder=7)
        ax.annotate(f"{d // 1000} km", (x[i], y[i]), textcoords="offset points",
                    xytext=(14, 30), fontsize=11.5, color=LINE,
                    fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))

    # 方向箭头（只取直线段）
    straight = [i for i in range(len(seg))
                if seg[i] > 45 and (i == 0 or abs(math.degrees(math.atan2(
                    seg[i], max(x[i + 1] - x[i], 1e-9))) - math.degrees(math.atan2(
                    seg[i - 1], max(x[i] - x[i - 1], 1e-9)))) < 22)]
    if len(straight) > 9:
        straight = [straight[i] for i in np.linspace(0, len(straight) - 1, 9).astype(int)]
    for i in straight:
        ax.add_patch(FancyArrowPatch((x[i], y[i]), (x[i + 1], y[i + 1]),
                                      arrowstyle="-|>", mutation_scale=24,
                                      color=LINE, linewidth=0, zorder=6,
                                      shrinkA=0, shrinkB=0))

    # 路名标注：每条路只标一次（取该路最长的一段）
    segs = (res.get("road_attribution") or {}).get("segments") or []
    runs: list[list] = []
    for sg in segs:
        if runs and runs[-1][0] == sg["road"]:
            runs[-1][1] = sg["end_index"]
            runs[-1][2] += sg["length_m"]
        else:
            runs.append([sg["road"], sg["end_index"], sg["length_m"]])
    best: dict[str, tuple] = {}
    for nm, i1, ln in runs:
        if nm not in best or ln > best[nm][1]:
            best[nm] = (i1, ln)
    major = [(nm, i1, ln) for nm, (i1, ln) in best.items() if ln >= 60]
    major.sort(key=lambda m: -m[2])

    # 先固定坐标范围，才能用显示坐标做碰撞检测
    ax.set_aspect("equal")
    ax.autoscale_view()
    fig.canvas.draw()

    def disp(px, py):
        t = ax.transData.transform((px, py))
        return float(t[0]), float(t[1])

    # 公里刻度位置（显示坐标），先占位
    # annotate 的 offset 单位是 points，1 point = dpi/72 显示像素
    PT = fig.dpi / 72.0

    def _scale():
        return PT, PT

    km_pos: list[tuple[float, float]] = []
    for d in range(1000, int(total), 1000):
        i = int(np.searchsorted(cum, d))
        kx_, ky_ = disp(x[i], y[i])
        sx, sy = _scale()
        km_pos.append((kx_ + 14 * sx, ky_ - 30 * sy))

    placed: list[tuple[float, float]] = []
    cands = [(0, 19), (0, -29), (52, 0), (-52, 0), (0, 45), (0, -55),
             (52, 26), (-52, -26), (52, -26), (-52, 26)]
    for nm, i1, ln in major:
        i = int(np.clip(i1, 0, len(x) - 1))
        ax_, ay_ = disp(x[i], y[i])
        chosen = cands[0]
        sx, sy = _scale()
        for cand in cands:
            tx_, ty_ = ax_ + cand[0] * sx, ay_ - cand[1] * sy
            ok = all(math.hypot(tx_ - qx, ty_ - qy) > 96 for qx, qy in placed + km_pos)
            if ok:
                chosen = cand
                break
        dxp, dyp = chosen
        ax.annotate(nm, (x[i], y[i]), textcoords="offset points", xytext=(dxp, dyp),
                    fontsize=13, color=AMBER, fontproperties=cn, fontweight="bold",
                    zorder=9, bbox=dict(boxstyle="round,pad=0.25", fc="#fffaf0",
                                        ec="#e8d9b8", alpha=0.96))
        placed.append((ax_ + dxp * sx, ay_ - dyp * sy))

    ax.set_aspect("equal")
    ax.set_axis_off()

    rf = res.get("road_fit") or {}
    dg = res["diagnostics"]
    roads = res.get("road_attribution", {}).get("roads", [])
    head = (f"{res['track_points']} pts   {total / 1000:.3f} km   "
            f"conf {res['confidence']:.3f}   scale {dg['scale_ground_m_per_px']:.3f} m/px")
    sub = ("road-fit  E%+.0f N%+.0f m   road-dist %.0f→%.0f m   |   %s"
           % (rf.get("shift_e_m", 0), rf.get("shift_n_m", 0),
              rf.get("median_before_m", 0), rf.get("median_after_m", 0),
              " → ".join(roads[:7])))
    fig.text(0.012, 0.975, "美团骑行截图 → GeoJSON 轨迹识别", fontsize=19,
             fontweight="bold", color="#14261f", fontproperties=cn)
    fig.text(0.012, 0.938, head, fontsize=12, color="#3d4f48", fontproperties=cn)
    fig.text(0.012, 0.905, sub, fontsize=11, color="#6b7a74", fontproperties=cn)

    # 比例尺
    bx, by = x.min() + 0.05 * (x.max() - x.min()), y.min() + 0.10 * (y.max() - y.min())
    ax.plot([bx, bx + 1000], [by, by], color="#14261f", lw=3, zorder=9)
    ax.plot([bx, bx], [by - 60, by + 60], color="#14261f", lw=2, zorder=9)
    ax.plot([bx + 1000, bx + 1000], [by - 60, by + 60], color="#14261f", lw=2, zorder=9)
    ax.annotate("1 km", (bx + 500, by), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=11.5, color="#14261f", fontweight="bold",
                fontproperties=cn, zorder=9)

    fig.legend(handles=[
        plt.Line2D([], [], color=LINE, lw=4.2, label="识别轨迹"),
        plt.Line2D([], [], color=LINE, lw=0, marker="o", ms=12,
                   mfc=LINE, mec="white", label="起点"),
        plt.Line2D([], [], color=LINE2, lw=0, marker="o", ms=12,
                   mfc=LINE2, mec="white", label="终点"),
        plt.Line2D([], [], color=AMBER, lw=0, marker="s", ms=10,
                   mfc="#fffaf0", mec="#e8d9b8", label="经过的道路"),
    ], loc="upper right", bbox_to_anchor=(0.995, 0.985), frameon=False,
        prop=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=11.5))

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(out_dir, _tag(sample) + "trajectory_chart.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out
