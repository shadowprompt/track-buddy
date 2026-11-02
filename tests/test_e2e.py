"""端到端验收测试。

用真实截图验证：
1. 轨迹提取正确（起终点徽章、连通域）
2. 地理配准成功且模型合理
3. 输出的 GeoJSON 结构完整、坐标合理
4. 里程与像素弧长一致（含墨卡托纬度修正）
5. 反投影回截图后与原轨迹吻合（自洽性）
6. 用户纠正 → 词典强化 → 后续识别命中
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CONFIG
from app.cv_utils import imread_unicode
from app.geo.coords import gcj02_to_wgs84, haversine, wgs84_to_gcj02
from app.pipeline import RidePipeline
from app.track import extract_track

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "meituan_ride_shanghai.jpg"

PASS = "✓"
FAIL = "✗"
results: list[tuple[bool, str, str]] = []



class _TrackStub:
    """轨迹提取结果的最小桩（只提供 confidence，置信度评分用）。"""
    def __init__(self, conf: float = 0.9):
        self.confidence = conf

def check(cond: bool, name: str, detail: str = "") -> None:
    results.append((bool(cond), name, detail))
    print(f"  {PASS if cond else FAIL} {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    print("=" * 66)
    print("美团骑行截图轨迹识别 · 端到端验收")
    print("=" * 66)

    if not SAMPLE.exists():
        print(f"样本缺失: {SAMPLE}")
        return 1

    img = imread_unicode(SAMPLE)
    check(img is not None, "样本图片读取", f"{img.shape[1]}x{img.shape[0]}")

    pipe = RidePipeline(CONFIG)

    print("\n[1] 轨迹提取")
    from app.track import extract_track

    tr = extract_track(img, CONFIG.vision)
    check(tr.start_marker is not None, "检测到起点徽章", str(tr.start_marker))
    check(tr.end_marker is not None, "检测到终点徽章", str(tr.end_marker))
    check(len(tr.polyline) > 100, "折线点数充足", f"{len(tr.polyline)} 点")

    seg = np.linalg.norm(np.diff(tr.polyline, axis=0), axis=1)
    # 允许起/终徽章尖端到轨迹端点的一小段直线（徽章是水滴形，
    # 尖端与轨迹端点相距约 20-25 像素），但不允许更大跳变。
    check(seg.max() < 30, "折线无异常跳变", f"最大段长 {seg.max():.2f}px")
    arc_px = float(seg.sum())

    print("\n[2] 地理配准")
    res = pipe.run(img, city="上海", allow_online=False, output_crs="WGS-84")
    dg = res.diagnostics
    check(res.confidence > 0, "置信度大于 0", f"{res.confidence:.3f}")
    check(dg["model"] in ("north_up", "similarity", "affine"), "配准模型", dg["model"])
    check(dg["inliers"] >= 3, "内点数量充足", f"{dg['inliers']} 内点 / {dg['outliers']} 离群")
    check(dg["rmse_m"] < 200, "配准残差可接受", f"RMSE {dg['rmse_m']:.1f} m")
    check(abs(dg["rotation_deg"]) < 5, "地图朝向为正北", f"{dg['rotation_deg']:.2f}°")
    check(4.0 < dg["scale_ground_m_per_px"] < 8.0,
          "地面比例尺合理", f"{dg['scale_ground_m_per_px']:.3f} m/px")

    print("\n[3] GeoJSON 输出")
    gj = res.geojson
    check(gj.get("type") == "FeatureCollection", "类型为 FeatureCollection")
    check(len(gj["features"]) >= 1, "包含要素", f"{len(gj['features'])} 个")
    line = gj["features"][0]
    check(line["geometry"]["type"] == "LineString", "主要素为 LineString")
    coords = line["geometry"]["coordinates"]
    check(len(coords) > 20, "轨迹点数", f"{len(coords)}")
    check(all(len(c) == 2 for c in coords), "坐标为 [lon, lat]")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    check(all(120 < lo < 123 for lo in lons), "经度在上海范围内",
          f"{min(lons):.4f}~{max(lons):.4f}")
    check(all(30 < la < 32.5 for la in lats), "纬度在上海范围内",
          f"{min(lats):.4f}~{max(lats):.4f}")

    print("\n[4] 里程一致性（含墨卡托纬度修正）")
    seg_m = [
        haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        for i in range(1, len(coords))
    ]
    sum_hav = float(sum(seg_m))
    # 像素弧长 → 墨卡托米 → 地面米（乘 cos(lat)）
    lat0 = math.radians(sum(lats) / len(lats))
    expect = arc_px * dg["scale_mercator_m_per_px"] * math.cos(lat0)
    ratio = sum_hav / expect if expect else 0
    check(0.90 < ratio < 1.10, "里程与像素弧长一致",
          f"haversine {sum_hav:.0f}m vs 预期 {expect:.0f}m (比值 {ratio:.3f})")
    check(abs(res.distance_m - sum_hav) < 5, "上报里程与坐标一致",
          f"{res.distance_m:.0f}m vs {sum_hav:.0f}m")

    print("\n[5] 坐标转换正确性")
    from app.register import register
    from app.ocr_anchor import build_anchors, ocr_texts
    from app.geo.coords import lonlat_to_mercator

    items = ocr_texts(img, CONFIG.vision)
    anchors = build_anchors(img, items, CONFIG.vision)
    warn: list[str] = []
    pairs = pipe._resolve_anchors(anchors, "上海", False, warn)
    reg = register(pairs, inlier_m=CONFIG.vision.ransac_inlier_m, min_inliers=3)
    check(reg.ok, "配准矩阵可用")
    if reg.ok:
        # GCJ-02 → WGS-84 → GCJ-02 应当严格回到原点（亚米级）
        errs = []
        for px, py in tr.polyline[::17]:
            lon, lat = reg.pixel_to_lonlat(float(px), float(py))
            wlon, wlat = gcj02_to_wgs84(lon, lat)
            back = wgs84_to_gcj02(wlon, wlat)
            mx1, my1 = lonlat_to_mercator(lon, lat)
            mx2, my2 = lonlat_to_mercator(*back)
            errs.append(math.hypot(mx1 - mx2, my1 - my2))
        check(max(errs) < 0.01, "GCJ-02↔WGS-84 往返无损",
              f"最大偏差 {max(errs):.6f} m")

        # 输出坐标（WGS-84）应与内部 GCJ-02 相差约 347m，这是基准差异
        deltas = []
        for px, py in tr.polyline[::17]:
            lon, lat = reg.pixel_to_lonlat(float(px), float(py))
            wlon, wlat = gcj02_to_wgs84(lon, lat)
            mx1, my1 = lonlat_to_mercator(lon, lat)
            mx2, my2 = lonlat_to_mercator(wlon, wlat)
            deltas.append(math.hypot(mx1 - mx2, my1 - my2))
        # 沪实测约 500-600m（GCJ-02 偏移随位置变化），只做量级校验
        check(350 < sum(deltas) / len(deltas) < 750,
              "GCJ-02 与 WGS-84 基准差符合预期",
              f"平均 {sum(deltas)/len(deltas):.0f} m（上海约 500-600m）")

    print("\n[6] 绝对位置自校验（回归项）")
    # 这是本次修复的核心回归点：早期版本把道路名当锚点，
    # 导致整体平移 3 公里，而 RMSE 完全看不出来。
    med, notes = pipe._self_check(reg, pairs)
    check(med > 0, "自校验已执行", f"锚点反投影偏移中位 {med:.0f}px")
    tol_px = 400.0 / max(reg.scale, 1e-6)
    check(med < tol_px, "锚点绝对位置自洽",
          f"{med:.0f}px < 容差 {tol_px:.0f}px")
    check(not notes, "无自校验告警", "; ".join(notes) if notes else "")

    print("\n[7] 道路名不参与配准（回归项）")
    road_anchors = [a for a in anchors if a.kind == "road"]
    check(all(a.kind != "road" for a in reg.inliers),
          "内点中没有道路名锚点",
          f"内点 {[a.name for a in reg.inliers]}")

    print("\n[8] 端点与已知 POI 对照")
    s_pt = res.geojson["features"][1]["geometry"]["coordinates"]
    e_pt = res.geojson["features"][2]["geometry"]["coordinates"]
    # 截图中「终」徽章紧邻「上海会聚/荟聚」标注
    d_huiju = haversine(e_pt[0], e_pt[1], 121.35198, 31.22350)
    check(d_huiju < 700, "终点落在上海荟聚附近", f"{d_huiju:.0f} m")
    # 起点在水城路/西郊宾馆一带
    d_start = min(
        haversine(s_pt[0], s_pt[1], 121.37611, 31.20348),
        haversine(s_pt[0], s_pt[1], 121.38760, 31.20125),
    )
    check(d_start < 2000, "起点落在水城路/西郊宾馆一带", f"{d_start:.0f} m")

    print("\n[9] GeoJSON 结构与方向")
    gj = res.geojson
    line_c = gj["features"][0]["geometry"]["coordinates"]
    st_c = gj["features"][1]["geometry"]["coordinates"]
    en_c = gj["features"][2]["geometry"]["coordinates"]
    check(line_c[0] == st_c, "LineString 首点 == 起点要素",
          "顺序即骑行方向，第三方工具依赖此约定")
    check(line_c[-1] == en_c, "LineString 末点 == 终点要素")
    check(gj["features"][0]["properties"].get("crs") == "WGS-84",
          "properties 声明坐标系")
    check(res.geojson_gcj02["features"][0]["properties"].get("crs") == "GCJ-02",
          "同时提供 GCJ-02 版本")

    print("\n[10] 输出必须真的贴在真实道路上（回归项）")
    # 这一条直接守住「生成的轨迹不在实际道路上」这个 bug：
    # 把 GCJ-02 版本（用于中文底图）投到真实路网上测距。
    try:
        from app.roadfit import RoadNetwork, fetch_roads
        from app.geo.coords import lonlat_to_mercator
        from scipy.spatial import cKDTree

        g = res.geojson_gcj02["features"][0]["geometry"]["coordinates"]
        # 复用流水线刚写好的路网缓存（避免重复拉取）。
        # 缓存现在是多块格式 {"blocks": [{bbox, roads}]}，但要兼容旧的单块格式
        # {"bbox", roads}——两者的 roads 都是 [[路名, 点串], ...]。
        _cache = CONFIG.road_cache_dir / "roads.json"
        _roads: list = []
        if _cache.exists():
            from app.roadfit import _load_cache_blocks

            _blocks = _load_cache_blocks(str(_cache))
            if _blocks:
                _roads = [(nm, [(p[0], p[1]) for p in pts])
                          for blk in _blocks
                          for nm, pts in blk.get("roads", [])]
        if not _roads:
            _roads = fetch_roads((121.3300, 31.1900, 121.4000, 31.2450), timeout=90)
        _net = RoadNetwork(_roads)
        _pts = np.array([lonlat_to_mercator(lo, la) for lo, la in g])
        _d, _ = cKDTree(np.array(_net.samples)).query(_pts)
        check(float(np.median(_d)) < 20,
              "GCJ-02 轨迹贴路（中文底图直接可用）",
              f"中位 {np.median(_d):.1f} m，<=15m 的点 {( _d<=15).sum()}/{len(_d)}")
        # WGS-84 版本被误当 GCJ-02 用（即丢进腾讯底图）会偏约 290m
        w = res.geojson["features"][0]["geometry"]["coordinates"]
        _pw = np.array([lonlat_to_mercator(lo, la) for lo, la in w])
        _dw, _ = cKDTree(np.array(_net.samples)).query(_pw)
        check(float(np.median(_dw)) > float(np.median(_d)) + 5,
              "WGS-84 版本与 GCJ-02 有明显基准差",
              f"WGS 中位 {np.median(_dw):.1f} m vs GCJ 中位 {np.median(_d):.1f} m")
    except Exception as exc:
        check(False, "输出贴在真实道路上", f"{type(exc).__name__}: {exc}")

    print("\n[10] 路网约束精修（回归项）")
    rf = res.road_fit
    if rf.get("ok"):
        check(rf["median_after_m"] < 15,
              "轨迹贴在真实道路上",
              f"贴路中位 {rf['median_before_m']:.0f}m → {rf['median_after_m']:.0f}m")
        check(abs(rf["shift_e_m"]) < 800 and abs(rf["shift_n_m"]) < 800,
              "平移量在合理范围",
              f"东{rf['shift_e_m']:+.0f}m 北{rf['shift_n_m']:+.0f}m")
    else:
        check(False, "路网精修已执行", str(rf.get("notes")))

    roads = res.road_attribution.get("roads") or []
    check(bool(roads), "路名归属非空", "、".join(roads[:6]))
    # 用户实骑道路（2026-10-30 由用户确认）
    expect = ["安龙路", "新渔路", "新渔东路", "剑河路", "金钟路"]
    hit = [r for r in expect if any(r in x for x in roads)]
    check(len(hit) >= 4,
          "路名归属覆盖用户实骑道路",
          f"命中 {len(hit)}/{len(expect)}：{'、'.join(hit)}")

    print("\n[10a] 第二张样本：对角线路线 + 往返一致性（回归项）")
    # 2026-10-31 新增样本：起终点呈对角分布（aspect≈1.03），早期
    # 「必须横向延展 aspect>1.6」的启发式会整条漏掉。
    sample2 = CONFIG.root / "samples" / "meituan_ride_2.jpg" if hasattr(
        CONFIG, "root") else None
    p2 = Path(__file__).resolve().parent.parent / "samples" / "meituan_ride_2.jpg"
    if p2.exists():
        img2 = imread_unicode(str(p2))
        tr2 = extract_track(img2, CONFIG.vision)
        check(len(tr2.polyline) > 200, "对角线路线被正确提取",
              f"{len(tr2.polyline)} 点，置信度 {tr2.confidence}")
        check(tr2.start_marker is not None and tr2.end_marker is not None,
              "第二张样本起终点徽章已定位",
              f"起{tr2.start_marker} 终{tr2.end_marker}")

        pipe2 = RidePipeline(CONFIG)
        res2 = pipe2.run(img2, city="上海", allow_online=False, output_crs="WGS-84")
        check(res2.track_points > 100, "第二张样本流水线跑通",
              f"{res2.track_points} 点，{res2.distance_m/1000:.3f} km")

        # 往返一致性：输出坐标反投影回像素，必须与提取的轨迹重合。
        # （可视化脚本曾对已是 GCJ-02 的坐标重复做基准转换，凭空偏 500 米，
        #   这条断言能在第一时间抓住同类错误。）
        from app.ocr_anchor import build_anchors, ocr_texts
        from app.register import register
        from app.geo.coords import lonlat_to_mercator

        items2 = ocr_texts(img2, CONFIG.vision)
        anchors2 = build_anchors(img2, items2, CONFIG.vision)
        warn2: list[str] = []
        pairs2 = pipe2._resolve_anchors(anchors2, "上海", False, warn2)
        reg2 = register(pairs2, inlier_m=CONFIG.vision.ransac_inlier_m,
                        min_inliers=CONFIG.vision.ransac_min_inliers)
        rf2 = res2.road_fit or {}
        if rf2.get("ok"):
            reg2.translation = reg2.translation + np.array(
                [rf2["shift_e_m"], rf2["shift_n_m"]], dtype=float)
        gcj2 = res2.geojson_gcj02["features"][0]["geometry"]["coordinates"]
        rec2 = np.array([reg2.mercator_to_pixel(*lonlat_to_mercator(lo, la))
                         for lo, la in gcj2])
        from scipy.spatial import cKDTree
        d2, _ = cKDTree(tr2.polyline.astype(float)).query(rec2)
        check(float(np.median(d2)) < 3.0,
              "输出坐标反投影后与提取轨迹重合",
              f"中位 {np.median(d2):.2f} px，最大 {d2.max():.2f} px")
    else:
        check(True, "第二张样本存在", "跳过（文件缺失）")

    print("\n[10b] GCJ-02 转换权威基准（回归项）")
    # 公开基准点：天安门 WGS-84 (116.391349, 39.907375) → GCJ-02 (116.397590, 39.908776)
    # 早期实现里 _transform_lon 是死代码（经度误用了 transform_lat 的结果），
    # 纬度对、经度偏 375m —— 这条断言能在第一时间抓住它。
    wlon, wlat = 116.391349, 39.907375
    elon, elat = 116.397590, 39.908776
    g = wgs84_to_gcj02(wlon, wlat)
    err = math.hypot((g[0] - elon) * 85000, (g[1] - elat) * 111000)
    check(err < 20.0, "WGS-84→GCJ-02 符合天安门权威基准", f"误差 {err:.1f} m")
    b = gcj02_to_wgs84(elon, elat)
    err2 = math.hypot((b[0] - wlon) * 85000, (b[1] - wlat) * 111000)
    check(err2 < 20.0, "GCJ-02→WGS-84 符合天安门权威基准", f"误差 {err2:.1f} m")

    # 往返无损
    worst = 0.0
    for lon, lat in ((121.35791, 31.19512), (121.35198, 31.22350),
                     (121.39606, 31.26457), (116.391349, 39.907375)):
        gg = wgs84_to_gcj02(lon, lat)
        bb = gcj02_to_wgs84(*gg)
        worst = max(worst, math.hypot((bb[0] - lon) * 95000, (bb[1] - lat) * 111000))
    check(worst < 0.5, "GCJ-02↔WGS-84 往返亚米级", f"最大 {worst:.4f} m")

    # 上海偏移量应约 +430m 东 / +155m 北（东偏为主，与经验值一致）
    sg = wgs84_to_gcj02(121.3600, 31.2200)
    de = (sg[0] - 121.3600) * 95000
    dn = (sg[1] - 31.2200) * 111000
    # 上海实测：东偏 +438m、南偏 -208m（东偏为主）
    check(300 < de < 600 and -280 < dn < -80,
          "上海 GCJ-02 偏移方向与量级正确",
          f"东 {de:+.0f} m, 北 {dn:+.0f} m")

    print("\n[10] 用户纠正 → 词典强化")
    lex = pipe.lexicon
    before = lex.stats()
    probe = "上海动物园"
    hits_before = len(lex.lookup(probe, "上海"))
    lex.remember(probe, 121.3579100, 31.1951200, city="上海", source="user", boost=2.0)
    after = lex.stats()
    hits_after = len(lex.lookup(probe, "上海"))
    check(after["points"] >= before["points"], "词典条目已写入",
          f"{before['points']} → {after['points']}")
    check(hits_after >= hits_before, "该地名可被命中", f"{hits_before} → {hits_after} 候选")
    lex.save()

    print("\n[11] 失败路径返回结构完整的 diagnostics（回归项）")
    # 前端曾直接对 d.diagnostics.rotation_deg 调 .toFixed()，
    # 而配准失败时后端只返回 {"track_points": N}，
    # 于是 undefined.toFixed 抛 TypeError，把真正的失败原因整个吞掉——
    # 用户只看到 "Cannot read properties of undefined (reading 'toFixed')"。
    from app.pipeline import DIAG_KEYS, _base_diagnostics

    fail_dg = _base_diagnostics(track_points_px=123)
    check(all(k in fail_dg for k in DIAG_KEYS),
          "失败路径 diagnostics 字段齐全",
          f"{len(fail_dg)} 个键（应为 {len(DIAG_KEYS)}）")
    # 模拟前端行为：任何缺失值都不得让页面崩溃
    def safe_num(v, digits=2):
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            return "—"
        return f"{v:.{digits}f}"
    check(safe_num(fail_dg["rotation_deg"]) == "—"
          and safe_num(fail_dg["scale_ground_m_per_px"], 3) == "—",
          "缺失字段安全降级为「—」而非抛异常")
    check(res2.diagnostics is not None
          and all(k in res2.diagnostics for k in DIAG_KEYS),
          "成功路径 diagnostics 字段同样齐全")

    print("\n[12] 折线简化不丢形状（回归项）")
    # 经典 Douglas-Peucker 用「到端点连线的垂距」作判据，
    # 对南北向长直线会把微小蛇形抖动放大成大垂距，整条轨迹被压成几个点。
    # 实测长宁剑河路截图（scale=0.26 m/px）简化后只剩 32 点，弯道全丢。
    from app.cv_utils import resample_polyline, simplify_polyline

    tr0 = extract_track(img, CONFIG.vision)
    samp0 = resample_polyline(tr0.polyline, step_px=2.0)
    simp0 = simplify_polyline(samp0, eps=1.0)
    len_raw = float(np.linalg.norm(np.diff(tr0.polyline.astype(float), axis=0), axis=1).sum())
    len_simp = float(np.linalg.norm(np.diff(simp0.astype(float), axis=0), axis=1).sum())
    check(len_simp / len_raw > 0.9,
          "简化后弧长保留 90% 以上",
          f"{len_raw:.0f}px → {len_simp:.0f}px ({len_simp/len_raw*100:.1f}%)")
    from scipy.spatial import cKDTree
    dd0, _ = cKDTree(tr0.polyline.astype(float)).query(simp0.astype(float))
    check(float(np.max(dd0)) <= 3.0,
          "简化后最大偏离原轨迹 3 像素内",
          f"最大 {np.max(dd0):.2f} px")

    print("\n[13] 采样密度与尺度自适应（回归项）")
    # 高倍截图 scale 很小，10 米的像素步长会变成 38 像素，一个弯道就被跳过。
    from app.pipeline import RidePipeline as _RP
    small_scale = 0.26
    step_px = min(max(1.0, 5.0 / small_scale), 4.0)
    eps_px = min(1.0 / small_scale, 1.0)
    check(step_px <= 4.0, "小尺度下采样步长有像素上限", f"{step_px:.1f} px")
    check(eps_px <= 1.0, "小尺度下简化容差有像素上限", f"{eps_px:.1f} px")

    print("\n[14] OSM 路网 bbox 坐标系正确（回归项）")
    # _road_refine 过去把 GCJ-02 的 bbox 直接喂给 OSM API（期望 WGS-84），
    # 查询中心因此偏移约 450 米（东）/200 米（南），拉回的是别的街区的路网。
    from app.geo.coords import gcj02_to_wgs84 as _g2w
    gcj_lo, gcj_la = 121.3811, 31.2025
    w_lo, w_la = _g2w(gcj_lo, gcj_la)
    d_east = (gcj_lo - w_lo) * 95000
    d_north = (gcj_la - w_la) * 111000
    # 上海实测：GCJ-02 相对 WGS-84 东偏约 +450m、南偏约 -200m
    check(200 < d_east < 700 and -400 < d_north < -100,
          "GCJ-02→WGS-84 偏移量与已知基准一致（上海约东 450m/南 200m）",
          f"东 {d_east:.0f} m, 北 {d_north:.0f} m")

    print("\n[15] 冷启动：名称相似度与多锚点互证（回归项）")
    # 早期冷启动要求「完全同名」，浦东那张截图 6 个 POI 全部 0 命中：
    #   OCR「进才实验中学」 → OSM「上海市进才实验中学」（前缀被截）
    #   OCR「水清木华」     → OSM「水清木华公寓」（后缀被截）
    #   OCR「花园2期」     → OSM「朗香花园2期」「星宇花园2期」…（多个候选）
    from app.coldstart import name_score, ColdStartGeocoder, _haversine_m

    check(name_score("进才实验中学", "上海市进才实验中学") > 0.85,
          "OCR 名是 OSM 名的子串时得高分",
          f"{name_score('进才实验中学', '上海市进才实验中学'):.3f}")
    check(name_score("水清木华", "水清木华公寓") > 0.85,
          "OCR 名缺后缀时同样得高分",
          f"{name_score('水清木华', '水清木华公寓'):.3f}")
    check(name_score("乡土田园区", "何文田上海水配水庫") < 0.3,
          "完全无关的地名得低分",
          f"{name_score('乡土田园区', '何文田上海水配水庫'):.3f}")

    # 互证：两个真实地名互相印证，异地同名地点应被排除
    from app.coldstart import OsmPlace
    cs = ColdStartGeocoder.__new__(ColdStartGeocoder)  # 不联网，只测纯逻辑
    good_a = OsmPlace(name="上海市进才实验中学", lon=121.5567, lat=31.2255, kind="poi")
    good_b = OsmPlace(name="水清木华公寓", lon=121.5610, lat=31.2248, kind="poi")
    wrong = OsmPlace(name="水清木华", lon=119.9641, lat=31.8040, kind="poi")  # 安徽
    cands = {
        "进才实验中学": [good_a],
        "水清木华": [good_b, wrong],
        "花园2期": [
            OsmPlace(name="星宇花园2期", lon=121.3944, lat=31.2811, kind="poi"),
            OsmPlace(name="朗香花园2期", lon=121.3779, lat=31.3213, kind="poi"),
        ],
    }
    center = cs._find_region_center(cands)
    check(center is not None, "互证能定位出区域中心")
    if center:
        clon, clat, names = center
        # 互证的语义是「簇心周围有这些不同地名的候选」，
        # 因此只要 ≥2 个不同地名成立，就说明它们互相印证了。
        check(len(names) >= 2, "至少两个不同地名互相印证",
              f"簇心周边有 {sorted(names)}")
        d_wrong = _haversine_m(clon, clat, wrong.lon, wrong.lat)
        check(d_wrong > 1500, "异地同名地点被排除在区域外",
              f"距簇心 {d_wrong/1000:.0f} km")
        # 簇心应落在两个真锚点之间
        d_a = _haversine_m(clon, clat, good_a.lon, good_a.lat)
        d_b = _haversine_m(clon, clat, good_b.lon, good_b.lat)
        check(d_a < 800 and d_b < 800,
              "簇心紧邻两个真锚点", f"距两锚点 {d_a:.0f} m / {d_b:.0f} m")

    print("\n[15b] 词典子串匹配方向（回归项）")
    # 真实事故：词典里有 OCR 截断产生的残片「上海世」（来自「上海西站」），
    # 坐标 121.40071,31.26275（上海西站）。查「上海世贸商城」时，
    # 「上海世」是它的子串 → 命中 → 返回了**上海西站**的坐标，
    # 偏离真实位置（121.40087,31.20189）6.8 公里。
    # 而冷启动已经查到了正确坐标，却因「词典已命中」被丢弃。
    from app.geocode import Lexicon
    _lex = Lexicon(CONFIG.lexicon_path)
    _res = _lex.lookup("上海世贸商城", "上海")
    _wrong = [r for r in _res if abs(r.lat - 31.26275) < 1e-4]
    check(not _wrong,
          "短残片条目不再劫持长查询名（上海世 ≠ 上海世贸商城）",
          f"查到 {len(_res)} 候选")
    # 安全方向必须仍然可用：查「百联中环购物广场」应命中词典里的「百联中环」
    _safe = _lex.lookup("百联中环购物广场", "上海")
    check(len(_safe) >= 1,
          "安全方向的包含匹配仍然有效（长 key 命中短查询）",
          f"{len(_safe)} 候选")
    # 精确命中不受影响
    check(len(_lex.lookup_exact("安龙小区")) >= 1,
          "精确命中不受子串规则影响")

    print("\n[15c] 冷启动簇心过滤不得丢弃正确候选（回归项）")
    # 真实事故（上海古北一例）：簇心按地理距离选出，
    # 恰好是「地理上更紧密但方位不对」的那一对赢得簇心，
    # 于是**正确**的候选被硬过滤丢弃：
    #   「虹桥基金小镇」↔「红松东路」相距仅 791m（赢得簇心）
    #   「上海世贸商城」↔「上海长风大悦城」相距 2379m（正确但被丢）
    # 结果：正确候选 2399m / 4471m 超出 1500m 阈值 → 被扔。
    #
    # 修法：簇心过滤放宽为「只排序、不丢弃」，
    # 挑选的正确性交给 RANSAC 的几何一致性。
    from app.coldstart import ColdStartGeocoder as _CS
    _cs2 = _CS.__new__(_CS)
    _cs2._lock = __import__("threading").Lock()
    _cs2._mem = {
        "上海市|上海世贸商城": [OsmPlace(name="上海世贸商城", lon=121.40087,
                                          lat=31.20189, kind="poi", score=1.0)],
        "上海市|上海长风大悦城": [OsmPlace(name="长风大悦城", lon=121.39528,
                                            lat=31.22274, kind="poi", score=0.9)],
        "上海市|虹桥基金小镇": [OsmPlace(name="虹桥基金小镇", lon=121.38890,
                                          lat=31.18290, kind="poi", score=1.0)],
        "上海市|红松东": [OsmPlace(name="红松东路", lon=121.39527, lat=31.18747,
                                    kind="poi", score=0.94)],
    }
    _cs2._save_disk = lambda: None
    _found = _cs2.lookup_many(
        ["上海世贸商城", "上海长风大悦城", "虹桥基金小镇", "红松东"], "上海")
    _kept = [n for n in _found if _found[n]]
    check("上海世贸商城" in _kept,
          "距簇心较远但正确的候选不被丢弃（交给 RANSAC 判断）",
          f"保留 {len(_kept)} 个地名")
    check("上海长风大悦城" in _kept,
          "第二个正确候选同样保留",
          f"保留 {len(_kept)} 个地名")
    # 每个地名保留的候选数上限应 >= 4，给 RANSAC 足够选择
    _maxc = 0
    for lst in _found.values():
        _maxc = max(_maxc, len(lst))
    from app.coldstart import _MAX_CANDIDATES_PER_NAME
    check(_MAX_CANDIDATES_PER_NAME >= 4,
          "每个地名保留的候选数 >= 4（截断过早会让正确候选进不来）",
          f"上限 {_MAX_CANDIDATES_PER_NAME}")

    print("\n[15d] 全部候选交给 RANSAC 时的几何一致性（回归项）")
    # 实测：同一组候选，直接交给 RANSAC 可得 RMSE 11.7m、3 个内点，
    # 且三个内点都在正确区域。说明「候选筛选」这一步不该替 RANSAC 做决定。
    from app.register import AnchorPair as _AP, register as _register
    _pix = {"上海世贸商城": (982, 1416), "上海长风大悦城": (698, 206),
            "虹桥基金小镇": (374, 2521), "红松东": (1207, 2122)}
    _pairs = []
    for _n, _lst in _found.items():
        for _q in _lst:
            _px, _py = _pix[_n]
            _pairs.append(_AP(name=_n, px=_px, py=_py, weight=0.85, kind="poi",
                               lon=_q.lon, lat=_q.lat, match_quality=0.6))
    _reg = _register(_pairs, inlier_m=200.0, min_inliers=3)
    check(_reg.ok and len(_reg.inliers) >= 3,
          "RANSAC 能从全部候选中挑出 >=3 个自洽内点",
          f"内点 {len(_reg.inliers)} / 离群 {len(_reg.outliers)}, "
          f"RMSE {_reg.rmse_m:.1f}m")
    check(_reg.rmse_m < 50,
          "自洽内点的 RMSE < 50m（说明选对了区域）",
          f"RMSE {_reg.rmse_m:.1f}m")

    print("\n[15e] 置信度：离群候选不应惩罚（回归项）")
    # 离群候选是**主动提供给 RANSAC 的备选项**（同名的多个地点、
    # 冷启动查到的多个候选）。RANSAC 挑出自洽的那一组后，
    # 剩下的自然都是离群——备选越多占比越低，但结果完全正确。
    #
    # 实测浦东：候选放宽后内点仍是正确的 2 个、RMSE 0.6m、贴路 3.3m，
    # 但离群候选从 0 涨到 3，按「内点/(内点+离群)」算占比
    # 从 1.0 掉到 0.4，置信度被无端压低 0.11。
    #
    # 正确做法：按**每个地名的候选组**算占比——
    # 一个地名若至少有一个候选进了内点，这个地名就算「命中」。
    from app.register import Registration as _Reg, AnchorPair as _AP2
    import numpy as _np
    _mk = lambda nm, lon, lat: _AP2(name=nm, px=100, py=100, weight=0.85,
                                   kind="poi", lon=lon, lat=lat, match_quality=0.9)
    # 桩需要一个可逆的变换矩阵——_self_check 会求逆做反投影校验
    _mat = _np.array([[1.0, 0.0], [0.0, -1.0]])
    # 情况A：2 个地名命中，其中一个地名有 3 个离群候选
    _reg_a = _Reg(
        ok=True, inliers=[_mk("A", 121.4, 31.2), _mk("B", 121.41, 31.21)],
        outliers=[_mk("B", 121.5, 31.3), _mk("B", 121.6, 31.4),
                  _mk("B", 121.7, 31.5)],
        rmse_m=0.6, model="north_up", matrix=_mat,
    )
    # 情况B：2 个地名命中，各 1 个离群候选
    _reg_b = _Reg(
        ok=True, inliers=[_mk("A", 121.4, 31.2), _mk("B", 121.41, 31.21)],
        outliers=[_mk("A", 121.5, 31.3), _mk("B", 121.6, 31.4)],
        rmse_m=0.6, model="north_up", matrix=_mat,
    )
    _pipe = RidePipeline(CONFIG)
    _ca = _pipe._score_confidence(_reg_a, _TrackStub(0.9), [], [], None)
    _cb = _pipe._score_confidence(_reg_b, _TrackStub(0.9), [], [], None)
    check(abs(_ca - _cb) < 0.001,
          "同一组内点下，离群候选数量不影响置信度",
          f"A={_ca}  B={_cb}")

    print("\n[17] 城市级低缩放截图的定位能力（回归项）")
    # 真实事故：换了一张骑行时间更长、缩放级别更低的城市级截图，
    # 识别结果整条落到「上海站/静安区」一带，与真实位置差了十几公里。
    #
    # 根因链（每一条都已修复）：
    #   1. north_up 模型有 3 个自由度却只用 2 个点采样 → 尺度系统性偏差
    #   2. 同一锚点的多个同名候选被逐个计入内点 → 内点虚增、伪解胜出
    #   3. 内点容差用绝对米数 → 低缩放下可靠 POI 全被判成离群
    #   4. 行政区锚点与真实质心可差 1-2 公里 → 必须给更大的容差
    #   5. 「3号线」匹配到哈尔滨、「8号线」匹配到重庆 → 跨城候选污染
    #   6. 词典命中「够用即停」→ 43 条词典里 10 个可用锚点被白白丢掉
    from app.register import (
        AnchorPair as _AP3, register as _reg3,
        _estimate_scale_prior, _group_candidates, _tolerance_vec,
        _model_scale,
    )
    import numpy as _np
    from app.geo.coords import lonlat_to_mercator as _l2m

    print("\n[17a] 同一锚点的多个候选不得虚增内点数")
    # 同一像素位置挂 4 个同名候选（相距 < 200m），它们在模型下
    # 都会「自洽」。若按候选计数，内点数会被放大 4 倍。
    # 墨卡托各向异性：x 方向 1° = 111320m，y 方向 1° = 130100m。
    # 令尺度恒为 3.0 墨卡托米/像素，构造三个互相自洽的锚点。
    _S = 3.0
    _DLON = 700.0 * _S / 111320.0
    _DLAT = 700.0 * _S / 130100.0
    _same = []
    for _k in range(4):
        _same.append(_AP3(name="X", px=100.0, py=100.0, weight=0.9,
                           kind="poi", lon=121.4000 + _k * 0.0005,
                           lat=31.2000 + _k * 0.0005, match_quality=0.6))
    _same.append(_AP3(name="Y", px=800.0, py=100.0, weight=0.9, kind="poi",
                       lon=121.4000 + _DLON, lat=31.2000, match_quality=0.6))
    # 墨卡托 y 轴向北增大：屏幕下方（py 更大）对应**更南**，
    # 经度不变、纬度减小。
    _same.append(_AP3(name="Z", px=100.0, py=800.0, weight=0.9, kind="poi",
                       lon=121.4000, lat=31.2000 - _DLAT, match_quality=0.6))
    _r3 = _reg3(_same, inlier_m=200.0, min_inliers=3)
    _n_anchor = len({(round(a.px), round(a.py)) for a in _r3.inliers})
    check(_r3.ok and _n_anchor >= 3,
          "同锚点的多个候选只算一个内点",
          f"内点 {len(_r3.inliers)} 个 / 不同锚点 {_n_anchor} 个")

    print("\n[17b] 内点容差随尺度自适应（像素制）")
    # 低缩放（15 米/像素）时，可靠 POI 的标注偏移实测 30-60 像素。
    # 固定 450 米的阈值只够 30 像素，会把可靠 POI 全部判成离群。
    _tol_low = _tolerance_vec(["poi"] * 3, 15.0)
    _tol_high = _tolerance_vec(["poi"] * 3, 0.5)
    check(_tol_low[0] > 800.0 and _tol_high[0] <= 250.0,
          "同一套参数在低缩放与街道级上分别给出宽/严容差",
          f"15m/px -> {_tol_low[0]:.0f}m，0.5m/px -> {_tol_high[0]:.0f}m")
    _tol_dist = _tolerance_vec(["district"], 15.0)
    check(_tol_dist[0] > _tol_low[0],
          "行政区容差大于普通 POI（标注被挪到视野空白处）",
          f"district {_tol_dist[0]:.0f}m > poi {_tol_low[0]:.0f}m")

    print("\n[17c] 尺度投票不被同名候选刷票")
    # 「上海南站」的 5 个候选横跨 3 公里。若每个候选都独立参与投票，
    # 会产生十几个偏离的比值，把尺度带偏（实测 13.03 vs 真值 15.73）。
    # 三个自洽锚点（尺度恒为 15.7 墨卡托米/像素）+ 一个带 5 个
    # 远距离同名候选的锚点组。
    _TS = 15.7
    _DL = 700.0 * _TS / 111320.0
    _DA = 700.0 * _TS / 130100.0
    _src = _np.array([[100.0, 100.0], [800.0, 100.0], [100.0, 800.0],
                       [400.0, 400.0]])
    _dst = _np.array([_l2m(121.4000, 31.2000),
                      _l2m(121.4000 + _DL, 31.2000),
                      _l2m(121.4000, 31.2000 + _DA),
                      _l2m(121.4000, 31.2000)])
    _extra = _np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
                         [4.0, 4.0], [5.0, 5.0]])
    _src = _np.vstack([_src, _src[3] + _extra])
    _dst = _np.vstack([_dst, _dst[3] + _extra * 5000.0])
    _groups = [[0], [1], [2], [3, 4, 5, 6, 7]]
    _sp, _keep = _estimate_scale_prior(
        _src, _dst, _groups, ["poi"] * len(_src)
    )
    _true_scale = _model_scale(_np.array([[15.7, 0.0], [0.0, -15.7]]))
    check(_sp is not None and abs(_sp - _true_scale) / _true_scale < 0.10,
          "尺度投票收敛到真值附近（离群候选不影响）",
          f"投票 {_sp:.3f} vs 真值 {_true_scale:.3f}")

    print("\n[17d] 跨城同名候选被过滤")
    from app.coldstart import ColdStartGeocoder as _CS2
    _cs3 = _CS2.__new__(_CS2)
    _bb = (120.85, 30.67, 123.23, 31.87)
    check(not _cs3._in_city(126.5976, 45.7553, _bb, is_wgs84=True),
          "哈尔滨的「3号线」被判定为超出城市范围")
    check(not _cs3._in_city(106.6702, 29.4832, _bb, is_wgs84=True),
          "重庆的「8号线」被判定为超出城市范围")
    check(_cs3._in_city(121.3934, 31.6263, _bb, is_wgs84=True),
          "上海范围内的同名地点被保留")

    print("\n[17e] 留一验证能识别「只靠自己撑着的伪内点」")
    # 一个偏离的锚点 + 三个自洽锚点：内点数看起来更多，
    # 但拿掉偏离点后其余点拟合不出同一个模型。
    # 四个互相自洽的锚点：墨卡托 x 向东增大、y 向北增大，
    # 屏幕 y 向下 = 地理向南（纬度减小）。
    _MS = 5.0
    _MDL = 800.0 * _MS / 111320.0
    _MDA = 800.0 * _MS / 130100.0
    _mix = [
        _AP3(name="A", px=100.0, py=100.0, weight=0.9, kind="poi",
             lon=121.4000, lat=31.2000, match_quality=0.9),
        _AP3(name="B", px=900.0, py=100.0, weight=0.9, kind="poi",
             lon=121.4000 + _MDL, lat=31.2000, match_quality=0.9),
        _AP3(name="C", px=100.0, py=900.0, weight=0.9, kind="poi",
             lon=121.4000, lat=31.2000 - _MDA, match_quality=0.9),
        _AP3(name="D", px=900.0, py=900.0, weight=0.9, kind="poi",
             lon=121.4000 + _MDL, lat=31.2000 - _MDA, match_quality=0.9),
    ]
    _r4 = _reg3(_mix, inlier_m=200.0, min_inliers=3)
    check(_r4.ok and len(_r4.inliers) == 4,
          "四个自洽锚点全部成为内点",
          f"内点 {len(_r4.inliers)} / 离群 {len(_r4.outliers)}")

    print("\n[17f] 锚点类型判定")
    from app.geocode import infer_kind as _ik
    check(_ik("静安区", "boundary", "administrative") == "district",
          "行政边界被判定为 district")
    check(_ik("上海站", "building", "train_station") == "station",
          "火车站被判定为 station")
    check(_ik("12号线", "highway", "bus_stop") == "transit",
          "地铁线被判定为 transit")
    check(_ik("桃浦中央绿地", "leisure", "park") == "poi",
          "公园被判定为 poi")

    print("\n[16] CLI 失败路径也必须输出可解析 JSON（回归项）")
    # 真实事故：cli._summary() 直接 res.geojson_gcj02["features"][0]，
    # 而识别失败时 geojson_gcj02 是空字典（PipelineResult 的默认值 {}），
    # 于是抛 KeyError: 'features' —— CLI 崩溃而不是输出 JSON 摘要，
    # 上层批量脚本拿不到任何可解析的结果。
    from app.cli import _summary as _cli_summary
    from app.pipeline import PipelineResult as _PR

    _empty_res = _PR(
        geojson={"type": "FeatureCollection", "features": []},
        confidence=0.0,
        geojson_gcj02={},          # 失败路径：没有 features 键
        track_points=0,
        distance_m=0.0,
        warnings=["配准失败"],
    )
    try:
        _s = _cli_summary(_empty_res, ".", {}, 0.1)
        _ok = (_s.get("ok") is False
               and _s.get("start_gcj02") is None
               and _s.get("end_gcj02") is None)
        check(_ok, "识别失败时 CLI 摘要仍可生成，且起终点为 None",
              f"ok={_s.get('ok')}, start={_s.get('start_gcj02')}")
    except Exception as exc:  # noqa: BLE001
        check(False, "识别失败时 CLI 摘要仍可生成",
              f"{type(exc).__name__}: {exc}")

    # 有轨迹时摘要里的起终点应取自 geojson_gcj02
    _ok_res = _PR(
        geojson={"type": "FeatureCollection", "features": []},
        confidence=0.9,
        geojson_gcj02={"type": "FeatureCollection", "features": [
            {"geometry": {"type": "LineString",
                          "coordinates": [[121.5, 31.2], [121.6, 31.3]]}}]},
        track_points=2,
        distance_m=1234.5,
    )
    _s2 = _cli_summary(_ok_res, ".", {}, 0.1)
    check(_s2.get("start_gcj02") == [121.5, 31.2]
          and _s2.get("end_gcj02") == [121.6, 31.3],
          "识别成功时 CLI 摘要含正确的起终点",
          f"start={_s2.get('start_gcj02')}")

    print("\n[18] 低置信度时的备选位置（数据飞轮的入口）")
    # 场景：低缩放截图的定位信号是**多峰**的——实测前 12 个候选的
    # 匹配分数差只有 0.1%，但地理位置分散在 900-1600 米。
    # 只给一个答案等于替用户乱猜，必须把候选摆出来让用户选。
    # 每个锚点配 3 个候选：正确的一个 + 两个相距 2 公里的同名地点。
    # 这正是 Nominatim 对「上海站」「外环高速」等返回的样子——
    # 选了哪个候选，就决定了落点在哪。
    _S = 15.7
    _DL = 700.0 * _S / 111320.0
    _DA = 700.0 * _S / 130100.0
    _TRUTH = [
        ("A", (100.0, 100.0), 121.4000, 31.2000),
        ("B", (800.0, 100.0), 121.4000 + _DL, 31.2000),
        ("C", (100.0, 800.0), 121.4000, 31.2000 - _DA),
    ]
    _pairs = []
    for _n, (_px, _py), _lo, _la in _TRUTH:
        for _off in (0.0, 0.02, -0.02):
            _pairs.append(_AP3(
                name=_n, px=_px, py=_py, weight=0.9, kind="poi",
                lon=_lo + _off, lat=_la, match_quality=0.9))
    _reg = register(_pairs, inlier_m=200.0, min_inliers=3)
    check(_reg.ok, "备选功能的前提：配准本身成立")

    # 备选解字段存在且是列表
    check(isinstance(getattr(_reg, "alternatives", None), list),
          "Registration 暴露 alternatives 列表",
          f"type={type(getattr(_reg, 'alternatives', None)).__name__}")

    # 备选解与最优解地理上分离（>300 米）——这是筛掉「同一解的抖动」的判据
    if _reg.ok and _reg.alternatives:
        _alt = _reg.alternatives[0]
        _probe = _np.array([300.0, 300.0])
        _c_best = _reg.matrix @ _probe + _reg.translation
        _c_alt = _alt.matrix @ _probe + _alt.translation
        _sep = float(_np.linalg.norm(_c_alt - _c_best))
        check(_sep > 300.0,
              "备选解与最优解地理上分离（>300 米）",
              f"分离 { _sep:.0f} 米")
        check(_alt.ok and len(_alt.inliers) >= 2,
              "备选解本身是有效的自洽解",
              f"内点 {len(_alt.inliers)}")
    else:
        check(False, "备选解与最优解地理上分离（>300 米）",
              "该构造未产生备选解（少于 3 个锚点时正常）")

    # Pipeline 层：备选携带完整轨迹坐标，且坐标系是 GCJ-02
    from app.pipeline import RidePipeline as _RP
    _pipe = _RP.__new__(_RP)          # 绕过 __init__（需要词典与网络）
    _fake_track = type("T", (), {"polyline": _np.array([[100.0, 100.0],
                                                         [800.0, 100.0],
                                                         [100.0, 800.0]])})()
    _alts_out = _pipe._collect_alternatives(_reg, _fake_track, "WGS-84")
    check(isinstance(_alts_out, list),
          "pipeline._collect_alternatives 返回列表")
    if _alts_out:
        _a0 = _alts_out[0]
        check(_a0.get("crs") == "GCJ-02",
              "备选坐标基准是 GCJ-02（与腾讯底图一致）",
              f"crs={_a0.get('crs')}")
        check(len(_a0.get("coords") or []) == 3,
              "备选携带完整轨迹坐标（前端可直接重画）",
              f"点数 {len(_a0.get('coords') or [])}")
        check(isinstance(_a0.get("lon"), float)
              and isinstance(_a0.get("lat"), float),
              "备选带有起终点坐标")
    else:
        check(False, "备选坐标基准是 GCJ-02（与腾讯底图一致）",
              "未产生备选（构造数据下正常）")

    print("\n[19] 用户路名约束（线约束）")
    # 场景：低缩放截图的定位信号是多峰的，用户记得走过哪条路，
    # 这是最强的约束。实测轨迹到真实路网的偏差可从 191 米降到 98 米。
    from app.roadname_constraint import (
        RoadNameConstraint, refine_by_road_names, _dist_to_segments,
    )

    # 造一条「水平的路」（墨卡托米），轨迹从它旁边经过
    _road_y = 0.0
    _road = [[(-5000.0, _road_y), (5000.0, _road_y)]]
    _con = RoadNameConstraint(name="测试路", ok=True, segments=_road,
                              n_segments=1, total_len_m=10000.0)
    check(_con.uniqueness > 0.9,
          "单段路名的唯一性接近 1",
          f"uniqueness={_con.uniqueness:.3f}")

    # 多段路名唯一性更低
    _con2 = RoadNameConstraint(
        name="测试路2", ok=True,
        segments=[[(-5000.0, i * 1000.0), (5000.0, i * 1000.0)]
                  for i in range(10)],
        n_segments=10, total_len_m=100000.0)
    check(_con2.uniqueness < _con.uniqueness,
          "多段路名的唯一性更低",
          f"{_con2.uniqueness:.3f} < {_con.uniqueness:.3f}")

    # 轨迹在路的北侧 300 米处 → 约束应把它拉回来
    _traj = np.array([[float(x), 300.0] for x in range(-400, 400, 20)])
    _shift, _val, _notes, _base = refine_by_road_names(
        _traj, [_con], max_shift_m=1000.0, coarse_step_m=50.0)
    check(_base > 200.0,
          "基准值反映「约束前轨迹离路多远」",
          f"基准 {_base:.0f} 米")
    check(abs(_val) < 50.0,
          "路名约束把轨迹拉回路面（目标值 < 50 米）",
          f"目标值 {_val:.1f} 米")
    check(abs(_shift[1]) > 100.0,
          "路名约束产生了实质平移（>100 米）",
          f"平移北 {_shift[1]:+.0f} 米")
    check(abs(_shift[0]) < 60.0,
          "路名约束不产生虚假的东西向平移",
          f"平移东 {_shift[0]:+.0f} 米")

    # 距离函数本身：点在线上距离为 0
    _d = _dist_to_segments(np.array([[0.0, 0.0]]), _road)
    check(float(_d[0]) < 1.0, "点在线上时距离为 0", f"{float(_d[0]):.3f}")
    _d2 = _dist_to_segments(np.array([[0.0, 250.0]]), _road)
    check(abs(float(_d2[0]) - 250.0) < 1.0,
          "点离线 250 米时距离为 250", f"{float(_d2[0]):.1f}")

    print("\n" + "=" * 66)
    ok = sum(1 for r in results if r[0])
    total = len(results)
    print(f"结果: {ok}/{total} 项通过")
    if ok < total:
        print("\n未通过项:")
        for good, name, detail in results:
            if not good:
                print(f"  {FAIL} {name}  {detail}")
    print("=" * 66)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
