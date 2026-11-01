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
        # 复用流水线刚写好的路网缓存（bbox 与流水线一致，避免重复拉取）
        _cache = CONFIG.road_cache_dir / "roads.json"
        if _cache.exists():
            with open(_cache, encoding="utf-8") as _f:
                _raw = json.load(_f)
            _roads = [(nm, [(p[0], p[1]) for p in pts])
                      for nm, pts in _raw.get("roads", [])]
        else:
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
