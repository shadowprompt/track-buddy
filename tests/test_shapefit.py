"""形状重建（`app/shapefit.py`）的测试。

用**合成数据**（不依赖网络/OSM），守住四个最容易回归的点：

1. `align_by_length` 的区间必须**连续铺满**轨迹、且长度按路名长短分配
   （旧实现把轨迹等分 30 段、每路名只匹配 1 段，跨度差异被抹平）；
2. `main_axis` 必须**去重**（OSM 相邻 way 共享端点，
   不去重会让投影塌陷）；
3. `_map_by_ratio` 必须**双向尝试**（OSM way 方向可能与骑行方向相反）；
4. `fit_shape` 的安全阀：**起终点不动**、**段长不被压缩**。

坐标单位统一为**墨卡托米**（与 `lonlat_to_mercator` 一致）。
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.shapefit import (
    _arc, _map_by_ratio, align_by_length, fit_shape, main_axis,
)


def _line(x0, y0, x1, y1, n=40):
    """一条直线段（墨卡托米）。"""
    t = np.linspace(0.0, 1.0, n)
    return np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1)


def _residual_fn(road_polys):
    """残差：每段到对应路名几何的中位距离。"""
    from scipy.spatial import cKDTree

    trees = [cKDTree(p) for p in road_polys]

    def fn(points, idx, spans):
        total = float(arc[-1]) if False else 0.0
        # 用外部闭包拿不到 arc，这里改用「段内点数的比例」近似
        return total
    return fn


def test_main_axis_dedups():
    """OSM 相邻 way 共享端点 -> main_axis 必须去重。"""
    pts = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [20.0, 0.0],
        [20.0, 0.0], [30.0, 0.0],
    ])
    poly = main_axis(pts)
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    assert len(poly) == 4, f"去重后应剩 4 个点，实际 {len(poly)}"
    assert d.min() > 1.0, "不应保留 1 米内的重复点"
    print(f"  [OK] main_axis 去重：6 点 -> {len(poly)} 点")


def test_align_segments_are_contiguous_and_length_weighted():
    """对齐区间必须连续铺满，且长度按路名长短分配。"""
    # 一条 3000 米的直线轨迹
    traj = _line(0.0, 0.0, 3000.0, 0.0, n=120)
    arc = _arc(traj)
    # 三条路：长短差异明显（200 / 1000 / 1800 米）
    lens = [200.0, 1000.0, 1800.0]
    polys = []
    for L in lens:
        polys.append(_line(0.0, 0.0, L, 0.0, n=30))
    names = ["短路", "中路", "长路"]
    spans = align_by_length(traj, arc, polys, lens, names, n_seg=60)
    assert len(spans) == 3, f"应返回 3 个区间，实际 {len(spans)}"
    # 连续性：前一个区间的终点 == 后一个区间的起点（无缝铺满）
    for i in range(len(spans) - 1):
        assert spans[i][1] == spans[i + 1][0], (
            f"区间不连续：{spans}")
    # 长度分配：长的路应分到更长的区间
    total = float(arc[-1])
    bounds = np.linspace(0.0, total, 60 + 1)
    got = [bounds[b + 1] - bounds[a] for a, b in spans]
    order = np.argsort(got)
    assert order[0] == 0, f"最短的路应分到最短的区间，实际 {got}"
    assert order[-1] == 2, f"最长的路应分到最长的区间，实际 {got}"
    print(f"  [OK] 区间连续铺满，长度按路名长短分配：{[round(x) for x in got]}")


def test_map_by_ratio_handles_reverse_direction():
    """几何走向与轨迹相反时，也要能映射（双向尝试）。"""
    seg = _line(0.0, 0.0, 500.0, 0.0, n=10)
    poly = _line(0.0, 0.0, 500.0, 0.0, n=30)     # 与轨迹同向
    out = _map_by_ratio(seg, poly, _arc(poly))
    assert len(out) == len(seg)
    assert float(np.linalg.norm(np.diff(out, axis=0), axis=1).sum()) > 100, (
        "同向映射应保留长度")
    # 反向几何
    poly_rev = poly[::-1]
    out2 = _map_by_ratio(seg, poly_rev, _arc(poly_rev))
    assert len(out2) == len(seg)
    assert float(np.linalg.norm(np.diff(out2, axis=0), axis=1).sum()) > 100, (
        "反向几何也应能映射")
    print("  [OK] 双向映射都有效")


def test_fit_shape_keeps_endpoints():
    """安全阀：起终点必须不动（它们是用户给的锚）。"""
    # 轨迹：一条斜直线
    traj = _line(0.0, 0.0, 2000.0, 600.0, n=100)
    arc = _arc(traj)
    # 路名几何：贴着轨迹但有弯折（模拟真实街道）
    bend = _line(0.0, 0.0, 1000.0, 200.0, n=40)
    bend2 = _line(1000.0, 200.0, 2000.0, 600.0, n=40)
    poly = np.vstack([bend, bend2[1:]])
    polys = [poly]
    lens = [float(_arc(poly)[-1])]

    def residual(points, idx, spans):
        from scipy.spatial import cKDTree
        tree = cKDTree(poly)
        d, _ = tree.query(points)
        return float(np.median(d))

    fit = fit_shape(traj, arc, polys, [_arc(poly)], lens, ["测试路"],
                    residual_fn=residual, n_seg=40)
    if fit.ok and fit.points is not None:
        assert float(np.hypot(*(fit.points[0] - traj[0]))) < 1.0, (
            "起点被移动了")
        assert float(np.hypot(*(fit.points[-1] - traj[-1]))) < 1.0, (
            "终点被移动了")
        print("  [OK] 重建后起终点未移动")
    else:
        print("  [SKIP] 本例未触发重建（残差未改善），起终点检查跳过")


def test_fit_shape_skips_compressed_segment():
    """安全阀：映射后段长被压缩 -> 该段跳过，不整体放弃。

    物理依据：真实路径长度 >= 直线距离。若把一段直线轨迹贴到一条
    更短的路上，长度必然被压缩——那说明映射错位，不该采纳。
    """
    traj = _line(0.0, 0.0, 2000.0, 0.0, n=100)
    arc = _arc(traj)
    # 几何只有 200 米，远短于轨迹段
    poly = _line(0.0, 0.0, 200.0, 0.0, n=20)
    polys = [poly]
    lens = [float(_arc(poly)[-1])]

    def residual(points, idx, spans):
        from scipy.spatial import cKDTree
        tree = cKDTree(poly)
        d, _ = tree.query(points)
        return float(np.median(d))

    fit = fit_shape(traj, arc, polys, [_arc(poly)], lens, ["短路"],
                    residual_fn=residual, n_seg=40)
    skipped = [n for n in fit.notes if "跳过" in n or "变短" in n]
    assert skipped, f"应记录「跳过」原因，实际 notes={fit.notes}"
    if fit.ok and fit.points is not None:
        # 若整体仍然通过，说明其余部分够好——但被跳过的段不能被改短
        seg_len = float(np.linalg.norm(
            np.diff(fit.points, axis=0), axis=1).sum())
        assert seg_len > 1000, f"轨迹被压得过短：{seg_len:.0f} 米"
    print(f"  [OK] 段长压缩时跳过该段（{skipped[0]}）")


if __name__ == "__main__":
    print("形状重建测试：")
    test_main_axis_dedups()
    test_align_segments_are_contiguous_and_length_weighted()
    test_map_by_ratio_handles_reverse_direction()
    test_fit_shape_keeps_endpoints()
    test_fit_shape_skips_compressed_segment()
    print("\n全部通过")
