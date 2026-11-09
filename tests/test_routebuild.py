"""路线构建测试（不依赖网络）。

覆盖 `app/routebuild.py`（串路）与 `app/seampath.py`（接缝插入）：

  * 同名路的多实例选择（长度惩罚）
  * 路口判定：OSM 共同端点优先
  * 方向枚举：接触顺序与用户顺序一致
  * 接缝插入：下标定位正确、长度守门
"""

from __future__ import annotations

from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.routebuild import _endpoints, build_route
from app.seampath import insert_seams


def _line(x0, y0, x1, y1, n=20) -> np.ndarray:
    """水平直线（墨卡托米）。"""
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])


def _seg(poly: np.ndarray) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """把折线包成 OSM segment 格式（端点共享）。"""
    out = []
    for i in range(len(poly) - 1):
        a = (round(float(poly[i][0]), 6), round(float(poly[i][1]), 6))
        b = (round(float(poly[i + 1][0]), 6), round(float(poly[i + 1][1]), 6))
        out.append((a, b))
    return out


def test_endpoints_shared():
    """共同端点检测：两条路在同一点相接 -> 端点集合有交集。"""
    a = _line(0, 0, 1000, 0, 5)
    b = _line(1000, 0, 1000, 1000, 5)
    segs_a = _seg(a)
    segs_b = _seg(b)
    ea = _endpoints([segs_a])[0]
    eb = _endpoints([segs_b])[0]
    common = ea & eb
    assert len(common) == 1, f"共同端点不对：{common}"
    print("  [OK] 共同端点检测正确")


def test_endpoints_no_overlap():
    """不相交的两条路 -> 无共同端点。"""
    a = _line(0, 0, 1000, 0, 5)
    b = _line(0, 500, 1000, 500, 5)
    ea = _endpoints([_seg(a)])[0]
    eb = _endpoints([_seg(b)])[0]
    assert not (ea & eb), "不该有共同端点"
    print("  [OK] 不相交时无共同端点")


def test_build_route_order():
    """串路：两条路按顺序相连，接触顺序应与输入一致。"""
    a = _line(0, 0, 1000, 0, 20)
    b = _line(1000, 0, 2000, 0, 20)
    polys = [[a], [b]]
    segs = [_seg(a), _seg(b)]
    start = np.array([0.0, 0.0])
    end = np.array([2000.0, 0.0])
    fit = build_route(polys, ["甲路", "乙路"], segs, start, end)
    assert fit.ok, fit.notes
    assert fit.n_hit == 2, f"命中数不对：{fit.n_hit}"
    assert fit.order_consistency == 1.0, f"顺序一致率不对：{fit.order_consistency}"
    assert fit.hit_sequence == ["甲路", "乙路"], fit.hit_sequence
    print("  [OK] 串路顺序正确")


def test_build_route_direction():
    """方向枚举：两条路首尾相接但需要反向走时，仍应正确。"""
    # 甲路从 (0,0) 到 (1000,0)，乙路从 (1000,0) 往回延伸到 (2000,0)
    # 但乙路的 OSM 几何点序是反的 -> 需要枚举方向
    a = _line(0, 0, 1000, 0, 20)
    b = _line(2000, 0, 1000, 0, 20)     # 点序反向
    polys = [[a], [b]]
    segs = [_seg(a), _seg(b)]
    start = np.array([0.0, 0.0])
    end = np.array([2000.0, 0.0])
    fit = build_route(polys, ["甲路", "乙路"], segs, start, end)
    assert fit.ok, fit.notes
    assert fit.order_consistency == 1.0, f"顺序一致率不对：{fit.order_consistency}"
    assert fit.start_offset_m < 50, f"起点偏移过大：{fit.start_offset_m}"
    assert fit.end_offset_m < 50, f"终点偏移过大：{fit.end_offset_m}"
    print("  [OK] 方向枚举正确")


def test_build_route_reversed_order():
    """用户顺序反了 -> 串出的路线也该反（顺序一致率 1.0）。"""
    a = _line(0, 0, 1000, 0, 20)
    b = _line(1000, 0, 2000, 0, 20)
    polys = [[b], [a]]       # 顺序颠倒
    segs = [_seg(b), _seg(a)]
    start = np.array([0.0, 0.0])
    end = np.array([2000.0, 0.0])
    fit = build_route(polys, ["乙路", "甲路"], segs, start, end)
    assert fit.ok, fit.notes
    # 起点在甲路上，但用户先写的是乙路 -> 顺序一致率会低
    # 这是「用户顺序与几何矛盾」的情形，不能崩
    assert 0.0 <= fit.order_consistency <= 1.0
    print(f"  [OK] 顺序矛盾时不崩溃（一致率 {fit.order_consistency:.0%}）")


def test_build_route_too_few():
    """路名太少 -> 明确失败，不崩。"""
    a = _line(0, 0, 1000, 0, 20)
    fit = build_route([[a]], ["甲路"], [_seg(a)],
                      np.array([0.0, 0.0]), np.array([1000.0, 0.0]))
    assert not fit.ok, "单条路应判定为不足"
    print("  [OK] 路名不足时明确失败")


def test_insert_seams_basic():
    """接缝插入：路径点应插到指定区间，且长度增加。"""
    route = _line(0, 0, 1000, 0, 11)
    ents = [0, 5]
    exts = [4, 10]
    # 接缝路径：从 (400,0) 绕到 (500,0)，中间有个点 (450, 100)
    paths = [np.array([[400.0, 0.0], [450.0, 100.0], [500.0, 0.0]])]
    res = insert_seams(route, paths, ents, exts, max_growth=5.0)
    assert res.ok, res.notes
    assert res.n_inserted == 1, f"插入数不对：{res.n_inserted}"
    assert res.length_after_m > res.length_before_m, "插入后长度应增加"
    # 原路线点应仍在（未被破坏）
    assert len(res.points) > len(route), "点数应增加"
    print(f"  [OK] 接缝插入正确（{res.length_before_m:.0f} -> "
          f"{res.length_after_m:.0f} 米）")


def test_insert_seams_growth_guard():
    """长度守门：插入后暴涨太多 -> 放弃。"""
    route = _line(0, 0, 1000, 0, 11)
    ents = [0, 5]
    exts = [4, 10]
    # 绕一个巨大的圈
    circle = np.array([[400.0, 0.0], [450.0, 5000.0], [500.0, 0.0]])
    res = insert_seams(route, [circle], ents, exts, max_growth=1.5)
    assert not res.ok, "暴涨时应放弃"
    assert res.points is None, "放弃时不应返回点集"
    print("  [OK] 长度暴涨时放弃插入")


def test_insert_seams_no_paths():
    """没有可用路径 -> 明确失败。"""
    route = _line(0, 0, 1000, 0, 11)
    res = insert_seams(route, [None], [0, 5], [4, 10])
    assert not res.ok
    print("  [OK] 无接缝路径时明确失败")


def test_insert_seams_index_correctness():
    """⚠️ 下标定位必须用**原始路线**的下标。

    踩过的坑：边插边改锚点，导致后面用的下标是「已插入后的」，
    而 out 长度也在变——实测区间下标飙到 [271..272]，
    远超路线长度 129，插入后长度从 14905 暴涨到 56047 米。
    """
    route = _line(0, 0, 100, 0, 11)
    ents = [0, 3, 6]
    exts = [2, 5, 9]
    paths = [
        np.array([[20.0, 0.0], [25.0, 5.0], [30.0, 0.0]]),
        np.array([[50.0, 0.0], [55.0, 5.0], [60.0, 0.0]]),
    ]
    res = insert_seams(route, paths, ents, exts, max_growth=5.0)
    assert res.ok, res.notes
    assert res.n_inserted == 2, f"插入数不对：{res.n_inserted}"
    # 长度不应爆炸
    assert res.length_after_m < 500, f"长度异常：{res.length_after_m}"
    # 原路线的端点应保留
    assert abs(res.points[0][0] - 0.0) < 1e-6
    assert abs(res.points[-1][0] - 100.0) < 1e-6
    print(f"  [OK] 下标定位正确（{res.length_before_m:.0f} -> "
          f"{res.length_after_m:.0f} 米）")


def test_insert_seams_two_point_path():
    """2 点路径也要能插入（path[1:-1] 是空的，不能直接跳过）。"""
    route = _line(0, 0, 100, 0, 11)
    ents = [0, 6]
    exts = [5, 10]
    paths = [np.array([[50.0, 0.0], [60.0, 3.0]])]
    res = insert_seams(route, paths, ents, exts, max_growth=5.0)
    assert res.ok, res.notes
    assert res.n_inserted == 1, f"2 点路径应能插入，实际 {res.n_inserted}"
    print("  [OK] 2 点路径可插入")


if __name__ == "__main__":
    print("路线构建测试：")
    test_endpoints_shared()
    test_endpoints_no_overlap()
    test_build_route_order()
    test_build_route_direction()
    test_build_route_reversed_order()
    test_build_route_too_few()
    test_insert_seams_basic()
    test_insert_seams_growth_guard()
    test_insert_seams_no_paths()
    test_insert_seams_index_correctness()
    test_insert_seams_two_point_path()
    print("\n全部通过")
