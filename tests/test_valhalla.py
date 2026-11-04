"""Valhalla 客户端测试（不依赖网络的部分）。

覆盖：
  * encoded polyline 解码的正确性
  * location 数量上限（公共实例 max 10）
  * 缓存键稳定性（必须用 md5，不能用内置 hash）
  * 路名命中与**顺序一致性**判定
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, r"D:\Dev\meituan-ride")

from app.valhalla import (
    MAX_LOCATIONS,
    ValhallaClient,
    ValhallaRoute,
    _thin_locations,
    decode_polyline,
    match_ratio,
    order_consistency,
    polyline_length,
)


def test_decode_polyline():
    """解码必须正确。

    ⚠️ **标准 polyline 的第一组 delta 是「从原点到首点」**，
    不是「首点到第二点」——所以第一个解码出来的点就是首点本身。
    （写测试时我一开始把这一点搞反了，误以为解码器有问题；
    后来用真实 Valhalla 返回的 shape 交叉验证——
    821 字符解出 188 点，首末点与请求坐标误差 < 10 米，
    确认解码器是对的，错的是期望值。）
    """
    def _enc(v: int) -> str:
        v = v << 1 if v >= 0 else ~(v << 1)
        out = ""
        while True:
            b = v & 0x1F
            v >>= 5
            if v:
                b |= 0x20
            out += chr(b + 63)
            if not v:
                break
        return out

    # 首点 delta = (lat 0.001, lon 0.001)，第二点 delta 同理
    enc = _enc(1000) + _enc(1000) + _enc(1000) + _enc(1000)
    pts = decode_polyline(enc, precision=6)
    assert len(pts) == 2, f"点数不对：{len(pts)}"
    assert abs(pts[0][0] - 0.001) < 1e-9, f"首点经度不对：{pts[0]}"
    assert abs(pts[0][1] - 0.001) < 1e-9, f"首点纬度不对：{pts[0]}"
    assert abs(pts[1][0] - 0.002) < 1e-9, f"末点经度不对：{pts[1]}"
    assert abs(pts[1][1] - 0.002) < 1e-9, f"末点纬度不对：{pts[1]}"
    print("  [OK] encoded polyline 解码正确")


def test_decode_polyline_negative():
    """负 delta（西南方向）必须正确处理。"""
    def _enc(v: int) -> str:
        v = ~(v << 1) if v < 0 else (v << 1)
        out = ""
        while True:
            b = v & 0x1F
            v >>= 5
            if v:
                b |= 0x20
            out += chr(b + 63)
            if not v:
                break
        return out

    # 往西南：lat -0.001, lon -0.001
    enc = _enc(-1000) + _enc(-1000)
    pts = decode_polyline(enc, precision=6)
    assert len(pts) == 1
    assert abs(pts[0][0] + 0.001) < 1e-9, f"负 delta 解码不对：{pts[0]}"
    print("  [OK] 负 delta 解码正确")


def test_decode_polyline_empty():
    assert decode_polyline("") == []
    assert decode_polyline(None) == []
    print("  [OK] 空输入返回空列表")


def test_max_locations_thinning():
    """超过上限时必须截断到上限，且首尾保留。"""
    pts = [(121.0 + i * 0.01, 31.0 + i * 0.01) for i in range(20)]
    out = _thin_locations(pts, MAX_LOCATIONS)
    assert len(out) <= MAX_LOCATIONS, f"仍超限：{len(out)}"
    assert out[0] == pts[0], "起点必须保留"
    assert out[-1] == pts[-1], "终点必须保留"
    # 首尾之间必须单调（不能乱序）
    for i in range(1, len(out)):
        assert out[i][0] >= out[i - 1][0], "顺序被打乱"
    print(f"  [OK] 20 个点截断到 {len(out)} 个，首尾保留且顺序单调")


def test_thinning_below_limit():
    pts = [(121.0, 31.0), (121.1, 31.1)]
    assert _thin_locations(pts, MAX_LOCATIONS) == pts
    print("  [OK] 未超限时不改动")


def test_cache_key_stable():
    """缓存键必须跨进程稳定。

    ⚠️ 内置 `hash()` 对字符串受 PYTHONHASHSEED 随机化影响，
    每次进程启动都不一样——缓存永远不命中（实测端到端耗时
    从 150 秒涨到 528 秒）。
    """
    c1 = ValhallaClient(cache_dir="data/cache")
    c2 = ValhallaClient(cache_dir="data/cache")
    pts = [(121.4, 31.2), (121.5, 31.3)]
    k1 = c1._cache_key("pedestrian", pts)
    k2 = c2._cache_key("pedestrian", pts)
    assert k1 == k2, "缓存键不稳定"
    # 不同 costing 必须不同键
    k3 = c1._cache_key("auto", pts)
    assert k1 != k3, "不同 costing 共用了缓存键"
    assert len(k1) == 16, f"键长度异常：{len(k1)}"
    print(f"  [OK] 缓存键稳定（{k1}）")


def test_match_ratio():
    names = ["中山南二路", "东安路", "肇嘉浜路", "虹桥路"]
    user = ["中山南二路", "东安路", "肇嘉浜路"]
    r = match_ratio(names, user)
    assert abs(r - 1.0) < 1e-9, f"命中率不对：{r}"
    r2 = match_ratio(names, ["完全不存在的路"])
    assert r2 == 0.0
    print("  [OK] 命中率计算正确")


def test_match_ratio_substring():
    """子串双向匹配：「虹桥路」应命中「虹桥路1829弄」。"""
    names = ["虹桥路1829弄"]
    user = ["虹桥路"]
    assert match_ratio(names, user) == 1.0, "子串匹配失败"
    user2 = ["虹桥路1829弄"]
    assert match_ratio(names, user2) == 1.0, "反向子串匹配失败"
    print("  [OK] 子串双向匹配正确")


def test_order_consistency():
    """顺序一致性：命中的路名**顺序也要对**。"""
    road_names = ["中山南二路", "东安路", "肇嘉浜路", "虹桥路", "水城路"]
    user = ["小木桥路", "中山南二路", "东安路", "肇嘉浜路", "虹桥路", "水城路"]
    score, seq = order_consistency(road_names, user)
    assert score == 1.0, f"顺序一致率不对：{score}"
    assert seq == ["中山南二路", "东安路", "肇嘉浜路", "虹桥路", "水城路"], \
        f"顺序序列不对：{seq}"
    print("  [OK] 顺序一致性判定正确（100%）")

    # 顺序颠倒时应得分低
    road_names2 = ["水城路", "虹古路", "虹桥路", "肇嘉浜路", "东安路", "中山南二路"]
    score2, _ = order_consistency(road_names2, user)
    assert score2 < 0.5, f"逆序未被识别：{score2}"
    print(f"  [OK] 逆序被识别（一致率 {score2:.0%}）")


def test_order_consistency_too_few():
    score, seq = order_consistency(["某路"], ["某路"])
    assert score == 0.0 and seq == ["某路"]
    print("  [OK] 命中不足 2 条时返回 0")


def test_route_rejects_bad_input():
    c = ValhallaClient(cache_dir="data/cache")
    r = c.route([], costing="pedestrian")
    assert not r.ok and "2 个点" in r.error, r.error
    r2 = c.route([(121.0, 31.0)], costing="pedestrian")
    assert not r2.ok
    r3 = c.route([(121.0, 31.0), (121.1, 31.1)], costing="unicycle")
    assert not r3.ok and "costing" in r3.error, r3.error
    print("  [OK] 非法输入被正确拒绝")


def test_route_disabled():
    c = ValhallaClient(cache_dir="data/cache", enabled=False)
    r = c.route([(121.0, 31.0), (121.1, 31.1)])
    assert not r.ok and "未启用" in r.error
    print("  [OK] 禁用时直接返回")


def test_polyline_length():
    pts = [(121.0, 31.0), (121.0, 31.01)]
    L = polyline_length(pts)
    assert abs(L - 1113.2) < 5.0, f"长度不对：{L}"
    assert polyline_length([]) == 0.0
    assert polyline_length([(1.0, 1.0)]) == 0.0
    print(f"  [OK] 折线长度计算正确（{L:.0f} 米）")


def test_route_dataclass_roundtrip():
    r = ValhallaRoute(ok=True, costing="bicycle", distance_m=1234.5,
                      road_names=["中山南二路"], geometry=[(121.0, 31.0)])
    d = r.to_dict()
    assert d["ok"] and d["costing"] == "bicycle"
    assert d["n_geometry"] == 1
    print("  [OK] 结果对象序列化正确")


if __name__ == "__main__":
    print("Valhalla 客户端测试：")
    test_decode_polyline()
    test_decode_polyline_negative()
    test_decode_polyline_empty()
    test_max_locations_thinning()
    test_thinning_below_limit()
    test_cache_key_stable()
    test_match_ratio()
    test_match_ratio_substring()
    test_order_consistency()
    test_order_consistency_too_few()
    test_route_rejects_bad_input()
    test_route_disabled()
    test_polyline_length()
    test_route_dataclass_roundtrip()
    print("\n全部通过")
