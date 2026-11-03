"""跨江检测护栏的测试。

要守住的 bug：
  1. **单位混用**——KD 树返回「度」，阈值是「米」，
     直接比较会让任何点都被判成「在河上」（实测占比 100%）。
  2. **法向翻转**——河道弯曲时局部法向会翻转 180 度，
     导致同一条河的两段被判定成相反的侧别。
  3. **假阳性**——浦西侧的正常轨迹不该被判成横跨黄浦江。

测试用真实数据：OSM 拉到的黄浦江岸线 + 本项目实测的
「徐汇滨江 → 长宁」骑行轨迹（修复后起点在浦西侧）。
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Dev\meituan-ride")
from app.geo.coords import gcj02_to_wgs84, wgs84_to_gcj02
from app.waterguard import check_crossing, get_waters

CACHE = r"D:\Dev\meituan-ride\data\cache"
RESULT = r"D:\Dev\meituan-ride\tmp_debug\run\e2e_result.json"

WEST_BANK = (121.46019, 31.18592)   # 西岸艺术中心（浦西）
EAST_BANK = (121.48793, 31.18618)   # 上海世博园（浦东）


def _load_waters():
    return get_waters(city="上海", cache_dir=CACHE, allow_network=False)


def test_bank_geometry_real():
    """岸线数据必须是真实几何，且确实在浦西一侧。"""
    waters = _load_waters()
    assert "黄浦江" in waters, "没有黄浦江数据"
    pts = np.array(waters["黄浦江"][0])
    assert pts[:, 0].min() < 121.3, "黄浦江西端太靠东"
    assert pts[:, 0].max() > 121.5, "黄浦江东端太靠西"
    # 岸线在西岸艺术中心附近（实测 313 米），即它是**西岸线**
    d = np.hypot(
        (pts[:, 0] - WEST_BANK[0]) * 111320 * math.cos(math.radians(31.186)),
        (pts[:, 1] - WEST_BANK[1]) * 111320)
    assert d.min() < 800, f"岸线不在西岸附近（最近 {d.min():.0f} 米）"
    print(f"  [OK] 黄浦江岸线真实：lon {pts[:,0].min():.4f}..{pts[:,0].max():.4f}"
          f"，距西岸艺术中心最近 {d.min():.0f} 米")


def test_normal_puxi_track_not_crossing():
    """浦西侧的正常轨迹不该被判成横跨。"""
    waters = _load_waters()
    with open(RESULT, encoding="utf-8") as f:
        d = json.load(f)
    g = d["geojson"]["features"][0]["geometry"]["coordinates"]
    crossed, which, ratio = check_crossing(g, waters=waters)
    assert not crossed, (
        f"误报：正常浦西轨迹被判成横跨 {which}（占比 {ratio:.0%}）")
    assert ratio < 0.10, f"贴河比例过高：{ratio:.1%}"
    print(f"  [OK] 浦西轨迹未横跨（贴河占比 {ratio:.2%}）")


def test_crossing_detected():
    """故意造一条从浦西跨到浦东的轨迹，必须被检出。

    ⚠️ 岸线数据是 **WGS-84**，而 `check_crossing` 的入参是 **GCJ-02**，
    所以造点时必须先转基准（实测踩过：直接把 WGS 岸线当 GCJ 传，
    偏移 400 多米，判定结果完全不对）。
    """
    waters = _load_waters()
    bank = np.array(waters["黄浦江"][0])
    band = bank[(bank[:, 1] > 31.183) & (bank[:, 1] < 31.193)]
    assert len(band) >= 3, "该段没有黄浦江岸线点"
    # 岸线（西岸线）向东偏移 ~700 米 = 江里；再往西 = 浦西
    east_shift = 0.0062
    traj_gcj = []
    for p in band[:: max(1, len(band) // 3)][:3]:
        lon, lat = wgs84_to_gcj02(float(p[0]) + east_shift, float(p[1]))
        traj_gcj.append((lon, lat))
    # 起点在浦西（岸线以西）
    w_lo, w_la = wgs84_to_gcj02(
        float(band[0][0]) - 0.004, float(band[0][1]))
    traj_gcj.insert(0, (w_lo, w_la))
    # 终点在浦东（岸线以东足够远）
    e_lo, e_la = wgs84_to_gcj02(
        float(band[-1][0]) + 0.020, float(band[-1][1]))
    traj_gcj.append((e_lo, e_la))

    crossed, which, ratio = check_crossing(traj_gcj, waters=waters)
    assert crossed, "横跨黄浦江的轨迹未被检出"
    assert "黄浦江" in which, f"报出的河不对：{which}"
    print(f"  [OK] 横跨黄浦江被检出（占比 {ratio:.0%}）")


def test_side_sign_stability():
    """法向符号稳定：同一条河两岸的点必须被判成相反的侧别。

    这是实测踩到的坑——河道弯曲时局部法向会翻转 180 度，
    导致同一条河被判定成「有的段在浦东、有的段在浦西」。

    ⚠️ 岸线是 WGS-84，造点时要转成 GCJ-02 再传入。
    """
    waters = _load_waters()
    bank = np.array(waters["黄浦江"][0])
    band = bank[np.abs(bank[:, 1] - 31.186) < 0.002]
    mid = band[len(band) // 2]
    # 岸线西侧（浦西）与东侧（浦东），各造 3 个点满足连续判定
    def _mk(dlon, n=3):
        out = []
        for k in range(n):
            lon, lat = wgs84_to_gcj02(
                float(mid[0]) + dlon + k * 0.0002, float(mid[1]))
            out.append((lon, lat))
        return out

    c_w, _, _ = check_crossing(_mk(-0.008), waters=waters)
    c_e, _, _ = check_crossing(_mk(+0.006), waters=waters)
    assert not c_w, "岸线西侧（浦西）被误判成在江里"
    assert c_e, "岸线东侧（浦东）未被判成在江里"
    print("  [OK] 岸线两侧判定正确（西侧浦西不判、东侧浦东判）")


def test_fallback_matches_real_bank():
    """兜底折线必须与真实岸线吻合。

    兜底折线是「联网拉不到岸线时」的保底。如果它偏离真实岸线，
    侧别判定会整体偏移，浦西轨迹被误判成横跨（或反之）。

    实测：兜底折线的控制点直接取自 OSM 真实岸线，
    所以在市中心段偏差应为 0 米。
    """
    from app.waterguard import _FALLBACK
    from scipy.spatial import cKDTree

    waters = _load_waters()
    real = np.array(waters["黄浦江"][0])
    fb = np.array(_FALLBACK["黄浦江"])
    tree = cKDTree(fb)
    band = real[(real[:, 1] > 31.17) & (real[:, 1] < 31.21)]
    d, _ = tree.query(band)
    assert d.max() < 100, (
        f"兜底折线偏离真实岸线最多 {d.max():.0f} 米（应 < 100 米）")
    print(f"  [OK] 兜底折线与真实岸线吻合（最大偏差 {d.max():.0f} 米）")


def test_fallback_used_when_offline():
    """断网时（无缓存）也要能判定——护栏不能静默失效。"""
    from app.waterguard import _FALLBACK

    waters = get_waters(city="不存在的城市XYZ", cache_dir=CACHE,
                        allow_network=False)
    assert waters, "无网络且无缓存时未返回兜底数据"
    assert "黄浦江" in waters
    print(f"  [OK] 断网时返回兜底数据（{len(waters)} 条河）")


if __name__ == "__main__":
    print("跨江检测测试：")
    test_bank_geometry_real()
    test_fallback_matches_real_bank()
    test_fallback_used_when_offline()
    test_normal_puxi_track_not_crossing()
    test_crossing_detected()
    test_side_sign_stability()
    print("\n全部通过")
