"""起终点锚定的坐标系一致性测试。

这类 bug（矩阵约定不一致导致轨迹南北镜像）非常隐蔽：
结果看起来「有坐标、有置信度」，只是位置错了。
必须有测试守住。

验证：
  1. `two_point_similarity` + `_apply` 能把像素点还原到指定的地理点
  2. 与 `Registration.pixel_to_mercator` 的约定一致（同样的矩阵能互换）
  3. 尺度与旋转的符号正确（北在上、东在右）
"""
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\Dev\meituan-ride")
from app.endpointfit import two_point_similarity, _apply, _compose
from app.register import Registration
from app.geo.coords import lonlat_to_mercator, mercator_to_lonlat


def test_two_point_roundtrip():
    """两点定相似变换后，能把这两个像素点映射回指定的地理点。"""
    p1 = (968.0, 1414.0)
    p2 = (292.0, 1188.0)
    g1 = (121.45864, 31.19085)
    g2 = (121.37718, 31.21355)
    m1 = lonlat_to_mercator(*g1)
    m2 = lonlat_to_mercator(*g2)

    r = two_point_similarity(p1, p2, m1, m2)
    assert r is not None, "两点定解失败"
    s, th, t = r
    out = _apply(np.array([p1, p2], dtype=float), s, th, t)
    err1 = math.hypot(out[0][0] - m1[0], out[0][1] - m1[1])
    err2 = math.hypot(out[1][0] - m2[0], out[1][1] - m2[1])
    assert err1 < 1e-6, f"起点还原误差 {err1:.2e} 米"
    assert err2 < 1e-6, f"终点还原误差 {err2:.2e} 米"
    print(f"  [OK] 两点还原误差 {err1:.2e} / {err2:.2e} 米")
    print(f"       尺度 {s:.4f} m/px  旋转 {math.degrees(th):+.4f}°")


def test_registration_convention():
    """与 `Registration` 的约定一致：matrix 直接作用在原始像素上。"""
    p1 = (968.0, 1414.0)
    p2 = (292.0, 1188.0)
    g1 = (121.45864, 31.19085)
    g2 = (121.37718, 31.21355)
    m1 = lonlat_to_mercator(*g1)
    m2 = lonlat_to_mercator(*g2)
    s, th, t = two_point_similarity(p1, p2, m1, m2)

    reg = Registration(ok=True, matrix=_compose(s, th, t), translation=t)
    out = reg.pixel_to_mercator(*p1)
    err = math.hypot(out[0] - m1[0], out[1] - m1[1])
    assert err < 1e-6, f"与 Registration 约定不一致，误差 {err:.2e} 米"
    print(f"  [OK] 与 Registration 约定一致，误差 {err:.2e} 米")


def test_orientation():
    """朝向正确：地理上「北」对应像素 y 变小，「东」对应像素 x 变大。"""
    # 构造一个纯北向上的变换：北 1000 米 = 像素 y 减小
    s = 10.0
    p_center = (600.0, 1200.0)
    g_center = lonlat_to_mercator(121.46, 31.19)
    p_north = (600.0, 1200.0 - 100.0)   # 像素 y 减小 100 -> 应该是北
    g_north = lonlat_to_mercator(121.46, 31.19 + 100.0 * s / 111319.49)
    p_east = (600.0 + 100.0, 1200.0)
    g_east = lonlat_to_mercator(121.46 + 100.0 * s / 111319.49, 31.19)

    r = two_point_similarity(p_center, p_north, g_center, g_north)
    assert r is not None
    s2, th2, t2 = r
    out = _apply(np.array([p_east], dtype=float), s2, th2, t2)
    # 应当向东
    d_lon = mercator_to_lonlat(out[0][0], out[0][1])[0] - 121.46
    assert d_lon > 0, f"东向不对：lon 偏移 {d_lon:.6f}"
    print(f"  [OK] 朝向正确：东向偏移 lon +{d_lon:.6f}°")


def test_no_south_flip():
    """防回归：北向不能变成南向。

    这是本次实际踩到的 bug——矩阵约定不一致导致整条轨迹南北镜像，
    起点纬度与终点纬度正好互换，但所有数值看起来都「正常」。
    """
    p1 = (968.0, 1414.0)
    p2 = (292.0, 1188.0)
    # 起点在南、终点在北
    g1 = lonlat_to_mercator(121.45864, 31.19085)
    g2 = lonlat_to_mercator(121.37718, 31.21355)
    s, th, t = two_point_similarity(p1, p2, g1, g2)
    out = _apply(np.array([p1, p2], dtype=float), s, th, t)
    assert out[0][1] < out[1][1], (
        f"南北颠倒了：起点 y={out[0][1]:.0f} 终点 y={out[1][1]:.0f}")
    la1 = mercator_to_lonlat(out[0][0], out[0][1])[1]
    la2 = mercator_to_lonlat(out[1][0], out[1][1])[1]
    assert abs(la1 - 31.19085) < 1e-6, f"起点纬度错了：{la1}"
    assert abs(la2 - 31.21355) < 1e-6, f"终点纬度错了：{la2}"
    print(f"  [OK] 南北未颠倒：起点 lat {la1:.5f}，终点 lat {la2:.5f}")


if __name__ == "__main__":
    print("坐标系一致性测试：")
    test_two_point_roundtrip()
    test_registration_convention()
    test_orientation()
    test_no_south_flip()
    print("\n全部通过")
