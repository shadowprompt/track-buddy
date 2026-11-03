"""「缩放级别评估」的测试。

这个判断决定要不要告诉用户「这张图缩放太低，填路名或截大点的图」。
判错两个方向都有代价：

* **漏判**（该提示却没提示）-> 用户拿到一个错的结果还以为是准的；
* **误判**（不该提示却提示）-> 让用户白折腾（去截一张更大的图）。

实测数据（上海城市级截图 1260x2720，覆盖整个上海）：
  19 个地标，0 个街道级，11 个大范围（行政区 + 地铁线），
  每个锚点的「文字标注位置 -> 真实位置」偏差 800 米 ~ 16.7 公里，
  而整条轨迹只有 11 公里长——锚点偏移比轨迹还大，原理上不可解。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RidePipeline, _BROAD_KINDS, _EXPRESSWAY_MARKERS
from app.register import AnchorPair


class _Cfg:
    """_assess_zoom_level 不依赖配置，用一个空壳即可。"""


def _pair(name, kind):
    return AnchorPair(name=name, px=0.0, py=0.0, weight=1.0,
                      kind=kind, lon=121.4, lat=31.2)


def _assess(pairs, road_names=None):
    return RidePipeline._assess_zoom_level(_Cfg(), pairs, road_names)


def test_city_level_screenshot_flagged():
    """城市级截图：无街道级锚点 + 大范围地标占多数 -> 必须判为缩放过低。"""
    pairs = [
        _pair("静安区", "district"), _pair("徐汇区", "poi"),
        _pair("普陀区", "poi"), _pair("黄浦区", "poi"),
        _pair("长宁区", "poi"), _pair("闵行区", "poi"),
        _pair("3号线", "transit"), _pair("8号线", "transit"),
        _pair("12号线", "transit"), _pair("1号线", "transit"),
        _pair("上海西站", "station"), _pair("上海站", "station"),
        _pair("上海南站", "station"), _pair("外环高速", "road"),
        _pair("桃浦中央绿地", "poi"), _pair("西岸自然", "poi"),
        _pair("机场联络线", "transit"), _pair("4号线内圈", "poi"),
        _pair("锦江乐园", "poi"),
    ]
    r = _assess(pairs)
    assert r["zoom_too_low"], "城市级截图未被判定为缩放过低"
    assert r["n_street_anchors"] == 0, "高速公路不该算街道级锚点"
    assert r["n_broad_anchors"] >= 10, f"大范围地标数不对：{r}"
    assert r["broad_ratio"] >= 0.45, f"大范围占比不对：{r}"
    print(f"  [OK] 城市级截图被判为缩放过低"
          f"（{r['n_anchor_names']} 地标 / 街道级 {r['n_street_anchors']} /"
          f" 大范围 {r['n_broad_anchors']}）")


def test_street_level_not_flagged():
    """街道级截图：有普通道路锚点 -> 不能判为缩放过低。"""
    pairs = [
        _pair("小木桥路", "road"), _pair("龙华中路", "road"),
        _pair("船厂路", "road"), _pair("中山南二路", "road"),
        _pair("东安路", "road"), _pair("肇嘉浜路", "road"),
        _pair("虹桥路", "road"), _pair("虹古路", "road"),
        _pair("安龙新村", "poi"),
    ]
    r = _assess(pairs)
    assert not r["zoom_too_low"], "街道级截图被误判为缩放过低"
    assert r["n_street_anchors"] >= 8, f"街道级锚点数不对：{r}"
    print(f"  [OK] 街道级截图未被误判（街道级锚点 {r['n_street_anchors']} 个）")


def test_user_road_names_disable_flag():
    """用户填了路名 -> 即使没有街道级锚点也不该提示缩放过低。

    理由：路名约束提供了街道级信息（起终点锚定 + 路名链），
    定位是可靠的，不该再让用户去折腾截图。
    """
    pairs = [_pair(n, k) for n, k in [
        ("静安区", "district"), ("3号线", "transit"),
        ("上海西站", "station"), ("8号线", "transit"),
        ("1号线", "transit"), ("12号线", "transit"),
    ]]
    r = _assess(pairs, ["小木桥路", "龙华中路"])
    assert not r["zoom_too_low"], "填了路名仍被判为缩放过低"
    print("  [OK] 用户填了路名后不再提示缩放过低")


def test_expressway_not_street_level():
    """高速公路 / 国道 / 高架 都算大对象，不能算街道级锚点。"""
    assert "高速" in _EXPRESSWAY_MARKERS
    assert "高架" in _EXPRESSWAY_MARKERS
    pairs = [_pair(n, "road") for n in
             ("外环高速", "内环高架", "G2 京沪高速", "沪宁高速")]
    r = _assess(pairs)
    assert r["n_street_anchors"] == 0, f"高速被算成街道级：{r}"
    print("  [OK] 高速公路类不算街道级锚点")


def test_too_few_anchors_not_flagged():
    """锚点太少时不该判缩放过低——那是「识别不出地标」，不是「缩放低」。"""
    r = _assess([_pair("静安区", "district"), _pair("3号线", "transit")])
    assert not r["zoom_too_low"], "锚点过少被误判为缩放过低"
    print("  [OK] 锚点过少不判缩放过低")


if __name__ == "__main__":
    print("缩放级别评估测试：")
    test_city_level_screenshot_flagged()
    test_street_level_not_flagged()
    test_user_road_names_disable_flag()
    test_expressway_not_street_level()
    test_too_few_anchors_not_flagged()
    print("\n全部通过")
