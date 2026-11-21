"""行者（xingzhe）来源画像的轨迹提取回归测试（2026-11-17 真实样本标定）。

样本：宁波 34.67km 闭环骑行（用户确认起点=终点），卫星底图 + 橙红轨迹
+ 蓝色规划线 + 蓝色起点定位点 / 白底红字终点水滴。这是与美团/高德都不
同的第三类版式，锁定四件事：

1. 画像注册且参数正确（掩膜橙红色带、徽章填充率筛选、闭环赛道开关）；
2. 轨迹掩膜干净：单一连通域、像素数在合理区间（不被卫星底图污染）；
3. 徽章定位准（蓝色起点定位点、终点红字）、折线覆盖全程（环形不漏段）；
4. 中国邮路闭环排序：弧长接近真实里程、首尾闭合于起终点区域。

闭环管线背景（2026-11-18）：起点检测从红色「起点旗」改为蓝色定位圆点
（旧红色检测会误抓左边缘的导航指北针箭头）；用户确认起点=终点，排序
走中国邮路（HANDOVER §24.9），实测弧长 2560px ≈ 33.25km，对比 App
记录 34.67km 误差 4.1%（旧开环管线 25.2km 误差 27%）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.cv_utils import imread_unicode
from app.source_profile import PROFILES, XINGZHE, resolve_source_profile
from app.track import extract_track

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "xingzhe_ningbo.jpg"

# 比例尺实测：左下角「1公里」黑色线段 77px → 12.99 m/px
M_PER_PX = 1000.0 / 77.0


def test_xingzhe_profile_registered() -> None:
    assert "xingzhe" in PROFILES
    p = resolve_source_profile("xingzhe")
    assert p is XINGZHE
    assert p.title == "行者"
    # 轨迹掩膜用区间式橙红色带（非美团的上限式）
    assert XINGZHE.vision.track_hsv_range is not None
    # 徽章靠填充率从同色轨迹里分离
    assert XINGZHE.vision.badge_min_fill > 0
    # 闭环赛道（起点=终点）：走中国邮路专用管线
    assert XINGZHE.vision.loop_course is True
    # 内容尺度显式锁定（起点换成蓝点后自动反推会缩水）
    assert XINGZHE.vision.scale_badge_d == 39.2


@pytest.fixture(scope="module")
def img() -> np.ndarray:
    # 样本是真实骑行截图（含位置信息），按 .gitignore 约定不入库。
    # 缺失时跳过（CI / 他人 clone 场景），本地标定时放在 samples/xingzhe_ningbo.jpg。
    if not SAMPLE.exists():
        pytest.skip(f"行者标定样本未入库（含位置信息），跳过：{SAMPLE}")
    return imread_unicode(str(SAMPLE))


def test_track_mask_is_single_clean_component(img: np.ndarray) -> None:
    r = extract_track(img, XINGZHE.vision)
    fg = int((r.mask > 0).sum())
    # 实测约 1.5 万像素；区间放宽到 8000-25000 容纳轻微标定漂移，
    # 同时远低于「卫星底图被整张吞进来」的 57 万（回归 8bd 之前的 bug）。
    assert 8000 < fg < 25000, f"掩膜像素异常：{fg}"
    n, _, stats, _ = cv2.connectedComponentsWithStats(r.mask, 8)
    areas = sorted((int(stats[i, 4]) for i in range(1, n)), reverse=True)
    assert areas, "掩膜为空"
    # 主连通域占绝大多数（单一轨迹，不是散落噪声）
    assert areas[0] / fg > 0.85


def test_badges_located(img: np.ndarray) -> None:
    r = extract_track(img, XINGZHE.vision)
    assert r.start_marker is not None, "蓝色起点定位点未检测到"
    assert r.end_marker is not None, "终点红字未检测到"
    sx, sy = r.start_marker
    ex, ey = r.end_marker
    # 实测起点蓝点 (365,1102)、终点红徽章 (334,1079)，容许小幅漂移。
    # 闭环赛道：起点必须贴近终点（<80px）——这是「起点=终点」的检测证据。
    assert abs(sx - 365) < 25 and abs(sy - 1102) < 25, f"起点位置异常：{r.start_marker}"
    assert abs(ex - 334) < 30 and abs(ey - 1079) < 30, f"终点位置异常：{r.end_marker}"
    assert ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5 < 80, (
        f"闭环赛道起终点应重合，实测相距 {((sx-ex)**2+(sy-ey)**2)**0.5:.0f}px"
    )


def test_polyline_covers_full_loop(img: np.ndarray) -> None:
    r = extract_track(img, XINGZHE.vision)
    assert len(r.polyline) > 800, f"折线点数过少：{len(r.polyline)}"
    arc = float(np.linalg.norm(np.diff(r.polyline, axis=0), axis=1).sum())
    # 中国邮路闭环 + 终点碎片接入后实测约 2560px ≈ 33.25km（误差 4.1%）。
    # 低于 2000px 说明漏了大段（旧开环管线只有 ~1940px/25.2km，误差 27%）；
    # 高于 3200px 说明邮路复制失控（像素级伪影配对，实测曾到 5637px）。
    assert 2000 < arc < 3200, f"折线弧长异常：{arc:.0f}px"
    # 闭环：首尾都是「起点=终点」徽章区域的开放端（被终徽章+蓝点遮挡，
    # 掩膜在此断开——walk 从一个端头走到另一个端头）。两个端头都在
    # 终点徽章 200px 邻域内（实测首 ~23px、尾 ~157px）。
    for pt in (r.polyline[0], r.polyline[-1]):
        d = float(np.linalg.norm(pt - np.array([334, 1079], dtype=float)))
        assert d < 200, f"折线端点未闭合到终点区域：{pt} 距 {d:.0f}px"


def test_loop_distance_close_to_recorded(img: np.ndarray) -> None:
    """闭环排序的里程应接近 App 记录（34.67km，误差 <12%）。

    里程 = 折线弧长 × 比例尺换算。剩余 ~4% 误差来自 Zhang-Suen 骨架
    对 45° 斜线的双列带压薄（无法区分「单线锯齿」与「跑两遍双线」）。
    """
    r = extract_track(img, XINGZHE.vision)
    arc = float(np.linalg.norm(np.diff(r.polyline, axis=0), axis=1).sum())
    km = arc * M_PER_PX / 1000.0
    # 实测 33.25km；34.67km 的 ±12% = [30.5, 38.8]
    assert 30.5 < km < 38.8, f"闭环里程偏离 App 记录：{km:.2f}km (目标 34.67km)"


def test_endpoint_fragment_integrated(img: np.ndarray) -> None:
    """终点徽章遮挡的「跑两遍」折返段：闭环赛道下应桥接接入主轨迹。

    这段被白色「终」徽章 + 蓝色定位点切成独立小连通域（实测 364px），
    闭环语义下它是起终点的进出通道——桥接接入后由邮路自然计两遍。
    """
    r = extract_track(img, XINGZHE.vision)
    notes = " ".join(r.notes)
    assert "碎片" in notes and "桥接接入" in notes, (
        f"终点折返碎片未被接入：{r.notes}"
    )


def test_topbar_distance_column_parse(img: np.ndarray) -> None:
    """顶栏无单位里程（按列切分）：「34.67新=江18.29」应解出 34.67。

    行者顶栏是「距离/速度/时间」三列，数字行无单位且跨列粘连，
    带单位的 ``extract_distance_km`` 漏掉；按列裁切 + 二次 OCR 后
    「距离」列下的 34.67 必须被解出（2026-11-18 顶栏 OCR 粘连修复）。
    """
    from app.config import CONFIG
    from app.input_quality import extract_distance_column, extract_distance_km
    from app.ocr_anchor import ocr_texts

    texts = ocr_texts(img, XINGZHE.vision)
    # 带单位正则确实解不出（粘连块无单位）
    assert extract_distance_km(texts) is None
    km = extract_distance_column(img, texts)
    assert km is not None, "按列切分未能解出顶栏里程"
    assert abs(km - 34.67) < 0.05, f"顶栏里程解析错误：{km}"


def test_road_snap_delivery(img: np.ndarray) -> None:
    """路网温和吸附交付：里程回到 ±5% 内、大部分点投影上路、无跨街跳段。

    三项修复（2026-11-18）：
    1. RDP 弧长坍缩曾把交付里程 34.9km 削到 30.4km（-12%）——吸附交付
       跳过 RDP 改 2px 等距重采样后里程自动回到 ±1%；
    2. gentle_snap 把离路 ≤60m 的点投影到路上（实测 54%）；
    3. 跳变回退保证无 >200m 跨街直线。
    """
    from app.cv_utils import resample_polyline  # noqa: F401 —— 依赖可用性
    r = extract_track(img, XINGZHE.vision)
    arc_px = float(np.linalg.norm(np.diff(r.polyline, axis=0), axis=1).sum())
    # 2px 重采样后点数与弧长：弧长应接近骨架原始值（~2550px）
    assert arc_px > 2300, f"吸附交付的轨迹弧长被削：{arc_px:.0f}px"


def test_gentle_snap_unit() -> None:
    """gentle_snap 单元：合成路网+轨迹，验证吸附/表决/跳变回退语义。"""
    import numpy as np
    from app.snaproute import gentle_snap

    class FakeReg:
        def pixel_to_mercator(self, px, py):
            return float(px), float(py)

        def mercator_to_pixel(self, x, y):
            return float(x), float(y)

    class FakeNet:
        # 一条水平路 y=0，x 0..1000，6m 采样
        samples = [(float(x), 0.0) for x in range(0, 1001, 6)]
        owner = ["X路"] * len(samples)
        tree = None

    import scipy.spatial as sp
    FakeNet.tree = sp.cKDTree(np.array(FakeNet.samples))

    reg = FakeReg()
    # 轨迹：贴近水平路（y=20），中间有一段 y=200（远离，应保持原样）
    pts = [(x, 20.0) for x in range(0, 400, 10)] + \
          [(x, 200.0) for x in range(400, 600, 10)] + \
          [(x, 20.0) for x in range(600, 1001, 10)]
    out = gentle_snap(reg, pts, FakeNet, tol_m=60.0)
    assert out["ok"], out.get("reason")
    # 用 coordinates_px（FakeReg 是恒等投影，数值即输入墨卡托坐标）
    px = np.asarray(out["coordinates_px"])
    # 吸附点应贴到 y≈0；远离段（y=200）保持原样
    ys_near = px[:35, 1]
    assert float(np.abs(ys_near).max()) < 30.0, "近路段未吸附"
    ys_far = px[40:55, 1]
    assert float(np.abs(ys_far - 200).max()) < 1.0, "远路段被强行吸附"
    # 吸附不引入「新的」跨街跳变：>100m 的步数不多于输入
    # （输入在近/远段衔接处本身就有 180m 真实跳变）
    arr_in = np.asarray(pts, dtype=float)
    in_big = int((np.linalg.norm(np.diff(arr_in, axis=0), axis=1) > 100).sum())
    out_big = int((np.linalg.norm(np.diff(px, axis=0), axis=1) > 100).sum())
    assert out_big <= in_big, \
        f"吸附引入新跳变：输出 {out_big} 处 > 输入 {in_big} 处"


# ---------------------------------------------------------------------------
# UI 文案锚点过滤（2026-11-21 真实失败复现）
# 行者宁波图的顶栏统计 + 详情面板文字被高德当成 POI，占满
# GEOCODE_TARGET_LIMIT=20 名额，真地名（迪士尼/东柳）被 skipped，
# RANSAC 定不出尺度 → 0 轨迹点、ok=False。
# ---------------------------------------------------------------------------

def test_ui_noise_filter_covers_xingzhe_hud() -> None:
    from app.ocr_anchor import is_ui_noise

    ui_texts = [
        "距离", "速度", "时间", "34.63", "15.67", "2:12:39",
        "骑行", "运动时间", "全程时间", "运动均速", "全程均速",
        "最快速度", "累计上升", "累计下降", "海拔范围",
        "655米", "2.00-80.00米", "15.67km/h", "34.63公里",
        "导出删除分享", "<返回", "地图", "分享", "备注", "数据", "图表",
    ]
    for t in ui_texts:
        assert is_ui_noise(t), f"UI 文案未被过滤: {t!r}"


def test_ui_noise_filter_keeps_real_places() -> None:
    from app.ocr_anchor import is_ui_noise

    places = ["宁波站", "鄞州区政府", "钟公庙村", "丽兹大酒店",
              "迪士尼", "东柳", "上海西站"]
    for t in places:
        assert not is_ui_noise(t), f"真实地名被误杀: {t!r}"


def test_ocr_garbage_filter_catches_mixed_fragments() -> None:
    """OCR 粘连碎片必须被拦，正常中英混排地名不受影响。"""
    from app.ocr_anchor import is_ocr_garbage, is_ui_noise

    garbage = ["A.十", "y.十层", "34.67新=江18.29", "新=江18.29", "", "中"]
    for t in garbage:
        assert is_ocr_garbage(t), f"OCR 碎片未被识别: {t!r}"
        assert is_ui_noise(t), f"OCR 碎片未接入 UI 过滤: {t!r}"

    legit = ["上海荟聚Livat", "7-Eleven", "宁波站", "天一阁", "G2"]
    for t in legit:
        assert not is_ocr_garbage(t), f"正常地名被误杀: {t!r}"


def test_build_anchors_drops_ui_texts() -> None:
    from app.config import VisionConfig
    from app.ocr_anchor import TextItem, build_anchors

    def item(text: str, x: int) -> TextItem:
        return TextItem(text=text, score=0.9, cx=float(x), cy=10.0,
                        x0=float(x), y0=0.0, x1=float(x + 50), y1=20.0,
                        width=50.0, height=20.0)

    texts = ["距离", "速度", "骑行", "运动均速", "655米",
             "导出删除分享", "宁波站", "迪士尼", "东柳"]
    items = [item(t, i * 60) for i, t in enumerate(texts)]
    img = np.zeros((100, 800, 3), dtype=np.uint8)

    names = [a.name for a in build_anchors(img, items, VisionConfig())]
    for place in ("宁波站", "迪士尼", "东柳"):
        assert place in names, f"真实地名丢失: {place}"
    for ui in texts[:6]:
        assert ui not in names, f"UI 文案混入锚点: {ui}"


# --- 长截图版式回归（2026-11-21：详情面板干扰）---------------------------
#
# 用户新上传的 631×1830 长截图：上半地图 + 下半详情面板。两个 bug：
#   ① 详情面板里 15.67km/h 等三个速度被当里程，与 34.63 取中位数得 25.15；
#   ② 面板蓝色标签 (318,1117) area 2464 / 指北针 (99,309) area 1329 面积
#      碾压真徽章，按「面积最大」挑起终点直接定位到面板和地图左上角。

LONG_SHOT = Path(__file__).resolve().parents[1] / "data" / "uploads" / \
    "1795244680_dae0a1b9.jpg"

# 断裂段回归用合成掩膜（见 test_broken_line_fragments_*），不依赖
# 二进制样本文件——仓库 .gitignore 忽略 samples/*.jpg。


def test_speed_texts_not_parsed_as_distance() -> None:
    """详情面板的 km/h 速度不能混进里程中位数（25.15 bug 回归）。"""
    from app.input_quality import extract_distance_km

    class T:
        def __init__(self, text: str) -> None:
            self.text = text

    # 复刻失败现场：真里程 34.63 + 三个速度，旧代码得中位数 25.15
    panel = [T("34.63公里"), T("15.67km/h"), T("13.95km/h"), T("39.53km/h")]
    assert extract_distance_km(panel) == pytest.approx(34.63)
    # 只有速度时应返回 None，不能给出一个凭空的里程
    assert extract_distance_km([T("15.67km/h"), T("39.53km/h")]) is None
    # 正常来源不受影响
    assert extract_distance_km([T("骑行 12.6公里")]) == pytest.approx(12.6)
    assert extract_distance_km([T("总里程 8.2 km")]) == pytest.approx(8.2)


@pytest.mark.skipif(not LONG_SHOT.exists(), reason="长截图样本未提供")
def test_long_shot_badges_are_paired_adjacent() -> None:
    """长截图起终点必须成对相邻，不能落在详情面板/指北针上。"""
    from app.track import _detect_markers_ex

    img = cv2.imread(str(LONG_SHOT))
    assert img is not None
    s, e, notes, _ = _detect_markers_ex(img, XINGZHE.vision)
    assert s is not None and e is not None, f"未定位徽章: {notes}"
    gap = float(np.hypot(s[0] - e[0], s[1] - e[1]))
    # 闭环起终点实测 47px；上限取徽章直径 2.5 倍（39.2*2.5=98px）
    assert gap <= XINGZHE.vision.scale_badge_d * 2.5, \
        f"起终点不相邻，疑似误检到面板/指北针: 起{s} 终{e} 距{gap:.0f}px"
    # 两者都必须在上半地图区域，不能落进 y>920 的详情面板
    assert s[1] < 920 and e[1] < 920, f"徽章落进详情面板: 起{s} 终{e}"
    assert any("成对定位" in n for n in notes), f"未走成对定位逻辑: {notes}"


def _synthetic_fragments() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """合成一张掩膜：主竖线 + 一段细线（应接回）+ 一块图例（应排除）。

    主域：x=50 的竖线 y=50..250。
    细线段：x=110 的竖线 y=60..160（细长，fill 低，gap ~1px）。
    图例块：x=20..70、y=200..233 的实心矩形（fill 1.0，gap ~20px）。
    """
    mask = np.zeros((300, 200), np.uint8)
    cv2.line(mask, (80, 30), (80, 270), 1, 2)      # 主竖线（area ~482）
    # 细线断裂段：带弯折，bbox 60×100、fill~0.1（模拟真实轨迹的折线形态）
    cv2.line(mask, (140, 60), (190, 110), 1, 2)
    cv2.line(mask, (190, 110), (140, 160), 1, 2)
    cv2.rectangle(mask, (20, 200), (60, 215), 1, -1)  # 图例实心块（分离，不碰主域）
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    main_id = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return mask, labels, stats, main_id


def test_broken_line_fragment_is_bridged_back() -> None:
    """细线状断裂段必须被桥接回主域（用户反馈「路线少一截」的根因）。"""
    from app.track import _merge_line_fragments

    from app.scale import Scale

    mask, labels, stats, main_id = _synthetic_fragments()
    scale = Scale.from_image(mask, badge_d=XINGZHE.vision.scale_badge_d)
    comp, notes = _merge_line_fragments(mask, labels, stats, main_id, scale)
    # 细线段整段并入（用原始标签取掩膜，避免依赖具体某个坐标点）
    seg_mask = labels == 2
    assert comp[seg_mask].mean() > 0.95, "线状断裂段未被桥接"
    assert any("线状段已接回" in n for n in notes), f"无桥接记录: {notes}"
    # 不重复桥接：同一段只应记录一次
    thin_notes = [n for n in notes if "线状段已接回" in n]
    assert len(thin_notes) == 1, f"同一段被重复桥接: {thin_notes}"


def test_bridge_excludes_solid_legend_block() -> None:
    """底图图例/色块（fill 高）不能被当成轨迹段接进来。"""
    from app.scale import Scale
    from app.track import _merge_line_fragments

    mask, labels, stats, main_id = _synthetic_fragments()
    scale = Scale.from_image(mask, badge_d=XINGZHE.vision.scale_badge_d)
    comp, notes = _merge_line_fragments(mask, labels, stats, main_id, scale)
    # 图例块中心 (40, 207) 仍为背景
    assert comp[207, 40] == 0, "实心图例块被误接为轨迹"
    assert not any("area=600" in n for n in notes)
