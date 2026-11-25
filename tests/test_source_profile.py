"""输入源画像（Source Profile）契约测试。

背景（2026-11-14 演化决策）：识别链路的下游（拼接 / 模板 / 地理编码）
与截图来源无关，上游视觉参数与 OCR 文案词表按来源绑定。本文件锁定
三件事：

1. 默认画像必须是 ``meituan`` 且参数值与模块化之前完全一致（行为零
   变化是本次重构的硬要求）；
2. ``SOURCE_PROFILE`` 是来源选择的**唯一**环境变量（遵守「同一功能
   一个环境变量」约定），未知 id 必须报错并列出可用值，绝不静默回退；
3. OCR 里程词表跟随画像：切换画像后 ``extract_distance_km`` 行为
   立即改变，无需重启或重建 pipeline。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import CONFIG
from app.input_quality import extract_distance_km
from app.pipeline import _base_diagnostics
from app.source_profile import (
    DEFAULT_PROFILE_ID,
    MEITUAN,
    PROFILES,
    SourceProfile,
    resolve_source_profile,
)


def _items(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=t) for t in texts]


def test_default_profile_is_meituan_with_legacy_values() -> None:
    src = resolve_source_profile(None)
    assert src.id == "meituan"
    assert src.title == "美团骑行"
    # 视觉参数 = 模块化之前 VisionConfig 的默认值（逐一锁定，防漂移）
    v = src.vision
    assert v.start_hsv_range == (35, 90, 90, 85, 255, 255)
    assert v.end_hsv_range == (0, 90, 90, 12, 255, 255)
    assert v.end_hsv_range2 == (170, 90, 90, 179, 255, 255)
    assert (v.track_max_value, v.track_max_saturation) == (200, 90)
    # 美团标定样本短边 1260（分辨率归一化目标）
    assert v.vision_ref_short_side == 1260
    assert v.ransac_inlier_m == 200.0
    # 里程词表 = 原 input_quality 模块常量（原样迁移，一字不差）
    assert src.distance_positive == (
        "里程", "本次", "骑行", "行程", "全程", "共计", "合计", "距离")
    assert src.distance_negative == (
        "免费", "累计", "套餐", "赠送", "优惠", "已骑", "封顶")


def test_config_source_matches_default_profile() -> None:
    assert CONFIG.source_profile_id == DEFAULT_PROFILE_ID
    assert CONFIG.source.id == "meituan"
    assert CONFIG.source.vision is CONFIG.vision or (
        CONFIG.source.vision.start_hsv_range == CONFIG.vision.start_hsv_range)


def test_unknown_profile_id_raises_with_available_values() -> None:
    with pytest.raises(ValueError) as exc:
        resolve_source_profile("keep")
    msg = str(exc.value)
    # 报错里列出可用值（含已注册的 meituan 与 amap）
    assert "keep" in msg and "meituan" in msg and "amap" in msg


def test_source_profile_env_switches_profile_and_keywords(monkeypatch) -> None:
    """SOURCE_PROFILE 是唯一开关：设置后画像与 OCR 词表立即跟随。

    用桩画像 ``keep``（避免遮蔽已注册的真实 amap 画像）。
    """
    fake = SourceProfile(
        id="keep", title="Keep", vision=MEITUAN.vision,
        distance_positive=("总里程",),
        distance_negative=("历史",),
        ride_labels=("总里程", "消耗", "减碳"),
    )
    monkeypatch.setitem(PROFILES, "keep", fake)
    monkeypatch.setenv("SOURCE_PROFILE", "keep")
    assert CONFIG.source_profile_id == "keep"
    assert CONFIG.source.id == "keep"
    # 词表跟随画像：「总里程」成为正面词；美团正面词不再提供优先级
    # （词表只影响优先级排序，裸数字兜底依然有效——与主流程语义一致）
    assert extract_distance_km(_items("总里程 12.3 公里")) == pytest.approx(12.3)
    assert extract_distance_km(_items("本次骑行 9.6 公里")) == pytest.approx(9.6)
    # amap 正面词命中时优先于裸数字
    assert extract_distance_km(
        _items("总里程 12.3 公里", "杂项 5.5 公里")) == pytest.approx(12.3)
    # 干扰词整块排除
    assert extract_distance_km(_items("历史骑行 100 公里")) is None


def test_meituan_keywords_unchanged_by_default(monkeypatch) -> None:
    """默认（无 SOURCE_PROFILE）行为与模块化之前完全一致。"""
    monkeypatch.delenv("SOURCE_PROFILE", raising=False)
    assert extract_distance_km(_items("本次骑行 9.6 公里")) == pytest.approx(9.6)
    assert extract_distance_km(_items("累计骑行 1024 公里")) is None
    # 干扰词整块排除：即使块里也有正面词
    assert extract_distance_km(_items("里程免费领取 3 公里")) is None
    # 无正面词的裸数字低优先级，与正面词命中共存时取后者
    assert extract_distance_km(_items("约 2 公里", "全程 8.8 公里")) == pytest.approx(8.8)


def test_base_diagnostics_carries_source_profile() -> None:
    dg = _base_diagnostics()
    assert dg["source"]["id"] == "meituan"
    assert dg["source"]["title"] == "美团骑行"
    # over 参数不被 source 键覆盖
    dg2 = _base_diagnostics(track_points=122)
    assert dg2["track_points"] == 122
    assert dg2["source"]["id"] == "meituan"


def test_profiles_registry_matches_registered_ids() -> None:
    """注册表内容锁定：meituan + amap + xingzhe + xiaomi。

    amap / xingzhe 按真实样本精标；xiaomi 是 2026-11-23 首张真实截图
    驱动的可测试初版。此处锁「结构完整性」——每个来源都必须显式注册，
    且不能因新增来源而静默删除既有画像。
    """
    assert set(PROFILES) == {"meituan", "amap", "xingzhe", "xiaomi"}


def test_amap_profile_initial_params_locked() -> None:
    """amap 参数锚点（2026-11-16 真实样本精标）：防无意识漂移。

    关键事实：高德轨迹是速度渐变线（红→橙→黄→绿），掩膜必须是
    多色带并集；终点红标（H≈0-7）与轨迹红段（H≥8）错开。
    """
    src = resolve_source_profile("amap")
    assert src.id == "amap"
    assert src.title == "高德运动"
    v = src.vision
    # 多色带：橙红黄主体 + 高饱和绿快速段（避开薄荷绿公园 S≈50-90）
    assert v.track_hsv_range == (
        (8, 150, 120, 30, 255, 255),
        (35, 180, 120, 70, 255, 255),
    )
    # 地图区裁剪：整屏截图下方 UI 卡片与轨迹同色系，拼接前逐图裁掉
    assert v.map_crop_v == (0.065, 0.44)
    # 终点徽章与轨迹红段分离；洋红端关闭（红心装饰误判源）
    assert v.end_hsv_range == (0, 110, 110, 7, 255, 255)
    assert v.end_hsv_range2 == (0, 0, 0, 0, 0, 0)
    # 2026-11-25 补测：标定样本 1260×2720，map_crop_v 裁剪后短边 1020
    assert v.vision_ref_short_side == 1020
    assert v.ransac_inlier_m == 200.0
    # 干扰词必须含累计类文案（个人页统计混入详情页是高德的主要干扰源）
    assert "累计" in src.distance_negative
    assert "里程" in src.distance_positive


def test_track_mask_multiband_union() -> None:
    """多色带掩膜 = 各色带并集：渐变轨迹的两个色相带都被提取。"""
    import cv2
    import numpy as np

    from app.track import build_track_mask

    img = np.full((60, 180, 3), 255, np.uint8)
    # 橙色段（左）：#FF8C1A 一族 → OpenCV H≈21, S≈235
    cv2.line(img, (5, 30), (55, 30), (26, 140, 255), 4)
    # 绿色段（中）：#00E63C 一族 → OpenCV H≈68, S=255（落在绿带 [35,70]）
    cv2.line(img, (65, 30), (115, 30), (60, 230, 0), 4)
    # 底图薄荷绿（右，S≈60）：必须被 S>=180 排除
    cv2.line(img, (125, 30), (175, 30), (176, 216, 168), 4)

    mask = build_track_mask(img, resolve_source_profile("amap").vision)
    assert int(mask[25:35, 5:55].sum()) > 0, "应提取橙色段"
    assert int(mask[25:35, 65:115].sum()) > 0, "应提取高饱和绿段"
    assert int(mask[25:35, 125:175].sum()) == 0, "低饱和薄荷绿应被排除"


def test_map_crop_region() -> None:
    """map_crop_v：画像配置时按比例裁出地图区，None 时原样返回。"""
    import numpy as np

    from app.config import VisionConfig
    from app.cv_utils import crop_map_region

    img = np.zeros((1000, 500, 3), np.uint8)
    assert crop_map_region(img, VisionConfig()) is img, "无配置时零变化"
    out = crop_map_region(img, VisionConfig(map_crop_v=(0.065, 0.44)))
    assert out.shape[0] == int(round(1000 * 0.44)) - int(round(1000 * 0.065))
    # 防御：比例配错（裁出的行数 < 16）不返回空图
    assert crop_map_region(img, VisionConfig(map_crop_v=(0.98, 0.99))) is img


def test_track_fragments_chained_without_badges() -> None:
    """碎片串接：被里程牌/接缝切断的轨迹应被串成一条链（高德场景）。

    无徽章画布上，旧逻辑只取最大段会丢 30-40% 路线；新逻辑从最大段
    出发按端点邻近串接，远处的孤立团块（底图色块）绝不并入。
    """
    import cv2
    import numpy as np

    from app.scale import Scale
    from app.track import _merge_track_fragments

    mask = np.zeros((80, 300), np.uint8)
    # 同一条水平线切成 3 段（间隔 20px，模拟里程牌白胶囊切断）
    mask[40:46, 10:90] = 1
    mask[40:46, 110:190] = 1
    mask[40:46, 210:290] = 1
    # 远处的孤立团块（模拟漏网底图色块）：必须不被串进来
    mask[5:25, 140:170] = 1

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    scale = Scale.from_image(mask, badge_d=40.0)
    comp, bbox = _merge_track_fragments(labels, stats, count, scale, [])

    assert comp is not None
    ys, xs = np.where(comp > 0)
    # 三段全部在链里：横跨 x≈10..290
    assert xs.min() <= 15 and xs.max() >= 285, "三段轨迹碎片应全部串接"
    # 团块没被串进来：y<30 区域应为空
    assert (ys < 30).sum() == 0, "孤立团块不得被串接"
    assert bbox[2] >= 270


def test_track_mask_range_mode_for_saturated_blue_track() -> None:
    """掩膜双模式行为：美团上限式提浅荧光绿；单色带区间式提高饱和亮蓝。

    高德系亮蓝（#1677FF 一族，S≈233）在上限式掩膜里因 S 超限被排除——
    这正是区间式掩膜存在的理由（amap 画像本身现用橙绿多色带，见
    ``test_track_mask_multiband_union``；此处用自定义配置锁单带模式）。
    """
    import cv2
    import numpy as np

    from app.config import VisionConfig
    from app.track import build_track_mask

    img = np.full((60, 120, 3), 255, np.uint8)
    # 亮蓝轨迹线（左半幅，BGR）
    cv2.line(img, (5, 30), (55, 30), (255, 119, 22), 4)
    # 浅荧光绿轨迹线（右半幅，V≈190<200 且 S≈80<90，命中美团上限式）
    cv2.line(img, (65, 30), (115, 30), (150, 190, 130), 4)

    meituan_mask = build_track_mask(img, resolve_source_profile("meituan").vision)
    blue_cfg = VisionConfig(track_hsv_range=(95, 120, 150, 120, 255, 255))
    blue_mask = build_track_mask(img, blue_cfg)

    assert int(meituan_mask[25:35, 65:115].sum()) > 0, "美团掩膜应提取浅荧光绿线"
    assert int(meituan_mask[25:35, 5:55].sum()) == 0, "上限式掩膜应排除高饱和亮蓝"
    assert int(blue_mask[25:35, 5:55].sum()) > 0, "单带区间式掩膜应提取亮蓝线"
    assert int(blue_mask[25:35, 65:115].sum()) == 0, "蓝色相带应排除浅荧光绿"
