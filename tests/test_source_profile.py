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
    assert (v.track_close_kernel, v.track_open_kernel) == (11, 5)
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
        resolve_source_profile("amap")
    msg = str(exc.value)
    assert "amap" in msg and "meituan" in msg  # 报错里列出可用值


def test_source_profile_env_switches_profile_and_keywords(monkeypatch) -> None:
    """SOURCE_PROFILE 是唯一开关：设置后画像与 OCR 词表立即跟随。"""
    fake = SourceProfile(
        id="amap", title="高德运动", vision=MEITUAN.vision,
        distance_positive=("总里程",),
        distance_negative=("历史",),
        ride_labels=("总里程", "消耗", "减碳"),
    )
    monkeypatch.setitem(PROFILES, "amap", fake)
    monkeypatch.setenv("SOURCE_PROFILE", "amap")
    assert CONFIG.source_profile_id == "amap"
    assert CONFIG.source.id == "amap"
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


def test_profiles_registry_has_no_unverified_placeholder() -> None:
    """注册表里只允许已标定的画像——防止「切换到盲参数」的隐患。"""
    assert set(PROFILES) == {"meituan"}
