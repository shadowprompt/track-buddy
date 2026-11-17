"""行者（xingzhe）来源画像的轨迹提取回归测试（2026-11-17 真实样本标定）。

样本：宁波 34.67km 骑行，卫星底图 + 橙红轨迹 + 蓝色规划线 + 红色起点旗 /
白底红字终点水滴。这是与美团/高德都不同的第三类版式，锁定三件事：

1. 画像注册且参数正确（掩膜橙红色带、徽章填充率筛选）；
2. 轨迹掩膜干净：单一连通域、像素数在合理区间（不被卫星底图污染）；
3. 徽章定位准（起点红旗、终点红字）、折线覆盖全程（环形不漏段）。
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


def test_xingzhe_profile_registered() -> None:
    assert "xingzhe" in PROFILES
    p = resolve_source_profile("xingzhe")
    assert p is XINGZHE
    assert p.title == "行者"
    # 轨迹掩膜用区间式橙红色带（非美团的上限式）
    assert XINGZHE.vision.track_hsv_range is not None
    # 徽章靠填充率从同色轨迹里分离
    assert XINGZHE.vision.badge_min_fill > 0


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
    # 实测约 1.4 万像素；区间放宽到 8000-25000 容纳轻微标定漂移，
    # 同时远低于「卫星底图被整张吞进来」的 57 万（回归 8bd 之前的 bug）。
    assert 8000 < fg < 25000, f"掩膜像素异常：{fg}"
    n, _, stats, _ = cv2.connectedComponentsWithStats(r.mask, 8)
    areas = sorted((int(stats[i, 4]) for i in range(1, n)), reverse=True)
    assert areas, "掩膜为空"
    # 主连通域占绝大多数（单一轨迹，不是散落噪声）
    assert areas[0] / fg > 0.85


def test_badges_located(img: np.ndarray) -> None:
    r = extract_track(img, XINGZHE.vision)
    assert r.start_marker is not None, "起点红旗未检测到"
    assert r.end_marker is not None, "终点红字未检测到"
    sx, sy = r.start_marker
    ex, ey = r.end_marker
    # 实测起点 (42,315)、终点 (334,1079)，容许小幅漂移
    assert abs(sx - 42) < 25 and abs(sy - 315) < 25, f"起点位置异常：{r.start_marker}"
    assert abs(ex - 334) < 30 and abs(ey - 1079) < 30, f"终点位置异常：{r.end_marker}"


def test_polyline_covers_full_loop(img: np.ndarray) -> None:
    r = extract_track(img, XINGZHE.vision)
    assert len(r.polyline) > 800, f"折线点数过少：{len(r.polyline)}"
    arc = float(np.linalg.norm(np.diff(r.polyline, axis=0), axis=1).sum())
    # 实测约 1960px；环形全程必须 >1500，低于此说明只走了半环（图搜索
    # 抄近道的 bug：曾得到 976 点 / 1145px，漏掉左半环）。
    assert arc > 1500, f"折线过短，疑似漏段：{arc:.0f}px"
    # 首尾接近（环形特征）：闭合度小
    straight = float(np.linalg.norm(r.polyline[-1] - r.polyline[0]))
    assert straight / arc < 0.3, "不像环形轨迹"
