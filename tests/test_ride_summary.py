"""运动概要解析（ride_summary）契约测试。

样例值来自用户 2026-11-14 提供的真实美团订单详情页截图：
3.5km 本次骑行、骑行20分35秒、203kcal 本次消耗、172g 本次减碳、
2026-11-11 20:20:26，页面上还有「1314km」累计里程与「已省 7039.6 元」
等干扰文本。解析必须在这些干扰下取对「本次」值。
"""
from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.ride_summary import parse_ride_summary, summary_properties

# 模拟 OCR 按阅读顺序输出的订单详情页文本块（横排卡片 + 下方行程行）
ORDER_PAGE_TOKENS = [
    "订单详情",
    "卡路里粉碎机",
    "1314km",        # 累计里程（干扰）
    "203kcal",
    "本次消耗",
    "172g",
    "本次减碳",
    "3.5km",
    "本次骑行",
    "行程免费",
    "2026-11-11 20:20:26",
    "骑行20分35秒",
    "已省 7039.6 元，该卡截至本单省 149.5 元",  # 运营文案（干扰）
]

MEITUAN_LABELS = ("本次骑行", "本次消耗", "本次减碳")


def _items(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=t) for t in texts]


def test_parse_meituan_order_page_full_fields() -> None:
    out = parse_ride_summary(_items(*ORDER_PAGE_TOKENS),
                             ride_labels=MEITUAN_LABELS)
    assert out["available"] is True
    assert out["distance_km"] == pytest.approx(3.5)
    assert out["duration_s"] == 20 * 60 + 35
    assert out["duration_text"] == "骑行20分35秒"
    assert out["calories_kcal"] == pytest.approx(203)
    assert out["carbon_g"] == pytest.approx(172)
    assert out["started_at"] == "2026-11-11T20:20:26"
    assert set(out["fields_found"]) == {
        "distance_km", "duration_s", "calories_kcal", "carbon_g", "started_at"}


def test_distance_binds_label_neighbor_not_cumulative() -> None:
    # 候选 [1314km, 3.5km]：「本次骑行」标签近邻必须选中 3.5
    out = parse_ride_summary(_items(*ORDER_PAGE_TOKENS),
                             ride_labels=MEITUAN_LABELS)
    assert out["distance_km"] == pytest.approx(3.5)
    assert out["distance_label"] == "本次骑行"
    assert any("2 个公里数候选" in n for n in out["notes"])


def test_distance_without_label_falls_back_to_min() -> None:
    out = parse_ride_summary(
        _items("1314km", "3.5km", "2026-11-11 20:20:26"),
        ride_labels=MEITUAN_LABELS)
    assert out["distance_km"] == pytest.approx(3.5)  # 无标签取最小（累计>本次）


def test_timestamp_survives_ocr_glued_separator() -> None:
    """回归锁定（2026-11-14 真实落盘数据）：OCR 吞掉日期与时间之间的
    空格——「2026-11-11 20:20:26骑行20分35秒」粘连成
    「2026-11-1120:20:26骑行20分35秒」——旧正则要求分隔符必选，
    started_at 解析为 null，整条 GPX 时间轴跟着消失。"""
    out = parse_ride_summary(
        _items("2026-11-1120:20:26骑行20分35秒"),
        ride_labels=MEITUAN_LABELS)
    assert out["started_at"] == "2026-11-11T20:20:26"
    assert out["duration_s"] == 20 * 60 + 35
    assert "started_at" in out["fields_found"]


def test_timestamp_with_space_still_parsed() -> None:
    out = parse_ride_summary(
        _items("2026-11-11 20:20:26"), ride_labels=MEITUAN_LABELS)
    assert out["started_at"] == "2026-11-11T20:20:26"


def test_timestamp_range_validation_rejects_garbled_values() -> None:
    """粘连错位读出 92 时 90 分之类的胡值必须丢弃，宁缺勿错。"""
    out = parse_ride_summary(
        _items("2026-11-1192:90:26骑行20分35秒"),
        ride_labels=MEITUAN_LABELS)
    assert out["started_at"] is None
    assert out["duration_s"] == 20 * 60 + 35


def test_duration_without_start_warns_about_missing_gpx_time_axis() -> None:
    out = parse_ride_summary(
        _items("骑行20分35秒"), ride_labels=MEITUAN_LABELS)
    assert out["started_at"] is None
    assert out["duration_s"] == 20 * 60 + 35
    assert any("GPX" in n and "时间轴" in n for n in out["notes"])


def test_carbon_requires_label_neighbor() -> None:
    # g 候选存在但无「减碳」标签近邻 → 宁缺勿错
    out = parse_ride_summary(_items("172g", "行程免费", "203kcal"),
                             ride_labels=MEITUAN_LABELS)
    assert out["carbon_g"] is None
    assert any("宁缺勿错" in n for n in out["notes"])


def test_duration_with_hours() -> None:
    out = parse_ride_summary(_items("骑行1小时5分10秒"),
                             ride_labels=MEITUAN_LABELS)
    assert out["duration_s"] == 3600 + 5 * 60 + 10


def test_cross_check_consistent_and_inconsistent() -> None:
    ok = parse_ride_summary(_items(*ORDER_PAGE_TOKENS),
                            geom_distance_m=3406.1,
                            ride_labels=MEITUAN_LABELS)
    assert ok["cross_check"]["available"] is True
    assert ok["cross_check"]["consistent"] is True

    bad = parse_ride_summary(_items(*ORDER_PAGE_TOKENS),
                             geom_distance_m=10000.0,
                             ride_labels=MEITUAN_LABELS)
    assert bad["cross_check"]["consistent"] is False
    assert "量级不符" in bad["cross_check"]["note"]


def test_empty_tokens_reports_unavailable() -> None:
    out = parse_ride_summary([], ride_labels=MEITUAN_LABELS)
    assert out["available"] is False
    assert "未识别到任何概要字段" in out["notes"][0]


def test_labels_default_from_source_profile(monkeypatch) -> None:
    """不传 ride_labels 时从 CONFIG.source 取（当前=meituan）。"""
    out = parse_ride_summary(_items(*ORDER_PAGE_TOKENS))
    assert out["distance_km"] == pytest.approx(3.5)


def test_summary_properties_only_non_null() -> None:
    out = parse_ride_summary(_items(*ORDER_PAGE_TOKENS),
                             ride_labels=MEITUAN_LABELS)
    props = summary_properties(out)
    assert props["distance_km"] == pytest.approx(3.5)
    assert props["started_at"] == "2026-11-11T20:20:26"
    # 没有的字段不出现在 properties 里
    out2 = parse_ride_summary(_items("3.5km"), ride_labels=MEITUAN_LABELS)
    props2 = summary_properties(out2)
    assert "calories_kcal" not in props2
    assert props2["distance_km"] == pytest.approx(3.5)


# --- server 集成：_attach_ride_summary 直测（真 jpg + 打桩 OCR） ---

def _tiny_jpg() -> bytes:
    img = np.full((40, 200, 3), 250, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _fake_ocr(img, cfg):
    return _items(*ORDER_PAGE_TOKENS)


def _gj_payload() -> dict:
    feat = {"type": "Feature",
            "properties": {"crs": "GCJ-02"},
            "geometry": {"type": "LineString", "coordinates": [[121.3, 31.2]]}}
    return {"distance_m": 3406.1, "warnings": [],
            "geojson_gcj02": {"type": "FeatureCollection", "features": [feat]}}


def _make_upload(data: bytes) -> SimpleNamespace:
    from fastapi import UploadFile
    return UploadFile(filename="order.jpg", file=io.BytesIO(data))


def test_attach_ride_summary_merges_payload_and_geojson(monkeypatch) -> None:
    from app import server as S

    monkeypatch.setattr("app.ocr_anchor.ocr_texts", _fake_ocr)
    payload = _gj_payload()
    asyncio.run(S._attach_ride_summary(payload, _make_upload(_tiny_jpg())))
    rs = payload["ride_summary"]
    assert rs["available"] is True
    assert rs["distance_km"] == pytest.approx(3.5)
    assert rs["cross_check"]["consistent"] is True
    # GeoJSON properties 合并（GCJ 版本）
    props = payload["geojson_gcj02"]["features"][0]["properties"]
    assert props["ride_summary"]["distance_km"] == pytest.approx(3.5)
    assert props["ride_summary"]["started_at"] == "2026-11-11T20:20:26"
    assert payload["warnings"] == []


def test_attach_ride_summary_skips_when_absent(monkeypatch) -> None:
    from app import server as S

    payload = _gj_payload()
    asyncio.run(S._attach_ride_summary(payload, None))
    assert "ride_summary" not in payload
    # 坏图：告警但绝不抛异常
    payload2 = _gj_payload()
    asyncio.run(S._attach_ride_summary(
        payload2, _make_upload(b"not-an-image")))
    assert "ride_summary" not in payload2
    assert any("无法解码" in w for w in payload2["warnings"])


def test_recognize_endpoints_accept_summary_form() -> None:
    """两个识别端点的 Form 契约：可选 summary 文件 + 落盘前并入。"""
    import inspect

    from app import server as S

    for fn in (S.recognize, S.recognize_multi):
        src = inspect.getsource(fn)
        assert "summary: UploadFile | None = File(None)" in src
        assert "await _attach_ride_summary(" in src
        # 记录存 COS 时还需要把概要图字节交给 byte_sink（不落盘）
        assert "byte_sink=summary_sink if record_uid else None" in src
        # 概要必须在落盘之前并入（否则对照页看不到）
        assert src.index("_attach_ride_summary") < src.index("_persist_last_result")


# --- 高德运动详情页（2026-11-16 真实截图 OCR 实测） ---
# 页面特征：上半幅地图（含 3KM/2KM/1KM 里程牌、比例尺「1公里」、
# 状态栏时钟「08:34」），下半幅概要卡片：骑行 3.68 干米（OCR 把
# 「千」误读成「干」，且数字与单位是两个文本块）、00:14:14、79大卡、
# 运动用时/平均速度/消耗热量标签，日期点分且与时间粘连。
def _boxed(*rows: list[str]) -> list[SimpleNamespace]:
    """按行构造带几何框的 OCR 块（还原真实 TextItem 布局）。

    每行内块从左到右排，行间留出足够间距（不同行绝不共线）。
    """
    items: list[SimpleNamespace] = []
    for r, row in enumerate(rows):
        x = 40.0
        y = 100.0 + r * 90.0
        for tok in row:
            w = 30.0 + 18.0 * len(tok)
            items.append(SimpleNamespace(
                text=tok, score=0.9,
                x0=x, y0=y, x1=x + w, y1=y + 44.0,
                cx=x + w / 2, cy=y + 22.0, width=w, height=44.0))
            x += w + 24.0
    return items


def _amap_page() -> list[SimpleNamespace]:
    """高德运动详情页真实布局（2026-11-16 截图 OCR 实测）。

    地图区（上）：状态栏、里程牌、比例尺；概要卡片区（下）：
    「骑行 3.68 干米 | 00:14:14 | 79大卡」一行，标签行在其下。
    """
    return _boxed(
        ["08:34", "骑行结束"],
        ["2026.11.1608:13:27"],
        ["3KM"],
        ["2KM"],
        ["1KM"],
        ["1公里"],
        ["骑行", "3.68", "干米"],
        ["最慢", "最快"],
        ["00:14:14", "15千米/时", "79大卡"],
        ["运动用时", "平均速度", "消耗热量"],
        ["分享运动记录"],
    )


AMAP_LABELS = ("骑行", "消耗热量")


def test_parse_amap_detail_page_full_fields() -> None:
    out = parse_ride_summary(_amap_page(), ride_labels=AMAP_LABELS)
    assert out["available"] is True
    assert out["started_at"] == "2026-11-16T08:13:27"
    assert out["duration_s"] == 14 * 60 + 14
    assert out["duration_text"] == "00:14:14"
    assert out["calories_kcal"] == pytest.approx(79)
    # 里程必须绑定「骑行」标签的 3.68，而不是里程牌/比例尺的 3/2/1
    assert out["distance_km"] == pytest.approx(3.68)
    assert out["distance_label"] == "骑行"
    # 高德骑行页没有减碳字段，宁缺勿错
    assert out["carbon_g"] is None


def test_amap_distance_ignores_milestones_and_speed() -> None:
    """干扰隔离：里程牌 3/2/1 KM、比例尺 1公里、配速 15千米/时 都
    不能成为距离候选的选中值；状态栏时钟不能成为时长。"""
    out = parse_ride_summary(_amap_page(), ride_labels=AMAP_LABELS)
    assert out["distance_km"] != 3.0 and out["distance_km"] != 1.0
    assert out["duration_s"] == 14 * 60 + 14  # 不是 08:34 → 514s


def test_timestamp_separators_dot_and_slash() -> None:
    """时间戳分隔符族谱：点分（高德）与横杠（美团）都要吃。"""
    out = parse_ride_summary(_items("2026.11.1608:13:27"))
    assert out["started_at"] == "2026-11-16T08:13:27"
    out2 = parse_ride_summary(_items("2026/11/16 08:13:27"))
    assert out2["started_at"] == "2026-11-16T08:13:27"
