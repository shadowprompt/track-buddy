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
