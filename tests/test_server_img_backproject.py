"""API 响应层像素坐标反投影契约的回归测试。

事故：`image_size` 下发的是裁剪后 prepared 尺寸，前端却把轨迹画在
完整原图上——纵向比例 = 原图高 / 裁剪高（小米 1920/1037 ≈ 1.85×），
绿线纵向拉伸、环下坠、尾线冲出起终点标记（2026-11-25 用户实测）。

修复：`server._to_original_img_px` 在两个 payload 出口统一把
`track_polyline_img_px` / 锚点 `px,py` / `candidate_shift_px` /
`image_size` 换算回原图坐标系；`track_polyline_px` 保持 prepared
语义（模板真值链路依赖）。

换算只依赖 diagnostics.vision_prep 元数据，与画像无关——四个画像
（meituan/xingzhe/xiaomi/amap）共用同一条路径，本测试用合成元数据
覆盖各形态，不绑定具体截图。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.pipeline import backproject_prep_px
from app.server import _img_px_transform, _to_original_img_px


def _result(
    orig: list[int] | None,
    prep: list[int] | None,
    crop: list[float] | None = None,
    polyline: list[list[float]] | None = None,
    anchors: list[dict] | None = None,
    shift: list[float] | None = None,
) -> SimpleNamespace:
    diag: dict = {}
    if orig is not None and prep is not None:
        diag["vision_prep"] = {"original_hw": orig, "prepared_hw": prep}
    if crop:
        diag["map_crop_applied"] = list(crop)
    return SimpleNamespace(
        diagnostics=diag,
        track_polyline_px=polyline or [],
        anchors_used=anchors or [],
        anchors_rejected=[],
        road_fit={"ok": False, "candidate_shift_px": shift},
        image_size=[prep[1], prep[0]] if prep else [0, 0],
    )


def _payload(result: SimpleNamespace) -> dict:
    return _to_original_img_px({"_result_ref": result,
                                "image_size": list(result.image_size),
                                "track_polyline_px": result.track_polyline_px,
                                "anchors_used": result.anchors_used,
                                "anchors_rejected": result.anchors_rejected,
                                "road_fit": result.road_fit})


def test_crop_scale_xiaomi_shape():
    """裁剪+缩放：纵向含 y0 偏移，横向只按缩放比。

    合成：原图 2000h×1000w，裁剪 10%~90%（留 1600 行），缩放 0.9 →
    prepared 1440h×900w。prepared (450, 720) → 原图 (500, 1000)。
    """
    r = _result(
        orig=[2000, 1000], prep=[1440, 900], crop=[0.10, 0.90],
        polyline=[[450.0, 720.0]],
        anchors=[{"name": "A", "px": 450.0, "py": 720.0}],
    )
    t = _img_px_transform(r)
    assert t is not None
    s, y0, ow, oh = t
    assert (ow, oh) == (1000, 2000)
    assert abs(s - 1000 / 900) < 1e-9
    assert abs(y0 - 200.0) < 1e-9
    p = _payload(r)
    assert p["image_size"] == [1000, 2000]
    assert p["track_polyline_img_px"] == [[500.0, 1000.0]]
    assert p["anchors_used"][0]["px"] == 500.0
    assert p["anchors_used"][0]["py"] == 1000.0
    # prepared 语义字段不动
    assert p["track_polyline_px"] == [[450.0, 720.0]]


def test_scale_only_no_crop():
    """只缩放不裁剪：y0 = 0，纯等比反投影。"""
    r = _result(orig=[2720, 1260], prep=[1020, 473],
                polyline=[[236.5, 510.0]])
    p = _payload(r)
    s = 1260 / 473
    x, y = p["track_polyline_img_px"][0]
    assert abs(x - 236.5 * s) < 0.2
    assert abs(y - 510.0 * s) < 0.2
    assert p["image_size"] == [1260, 2720]


def test_identity_when_prepared_equals_original():
    """无裁剪无缩放（原分辨率输入）：反投影是恒等映射。"""
    r = _result(orig=[864, 600], prep=[864, 600],
                polyline=[[100.0, 200.0]])
    p = _payload(r)
    assert p["track_polyline_img_px"] == [[100.0, 200.0]]
    assert p["image_size"] == [600, 864]


def test_missing_meta_untouched():
    """旧结果 / 元数据缺失：payload 保持原样，不得新增换算字段。"""
    r = _result(orig=None, prep=[864, 600],
                polyline=[[100.0, 200.0]],
                anchors=[{"name": "A", "px": 1.0, "py": 2.0}])
    p = _payload(r)
    assert "track_polyline_img_px" not in p
    assert p["image_size"] == [600, 864]
    assert p["anchors_used"][0]["px"] == 1.0


def test_paired_mode_canvas_coords_untouched():
    """配对/拼接模式（prep_meta=None）：overlay 画在拼接画布上，
    与 track_polyline_px 同坐标系，不得换算。"""
    r = _result(orig=None, prep=None,
                polyline=[[50.0, 60.0]], shift=[3.0, 4.0])
    p = _payload(r)
    assert "track_polyline_img_px" not in p
    assert p["road_fit"]["candidate_shift_px"] == [3.0, 4.0]


def test_candidate_shift_scales_without_y0():
    """位移增量按缩放比换算，不加 y0（平移量不是绝对坐标）。"""
    r = _result(orig=[2000, 1000], prep=[1440, 900],
                crop=[0.10, 0.90], shift=[90.0, 111.0])
    p = _payload(r)
    s = 1000 / 900
    assert p["road_fit"]["candidate_shift_px"] == [
        round(90.0 * s, 1), round(111.0 * s, 1)]


def test_matches_pipeline_backproject():
    """server 出口换算与 pipeline.backproject_prep_px 逐点一致。"""
    orig, prep, crop = [2000, 1000], [1440, 900], [0.10, 0.90]
    r = _result(orig=orig, prep=prep, crop=crop,
                polyline=[[123.4, 567.8]])
    t = _img_px_transform(r)
    assert t is not None
    s, y0, _, _ = t
    assert s == orig[1] / prep[1]
    assert y0 == crop[0] * orig[0]
    bp = backproject_prep_px(
        [[123.4, 567.8]], tuple(crop), prep, orig)
    assert bp is not None
    bx, by = bp[0]
    px, py = (123.4 / s, 567.8 / s + y0)
    assert abs(bx - px) < 1e-6
    assert abs(by - py) < 1e-6
