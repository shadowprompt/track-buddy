"""后台精修复用主请求中间结果的回归测试。

事故：后台精修原先会 ``imread_unicode`` + ``Pipeline.run(refine=True)``，
把 OCR / 轨迹提取 / 在线锚点地理编码 / 配准整条重跑一遍。12 图拼接画布
实测多花 4 分钟，而且后台重新取在线锚点，配额一波动就和主请求判定不一致。

现在主请求把中间产物留在 ``PipelineResult.refine_context``，后台只调
``refine_existing`` 补算平移量。这里验证编排逻辑、重试安全（deepcopy）、
强约束保护，以及「绝不重跑 pipeline」这条底线。
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import pipeline as pipeline_mod
from app.pipeline import PipelineResult, RefineContext, RidePipeline


SHIFT = np.array([100.0, -50.0], dtype=float)


def _base(road_name_fit: dict | None = None,
          warnings: list[str] | None = None) -> PipelineResult:
    ctx = RefineContext(
        reg=SimpleNamespace(translation=np.zeros(2, dtype=float)),
        track=SimpleNamespace(),
        simplified=np.array([[0.0, 0.0], [10.0, 10.0]]),
        anchors=[], pairs=[], texts=[], zoom={}, alternatives=[],
        road_name_fit=road_name_fit or {}, med_px=0.0,
        output_crs="WGS-84",
    )
    return PipelineResult(
        geojson={"features": [{"geometry": {
            "coordinates": [[121.0, 31.0], [121.0001, 31.0001]]},
            "properties": {}}]},
        geojson_gcj02={"features": [{"geometry": {
            "coordinates": [[121.0, 31.0], [121.0001, 31.0001]]},
            "properties": {}}]},
        confidence=0.2,
        crs="WGS-84",
        track_points=2,
        distance_m=100.0,
        diagnostics={"scale_ground_m_per_px": 1.0},
        road_name_fit=road_name_fit or {},
        warnings=warnings if warnings is not None else [
            "本次跳过路网精修（异步模式下由后台任务执行）",
            "路名归属（amap）：某路",
        ],
        elapsed_ms=1000,
        image_size=(100, 100),
        refine_context=ctx,
    )


def _install_fakes(monkeypatch, *, ok: bool = True) -> None:
    """把与「补算平移」无关的重活替换成确定性桩件。"""
    fit = {
        "ok": ok,
        "shift_e_m": 100.0 if ok else 0.0,
        "shift_n_m": -50.0 if ok else 0.0,
        "median_before_m": 25.0,
        "median_after_m": 9.0,
        "source": "osm",
        "notes": [] if ok else ["路网精修未采纳（测试桩）"],
    }

    def fake_road_refine(self, reg, track, mode="auto"):
        assert mode in ("auto", "verify")
        if ok:
            # 模拟 _road_refine 把平移并入配准矩阵
            reg.translation = reg.translation + SHIFT
        return dict(fit), object()

    def fake_apply(reg, poly):
        t = reg.translation
        return [(121.0 + float(t[0]) / 1e5, 31.0 + float(t[1]) / 1e5)
                for _ in poly]

    def fake_geojson(lonlat, reg, track, source_id=""):
        return {"features": [{"geometry": {
            "coordinates": [[float(lo), float(la)] for lo, la in lonlat]},
            "properties": {}}]}

    attr = SimpleNamespace(
        segments=[{"road": "某路"}], roads=["某路"],
        source="road_network",
        to_dict=lambda: {"source": "road_network", "segments": [{"road": "某路"}]},
    )

    monkeypatch.setattr(RidePipeline, "_road_refine", fake_road_refine)
    monkeypatch.setattr(RidePipeline, "_score_confidence",
                        lambda *a, **kw: 0.83)
    monkeypatch.setattr(RidePipeline, "_attribute_roads",
                        lambda *a, **kw: attr)
    monkeypatch.setattr(pipeline_mod, "apply_to_polyline", fake_apply)
    monkeypatch.setattr(pipeline_mod, "_build_geojson", fake_geojson)


def _pipeline() -> RidePipeline:
    # 绕过 __init__：refine_existing 只依赖被替换掉的类方法，
    # 不需要 Lexicon / Geocoder 这些带文件 IO 的状态。
    return RidePipeline.__new__(RidePipeline)


def test_refine_existing_reprojects_and_updates_fields(monkeypatch):
    _install_fakes(monkeypatch)
    base = _base()
    out = _pipeline().refine_existing(base)

    assert out.road_fit["ok"] is True
    assert out.confidence == 0.83
    assert out.road_attribution["source"] == "road_network"
    assert out.diagnostics["refine_reused"] is True
    # 平移真的进了输出坐标，且 base 本身没被改。
    # GCJ-02 那份不受输出 CRS 换算影响，直接量平移量；WGS-84 那份
    # 经过 GCJ→WGS 转换，只能断言「确实变了」。
    gcj_first = out.geojson_gcj02["features"][0]["geometry"]["coordinates"][0]
    assert gcj_first[0] == pytest.approx(121.001, abs=1e-6)
    assert gcj_first[1] == pytest.approx(30.9995, abs=1e-6)
    assert out.geojson["features"][0]["geometry"]["coordinates"][0][0] != 121.0
    assert base.geojson["features"][0]["geometry"]["coordinates"][0][0] == 121.0
    # 过期的说明被换掉
    assert not any("跳过路网精修" in w for w in out.warnings)
    assert not any("路名归属（amap）" in w for w in out.warnings)
    assert any("路网精修：平移" in w for w in out.warnings)
    assert any("路名归属（road_network）" in w for w in out.warnings)


def test_refine_existing_does_not_mutate_context_reg(monkeypatch):
    """任务重试时不能在上一次的平移上再叠一层。"""
    _install_fakes(monkeypatch)
    base = _base()
    first = _pipeline().refine_existing(base)
    second = _pipeline().refine_existing(base)
    assert np.allclose(base.refine_context.reg.translation, [0.0, 0.0])
    assert (first.geojson["features"][0]["geometry"]["coordinates"]
            == second.geojson["features"][0]["geometry"]["coordinates"])


def test_road_name_constraint_blocks_background_shift(monkeypatch):
    """用户路名约束是强约束，后台平移不能把它拉走。"""
    _install_fakes(monkeypatch)
    base = _base(road_name_fit={"ok": True})
    out = _pipeline().refine_existing(base)
    assert out.road_fit["ok"] is False
    assert "已由用户路名约束定位" in out.road_fit["notes"][0]
    assert out.geojson == base.geojson
    assert out.confidence == base.confidence


def test_unadopted_refine_keeps_coordinates(monkeypatch):
    _install_fakes(monkeypatch, ok=False)
    base = _base()
    out = _pipeline().refine_existing(base)
    assert out.geojson == base.geojson
    assert any("测试桩" in w for w in out.warnings)


def test_missing_context_fails_loudly():
    base = _base()
    base.refine_context = None
    with pytest.raises(ValueError, match="精修上下文"):
        _pipeline().refine_existing(base)


def test_verify_mode_is_passed_through(monkeypatch):
    _install_fakes(monkeypatch)
    seen: list[str] = []
    orig = RidePipeline._road_refine

    def spy(self, reg, track, mode="auto"):
        seen.append(mode)
        return orig(self, reg, track, mode=mode)

    monkeypatch.setattr(RidePipeline, "_road_refine", spy)
    _pipeline().refine_existing(_base(), mode="verify")
    assert seen == ["verify"]


def test_verify_mode_never_moves_or_rescores(monkeypatch):
    """多图默认档：只校验 + 命名，坐标与置信度都保持主请求的值。"""
    _install_fakes(monkeypatch)
    base = _base()
    out = _pipeline().refine_existing(base, mode="verify")
    assert out.road_fit["mode"] == "verify"
    assert out.geojson == base.geojson
    assert out.geojson_gcj02 == base.geojson_gcj02
    assert out.confidence == base.confidence
    assert out.track_points == base.track_points
    # 命名仍然换成几何判定的结果
    assert out.road_attribution["source"] == "road_network"
    assert any("路名归属（road_network）" in w for w in out.warnings)
    assert not any("路网精修：平移" in w for w in out.warnings)


# --- server 层：确认后台任务绝不重跑 pipeline ---------------------------

def test_background_refine_never_reruns_pipeline(monkeypatch, tmp_path):
    from app import server

    base = _base()

    def boom(*args, **kwargs):
        raise AssertionError("后台精修不该重跑 Pipeline.run")

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server._pipeline, "run", boom)
    monkeypatch.setattr(
        server._pipeline, "refine_existing",
        lambda b, *, mode="auto": _finished(b),
    )
    server._refine_jobs.clear()
    server._refine_ctx.clear()

    server._start_refine("sid", Path("sid.jpg"),
                         base_result=base, mode="auto")
    # state=done 写在 finally 之前，要等上下文也释放完
    for _ in range(300):
        if (server._refine_jobs["sid"]["state"] != "running"
                and not server._refine_ctx):
            break
        time.sleep(0.01)

    job = server._refine_jobs["sid"]
    assert job["state"] == "done", job
    assert job["payload"]["shot_id"] == "sid"
    # 上下文用完即释放
    assert server._refine_ctx == {}


def _finished(base: PipelineResult) -> PipelineResult:
    from dataclasses import replace
    return replace(
        base,
        road_fit={"ok": True, "shift_e_m": 25.0, "shift_n_m": -50.0,
                  "median_before_m": 25.0, "median_after_m": 9.0},
        confidence=0.83,
    )


def test_missing_server_context_reports_error(monkeypatch, tmp_path):
    """上下文丢了要直接报错，不能静默降级回旧的重跑逻辑。"""
    from app import server

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    server._refine_jobs.clear()
    server._refine_ctx.clear()
    server._run_refine("missing", Path("missing.jpg"))
    job = server._refine_jobs["missing"]
    assert job["state"] == "error"
    assert "精修上下文" in job["message"]
