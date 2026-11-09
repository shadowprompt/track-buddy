"""多图接口接入后台精修的集成测试。

改造前：``/api/recognize_multi`` 默认 ``auto`` 直接跳过后台精修，只有手动
传 ``refine=on`` 才会同步跑一遍（12 图拼接实测 263 秒）。改造后默认启动
后台任务，且**多图只校验 + 命名不平移**，显式 ``refine=on`` 仍保留同步
修正模式。

把拼接与流水线都替换成桩件，只验证接口层的接线、轮询字段、落盘同步，
不触发真实 OCR / Overpass。
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.pipeline import PipelineResult, RefineContext


def _result() -> PipelineResult:
    gj = {"features": [{"geometry": {
        "coordinates": [[121.0, 31.0], [121.001, 31.001]]},
        "properties": {}}]}
    return PipelineResult(
        geojson=gj,
        geojson_gcj02=json.loads(json.dumps(gj)),
        confidence=0.4,
        crs="WGS-84",
        track_points=2,
        distance_m=100.0,
        road_fit={"ok": False, "notes": [
            "本次跳过路网精修（异步模式下由后台任务执行）"]},
        diagnostics={"scale_ground_m_per_px": 1.0},
        warnings=["本次跳过路网精修（异步模式下由后台任务执行）"],
        elapsed_ms=100,
        image_size=(10, 10),
        refine_context=RefineContext(
            reg=SimpleNamespace(translation=np.zeros(2, dtype=float)),
            track=SimpleNamespace(),
            simplified=np.array([[0.0, 0.0], [1.0, 1.0]]),
            anchors=[], pairs=[], texts=[], zoom={}, alternatives=[],
            road_name_fit={}, med_px=0.0,
        ),
    )


def _finished(base: PipelineResult) -> PipelineResult:
    return replace(base, road_fit={
        "ok": True, "mode": "verify", "shift_e_m": 0.0, "shift_n_m": 0.0,
        "median_before_m": 25.0, "median_after_m": 25.0,
        "notes": ["路网校验：贴路中位 25 米"],
    })


@pytest.fixture
def env(monkeypatch, tmp_path):
    from app import server

    small = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(server, "imread_unicode", lambda _p: small)
    monkeypatch.setattr(
        server, "stitch_images",
        lambda imgs: SimpleNamespace(
            ok=True, canvas=small,
            to_dict=lambda: {"ok": True, "notes": [], "shots": [
                {"index": i, "box": [0, 0, 10, 10]}
                for i in range(len(imgs))]},
        ),
    )
    res = _result()
    monkeypatch.setattr(server._pipeline, "run", lambda *a, **kw: res)
    monkeypatch.setattr(
        server._pipeline, "refine_existing",
        lambda base, *, mode="auto": _finished(base),
    )
    server._refine_jobs.clear()
    server._refine_ctx.clear()
    yield server, res, tmp_path
    # 等后台线程收尾，避免它拿着已被还原的 monkeypatch 全局继续跑
    for _ in range(300):
        if not any(j["state"] == "running" for j in server._refine_jobs.values()):
            break
        time.sleep(0.01)
    server._refine_jobs.clear()
    server._refine_ctx.clear()


def _spy(server, monkeypatch, seen: dict) -> None:
    orig = server._start_refine

    def spy(*args, **kwargs):
        seen["mode"] = kwargs.get("mode")
        seen["extra"] = kwargs.get("extra")
        return orig(*args, **kwargs)

    monkeypatch.setattr(server, "_start_refine", spy)


def _post(client, *, refine: str = "auto", n: int = 3):
    files = [("files", (f"s{i}.jpg", b"fake", "image/jpeg")) for i in range(n)]
    return client.post(
        "/api/recognize_multi",
        files=files,
        data={"city": "", "output_crs": "WGS-84",
              "allow_online": "true", "refine": refine, "road_names": ""},
    )


def _wait_done(server, shot_id: str) -> dict:
    for _ in range(300):
        job = server._refine_jobs.get(shot_id)
        if job and job["state"] != "running":
            return job
        time.sleep(0.01)
    return server._refine_jobs.get(shot_id) or {}


def test_multi_auto_starts_verify_background_refine(env, monkeypatch):
    server, _, tmp_path = env
    seen: dict = {}
    _spy(server, monkeypatch, seen)

    with TestClient(server.app) as client:
        resp = _post(client)
        assert resp.status_code == 200, resp.text
        data = resp.json()

    assert data["refine_pending"] is True
    assert data["refine_state"] == "running"
    assert seen["mode"] == "verify"
    assert set(seen["extra"]) >= {"stitch", "n_shots", "stitched_url"}
    assert seen["extra"]["n_shots"] == 3
    warns = "\n".join(data["warnings"])
    assert "多图后台精修已启动" in warns
    assert "不执行后台路网精修" not in warns
    assert "跳过路网精修" not in warns

    shot_id = data["shot_id"]
    job = _wait_done(server, shot_id)
    assert job["state"] == "done", job
    payload = job["payload"]
    assert payload["n_shots"] == 3
    assert payload["stitch"]["ok"] is True
    assert payload["stitched_url"]
    assert payload["road_fit"]["mode"] == "verify"

    # 核对页两份落盘结果都要是精修后的
    for name in ("last_result.json", "last_stitch.json"):
        saved = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert saved["shot_id"] == shot_id
        assert saved["road_fit"]["ok"] is True
        assert "stitch" in saved

    with TestClient(server.app) as client:
        polled = client.get(f"/api/refine/{shot_id}").json()
    assert polled["state"] == "done"
    assert polled["payload"]["road_fit"]["ok"] is True


def test_rejected_background_refine_is_persisted(env, monkeypatch):
    """护栏否决时也必须落盘，否则刷新后橙线与候选切换按钮都会消失。"""
    server, base, tmp_path = env

    def rejected(base_result, *, mode="auto"):
        return replace(base_result, road_fit={
            "ok": False,
            "shift_e_m": 0.0, "shift_n_m": 0.0,
            "median_before_m": 78.1, "median_after_m": 78.1,
            "candidate_shift_e_m": -55.0, "candidate_shift_n_m": -175.0,
            "candidate_median_after_m": 6.4,
            "candidate_shift_px": [-19.73, 62.79],
            "notes": ["路网精修未采纳（可靠锚点残差 52m → 222m）"],
        })

    monkeypatch.setattr(server._pipeline, "refine_existing", rejected)

    with TestClient(server.app) as client:
        resp = _post(client, n=3)
        assert resp.status_code == 200, resp.text
        shot_id = resp.json()["shot_id"]

    job = _wait_done(server, shot_id)
    assert job["state"] == "done", job
    for name in ("last_result.json", "last_stitch.json"):
        saved = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert saved["shot_id"] == shot_id
        rf = saved["road_fit"]
        assert rf["ok"] is False
        assert rf["candidate_shift_px"] == [-19.73, 62.79]
        # 否决时输出位移归零，落盘不能把坐标改掉
        assert rf["shift_e_m"] == 0.0 and rf["shift_n_m"] == 0.0


def test_multi_sync_refine_on_has_no_background_job(env, monkeypatch):
    server, _, _ = env
    seen: dict = {}
    _spy(server, monkeypatch, seen)

    with TestClient(server.app) as client:
        data = _post(client, refine="on").json()

    assert "refine_pending" not in data
    assert seen == {}
    assert server._refine_ctx == {}


def test_multi_refine_off_has_no_background_job(env, monkeypatch):
    server, _, _ = env
    seen: dict = {}
    _spy(server, monkeypatch, seen)

    with TestClient(server.app) as client:
        data = _post(client, refine="off").json()

    assert "refine_pending" not in data
    assert seen == {}


def test_single_file_uses_full_mode(env, monkeypatch):
    """只传 1 张图时不该被「多图」规则误伤，仍走完整修正模式。"""
    server, _, _ = env
    seen: dict = {}
    _spy(server, monkeypatch, seen)

    with TestClient(server.app) as client:
        data = _post(client, n=1).json()

    assert data["refine_pending"] is True
    assert seen["mode"] == "auto"
    _wait_done(server, data["shot_id"])


def test_refine_status_reports_elapsed_seconds():
    """长任务前端要能拿到已耗时，才能给出「仍在计算」的提示。"""
    from app import server

    server._refine_jobs.clear()
    server._refine_jobs["sid"] = {
        "state": "running", "started_at": time.time() - 12.5,
    }
    try:
        with TestClient(server.app) as client:
            job = client.get("/api/refine/sid").json()
        assert job["state"] == "running"
        assert job["elapsed_s"] >= 12.0
    finally:
        server._refine_jobs.clear()


def test_refine_status_done_has_no_elapsed():
    from app import server

    server._refine_jobs.clear()
    server._refine_jobs["sid"] = {"state": "done", "payload": {"ok": True}}
    try:
        with TestClient(server.app) as client:
            job = client.get("/api/refine/sid").json()
        assert job["state"] == "done"
        assert "elapsed_s" not in job
    finally:
        server._refine_jobs.clear()


def test_no_coordinates_skips_background_refine(env, monkeypatch):
    server, res, _ = env
    res.geojson = {"features": [{"geometry": {"coordinates": []},
                                 "properties": {}}]}
    seen: dict = {}
    _spy(server, monkeypatch, seen)

    with TestClient(server.app) as client:
        data = _post(client).json()

    assert "refine_pending" not in data
    assert seen == {}
    assert "本次未启动后台精修" in "\n".join(data["warnings"])
