"""运动记录云存储（腾讯 COS）后端契约：全桩件、不联网、不写真实 data/。

覆盖：HMAC token、微信登录不泄密、uid 隔离、识别端点带 token 自动保存、
无 token 零回归、保存失败降级、列表分页/index 自愈、401/403/404/503、
图片代理流、删除、health 与静态接线。
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import records
from app.config import CONFIG
from app.route_template import TemplateStore
from app.server import app as S  # noqa: F401  (导入即注册路由)
import app.server as server

SECRET = "unit-test-record-secret"


# --- 内存 COS 桩件 -------------------------------------------------------

class FakeCOS:
    """只实现 store.py 用到的 5 个 SDK 方法。"""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    @staticmethod
    def _etag(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()  # noqa: S324 — 仅测试用

    def put_object(self, Bucket, Key, Body, **kw):
        data = Body if isinstance(Body, bytes) else Body.read()
        self.objects[Key] = data
        return {"ETag": f'"{self._etag(data)}"'}

    def get_object(self, Bucket, Key, **kw):
        if Key not in self.objects:
            raise KeyError(Key)
        body = SimpleNamespace(
            get_raw_stream=lambda: io.BytesIO(self.objects[Key]))
        return {"Body": body, "ETag": f'"{self._etag(self.objects[Key])}"'}

    def head_object(self, Bucket, Key, **kw):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ETag": f'"{self._etag(self.objects[Key])}"'}

    def list_objects(self, Bucket, Prefix="", Delimiter="", Marker="",
                     MaxKeys=1000, **kw):
        candidates = [k for k in sorted(self.objects)
                      if k.startswith(Prefix) and k > Marker]
        page = candidates[:MaxKeys]
        contents = [{"Key": k, "Size": len(self.objects[k])} for k in page]
        common: list[dict] = []
        if Delimiter:
            seen: list[str] = []
            for key in page:
                rest = key[len(Prefix):]
                if Delimiter in rest:
                    prefix = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                    if prefix not in seen:
                        seen.append(prefix)
            contents = [c for c in contents
                        if Delimiter not in c["Key"][len(Prefix):]]
            common = [{"Prefix": p} for p in seen]
        truncated = len(candidates) > len(page)
        out = {
            "Contents": contents,
            "CommonPrefixes": common,
            "IsTruncated": "true" if truncated else "false",
        }
        if truncated and page:
            out["NextMarker"] = page[-1]
        return out

    def delete_objects(self, Bucket, Delete, **kw):
        for item in Delete.get("Object", []):
            self.objects.pop(item["Key"], None)
        return {}


def _jpg() -> bytes:
    _, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 255, np.uint8))
    return buf.tobytes()


def _fake_result():
    gj = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "LineString",
                     "coordinates": [[121.47, 31.23], [121.48, 31.24]]}}]}
    return SimpleNamespace(
        confidence=0.82, track_points=2, geojson=gj, geojson_gcj02=gj,
        crs="WGS-84", distance_m=1500.0, anchors_used=[], anchors_rejected=[],
        road_attribution={}, road_fit={}, road_name_constraints=[],
        road_name_fit={}, alternatives=[],
        diagnostics={"input_quality": {"usable": False}}, warnings=[],
        elapsed_ms=1, track_polyline_px=[[0, 0], [10, 10]],
        map_matched=False, image_size=(100, 100, 3))


@pytest.fixture
def wired(monkeypatch, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(server, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_templates",
                        TemplateStore(tmp_path / "route_templates.json"))
    monkeypatch.setenv("RECORDS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("RECORDS_TOKEN_TTL", "3600")
    monkeypatch.setattr(records, "records_enabled", lambda: True)
    monkeypatch.setattr(records.store, "cos_sdk_available", lambda: True)

    fake = FakeCOS()
    store = records.store.CosRecordStore(
        client=fake, bucket="unit-bucket", prefix="ride-records/")
    monkeypatch.setattr(records.store, "get_record_store", lambda: store)

    def fake_code2session(code, *, http_get=None):
        if code == "codeA":
            return {"openid": "openid-A", "session_key": "SECRET-KEY-A"}
        if code == "codeB":
            return {"openid": "openid-B", "session_key": "SECRET-KEY-B"}
        raise records.TokenError("bad code")

    monkeypatch.setattr(records.auth, "code2session", fake_code2session)
    monkeypatch.setattr(server._pipeline, "run", lambda *a, **k: _fake_result())
    monkeypatch.setattr(server, "_need_background_refine",
                        lambda *a, **k: False)
    monkeypatch.setattr(server, "_start_refine", lambda *a, **k: None)

    def fake_exports(payload):
        payload["ride_summary"] = {
            "started_at": "2026-11-14T20:20:26", "duration_s": 1235,
            "distance_km": 1.5, "calories_kcal": 80, "available": True,
        }
        payload["gpx"] = "<gpx/>"
        payload["tcx"] = "<TrainingCenterDatabase/>"

    monkeypatch.setattr(server, "_attach_exports", fake_exports)
    return SimpleNamespace(store=store, cos=fake, client=TestClient(server.app))


def _login(wired, code="codeA") -> str:
    r = wired.client.post("/api/auth/wxlogin", json={"code": code})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _recognize(wired, token: str | None = None, *, summary=False):
    files = [("file", ("overview.jpg", _jpg(), "image/jpeg"))]
    if summary:
        files.append(("summary", ("summary.jpg", b"fake-summary-bytes",
                                 "image/jpeg")))
    r = wired.client.post(
        "/api/recognize", files=files, data={"city": "上海", "refine": "off"},
        headers=_auth(token) if token else {})
    assert r.status_code == 200, r.text
    return r.json()


# --- 1. token ------------------------------------------------------------

def test_token_roundtrip_and_uid_derivation():
    uid = records.uid_from_openid("openid-A", SECRET)
    assert len(uid) == 32 and uid != "openid-A"
    assert uid != records.uid_from_openid("openid-A", "other-secret")
    token = records.issue_token(uid, secret=SECRET, ttl=3600, now=1000)
    assert records.verify_token(token, secret=SECRET, now=1001)["uid"] == uid


def test_token_expired_tampered_wrong_secret():
    token = records.issue_token("u", secret=SECRET, ttl=60, now=1000)
    with pytest.raises(records.TokenError):
        records.verify_token(token, secret=SECRET, now=1061)
    with pytest.raises(records.TokenError):
        records.verify_token(token + "x", secret=SECRET, now=1001)
    with pytest.raises(records.TokenError):
        records.verify_token(token, secret="another", now=1001)
    with pytest.raises(records.TokenError):
        records.verify_token("not-a-token", secret=SECRET, now=1001)


# --- 2. 微信登录 ---------------------------------------------------------

def test_wxlogin_does_not_leak_identity(wired):
    r = wired.client.post("/api/auth/wxlogin", json={"code": "codeA"})
    assert r.status_code == 200
    body = r.text
    for secret in ("openid-A", "SECRET-KEY-A", "session_key", "errcode"):
        assert secret not in body
    assert set(r.json()) == {"token", "expires_in"}


def test_wxlogin_bad_code_and_no_code(wired):
    r = wired.client.post("/api/auth/wxlogin", json={"code": "bad"})
    assert r.status_code == 401
    assert "errcode" not in r.text and "errmsg" not in r.text
    assert wired.client.post("/api/auth/wxlogin", json={}).status_code == 400


def test_wxlogin_disabled_returns_503(wired, monkeypatch):
    monkeypatch.setattr(records, "records_enabled", lambda: False)
    assert wired.client.post("/api/auth/wxlogin",
                             json={"code": "codeA"}).status_code == 503


# --- 3. 鉴权与零回归 -----------------------------------------------------

@pytest.mark.parametrize("path", ["/api/records", "/api/records/x"])
def test_records_require_token(wired, path):
    assert wired.client.get(path).status_code == 401
    assert wired.client.get(path, headers=_auth("bad.token.x")).status_code == 401


def test_records_disabled_returns_403(wired, monkeypatch):
    token = _login(wired)
    monkeypatch.setattr(records, "records_enabled", lambda: False)
    assert wired.client.get("/api/records",
                            headers=_auth(token)).status_code == 403


def test_no_token_zero_regression(wired):
    payload = _recognize(wired, None)
    assert wired.cos.objects == {}, "无 token 时一次 COS 调用都不能有"
    assert "record_id" not in payload and "saved_at" not in payload
    assert payload["shot_url"].startswith("/api/shots/")


def test_autosave_payload_and_images(wired):
    token = _login(wired)
    payload = _recognize(wired, token)
    meta_key = records.keys.meta_key(
        records.uid_from_openid("openid-A", SECRET), payload["shot_id"],
        prefix="ride-records/")
    meta = json.loads(wired.cos.objects[meta_key])
    assert meta["gpx"] == "<gpx/>" and meta["tcx"] == "<TrainingCenterDatabase/>"
    assert meta["ride_summary"]["started_at"] == "2026-11-14T20:20:26"
    assert "coords" not in meta and "coords_a" not in meta
    assert meta["shot_url"].startswith("/api/records/")
    assert any(k.endswith("overview.jpg") or k.endswith(".jpg")
               for k in wired.cos.objects if "/images/" in k)


def test_autosave_failure_does_not_break_response(wired, monkeypatch, caplog):
    token = _login(wired)

    def boom(*a, **k):
        raise RuntimeError("COS down")

    monkeypatch.setattr(wired.store, "put_record", boom)
    with caplog.at_level("WARNING"):
        payload = _recognize(wired, token)
    assert payload["ok"] is True
    assert "运动记录保存 COS 失败" in caplog.text


def test_summary_image_persisted(wired):
    token = _login(wired)
    payload = _recognize(wired, token, summary=True)
    uid = records.uid_from_openid("openid-A", SECRET)
    key = records.keys.image_key(
        uid, payload["shot_id"], f"{payload['shot_id']}_summary.jpg",
        prefix="ride-records/")
    assert wired.cos.objects.get(key) == b"fake-summary-bytes"


# --- 4. 隔离 / 列表 / 删除 -----------------------------------------------

def test_uid_isolation(wired):
    token_a, token_b = _login(wired, "codeA"), _login(wired, "codeB")
    pa, pb = _recognize(wired, token_a), _recognize(wired, token_b)
    la = wired.client.get("/api/records", headers=_auth(token_a)).json()
    lb = wired.client.get("/api/records", headers=_auth(token_b)).json()
    assert [x["id"] for x in la["items"]] == [pa["shot_id"]]
    assert [x["id"] for x in lb["items"]] == [pb["shot_id"]]
    assert la["total"] == 1 and lb["total"] == 1
    assert wired.client.get(f"/api/records/{pb['shot_id']}",
                            headers=_auth(token_a)).status_code == 404
    assert all("openid-A" not in k and "openid-B" not in k
               for k in wired.cos.objects)


def test_list_pagination_and_bad_cursor(wired):
    token = _login(wired)
    uid = records.uid_from_openid("openid-A", SECRET)
    for i in range(45):
        rec_id = f"rec_{i:03d}"
        wired.store.put_record(
            uid, rec_id, meta={"ok": True},
            summary={"id": rec_id, "created_at": 1000 + i},
            images={})
    seen, cursor, pages = [], None, 0
    while True:
        params = {"limit": 20}
        if cursor:
            params["cursor"] = cursor
        r = wired.client.get("/api/records", params=params,
                             headers=_auth(token))
        assert r.status_code == 200, r.text
        data = r.json()
        seen.extend(x["id"] for x in data["items"])
        pages += 1
        cursor = data["next_cursor"]
        if not cursor:
            break
        assert pages < 5
    assert len(seen) == len(set(seen)) == 45
    assert "rec_044" in seen and "rec_000" in seen
    assert wired.client.get("/api/records?cursor=!!!",
                            headers=_auth(token)).status_code == 400


def test_empty_list(wired):
    token = _login(wired)
    r = wired.client.get("/api/records", headers=_auth(token))
    assert r.json() == {"items": [], "next_cursor": None, "total": 0}


def test_index_rebuild_after_loss(wired):
    token = _login(wired)
    _recognize(wired, token)
    uid = records.uid_from_openid("openid-A", SECRET)
    wired.cos.objects.pop(records.keys.index_key(uid, prefix="ride-records/"))
    r = wired.client.get("/api/records", headers=_auth(token))
    assert r.status_code == 200 and r.json()["total"] == 1


def test_concurrent_same_uid_writes_keep_both(wired):
    uid = records.uid_from_openid("openid-A", SECRET)
    errors: list[Exception] = []

    def put(i):
        try:
            wired.store.put_record(
                uid, f"rec_{i}", meta={},
                summary={"id": f"rec_{i}", "created_at": 1000 + i}, images={})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=put, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    items, _, total = wired.store.list_records(uid, cursor=None, limit=20)
    assert total == 2 and {x["id"] for x in items} == {"rec_0", "rec_1"}


def test_record_detail_image_and_delete(wired):
    token = _login(wired)
    payload = _recognize(wired, token)
    rec_id = payload["shot_id"]
    detail = wired.client.get(f"/api/records/{rec_id}",
                              headers=_auth(token)).json()
    assert detail["payload"]["geojson"]["type"] == "FeatureCollection"
    assert detail["images"], "记录必须列出已保存的图片"
    url = detail["images"][0]["url"]
    img = wired.client.get(url, headers=_auth(token))
    assert img.status_code == 200 and img.content.startswith(b"\xff\xd8")
    assert wired.client.get(url).status_code == 401
    assert wired.client.get(f"/api/records/{rec_id}/images/nope.jpg",
                            headers=_auth(token)).status_code == 404
    assert wired.client.delete(f"/api/records/{rec_id}",
                               headers=_auth(token)).json() == {"ok": True}
    assert not any(f"/records/{rec_id}/" in k for k in wired.cos.objects)
    assert wired.client.delete(f"/api/records/{rec_id}",
                               headers=_auth(token)).status_code == 404


def test_cross_uid_delete_not_found(wired):
    token_a, token_b = _login(wired, "codeA"), _login(wired, "codeB")
    payload = _recognize(wired, token_b)
    assert wired.client.delete(f"/api/records/{payload['shot_id']}",
                               headers=_auth(token_a)).status_code == 404


# --- 5. health 与静态接线契约 --------------------------------------------

def test_health_reports_records_enabled(wired):
    r = wired.client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["records"]["enabled"] is True


def test_static_wiring_contract():
    root = Path(__file__).resolve().parents[1]
    server_src = (root / "src" / "app" / "server.py").read_text(encoding="utf-8")
    req = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "cos-python-sdk-v5" in req
    for needle in ("_maybe_save_record(", "records.optional_record_uid",
                   "records.require_record_uid", "/api/records/{rec_id}"):
        assert needle in server_src
    assert server_src.count("_maybe_save_record(") >= 3  # 定义 + 两个端点


def test_config_properties_exist():
    assert CONFIG.cos_prefix.endswith("/")
    assert CONFIG.records_token_ttl >= 60
    assert isinstance(CONFIG.records_config_ready, bool)
