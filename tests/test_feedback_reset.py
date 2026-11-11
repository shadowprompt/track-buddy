"""「清空全部人工纠正」的后端回归测试。

背景：用户点位在 ``Geocoder.geocode()`` 里会短路掉在线地理编码，一条
点错的纠正会长期生效；而过去没有删除入口，只能手工改 data/lexicon.json
再重启服务（用户明确要求由自己重启，助手不能动服务进程）。

因此提供 ``Lexicon.clear_user_points()`` + ``POST /api/feedback/reset``：
只删 ``verified`` / ``source=user`` 点位，保留自动写入的证据，内存与磁盘
同步、幂等、无需重启。本测试全部用 tmp_path 的临时词典，不碰真实 data。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.geocode import Lexicon  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _seed(path: Path) -> Lexicon:
    """构造混合词典：纯自动 / 被纠正就地升级 / 纯人工 / 同名多点。"""
    lex = Lexicon(path)
    lex.remember("同仁医院", 121.4000, 31.2000, city="上海",
                 source="tianditu", boost=0.5)
    # 生产里自动写入由 flush_capture 带 confidence -> 会留下 last_confidence
    lex.remember("同仁医院", 121.4001, 31.2001, city="上海",
                 source="tianditu", boost=0.5, confidence=0.54)
    # 同一坐标纠正：把自动条目就地升级成 user / verified
    lex.remember("同仁医院", 121.4001, 31.2001, city="上海",
                 source="user", boost=2.0)
    lex.remember("交通大学", 121.4300, 31.2000, city="上海",
                 source="user", boost=2.0)
    lex.remember("交通大学", 121.4400, 31.2100, city="上海",
                 source="user", boost=2.0)
    lex.remember("自动保留点", 121.5000, 31.3000, city="上海",
                 source="tianditu", boost=0.5)
    return lex


def test_clear_only_user_points_and_persists(tmp_path):
    path = tmp_path / "lexicon.json"
    lex = _seed(path)
    before = lex.stats()
    assert before["verified_points"] >= 3
    assert before["auto_points"] >= 2

    report = lex.clear_user_points()

    assert report["removed_points"] == before["verified_points"]
    assert report["remaining_points"] == before["auto_points"]
    # 被就地升级的自动点位无法还原，必须如实计数，不能假装完全无损
    assert report["upgraded_auto_points_lost"] == 1
    assert report["removed_keys"] >= 1

    after = lex.stats()
    assert after["verified_points"] == 0
    assert after["auto_points"] == before["auto_points"]
    # 磁盘与内存一致；纯人工的 key 要整个删掉，不能留下空列表
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(disk) == set(lex._data)  # noqa: SLF001 — 只读校验
    assert "交通大学" not in disk
    assert all(entries for entries in disk.values())
    assert all(not e.get("verified") and e.get("source") != "user"
               for entries in disk.values() for e in entries)
    # 重新加载后仍然一致（防止只改了内存没落盘）
    assert Lexicon(path).stats() == after


def test_clear_is_idempotent_and_does_not_rewrite_when_empty(tmp_path):
    path = tmp_path / "lexicon.json"
    lex = _seed(path)
    lex.clear_user_points()
    snapshot = path.read_bytes()

    second = lex.clear_user_points()
    assert second["removed_points"] == 0
    assert second["removed_keys"] == 0
    assert second["remaining_points"] == lex.stats()["points"]
    assert path.read_bytes() == snapshot, "空操作时不应重写磁盘文件"


def test_reset_endpoint_requires_explicit_confirm(tmp_path, monkeypatch):
    import app.server as srv

    lex = _seed(tmp_path / "lexicon.json")
    monkeypatch.setattr(srv._pipeline, "lexicon", lex)
    with TestClient(srv.app) as client:
        assert client.post("/api/feedback/reset", json={}).status_code == 400
        assert client.post("/api/feedback/reset").status_code == 400
        assert client.post("/api/feedback/reset",
                           json={"confirm": "true"}).status_code == 400
        assert lex.stats()["verified_points"] > 0, "被拒的请求不能改词典"


def test_reset_endpoint_clears_submitted_feedback(tmp_path, monkeypatch):
    import app.server as srv

    lex = Lexicon(tmp_path / "lexicon.json")
    monkeypatch.setattr(srv._pipeline, "lexicon", lex)
    # 自动点位直接用 Lexicon 造（/api/feedback 永远是 user 写入）
    lex.remember("自动保留点", 121.5000, 31.3000, city="上海",
                 source="tianditu", boost=0.5)
    auto_before = lex.stats()["auto_points"]
    with TestClient(srv.app) as client:
        submitted = client.post("/api/feedback", json={
            "name": "交通大学", "lon": 121.43, "lat": 31.20,
            "crs": "GCJ-02", "city": "上海",
        })
        assert submitted.status_code == 200
        assert lex.stats()["verified_points"] == 1

        resp = client.post("/api/feedback/reset", json={"confirm": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["removed_points"] == 1
        assert data["lexicon"]["verified_points"] == 0
        assert data["lexicon"]["auto_points"] == auto_before

        listed = client.get("/api/lexicon").json()
        assert listed["stats"]["verified_points"] == 0
        assert "自动保留点" in listed["entries"]

        # 幂等：再清一次仍是 200，计数为 0
        again = client.post("/api/feedback/reset", json={"confirm": True})
        assert again.status_code == 200
        assert again.json()["removed_points"] == 0
