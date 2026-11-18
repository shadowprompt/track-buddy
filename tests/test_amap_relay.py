"""高德中转（app.amap_relay）的单元测试。

覆盖三块：
  1. client.rewrite_amap_request 的改写语义（端点/完整 URL/非高德/未配置）；
  2. server 的鉴权与路径白名单（不起真实网络，直接调路由函数）；
  3. 端到端转发链路：本地起 TestClient，上游用 monkeypatch 替身，
     验证 key 注入与 relay_token 剥离。

不依赖真实高德 key——线上链路是否通由 probe / healthz 判断。
"""

from __future__ import annotations

import os
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# 1. client：URL/参数改写
# ---------------------------------------------------------------------------

class _FakeCfg:
    """只暴露中转两个属性的轻量替身（避免真实 .env 干扰断言）。"""

    def __init__(self, relay_url: str | None, token: str | None = None):
        self.amap_relay_url = relay_url
        self.amap_relay_token = token


def test_rewrite_disabled_passthrough():
    """未配置中转时原样返回——线上零回归的根基。"""
    from app.amap_relay.client import rewrite_amap_request

    url = "https://restapi.amap.com/v3/place/text"
    params = {"key": "abc", "keywords": "上海站"}
    out_url, out_params = rewrite_amap_request(url, params, _FakeCfg(None))
    assert out_url == url
    assert out_params == params


def test_rewrite_non_amap_untouched():
    """非高德 host（腾讯/天地图）绝不改写。"""
    from app.amap_relay.client import rewrite_amap_request

    url = "https://apis.map.qq.com/ws/place/v1/search"
    out_url, out_params = rewrite_amap_request(
        url, {"key": "k"}, _FakeCfg("http://relay:8849", "tok"))
    assert out_url == url
    assert out_params == {"key": "k"}


def test_rewrite_endpoint_and_params():
    from app.amap_relay.client import rewrite_amap_request

    url, params = rewrite_amap_request(
        "https://restapi.amap.com/v3/place/text",
        {"key": "LOCAL_DUMMY", "keywords": "上海站"},
        _FakeCfg("http://relay:8849/", "tok"),
    )
    assert url == "http://relay:8849/amap/v3/place/text"
    assert "key" not in params          # 占位 key 绝不外发
    assert params["relay_token"] == "tok"
    assert params["keywords"] == "上海站"  # 业务参数原样保留


def test_rewrite_full_url_with_query():
    """staticmap 风格：key 拼在 URL query 里也要被拆出来丢弃。"""
    from app.amap_relay.client import rewrite_amap_request

    url, params = rewrite_amap_request(
        "https://restapi.amap.com/v3/staticmap?location=121,31&key=DUMMY",
        None,
        _FakeCfg("http://relay:8849", "tok"),
    )
    assert url == "http://relay:8849/amap/v3/staticmap"
    assert params["location"] == "121,31"
    assert "key" not in params
    assert params["relay_token"] == "tok"


def test_rewrite_preserves_priority_params_over_query():
    """URL query 与 params 冲突时 params 优先（与 requests 语义一致）。"""
    from app.amap_relay.client import rewrite_amap_request

    url, params = rewrite_amap_request(
        "https://restapi.amap.com/v3/x?keywords=from_url",
        {"keywords": "from_params"},
        _FakeCfg("http://relay:8849", None),
    )
    assert params["keywords"] == "from_params"
    assert "relay_token" not in params  # 没配令牌就不带（服务端会拒绝）


def test_relay_enabled_flag():
    from app.amap_relay.client import amap_relay_enabled

    assert amap_relay_enabled(_FakeCfg(None)) is False
    assert amap_relay_enabled(_FakeCfg("http://relay:8849")) is True


# ---------------------------------------------------------------------------
# 2. server：鉴权与路径白名单
# ---------------------------------------------------------------------------

@pytest.fixture()
def relay_app(monkeypatch):
    """构造带测试令牌的中转应用。

    用替身替换 ``Config._file_env_value``（.env 的实际读取点），避免
    读到开发机的真实 .env；``_env_value`` 走到 file_val 分支即拿到
    这里的固定值。
    """
    fake_env = {"AMAP_KEY": "server-side-key", "AMAP_RELAY_TOKEN": "testtok"}
    from app.config import AppConfig

    monkeypatch.setattr(
        AppConfig, "_file_env_value", lambda self, name: fake_env.get(name))
    from app.amap_relay.server import create_app

    return create_app()


def test_healthz_reports_config_status(relay_app):
    from fastapi.testclient import TestClient

    with TestClient(relay_app) as client:
        data = client.get("/healthz").json()
    assert data["ok"] is True
    assert data["token_configured"] is True


def test_relay_rejects_bad_token(relay_app):
    from fastapi.testclient import TestClient

    with TestClient(relay_app) as client:
        resp = client.get(
            "/amap/v3/place/text",
            params={"relay_token": "wrong", "keywords": "x"})
    assert resp.status_code == 401
    assert resp.json()["info"] == "INVALID_RELAY_TOKEN"


def test_relay_rejects_when_token_missing(relay_app):
    """没带令牌也拒绝（header / query 两种途径都试）。"""
    from fastapi.testclient import TestClient

    with TestClient(relay_app) as client:
        assert client.get("/amap/v3/place/text").status_code == 401
        assert client.get(
            "/amap/v3/place/text",
            headers={"X-Relay-Token": "testtok"}).status_code != 401


def test_relay_rejects_non_version_path(relay_app):
    """路径白名单：v3/v4 之外的路径直接 403，不给探路空间。"""
    from fastapi.testclient import TestClient

    with TestClient(relay_app) as client:
        resp = client.get(
            "/amap/not-a-version/evil",
            params={"relay_token": "testtok"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 3. server：转发链路（上游替身）
# ---------------------------------------------------------------------------

def test_relay_forwards_and_injects_key(relay_app, monkeypatch):
    """验证四件事：上游 URL 正确、key 被替换、relay_token 被剥离、
    响应原样透传。"""
    import app.amap_relay.server as server_mod
    from fastapi.testclient import TestClient

    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200
        content = b'{"status":"1","info":"OK"}'
        headers = {"content-type": "application/json"}

    def _fake_request(method, url, params=None, data=None,
                      headers=None, timeout=None):
        captured.update(method=method, url=url, params=params,
                        data=data, headers=headers)
        return _FakeResp()

    relay_app.state.session = _FakeSession(_fake_request)

    with TestClient(relay_app) as client:
        resp = client.get(
            "/amap/v3/place/text",
            params={"key": "CLIENT_DUMMY", "keywords": "上海站",
                    "relay_token": "testtok"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "1"
    assert captured["url"].startswith(server_mod.UPSTREAM)
    assert "/v3/place/text" in captured["url"]
    assert captured["params"]["key"] == "server-side-key"  # 服务器侧注入
    assert "relay_token" not in captured["params"]          # 令牌不出机房


class _FakeSession:
    """requests.Session 的最小替身：只暴露 request()。"""

    def __init__(self, fn):
        self._fn = fn
        self.headers: dict[str, str] = {}

    def request(self, method, url, **kwargs):
        return self._fn(method, url, **kwargs)


def test_relay_upstream_error_returns_502(relay_app, monkeypatch):
    import app.amap_relay.server as server_mod
    from fastapi.testclient import TestClient

    def _boom(*args, **kwargs):
        raise ConnectionError("network down")

    relay_app.state.session = _FakeSession(_boom)

    with TestClient(relay_app) as client:
        resp = client.get(
            "/amap/v3/place/text", params={"relay_token": "testtok"})
    assert resp.status_code == 502
    assert resp.json()["info"] == "UPSTREAM_ERROR"
