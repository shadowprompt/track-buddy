"""小程序记录链路（JS 侧）契约：语法、静态接线、Node 桩件行为。

后端运动记录云存储的契约在 tests/test_records_cos.py；本文件只管 ridewx 的
JS 逻辑：登录态、Bearer 注入、云列表 + 本机兜底、历史详情与删除。
所有断言只读 .js 源码（这些文件是本轮随代码一起提交的），不依赖用户本地
未提交的布局 WIP。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RIDEWX = ROOT / "ridewx"
JS_FILES = sorted(RIDEWX.rglob("*.js"))


def _read(rel: str) -> str:
    return (RIDEWX / rel).read_text(encoding="utf-8")


# --- 1. 语法 ---------------------------------------------------------------

def test_all_miniprogram_js_parse():
    """每个 .js 都必须能被 JS 解析器接受（小程序编译期错误成本很高）。"""
    if shutil.which("node") is None:
        pytest.skip("未安装 Node.js，跳过 JS 语法检查")
    assert JS_FILES
    for f in JS_FILES:
        cp = subprocess.run(["node", "--check", str(f)],
                            capture_output=True, text=True, timeout=60)
        assert cp.returncode == 0, f"{f.name}: {cp.stderr}"


# --- 2. 静态接线契约 -------------------------------------------------------

def test_app_token_lifecycle_wired():
    app_js = _read("app.js")
    for needle in ("/api/health", "api.wxlogin", "ensureToken",
                   "invalidateToken", "wx.login", "expires_at"):
        assert needle in app_js, needle
    # 冷启动必须先看开关，未启用时不做微信登录
    assert "d.records.enabled" in app_js or "records.enabled" in app_js


def test_api_records_endpoints_and_bearer():
    api_js = _read("utils/api.js")
    for needle in ("/api/auth/wxlogin", "/api/records",
                   "listRecords", "getRecord", "deleteRecord",
                   "downloadRecordImage", "Authorization", "Bearer ",
                   "err.statusCode !== 401", "force: true"):
        assert needle in api_js, needle
    # 识别端点必须走带鉴权的请求（服务端据此自动写 COS）
    assert "authedRequest('POST', '/api/recognize'" in api_js
    # wxlogin 不能要求 Bearer（鸡生蛋问题）
    assert "rawRequest('POST', '/api/auth/wxlogin'" in api_js
    # 客户端绝对不能出现云端凭据
    for secret in ("COS_SECRET_ID", "COS_SECRET_KEY", "WX_APP_SECRET",
                   "qcloud_cos"):
        assert secret not in api_js, secret


def test_pages_use_cloud_records_with_local_fallback():
    rec = _read("pages/records/records.js")
    me = _read("pages/me/me.js")
    idx = _read("pages/index/index.js")
    result = _read("pages/result/result.js")
    util = _read("utils/records.js")

    assert "api.listRecords" in rec and "records.merge" in rec
    assert "api.deleteRecord" in rec and "openRecord" in rec
    assert "api.listRecords" in me and "records.statsOf" in me
    # 识别成功后写一份本机摘要（云端不可用时兜底）
    assert "records.upsert(" in idx
    # 历史记录详情：按 recordId 从云端拉完整 payload + 代理流图片
    assert "recordId" in result and "api.getRecord(" in result
    assert "api.downloadRecordImage(" in result
    # 页面不再直接读本地 Storage，统一走 utils/records.js
    for src in (rec, me, idx, result):
        assert "getStorageSync('ride_records_v1')" not in src
    assert "ride_records_v1" in util
    assert "cardToView" in util and "statsOf" in util


def test_miniprogram_wording_matches_web():
    """「模板」「订单详情」是内部/旧话术，用户可见文案对齐网页版。"""
    for rel in ("app.js", "config.js", "utils/api.js", "utils/records.js",
                "utils/multipart.js", "pages/index/index.js",
                "pages/records/records.js", "pages/me/me.js",
                "pages/result/result.js"):
        src = _read(rel)
        for old in ("订单详情", "订单里程", "订单摘要", "参考历史模板",
                    "优先模板", "禁用模板"):
            assert old not in src, f"{rel}: {old}"
    assert "参考历史记录" in _read("pages/result/result.js")
    assert "概要里程" in _read("pages/result/result.js")
    assert "优先历史记录" in _read("pages/index/index.js")


# --- 3. Node 桩件行为 ------------------------------------------------------

def test_miniprogram_records_runtime():
    """假 wx + 假后端跑真实模块：登录/零回归/401 重试/列表/兜底/详情/删除。"""
    if shutil.which("node") is None:
        pytest.skip("未安装 Node.js，跳过小程序 JS 桩件测试")
    harness = Path(__file__).with_name("miniprogram_records_test.js")
    cp = subprocess.run(["node", str(harness)],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(ROOT))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "全部通过" in cp.stdout
    assert "✗" not in cp.stdout
