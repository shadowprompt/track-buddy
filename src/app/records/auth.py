"""微信登录、HMAC 签名 token 与请求级身份解析。

安全边界：

* ``session_key`` / ``openid`` / 微信的 ``errcode``/``errmsg`` 一律不进响应；
* token 只含派生的 ``uid``，用 stdlib HMAC 签名（不引入 PyJWT）；
* ``optional_record_uid`` 用于识别端点：任何异常都返回 None，**零回归**；
* ``require_record_uid`` 用于 /api/records*：按 403 / 401 明确拒绝。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import requests
from fastapi import HTTPException, Request

from ..config import CONFIG

WX_JSCODE2SESSION_URL = "https://api.weixin.qq.com/sns/jscode2session"


class TokenError(Exception):
    """token 无效 / 过期 / 微信 code 不合法（统一不向客户端解释原因）。"""


class WeChatServiceError(Exception):
    """微信接口网络故障（端点据此返回 502）。"""


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64u(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_token(uid: str, *, secret: str, ttl: int,
                now: int | None = None) -> str:
    """签发 ``head.body.sig`` 三段式 token（算法固定 HS256，不解析 alg）。"""
    ts = int(now if now is not None else time.time())
    head = _b64u(b'{"alg":"HS256","typ":"RIDE"}')
    claims = {"uid": uid, "iat": ts, "exp": ts + int(ttl), "v": 1}
    body = _b64u(json.dumps(claims, separators=(",", ":"),
                            ensure_ascii=False).encode("utf-8"))
    signing = f"{head}.{body}"
    sig = _b64u(hmac.new(secret.encode("utf-8"), signing.encode("utf-8"),
                         hashlib.sha256).digest())
    return f"{signing}.{sig}"


def verify_token(token: str, *, secret: str,
                 now: int | None = None) -> dict[str, Any]:
    """校验签名/结构/有效期；任一异常统一抛 ``TokenError``。"""
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise TokenError("bad token")
    head, body, sig = parts
    try:
        json.loads(_unb64u(head))
        claims = json.loads(_unb64u(body))
    except Exception as exc:  # noqa: BLE001 — 结构错误统一处理
        raise TokenError("bad token") from exc
    expected = _b64u(hmac.new(secret.encode("utf-8"),
                              f"{head}.{body}".encode("utf-8"),
                              hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise TokenError("bad signature")
    if not isinstance(claims, dict) or claims.get("v") != 1:
        raise TokenError("bad claims")
    uid, iat, exp = claims.get("uid"), claims.get("iat"), claims.get("exp")
    if not isinstance(uid, str) or not uid:
        raise TokenError("bad uid")
    if not isinstance(iat, int) or not isinstance(exp, int) or exp <= iat:
        raise TokenError("bad expiry")
    if int(now if now is not None else time.time()) >= exp:
        raise TokenError("expired")
    return claims


def code2session(code: str, *, http_get=None) -> dict[str, Any]:
    """用 wx.login 的 code 换 openid（session_key 由调用方即刻丢弃）。

    ``http_get`` 可注入，测试无需联网。失败一律抛统一异常，不回显微信
    返回的 errcode/errmsg（避免探测配置是否正确）。
    """
    appid, app_secret = CONFIG.wx_appid, CONFIG.wx_app_secret
    if not (appid and app_secret) or not (code or "").strip():
        raise TokenError("wxlogin unavailable")
    get = http_get or requests.get
    try:
        resp = get(WX_JSCODE2SESSION_URL, params={
            "appid": appid, "secret": app_secret,
            "js_code": code.strip(), "grant_type": "authorization_code",
        }, timeout=CONFIG.http_timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — 网络/解析都算上游故障
        raise WeChatServiceError(str(type(exc).__name__)) from exc
    if not isinstance(data, dict) or data.get("errcode") or not data.get("openid"):
        raise TokenError("wx code rejected")
    return data


def _records_enabled() -> bool:
    # 运行时解析包属性，便于测试用 monkeypatch 覆盖功能开关。
    from . import records_enabled
    return bool(records_enabled())


def _bearer_uid(request: Request) -> str | None:
    raw = request.headers.get("authorization", "")
    if not raw.startswith("Bearer "):
        return None
    secret = CONFIG.records_token_secret
    if not secret:
        return None
    try:
        return str(verify_token(raw[7:].strip(), secret=secret)["uid"])
    except TokenError:
        return None


async def optional_record_uid(request: Request) -> str | None:
    """识别端点用：功能未启用 / 无 token / token 无效 → None（跳过保存）。"""
    if not _records_enabled():
        return None
    return _bearer_uid(request)


async def require_record_uid(request: Request) -> str:
    """记录端点用：403=功能未启用，401=缺 token 或 token 无效。"""
    if not _records_enabled():
        raise HTTPException(status_code=403, detail="运动记录功能未启用")
    uid = _bearer_uid(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="缺少有效 Bearer token")
    return uid
