"""COS 对象键与列表卡片：**唯一**的键构造处。

设计要点（见 README「运动记录云存储」）：

* ``uid`` 由 openid 经 HMAC-SHA256 派生，明文 openid 绝不出现在对象键、
  meta 或 token 里；
* 每条记录的对象是**不可变、只追加**的，绝不读-改-写同一记录对象；
* ``index.json`` 只是派生缓存，丢失/损坏可由记录对象重建。
"""
from __future__ import annotations

import hashlib
import hmac
import re
import time
from typing import Any

DEFAULT_PREFIX = "ride-records/"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def uid_from_openid(openid: str, secret: str) -> str:
    """openid → 稳定的 32 位十六进制 uid（换密钥等于换用户，需迁移数据）。"""
    return hmac.new(
        secret.encode("utf-8"), openid.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:32]


def safe_component(value: str) -> bool:
    """记录 id / 图片名必须是最简单的安全文件名（防路径穿越）。"""
    return bool(_SAFE_COMPONENT.match(value or "")) and ".." not in value


def user_prefix(uid: str, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{prefix}users/{uid}/"


def index_key(uid: str, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{user_prefix(uid, prefix)}index.json"


def record_prefix(uid: str, rec_id: str,
                  prefix: str = DEFAULT_PREFIX) -> str:
    return f"{user_prefix(uid, prefix)}records/{rec_id}/"


def meta_key(uid: str, rec_id: str, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{record_prefix(uid, rec_id, prefix)}meta.json"


def summary_key(uid: str, rec_id: str, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{record_prefix(uid, rec_id, prefix)}summary.json"


def image_key(uid: str, rec_id: str, name: str,
              prefix: str = DEFAULT_PREFIX) -> str:
    return f"{record_prefix(uid, rec_id, prefix)}images/{name}"


def images_prefix(uid: str, rec_id: str,
                  prefix: str = DEFAULT_PREFIX) -> str:
    return f"{record_prefix(uid, rec_id, prefix)}images/"


def build_summary_card(payload: dict[str, Any], city: str, rec_id: str, *,
                       images: dict[str, bytes] | None = None,
                       created_at: int | None = None) -> dict[str, Any]:
    """列表卡片：字段自包含，列表接口无需再逐条读 meta（避免 N+1 GET）。"""
    rs = payload.get("ride_summary") or {}
    geo = payload.get("geojson") or {}
    try:
        coords = ((geo.get("features") or [{}])[0]
                  .get("geometry", {}).get("coordinates") or [])
    except (AttributeError, IndexError, TypeError):
        coords = []
    exports = [fmt for fmt in ("gpx", "tcx") if payload.get(fmt)]
    if geo:
        exports.insert(0, "geojson")
    thumb = next(iter(images or {}), None)
    return {
        "id": rec_id,
        "created_at": int(created_at if created_at is not None else time.time()),
        "city": (city or "").strip(),
        "distance_m": payload.get("distance_m"),
        "duration_s": rs.get("duration_s"),
        "calories_kcal": rs.get("calories_kcal"),
        "started_at": rs.get("started_at"),
        "distance_km": rs.get("distance_km"),
        "confidence": payload.get("confidence"),
        "track_points": payload.get("track_points"),
        "has_track": len(coords) >= 2,
        "exports": exports,
        "thumb": thumb,
    }
