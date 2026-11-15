"""记录保存服务：把一次识别的 payload + 图片整理成 COS 记录。

刻意与 server 解耦：所有异常在这里吞掉并只记 warning，
识别接口绝不能因为云存储失败而返回 500。
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

from .keys import build_summary_card
from .store import RecordStore

log = logging.getLogger(__name__)

_LOCAL_URL_KEYS = ("shot_url", "overview_url", "stitched_url")


def normalize_payload_for_store(payload: dict[str, Any], rec_id: str) -> dict:
    """复制一份完整 payload，并把只在本机有效的 /api/shots URL 改写到记录端点。

    用 ``deepcopy`` 是硬要求：调用方还要把原 payload 返回给客户端，
    原地改写会把本次响应里的截图链接弄坏。
    """
    meta = copy.deepcopy(payload)
    for key in _LOCAL_URL_KEYS:
        url = meta.get(key)
        if isinstance(url, str) and "/api/shots/" in url:
            name = url.rsplit("/", 1)[-1]
            meta[key] = f"/api/records/{rec_id}/images/{name}"
    meta["record_id"] = rec_id
    meta["saved_at"] = int(time.time())
    return meta


def try_save_record(*, uid: str, rec_id: str, payload: dict[str, Any],
                    image_paths: list[Path] | None = None,
                    summary_bytes: bytes = b"",
                    summary_name: str | None = None,
                    city: str = "", union_uid: str = "",
                    store: RecordStore | None) -> bool:
    """带合法 token 时才调用；失败一律返回 False，不抛给识别端点。"""
    if store is None:
        return False
    try:
        images: dict[str, bytes] = {}
        for path in image_paths or []:
            if not isinstance(path, Path) or not path.is_file():
                continue
            data = path.read_bytes()
            if data:
                images[path.name] = data
        if summary_bytes:
            images[summary_name or f"{rec_id}_summary.jpg"] = summary_bytes
        meta = normalize_payload_for_store(payload, rec_id)
        if union_uid:
            # 只写脱敏锚点，不写明文 unionid / openid
            meta["union_uid"] = union_uid
        card = build_summary_card(payload, city, rec_id, images=images)
        store.put_record(uid, rec_id, meta=meta, summary=card, images=images)
        if union_uid:
            # 别名索引是派生缓存：写失败只记 warning，记录本身已经保存成功。
            try:
                store.register_alias(union_uid, uid)
            except Exception as exc:  # noqa: BLE001
                log.warning("跨端身份别名写入失败（不影响记录）：%s", exc)
        return True
    except Exception as exc:  # noqa: BLE001 — 云存储失败不能影响识别结果
        log.warning("运动记录保存 COS 失败（不影响本次识别）：%s", exc)
        return False


def record_to_response(record: dict[str, Any]) -> dict[str, Any]:
    """端点返回前的最后一道整形（当前为恒等，保留扩展位）。"""
    return {
        "id": record.get("id"),
        "created_at": record.get("created_at"),
        "summary": record.get("summary") or {},
        "payload": record.get("payload") or {},
        "images": record.get("images") or [],
    }
