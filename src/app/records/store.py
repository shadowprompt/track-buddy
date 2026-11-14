"""运动记录的 COS 存储层。

* ``RecordStore`` 是协议：测试注入内存 Fake，生产用 ``CosRecordStore``；
* 官方建议每 region 复用单例 ``CosS3Client``，这里按凭据做进程级缓存；
* 记录对象只追加、不读-改-写；唯一共享可变对象是 ``index.json``，
  用 uid 锁串行化 + 写后 ETag 复核 + 可重建来自愈。
"""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any, Protocol

from ..config import CONFIG
from . import keys

DEFAULT_PAGE = 20
MAX_PAGE = 100
_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}


class RecordStore(Protocol):
    """server / service 只依赖这 6 个方法，便于桩件化。"""

    def put_record(self, uid: str, rec_id: str, *, meta: dict,
                   summary: dict, images: dict[str, bytes]) -> None: ...

    def list_records(self, uid: str, *, cursor: str | None,
                     limit: int) -> tuple[list[dict], str | None, int]: ...

    def get_record(self, uid: str, rec_id: str) -> dict | None: ...

    def get_image(self, uid: str, rec_id: str,
                  name: str) -> tuple[bytes, str] | None: ...

    def delete_record(self, uid: str, rec_id: str) -> bool: ...

    def rebuild_index(self, uid: str) -> int: ...


def cos_sdk_available() -> bool:
    """惰性探测 SDK；未装包也必须保证 ``import app.server`` 正常。"""
    try:
        import qcloud_cos  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


_CLIENT = None
_CLIENT_KEY: tuple | None = None
_CLIENT_LOCK = threading.Lock()


def _cos_client():
    """按 (id, key, region) 缓存单例；换凭据时自动重建。"""
    global _CLIENT, _CLIENT_KEY
    cfg_key = (CONFIG.cos_secret_id, CONFIG.cos_secret_key, CONFIG.cos_region)
    if _CLIENT is not None and _CLIENT_KEY == cfg_key:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT_KEY != cfg_key:
            from qcloud_cos import CosConfig, CosS3Client
            conf = CosConfig(Region=cfg_key[2], SecretId=cfg_key[0],
                             SecretKey=cfg_key[1], Scheme="https")
            _CLIENT = CosS3Client(conf)
            _CLIENT_KEY = cfg_key
    return _CLIENT


_UID_LOCKS: dict[str, threading.Lock] = {}
_UID_LOCKS_GUARD = threading.Lock()


def _uid_lock(uid: str) -> threading.Lock:
    with _UID_LOCKS_GUARD:
        return _UID_LOCKS.setdefault(uid, threading.Lock())


def encode_cursor(created_at: int, rec_id: str) -> str:
    raw = json.dumps({"created_at": int(created_at), "id": rec_id},
                     separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_cursor(cursor: str | None) -> tuple[int, str] | None:
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        obj = json.loads(raw)
        created_at, rec_id = obj["created_at"], obj["id"]
        if not isinstance(created_at, int) or not isinstance(rec_id, str):
            raise ValueError("bad cursor")
        return created_at, rec_id
    except Exception as exc:  # noqa: BLE001
        raise ValueError("bad cursor") from exc


def _sort_key(card: dict) -> tuple[int, str]:
    try:
        ts = int(card.get("created_at") or 0)
    except (TypeError, ValueError):
        ts = 0
    return ts, str(card.get("id") or "")


class CosRecordStore:
    """用 COS SDK 实现的记录存储（客户端可注入，便于离线测试）。"""

    def __init__(self, client=None, *, bucket: str | None = None,
                 prefix: str | None = None):
        self._client = client if client is not None else _cos_client()
        self.bucket = bucket or CONFIG.cos_bucket
        self.prefix = prefix or CONFIG.cos_prefix

    # --- 底层原语 -----------------------------------------------------
    def _k(self, fn, *args) -> str:
        return fn(*args, prefix=self.prefix)

    def _put_bytes(self, key: str, data: bytes) -> str | None:
        resp = self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        etag = (resp or {}).get("ETag") if isinstance(resp, dict) else None
        return str(etag).strip('"') if etag else None

    def _put_json(self, key: str, value: Any) -> str | None:
        return self._put_bytes(
            key, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def _get_bytes(self, key: str) -> bytes | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].get_raw_stream().read()
        except Exception:  # noqa: BLE001 — 对象不存在/网络错误都视为无数据
            return None

    def _get_json(self, key: str) -> Any | None:
        raw = self._get_bytes(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return None

    def _etag(self, key: str) -> str | None:
        try:
            resp = self._client.head_object(Bucket=self.bucket, Key=key)
            return str((resp or {}).get("ETag", "")).strip('"') or None
        except Exception:  # noqa: BLE001
            return None

    def _list_all_keys(self, prefix: str) -> list[dict]:
        """列某前缀下全部对象（自动翻页），返回 Contents 元素列表。"""
        out: list[dict] = []
        marker = ""
        for _ in range(100):  # 硬上限，防异常数据导致死循环
            resp = self._client.list_objects(
                Bucket=self.bucket, Prefix=prefix, Delimiter="",
                Marker=marker, MaxKeys=1000)
            out.extend(resp.get("Contents") or [])
            nxt = resp.get("NextMarker") or ""
            truncated = str(resp.get("IsTruncated", "false")).lower() == "true"
            if not truncated and not nxt:
                break
            if not nxt or nxt == marker:
                break
            marker = nxt
        return out

    # --- RecordStore --------------------------------------------------
    def put_record(self, uid: str, rec_id: str, *, meta: dict,
                   summary: dict, images: dict[str, bytes]) -> None:
        if not (keys.safe_component(uid) and keys.safe_component(rec_id)):
            raise ValueError("unsafe uid or record id")
        # 1) 图片先写：它们是独立不可变对象，失败也只在 index 之前留下孤儿。
        for name, data in images.items():
            if not keys.safe_component(name):
                raise ValueError(f"unsafe image name: {name}")
            self._put_bytes(self._k(keys.image_key, uid, rec_id, name), data)
        # 2) 权威记录对象
        self._put_json(self._k(keys.meta_key, uid, rec_id), meta)
        self._put_json(self._k(keys.summary_key, uid, rec_id), summary)
        # 3) 派生 index（唯一一处读-改-写）
        self._update_index(uid, summary)

    def _update_index(self, uid: str, card: dict) -> None:
        index = self._k(keys.index_key, uid)
        with _uid_lock(uid):
            for _ in range(3):
                items = self._get_json(index)
                if not isinstance(items, list):
                    items = []
                items = [x for x in items
                         if isinstance(x, dict) and x.get("id") != card.get("id")]
                items.append(dict(card))
                items.sort(key=_sort_key, reverse=True)
                wrote = self._put_json(index, items)
                # best-effort：写完后若 ETag 已不是我们写下的那个，说明
                # 期间有别的进程覆盖过，重读合并再写（不依赖 COS 条件写）。
                if wrote and self._etag(index) not in (None, wrote):
                    continue
                return

    def list_records(self, uid: str, *, cursor: str | None,
                     limit: int) -> tuple[list[dict], str | None, int]:
        limit = max(1, min(int(limit or DEFAULT_PAGE), MAX_PAGE))
        point = decode_cursor(cursor)
        index = self._k(keys.index_key, uid)
        items = self._get_json(index)
        if not isinstance(items, list):
            self.rebuild_index(uid)
            items = self._get_json(index)
            if not isinstance(items, list):
                items = []
        items = [x for x in items if isinstance(x, dict)]
        items.sort(key=_sort_key, reverse=True)
        total = len(items)
        if point is not None:
            items = [x for x in items if _sort_key(x) < point]
        page = items[:limit]
        next_cursor = None
        if len(items) > limit and page:
            last = page[-1]
            next_cursor = encode_cursor(_sort_key(last)[0], str(last.get("id") or ""))
        return page, next_cursor, total

    def rebuild_index(self, uid: str) -> int:
        """index 丢失/损坏时，从不可变的 summary.json 全量重建。"""
        prefix = self._k(keys.user_prefix, uid) + "records/"
        cards: list[dict] = []
        marker = ""
        for _ in range(100):
            resp = self._client.list_objects(
                Bucket=self.bucket, Prefix=prefix, Delimiter="/",
                Marker=marker, MaxKeys=1000)
            for entry in resp.get("CommonPrefixes") or []:
                rec_id = str(entry.get("Prefix", "")).rstrip("/").rsplit("/", 1)[-1]
                if not keys.safe_component(rec_id):
                    continue
                card = self._get_json(self._k(keys.summary_key, uid, rec_id))
                if isinstance(card, dict):
                    cards.append(card)
            nxt = resp.get("NextMarker") or ""
            truncated = str(resp.get("IsTruncated", "false")).lower() == "true"
            if not truncated and not nxt:
                break
            if not nxt or nxt == marker:
                break
            marker = nxt
        cards.sort(key=_sort_key, reverse=True)
        self._put_json(self._k(keys.index_key, uid), cards)
        return len(cards)

    def get_record(self, uid: str, rec_id: str) -> dict | None:
        if not (keys.safe_component(uid) and keys.safe_component(rec_id)):
            return None
        meta = self._get_json(self._k(keys.meta_key, uid, rec_id))
        if meta is None:
            return None
        summary = self._get_json(self._k(keys.summary_key, uid, rec_id)) or {}
        images = []
        for entry in self._list_all_keys(self._k(keys.images_prefix, uid, rec_id)):
            name = str(entry.get("Key", "")).rsplit("/", 1)[-1]
            if not name or not keys.safe_component(name):
                continue
            images.append({
                "name": name,
                "url": f"/api/records/{rec_id}/images/{name}",
                "size": entry.get("Size"),
            })
        return {
            "id": rec_id,
            "created_at": summary.get("created_at")
            or int(time.time()),
            "summary": summary,
            "payload": meta,
            "images": images,
        }

    def get_image(self, uid: str, rec_id: str,
                  name: str) -> tuple[bytes, str] | None:
        if not all(keys.safe_component(x) for x in (uid, rec_id, name)):
            return None
        raw = self._get_bytes(self._k(keys.image_key, uid, rec_id, name))
        if raw is None:
            return None
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        return raw, _MEDIA_TYPES.get(suffix, "application/octet-stream")

    def delete_record(self, uid: str, rec_id: str) -> bool:
        if not (keys.safe_component(uid) and keys.safe_component(rec_id)):
            return False
        objects = self._list_all_keys(self._k(keys.record_prefix, uid, rec_id))
        keys_to_delete = [str(o.get("Key")) for o in objects if o.get("Key")]
        if not keys_to_delete:
            return False
        for start in range(0, len(keys_to_delete), 1000):
            batch = keys_to_delete[start:start + 1000]
            self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Quiet": "true",
                        "Object": [{"Key": k} for k in batch]})
        index = self._k(keys.index_key, uid)
        with _uid_lock(uid):
            items = self._get_json(index)
            if isinstance(items, list):
                self._put_json(index, [x for x in items
                                       if not (isinstance(x, dict)
                                               and x.get("id") == rec_id)])
        return True


_STORE_OVERRIDE: RecordStore | None = None


def get_record_store() -> RecordStore | None:
    """工厂：测试用 ``_STORE_OVERRIDE`` 注入；未启用时返回 None。"""
    if _STORE_OVERRIDE is not None:
        return _STORE_OVERRIDE
    from . import records_enabled
    if not records_enabled():
        return None
    return CosRecordStore()
