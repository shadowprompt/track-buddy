"""运动记录云存储门面（server 只 import 这个模块）。

功能开关是双保险：7 项配置齐全 **且** ``qcloud_cos`` 可导入。
未启用时识别端点完全保持旧行为（``optional_record_uid`` 恒为 None），
``/api/records*`` 返回 403，``/api/auth/wxlogin`` 返回 503。
"""
from __future__ import annotations

from ..config import CONFIG
from . import auth, keys, service, store
from .auth import (
    TokenError, WeChatServiceError, code2session, issue_token,
    optional_record_uid, optional_union_uid, require_record_identity,
    require_record_uid, verify_token,
)
from .keys import (alias_key, build_summary_card, uid_from_openid,
                   union_uid_from_unionid)
from .service import normalize_payload_for_store, record_to_response, try_save_record
from .store import (
    CosRecordStore, RecordStore, cos_sdk_available, get_record_store,
)

# 测试可覆盖（monkeypatch records.RECORDS_ENABLED_OVERRIDE 或 records.records_enabled）
RECORDS_ENABLED_OVERRIDE: bool | None = None


def records_enabled() -> bool:
    if RECORDS_ENABLED_OVERRIDE is not None:
        return bool(RECORDS_ENABLED_OVERRIDE)
    return bool(CONFIG.records_config_ready and store.cos_sdk_available())


__all__ = [
    "auth", "keys", "service", "store", "records_enabled",
    "TokenError", "WeChatServiceError", "code2session", "issue_token",
    "verify_token", "optional_record_uid", "optional_union_uid",
    "require_record_identity", "require_record_uid",
    "uid_from_openid", "union_uid_from_unionid", "alias_key",
    "build_summary_card",
    "normalize_payload_for_store", "try_save_record", "record_to_response",
    "RecordStore", "CosRecordStore", "get_record_store", "cos_sdk_available",
]
