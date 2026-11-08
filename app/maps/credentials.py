"""统一凭证解析：环境变量 > .env > 未配置。

过去 ``roadname._amap_key()`` 自己 ``os.environ.get``、完全绕开
AppConfig 的 .env 支持；``config.tianditu_key`` 又单独维护一份别名表。
这里收口成一处：provider 只声明 ``credential_aliases``，取值逻辑只有
这一份，新增厂商不用再写一遍「找 key」的代码。

为了不引入第二套 .env 解析语义（环境变量优先、.env 热更新），默认
委托给 ``AppConfig._env_value``；传入简单配置桩时退回 ``os.environ``。
"""

from __future__ import annotations

import os
from typing import Any


class CredentialStore:
    """按别名顺序解析密钥；别名顺序即优先级。"""

    def __init__(self, cfg: Any | None = None):
        self.cfg = cfg

    def _value(self, name: str) -> str | None:
        getter = getattr(self.cfg, "_env_value", None)
        if callable(getter):
            return getter(name)
        return os.environ.get(name)

    def resolve(self, *aliases: str) -> str | None:
        """返回第一个非空值；全都没有时返回 None（不抛异常）。"""
        for name in aliases:
            if not name:
                continue
            raw = self._value(name)
            if raw is None:
                continue
            val = str(raw).strip()
            if val:
                return val
        return None

    def has(self, *aliases: str) -> bool:
        return self.resolve(*aliases) is not None

    def resolve_for(self, meta: Any) -> str | None:
        """按 ProviderMeta.credential_aliases 解析，供注册表统一调用。"""
        return self.resolve(*getattr(meta, "credential_aliases", ()))
