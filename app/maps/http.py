"""统一 HTTP 基座：Session / UA / 限流 / 重试 / 可选磁盘缓存。

目前 geocode.py、roadname.py、roadfit.py、osrm.py、valhalla.py 各自
维护一份 Session 与限流字典，Overpass 镜像表还重复了三份。阶段 2
会把这些调用收敛到这里；本模块先提供可独立测试的实现，不改任何
现有调用点。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests


class HttpClient:
    """一个能力内共享的 HTTP 客户端。

    ``session`` 可注入假实现，便于离线测试；``cache=True`` 时按
    URL + 参数做磁盘缓存，用于路网这类「同一区域反复请求」的数据。
    """

    def __init__(
        self,
        *,
        timeout: float = 12.0,
        user_agent: str = "meituan-ride/0.1 (+local)",
        min_interval: float = 0.0,
        retries: int = 1,
        cache_dir: Path | None = None,
        cache_ttl_s: float = 86400.0,
        session: Any | None = None,
    ):
        self.timeout = float(timeout)
        self.min_interval = float(min_interval)
        self.retries = max(0, int(retries))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl_s = float(cache_ttl_s)
        self._session = session if session is not None else requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def session(self) -> Any:
        return self._session

    @staticmethod
    def _host(url: str) -> str:
        return url.split("/")[2] if "//" in url else url

    def _throttle(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self.min_interval - (time.time() - self._last_call.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.time()

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _cache_read(self, path: Path) -> Any | None:
        try:
            if not path.exists():
                return None
            if time.time() - path.stat().st_mtime > self.cache_ttl_s:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _cache_key(url: str, params: dict[str, Any] | None) -> str:
        return url + "?" + json.dumps(
            params or {}, sort_keys=True, ensure_ascii=False
        )

    def get_json(self, url: str, *, params: dict[str, Any] | None = None,
                 cache: bool = False) -> Any:
        """GET 并解析 JSON；失败重试后抛 RuntimeError。"""
        path = self._cache_path(self._cache_key(url, params)) if cache else None
        if path is not None:
            hit = self._cache_read(path)
            if hit is not None:
                return hit
        host = self._host(url)
        last_exc: Exception | None = None
        for _ in range(self.retries + 1):
            self._throttle(host)
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                data = response.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                continue
            if path is not None:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                except (OSError, TypeError, ValueError):
                    pass  # 缓存写失败不影响本次结果
            return data
        raise RuntimeError(f"HTTP 请求失败: {url} ({last_exc})")

    def try_get_json(self, url: str, *, params: dict[str, Any] | None = None,
                     cache: bool = False) -> Any | None:
        """宽松版本：任何失败返回 None，供「返回空就落下一个源」使用。"""
        try:
            return self.get_json(url, params=params, cache=cache)
        except Exception:  # noqa: BLE001 - 回退链路由调用方决定
            return None
