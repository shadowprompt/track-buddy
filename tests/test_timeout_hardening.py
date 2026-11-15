"""超时加固契约：冷启动预热、COS 显式超时、云存储后台化、504 文案。

纯静态 + 纯函数级：不联网、不加载 OCR 模型、不启任何服务进程。
起因：线上首次识别撞上 Nginx 默认 60s 读超时，返回 504 Gateway Time-out。
"""
from __future__ import annotations

from pathlib import Path

from app import ocr_anchor
from app.config import CONFIG

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_startup_warms_up_ocr_without_blocking() -> None:
    src = _read("run.py")
    assert "_warmup_ocr_async()" in src
    assert "warmup_ocr" in src
    assert "daemon=True" in src          # 不阻塞 uvicorn 起监听
    # 绝不能放进 app.server 的 startup：TestClient 会触发，几百个用例
    # 都要加载一遍 onnxruntime 模型。
    assert "warmup_ocr" not in _read("src/app/server.py")


def test_ocr_singleton_is_locked() -> None:
    src = _read("src/app/ocr_anchor.py")
    assert "def warmup_ocr(" in src
    assert "self._lock = threading.Lock()" in src
    assert src.count("if self._engine is None:") >= 2  # 双重检查
    assert hasattr(ocr_anchor._OCR, "_lock")


def test_warmup_never_raises(monkeypatch) -> None:
    def boom(self):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ocr_anchor.OcrEngine, "_get", boom)
    assert ocr_anchor.warmup_ocr() is False


def test_cos_http_timeout_bounds() -> None:
    value = CONFIG.cos_http_timeout
    assert isinstance(value, float)
    assert 3.0 <= value <= 120.0


def test_cos_client_sets_explicit_timeout() -> None:
    src = _read("src/app/records/store.py")
    assert "CONFIG.cos_http_timeout" in src
    assert "Timeout=cfg_key[3]" in src


def test_record_save_moved_to_background_tasks() -> None:
    src = _read("src/app/server.py")
    assert "    BackgroundTasks,\n" in src
    assert src.count("background_tasks: BackgroundTasks,") == 2
    assert src.count("background_tasks.add_task(") == 2
    assert src.count("        _maybe_save_record,\n") == 2


def test_miniprogram_explains_gateway_timeout() -> None:
    src = _read("wxmin/utils/api.js")
    assert "code === 504 || code === 502" in src
    assert "proxy_read_timeout" in src
    assert "/timeout/i.test(msg)" in src
