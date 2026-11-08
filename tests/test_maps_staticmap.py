"""能力⑥（静态地图素材）的回归测试（阶段 4）。

三份实验脚本曾经各抄一份「拼 URL + 取图」的实现。这组测试锁住：

1. URL 拼装正确（参数、坐标顺序、size 格式）；
2. 三个实验脚本**不再**自带 URL 拼接，全部走统一 provider；
3. 缺 key / 请求失败时返回 None 而不是抛异常（实验脚本据此跳过）；
4. 腾讯与高德两个 provider 的参数形态各自正确。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ provider


def test_amap_url_shape() -> None:
    from app.maps.staticmap import fetch_static_map

    captured: dict[str, object] = {}

    class FakeResp:
        def read(self) -> bytes:
            return b"PNGDATA"

        def __enter__(self) -> "FakeResp":
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def opener(req, timeout=None):  # noqa: ANN001, ANN202
        captured["url"] = req.full_url
        captured["ua"] = req.get_header("User-agent")
        return FakeResp()

    res = fetch_static_map(121.46, 31.20, 17, 1024,
                           provider="amap", key="K1", opener=opener)
    assert res is not None
    url = str(captured["url"])
    assert "restapi.amap.com/v3/staticmap" in url
    assert "location=121.46%2C31.2" in url or "location=121.46,31.2" in url
    assert "zoom=17" in url
    assert "size=1024%2A1024" in url or "size=1024*1024" in url
    assert "key=K1" in url
    assert res.data == b"PNGDATA"
    assert res.provider == "amap"


def test_tmap_url_shape() -> None:
    from app.maps.staticmap import fetch_static_map

    captured: dict[str, object] = {}

    class FakeResp:
        def read(self) -> bytes:
            return b"X"

        def __enter__(self) -> "FakeResp":
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def opener(req, timeout=None):  # noqa: ANN001, ANN202
        captured["url"] = req.full_url
        return FakeResp()

    res = fetch_static_map(121.46, 31.20, 17, 800,
                           provider="tmap", key="K2", opener=opener)
    assert res is not None
    url = str(captured["url"])
    assert "apis.map.qq.com/ws/staticmap" in url
    # 腾讯用 center=lat,lon 且尺寸分开写
    assert "center=31.2%2C121.46" in url or "center=31.2,121.46" in url
    assert "width=800" in url and "height=800" in url
    assert res.provider == "tmap"


def test_missing_key_returns_none() -> None:
    from app.maps.staticmap import fetch_static_map

    assert fetch_static_map(121.46, 31.20, 17, key=None) is None
    assert fetch_static_map(121.46, 31.20, 17, key="") is None


def test_request_failure_returns_none() -> None:
    """请求失败返回 None（实验脚本据此跳过），不能抛异常打断整批实验。"""
    from app.maps.staticmap import fetch_static_map

    def opener(req, timeout=None):  # noqa: ANN001, ANN202
        raise OSError("network down")

    assert fetch_static_map(121.46, 31.20, 17, key="K",
                            opener=opener) is None


def test_amap_key_reads_config() -> None:
    from app.maps.staticmap import amap_key

    class Cfg:
        def amap_key(self) -> str:
            return "FROM_CFG"

    assert amap_key(Cfg()) == "FROM_CFG"


# ------------------------------------------------------------------ tools


_TOOL_FILES = [
    "exp_basemap_match.py",
    "exp_e2e_multishot.py",
    "exp_overlap_need.py",
]


@pytest.mark.parametrize("name", _TOOL_FILES)
def test_tools_use_shared_provider(name: str) -> None:
    """三个实验脚本必须走统一 provider，不能再自带 URL 拼接。"""
    path = TOOLS / name
    assert path.exists(), f"缺少脚本 {name}"
    src = path.read_text(encoding="utf-8")
    assert "from app.maps.staticmap import" in src, \
        f"{name} 未使用统一静态地图 provider"
    assert "restapi.amap.com" not in src, \
        f"{name} 仍在手写高德静态地图 URL"


@pytest.mark.parametrize("name", _TOOL_FILES)
def test_tools_parse_and_import_provider(name: str) -> None:
    """脚本能被解析，且确实 import 了 provider（防止改坏导入）。"""
    path = TOOLS / name
    tree = _parse(path)
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "app.maps.staticmap":
                imported = True
    assert imported, f"{name} 未真正 import app.maps.staticmap"


def test_no_duplicate_staticmap_url_in_repo() -> None:
    """仓库里静态地图端点只应出现在统一实现里。"""
    hits: list[str] = []
    for base in (ROOT / "app", ROOT / "tools"):
        for p in base.rglob("*.py"):
            if "staticmap" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "restapi.amap.com/v3/staticmap" in text:
                hits.append(str(p.relative_to(ROOT)))
    assert not hits, f"以下文件仍在手写静态地图 URL: {hits}"
