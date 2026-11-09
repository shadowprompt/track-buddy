"""前端底图适配层（阶段 3）的回归测试。

覆盖三层：

1. **结构层**（pytest 直接读文件）：业务代码不再直接出现 ``new TMap.*``，
   适配层是唯一的厂商接触面；服务端会注入底图服务商名。
2. **行为层**（Node 跑真实的 ``web/js/basemap.js``）：适配层在腾讯桩上
   仍能「原地更新几何」，且换 provider 不需要改业务代码。
3. **服务端层**：``BASEMAP_PROVIDER`` 能切服务商；没配高德 JS key 时
   自动回退腾讯，避免页面白屏。
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "src" / "web" / "index.html"
ADAPTER_JS = ROOT / "src" / "web" / "js" / "basemap.js"
SCRIPT = Path(__file__).resolve().parent / "frontend_basemap_test.js"


# ---------------------------------------------------------------- 结构层


def test_business_code_has_no_direct_tmap_calls() -> None:
    """业务代码里不应再出现 ``new TMap.*`` 这类厂商调用。"""
    html = HTML.read_text(encoding="utf-8")
    # 只看内联业务脚本（最后一个 <script> 块），适配层在独立文件里。
    blocks = [
        m.group(1) for m in
        __import__("re").finditer(r"<script>([\s\S]*?)</script>", html)
    ]
    code = blocks[-1]
    assert "new TMap." not in code, "业务脚本仍在直接调用 TMap"
    assert "TMap.LatLng" not in code, "业务脚本仍在直接构造 LatLng"
    assert "new TMap.Map" not in code, "业务脚本仍在直接建地图"
    # 适配层入口必须被引入
    assert "/static/js/basemap.js" in html, "页面未引入底图适配层"
    assert "RideBasemap.create" in code, "业务脚本未通过适配层创建底图"


def test_adapter_file_is_self_contained() -> None:
    """适配层必须自带两个服务商实现，且不依赖业务代码。"""
    src = ADAPTER_JS.read_text(encoding="utf-8")
    assert "register('tmap'" in src
    assert "register('amap'" in src
    # 统一接口
    for method in ("createMap", "createTrack", "updateTrack",
                   "removeTrack", "fitBounds", "onClick"):
        assert method in src, f"适配层缺少统一接口方法 {method}"


def test_static_mount_serves_adapter() -> None:
    """``/static`` 挂载的是 web 目录，适配层必须能被访问到。"""
    assert ADAPTER_JS.exists()
    assert ADAPTER_JS.parent.name == "js"


# ---------------------------------------------------------------- 行为层


def test_adapter_behaviour_on_node() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("未安装 Node.js，跳过前端适配层测试")
    proc = subprocess.run(
        [node, str(SCRIPT), str(ADAPTER_JS)],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT),
    )
    assert proc.returncode == 0, (
        "底图适配层测试失败:\n" + proc.stdout[-2500:]
        + "\n--- stderr ---\n" + proc.stderr[-1200:]
    )


# ---------------------------------------------------------------- 服务端层


def test_render_page_injects_basemap_provider() -> None:
    from app.tmap_proxy import render_page

    html = render_page("<b>__WB_BASEMAP_PROVIDER__</b>", 8848, "sec",
                        basemap_provider="amap")
    assert "amap" in html
    assert "__WB_BASEMAP_PROVIDER__" not in html


def test_page_placeholder_not_nested_in_variable_name() -> None:
    """页面里的占位符不能出现在 JS 变量名中。

    否则渲染时占位符子串会被一并替换：
    ``window.__WB_BASEMAP_PROVIDER__`` 会变成 ``window.tmap``——
    变量名被悄悄改掉，语义全错，而且**不报任何错**。

    只看代码行（跳过注释），注释里可以自由提及占位符名。
    """
    html = HTML.read_text(encoding="utf-8")
    code_lines = [
        ln for ln in html.splitlines()
        if "__WB_BASEMAP_PROVIDER__" in ln and not ln.strip().startswith("//")
    ]
    for ln in code_lines:
        assert "window.__WB_BASEMAP_PROVIDER__" not in ln, \
            f"占位符名被用作 JS 变量名，渲染时会被连带替换: {ln.strip()}"
    assert any(
        "window.__WB_MAP_PROVIDER__ = '__WB_BASEMAP_PROVIDER__'" in ln
        for ln in code_lines
    ), "provider 变量名与占位符冲突"


def test_render_page_does_not_corrupt_variable_names() -> None:
    """渲染后的页面里变量名必须保持原样。"""
    from app.tmap_proxy import render_page

    html = render_page(HTML.read_text(encoding="utf-8"), 8848, "sec",
                        basemap_provider="tmap")
    assert "window.__WB_MAP_PROVIDER__ = 'tmap'" in html, \
        "provider 变量名在渲染后被占位符替换破坏了"
    # 代码行里不能再出现被连带替换出来的 window.tmap
    code_lines = [
        ln for ln in html.splitlines()
        if not ln.strip().startswith("//")
    ]
    assert not any("window.tmap" in ln for ln in code_lines), \
        "渲染后代码行里出现了 window.tmap（占位符连带替换）"


def test_render_page_amap_sdk_only_when_selected() -> None:
    from app.tmap_proxy import render_page

    tmap_html = render_page("__WB_AMAP_SDK__", 8848, "sec",
                             basemap_provider="tmap")
    assert "webapi.amap.com" not in tmap_html
    assert "__WB_AMAP_SDK__" not in tmap_html

    amap_html = render_page("__WB_AMAP_SDK__", 8848, "sec",
                             basemap_provider="amap",
                             amap_js_key="JSKEY123",
                             amap_js_security="SEC456")
    assert "webapi.amap.com/maps" in amap_html
    assert "JSKEY123" in amap_html
    assert "SEC456" in amap_html


def test_amap_sdk_absent_without_key() -> None:
    """选了 amap 但没配 JS key 时不注入 SDK（否则页面会白屏）。"""
    from app.tmap_proxy import render_page

    html = render_page("__WB_AMAP_SDK__", 8848, "sec",
                        basemap_provider="amap", amap_js_key=None)
    assert "webapi.amap.com" not in html
    assert "__WB_AMAP_SDK__" not in html


def test_config_basemap_provider_defaults_to_tmap() -> None:
    from app.config import AppConfig

    cfg = AppConfig()
    assert cfg.basemap_provider in ("tmap", "amap")


def test_config_amap_requires_js_key() -> None:
    """没配高德 JS key 时，即使 BASEMAP_PROVIDER=amap 也要回退腾讯。"""
    from app.config import AppConfig

    cfg = AppConfig()
    if not cfg.amap_js_key:
        assert cfg.basemap_provider == "tmap"


def test_config_basemap_provider_reads_env(monkeypatch) -> None:
    from app.config import AppConfig

    monkeypatch.setenv("BASEMAP_PROVIDER", "amap")
    monkeypatch.setenv("AMAP_JS_KEY", "JSKEY")
    cfg = AppConfig()
    assert cfg.basemap_provider == "amap"


def test_health_reports_basemap_provider() -> None:
    """health 里应能看到当前底图服务商，方便验收时确认切换生效。"""
    from app.config import CONFIG
    from app.server import app

    with __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
        app
    ) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "maps" in data
        assert "basemap" in data["maps"]
        assert data["maps"]["basemap"]["provider"] in ("tmap", "amap")


def test_json_snapshot_of_health_shape() -> None:
    """health 的 maps 段结构固定，避免以后改坏前端契约。"""
    from app.server import app

    with __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
        app
    ) as client:
        data = client.get("/api/health").json()
    maps = data["maps"]
    assert set(maps) >= {"road_geometry", "route_validation", "basemap"}
    print(json.dumps(maps["basemap"], ensure_ascii=False))
