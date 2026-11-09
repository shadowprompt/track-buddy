"""真实浏览器验证底图适配层（Playwright）。

为什么需要它：阶段 3 的 Node 测试都跑在**桩**上，只能证明逻辑正确；
「腾讯 GL JS + 适配层」这条真实链路是否通，只有浏览器能回答。
历史上「Unhandled error. (undefined)」这类问题就是这么漏过去的。

前置：服务已在 127.0.0.1:8848 运行。

⚠️ **可运行性探测**：某些沙箱环境不允许浏览器进程存活——启动会
挂起或被直接终止，而不是抛异常。本脚本先用一个**独立子进程**做
5 秒探测，探不通就打印「跳过」并以 0 退出（环境限制，不是代码问题）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8848"
SHOT = ROOT / "samples" / "meituan_ride_shanghai.jpg"

failures: list[str] = []

_PROBE = """
import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True,
                           args=["--no-sandbox", "--disable-gpu"])
    b.close()
print("PROBE_OK")
"""


def browser_available() -> bool:
    """探测浏览器能否在 20 秒内启动并退出。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True, text=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001
        return False
    return "PROBE_OK" in (proc.stdout or "")


def check(cond: bool, label: str, detail: str = "") -> None:
    print(("  ✓ " if cond else "  ✗ ") + label + ("" if cond else "  — " + detail))
    if not cond:
        failures.append(label)


def main() -> int:
    if not SHOT.exists():
        print(f"样本缺失: {SHOT}")
        return 1

    if not browser_available():
        print("浏览器不可用（沙箱限制），跳过真实浏览器验证")
        return 0

    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" else None)

        page.goto(BASE, wait_until="networkidle", timeout=60000)

        # ---- 1. 适配层就绪 ----
        print("\n【适配层加载】")
        info = page.evaluate(
            """() => ({
                hasRideBasemap: typeof window.RideBasemap !== 'undefined',
                providers: (typeof window.RideBasemap !== 'undefined')
                    ? window.RideBasemap.listProviders() : [],
                injected: window.__WB_MAP_PROVIDER__ || null,
                hasTMap: typeof window.TMap !== 'undefined',
                ride: typeof window.__ride !== 'undefined',
            })"""
        )
        check(info["hasRideBasemap"], "适配层脚本已加载 (window.RideBasemap)")
        check("tmap" in (info["providers"] or []),
              "tmap provider 已注册", str(info["providers"]))
        check(info["injected"] == "tmap",
              "服务端注入的 provider 正确", str(info["injected"]))
        check(info["hasTMap"], "腾讯地图 SDK 已加载")
        check(info["ride"], "调试钩子 window.__ride 存在")

        # ---- 2. 上传并识别，验证地图真的画出来 ----
        print("\n【识别后地图渲染】")
        page.set_input_files("#file", str(SHOT))
        page.click("#go")
        try:
            page.wait_for_function(
                "() => { const s = document.getElementById('status');"
                " return s && s.innerText && s.innerText.length > 10; }",
                timeout=240000,
            )
        except Exception as exc:  # noqa: BLE001
            check(False, "识别在超时前返回结果", str(exc)[:200])

        page.wait_for_timeout(3000)
        state = page.evaluate(
            """() => {
                const R = window.__ride || {};
                const m = R.getMap ? R.getMap() : null;
                const p = R.getPolyline ? R.getPolyline() : null;
                let paths = [];
                if (p && p.getGeometries) {
                    const g = p.getGeometries();
                    paths = (g && g[0] && g[0].paths) || [];
                }
                const markers = R.getMarkers ? R.getMarkers() : [];
                const el = document.getElementById('map');
                return {
                    hasMap: !!m,
                    trailPoints: paths.length,
                    markerCount: markers.length,
                    mapWidth: el ? el.clientWidth : 0,
                    mapHeight: el ? el.clientHeight : 0,
                    basemapName: R.getBasemap ? R.getBasemap().providerName : null,
                    lastPathLen: R.getLastPath ? R.getLastPath().length : 0,
                };
            }"""
        )
        check(state["hasMap"], "适配层创建了地图实例")
        check(state["basemapName"] == "tmap",
              "运行时 provider 为 tmap", str(state["basemapName"]))
        check(state["trailPoints"] > 0,
              "地图上有轨迹几何", f"points={state['trailPoints']}")
        check(state["markerCount"] == 2,
              "起点/终点标记都画上了", f"markers={state['markerCount']}")
        check(state["mapWidth"] > 0 and state["mapHeight"] > 0,
              "地图容器有尺寸", f"{state['mapWidth']}x{state['mapHeight']}")

        # ---- 3. 无 JS 异常 ----
        print("\n【JS 异常】")
        fatal = [e for e in errors
                 if "Unhandled error" in e or "TypeError" in e
                 or "ReferenceError" in e]
        check(not fatal, "没有致命 JS 异常", " | ".join(fatal[:3])[:300])

        browser.close()

    print("\n" + (f"{len(failures)} 项失败: " + "、".join(failures)
                  if failures else "全部通过"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
