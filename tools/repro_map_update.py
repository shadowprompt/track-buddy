"""真实浏览器复现：连续识别两张不同截图，抓取地图图层状态与异常。

背景：用户反馈「不报错了，但识别的轨迹图并没有更新」。
要确认：轨迹线是新图的还是旧图的？视野有没有跟着变？
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = [
    ("长宁安龙路", "data/uploads/1793581518_a3432938.jpg"),
    ("浦东锦绣路", "data/uploads/1793581569_148a669c.jpg"),
]
OUT = Path("tmp_debug")


def probe(page, tag):
    st = page.evaluate(
        """() => {
            const R = window.__ride || {};
            const m = R.getMap ? R.getMap() : null;
            const p = R.getPolyline ? R.getPolyline() : null;
            const paths = p && p.getGeometries ? (p.getGeometries()[0].paths || []) : [];
            const el = document.getElementById('map');
            const pv = document.getElementById('preview');
            return {
                activeTab: R.getActiveTab ? R.getActiveTab() : null,
                previewDisplay: getComputedStyle(pv).display,
                mapDisplay: getComputedStyle(el).display,
                mapSize: [el.clientWidth, el.clientHeight],
                center: m ? [Number(m.getCenter().lat.toFixed(5)),
                             Number(m.getCenter().lng.toFixed(5))] : null,
                zoom: m ? m.getZoom() : null,
                trailCount: paths.length,
                trailHead: paths.slice(0, 2).map(x => [Number(x.lat.toFixed(5)),
                                                      Number(x.lng.toFixed(5))]),
                trailTail: paths.slice(-1).map(x => [Number(x.lat.toFixed(5)),
                                                    Number(x.lng.toFixed(5))]),
                markerCount: R.getMarkers ? R.getMarkers().length : null,
                lastPathLen: R.getLastPath ? R.getLastPath().length : null,
                status: (document.getElementById('status').innerText || '').slice(0, 90),
            };
        }"""
    )
    print(f"--- {tag} ---")
    for k, v in st.items():
        print(f"    {k}: {v}")
    return st


def main() -> int:
    errors: list[str] = []
    console: list[str] = []

    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    launch_kw = {"channel": "msedge"} if edge.exists() else {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch_kw)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: console.append(f"{m.type}: {m.text[:300]}"))

        page.goto("http://127.0.0.1:8848", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2500)

        states = []
        for idx, (tag, path) in enumerate(SHOTS):
            print(f"\n===== 第 {idx+1} 次：{tag} =====")
            page.set_input_files("#file", path)
            page.click("#go")
            page.wait_for_function(
                """() => {
                    const t = document.getElementById('status').innerText || '';
                    return t.includes('识别完成') || t.includes('未能得到坐标')
                           || t.includes('识别失败') || t.includes('地图渲染出错');
                }""",
                timeout=240000,
            )
            page.wait_for_timeout(4500)
            states.append(probe(page, tag))
            page.screenshot(path=str(OUT / f"mapupd_{idx+1}.png"), full_page=False)

        print("\n===== 两次识别的轨迹是否真的换了 =====")
        if len(states) == 2:
            a, b = states
            print(f"    轨迹点数: {a['trailCount']} -> {b['trailCount']}")
            print(f"    轨迹首点: {a['trailHead']} -> {b['trailHead']}")
            same = a["trailHead"] == b["trailHead"] and a["trailCount"] == b["trailCount"]
            print(f"    >>> {'轨迹未更新（相同）✗' if same else '轨迹已更新 ✓'}")
            print(f"    地图中心: {a['center']} -> {b['center']}")
            print(f"    zoom: {a['zoom']} -> {b['zoom']}")

        print("\n===== console 错误/警告 =====")
        errs = [c for c in console if c.startswith("error") or c.startswith("warning")]
        for c in errs[-20:]:
            print("  ", c)
        if not errs:
            print("   无")

        print("\n===== JS 异常 =====")
        for e in errors[-10:]:
            print("  X", e[:400])
        if not errors:
            print("   无")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
