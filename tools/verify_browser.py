"""真实浏览器端到端验证：连续识别三张不同截图。

检查四件事：
1. 每次识别后地图上的轨迹确实是这一张的（轨迹首点/点数随图变化）
2. 地图视野跟着轨迹走（中心点随之改变）
3. 「原图叠加」标签页正常显示，且与「地图」互斥
4. 无 JS 异常、无 removeLayer 报错
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = [
    ("①长宁安龙路", "data/uploads/1793581518_a3432938.jpg"),
    ("②浦东锦绣路", "data/uploads/1793581569_148a669c.jpg"),
    ("③普陀祁连山南路", "data/uploads/1793581460_58ac0e34.jpg"),
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
                tab: R.getActiveTab ? R.getActiveTab() : null,
                previewDisplay: getComputedStyle(pv).display,
                mapDisplay: getComputedStyle(el).display,
                center: m ? [Number(m.getCenter().lat.toFixed(5)),
                             Number(m.getCenter().lng.toFixed(5))] : null,
                trailCount: paths.length,
                trailHead: paths.length ? [Number(paths[0].lat.toFixed(5)),
                                           Number(paths[0].lng.toFixed(5))] : null,
                status: (document.getElementById('status').innerText || '').slice(0, 80),
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
        page.on("console", lambda m: console.append(f"{m.type}: {m.text[:260]}"))

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
            page.screenshot(path=str(OUT / f"e2e_{idx+1}.png"))

        print("\n===== 每次识别的轨迹是否都换了 =====")
        heads = [s["trailHead"] for s in states]
        counts = [s["trailCount"] for s in states]
        centers = [s["center"] for s in states]
        print(f"    轨迹首点: {heads}")
        print(f"    轨迹点数: {counts}")
        print(f"    地图中心: {centers}")
        unique_heads = len({tuple(h) for h in heads if h}) == len(heads)
        print(f"    >>> {'每次轨迹都不同 ✓' if unique_heads else '有重复轨迹 ✗'}")
        unique_centers = len({tuple(c) for c in centers if c}) == len(centers)
        print(f"    >>> {'每次视野都跟着变 ✓' if unique_centers else '有视野没变 ✗'}")

        print("\n===== 标签页切换 =====")
        page.click(".tab[data-v='preview']")
        page.wait_for_timeout(1200)
        pv = page.evaluate(
            """() => ({
                preview: getComputedStyle(document.getElementById('preview')).display,
                map: getComputedStyle(document.getElementById('map')).display,
                imgH: document.getElementById('prevImg').clientHeight,
            })"""
        )
        print(f"    切到原图叠加: {pv}")
        page.screenshot(path=str(OUT / "e2e_preview.png"))

        page.click(".tab[data-v='map']")
        page.wait_for_timeout(1500)
        mv = page.evaluate(
            """() => ({
                preview: getComputedStyle(document.getElementById('preview')).display,
                map: getComputedStyle(document.getElementById('map')).display,
            })"""
        )
        print(f"    切回地图: {mv}")
        page.screenshot(path=str(OUT / "e2e_map.png"))

        print("\n===== console 错误 =====")
        errs = [c for c in console if c.startswith("error")]
        for c in errs[-12:]:
            print("  ", c)
        if not errs:
            print("   无")

        print("\n===== JS 异常 =====")
        for e in errors[-8:]:
            print("  X", e[:300])
        if not errors:
            print("   无")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
