"""真实浏览器验证：低置信度时的「换个位置」功能。

检查三件事：
1. 低置信度截图的识别结果里确实出现备选候选卡片
2. 点「用这个」后地图轨迹真的换成了候选位置（首点经纬度变了）
3. 切换过程中无 JS 异常
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOT = "data/uploads/1793589815_20da8357.jpg"
OUT = Path("tmp_debug")


def probe(page):
    return page.evaluate(
        """() => {
            const R = window.__ride || {};
            const p = R.getPolyline ? R.getPolyline() : null;
            const paths = p && p.getGeometries ? (p.getGeometries()[0].paths || []) : [];
            const altCard = document.getElementById('altCard');
            const altList = document.getElementById('altList');
            return {
                altVisible: altCard ? getComputedStyle(altCard).display : 'none',
                altCount: altList ? altList.querySelectorAll('[data-alt]').length : 0,
                trailCount: paths.length,
                trailHead: paths.length
                    ? [Number(paths[0].lat.toFixed(5)), Number(paths[0].lng.toFixed(5))]
                    : null,
                trailTail: paths.length
                    ? [Number(paths[paths.length-1].lat.toFixed(5)),
                       Number(paths[paths.length-1].lng.toFixed(5))]
                    : null,
                status: (document.getElementById('status').innerText || '').slice(0, 120),
            };
        }"""
    )


def main() -> int:
    OUT.mkdir(exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append("console." + m.type + ": " + m.text)
                if m.type == "error" else None)

        page.goto("http://127.0.0.1:8848/", wait_until="networkidle")
        page.wait_for_timeout(2500)

        print("上传截图…")
        page.set_input_files("#file", SHOT)
        page.wait_for_timeout(500)
        page.click("#go")

        # 等识别完成（低置信度截图约 12 秒）
        for _ in range(90):
            page.wait_for_timeout(1000)
            st = probe(page)
            if st["altCount"] > 0:
                break
        st = probe(page)
        print(f"备选卡片 display={st['altVisible']}  候选数={st['altCount']}")
        print(f"当前轨迹点数={st['trailCount']} 起点={st['trailHead']}")
        if st["altCount"] == 0:
            print("✗ 没有出现备选候选")
            browser.close()
            return 1

        page.screenshot(path=str(OUT / "alt_before.png"), full_page=True)

        # 点第一个候选
        before = st["trailHead"]
        page.click("[data-alt='1']")
        page.wait_for_timeout(2500)
        st2 = probe(page)
        print(f"切换后 轨迹点数={st2['trailCount']} 起点={st2['trailHead']}")
        print(f"状态: {st2['status']}")
        page.screenshot(path=str(OUT / "alt_after.png"), full_page=True)

        changed = (before != st2["trailHead"])
        if changed:
            print("✓ 轨迹已切换到候选位置")
        else:
            print("✗ 轨迹没有变化")

        if errors:
            print("JS 异常：")
            for e in errors[:8]:
                print("   ", e)
        else:
            print("✓ 无 JS 异常")

        browser.close()
        return 0 if changed and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
