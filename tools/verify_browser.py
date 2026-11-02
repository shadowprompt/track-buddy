"""真实浏览器验证：连续识别两张，是否还抛 Unhandled error。

这是决定性测试——之前所有分析都是推断，只有真浏览器能确认：
1) 腾讯地图 SDK 鉴权是否成功（底图是否真的加载出来）
2) 连续识别第二张时页面是否还报错
3) 原图预览是否完整显示（下半部分不被裁切）
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8848"
SHOT_A = ROOT / "tmp_debug" / "user" / "1793531960_55b77b54.jpg"
SHOT_B = ROOT / "tmp_debug" / "user" / "1793531918_20a0171f.jpg"


def main() -> int:
    errors: list[str] = []
    console: list[str] = []

    with sync_playwright() as pw:
        # 用系统 Edge：沙箱环境下 Playwright 自带的 Chromium
        # 可能因 WebGL/沙箱限制启动失败。
        edge = Path(r"C://Program Files (x86)//Microsoft//Edge//Application//msedge.exe")
        launch_kw = {"channel": "msedge"} if edge.exists() else {}
        browser = pw.chromium.launch(**launch_kw)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))

        print("打开页面…")
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2500)

        # 底图是否真的加载出来了
        has_tmap = page.evaluate("() => typeof TMap !== 'undefined'")
        print(f"  TMap SDK 已加载: {has_tmap}")
        canvases = page.evaluate("() => document.querySelectorAll('#map canvas').length")
        print(f"  地图 canvas 数量: {canvases}")

        # 第一次识别
        print("\n第一次识别…")
        page.set_input_files("#file", str(SHOT_A))
        page.click("#go")
        page.wait_for_function(
            """() => {
                const t = document.getElementById('status').innerText || '';
                return t.includes('识别完成') || t.includes('未能得到坐标')
                       || t.includes('识别失败') || t.includes('已完成');
            }""",
            timeout=240000,
        )
        page.wait_for_timeout(3000)
        st1 = page.inner_text("#status")
        print(f"  状态: {st1.strip()[:120]}")

        # 切到原图叠加页（这是之前必崩的路径）
        print("\n切到「原图叠加」页…")
        page.click(".tab[data-v='preview']")
        page.wait_for_timeout(800)

        # 预览图是否完整（高度应等于图片按宽度缩放后的高度）
        dims = page.evaluate("""() => {
            const img = document.getElementById('prevImg');
            const box = document.getElementById('preview');
            const c = document.getElementById('prevCanvas');
            return {
                imgH: img.clientHeight, imgW: img.clientWidth,
                boxH: box.clientHeight, boxScrollH: box.scrollHeight,
                canvasH: c.clientHeight, canvasW: c.clientWidth,
                natH: img.naturalHeight, natW: img.naturalWidth,
                overflow: getComputedStyle(box).overflow,
            };
        }""")
        print(f"  预览: 图片 {dims['imgW']}x{dims['imgH']}, "
              f"容器 {dims['boxH']}px (scroll {dims['boxScrollH']}px), "
              f"画布 {dims['canvasW']}x{dims['canvasH']}")
        print(f"  overflow: {dims['overflow']}")
        expected_h = 0.0
        if dims["natW"]:
            expected_h = dims["natH"] * dims["imgW"] / dims["natW"]
        not_cropped = (
            expected_h > 0
            and dims["boxH"] >= expected_h - 2
            and dims["canvasH"] >= expected_h - 2
        )
        print(f"  期望高度 {expected_h:.0f}px -> "
              f"{'完整显示 ✓' if not_cropped else '被裁切 ✗'}")

        # 第二次识别（关键场景）
        print("\n第二次识别（停在原图叠加页）…")
        page.set_input_files("#file", str(SHOT_B))
        page.click("#go")
        page.wait_for_function(
            """() => {
                const t = document.getElementById('status').innerText || '';
                return t.includes('识别完成') || t.includes('未能得到坐标')
                       || t.includes('识别失败') || t.includes('已完成');
            }""",
            timeout=240000,
        )
        page.wait_for_timeout(4000)
        st2 = page.inner_text("#status")
        print(f"  状态: {st2.strip()[:160]}")

        has_unhandled = "Unhandled error" in st2
        print(f"\n  出现 'Unhandled error': {has_unhandled}")

        # 是否自动切回地图页
        tab = page.evaluate("""() => {
            const map = document.getElementById('map');
            const on = document.querySelector('.tab.on');
            return { mapDisplay: map.style.display, activeTab: on ? on.dataset.v : '' };
        }""")
        print(f"  自动切回地图页: {tab}")

        # 地图上是否有轨迹图层
        info = page.evaluate("""() => {
            const c = document.querySelectorAll('#map canvas');
            let nonEmpty = 0;
            c.forEach(cv => { if (cv.width > 0 && cv.height > 0) nonEmpty++; });
            return { canvases: c.length, sized: nonEmpty };
        }""")
        print(f"  地图 canvas: {info['canvases']} 个, 已分配尺寸 {info['sized']} 个")

        page.screenshot(path=str(ROOT / "tmp_debug" / "browser_check.png"), full_page=False)
        browser.close()

    print("\n=== 浏览器控制台错误 ===")
    real_errors = [e for e in errors if "favicon" not in e.lower()]
    if real_errors:
        for e in real_errors[:10]:
            print("  ✗", e[:200])
    else:
        print("  无 JS 异常 ✓")

    auth_fail = [c for c in console if "鉴权" in c or "AUTH" in c]
    if auth_fail:
        print("\n=== 鉴权相关日志 ===")
        for c in auth_fail[:8]:
            print("  !", c[:200])
    else:
        print("\n  无鉴权失败日志 ✓")

    ok = not has_unhandled and not real_errors and not_cropped
    print(f"\n结果: {'全部通过 ✓' if ok else '仍有问题 ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
