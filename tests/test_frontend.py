"""前端回归测试：连续识别不报错 + 原图预览不裁切。

这两个问题都是纯前端的，服务端日志完全正常（全 200），
所以光跑后端测试永远发现不了——必须单独测前端逻辑。

问题 1：连续识别第二张时报「Unhandled error. (undefined)」
    根因：腾讯地图 GL JS 在 0 尺寸容器（display:none）上做 fitBounds
    会抛这个错。用户若停在「原图叠加」标签页直接点第二次「开始识别」，
    识别本身是成功的，但渲染阶段的异常被 catch 捕获，显示成「识别失败」。

问题 2：原图预览下半部分显示不全
    根因：#preview 固定 height:480px + overflow:hidden，
    而截图是 1260x2720 的竖长图，按宽度缩放后必然被裁。

本测试用 Node 跑（无浏览器依赖）：把 index.html 里的相关函数抽出来，
配一个会复现真实腾讯地图行为的桩，逐个场景断言。

用法：python tests/test_frontend.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "web" / "index.html"

PASS = "✓"
FAIL = "✗"
results: list[tuple[bool, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    results.append((bool(cond), name, detail))
    print(f"  {PASS if cond else FAIL} {name}" + (f"  — {detail}" if detail else ""))


# ---- Node 端测试脚本 ----
NODE_SCRIPT = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const m = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const code = m[m.length - 1][1];

function extract(name) {
  const re = new RegExp('function ' + name + '\\([^)]*\\) \\{[\\s\\S]*?\\n  \\}');
  const f = code.match(re);
  if (!f) throw new Error('未找到 ' + name);
  return f[0];
}

let mapElDisplay = 'block';
let mapElW = 800, mapElH = 480;
const tabEls = [
  { dataset: { v: 'map' }, classList: { toggle(){}, add(){}, remove(){} } },
  { dataset: { v: 'preview' }, classList: { toggle(){}, add(){}, remove(){} } },
];
const els = {
  map: {
    get style() { return { set display(v) { mapElDisplay = v; }, get display() { return mapElDisplay; } }; },
    get clientWidth() { return mapElDisplay === 'none' ? 0 : mapElW; },
    get clientHeight() { return mapElDisplay === 'none' ? 0 : mapElH; },
  },
  preview: { style: { display: 'none' } },
  prevImg: { clientWidth: 800, naturalWidth: 1260, naturalHeight: 2720, onload: null, src: '' },
  prevCanvas: { style: {}, width: 0, height: 0, getContext: () => ({
    setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){},
    arc(){}, fill(){}, fillText(){}, set strokeStyle(v){}, set fillStyle(v){},
    set lineWidth(v){}, set lineJoin(v){}, set font(v){},
  }) },
  status: { className: '', innerHTML: '' },
  resultCard: { style: {} }, anchorCard: { style: {} }, roadCard: { style: {} },
  stats: { innerHTML: '' }, diag: { innerHTML: '' },
  roadSummary: { innerHTML: '' }, roadRows: { innerHTML: '' }, roadNote: { innerHTML: '' },
  anchorRows: { innerHTML: '', querySelectorAll: () => [] },
  dl: {}, dlG: { style: {} }, correctName: { textContent: '' },
  correctBox: { style: {} }, mapNote: {},
};
const $ = (id) => els[id] || { style: {}, innerHTML: '', textContent: '' };
const docStub = { querySelectorAll: (sel) => (sel === '.tab' ? tabEls : []) };

const calls = [];
class TMapStub {
  static LatLng = class { constructor(lat, lng) { this.lat = lat; this.lng = lng; } };
  static LatLngBounds = class { constructor() { this.n = 0; } extend() { this.n++; } };
  static PolylineStyle = class { constructor(o) { Object.assign(this, o); } };
  static MarkerStyle = class { constructor(o) { Object.assign(this, o); } };
  static MultiPolyline = class { constructor() { calls.push('polyline'); } };
  static MultiMarker = class { constructor() { calls.push('marker'); } };
  static Map = class {
    constructor() { calls.push('newMap'); }
    removeLayer() { calls.push('removeLayer'); }
    fitBounds() {
      // 复现真实腾讯地图 GL JS：容器 0 尺寸时抛这个错
      calls.push('fitBounds');
      if (mapElW === 0 || mapElH === 0) throw new Error('Unhandled error. (undefined)');
    }
    on() {}
    resize() { calls.push('resize'); }
  };
}

const factory = new Function('TMap', '$', 'document', `
  let map = null, polyline = null, markers = [];
  let lastResult = null, correcting = null, lastPath = [];
  ${extract('esc')}
  ${extract('num')}
  ${extract('int')}
  ${extract('text')}
  ${extract('drawMap')}
  ${extract('drawMapInner')}
  ${extract('fitMapToTrack')}
  ${extract('switchTab')}
  return { drawMap, fitMapToTrack, switchTab,
    setResult(d) { lastResult = d; }, getMap() { return map; } };
`);

const D1 = {
  geojson_gcj02: { features: [{ geometry: { type: 'LineString',
    coordinates: [[121.55, 31.22], [121.56, 31.23]] } }] },
  image_size: [1260, 2720], track_polyline_px: [[100, 200], [300, 400]],
  anchors_used: [], anchors_rejected: [],
};
const out = {};

// 场景1：地图页连续识别两张
{
  const api = factory(TMapStub, $, docStub);
  mapElDisplay = 'block'; mapElW = 800; mapElH = 480;
  calls.length = 0;
  let err = null;
  try { api.drawMap(D1); api.drawMap(D1); } catch (e) { err = e.message; }
  out.s1_error = err;
  out.s1_fitBounds = calls.filter(c => c === 'fitBounds').length;
}

// 场景2：停在原图叠加页时识别（关键场景）
// 注意：fitMapToTrack 内部有 try-catch，会把腾讯地图的异常吞掉，
// 所以这里不能只看「有没有抛出来」，要看：
//   (a) 有没有在 0 尺寸容器上创建/操作地图
//   (b) fitBounds 有没有被调用（bug 版本会真的调进去）
{
  const api = factory(TMapStub, $, docStub);
  mapElDisplay = 'none'; mapElW = 0; mapElH = 0;
  calls.length = 0;
  let err = null;
  try { api.drawMap(D1); } catch (e) { err = e.message; }
  out.s2_error = err;
  out.s2_mapCreated = !!api.getMap();
  out.s2_fitBoundsCalled = calls.includes('fitBounds');
}

// 场景3：切回地图页应补 fitBounds
{
  const api = factory(TMapStub, $, docStub);
  mapElDisplay = 'block'; mapElW = 800; mapElH = 480;
  api.setResult(D1);
  calls.length = 0;
  let err = null;
  try { api.switchTab('map'); } catch (e) { err = e.message; }
  out.s3_error = err;
  out.s3_fitBounds = calls.includes('fitBounds');
}

// 场景4：首次识别时容器就隐藏（地图从未初始化）
{
  const api = factory(TMapStub, $, docStub);
  mapElDisplay = 'none'; mapElW = 0; mapElH = 0;
  calls.length = 0;
  let err = null;
  try {
    api.drawMap(D1);
    mapElDisplay = 'block'; mapElW = 800; mapElH = 480;
    api.setResult(D1);
    api.switchTab('map');
  } catch (e) { err = e.message; }
  out.s4_error = err;
  out.s4_mapCreated = !!api.getMap();
  out.s4_fitBounds = calls.includes('fitBounds');
}

// 场景5：预览容器样式（不裁切）
out.preview_css = (html.match(/#preview\s*\{[^}]*\}/) || [''])[0];
// 旧版是 `#map, #preview { ... height:480px; overflow:hidden }`，
// 新版把 #preview 拆出来且不再设固定高度。
out.has_old_fixed = /#map,\s*#preview\s*\{[^}]*height:480px/.test(html);
out.preview_has_fixed_height = /#preview\s*\{[^}]*height:\s*\d+px/.test(html);

console.log(JSON.stringify(out));
"""




def test_proxy() -> int:
    """腾讯地图本地代理测试（需要服务在 8848 端口运行）。"""
    print("\n[3] 腾讯地图本地代理（回归项）")
    import requests

    s = requests.Session()
    s.trust_env = False
    try:
        html = s.get("http://127.0.0.1:8848/", timeout=10).text
    except Exception as exc:  # noqa: BLE001
        print(f"  服务未运行，跳过：{exc}")
        return 0

    # 占位符必须被替换
    check("__WB_HTTP_PORT__" not in html,
          "页面的端口占位符已替换", "")
    check("__WB_TMAP_SECRET__" not in html,
          "页面的代理标记占位符已替换", "")
    check("_TMapService/_wbt/" in html,
          "serviceHost 指向本地代理")

    # 取出渲染后的 serviceHost
    host = None
    for line in html.split("\n"):
        if "serviceHost" in line and "'" in line:
            host = line.split("'")[1]
            break
    if not host:
        check(False, "能解析出 serviceHost")
        return 0
    check(True, "能解析出 serviceHost", host)

    # checkKey 必须鉴权成功（否则 SDK 会抛 Unhandled error）
    url = host.rstrip("/") + "/checkKey?callback=mkeycb&platform=JSAPI&appId=jsapi_v3"
    try:
        r = s.get(url, timeout=25)
        body = r.text
        ok = r.status_code == 200 and '"error":0' in body
        check(ok, "代理转发 checkKey 且鉴权成功",
              f"HTTP {r.status_code}, {len(body)} 字节")
        check(body.startswith("mkeycb&&mkeycb("),
              "checkKey 响应保持 JSONP 格式（SDK 靠这个识别）",
              body[:28])
    except Exception as exc:  # noqa: BLE001
        check(False, "代理转发 checkKey", str(exc))

    # 不带路径的形式（SDK 的 pingd 会这么请求）
    try:
        r = s.get(host.rstrip("/") + "?logid=pv&v=1.8.2.3", timeout=20,
                  allow_redirects=False)
        check(r.status_code == 200,
              "代理支持不带路径的请求（SDK pingd）", f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        check(False, "代理支持不带路径的请求", str(exc))

    return 0


def main() -> int:
    print("=" * 66)
    print("前端回归测试 · 连续识别与原图预览")
    print("=" * 66)

    if not HTML.exists():
        print(f"前端文件缺失: {HTML}")
        return 1

    node = shutil.which("node")
    if not node:
        print("未找到 node，跳过（前端测试依赖 Node.js）")
        return 0

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "frontend_test.js"
        script.write_text(NODE_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            [node, str(script), str(HTML)],
            capture_output=True, text=True, timeout=60,
        )
    if proc.returncode != 0:
        print("Node 脚本执行失败：")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return 1
    try:
        r = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print("无法解析 Node 输出：", proc.stdout[-500:])
        return 1

    print("\n[1] 连续识别不再抛异常（回归项）")
    check(r["s1_error"] is None, "地图页连续识别两张不报错",
          r["s1_error"] or "无异常")
    check(r["s2_error"] is None,
          "停在「原图叠加」页时识别不报 Unhandled error",
          r["s2_error"] or "无异常")
    check(r["s2_mapCreated"] is False,
          "容器隐藏时不创建地图实例（0 尺寸容器上操作会抛错）")
    check(r["s2_fitBoundsCalled"] is False,
          "容器隐藏时不调用 fitBounds（腾讯地图在此会抛 Unhandled error)")
    check(r["s3_error"] is None, "切回地图页不报错", r["s3_error"] or "无异常")
    check(r["s3_fitBounds"] is True, "切回地图页会自动把视野对准轨迹")
    check(r["s4_error"] is None, "首次识别时容器隐藏也不报错",
          r["s4_error"] or "无异常")
    check(r["s4_mapCreated"] is True, "切回地图页时补建地图实例")
    check(r["s4_fitBounds"] is True, "补建后视野对准轨迹")

    print("\n[2] 原图预览不再被裁切（回归项）")
    check(r["has_old_fixed"] is False,
          "旧的 #map, #preview 固定高度规则已移除")
    check(r["preview_has_fixed_height"] is False,
          "预览容器不再有像素固定高度（否则竖长图会被裁）",
          r["preview_css"].replace("\n", " ")[:90])

    test_proxy()

    print("\n" + "=" * 66)
    ok = sum(1 for x in results if x[0])
    total = len(results)
    print(f"结果: {ok}/{total} 项通过")
    if ok < total:
        print("\n未通过项:")
        for good, name, detail in results:
            if not good:
                print(f"  {FAIL} {name}  {detail}")
    print("=" * 66)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
