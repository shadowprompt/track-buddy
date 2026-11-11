"""锚点纠正的交互反馈回归测试（Node 跑真实前端函数）。

用户反馈的两个问题都是纯前端、服务端日志里看不出来的：

1. 点「纠正」后页面上「没有任何提示」——提示放在左栏的 correctBox /
   status 里，而 startCorrect 会把页面滚动到右侧地图，左栏被滚出视野。
2. 在地图上点击后「没有任何交互反应」——旧实现只更新左栏状态文字，
   点击处没有标记、地图没有状态变化，用户无法确认这一下是否生效。

修复方式是地图内的提示浮层（跟随滚动）+ 纠正态边框/十字光标 +
点击处临时 pin + 保存中/成功/失败三态反馈。本测试用 Node 直接执行
index.html 里的函数（配 DOM / fetch / 底图桩），并验证适配层两个
provider 的 showPin/clearPin 都能工作。

用法：pytest -q tests/test_frontend_correction_ui.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "src" / "web" / "index.html"
BASEMAP = ROOT / "src" / "web" / "js" / "basemap.js"

NODE_SCRIPT = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const bmPath = process.argv[3];
const code = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];
const out = {};

out.static = {
  shell: html.includes('id="mapShell"'),
  hint: html.includes('id="mapHint"'),
  hintInsideShell: html.indexOf('id="mapShell"') < html.indexOf('id="mapHint"'),
  crosshair: html.includes('cursor:crosshair'),
  correctingOutline: html.includes('.map-shell.correcting #map'),
  pinOnClick: code.includes('basemap.showPin(lat, lon)'),
  savingState: code.includes('正在保存'),
  successState: code.includes('已记录'),
  failureState: code.includes('纠正提交失败'),
  scrollToShell: code.includes("$('mapShell').scrollIntoView"),
};

function extract(name) {
  const re = new RegExp('(?:async\\s+)?function\\s+' + name +
    '\\([^)]*\\)\\s*\\{[\\s\\S]*?\\n  \\}');
  const m = code.match(re);
  if (!m) throw new Error('未找到函数 ' + name);
  return m[0];
}

let scrolled = false;
const statuses = [];
const els = {
  mapShell: { className: '', scrollIntoView() { scrolled = true; } },
  map: { style: {}, scrollIntoView() {} },
  mapHint: { className: '', innerHTML: '' },
  correctBox: { style: {} },
  correctName: { textContent: '' },
  hintDone: { onclick: null },
  status: { className: '', innerHTML: '' },
  city: { value: '上海' },
  cancelCorrect: { onclick: null },
};
const $ = (id) => els[id] || { style: {}, innerHTML: '', textContent: '', className: '' };

const bm = {
  pinCalls: [], clearCalls: 0,
  showPin(lat, lon) { this.pinCalls.push([lat, lon]); return true; },
  clearPin() { this.clearCalls++; },
  getMap() { return {}; },
};

let fetchBody = null, fetchMode = 'ok', resolveFetch = null;
const fetchStub = (url, opts) => {
  fetchBody = { url: url, body: JSON.parse(opts.body) };
  return new Promise((res) => {
    resolveFetch = () => res({
      ok: fetchMode !== 'fail',
      json: async () => (fetchMode === 'fail'
        ? { detail: '服务不可用' }
        : { ok: true, lexicon: { entries: 23 } }),
    });
  });
};
const showStatus = (m, k) => statuses.push({ m: m, k: k });
const hideStatus = () => statuses.push(null);
const ensureMapReady = () => {};
// 纠正提交成功后会刷新人工纠正清单；本沙箱只测提交链路，用桩记录调用。
const refreshCalls = [];
const refreshCorrections = () => { refreshCalls.push(1); };

const factory = new Function(
  '$', 'basemap', 'fetch', 'showStatus', 'hideStatus', 'ensureMapReady',
  'refreshCorrections', `
  let correcting = null;
  ${extract('esc')}
  ${extract('fmtLL')}
  ${extract('setMapHint')}
  ${extract('hideMapHint')}
  ${extract('setCorrectingUI')}
  ${extract('startCorrect')}
  ${extract('cancelCorrect')}
  ${extract('submitCorrection')}
  $('cancelCorrect').onclick = cancelCorrect;
  return { startCorrect, submitCorrection, cancelCorrect,
           getCorrecting() { return correcting; } };
`);

// ---- 适配层 showPin / clearPin：腾讯与高德两条路径都要能画、能删 ----
global.window = global;
const tmapRemoved = [];
global.TMap = {
  LatLng: class { constructor(lat, lng) { this.lat = lat; this.lng = lng; } },
  MarkerStyle: class { constructor(o) { Object.assign(this, o); } },
  MultiMarker: class { constructor() { this.id = 'pin-1'; } },
  Map: class {
    removeLayer(id) { tmapRemoved.push(id); }
    on() {} resize() {} fitBounds() {}
  },
};
const amapRemoved = [];
global.AMap = {
  Marker: class { constructor(o) { Object.assign(this, o); } },
  Map: class {
    add(o) { (this.added = this.added || []).push(o); }
    remove(o) { amapRemoved.push(o); }
    on() {} resize() {} setFitView() {}
  },
};
const RideBasemap = require(bmPath);

function checkPin(provider) {
  const a = RideBasemap.create(provider, { container: 'map' });
  a.ensureMap({ lat: 31.2, lon: 121.4 });
  const ok = a.showPin(31.2, 121.4);
  const hadPin = !!a.pin;
  a.clearPin();
  return { ok: ok, hadPin: hadPin, cleared: a.pin === null };
}
out.pin_tmap = checkPin('tmap');
out.pin_amap = checkPin('amap');
out.tmap_removed = tmapRemoved.slice();
out.amap_removed_count = amapRemoved.length;

(async () => {
  const api = factory($, bm, fetchStub, showStatus, hideStatus,
                      ensureMapReady, refreshCorrections);

  // 场景 1：进入纠正模式，地图内提示与纠正态必须可见
  api.startCorrect('交通大学');
  out.s1 = {
    shell: els.mapShell.className,
    cursor: els.map.style.cursor,
    hintClass: els.mapHint.className,
    hint: els.mapHint.innerHTML,
    box: els.correctBox.style.display,
    correctName: els.correctName.textContent,
    scrolled: scrolled,
    correcting: api.getCorrecting(),
    status: statuses[statuses.length - 1],
  };

  // 场景 2：点击地图后立刻有 pin + 「正在保存」，请求返回后切成成功态
  fetchMode = 'ok';
  const pending = api.submitCorrection(121.4343458, 31.2018605);
  out.s2_pending = {
    hint: els.mapHint.innerHTML,
    pins: bm.pinCalls.slice(),
    body: fetchBody,
    shell: els.mapShell.className,
    correcting: api.getCorrecting(),
  };
  resolveFetch();
  await pending;
  out.s2_done = {
    hintClass: els.mapHint.className,
    hint: els.mapHint.innerHTML,
    hasDoneButton: typeof els.hintDone.onclick === 'function',
    correcting: api.getCorrecting(),
  };
  els.hintDone.onclick();
  out.s2_close = {
    hintClass: els.mapHint.className,
    clearCalls: bm.clearCalls,
  };

  // 场景 3：提交失败要给出错误提示，并允许直接在地图上再点一次
  api.startCorrect('上海西站');
  fetchMode = 'fail';
  const failing = api.submitCorrection(121.4, 31.26);
  resolveFetch();
  await failing;
  out.s3 = {
    hintClass: els.mapHint.className,
    hint: els.mapHint.innerHTML,
    shell: els.mapShell.className,
    correcting: api.getCorrecting(),
  };

  // 场景 4：取消要清掉标记、提示与纠正态
  api.cancelCorrect();
  out.s4 = {
    shell: els.mapShell.className,
    cursor: els.map.style.cursor,
    hintClass: els.mapHint.className,
    box: els.correctBox.style.display,
    correcting: api.getCorrecting(),
    clearCalls: bm.clearCalls,
  };

  out.refresh_calls = refreshCalls.length;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_correction_ui_has_visible_feedback() -> None:
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("未找到 Node.js，跳过前端交互回归")

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "correction_ui_test.js"
        script.write_text(NODE_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            [node, str(script), str(HTML), str(BASEMAP)],
            capture_output=True, text=True, timeout=60,
        )
    assert proc.returncode == 0, (
        f"Node 脚本失败：\nstdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-800:]}"
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    st = out["static"]
    assert st["shell"] and st["hint"] and st["hintInsideShell"], "地图内提示浮层缺失"
    assert st["crosshair"] and st["correctingOutline"], "纠正态样式缺失"
    assert st["pinOnClick"], "点击后未落 pin"
    assert st["savingState"] and st["successState"] and st["failureState"], "三态反馈缺失"
    assert st["scrollToShell"], "仍滚动到旧目标（提示会被滚出视野）"

    assert out["pin_tmap"] == {"ok": True, "hadPin": True, "cleared": True}
    assert out["pin_amap"] == {"ok": True, "hadPin": True, "cleared": True}
    assert out["tmap_removed"] == ["pin-1"], "腾讯图层未按 id 移除"
    assert out["amap_removed_count"] == 1, "高德标记未移除"

    s1 = out["s1"]
    assert s1["shell"] == "map-shell correcting"
    assert s1["cursor"] == "crosshair"
    assert "show" in s1["hintClass"]
    assert "纠正模式" in s1["hint"] and "交通大学" in s1["hint"]
    assert s1["box"] == "block" and s1["correctName"] == "交通大学"
    assert s1["scrolled"] and s1["correcting"] == "交通大学"
    assert s1["status"] and "请在地图上点击" in s1["status"]["m"]

    p = out["s2_pending"]
    assert "正在保存" in p["hint"], "点击后没有即时反馈"
    assert p["pins"] == [[31.2018605, 121.4343458]], "pin 经纬度顺序错"
    assert p["body"]["url"] == "/api/feedback"
    assert p["body"]["body"] == {
        "name": "交通大学", "lon": 121.4343458, "lat": 31.2018605,
        "crs": "GCJ-02", "city": "上海",
    }
    assert p["shell"] == "map-shell" and p["correcting"] is None

    done = out["s2_done"]
    assert "ok" in done["hintClass"] and "已记录" in done["hint"]
    assert out["refresh_calls"] >= 1, "纠正成功后未刷新人工纠正清单"
    assert "121.434346" in done["hint"] and "31.2018" in done["hint"], \
        "成功提示未显示点击坐标"
    assert done["hasDoneButton"] and done["correcting"] is None
    assert out["s2_close"]["hintClass"] == "map-hint"
    assert out["s2_close"]["clearCalls"] >= 2, "关闭时未清除标记"

    s3 = out["s3"]
    assert "err" in s3["hintClass"] and "纠正提交失败" in s3["hint"]
    assert s3["shell"] == "map-shell correcting", "失败后未恢复纠正模式"
    assert s3["correcting"] == "上海西站", "失败后无法直接重试"

    s4 = out["s4"]
    assert s4["shell"] == "map-shell" and s4["cursor"] == ""
    assert s4["hintClass"] == "map-hint" and s4["box"] == "none"
    assert s4["correcting"] is None
