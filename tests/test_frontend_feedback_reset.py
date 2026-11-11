"""「人工纠正清单 + 一键清空」的前端回归测试（Node 跑真实前端函数）。

覆盖三件事：
1. 清单只列人工点位，并给出「本次是否用作内点」「疑似残片」「多个坐标」
   等可判断纠正准不准的信号；
2. 清空按钮必须二次确认，且请求体带 ``confirm:true``；成功后刷新清单；
3. 用户取消不发请求、服务端失败要有可见错误提示。

用法：pytest -q tests/test_frontend_feedback_reset.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "src" / "web" / "index.html"

NODE_SCRIPT = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const code = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];

const START = '  // ---------- 人工纠正清单 + 一键回滚 ----------';
const END = '  // 页面打开就先列一次';
const block = code.slice(code.indexOf(START), code.indexOf(END));
if (!block || block.length < 200) throw new Error('未找到人工纠正清单代码块');

function extract(name) {
  const re = new RegExp('(?:async\\s+)?function\\s+' + name +
    '\\([^)]*\\)\\s*\\{[\\s\\S]*?\\n  \\}');
  const m = code.match(re);
  if (!m) throw new Error('未找到函数 ' + name);
  return m[0];
}
const escSrc = extract('esc');
const fmtSrc = extract('fmtLL');

function makeApi(cfg) {
  cfg = cfg || {};
  const els = {
    correctionRows: { innerHTML: '' },
    correctionCount: { textContent: '0', className: '' },
    clearCorrections: { disabled: false, onclick: null },
  };
  const $ = (id) => els[id] || { style:{}, innerHTML:'', textContent:'', className:'' };
  const calls = [], statuses = [], confirms = [];
  let currentEntries = cfg.entries || {};
  const fetchStub = (url, opts) => {
    calls.push({url, body: opts && opts.body ? JSON.parse(opts.body) : null});
    if (url === '/api/lexicon') {
      return Promise.resolve({ok:true, json: async () => ({entries: currentEntries})});
    }
    if (url === '/api/feedback/reset') {
      currentEntries = cfg.afterEntries || {};
      const resp = cfg.reset || {ok:true};
      return Promise.resolve({ok: resp.ok !== false, json: async () => resp});
    }
    return Promise.reject(new Error('unexpected url ' + url));
  };
  const showStatus = (m, k) => statuses.push({m, k});
  const windowStub = {confirm: (msg) => { confirms.push(msg); return cfg.confirm !== false; }};
  const factory = new Function('$','fetch','showStatus','window','lastResult',
    escSrc + '\n' + fmtSrc + '\n' + block +
    '\nreturn {renderCorrections, refreshCorrections, getOnClick: () => $("clearCorrections").onclick};');
  const api = factory($, fetchStub, showStatus, windowStub, cfg.result || null);
  return {api, els, calls, statuses, confirms};
}

const entries = {
  '交通大学': [
    {lon:121.43, lat:31.20, weight:2, count:1, source:'user', verified:true},
    {lon:121.44, lat:31.21, weight:2, count:1, source:'user', verified:true},
  ],
  '上海世': [{lon:121.40, lat:31.26, weight:2, count:1, source:'user', verified:true}],
  '自动点': [{lon:121.50, lat:31.30, weight:0.5, count:1,
              source:'tianditu', verified:false}],
};
const result = {
  anchors_used: [{name:'交通大学'}],
  anchors_rejected: [{name:'上海世'}],
};

const out = {static:{}, render:{}, cleared:{}, cancelled:{}, failed:{}};
out.static = {
  panel: html.includes('id="correctionPanel"'),
  clearButton: html.includes('id="clearCorrections"'),
  rows: html.includes('id="correctionRows"'),
  explain: html.includes('怎么判断纠正准不准'),
  dangerStyle: html.includes('button.ghost.danger'),
  // anchorCard 默认隐藏；回滚入口必须是独立 card，否则页面刚打开看不到
  standaloneCard: html.includes('<div class="card" id="correctionPanel"'),
  anchorCardHidden: html.includes('id="anchorCard" style="margin-top:16px; display:none;"'),
  resetUrl: code.includes("'/api/feedback/reset'"),
  confirmBody: code.includes('JSON.stringify({ confirm: true })'),
  refreshCalls: (code.match(/refreshCorrections\(\);/g) || []).length,
  refreshInRender: code.includes('refreshCorrections();\n    return mapOk;'),
};

(async () => {
  {
    const c = makeApi({entries, result});
    const n = c.api.renderCorrections(entries, result);
    out.render = {
      count: n,
      badge: c.els.correctionCount.textContent,
      disabled: c.els.clearCorrections.disabled,
      html: c.els.correctionRows.innerHTML,
    };
  }
  {
    const c = makeApi({
      entries, result,
      afterEntries: {'自动点': entries['自动点']},
      reset: {ok:true, removed_points:3, removed_keys:2,
              remaining_points:1, upgraded_auto_points_lost:1},
    });
    c.api.renderCorrections(entries, result);
    await c.api.getOnClick()();
    out.cleared = {
      calls: c.calls, statuses: c.statuses, confirms: c.confirms.length,
      badge: c.els.correctionCount.textContent,
      disabled: c.els.clearCorrections.disabled,
      rows: c.els.correctionRows.innerHTML,
    };
  }
  {
    const c = makeApi({entries, result, confirm:false});
    c.api.renderCorrections(entries, result);
    await c.api.getOnClick()();
    out.cancelled = {calls:c.calls, statuses:c.statuses, confirms:c.confirms.length};
  }
  {
    const c = makeApi({entries, result, reset:{ok:false, detail:'服务不可用'}});
    c.api.renderCorrections(entries, result);
    await c.api.getOnClick()();
    out.failed = {statuses:c.statuses};
  }
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
"""


def test_feedback_reset_ui() -> None:
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("未找到 Node.js，跳过前端交互回归")

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "feedback_reset_ui.js"
        script.write_text(NODE_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            [node, str(script), str(HTML)],
            capture_output=True, text=True, timeout=60,
        )
    assert proc.returncode == 0, (
        f"Node 脚本失败：\nstdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-800:]}"
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    st = out["static"]
    assert all(st[k] for k in ("panel", "clearButton", "rows", "explain",
                               "dangerStyle", "standaloneCard", "anchorCardHidden",
                               "resetUrl", "confirmBody", "refreshInRender")), \
        f"清单/清空入口缺失或仍嵌在隐藏卡片内：{st}"
    assert st["refreshCalls"] >= 3, "识别完成 / 纠正提交 / 页面初始化都要刷新清单"

    r = out["render"]
    assert r["count"] == 3, "只应列出 3 条人工点位，自动点位不能混进来"
    assert r["badge"] == "3" and r["disabled"] is False
    html = r["html"]
    assert "已用作内点" in html and "本次未采纳" in html, "缺少本次识别结论"
    assert "多个坐标（纠错过）" in html, "同一地名多条坐标未标风险"
    assert "疑似 OCR 残片" in html, "短名称未标残片风险"
    assert "自动点" not in html, "自动点位被误列进人工纠正清单"

    c = out["cleared"]
    reset_calls = [x for x in c["calls"] if x["url"] == "/api/feedback/reset"]
    assert len(reset_calls) == 1 and reset_calls[0]["body"] == {"confirm": True}
    assert any(x["url"] == "/api/lexicon" for x in c["calls"]), "成功后未刷新清单"
    assert c["confirms"] == 1, "清空前没有二次确认"
    assert any("已清空 3 条" in s["m"] and "无法还原" in s["m"]
               for s in c["statuses"]), "成功提示未说明损失"
    assert c["badge"] == "0" and c["disabled"] is True, "清空后按钮/计数未复位"
    assert "暂无人工纠正" in c["rows"]

    n = out["cancelled"]
    assert n["confirms"] == 1
    assert not any(x["url"] == "/api/feedback/reset" for x in n["calls"]), \
        "用户取消后仍发送了清空请求"
    assert n["statuses"] == [], "取消不应产生成功/失败提示"

    assert any(s["k"] == "err" and "服务不可用" in s["m"]
               for s in out["failed"]["statuses"]), "清空失败没有可见提示"
