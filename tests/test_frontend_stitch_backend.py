"""前端「拼接方式」选择器与结果展示的回归测试。

静态契约锁住接线；再用 Node 沙箱跑真实的 `stitchBlock()` 与
`refreshStitchBackends()`，验证两件最容易骗人的事：

1. 纯大模型模式下 links 显示的是**模型置信度**，不是「0 内点 / 相似度 0」，
   也不会因为 inliers=0 被误标成「重叠偏弱」；
2. 没配大模型服务时对应选项必须 disabled，不能让用户选了以后
   其实跑的仍是 SIFT。
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

function extract(name) {
  const re = new RegExp('(?:async\\s+)?function\\s+' + name +
    '\\([^)]*\\)\\s*\\{[\\s\\S]*?\\n  \\}');
  const m = code.match(re);
  if (!m) throw new Error('未找到函数 ' + name);
  return m[0];
}
const escSrc = extract('esc');
const escFn = new Function(escSrc + '\nreturn esc;')();
const stitchBlock = new Function(
  escSrc + '\n' + extract('stitchBlock') + '\nreturn stitchBlock;')();

const out = {};
const base = {
  ok: true, used: [0, 1], unused: [], canvas_size: [600, 240],
  verify_score: 1.0, shots: [{index: 0}, {index: 1}],
  links: [{i: 0, j: 1, inliers: 314, ncc: 0.5, source: 'sift', weight: null}],
};
out.sift = stitchBlock(Object.assign({}, base, {backend: 'sift'}));
out.llm = stitchBlock(Object.assign({}, base, {
  backend: 'llm',
  backend_diagnostics: {model: 'stub-vlm', elapsed_ms: 12,
                        pairs_accepted: 1, pairs_rejected: 0},
  links: [{i: 0, j: 1, inliers: 0, ncc: 0, source: 'llm', weight: 0.8}],
}));
out.failed = stitchBlock(Object.assign({}, base, {
  backend: 'llm', ok: false, backend_error: true,
  backend_diagnostics: {
    model: 'gpt-6.1-sol',
    error: 'HTTP 400: Field max_tokens must be at most 65536',
  },
}));

const specs = [
  {id: 'sift', label: 'SIFT 特征配准', description: 'd1',
   available: true, reason: ''},
  {id: 'llm', label: '大模型语义配准', description: 'd2',
   available: false, reason: '未配置 LLM_BASE_URL'},
  {id: 'llm_guided', label: '大模型候选 + SIFT', description: 'd3',
   available: false, reason: '未配置 LLM_BASE_URL'},
];
const sel = {
  value: 'sift', innerHTML: '', options: [],
  appendChild(o) { this.options.push(o); },
};
const note = {innerHTML: ''};
const els = {stitchBackend: sel, stitchBackendNote: note};
const $ = (id) => els[id] || null;
const doc = {createElement: () => ({value: '', textContent: '',
                                    disabled: false, title: ''})};
let fetched = '';
const fetchStub = async (url) => {
  fetched = url;
  return {ok: true, json: async () => ({default: 'sift', backends: specs})};
};
const picker = new Function(
  '$', 'fetch', 'document', 'esc',
  extract('refreshStitchBackends') + '\nreturn refreshStitchBackends;'
)($, fetchStub, doc, escFn);

(async () => {
  await picker();
  out.picker = {
    url: fetched,
    ids: sel.options.map(o => o.value),
    disabled: sel.options.map(o => o.disabled),
    labels: sel.options.map(o => o.textContent),
    value: sel.value,
    note: note.innerHTML,
  };
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
"""


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_static_contract() -> None:
    h = _html()
    assert 'id="stitchBackend"' in h
    assert 'id="stitchBackendNote"' in h
    assert "refreshStitchBackends();" in h
    assert "fetch('/api/stitch-backends')" in h
    assert "o.disabled = !b.available" in h
    assert "fd.append('stitch_backend'" in h
    assert "if (paired || detailsOnly) {" in h
    assert "st.backend_diagnostics" in h
    assert "l.source === 'llm'" in h
    assert "st.backend_error" in h
    assert "bdg.error" in h          # 失败原因要留在结果卡片里


def test_rendered_output_and_picker() -> None:
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("未找到 Node.js，跳过前端交互回归")

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "stitch_backend_ui.js"
        script.write_text(NODE_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            [node, str(script), str(HTML)],
            capture_output=True, text=True, timeout=60,
        )
    assert proc.returncode == 0, (
        f"Node 脚本失败：\nstdout={proc.stdout[-1200:]}\n"
        f"stderr={proc.stderr[-800:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert "SIFT 特征配准" in out["sift"]
    assert "314 内点 / 相似度 0.5" in out["sift"]

    llm = out["llm"]
    assert "大模型语义配准" in llm
    assert "模型诊断" in llm and "stub-vlm" in llm
    assert "模型置信度 0.80" in llm
    assert "无几何内点" in llm
    # 内点 0 是「没做几何配准」，不是「重叠质量差」
    assert "0 内点 / 相似度 0" not in llm
    assert "重叠偏弱" not in llm
    assert "本次失败" in out["failed"]
    assert "模型诊断" in out["failed"]
    assert "max_tokens must be at most 65536" in out["failed"]

    p = out["picker"]
    assert p["url"] == "/api/stitch-backends"
    assert p["ids"] == ["sift", "llm", "llm_guided"]
    assert p["disabled"] == [False, True, True]
    assert p["value"] == "sift"
    assert "（未启用）" in p["labels"][1]
    assert "大模型后端未启用" in p["note"]
    assert "不会偷偷改用 SIFT" in p["note"]
