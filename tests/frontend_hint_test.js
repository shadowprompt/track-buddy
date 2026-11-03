// 验证前端「缩放级别过低 / 低置信度」提示的渲染逻辑。
//
// 用 Node 直接跑页面里的真实函数（从 web/index.html 抽取），
// 不依赖浏览器——比截图验证更精准，且能覆盖四种分支。
//
// 背景：这几句提示是用户唯一的行动指引，判错方向就有代价：
//   * 该提示却没提示 -> 用户拿到错结果却以为准的
//   * 不该提示却提示 -> 让用户白折腾（去截一张更大的图）
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

const els = { roadNames: { value: '' } };
const $ = (id) => els[id];

eval(extract('esc'));
eval(extract('lowConfidenceHint'));

const failures = [];
function check(cond, label, detail) {
  console.log((cond ? '  ✓ ' : '  ✗ ') + label + (cond ? '' : '  — ' + (detail || '')));
  if (!cond) failures.push(label);
}

// --- 场景 1：城市级截图 + 没填路名 ---
els.roadNames.value = '';
let out = lowConfidenceHint({
  confidence: 0.2,
  diagnostics: {
    zoom: {
      zoom_too_low: true, n_anchor_names: 19,
      n_street_anchors: 0, n_broad_anchors: 11,
      broad_examples: ['静安区', '徐汇区', '机场联络线'],
    },
  },
});
console.log('\n【场景1】缩放过低 + 未填路名\n  ' + out.replace(/<[^>]+>/g, '').slice(0, 220) + '\n');
check(out.includes('缩放级别过低'), '提示「地图缩放级别过低」', out.slice(0, 120));
check(out.includes('填'), '给出「填路名」建议');
check(out.includes('街道级'), '给出「放大到街道级」建议');
check(out.includes('静安区'), '列出具体的大范围地标示例');
check(out.includes('20%'), '显示置信度百分比');

// --- 场景 2：城市级截图 + 已填路名（位置已修正，但形状仍不可恢复）---
els.roadNames.value = '小木桥路,龙华中路';
out = lowConfidenceHint({
  confidence: 0.776,
  diagnostics: { zoom: { zoom_too_low: true, position_ok: true,
                          broad_examples: ['静安区'] } },
});
console.log('\n【场景2】缩放过低 + 已填路名\n  ' + out.replace(/<[^>]+>/g, '').slice(0, 220) + '\n');
check(out.includes('缩放级别太低'), '仍提示缩放级别过低');
check(out.includes('位置已修正'), '明确说明位置已修正');
check(out.includes('走向'), '说明街道级走向无法恢复');
check(out.includes('放大'), '给出「放大到街道级重截」建议');
check(!out.includes('填写你记得走过的路名'),
      '已填路名时不再重复要填路名', out.slice(0, 200));

// --- 场景 3：普通低置信度 + 未填路名 ---
els.roadNames.value = '';
out = lowConfidenceHint({ confidence: 0.3, diagnostics: {} });
console.log('\n【场景3】普通低置信度 + 未填路名\n  ' + out.replace(/<[^>]+>/g, '').slice(0, 200) + '\n');
check(out.includes('填写你记得走过的路名'), '提示填路名');
check(!out.includes('缩放级别过低'), '不误报缩放级别');

// --- 场景 4：普通低置信度 + 已填路名 ---
els.roadNames.value = '小木桥路';
out = lowConfidenceHint({ confidence: 0.3, diagnostics: {} });
console.log('\n【场景4】普通低置信度 + 已填路名\n  ' + out.replace(/<[^>]+>/g, '').slice(0, 200) + '\n');
check(out.includes('路名是否与实际路线一致'), '提示检查路名是否正确');

// --- 场景 5：高置信度 -> 不显示提示 ---
out = lowConfidenceHint({ confidence: 0.78, diagnostics: {} });
check(out === '', '高置信度时不显示任何提示', JSON.stringify(out.slice(0, 80)));

// --- 场景 6：高置信度但缩放过低 -> 仍要提示（形状问题与置信度无关）---
out = lowConfidenceHint({
  confidence: 0.78,
  diagnostics: { zoom: { zoom_too_low: true, position_ok: false,
                          broad_examples: ['静安区'] } },
});
check(out.includes('缩放级别'), '缩放过低时即使置信度高也要提示');
check(out.includes('位置'), '区分「位置」与「走向」两个维度');

console.log('');
if (failures.length) {
  console.log('失败 ' + failures.length + ' 项：' + JSON.stringify(failures));
  process.exit(1);
}
console.log('全部通过');
