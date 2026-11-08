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

// ===================================================================
// 拼接信息块（多图识别）
//
// 为什么单独测：用户传了 5 张图却只拼上 2 张时，
// 页面必须告诉他「哪几张没拼上、重叠是不是不够」，
// 否则用户只会看到结果不对，却不知道该重截哪几张。
// ===================================================================
eval(extract('stitchBlock'));

console.log('\n【场景7】多图拼接成功');
let s = stitchBlock({
  ok: true,
  used: [0, 1, 2],
  unused: [],
  canvas_size: [828, 1024],
  links: [
    { i: 0, j: 1, inliers: 863, ncc: 0.94 },
    { i: 1, j: 2, inliers: 870, ncc: 0.99 },
  ],
});
console.log('  ' + s.replace(/<[^>]+>/g, '').slice(0, 200) + '\n');
check(s.includes('已拼接 3 张'), '显示用了几张图', s.slice(0, 160));
check(s.includes('828×1024'), '显示画布尺寸');
check(s.includes('第1↔第2张'), '显示每对图的重叠情况');
check(!s.includes('重叠偏弱'), '重叠良好时不提示');

console.log('\n【场景8】拼接失败');
s = stitchBlock({ ok: false, used: [], unused: [0, 1, 2], links: [] });
console.log('  ' + s.replace(/<[^>]+>/g, '').slice(0, 160) + '\n');
check(s.includes('未能拼接'), '明确告知拼接失败');

console.log('\n【场景9】有图没拼上 + 重叠偏弱');
s = stitchBlock({
  ok: true,
  used: [0, 1],
  unused: [2],
  canvas_size: [828, 1024],
  links: [{ i: 0, j: 1, inliers: 12, ncc: 0.31 }],
});
console.log('  ' + s.replace(/<[^>]+>/g, '').slice(0, 220) + '\n');
check(s.includes('1 张未用上'), '提示有图没拼上', s.slice(0, 200));
check(s.includes('重叠偏弱'), '提示重叠偏弱、建议多留重叠');
check(s.includes('12 内点'), '显示实际内点数');

// --- 场景 10：像素级复核分（回答「拼得对不对」）---
console.log('\n【场景10】拼接复核分');
s = stitchBlock({
  ok: true,
  used: [0, 1, 2],
  unused: [],
  canvas_size: [828, 1024],
  verify_score: 0.9515,
  verify: [
    { index: 0, similarity: 1.0, pixels: 194560 },
    { index: 1, similarity: 0.9515, pixels: 652288 },
    { index: 2, similarity: 0.9395, pixels: 457728 },
  ],
  shots: [
    { index: 0, box: [0, 0, 700, 1024] },
    { index: 1, box: [190, 0, 890, 1024] },
    { index: 2, box: [380, 0, 1080, 1024] },
  ],
  links: [{ i: 0, j: 1, inliers: 863, ncc: 0.94 }],
});
console.log('  ' + s.replace(/<[^>]+>/g, '').slice(0, 260) + '\n');
check(s.includes('拼接复核'), '显示拼接复核行');
check(s.includes('通过'), '高分判为通过', s.slice(0, 200));
check(s.includes('0.952'), '显示相似度数值');
check(s.includes('原图位置'), '提示预览图上的虚线框含义');

console.log('\n【场景11】复核分偏低 -> 判为可疑');
s = stitchBlock({
  ok: true, used: [0, 1], unused: [], canvas_size: [828, 1024],
  verify_score: 0.62, verify: [], shots: [], links: [],
});
console.log('  ' + s.replace(/<[^>]+>/g, '').slice(0, 200) + '\n');
check(s.includes('可疑'), '低分判为可疑');
check(s.includes('没对齐'), '明确说接缝可能没对齐');

console.log('');
if (failures.length) {
  console.log('失败 ' + failures.length + ' 项：' + JSON.stringify(failures));
  process.exit(1);
}
console.log('全部通过');
