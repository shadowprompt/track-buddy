// 小程序 JS 逻辑桩件测试：假 wx + 假后端，直接 require 真实模块运行。
//
// 覆盖：登录换 token / 云未启用零回归 / recognize 带 Bearer / 401 续期重试 /
//       云列表与本机兜底合并 / 统计 / 历史记录详情 / 删除 / 图片代理下载。
// 不联网、不依赖微信开发者工具、不改任何业务文件。
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const RIDEWX = path.join(ROOT, 'wxmin');
const MODULES = [
  'app.js', 'config.js', 'utils/api.js', 'utils/multipart.js',
  'utils/records.js', 'pages/index/index.js', 'pages/records/records.js',
  'pages/me/me.js', 'pages/result/result.js'
];

var failures = [];
function check(cond, label, detail) {
  console.log((cond ? '  ✓ ' : '  ✗ ') + label + (cond ? '' : '  — ' + (detail || '')));
  if (!cond) failures.push(label);
}

function samplePayload(shotUrl) {
  var coords = [[116.3, 39.9], [116.31, 39.91], [116.32, 39.92]];
  var fc = function () {
    return { type: 'FeatureCollection', features: [
      { type: 'Feature', geometry: { type: 'LineString', coordinates: coords }, properties: {} }
    ] };
  };
  return {
    shot_id: 'S1',
    track_points: 120,
    confidence: 0.8,
    geojson: fc(),
    geojson_gcj02: fc(),
    gpx: '<gpx/>',
    tcx: '<tcx/>',
    ride_summary: {
      available: true, started_at: '2026-11-11T20:20:26',
      duration_s: 1235, distance_km: 5.2, calories_kcal: 180
    },
    diagnostics: { template_derived: false },
    warnings: [],
    shot_url: shotUrl || '/api/shots/S1.jpg'
  };
}

function sampleCard(id, createdAt) {
  return {
    id: id, created_at: createdAt || 1763100000, city: '上海',
    distance_m: 5200, duration_s: 1235, calories_kcal: 180,
    started_at: '2026-11-11T20:20:26', distance_km: 5.2,
    confidence: 0.8, track_points: 120, has_track: true,
    exports: ['geojson', 'gpx', 'tcx'], thumb: 'overview.jpg'
  };
}

var state, storage, requests, downloads, navigations, capturedApp, capturedPages;

function makeWx() {
  return {
    env: { USER_DATA_PATH: '/tmp' },
    getStorageSync: function (k) { return storage[k]; },
    setStorageSync: function (k, v) { storage[k] = v; },
    removeStorageSync: function (k) { delete storage[k]; },
    getFileSystemManager: function () {
      return {
        readFile: function (o) { o.success({ data: new ArrayBuffer(4) }); },
        writeFile: function (o) { if (o.success) o.success({}); }
      };
    },
    login: function (o) {
      state.loginCalls++;
      o.success({ code: 'CODE_A' });
    },
    request: function (o) { route(o); },
    downloadFile: function (o) {
      downloads.push({ url: o.url, header: o.header || {} });
      o.success({ statusCode: 200, tempFilePath: '/tmp/record-image.png' });
    },
    showToast: function () {},
    showLoading: function () {},
    hideLoading: function () {},
    showModal: function (o) {
      if (o.success) o.success({ confirm: state.confirmDelete !== false, cancel: false });
    },
    navigateTo: function (o) { navigations.push(o.url); },
    navigateBack: function () {},
    switchTab: function () {},
    stopPullDownRefresh: function () {},
    chooseMedia: function () {},
    chooseLocation: function () {},
    shareFileMessage: function () {}
  };
}

function reply(o, code, data) { o.success({ statusCode: code, data: data }); }

function route(o) {
  var url = String(o.url || '');
  var method = String(o.method || 'GET').toUpperCase();
  var header = o.header || {};
  requests.push({ url: url, method: method, header: header, data: o.data });

  if (url.indexOf('/api/health') >= 0) {
    return reply(o, 200, {
      records: { enabled: state.healthEnabled, sdk_available: true },
      source_profile: { available: [] }
    });
  }
  if (url.indexOf('/api/auth/wxlogin') >= 0) {
    state.wxloginCalls++;
    if (state.wxloginStatus && state.wxloginStatus !== 200) {
      return reply(o, state.wxloginStatus, { detail: '微信登录服务暂不可用' });
    }
    return reply(o, 200, { token: state.token, expires_in: 3600 });
  }
  if (url.indexOf('/api/recognize') >= 0) {
    state.recognizeAuth.push(header.Authorization || '');
    return reply(o, 200, samplePayload());
  }
  if (url.indexOf('/api/records') >= 0) {
    if (method === 'DELETE') {
      state.deleted.push(url);
      return reply(o, 200, { ok: true });
    }
    var isDetail = url.indexOf('/images/') < 0 && /\/api\/records\/[^/?]+$/.test(url);
    if (isDetail) return reply(o, 200, state.detail);
    var st = state.listStatuses.length ? state.listStatuses.shift() : 200;
    if (st !== 200) return reply(o, st, { detail: '缺少有效 Bearer token' });
    state.listAuth.push(header.Authorization || '');
    return reply(o, 200, {
      items: state.items || [],
      next_cursor: state.nextCursor || null,
      total: (state.items || []).length
    });
  }
  if (url.indexOf('/api/refine/') >= 0) {
    state.refineCalls++;
    return reply(o, 200, { state: 'unknown' });
  }
  if (url.indexOf('/api/feedback') >= 0) return reply(o, 200, { ok: true });
  return reply(o, 404, { detail: 'not found' });
}

/** 重建一套干净的模块实例（清 require 缓存）+ 假运行环境。 */
function boot(overrides) {
  state = Object.assign({
    healthEnabled: true, token: 'TOKEN_A', wxloginStatus: 200,
    items: [], nextCursor: null, listStatuses: [], detail: null,
    confirmDelete: true,
    loginCalls: 0, wxloginCalls: 0, refineCalls: 0,
    recognizeAuth: [], listAuth: [], deleted: []
  }, overrides || {});
  storage = {};
  requests = [];
  downloads = [];
  navigations = [];
  capturedPages = [];
  capturedApp = null;

  global.wx = makeWx();
  global.App = function (opts) {
    var app = Object.assign({}, opts);
    app.globalData = Object.assign({}, opts.globalData || {});
    capturedApp = app;
    app.onLaunch.call(app);
  };
  global.Page = function (opts) {
    var p = Object.assign({}, opts);
    p.data = Object.assign({}, opts.data || {});
    p.setData = function (patch) { Object.assign(p.data, patch); };
    capturedPages.push(p);
  };
  global.getApp = function () { return capturedApp; };

  MODULES.forEach(function (f) {
    var full = path.join(RIDEWX, f);
    delete require.cache[require.resolve(full)];
  });
  var api = require(path.join(RIDEWX, 'utils/api.js'));
  require(path.join(RIDEWX, 'app.js')); // 触发 onLaunch
  var recordsUtil = require(path.join(RIDEWX, 'utils/records.js'));
  return { app: capturedApp, api: api, recordsUtil: recordsUtil };
}

function loadPage(rel) {
  var full = path.join(RIDEWX, rel);
  delete require.cache[require.resolve(full)];
  require(full);
  return capturedPages[capturedPages.length - 1];
}

async function flush(times) {
  for (var i = 0; i < (times || 10); i++) {
    await new Promise(function (r) { setImmediate(r); });
  }
}

function countRecordsGet() {
  return requests.filter(function (r) {
    return r.method === 'GET' && r.url.indexOf('/api/records?') >= 0;
  }).length;
}

async function main() {
  // --- 1. 云端未启用：不 wx.login、识别不带 token（零回归） -----------
  console.log('\n【1】records.enabled=false：零回归');
  var b = boot({ healthEnabled: false });
  var t = await b.app.ensureToken();
  check(t === '', 'ensureToken 返回空 token', String(t));
  check(state.loginCalls === 0 && state.wxloginCalls === 0,
    '未调用 wx.login / wxlogin', 'login=' + state.loginCalls + ' wxlogin=' + state.wxloginCalls);
  await b.api.recognize({ overview: 'a.jpg' });
  check(state.recognizeAuth[0] === '', '识别请求不带 Authorization',
    String(state.recognizeAuth[0]));
  check(b.app.globalData.recordsReady === false, 'recordsReady=false');

  // --- 2. 云端启用：登录换 token + 识别带 Bearer ----------------------
  console.log('\n【2】records.enabled=true：登录与自动保存接线');
  b = boot({ healthEnabled: true });
  t = await b.app.ensureToken();
  check(t === 'TOKEN_A', '拿到 token', String(t));
  check(state.loginCalls === 1 && state.wxloginCalls === 1,
    'wx.login → /api/auth/wxlogin 各一次');
  var wxloginReq = requests.filter(function (r) {
    return r.url.indexOf('/api/auth/wxlogin') >= 0;
  })[0];
  check(wxloginReq && wxloginReq.data && wxloginReq.data.code === 'CODE_A',
    'wxlogin 只传 code（不含任何密钥/openid）');
  check(!!storage.ride_records_token_v1, 'token 已写入 Storage 缓存');
  await b.api.recognize({ overview: 'a.jpg' });
  check(state.recognizeAuth[0] === 'Bearer TOKEN_A',
    '识别请求带 Bearer（服务端据此自动存 COS）', String(state.recognizeAuth[0]));

  // --- 3. 有有效缓存：冷启动不再登录 ----------------------------------
  console.log('\n【3】token 缓存');
  // 首启时关掉记录功能，避免产生 wxlogin 计数；随后预置缓存并重建 app 实例
  b = boot({ healthEnabled: false });
  storage.ride_records_token_v1 = {
    token: 'TOKEN_SAVED', expires_at: Date.now() + 3600000
  };
  delete require.cache[require.resolve(path.join(RIDEWX, 'app.js'))];
  require(path.join(RIDEWX, 'app.js'));
  var cached = await capturedApp.ensureToken();
  check(cached === 'TOKEN_SAVED', '冷启动复用缓存 token', String(cached));
  check(state.wxloginCalls === 0, '有缓存时不重复 wxlogin');

  // --- 4. 列表请求带 Bearer -------------------------------------------
  console.log('\n【4】记录列表');
  b = boot({ healthEnabled: true, items: [sampleCard('S1'), sampleCard('S2', 1763099000)] });
  await b.app.ensureToken();
  var d = await b.api.listRecords({ limit: 100 });
  check(d.items.length === 2 && d.total === 2, '返回 items 与 total', JSON.stringify(d));
  check(state.listAuth[0] === 'Bearer TOKEN_A', '列表请求带 Bearer');

  // --- 5. 401 → 强制续期 + 重试一次 ------------------------------------
  console.log('\n【5】401 续期重试');
  b = boot({ healthEnabled: true, listStatuses: [401, 200], items: [sampleCard('S1')] });
  await b.app.ensureToken();
  d = await b.api.listRecords({ limit: 20 });
  check(d.items.length === 1, '重试后拿到列表');
  check(countRecordsGet() === 2, 'GET /api/records 共两次', String(countRecordsGet()));
  check(state.wxloginCalls === 2, '401 后重新登录一次', String(state.wxloginCalls));
  var gets = requests.filter(function (r) { return r.method === 'GET' && r.url.indexOf('/api/records?') >= 0; });
  check(gets[1].header.Authorization === 'Bearer TOKEN_A', '重试请求带新 Bearer');

  // --- 6. 视图模型：云端卡片 / 本机兜底 / 去重 / 统计 -------------------
  console.log('\n【6】视图模型与本地兜底');
  b = boot({ healthEnabled: true });
  var util = b.recordsUtil;
  var view = util.cardToView(sampleCard('S1'));
  check(view.id === 'S1' && view.hasTrack === true, '云端卡片 id / 轨迹标记');
  check(view.distanceText === '5.20 km', '里程格式化', view.distanceText);
  check(view.pointsText === '120' && view.exportText === '3 种', '轨迹点 / 导出格式');
  check(view.confidenceText === '匹配度 80%', '置信度文案', view.confidenceText);
  check(view.rideSummary.indexOf('20:20') >= 0, '运动概要含开始时间', view.rideSummary);
  check(view.isLocal === false, '云端卡片不是本机记录');

  var local = util.upsert(samplePayload(), { city: '上海', roadNames: '中山路' });
  check(local.id === 'S1' && local.isLocal === true, '本机摘要用 shot_id 作 id');
  check(util.readLocal().length === 1, '本机摘要已写入 Storage');
  var merged = util.merge([sampleCard('S1'), sampleCard('S2', 1763099000)], 'S1');
  check(merged.length === 2, '同 id 去重：云端覆盖本机', String(merged.length));
  check(merged.filter(function (x) { return x.id === 'S1'; })[0].isLocal === false,
    '云为准');
  check(merged.filter(function (x) { return x.id === 'S1'; })[0].isCurrent === true,
    '本次结果标记 isCurrent');
  var st = util.statsOf(merged);
  check(st.recordCount === 2 && st.trackCount === 2, '统计记录数/有效轨迹');
  check(st.distanceText === '10.4 km', '累计里程 5.2×2', st.distanceText);
  util.removeLocal('S1');
  check(util.readLocal().length === 0, 'removeLocal 生效');

  // --- 7. 记录页：云列表 + 本机兜底 ------------------------------------
  console.log('\n【7】记录页 onShow');
  b = boot({ healthEnabled: true, items: [sampleCard('S1')] });
  var recPage = loadPage('pages/records/records.js');
  b.app.globalData.lastResult = samplePayload();
  recPage.onShow();
  await flush();
  check(recPage.data.records.length === 1, '云端记录已渲染', String(recPage.data.records.length));
  check(recPage.data.storageBadge === '已云端同步', '存储状态=云端');
  check(recPage.data.records[0].isCurrent === true, '本次结果可查看');
  check(recPage.data.loading === false, '读取完成后 loading 复位');

  b = boot({ healthEnabled: false });
  util = b.recordsUtil;
  util.upsert(samplePayload(), { city: '上海' });
  recPage = loadPage('pages/records/records.js');
  recPage.onShow();
  await flush();
  check(recPage.data.records.length === 1, '云端未启用仍有本机摘要');
  check(recPage.data.storageBadge === '仅本机保存', '存储状态=本机');
  check(String(recPage.data.cloudHint).indexOf('未启用') >= 0, '给出未启用提示');

  // --- 8. 我的页统计 ---------------------------------------------------
  console.log('\n【8】我的页 onShow');
  b = boot({ healthEnabled: true, items: [sampleCard('S1'), sampleCard('S2', 1763099000)] });
  var mePage = loadPage('pages/me/me.js');
  mePage.onShow();
  await flush();
  check(mePage.data.recordCount === 2 && mePage.data.trackCount === 2,
    '统计来自云端', String(mePage.data.recordCount));
  check(mePage.data.distanceText === '10.4 km', '累计里程', mePage.data.distanceText);
  check(String(mePage.data.storageTitle).indexOf('云端') >= 0, '存储提示=云端');

  // --- 9. 历史记录详情 + 图片代理下载 ----------------------------------
  console.log('\n【9】历史记录详情');
  b = boot({
    healthEnabled: true,
    detail: {
      id: 'S1', created_at: 1763100000,
      summary: sampleCard('S1'),
      payload: samplePayload('/api/records/S1/images/S1.jpg'),
      images: [{ name: 'S1.jpg', url: '/api/records/S1/images/S1.jpg' }]
    }
  });
  var resultPage = loadPage('pages/result/result.js');
  resultPage.onLoad({ recordId: 'S1' });
  await flush();
  check(resultPage.data.ready === true, '完整 payload 已渲染');
  check(resultPage.data.shotUrl === '/tmp/record-image.png',
    '截图走带 token 的 downloadFile 代理流', String(resultPage.data.shotUrl));
  check(downloads.length === 1, '只下载一次截图', String(downloads.length));
  check(downloads[0].header.Authorization === 'Bearer TOKEN_A',
    '截图下载带 Bearer');
  check(state.refineCalls === 0, '历史记录不启动精修轮询');

  // --- 10. 删除与打开历史 ----------------------------------------------
  console.log('\n【10】删除 / 打开历史');
  b = boot({ healthEnabled: true, items: [sampleCard('S1'), sampleCard('S2', 1763099000)] });
  recPage = loadPage('pages/records/records.js');
  b.app.globalData.lastResult = samplePayload();
  recPage.onShow();
  await flush();
  recPage.openRecord({ currentTarget: { dataset: { id: 'S2' } } });
  check(navigations.some(function (u) { return u.indexOf('recordId=S2') >= 0; }),
    '点历史卡片跳转带 recordId', JSON.stringify(navigations));
  recPage.confirmDelete({ currentTarget: { dataset: { id: 'S2' } } });
  await flush();
  check(state.deleted.some(function (u) { return u.indexOf('/api/records/S2') >= 0; }),
    '云端记录调用 DELETE /api/records/{id}');

  // --- 11. 客户端不得出现任何云端私钥 ----------------------------------
  console.log('\n【11】客户端安全');
  var fs = require('fs');
  var allJs = MODULES.map(function (f) {
    return fs.readFileSync(path.join(RIDEWX, f), 'utf8');
  }).join('\n');
  check(!/COS_SECRET|WX_APP_SECRET|SecretKey|qcloud_cos/.test(allJs),
    '小程序 JS 不含 COS/微信私钥或 SDK');
  check(/Authorization/.test(allJs) && /Bearer /.test(allJs),
    '鉴权只走 Bearer token');

  console.log('');
  if (failures.length) {
    console.log('失败 ' + failures.length + ' 项：' + JSON.stringify(failures));
    process.exit(1);
  }
  console.log('全部通过');
}

main().catch(function (err) {
  console.error('测试异常：', err && err.stack || err);
  process.exit(1);
});
