// 运动轨迹识别后端 API 封装。
// 契约见 src/app/server.py：
//   GET  /api/health                         能力清单 + records.enabled
//   POST /wxminTrackBuddy/login             wx.login code → 记录 token（gateway 签发，
//                                           降级端点 /api/auth/wxlogin 在 track-buddy）
//   POST /api/recognize                     multipart：file + details + summary
//   GET  /api/refine/{id}                   后台精修进度轮询
//   POST /api/feedback                      JSON：{name, lon, lat, crs, city}
//   GET    /api/records                     我的记录列表（keyset 分页）
//   GET    /api/records/{id}                单条完整记录
//   GET    /api/records/{id}/images/{name}  记录里的截图（后端代理流，需 token）
//   DELETE /api/records/{id}                删除我的记录
//
// 鉴权：除 health/wxlogin/recognize 的兜底路径外，/api/records* 必须带
// Authorization: Bearer <token>。token 由 app.js 统一获取/缓存，本文件只取用，
// 绝不接触 COS 私钥、openid 或 session_key。
var config = require('../config.js');
var multipart = require('./multipart.js');

var BASE_URL = config.BASE_URL;

function errText(res) {
  // 后端错误统一是 HTTPException，detail 里是人话原因
  if (res && res.data && res.data.detail) return String(res.data.detail);
  var code = res && res.statusCode;
  // 反代（Nginx）读超时会先于 wx.request 的 600s 超时返回，且 body 是
  // HTML：dataType:'json' 下 res.data 不是对象。别让用户只看到「HTTP 504」。
  if (code === 504 || code === 502) {
    return '识别耗时较长，网关等待超时（' + code + '）。请稍后重试；'
      + '若反复出现，需把服务器 Nginx 的 proxy_read_timeout 调大。';
  }
  if (code === 503) return '服务暂时不可用，请稍后重试（503）';
  if (res && res.errMsg) return res.errMsg;
  return 'HTTP ' + (code || '错误');
}

function extend(target, src) {
  var out = target || {};
  Object.keys(src || {}).forEach(function (k) { out[k] = src[k]; });
  return out;
}

function appToken(force) {
  var app = (typeof getApp === 'function') ? getApp() : null;
  if (!app || typeof app.ensureToken !== 'function') return Promise.resolve('');
  return Promise.resolve(app.ensureToken(force)).catch(function () { return ''; });
}

/** 在业务 header（如 multipart content-type）之上叠加 Bearer。 */
function authHeader(extra, force) {
  return appToken(force).then(function (token) {
    var h = extend({}, extra);
    if (token) h.Authorization = 'Bearer ' + token;
    return h;
  });
}

function rawRequest(method, path, opts) {
  opts = opts || {};
  return new Promise(function (resolve, reject) {
    wx.request({
      url: (opts.baseUrl || BASE_URL) + path,
      method: method,
      header: opts.header || {},
      data: opts.data,
      timeout: opts.timeout || 15000,
      dataType: 'json',
      success: function (res) {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(res.data);
          return;
        }
        var err = new Error(errText(res));
        err.statusCode = res.statusCode;
        reject(err);
      },
      fail: function (err) {
        var msg = (err && err.errMsg) ? String(err.errMsg) : '网络请求失败';
        if (/timeout/i.test(msg)) msg = '识别超时：请检查网络后重试';
        reject(new Error(msg));
      }
    });
  });
}

/** 带 Bearer 的请求；401 时强制重新登录并重试一次（避免死循环）。 */
function authedRequest(method, path, opts) {
  opts = opts || {};
  var force = !!opts.force;
  return authHeader(opts.header, force).then(function (header) {
    return rawRequest(method, path, {
      header: header,
      data: opts.data,
      timeout: opts.timeout
    });
  }).catch(function (err) {
    if (err.statusCode !== 401 || force) throw err;
    var app = (typeof getApp === 'function') ? getApp() : null;
    if (app && typeof app.invalidateToken === 'function') app.invalidateToken();
    return authedRequest(method, path, extend(extend({}, opts), { force: true }));
  });
}

function readAsBuffer(fs, filePath) {
  return new Promise(function (resolve, reject) {
    fs.readFile({
      filePath: filePath,
      success: function (res) { resolve(res.data); },
      fail: function (err) {
        reject(new Error('读取文件失败: ' + (err.errMsg || filePath)));
      }
    });
  });
}

function imgType(path) {
  return /\.png$/i.test(path) ? 'image/png' : 'image/jpeg';
}

/**
 * 识别运动轨迹截图。
 * 带合法 token 时服务端会自动把本次结果写入云端记录；无 token 时行为不变。
 * @param {Object} opts
 *   overview    {string} 轨迹概览图临时路径（必填）
 *   details     {Array<string>} 轨迹街道级细节图临时路径（可选，多张）
 *   summary     {string} 运动概要截图临时路径（可选）
 *   city        {string} 城市（可选）
 *   roadNames   {string} 路名约束，逗号分隔（可选）
 * @returns {Promise<Object>} 识别 payload（与网页版同一结构）
 */
function recognize(opts) {
  var fs = wx.getFileSystemManager();
  var reads = [readAsBuffer(fs, opts.overview)];
  (opts.details || []).forEach(function (p) {
    reads.push(readAsBuffer(fs, p));
  });
  if (opts.summary) reads.push(readAsBuffer(fs, opts.summary));

  return Promise.all(reads).then(function (buffers) {
    var files = [
      {
        name: 'file',
        // 文件名统一 ASCII，避免 multipart 转义问题；后端只看扩展名
        filename: /\.png$/i.test(opts.overview) ? 'overview.png' : 'overview.jpg',
        contentType: imgType(opts.overview),
        data: buffers[0]
      }
    ];
    var k = 1;
    (opts.details || []).forEach(function (p, idx) {
      files.push({
        name: 'details',
        filename: 'detail_' + idx + (/\.png$/i.test(p) ? '.png' : '.jpg'),
        contentType: imgType(p),
        data: buffers[k]
      });
      k++;
    });
    if (opts.summary) {
      files.push({
        name: 'summary',
        filename: /\.png$/i.test(opts.summary) ? 'summary.png' : 'summary.jpg',
        contentType: imgType(opts.summary),
        data: buffers[k]
      });
    }

    var fields = [
      { name: 'output_crs', value: 'WGS-84' },
      { name: 'allow_online', value: 'true' },
      { name: 'refine', value: 'auto' }
    ];
    if (opts.city) fields.push({ name: 'city', value: opts.city });
    if (opts.roadNames) fields.push({ name: 'road_names', value: opts.roadNames });
    // 输入源画像（可选）：截图来自哪个 App。空 = 走服务端 SOURCE_PROFILE 默认。
    if (opts.source) fields.push({ name: 'source', value: opts.source });
    // 参考历史记录策略：auto（默认）/ prefer（优先历史记录）/ off（不参考历史记录）。
    if (opts.templateMode) fields.push({ name: 'template', value: opts.templateMode });

    var body = multipart.build(fields, files);
    return authedRequest('POST', '/api/recognize', {
      header: { 'content-type': body.contentType },
      data: body.buffer,
      timeout: 600000 // 多图拼接可能到几分钟
    });
  });
}

/**
 * 用 wx.login 的 code 换记录 token（不要求鉴权）。
 * 签发收敛到 gateway（LOGIN_BASE_URL + /wxminTrackBuddy/login），响应契约与
 * track-buddy 的 /api/auth/wxlogin 逐字段一致。gateway 网络不可达 / 5xx / 404
 * 时降级回 track-buddy 自带端点；400/401 是 code 本身被微信拒绝，换端点结果
 * 相同，直接抛出不兜底。
 */
function wxlogin(code) {
  var loginOpts = {
    header: { 'content-type': 'application/json' },
    data: { code: code },
    timeout: 15000
  };
  if (!config.LOGIN_BASE_URL) {
    return rawRequest('POST', '/api/auth/wxlogin', loginOpts);
  }
  return rawRequest('POST', '/wxminTrackBuddy/login',
    extend(extend({}, loginOpts), { baseUrl: config.LOGIN_BASE_URL })
  ).catch(function (err) {
    var st = err && err.statusCode;
    if (st && st < 500 && st !== 404) throw err;
    return rawRequest('POST', '/api/auth/wxlogin', loginOpts);
  });
}

/** 当前 token 的跨端身份：{uid, union_uid, linked_uids}（仅本人可见）。 */
function identity() {
  return authedRequest('GET', '/api/records/identity', { timeout: 15000 });
}

/** 我的记录列表；返回 {items, next_cursor, total}。 */
function listRecords(opts) {
  opts = opts || {};
  var q = '?limit=' + encodeURIComponent(String(opts.limit || 20));
  if (opts.cursor) q += '&cursor=' + encodeURIComponent(opts.cursor);
  return authedRequest('GET', '/api/records' + q, { timeout: 15000 });
}

/** 单条完整记录：{id, created_at, summary, payload, images}。 */
function getRecord(id) {
  return authedRequest('GET', '/api/records/' + encodeURIComponent(id),
    { timeout: 15000 });
}

/** 删除我的整条记录。 */
function deleteRecord(id) {
  return authedRequest('DELETE', '/api/records/' + encodeURIComponent(id),
    { timeout: 15000 });
}

/**
 * 下载记录里的截图到本地临时文件，供 <image src> 使用。
 * 图片端点需要 Bearer，而 <image> 不能带自定义 header，因此必须走
 * wx.downloadFile（它支持 header）拿 tempFilePath，不能直接拼 URL。
 */
function downloadRecordImage(id, name) {
  return authHeader({}).then(function (header) {
    return new Promise(function (resolve, reject) {
      wx.downloadFile({
        url: BASE_URL + '/api/records/' + encodeURIComponent(id) +
          '/images/' + encodeURIComponent(name),
        header: header,
        timeout: 15000,
        success: function (res) {
          if (res.statusCode >= 200 && res.statusCode < 300 && res.tempFilePath) {
            resolve(res.tempFilePath);
          } else {
            reject(new Error('图片下载失败 HTTP ' + res.statusCode));
          }
        },
        fail: function (err) {
          reject(new Error(err.errMsg || '图片下载失败'));
        }
      });
    });
  });
}

/** 轮询后台路网精修。resolve job：{state, payload?, message?, elapsed_s?} */
function refinePoll(shotId) {
  return rawRequest('GET', '/api/refine/' + encodeURIComponent(shotId),
    { timeout: 15000 });
}

/** 提交地名纠正（越用越准的数据入口）。lon/lat 为 GCJ-02。 */
function feedback(name, lon, lat, city) {
  return rawRequest('POST', '/api/feedback', {
    header: { 'content-type': 'application/json' },
    data: { name: name, lon: lon, lat: lat, crs: 'GCJ-02', city: city || '' },
    timeout: 15000
  });
}

/** 拉取 /api/health（输入源画像、记录开关）。 */
function health() {
  return rawRequest('GET', '/api/health', { timeout: 10000 });
}

module.exports = {
  BASE_URL: BASE_URL,
  recognize: recognize,
  wxlogin: wxlogin,
  identity: identity,
  listRecords: listRecords,
  getRecord: getRecord,
  deleteRecord: deleteRecord,
  downloadRecordImage: downloadRecordImage,
  refinePoll: refinePoll,
  feedback: feedback,
  health: health
};
