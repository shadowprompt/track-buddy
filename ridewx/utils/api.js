// meituan-ride 后端 API 封装。
// 契约见 src/app/server.py：
//   POST /api/recognize      multipart：file(必填)+details(可多)+summary(可选)
//   GET  /api/refine/{id}    后台精修进度轮询
//   POST /api/feedback       JSON：{name, lon, lat, crs, city}
var config = require('../config.js');
var multipart = require('./multipart.js');

var BASE_URL = config.BASE_URL;

function errText(res) {
  // 后端错误统一是 HTTPException，detail 里是人话原因
  if (res && res.data && res.data.detail) return String(res.data.detail);
  if (res && res.errMsg) return res.errMsg;
  return 'HTTP ' + (res && res.statusCode ? res.statusCode : '错误');
}

function readAsBuffer(fs, filePath) {
  return new Promise(function (resolve, reject) {
    fs.readFile({
      filePath: filePath,
      success: function (res) {
        resolve(res.data);
      },
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
 * 识别骑行轨迹截图。
 * @param {Object} opts
 *   overview    {string} 概览截图临时路径（必填）
 *   details     {Array<string>} 街道级细节图临时路径（可选，多张）
 *   summary     {string} 订单详情截图临时路径（可选）
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
        // 文件名统一 ASCII，避免multipart 转义问题；后端只看扩展名
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
    // 模板复用策略：auto（默认）/ prefer（优先模板）/ off（禁用）。
    if (opts.templateMode) fields.push({ name: 'template', value: opts.templateMode });

    var body = multipart.build(fields, files);
    return new Promise(function (resolve, reject) {
      wx.request({
        url: BASE_URL + '/api/recognize',
        method: 'POST',
        header: { 'content-type': body.contentType },
        data: body.buffer,
        timeout: 600000, // 多图拼接可能到几分钟
        dataType: 'json',
        success: function (res) {
          if (res.statusCode >= 200 && res.statusCode < 300) {
            resolve(res.data);
          } else {
            reject(new Error(errText(res)));
          }
        },
        fail: function (err) {
          reject(new Error(err.errMsg || '网络请求失败'));
        }
      });
    });
  });
}

/** 轮询后台路网精修。resolve job：{state, payload?, message?, elapsed_s?} */
function refinePoll(shotId) {
  return new Promise(function (resolve, reject) {
    wx.request({
      url: BASE_URL + '/api/refine/' + encodeURIComponent(shotId),
      method: 'GET',
      timeout: 15000,
      dataType: 'json',
      success: function (res) {
        if (res.statusCode === 200) resolve(res.data);
        else reject(new Error(errText(res)));
      },
      fail: function (err) {
        reject(new Error(err.errMsg || '网络请求失败'));
      }
    });
  });
}

/**
 * 提交地名纠正（越用越准的数据入口）。
 * lon/lat 为 GCJ-02（wx.chooseLocation 返回的就是 GCJ-02，直接传）。
 */
function feedback(name, lon, lat, city) {
  return new Promise(function (resolve, reject) {
    wx.request({
      url: BASE_URL + '/api/feedback',
      method: 'POST',
      header: { 'content-type': 'application/json' },
      data: { name: name, lon: lon, lat: lat, crs: 'GCJ-02', city: city || '' },
      timeout: 15000,
      dataType: 'json',
      success: function (res) {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve(res.data);
        else reject(new Error(errText(res)));
      },
      fail: function (err) {
        reject(new Error(err.errMsg || '网络请求失败'));
      }
    });
  });
}

/** 拉取 /api/health（输入源画像清单等），供首页渲染来源选择器。 */
function health() {
  return new Promise(function (resolve, reject) {
    wx.request({
      url: BASE_URL + '/api/health',
      method: 'GET',
      timeout: 10000,
      dataType: 'json',
      success: function (res) {
        if (res.statusCode === 200) resolve(res.data);
        else reject(new Error(errText(res)));
      },
      fail: function (err) {
        reject(new Error(err.errMsg || '网络请求失败'));
      }
    });
  });
}

module.exports = {
  BASE_URL: BASE_URL,
  recognize: recognize,
  refinePoll: refinePoll,
  feedback: feedback,
  health: health
};
