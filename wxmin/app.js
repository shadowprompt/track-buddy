// 全局状态：最近一次识别结果 + 运动记录 token 的获取/缓存/续期。
//
// 后端契约（src/app/records/auth.py、server.py）：
//   GET  /api/health           → {records:{enabled}}  零网络开关
//   POST /api/auth/wxlogin     → {token, expires_in, union_uid}（脱敏跨端锚点）
//   /api/recognize            带 Bearer 才自动把识别结果写入 COS；不带则零回归
//   /api/records*             必须带 Bearer，按 openid 派生的 uid 隔离
//
// 设计要点：
//   * 先探测 records.enabled：服务端未配齐 .env 时**不调用 wx.login**，
//     识别请求也不带 token，行为与旧版逐字节一致；
//   * token 缓存在 Storage，有效期内冷启动不再登录（后台刷新一次开关）；
//   * 任何一步失败都静默降级到「本机摘要」，绝不阻塞识别主流程。
var api = require('./utils/api.js');

var TOKEN_KEY = 'ride_records_token_v1';
var EXPIRE_BUFFER_MS = 60 * 1000; // 提前一分钟判过期，避免请求路上刚好失效

function readCache() {
  try {
    var v = wx.getStorageSync(TOKEN_KEY);
    if (v && v.token && Number(v.expires_at) > Date.now() + EXPIRE_BUFFER_MS) {
      return v.token;
    }
  } catch (e) { /* Storage 不可用时按无缓存处理 */ }
  return '';
}

function writeCache(token, expiresIn) {
  var ttl = Number(expiresIn) > 0 ? Number(expiresIn) : 86400;
  try {
    wx.setStorageSync(TOKEN_KEY, {
      token: token,
      expires_at: Date.now() + ttl * 1000
    });
  } catch (e) { /* 只影响下次冷启动是否重新登录 */ }
}

function clearCache() {
  try { wx.removeStorageSync(TOKEN_KEY); } catch (e) { /* ignore */ }
}

App({
  globalData: {
    // 最近一次识别结果（/api/recognize 返回的 payload）；
    // index 写入、result 读取——含 GPX/TCX 文本，不适合走页面参数传递。
    lastResult: null,
    token: '',
    // 跨端身份锚点：服务端脱敏后的 union_uid，不是明文 unionId
    unionUid: '',
    // 云端记录是否可用；false 时页面显示本机摘要兜底
    recordsReady: false,
    recordsReason: '',
    recordSource: 'local' // 'cloud' | 'local'
  },

  onLaunch: function () {
    var cached = readCache();
    if (cached) {
      this.globalData.token = cached;
      this.globalData.recordsReady = true;
      this.globalData.recordSource = 'cloud';
      // 有有效缓存就不做 wx.login；后台刷新开关状态，不阻塞任何页面。
      // （即使服务端此时刚关掉记录功能，带 token 的识别也只是不落 COS，零回归）
      var that = this;
      api.health().then(function (d) {
        if (d && d.records) {
          that.globalData.recordsReady = !!d.records.enabled;
          if (!d.records.enabled) {
            that.globalData.recordsReason = '云端记录未启用';
            that.globalData.recordSource = 'local';
          }
        }
      }).catch(function () { /* 服务暂时不可达：保留缓存 token */ });
      this.loginPromise = Promise.resolve(cached);
      return;
    }
    this.loginPromise = this._bootstrap();
  },

  /** 冷启动：探测开关 → wx.login → 换 token。永远 resolve，绝不 reject。 */
  _bootstrap: function () {
    var that = this;
    return api.health().then(function (d) {
      var enabled = !!(d && d.records && d.records.enabled);
      if (!enabled) {
        that.globalData.recordsReady = false;
        that.globalData.recordsReason = '云端记录未启用（服务端未配置）';
        that.globalData.recordSource = 'local';
        return '';
      }
      return that._wxlogin();
    }).catch(function () {
      that.globalData.recordsReady = false;
      that.globalData.recordsReason = '无法连接记录服务';
      that.globalData.recordSource = 'local';
      return '';
    });
  },

  _wxlogin: function () {
    var that = this;
    return new Promise(function (resolve) {
      wx.login({
        success: function (r) {
          if (!r || !r.code) { that._loginFailed(resolve); return; }
          api.wxlogin(r.code).then(function (d) {
            var token = d && d.token;
            if (!token) { that._loginFailed(resolve); return; }
            writeCache(token, d.expires_in);
            that.globalData.token = token;
            that.globalData.unionUid = d.union_uid || '';
            that.globalData.recordsReady = true;
            that.globalData.recordsReason = '';
            that.globalData.recordSource = 'cloud';
            resolve(token);
          }).catch(function () { that._loginFailed(resolve); });
        },
        fail: function () { that._loginFailed(resolve); }
      });
    });
  },

  _loginFailed: function (resolve) {
    this.globalData.recordsReady = false;
    this.globalData.recordsReason = '登录未完成，记录暂存在本机';
    this.globalData.recordSource = 'local';
    resolve('');
  },

  /**
   * 页面/请求层统一取 token。
   * @param {boolean} force true=丢弃缓存强制重新登录（401 续期用）
   * @returns {Promise<string>} token 或 ''，永不被拒绝
   */
  ensureToken: function (force) {
    if (force) {
      clearCache();
      this.globalData.token = '';
      this.loginPromise = this._bootstrap();
    }
    return this.loginPromise || Promise.resolve(this.globalData.token || '');
  },

  /** token 被服务端判无效（401）时由 api.js 调用，随后触发一次强制续期。 */
  invalidateToken: function () {
    clearCache();
    this.globalData.token = '';
    this.globalData.unionUid = '';
    this.globalData.recordsReady = false;
    this.globalData.recordsReason = '登录状态已失效';
    this.globalData.recordSource = 'local';
    this.loginPromise = Promise.resolve('');
  }
});
