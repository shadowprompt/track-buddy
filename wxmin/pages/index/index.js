var api = require('../../utils/api.js');
var records = require('../../utils/records.js');

// 转发分享文案与缩略图：好友/群与朋友圈共用一套。
// 缩略图为网络图片（5:4 卡片图），需在微信内可访问。
var SHARE_TITLE = '居然还能从轨迹截图提取经纬度啊！';
var SHARE_IMAGE = 'https://wp-img.daozhao.com/fitconverter/wxmin_recognize.png';

Page({
  data: {
    overview: '',
    details: [],
    summary: '',
    city: '',
    roadNames: '',
    busy: false,
    error: '',
    // 输入源画像选择器：加载时从 /api/health 拉取可用清单。
    // sourceIndex=0 固定为「默认（服务端配置）」，即不传 source。
    sources: [{ id: '', title: '默认（服务端配置）' }],
    sourceIndex: 0,
    // 参考历史记录策略：auto（默认）/ prefer（优先历史记录）/ off（不参考历史记录）
    templateModes: [
      { id: 'auto', title: '自动（推荐）' },
      { id: 'prefer', title: '优先历史记录' },
      { id: 'off', title: '不参考历史记录' }
    ],
    templateModeIndex: 0
  },

  // 分享给好友 / 群：点右上角「转发」或页内分享按钮时触发。
  onShareAppMessage: function () {
    return {
      title: SHARE_TITLE,
      imageUrl: SHARE_IMAGE,
      path: '/pages/index/index'
    };
  },

  // 分享到朋友圈：基础库 2.11.3+，从右上角菜单进入。
  onShareTimeline: function () {
    return {
      title: SHARE_TITLE,
      imageUrl: SHARE_IMAGE
    };
  },

  onLoad: function () {
    var that = this;
    api.health().then(function (d) {
      var sp = (d && d.source_profile) || {};
      var list = (sp.available || []).map(function (p) {
        return { id: p.id, title: p.title };
      });
      if (!list.length) return;
      var sources = [{ id: '', title: '默认（服务端配置）' }].concat(list);
      var idx = 0;
      sources.forEach(function (s, i) {
        if (s.id && s.id === sp.id) idx = i;
      });
      that.setData({ sources: sources, sourceIndex: idx });
    }).catch(function () {
      // 服务未就绪：保留「默认」项，不阻塞其它功能
    });
  },

  onSourceChange: function (e) {
    this.setData({ sourceIndex: Number(e.detail.value) });
  },

  onTemplateModeChange: function (e) {
    this.setData({ templateModeIndex: Number(e.detail.value) });
  },

  pickOverview: function () {
    var that = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      success: function (res) {
        that.setData({ overview: res.tempFiles[0].tempFilePath, error: '' });
      }
    });
  },

  clearOverview: function () {
    this.setData({ overview: '' });
  },

  pickDetails: function () {
    var that = this;
    wx.chooseMedia({
      count: 9 - this.data.details.length,
      mediaType: ['image'],
      success: function (res) {
        var paths = res.tempFiles.map(function (f) {
          return f.tempFilePath;
        });
        that.setData({ details: that.data.details.concat(paths) });
      }
    });
  },

  removeDetail: function (e) {
    var idx = e.currentTarget.dataset.index;
    var list = this.data.details.slice();
    list.splice(idx, 1);
    this.setData({ details: list });
  },

  pickSummary: function () {
    var that = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      success: function (res) {
        that.setData({ summary: res.tempFiles[0].tempFilePath });
      }
    });
  },

  clearSummary: function () {
    this.setData({ summary: '' });
  },

  onCity: function (e) {
    this.setData({ city: e.detail.value });
  },

  onRoadNames: function (e) {
    this.setData({ roadNames: e.detail.value });
  },

  start: function () {
    var that = this;
    if (!this.data.overview) {
      this.setData({ error: '请先选择②轨迹概览图' });
      return;
    }
    this.setData({ busy: true, error: '' });
    var src = (this.data.sources[this.data.sourceIndex] || {}).id || '';
    api
      .recognize({
        overview: this.data.overview,
        details: this.data.details,
        summary: this.data.summary,
        city: this.data.city,
        roadNames: this.data.roadNames,
        source: src,
        templateMode: this.data.templateModes[this.data.templateModeIndex].id
      })
      .then(function (payload) {
        getApp().globalData.lastResult = payload;
        // 云端记录由服务端在识别端点带 token 时自动保存；这里同时写一份
        // 本机摘要，云端未启用/保存失败时记录页也不会空。两者以 shot_id
        // 为同一 id，云端恢复后合并时自动去重（云为准）。
        records.upsert(payload, {
          city: that.data.city,
          roadNames: that.data.roadNames
        });
        that.setData({ busy: false });
        wx.navigateTo({ url: '/pages/result/result' });
      })
      .catch(function (err) {
        that.setData({
          busy: false,
          error: '识别失败：' + (err.message || '未知错误')
        });
      });
  }
});
