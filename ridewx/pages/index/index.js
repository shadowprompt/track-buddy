var api = require('../../utils/api.js');

Page({
  data: {
    overview: '',
    details: [],
    summary: '',
    city: '',
    roadNames: '',
    busy: false,
    error: ''
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
      this.setData({ error: '请先选择①轨迹截图' });
      return;
    }
    this.setData({ busy: true, error: '' });
    api
      .recognize({
        overview: this.data.overview,
        details: this.data.details,
        summary: this.data.summary,
        city: this.data.city,
        roadNames: this.data.roadNames
      })
      .then(function (payload) {
        getApp().globalData.lastResult = payload;
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
