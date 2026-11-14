// 「我的」收录概况：优先用云端记录统计，云不可用退回本机摘要。
var api = require('../../utils/api.js');
var records = require('../../utils/records.js');

Page({
  data: {
    recordCount: 0,
    trackCount: 0,
    distanceText: '0 km',
    storageTitle: '记录保存在本机',
    storageDesc: '当前版本暂未接入云端，识别结果保存在本机。'
  },

  onShow: function () {
    this.refresh();
  },

  refresh: function () {
    var that = this;
    var app = getApp();
    return app.ensureToken().then(function (token) {
      if (!token || !app.globalData.recordsReady) {
        that._render(records.merge([], ''), false);
        return;
      }
      return api.listRecords({ limit: 100 }).then(function (d) {
        that._render(records.merge((d || {}).items || [], ''), true);
      }).catch(function () {
        that._render(records.merge([], ''), false);
      });
    }).catch(function () {
      that._render(records.merge([], ''), false);
    });
  },

  _render: function (items, cloud) {
    var s = records.statsOf(items);
    this.setData({
      recordCount: s.recordCount,
      trackCount: s.trackCount,
      distanceText: s.distanceText,
      storageTitle: cloud ? '记录已保存在云端' : '记录保存在本机',
      storageDesc: cloud
        ? '按微信账号隔离，换设备也能查看；识别成功后自动同步。'
        : '云端记录未启用，识别结果保存在本机。'
    });
  }
});
