// 记录列表：权威源是云端 /api/records（按微信账号隔离）；
// 云端未启用 / 请求失败时退回本机摘要，保证记录页不会空白。
var api = require('../../utils/api.js');
var records = require('../../utils/records.js');

var CLOUD_FOOTNOTE = '记录保存在云端，按微信账号隔离；未启用云端时自动保存在本机。';
var LOCAL_FOOTNOTE = '当前显示本机摘要；云端启用后会自动切换到账号记录。';

Page({
  data: {
    records: [],
    loading: false,
    cloudHint: '',
    storageBadge: '云端同步',
    footnote: CLOUD_FOOTNOTE,
    hasMore: false,
    nextCursor: ''
  },

  onShow: function () {
    this.refresh();
  },

  _currentId: function () {
    var p = getApp().globalData.lastResult;
    return p ? String(p.shot_id || p.record_id || p._local_record_id || '') : '';
  },

  /** 云端为主；失败/未启用时退回本机摘要。 */
  refresh: function () {
    var that = this;
    var app = getApp();
    this.setData({ loading: true, cloudHint: '' });
    return app.ensureToken().then(function (token) {
      if (!token || !app.globalData.recordsReady) {
        that._local('云端记录未启用，已显示本机摘要');
        return;
      }
      return api.listRecords({ limit: 100 }).then(function (d) {
        d = d || {};
        that.setData({
          records: records.merge(d.items || [], that._currentId()),
          loading: false,
          cloudHint: '',
          storageBadge: '已云端同步',
          footnote: CLOUD_FOOTNOTE,
          hasMore: !!d.next_cursor,
          nextCursor: d.next_cursor || ''
        });
      }).catch(function () {
        that._local('云端记录读取失败，已显示本机摘要');
      });
    }).catch(function () {
      that._local('暂时无法连接记录服务，已显示本机摘要');
    });
  },

  _local: function (hint) {
    this.setData({
      records: records.merge([], this._currentId()),
      loading: false,
      cloudHint: hint,
      storageBadge: '仅本机保存',
      footnote: LOCAL_FOOTNOTE,
      hasMore: false,
      nextCursor: ''
    });
  },

  /** 云端 keyset 分页：把下一页卡片追加进来，同 id 去重。 */
  loadMore: function () {
    var that = this;
    var cursor = this.data.nextCursor;
    if (!cursor || this.data.loading) return;
    this.setData({ loading: true });
    api.listRecords({ limit: 100, cursor: cursor }).then(function (d) {
      d = d || {};
      var seen = {};
      var list = that.data.records.slice();
      list.forEach(function (r) { seen[r.id] = true; });
      (d.items || []).forEach(function (card) {
        var v = records.cardToView(card);
        if (!v.id || seen[v.id]) return;
        seen[v.id] = true;
        v.isCurrent = !!(that._currentId() && v.id === that._currentId());
        list.push(v);
      });
      list.sort(function (a, b) {
        return Number(b.timestamp || 0) - Number(a.timestamp || 0);
      });
      that.setData({
        records: list,
        loading: false,
        hasMore: !!d.next_cursor,
        nextCursor: d.next_cursor || ''
      });
    }).catch(function () {
      that.setData({ loading: false, cloudHint: '加载更多失败，可稍后重试' });
    });
  },

  /** 点卡片：本次结果直接看；云端历史记录带 recordId 打开完整 payload。 */
  openRecord: function (e) {
    if (this._longPressed) { this._longPressed = false; return; }
    var id = String(e.currentTarget.dataset.id || '');
    if (!id) return;
    var item = null;
    for (var i = 0; i < this.data.records.length; i++) {
      if (this.data.records[i].id === id) { item = this.data.records[i]; break; }
    }
    if (!item) return;
    if (item.isCurrent) { this.openCurrent(); return; }
    if (item.isLocal) {
      wx.showToast({ title: '本机摘要暂无完整轨迹', icon: 'none' });
      return;
    }
    wx.navigateTo({
      url: '/pages/result/result?recordId=' + encodeURIComponent(id)
    });
  },

  openCurrent: function () {
    if (!getApp().globalData.lastResult) {
      wx.showToast({ title: '本次结果暂不可查看', icon: 'none' });
      return;
    }
    wx.navigateTo({ url: '/pages/result/result' });
  },

  /** 长按卡片删除：云端记录调 DELETE，本机摘要只删本地，两者都清一遍。 */
  confirmDelete: function (e) {
    var that = this;
    var id = String(e.currentTarget.dataset.id || '');
    if (!id) return;
    // longpress 后可能还会补发一次 tap，先屏蔽；800ms 后自动复位
    this._longPressed = true;
    setTimeout(function () { that._longPressed = false; }, 800);
    var isLocal = true;
    for (var i = 0; i < this.data.records.length; i++) {
      if (this.data.records[i].id === id) {
        isLocal = !!this.data.records[i].isLocal;
        break;
      }
    }
    wx.showModal({
      title: '删除记录',
      content: '将删除这条运动记录的轨迹与截图，操作不可恢复。',
      success: function (r) {
        if (!r.confirm) return;
        var finish = function () {
          records.removeLocal(id);
          wx.showToast({ title: '已删除', icon: 'success' });
          that.refresh();
        };
        if (isLocal) { finish(); return; }
        api.deleteRecord(id).then(finish).catch(function (err) {
          wx.showToast({
            title: '删除失败：' + (err.message || '未知错误'), icon: 'none'
          });
        });
      }
    });
  },

  goImport: function () { wx.switchTab({ url: '/pages/index/index' }); }
});
