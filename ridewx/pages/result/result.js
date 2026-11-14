var api = require('../../utils/api.js');

// WGS-84 轨迹的客户端侧里程（haversine），用于与概要里程并列展示
function haversineKm(coords) {
  var R = 6371.0088;
  var total = 0;
  for (var i = 1; i < coords.length; i++) {
    var a = coords[i - 1];
    var b = coords[i];
    var rad = Math.PI / 180;
    var dLat = (b[1] - a[1]) * rad;
    var dLon = (b[0] - a[0]) * rad;
    var s =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(a[1] * rad) * Math.cos(b[1] * rad) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    total += 2 * R * Math.asin(Math.sqrt(s));
  }
  return total;
}

function fmtDuration(sec) {
  sec = Math.round(sec);
  var h = Math.floor(sec / 3600);
  var m = Math.floor((sec % 3600) / 60);
  var s = sec % 60;
  if (h > 0) return h + '小时' + m + '分';
  return m + '分' + (s > 0 ? s + '秒' : '');
}

Page({
  data: {
    ready: false,
    hasTrack: false,
    stats: [],
    summaryCards: [],
    summaryText: '',
    warnings: [],
    exports: [],
    // 地图
    centerLat: 39.9,
    centerLon: 116.4,
    mapScale: 13,
    polyline: [],
    circles: [],
    shotUrl: '',
    // 精修轮询
    refineState: '',
    refineElapsed: 0,
    // 纠正
    fbName: '',
    fbLon: null,
    fbLat: null,
    fbPosText: '',
    fbBusy: false,
    fbDone: false,
    fbError: ''
  },

  onLoad: function (options) {
    // 从记录列表点开的历史记录：完整 payload 在云端，按 id 拉取。
    var recordId = options && (options.recordId || options.record_id);
    if (recordId) { this._loadCloudRecord(String(recordId)); return; }
    var payload = getApp().globalData.lastResult;
    if (!payload) {
      wx.showToast({ title: '没有识别结果', icon: 'none' });
      setTimeout(function () {
        wx.navigateBack();
      }, 1200);
      return;
    }
    this.renderPayload(payload);
    if (payload.refine_pending && payload.shot_id) {
      this.startRefinePolling(payload.shot_id);
    }
  },

  /** 云端历史记录：GET /api/records/{id} → payload；截图需带 token 下载。 */
  _loadCloudRecord: function (id) {
    var that = this;
    wx.showLoading({ title: '加载记录…' });
    api.getRecord(id).then(function (rec) {
      wx.hideLoading();
      var payload = rec && rec.payload;
      if (!payload) {
        wx.showToast({ title: '记录内容为空', icon: 'none' });
        return;
      }
      that._recordId = id;
      getApp().globalData.lastResult = payload;
      that.renderPayload(payload);
      // 服务端把 /api/shots/... 改写成了 /api/records/{id}/images/{name}，
      // <image> 不能带 header，所以用 downloadFile 换成本地临时文件。
      var url = payload.shot_url || payload.overview_url || payload.stitched_url;
      if (url && String(url).indexOf('/api/records/') >= 0) {
        var name = String(url).split('/').pop();
        api.downloadRecordImage(id, name).then(function (p) {
          that.setData({ shotUrl: p });
        }).catch(function () { /* 截图缺失不影响轨迹展示 */ });
      }
    }).catch(function (err) {
      wx.hideLoading();
      wx.showToast({
        title: '加载失败：' + (err.message || '未知错误'), icon: 'none'
      });
      setTimeout(function () { wx.navigateBack(); }, 1200);
    });
  },

  renderPayload: function (payload) {
    var data = this.data;

    // --- 指标卡 -------------------------------------------------------
    var stats = [
      { k: '置信度', v: (payload.confidence != null ? (payload.confidence * 100).toFixed(0) : 0) + '%' },
      { k: '轨迹点', v: String(payload.track_points || 0) }
    ];
    var coords = this.extractCoords(payload.geojson); // WGS-84
    var hasTrack = coords.length >= 2;
    if (hasTrack) {
      stats.push({ k: '轨迹里程', v: haversineKm(coords).toFixed(2) + ' km' });
    }
    stats.push({ k: '坐标系', v: 'WGS-84' });
    // 是否参考历史记录：参考级（历史路线插值）vs 本次独立配准
    var diag = payload.diagnostics || {};
    stats.push({
      k: '历史记录',
      v: diag.template_derived ? '⚠️ 参考历史记录' : '本次独立配准'
    });

    // --- 运动概要 -----------------------------------------------------
    var rs = payload.ride_summary || {};
    var summaryCards = [];
    var summaryParts = [];
    if (rs.available) {
      if (rs.started_at) {
        var st = String(rs.started_at).replace('T', ' ');
        summaryParts.push('开始 ' + st);
        summaryCards.push({ k: '开始时间', v: st });
      }
      if (rs.duration_s != null) {
        var du = fmtDuration(rs.duration_s);
        summaryParts.push('耗时 ' + du);
        summaryCards.push({ k: '骑行耗时', v: du });
      }
      if (rs.distance_km != null) {
        summaryParts.push('概要里程 ' + rs.distance_km + ' km');
        summaryCards.push({ k: '概要里程', v: rs.distance_km + ' km' });
      }
      if (rs.calories_kcal != null) {
        summaryParts.push(rs.calories_kcal + ' kcal');
        summaryCards.push({ k: '卡路里', v: rs.calories_kcal + ' kcal' });
      }
      if (rs.carbon_g != null) {
        summaryParts.push('减碳 ' + rs.carbon_g + 'g');
        summaryCards.push({ k: '减碳', v: rs.carbon_g + ' g' });
      }
      var cc = rs.cross_check || {};
      if (cc.available) {
        summaryParts.push(cc.consistent ? '✓ 里程一致' : '⚠ 里程量级不符');
      }
    }

    // --- 地图（map 组件用 GCJ-02，与 geojson_gcj02 一致）--------------
    var polyline = [];
    var circles = [];
    var centerLat = data.centerLat;
    var centerLon = data.centerLon;
    var mapScale = data.mapScale;
    var gj02 = this.extractCoords(payload.geojson_gcj02);
    if (gj02.length >= 2) {
      var pts = [];
      var minLat = 90, maxLat = -90, minLon = 180, maxLon = -90;
      for (var i = 0; i < gj02.length; i++) {
        var lon = gj02[i][0];
        var lat = gj02[i][1];
        pts.push({ latitude: lat, longitude: lon });
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
        if (lon < minLon) minLon = lon;
        if (lon > maxLon) maxLon = lon;
      }
      polyline = [{
        points: pts,
        color: '#1a73e8ff',
        width: 4,
        arrowLine: true
      }];
      // 起点绿圈、终点红圈
      circles = [
        {
          latitude: pts[0].latitude,
          longitude: pts[0].longitude,
          radius: 12,
          color: '#1a9c5b55',
          fillColor: '#1a9c5bcc',
          strokeWidth: 1
        },
        {
          latitude: pts[pts.length - 1].latitude,
          longitude: pts[pts.length - 1].longitude,
          radius: 12,
          color: '#e5484d55',
          fillColor: '#e5484dcc',
          strokeWidth: 1
        }
      ];
      centerLat = (minLat + maxLat) / 2;
      centerLon = (minLon + maxLon) / 2;
      // 粗略按纬度差换算合适缩放级别（map scale 3-20）
      var spanKm = Math.max(maxLat - minLat, (maxLon - minLon) * 0.7) * 111;
      mapScale = 15;
      if (spanKm > 1) mapScale = 14;
      if (spanKm > 3) mapScale = 13;
      if (spanKm > 8) mapScale = 12;
      if (spanKm > 20) mapScale = 11;
      if (spanKm > 50) mapScale = 10;
    }

    // --- 导出（统一 WGS-84）------------------------------------------
    var exports = [];
    if (payload.geojson) {
      exports.push({
        key: 'geojson',
        label: 'GeoJSON（GPS / GIS 工具）',
        file: 'trajectory_wgs84.geojson',
        content: JSON.stringify(payload.geojson, null, 2)
      });
    }
    if (payload.gpx) {
      exports.push({
        key: 'gpx',
        label: 'GPX（含时间轴，Strava/Garmin）',
        file: 'trajectory.gpx',
        content: payload.gpx
      });
    }
    if (payload.tcx) {
      exports.push({
        key: 'tcx',
        label: 'TCX（Garmin Connect）',
        file: 'trajectory.tcx',
        content: payload.tcx
      });
    }

    // 云端历史记录的 shot_url 是 /api/records/{id}/images/{name}，
    // 需要 Bearer 才能读，不能像 /api/shots 那样直接拼成 <image src>。
    var shotUrl = '';
    if (payload.shot_url && String(payload.shot_url).indexOf('/api/records/') < 0) {
      shotUrl = api.BASE_URL + payload.shot_url;
    }
    this.setData({
      ready: true,
      hasTrack: hasTrack,
      stats: stats,
      summaryCards: summaryCards,
      summaryText: summaryParts.join(' · '),
      warnings: payload.warnings || [],
      exports: exports,
      centerLat: centerLat,
      centerLon: centerLon,
      mapScale: mapScale,
      polyline: polyline,
      circles: circles,
      shotUrl: shotUrl
    });
  },

  extractCoords: function (gj) {
    var coords = [];
    if (gj && gj.features && gj.features.length > 0) {
      var geom = gj.features[0].geometry || {};
      var cs = geom.coordinates || [];
      for (var i = 0; i < cs.length; i++) {
        coords.push([Number(cs[i][0]), Number(cs[i][1])]);
      }
    }
    return coords;
  },

  // --- 后台精修轮询 -----------------------------------------------------
  startRefinePolling: function (shotId) {
    var that = this;
    var attempts = 0;
    var timer = setInterval(function () {
      attempts++;
      if (attempts > 200) {
        // 10 分钟还没完就放弃轮询（识别主结果不受影响）
        clearInterval(timer);
        that.setData({ refineState: 'error', refineMessage: '精修超时，可稍后刷新重试' });
        return;
      }
      api
        .refinePoll(shotId)
        .then(function (job) {
          if (job.state === 'running') {
            that.setData({
              refineState: 'running',
              refineElapsed: Math.round(job.elapsed_s || 0)
            });
          } else if (job.state === 'done' && job.payload) {
            clearInterval(timer);
            getApp().globalData.lastResult = job.payload;
            that.renderPayload(job.payload);
            that.setData({ refineState: 'done' });
            wx.showToast({ title: '精修完成', icon: 'success' });
          } else if (job.state === 'error') {
            clearInterval(timer);
            that.setData({
              refineState: 'error',
              refineMessage: job.message || '未知错误'
            });
          }
          // state unknown：服务端刚重启丢了任务，继续等几次没意义
          if (job.state === 'unknown' && attempts > 10) {
            clearInterval(timer);
            that.setData({ refineState: 'error', refineMessage: '精修任务已丢失' });
          }
        })
        .catch(function () {
          // 单次网络抖动不打断轮询
        });
    }, 3000);
    this._refineTimer = timer;
  },

  onUnload: function () {
    if (this._refineTimer) clearInterval(this._refineTimer);
  },

  // --- 导出 ------------------------------------------------------------
  doExport: function (e) {
    var key = e.currentTarget.dataset.key;
    var item = null;
    for (var i = 0; i < this.data.exports.length; i++) {
      if (this.data.exports[i].key === key) item = this.data.exports[i];
    }
    if (!item) return;
    var fs = wx.getFileSystemManager();
    var filePath = wx.env.USER_DATA_PATH + '/' + item.file;
    fs.writeFile({
      filePath: filePath,
      data: item.content,
      encoding: 'utf8',
      success: function () {
        wx.shareFileMessage({
          filePath: filePath,
          fileName: item.file,
          fail: function () {
            // 分享面板取消或环境不支持时，文件其实已写好
            wx.showToast({ title: '已保存: ' + item.file, icon: 'none' });
          }
        });
      },
      fail: function (err) {
        wx.showToast({
          title: '写文件失败: ' + (err.errMsg || ''),
          icon: 'none'
        });
      }
    });
  },

  // --- 纠正 ------------------------------------------------------------
  onFbName: function (e) {
    this.setData({ fbName: e.detail.value, fbDone: false });
  },

  pickLocation: function () {
    var that = this;
    wx.chooseLocation({
      success: function (res) {
        that.setData({
          fbLon: res.longitude,
          fbLat: res.latitude,
          fbPosText: (res.name || res.address || '已选点') +
            ' (' + res.longitude.toFixed(5) + ', ' + res.latitude.toFixed(5) + ')'
        });
        if (!that.data.fbName && res.name) {
          that.setData({ fbName: res.name });
        }
      }
    });
  },

  submitFeedback: function () {
    var that = this;
    var d = this.data;
    if (!d.fbName) {
      this.setData({ fbError: '请填写地名' });
      return;
    }
    if (d.fbLon == null) {
      this.setData({ fbError: '请先在地图上选点' });
      return;
    }
    this.setData({ fbBusy: true, fbError: '' });
    api
      .feedback(d.fbName, d.fbLon, d.fbLat, '')
      .then(function () {
        that.setData({ fbBusy: false, fbDone: true });
      })
      .catch(function (err) {
        that.setData({
          fbBusy: false,
          fbError: '提交失败：' + (err.message || '未知错误')
        });
      });
  },

  goBack: function () {
    wx.navigateBack();
  }
});
