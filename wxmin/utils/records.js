// 运动记录视图模型：把「云端摘要卡片」与「本机兜底摘要」统一成
// pages/records/records.wxml 需要的同一套字段，页面不需要关心数据来自哪里。
//
// 云端卡片（src/app/records/keys.py:build_summary_card）是 snake_case；
// 旧版本机记录（ride_records_v1）是驼峰。两端都在这里归一。
// 权威源是云端：同一 shot_id 同时存在时，云卡片覆盖本机摘要。
var LOCAL_KEY = 'ride_records_v1';
var LOCAL_MAX = 100;

function coordsFrom(payload) {
  var features = payload && payload.geojson && payload.geojson.features;
  if (!features || !features.length) return [];
  var geometry = features[0].geometry || {};
  var coordinates = geometry.coordinates || [];
  if (geometry.type === 'MultiLineString') {
    var flattened = [];
    coordinates.forEach(function (line) { flattened = flattened.concat(line || []); });
    return flattened;
  }
  return coordinates;
}

function distanceKm(coords) {
  var total = 0;
  var rad = Math.PI / 180;
  for (var i = 1; i < coords.length; i++) {
    var a = coords[i - 1], b = coords[i];
    var dLat = (Number(b[1]) - Number(a[1])) * rad;
    var dLon = (Number(b[0]) - Number(a[0])) * rad;
    var h = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(Number(a[1]) * rad) * Math.cos(Number(b[1]) * rad) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    total += 2 * 6371.0088 * Math.asin(Math.sqrt(Math.max(0, Math.min(1, h))));
  }
  return total;
}

function exportsOf(payload) {
  return ['geojson', 'gpx', 'tcx'].filter(function (key) { return !!payload[key]; });
}

function pad(n) { return n < 10 ? '0' + n : String(n); }

function dateText(ms) {
  var d = new Date(Number(ms) || Date.now());
  return d.getFullYear() + '/' + pad(d.getMonth() + 1) + '/' + pad(d.getDate()) +
    ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
}

function startedText(s) {
  if (!s) return '';
  return String(s).replace('T', ' ').slice(0, 16);
}

function summaryText(o) {
  if (!o) return '';
  var parts = [];
  if (o.started_at) parts.push(startedText(o.started_at));
  if (o.duration_s != null) parts.push(Math.round(Number(o.duration_s) / 60) + ' 分钟');
  if (o.distance_km != null) parts.push(o.distance_km + ' km');
  if (o.calories_kcal != null) parts.push(o.calories_kcal + ' kcal');
  return parts.join(' · ');
}

/** 补齐 wxml 里所有 *Text 字段；isCurrent 由页面在合并后统一设置。 */
function decorate(item) {
  item.distanceText = (item.hasTrack && item.distanceKm > 0)
    ? Number(item.distanceKm).toFixed(2) + ' km' : '—';
  item.pointsText = item.trackPoints ? String(item.trackPoints) : '—';
  item.confidenceText = item.confidence == null
    ? '' : '匹配度 ' + Math.round(Number(item.confidence) * 100) + '%';
  item.exportText = item.exports && item.exports.length
    ? item.exports.length + ' 种' : '—';
  return item;
}

/** 云端摘要卡片 → 视图模型。 */
function cardToView(card) {
  card = card || {};
  var km = card.distance_km != null ? Number(card.distance_km)
    : (card.distance_m != null ? Number(card.distance_m) / 1000 : 0);
  return decorate({
    id: String(card.id || ''),
    // 服务端 created_at 是秒；统一成毫秒与本机 timestamp 可比
    timestamp: Number(card.created_at || 0) * 1000,
    dateText: card.started_at ? startedText(card.started_at)
      : dateText(Number(card.created_at || 0) * 1000),
    city: card.city || '',
    routeLabel: card.city || '骑行路线',
    hasTrack: !!card.has_track,
    distanceKm: km,
    trackPoints: Number(card.track_points || 0),
    confidence: card.confidence == null ? null : Number(card.confidence),
    exports: card.exports || [],
    rideSummary: summaryText(card),
    isLocal: false
  });
}

/** 一次识别结果写一份**本机摘要**（云端不可用或保存失败时的兜底）。
 *  id 用 shot_id：与云端记录 id 一致，云端恢复后合并时自动去重、云为准。 */
function upsert(payload, meta) {
  if (!payload) return null;
  var coords = coordsFrom(payload);
  var rs = payload.ride_summary || {};
  var now = Date.now();
  var view = decorate({
    id: String(payload.shot_id || payload.record_id || now),
    timestamp: now,
    dateText: rs.started_at ? startedText(rs.started_at) : dateText(now),
    city: (meta && meta.city) || '',
    routeLabel: (meta && (meta.roadNames || meta.city)) || '骑行路线',
    hasTrack: coords.length >= 2,
    distanceKm: coords.length >= 2 ? distanceKm(coords)
      : (rs.distance_km != null ? Number(rs.distance_km) : 0),
    trackPoints: Number(payload.track_points || coords.length || 0),
    confidence: payload.confidence == null ? null : Number(payload.confidence),
    exports: exportsOf(payload),
    rideSummary: summaryText(rs),
    isLocal: true
  });
  var kept = readLocal().filter(function (r) { return r.id !== view.id; });
  kept.unshift(view);
  kept.sort(function (a, b) { return Number(b.timestamp || 0) - Number(a.timestamp || 0); });
  try {
    wx.setStorageSync(LOCAL_KEY, kept.slice(0, LOCAL_MAX));
  } catch (e) { /* 存储不可用时仍可查看当前结果 */ }
  return view;
}

function readLocal() {
  try {
    var v = wx.getStorageSync(LOCAL_KEY);
    return Object.prototype.toString.call(v) === '[object Array]' ? v : [];
  } catch (e) { return []; }
}

function removeLocal(id) {
  var kept = readLocal().filter(function (r) { return String(r.id) !== String(id); });
  try { wx.setStorageSync(LOCAL_KEY, kept); } catch (e) { /* ignore */ }
}

/**
 * 合并两种来源：云端卡片优先，本机只补云端还没有的记录。
 * @param {Array} cards 云端 /api/records 的 items（云不可用时传 []）
 * @param {string} currentId 本次识别结果的 shot_id（用于「查看本次识别结果」）
 */
function merge(cards, currentId) {
  var seen = {};
  var out = [];
  (cards || []).forEach(function (card) {
    var v = cardToView(card);
    if (!v.id || seen[v.id]) return;
    seen[v.id] = true;
    v.isCurrent = !!(currentId && v.id === currentId);
    out.push(v);
  });
  readLocal().forEach(function (item) {
    var id = String(item && item.id || '');
    if (!id || seen[id]) return;
    seen[id] = true;
    var v = decorate(extendView(item));
    v.isLocal = true;
    v.isCurrent = !!(currentId && id === currentId);
    out.push(v);
  });
  out.sort(function (a, b) {
    return Number(b.timestamp || 0) - Number(a.timestamp || 0);
  });
  return out;
}

// 旧版本机记录可能缺 isLocal/dateText 等字段，补齐不影响已有数据。
function extendView(item) {
  var v = {};
  Object.keys(item || {}).forEach(function (k) { v[k] = item[k]; });
  if (!v.dateText) v.dateText = dateText(v.timestamp || Date.now());
  if (!v.routeLabel) v.routeLabel = v.city || '骑行路线';
  if (!v.exports) v.exports = [];
  return v;
}

/** 「我的」页统计：记录数 / 有效轨迹数 / 累计里程。 */
function statsOf(items) {
  var trackCount = 0;
  var distance = 0;
  (items || []).forEach(function (r) {
    if (r.hasTrack) {
      trackCount++;
      distance += Number(r.distanceKm || 0);
    }
  });
  return {
    recordCount: (items || []).length,
    trackCount: trackCount,
    distanceText: distance ? distance.toFixed(1) + ' km' : '0 km'
  };
}

module.exports = {
  cardToView: cardToView,
  decorate: decorate,
  upsert: upsert,
  readLocal: readLocal,
  removeLocal: removeLocal,
  merge: merge,
  statsOf: statsOf
};
