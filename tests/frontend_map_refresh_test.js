// 验证「异步路网精修完成后，地图轨迹真的被刷新」。
//
// 背景：用户看到状态条变成「已用真实路网精修位置（贴路中位 5 米）」、
// 置信度也涨到 91%，但腾讯地图上的轨迹仍是精修前的粗配准（整体偏北）。
// 原因是第二次 render() 走 removeLayer(layer.id) + 重建图层，
// 图层实例未必有公开 id，旧轨迹没被清掉 / 重建直接抛异常，
// 而 drawMap() 把异常静默吞了，状态条照样报成功。
//
// ⚠️ 阶段 3 之后，底图调用已收敛到适配层（web/js/basemap.js），
//    业务代码只调 basemap.setTrack()，由适配层负责「原地更新几何」。
//    本测试加载**真实的适配层**（只把 TMap 换成桩），
//    覆盖：首次绘制 → 二次渲染（精修 payload）→ 几何是否原地更新。
const fs = require('fs');
const path = require('path');

const htmlPath = path.resolve(process.argv[2] || 'web/index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const code = scripts[scripts.length - 1][1];

const failures = [];
function check(cond, label, detail) {
  console.log((cond ? '  ✓ ' : '  ✗ ') + label + (cond ? '' : '  — ' + (detail || '')));
  if (!cond) failures.push(label);
}

// --- 静态检查：状态与结果不能被「只更新指标不更新地图」骗过去 ---
check(/function render\(d\) \{[\s\S]{0,400}?lastResult = d;/.test(code),
  'render() 同步 lastResult，避免精修后 ensureMapReady 用旧结果');
check(/mapOk = drawMap\(d\);/.test(code),
  'render() 记录地图刷新是否成功');
check(/function drawMap\(d\) \{[\s\S]{0,700}?return true;[\s\S]{0,300}?return false;/.test(code),
  'drawMap() 不再静默吞异常，返回成功与否');
check(/地图已刷新/.test(code) && /地图刷新失败/.test(code),
  '状态条区分「地图已刷新」与「刷新失败」');
// 阶段 3：业务代码走适配层，不再直接 new TMap.*
check(!/new TMap\./.test(code), '业务代码不再直接调用 TMap');
check(/basemap\.setTrack\(/.test(code), '业务代码通过适配层绘制轨迹');

// --- 动态检查：真的跑两次 drawMapInner ---
let markerSeq = 0;
function LatLng(lat, lng) { this.lat = lat; this.lng = lng; }
function MultiPolyline(opts) {
  this.id = 'polyline-1';
  this.geometries = opts.geometries;
  this.updateCount = 0;
  this.setGeometries = (g) => { this.geometries = g; this.updateCount += 1; };
}
function MultiMarker(opts) {
  this.id = 'marker-' + (++markerSeq);
  this.geometries = opts.geometries;
  this.setGeometries = (g) => { this.geometries = g; };
}
const TMap = {
  LatLng, MultiPolyline, MultiMarker,
  PolylineStyle: function () {}, MarkerStyle: function () {},
  LatLngBounds: function () { this.extend = () => {}; },
  Map: function () {
    this.__clickBound = false;
    this.on = () => {};
    this.fitBounds = () => {};
    this.resize = () => {};
  },
};

// 载入真实适配层
global.window = global;
global.TMap = TMap;
const adapterPath = path.join(path.dirname(htmlPath), 'js', 'basemap.js');
const RideBasemap = require(adapterPath);

let correcting = false;
const submitCorrection = () => {};
const $ = () => ({ style: { display: 'block' }, clientWidth: 640, clientHeight: 420 });

const fn = code.match(/function drawMapInner\([^)]*\) \{[\s\S]*?\n  \}/);
if (!fn) throw new Error('未找到 drawMapInner');
const fitFn = code.match(/function fitMapToTrack\([^)]*\) \{[\s\S]*?\n  \}/);
if (!fitFn) throw new Error('未找到 fitMapToTrack');

// drawMapInner 依赖闭包里的 basemap / lastPath / correcting / fitMapToTrack，
// 这里用 new Function 复刻页面里的那几个闭包变量。
const factory = new Function('RideBasemap', '$', 'submitCorrection', `
  let lastPath = [];
  let correcting = false;
  const basemap = RideBasemap.create('tmap', { container: 'map', zoom: 14 });
  ${fitFn[0]}
  ${fn[0]}
  return { drawMapInner, getPolyline: () => basemap.getPolyline(),
           getMarkers: () => basemap.getMarkers(),
           getLastPath: () => lastPath };
`);
const api = factory(RideBasemap, $, submitCorrection);
const { getPolyline, getMarkers, getLastPath } = api;

function payload(shift) {
  return {
    geojson_gcj02: {
      features: [{
        geometry: {
          coordinates: [
            [121.3800 + shift, 31.2100 + shift],
            [121.3900 + shift, 31.2200 + shift],
          ],
        },
      }],
    },
  };
}

console.log('\n【首次绘制 + 精修后二次绘制】');
api.drawMapInner(payload(0));
const polyline = getPolyline();
const markers = getMarkers();
const firstLat = polyline.geometries[0].paths[0].lat;
const firstMarkerLat = markers[0].geometries[0].position.lat;
check(polyline !== null && markers.length === 2, '首次绘制创建线与起终点标记');
check(Math.abs(firstLat - 31.2100) < 1e-9, '首次几何来自初次识别结果', firstLat);

api.drawMapInner(payload(0.0025));   // 约 270 米南/东移，模拟精修后的新坐标
const secondLat = polyline.geometries[0].paths[0].lat;
check(polyline.updateCount === 1, '二次渲染复用同一图层（未重建）',
  'updateCount=' + polyline.updateCount);
check(Math.abs(secondLat - firstLat - 0.0025) < 1e-9,
  '地图轨迹几何已换成精修后坐标', secondLat);
check(Math.abs(markers[0].geometries[0].position.lat - firstMarkerLat - 0.0025) < 1e-9,
  '起点标记同步移动');
check(Math.abs(markers[1].geometries[0].position.lat - 31.2200 - 0.0025) < 1e-9,
  '终点标记同步移动');
check(Math.abs(getLastPath()[0].lat - secondLat) < 1e-9, 'fitBounds 用的 lastPath 同步更新');

console.log('\n' + (failures.length
  ? failures.length + ' 项失败: ' + failures.join('、')
  : '全部通过'));
process.exit(failures.length ? 1 : 0);
