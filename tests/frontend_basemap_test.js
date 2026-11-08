// 底图适配层（web/js/basemap.js）的行为测试。
//
// 目标：证明「业务代码不再依赖 TMap.*」这件事没有牺牲行为——
//   * 二次渲染仍然**原地更新几何**（异步精修能刷地图）；
//   * 腾讯桩上的起点/终点标记跟着移动；
//   * 换 provider 只需注册实现，业务侧调用不变。
//
// 用法：node frontend_basemap_test.js path/to/basemap.js
const fs = require('fs');
const path = require('path');

const adapterPath = process.argv[2] || path.join(
  __dirname, '..', 'web', 'js', 'basemap.js');
const src = fs.readFileSync(adapterPath, 'utf8');

const failures = [];
function check(cond, label, detail) {
  console.log((cond ? '  ✓ ' : '  ✗ ') + label + (cond ? '' : '  — ' + (detail || '')));
  if (!cond) failures.push(label);
}

// ---- 腾讯地图桩（复刻真实 SDK 的关键行为）----
let layerSeq = 0;
function LatLng(lat, lng) { this.lat = lat; this.lng = lng; }
function MultiPolyline(opts) {
  this.id = 'poly-' + (++layerSeq);
  this.geometries = opts.geometries;
  this.updateCount = 0;
  this.setGeometries = (g) => { this.geometries = g; this.updateCount += 1; };
}
function MultiMarker(opts) {
  this.id = 'mark-' + (++layerSeq);
  this.geometries = opts.geometries;
  this.setGeometries = (g) => { this.geometries = g; };
}
const TMap = {
  LatLng, MultiPolyline, MultiMarker,
  PolylineStyle: function (o) { Object.assign(this, o || {}); },
  MarkerStyle: function (o) { Object.assign(this, o || {}); },
  LatLngBounds: function () { this.n = 0; this.extend = () => { this.n += 1; }; },
  Map: function (container, opts) {
    this.container = container;
    this.opts = opts;
    this.fitCalls = 0;
    this.resizeCalls = 0;
    this.clickHandlers = [];
    this.fitBounds = () => { this.fitCalls += 1; };
    this.resize = () => { this.resizeCalls += 1; };
    this.on = (evt, cb) => { if (evt === 'click') this.clickHandlers.push(cb); };
    this.removeLayer = (id) => { this.removed = (this.removed || []).concat(id); };
  },
};

global.window = global;
global.TMap = TMap;

// 载入适配层（UMD：module.exports 存在时导出，否则挂到 window）
const RideBasemap = (function () {
  const module = { exports: {} };
  const fn = new Function('module', 'exports', 'window', src);
  fn(module, module.exports, global);
  return module.exports.RideBasemap || module.exports;
})();

console.log('\n【适配层基本契约】');
check(typeof RideBasemap.create === 'function', '导出 create()');
check(RideBasemap.listProviders().includes('tmap'), '注册了 tmap provider');
check(RideBasemap.listProviders().includes('amap'), '注册了 amap provider');

const adapter = RideBasemap.create('tmap', { container: 'map', zoom: 14 });
check(adapter.providerName === 'tmap', 'adapter 记录了 provider 名');
check(adapter.isAvailable() === true, '腾讯桩下 provider 可用');

console.log('\n【首次绘制 + 二次更新】');
const p1 = [{ lat: 31.2100, lon: 121.3800 }, { lat: 31.2200, lon: 121.3900 }];
const ok1 = adapter.setTrack(p1);
const poly = adapter.getPolyline();
const markers = adapter.getMarkers();
check(ok1 === true, 'setTrack 首次成功');
check(poly !== null, '创建了轨迹图层');
check(markers.length === 2, '创建了起点/终点两个标记');
check(Math.abs(poly.geometries[0].paths[0].lat - 31.2100) < 1e-9,
  '首次几何来自输入坐标', poly.geometries[0].paths[0].lat);
check(adapter.getMap().opts.zoom === 14, '地图按配置 zoom 创建');

const p2 = [{ lat: 31.2125, lon: 121.3825 }, { lat: 31.2225, lon: 121.3925 }];
adapter.setTrack(p2);
const poly2 = adapter.getPolyline();
check(poly2 === poly, '二次绘制复用同一图层实例（未重建）');
check(poly2.updateCount === 1, '二次绘制走 setGeometries 原地更新',
  'updateCount=' + poly2.updateCount);
check(Math.abs(poly2.geometries[0].paths[0].lat - 31.2125) < 1e-9,
  '轨迹几何已换成新坐标');
check(Math.abs(adapter.getMarkers()[0].geometries[0].position.lat - 31.2125) < 1e-9,
  '起点标记同步移动');
check(Math.abs(adapter.getMarkers()[1].geometries[0].position.lat - 31.2225) < 1e-9,
  '终点标记同步移动');
check(adapter.getLastCoords().length === 2, 'lastCoords 记录了当前轨迹');

console.log('\n【视野与交互】');
adapter.fitBounds(40);
check(adapter.getMap().fitCalls === 1, 'fitBounds 转发到 SDK');
adapter.resize();
check(adapter.getMap().resizeCalls === 1, 'resize 转发到 SDK');
let clicked = null;
adapter.onClick((ll) => { clicked = ll; });
check(adapter.getMap().clickHandlers.length === 1, '点击回调只绑定一次');
adapter.onClick(() => {});
check(adapter.getMap().clickHandlers.length === 1, '重复绑定不会叠加监听');
adapter.getMap().clickHandlers[0]({ latLng: { lat: 31.5, lng: 121.5 } });
check(clicked && clicked.lng === 121.5 && clicked.lat === 31.5,
  '点击回调拿到统一结构的坐标', JSON.stringify(clicked));

console.log('\n【换 provider：业务调用不变】');
// 注册一个假 provider，验证「注册即接入」
const fakeCalls = [];
RideBasemap.register('fake', {
  name: 'fake',
  available: () => true,
  createMap: (container, opts) => {
    fakeCalls.push('createMap:' + container);
    return { fitCalls: 0, fitBounds() { this.fitCalls += 1; }, resize() {}, on() {} };
  },
  createTrack: () => ({ polyline: { set() {} }, start: {}, end: {} }),
  updateTrack: () => { fakeCalls.push('updateTrack'); },
  removeTrack: () => { fakeCalls.push('removeTrack'); },
  fitBounds: () => { fakeCalls.push('fitBounds'); },
  resize: () => { fakeCalls.push('resize'); },
  onClick: () => { fakeCalls.push('onClick'); },
});
const fake = RideBasemap.create('fake', { container: 'map' });
fake.setTrack(p1);
fake.setTrack(p2);
fake.fitBounds(40);
fake.resize();
fake.onClick(() => {});
check(fakeCalls.includes('createMap:map'), '新 provider 接管建图');
check(fakeCalls.includes('updateTrack'), '二次绘制走新 provider 的更新接口');
check(fakeCalls.includes('fitBounds') && fakeCalls.includes('resize')
  && fakeCalls.includes('onClick'), '视野/尺寸/点击全部走新 provider');

console.log('\n【异常 provider 不应炸掉业务】');
RideBasemap.register('broken', {
  name: 'broken',
  available: () => true,
  createMap: () => { throw new Error('SDK 未加载'); },
  createTrack: () => ({}), updateTrack: () => {}, removeTrack: () => {},
  fitBounds: () => {}, resize: () => {}, onClick: () => {},
});
const broken = RideBasemap.create('broken', { container: 'map' });
let broke = null;
try { broken.setTrack(p1); } catch (e) { broke = e.message; }
check(broke !== null && /SDK 未加载/.test(broke), 'provider 异常向上抛出可诊断信息', broke);

console.log('\n' + (failures.length
  ? failures.length + ' 项失败: ' + failures.join('、')
  : '全部通过'));
process.exit(failures.length ? 1 : 0);
