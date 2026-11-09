/**
 * 底图适配层（阶段 3）。
 *
 * 为什么要这一层：
 * 腾讯地图 GL JS 的对象模型（`new TMap.Map` / `MultiPolyline` / `MultiMarker`
 * / `LatLngBounds`）与业务逻辑是**两种东西**，但此前它们混在同一个
 * `drawMapInner()` 里。后果是：
 *
 *   * 换底图服务商（高德 JS API / 天地图 JS API / Leaflet）要改业务函数；
 *   * 测试必须 mock 整个 `TMap` 全局对象才能跑一条轨迹；
 *   * 厂商 API 一旦在某个浏览器版本上改名，业务代码跟着炸。
 *
 * 现在业务代码只依赖 `RideBasemap.Adapter` 这套统一接口：
 *
 *     const basemap = RideBasemap.create('tmap', { container: 'map' });
 *     basemap.setTrack([{ lat, lon }, ...]);
 *     basemap.fitBounds();
 *     basemap.onClick((latLng) => { ... });
 *
 * 新增服务商 = 在 `providers` 里加一个实现，业务代码零改动。
 * 每个 provider 只回答 7 个问题：能不能用、怎么建地图、怎么建轨迹、
 * 怎么更新轨迹、怎么删图层、怎么缩放视野、怎么监听点击。
 *
 * ⚠️ 坐标一律用 `{ lat, lon }` 且为 **GCJ-02**（美团/腾讯/高德底图坐标系）。
 * WGS-84 直接画上去会整体偏约 350~500 米。
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    (root || (typeof window !== 'undefined' ? window : globalThis)).RideBasemap = api;
  }
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  const G = (typeof window !== 'undefined' && window) ? window : globalThis;

  /** 服务商注册表：名称 -> 实现。 */
  const providers = Object.create(null);

  function register(name, impl) {
    providers[name] = impl;
    return impl;
  }

  function getProvider(name) {
    const impl = providers[name];
    if (!impl) {
      throw new Error('未注册的底图服务商: ' + name +
        '（已注册: ' + Object.keys(providers).join(', ') + '）');
    }
    return impl;
  }

  function listProviders() {
    return Object.keys(providers);
  }

  // ------------------------------------------------------------------
  // 腾讯地图 GL JS
  // ------------------------------------------------------------------
  const tmap = {
    name: 'tmap',
    label: '腾讯地图',
    available() {
      return typeof G.TMap !== 'undefined' && !!G.TMap.Map;
    },
    createMap(container, opts) {
      const T = G.TMap;
      return new T.Map(container, {
        center: new T.LatLng(opts.lat, opts.lon),
        zoom: opts.zoom || 14,
      });
    },
    createTrack(map, coords) {
      const T = G.TMap;
      const path = coords.map((c) => new T.LatLng(c.lat, c.lon));
      const polyline = new T.MultiPolyline({
        map,
        styles: {
          trail: new T.PolylineStyle({
            color: '#534ab7', width: 5, borderWidth: 2,
            borderColor: '#ffffff', lineCap: 'round',
          }),
        },
        geometries: [{ id: 'trail', styleId: 'trail', paths: path }],
      });
      const start = new T.MultiMarker({
        map,
        styles: {
          s: new T.MarkerStyle({
            width: 24, height: 24, anchor: { x: 12, y: 12 },
            color: '#0f6e56', borderWidth: 3, borderColor: '#ffffff',
          }),
        },
        geometries: [{ id: 's', styleId: 's', position: path[0] }],
      });
      const end = new T.MultiMarker({
        map,
        styles: {
          e: new T.MarkerStyle({
            width: 24, height: 24, anchor: { x: 12, y: 12 },
            color: '#d54941', borderWidth: 3, borderColor: '#ffffff',
          }),
        },
        geometries: [{ id: 'e', styleId: 'e', position: path[path.length - 1] }],
      });
      return { polyline: polyline, start: start, end: end };
    },
    // ⚠️ 必须「原地换几何」而不是 removeLayer + 重建：
    //    异步路网精修会带着新坐标再画一次，重建会让旧轨迹留在图上，
    //    用户看到的是「指标更新了、地图没动」。
    updateTrack(track, coords) {
      const T = G.TMap;
      const path = coords.map((c) => new T.LatLng(c.lat, c.lon));
      track.polyline.setGeometries([
        { id: 'trail', styleId: 'trail', paths: path },
      ]);
      track.start.setGeometries([
        { id: 's', styleId: 's', position: path[0] },
      ]);
      track.end.setGeometries([
        { id: 'e', styleId: 'e', position: path[path.length - 1] },
      ]);
    },
    removeTrack(map, track) {
      [track.polyline, track.start, track.end].forEach((layer) => {
        if (!layer) return;
        try {
          // ⚠️ 真实 SDK 的 removeLayer 要求**图层 id 字符串**，
          //    传对象会抛 "The layer '[object Object]' does not exist in map"。
          map.removeLayer(layer.id);
        } catch (e) { /* 图层可能已失效，忽略 */ }
      });
    },
    fitBounds(map, coords, padding, track) {
      const T = G.TMap;
      const bounds = new T.LatLngBounds();
      coords.forEach((c) => bounds.extend(new T.LatLng(c.lat, c.lon)));
      map.fitBounds(bounds, { padding: padding });
    },
    resize(map) { map.resize(); },
    onClick(map, cb) {
      // 统一回调结构为 { lat, lng }——腾讯的 evt.latLng 是 LatLng 实例，
      // 高德是 { lnglat: { lat, lng } }，两者形状不同，
      // 必须在这里抹平，否则业务代码要按厂商分支。
      map.on('click', (evt) => {
        const ll = evt && evt.latLng;
        if (!ll) return;
        cb({ lat: ll.lat, lng: ll.lng });
      });
    },
  };

  // ------------------------------------------------------------------
  // 高德地图 JS API 2.0
  //
  // 与腾讯的差异：AMap 用**单个** Polyline / Marker 覆盖物（没有 Multi*），
  // 更新靠 setPath / setPosition，删除靠 map.remove(overlay)。
  // 这些差异全部被关在这个实现里，业务代码看不到。
  //
  // 启用条件：页面已加载高德 JS SDK 且配置了 key 与安全密钥。
  // 由服务端按配置渲染 SDK 脚本与 _AMapSecurityConfig。
  // ------------------------------------------------------------------
  const amap = {
    name: 'amap',
    label: '高德地图',
    available() {
      return typeof G.AMap !== 'undefined' && !!G.AMap.Map;
    },
    createMap(container, opts) {
      const A = G.AMap;
      return new A.Map(container, {
        center: [opts.lon, opts.lat],
        zoom: opts.zoom || 14,
        viewMode: '2D',
      });
    },
    createTrack(map, coords) {
      const A = G.AMap;
      const path = coords.map((c) => [c.lon, c.lat]);
      const polyline = new A.Polyline({
        path: path,
        strokeColor: '#534ab7',
        strokeWeight: 5,
        strokeOpacity: 1,
        lineJoin: 'round',
        lineCap: 'round',
        showDir: false,
      });
      const start = new A.Marker({
        position: path[0],
        anchor: 'center',
        content: '<div style="width:18px;height:18px;border-radius:50%;'
          + 'background:#0f6e56;border:3px solid #fff;box-sizing:content-box"></div>',
      });
      const end = new A.Marker({
        position: path[path.length - 1],
        anchor: 'center',
        content: '<div style="width:18px;height:18px;border-radius:50%;'
          + 'background:#d54941;border:3px solid #fff;box-sizing:content-box"></div>',
      });
      map.add([polyline, start, end]);
      return { polyline: polyline, start: start, end: end };
    },
    updateTrack(track, coords) {
      const path = coords.map((c) => [c.lon, c.lat]);
      track.polyline.setPath(path);
      track.start.setPosition(path[0]);
      track.end.setPosition(path[path.length - 1]);
    },
    removeTrack(map, track) {
      [track.polyline, track.start, track.end].forEach((overlay) => {
        if (!overlay) return;
        try { map.remove(overlay); } catch (e) { /* 已移除 */ }
      });
    },
    fitBounds(map, coords, padding, track) {
      const A = G.AMap;
      const bounds = new A.Bounds();
      coords.forEach((c) => bounds.extend([c.lon, c.lat]));
      // AMap 的 setFitView 接受**覆盖物数组**，padding 参数名为 avoid
      // （上、右、下、左）。用覆盖物才能保证整条轨迹都在视野内。
      const overlays = [track.polyline, track.start, track.end].filter(Boolean);
      if (overlays.length) {
        map.setFitView(overlays, false, [padding, padding, padding, padding]);
      } else {
        map.setCenter(bounds.getCenter());
      }
    },
    resize(map) { map.resize(); },
    onClick(map, cb) {
      map.on('click', (evt) => cb({ lat: evt.lnglat.lat, lng: evt.lnglat.lng }));
    },
  };

  register('tmap', tmap);
  register('amap', amap);

  // ------------------------------------------------------------------
  // 统一适配器
  // ------------------------------------------------------------------

  /**
   * @param {string} providerName 已注册的服务商名
   * @param {{container: string, zoom?: number}} opts
   */
  function create(providerName, opts) {
    return new Adapter(providerName, opts || {});
  }

  class Adapter {
    constructor(providerName, opts) {
      this.provider = getProvider(providerName);
      this.container = opts.container || 'map';
      this.zoom = opts.zoom || 14;
      this.map = null;
      this.track = null;
      this.lastCoords = [];
      this.clickBound = false;
    }

    get providerName() { return this.provider.name; }

    isAvailable() {
      try {
        return this.provider.available();
      } catch (e) {
        return false;
      }
    }

    /** 懒建地图；已建好就直接返回。 */
    ensureMap(center) {
      if (this.map) return this.map;
      if (!this.isAvailable()) return null;
      this.map = this.provider.createMap(this.container, {
        lat: center.lat, lon: center.lon, zoom: this.zoom,
      });
      return this.map;
    }

    /**
     * 画/更新轨迹。
     *
     * 二次调用走「原地更新几何」；原地更新失败才回退到重建图层。
     * 返回 false 表示当前无法绘制（服务商不可用）。
     */
    setTrack(coords, center) {
      if (!coords || !coords.length) return false;
      const c = center || coords[0];
      const map = this.ensureMap(c);
      if (!map) return false;
      if (this.track) {
        try {
          this.provider.updateTrack(this.track, coords);
          this.lastCoords = coords.slice();
          return true;
        } catch (e) {
          console.warn('图层原地更新失败，回退到重建:', e);
          this.clearTrack();
        }
      }
      this.track = this.provider.createTrack(map, coords);
      this.lastCoords = coords.slice();
      return true;
    }

    clearTrack() {
      if (this.map && this.track) {
        this.provider.removeTrack(this.map, this.track);
      }
      this.track = null;
      this.lastCoords = [];
    }

    fitBounds(padding) {
      if (!this.map || !this.lastCoords.length) return false;
      this.provider.fitBounds(this.map, this.lastCoords,
        padding === undefined ? 40 : padding, this.track);
      return true;
    }

    resize() {
      if (!this.map) return false;
      this.provider.resize(this.map);
      return true;
    }

    /** 点击回调只绑定一次（SDK 的 on 是累加的）。 */
    onClick(cb) {
      if (!this.map || this.clickBound) return false;
      this.clickBound = true;
      this.provider.onClick(this.map, cb);
      return true;
    }

    // ---- 调试与测试用的读取口 ----
    getMap() { return this.map; }
    getPolyline() { return this.track ? this.track.polyline : null; }
    getMarkers() {
      if (!this.track) return [];
      return [this.track.start, this.track.end].filter(Boolean);
    }
    getLastCoords() { return this.lastCoords.slice(); }
  }

  return {
    register: register,
    create: create,
    getProvider: getProvider,
    listProviders: listProviders,
    Adapter: Adapter,
  };
});
