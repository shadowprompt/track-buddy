"""轨迹地图对比页：把识别结果画到真实腾讯地图上。

用途：低缩放截图的定位结果光看 GeoJSON 坐标看不出对错，
必须落到真实底图上才能一眼看出「落在哪个街区」。

这个页面同时叠加两条轨迹做对比：
  * 红 = 纯 OCR 锚点定位（无用户路名）
  * 蓝 = 加了用户路名约束后的定位

用法：
    打开 http://127.0.0.1:8848/traj-map?shot=1793621605_dc366c0b
    （shot 参数省略时用最近一次识别结果）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

from .server import DATA_DIR, _TMAP_SECRET, _server_port, render_page

# 最近一次识别结果（含路名约束前后的对比）
_LAST: dict[str, Any] = {}


def set_last_result(payload: dict[str, Any]) -> None:
    """记录最近一次识别结果，供地图页读取。"""
    _LAST.clear()
    _LAST.update(payload)


def load_result(shot_id: str | None) -> dict[str, Any]:
    if shot_id:
        p = DATA_DIR / "last_result.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    if _LAST:
        return _LAST
    p = DATA_DIR / "last_result.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="还没有识别结果")


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>轨迹地图对照 · 美团骑行截图识别</title>
<script src="https://map.qq.com/api/gljs?v=1.exp"></script>
<style>
  html, body { margin:0; height:100%; font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  #map { position:absolute; inset:0; }
  .panel {
    position:absolute; left:16px; top:16px; z-index:10;
    background:rgba(255,255,255,.96); border:1px solid rgba(0,0,0,.12);
    border-radius:12px; padding:14px 16px; min-width:290px;
    box-shadow:0 6px 24px rgba(0,0,0,.10); font-size:13px; line-height:1.65;
  }
  .panel h2 { margin:0 0 10px; font-size:14px; font-weight:500; }
  .row { display:flex; align-items:center; gap:8px; margin:3px 0; }
  .sw { width:16px; height:4px; border-radius:2px; flex:0 0 auto; }
  .kv { display:flex; justify-content:space-between; gap:14px; color:#555; }
  .kv b { color:#111; font-weight:500; }
  .sep { height:1px; background:rgba(0,0,0,.08); margin:10px 0; }
  .note { color:#666; font-size:12px; margin-top:8px; line-height:1.6; }
  .tag { display:inline-block; padding:1px 7px; border-radius:9px; font-size:11px; }
  .tag.a { background:#e6f1fb; color:#185fa5; }
  .tag.b { background:#e1f5ee; color:#0f6e56; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h2>识别轨迹 · 真实地图对照</h2>
  <div class="row"><span class="sw" style="background:#d54941"></span>
    <span>纯 OCR 锚点定位</span>
    <span class="kv" style="margin-left:auto"><b id="c1"></b></span></div>
  <div class="row"><span class="sw" style="background:#378add"></span>
    <span>加用户路名约束</span>
    <span class="kv" style="margin-left:auto"><b id="c2"></b></span></div>
  <div class="sep"></div>
  <div class="kv"><span>起点</span><b id="s"></b></div>
  <div class="kv"><span>终点</span><b id="e"></b></div>
  <div class="kv"><span>轨迹点</span><b id="n"></b></div>
  <div class="kv"><span>里程</span><b id="d"></b></div>
  <div class="note" id="roads"></div>
  <div class="note" style="margin-top:10px;color:#888">
    底图为腾讯地图（GCJ-02）。拖动可平移，滚轮可缩放。
  </div>
</div>

<script>
const DATA = __TRAJ_DATA__;
const PORT = __WB_HTTP_PORT__;
const SECRET = __WB_TMAP_SECRET__;

function show(id, v) { document.getElementById(id).textContent = v; }

const geo = DATA.geojson_gcj02 && DATA.geojson_gcj02.features
  ? DATA.geojson_gcj02 : DATA.geojson;
const feats = geo.features || [];
if (!feats.length) {
  document.querySelector('.panel').innerHTML = '<h2>没有可显示的轨迹</h2>';
}

const coords = feats[0] ? feats[0].geometry.coordinates : [];
show('c1', DATA.confidence_a !== undefined ? (DATA.confidence_a * 100).toFixed(0) + '%' : '—');
show('c2', DATA.confidence_b !== undefined ? (DATA.confidence_b * 100).toFixed(0) + '%' : '—');
show('n', DATA.track_points);
show('d', DATA.distance_m > 1000 ? (DATA.distance_m / 1000).toFixed(2) + ' km'
                        : Math.round(DATA.distance_m) + ' m');
if (coords.length) {
  const a = coords[0], b = coords[coords.length - 1];
  show('s', a[0].toFixed(5) + ', ' + a[1].toFixed(5));
  show('e', b[0].toFixed(5) + ', ' + b[1].toFixed(5));
}
if (DATA.roads && DATA.roads.length) {
  show('roads', '经过：' + DATA.roads.slice(0, 12).join('、'));
}

const map = new TMap.Map('map', {
  center: new TMap.LatLng(coords.length ? coords[0][1] : 31.19,
                          coords.length ? coords[0][0] : 121.47),
  zoom: 12,
  pitch: 0,
  rotation: 0
});
map.setServiceHost?.({
  serviceHost: 'http://127.0.0.1:' + PORT + '/_TMapService/_wbt/' + SECRET
});

function draw(pts, color, width, id) {
  if (!pts.length) return null;
  return new TMap.MultiPolyline({
    map,
    styles: { s: new TMap.PolylineStyle({
      color: color, width: width, borderWidth: 2,
      borderColor: '#ffffff', lineCap: 'round', lineJoin: 'round' }) },
    geometries: [{ id: id, styleId: 's',
      paths: pts.map(c => new TMap.LatLng(c[1], c[0])) }]
  });
}

// 轨迹 A：纯 OCR
const pathA = (DATA.coords_a || []).map(c => new TMap.LatLng(c[1], c[0]));
// 轨迹 B：加路名约束
const pathB = (DATA.coords_b || []).map(c => new TMap.LatLng(c[1], c[0]));

draw(pathA, '#d54941', 5, 'trailA');
draw(pathB, '#378add', 5, 'trailB');

function marker(lat, lng, color, label) {
  new TMap.MultiMarker({
    map,
    styles: { m: new TMap.MarkerStyle({
      width: 22, height: 22, anchor: { x: 11, y: 11 },
      color: color, borderWidth: 3, borderColor: '#ffffff' }) },
    geometries: [{ id: label, styleId: 'm', position: new TMap.LatLng(lat, lng) }]
  });
}
if (pathA.length) {
  marker(pathA[0].lat, pathA[0].lng, '#d54941', 'as');
  marker(pathA[pathA.length - 1].lat, pathA[pathA.length - 1].lng, '#a32d2d', 'ae');
}
if (pathB.length) {
  marker(pathB[0].lat, pathB[0].lng, '#378add', 'bs');
  marker(pathB[pathB.length - 1].lat, pathB[pathB.length - 1].lng, '#185fa5', 'be');
}

// 视野对准两条轨迹的并集
try {
  const all = pathA.concat(pathB);
  if (all.length) {
    const b = new TMap.LatLngBounds();
    all.forEach(p => b.extend(p));
    map.fitBounds(b, { padding: 60 });
  }
} catch (e) { console.warn('fitBounds 失败', e); }
</script>
</body>
</html>
"""


def render_traj_map(result: dict[str, Any]) -> str:
    """把识别结果填进地图页模板。"""
    payload = {
        "geojson": result.get("geojson") or {},
        "geojson_gcj02": result.get("geojson_gcj02") or {},
        "confidence_a": result.get("confidence"),
        "confidence_b": result.get("confidence"),
        "coords_a": result.get("coords_a") or [],
        "coords_b": result.get("coords_b") or result.get("coords") or [],
        "track_points": result.get("track_points"),
        "distance_m": result.get("distance_m") or 0,
        "roads": (result.get("road_attribution") or {}).get("roads") or [],
    }
    html = PAGE.replace("__TRAJ_DATA__", json.dumps(payload, ensure_ascii=False))
    return html
