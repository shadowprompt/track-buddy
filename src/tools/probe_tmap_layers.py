"""探测腾讯地图 GL JS 的图层 API 正确用法。

要回答三个问题：
1. MultiPolyline / MultiMarker 有没有 id 属性？值是什么？
2. map.removeLayer(正确的参数) 能不能真的删掉图层？
3. 复用同一个图层 + setGeometries 能不能更新轨迹（比重建更稳）？
"""

import sys

from playwright.sync_api import sync_playwright

JS = r"""
async () => {
  const out = {};
  // 在页面上新建一个独立的地图容器来测，不干扰主地图
  const holder = document.createElement('div');
  holder.id = '__probe_map';
  holder.style.cssText = 'position:fixed;left:0;top:0;width:400px;height:300px;z-index:9999';
  document.body.appendChild(holder);

  const m = new TMap.Map('__probe_map', { center: new TMap.LatLng(31.22, 121.55), zoom: 12 });

  const p1 = new TMap.MultiPolyline({
    map: m,
    styles: { t: new TMap.PolylineStyle({ color: '#f00', width: 6 }) },
    geometries: [{ id: 'trail', styleId: 't', paths: [
      new TMap.LatLng(31.220, 121.550), new TMap.LatLng(31.225, 121.555)] }],
  });
  const mk = new TMap.MultiMarker({
    map: m,
    styles: { s: new TMap.MarkerStyle({ width: 20, height: 20 }) },
    geometries: [{ id: 's', styleId: 's', position: new TMap.LatLng(31.220, 121.550) }],
  });

  out.polylineIdType = typeof p1.id;
  out.polylineId = String(p1.id);
  out.markerId = String(mk.id);
  out.hasSetGeometries = typeof p1.setGeometries === 'function';
  out.hasUpdateGeometries = typeof p1.updateGeometries === 'function';
  out.markerHasSetGeometries = typeof mk.setGeometries === 'function';

  // 试错法一：传对象（当前代码的做法）
  try { m.removeLayer(p1); out.removeWithObject = 'OK'; }
  catch (e) { out.removeWithObject = 'THROW: ' + e.message; }

  // 试错法二：传 id 字符串
  try { m.removeLayer(p1.id); out.removeWithId = 'OK'; }
  catch (e) { out.removeWithId = 'THROW: ' + e.message; }

  // 试错法三：传我们自己的 geometry id（'trail'）
  try { m.removeLayer('trail'); out.removeWithGeomId = 'OK'; }
  catch (e) { out.removeWithGeomId = 'THROW: ' + e.message; }

  // 复用图层：setGeometries 更新路径
  try {
    const p2 = new TMap.MultiPolyline({
      map: m,
      styles: { t: new TMap.PolylineStyle({ color: '#0f0', width: 6 }) },
      geometries: [{ id: 'trail2', styleId: 't', paths: [
        new TMap.LatLng(31.200, 121.500), new TMap.LatLng(31.205, 121.505)] }],
    });
    out.newLayerId = String(p2.id);
    p2.setGeometries([{ id: 'trail2', styleId: 't', paths: [
      new TMap.LatLng(31.300, 121.600), new TMap.LatLng(31.310, 121.610)] }]);
    const g = p2.getGeometries();
    out.afterSetGeometries = {
      count: g.length,
      head: g[0].paths ? [g[0].paths[0].lat, g[0].paths[0].lng] : null,
    };
  } catch (e) {
    out.setGeometriesErr = 'THROW: ' + e.message;
  }

  // 复用 marker：setGeometries 更新位置
  try {
    const mk2 = new TMap.MultiMarker({
      map: m,
      styles: { s: new TMap.MarkerStyle({ width: 20, height: 20 }) },
      geometries: [{ id: 's2', styleId: 's', position: new TMap.LatLng(31.220, 121.550) }],
    });
    mk2.setGeometries([{ id: 's2', styleId: 's', position: new TMap.LatLng(31.400, 121.700) }]);
    const g = mk2.getGeometries();
    out.afterMarkerSet = [g[0].position.lat, g[0].position.lng];
  } catch (e) {
    out.markerSetErr = 'THROW: ' + e.message;
  }

  // getLayers 是否可用
  try {
    out.getLayersLen = m.getLayers().length;
  } catch (e) { out.getLayersErr = 'THROW: ' + e.message; }

  holder.remove();
  return out;
}
"""


def main() -> int:
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    import os

    launch_kw = {"channel": "msedge"} if os.path.exists(edge) else {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch_kw)
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto("http://127.0.0.1:8848", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)
        res = page.evaluate(JS)
        for k, v in res.items():
            print(f"  {k}: {v}")
        if errs:
            print("JS 异常:", errs[:3])
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
