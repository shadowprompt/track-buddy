"""多图拼接的核对页：拼接图 + 原图位置框 + 轨迹，一眼看清对不对。

## 为什么需要这个页面

多图拼接涉及三样东西，用户没法在网页上一屏看全：

1. **拼接图**（大图，可能几千像素宽，网页要缩放才看得全）
2. **每张原图在画布上的位置框**（判断有没有对上图）
3. **识别出的轨迹**（判断轨迹有没有贴住底图）

只给「拼接成功 + 复核分 0.95」这样的数字，用户没法判断。
这个页面把三样东西**按同一坐标系**叠在一张画布上：
位置框明显不对、轨迹明显偏离，一眼就能看出来。

页面还会给出**可下载的对照图**（PNG），方便用户留存或发给别人看。
"""

from __future__ import annotations

import json
from typing import Any

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>多图拼接核对</title>
<style>
html,body{margin:0;height:100%;font-family:system-ui,"Microsoft YaHei",sans-serif;
  background:#f7f7f5;color:#2c2c2a}
.wrap{display:flex;flex-direction:column;height:100%}
.hd{padding:10px 16px;background:#fff;border-bottom:1px solid rgba(0,0,0,.12);
  display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.hd b{font-size:14px;font-weight:500}
.kv{font-size:12px;color:#5f5e5a}
.kv b{color:#2c2c2a;font-weight:500}
.bad{color:#a32d2d}
.good{color:#0f6e56}
.scroll{flex:1;overflow:auto;padding:16px}
.stage{position:relative;display:inline-block;background:#eee;
  border:1px solid rgba(0,0,0,.15)}
.stage img{display:block;max-width:100%}
.stage canvas{position:absolute;left:0;top:0}
.legend{font-size:12px;color:#5f5e5a;padding:8px 0}
.lg{display:inline-block;margin-right:16px}
.lg i{display:inline-block;width:11px;height:11px;margin-right:4px;vertical-align:-1px}
</style>
</head>
<body>
<div class="wrap">
  <div class="hd">
    <b>多图拼接核对</b>
    <span class="kv">拼接图：<b id="mCanvas"></b></span>
    <span class="kv">用上：<b id="mUsed"></b></span>
    <span class="kv">复核分：<b id="mVerify"></b></span>
    <span class="kv">轨迹：<b id="mTrack"></b></span>
    <span class="kv">里程：<b id="mDist"></b></span>
    <button id="dl">下载对照图</button>
    <button id="dlCanvas">下载拼接图</button>
  </div>
  <div class="scroll">
    <div class="legend">
      <span class="lg"><i style="background:#ffc400"></i>原图位置框（应首尾相接、彼此重叠）</span>
      <span class="lg"><i style="background:#00e05a"></i>识别出的轨迹</span>
      <span class="lg"><i style="background:#ffd400"></i>采用的锚点</span>
    </div>
    <div class="stage" id="stage">
      <img id="img">
      <canvas id="cv"></canvas>
    </div>
  </div>
</div>
<script>
const DATA = __STITCH_DATA__;
const img = document.getElementById('img');
const cv  = document.getElementById('cv');
const stage = document.getElementById('stage');

function put(id, v, cls) {
  const el = document.getElementById(id);
  el.textContent = v;
  if (cls) el.className = cls;
}

put('mCanvas', DATA.canvas_size ? DATA.canvas_size[0] + '×' + DATA.canvas_size[1] : '—');
put('mUsed', (DATA.used || []).length + ' 张' +
    (DATA.unused && DATA.unused.length ? '（' + DATA.unused.length + ' 张未用上）' : ''));
const vs = DATA.verify_score;
put('mVerify', (typeof vs === 'number' && vs > 0)
    ? vs.toFixed(4) + (vs >= 0.9 ? '（通过）' : '（可疑）')
    : '—',
    (typeof vs === 'number' && vs > 0)
        ? (vs >= 0.9 ? 'good' : 'bad') : '');
put('mTrack', DATA.track_points ? DATA.track_points + ' 个点' : '—');
put('mDist', DATA.distance_m ? Math.round(DATA.distance_m) + ' 米' : '—');

// 逐张复核分
if ((DATA.verify || []).length) {
  const parts = DATA.verify.map(v =>
    '第' + (v.index + 1) + '张 ' + (v.similarity === null ? '—' : v.similarity));
  put('mVerify', put('mVerify', document.getElementById('mVerify').textContent) +
      '　' + parts.join('　'));
}

function draw() {
  const W = img.clientWidth;
  const natW = img.naturalWidth, natH = img.naturalHeight;
  if (!natW) return;
  const H = natH * (W / natW);
  cv.style.width = W + 'px';
  cv.style.height = H + 'px';
  cv.width = Math.round(W * devicePixelRatio);
  cv.height = Math.round(H * devicePixelRatio);
  const ctx = cv.getContext('2d');
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const sx = W / natW, sy = H / natH;

  // 各原图的位置框
  (DATA.shots || []).forEach(function (s) {
    if (!s.box || s.box.length !== 4) return;
    const x = s.box[0] * sx, y = s.box[1] * sy;
    const w = (s.box[2] - s.box[0]) * sx, h = (s.box[3] - s.box[1]) * sy;
    ctx.strokeStyle = 'rgba(255,196,0,.95)';
    ctx.lineWidth = 2; ctx.setLineDash([7,5]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,196,0,.95)';
    ctx.fillRect(x, Math.max(0, y-18), 26, 18);
    ctx.fillStyle = '#111'; ctx.font = 'bold 12px sans-serif';
    ctx.fillText(String(s.index+1), x+8, Math.max(0, y-18)+13);
  });

  // 轨迹
  const pts = DATA.track_polyline_px || [];
  if (pts.length > 1) {
    ctx.strokeStyle = '#00e05a'; ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.beginPath();
    pts.forEach(function (p, i) {
      const x = p[0]*sx, y = p[1]*sy;
      i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
    });
    ctx.stroke();
  }

  // 锚点
  (DATA.anchors_used || []).forEach(function (a) {
    const x = a.px*sx, y = a.py*sy;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI*2);
    ctx.fillStyle = '#ffd400'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#111'; ctx.font = '11px sans-serif';
    ctx.fillText(a.name, x+9, y+3);
  });
}

img.onload = draw;
if (DATA.stitched_url) {
  img.src = DATA.stitched_url;
} else {
  put('mCanvas', '没有拼接图（拼接失败或只传了一张）', 'bad');
}

// 下载对照图
document.getElementById('dl').onclick = function () {
  const natW = img.naturalWidth, natH = img.naturalHeight;
  if (!natW) { alert('还没有图'); return; }
  const out = document.createElement('canvas');
  out.width = natW; out.height = natH;
  const c = out.getContext('2d');
  c.drawImage(img, 0, 0);
  const sx = natW / img.clientWidth * (img.clientWidth / natW);
  // 直接按原始画布坐标画
  (DATA.shots || []).forEach(function (s) {
    if (!s.box || s.box.length !== 4) return;
    c.strokeStyle = 'rgba(255,196,0,1)'; c.lineWidth = 3; c.setLineDash([10,7]);
    c.strokeRect(s.box[0], s.box[1], s.box[2]-s.box[0], s.box[3]-s.box[1]);
    c.setLineDash([]);
    c.fillStyle = 'rgba(255,196,0,1)';
    c.fillRect(s.box[0], Math.max(0, s.box[1]-24), 32, 24);
    c.fillStyle = '#111'; c.font = 'bold 16px sans-serif';
    c.fillText(String(s.index+1), s.box[0]+8, Math.max(0, s.box[1]-24)+17);
  });
  const pts = DATA.track_polyline_px || [];
  if (pts.length > 1) {
    c.strokeStyle = '#00e05a'; c.lineWidth = 4; c.lineJoin = 'round';
    c.beginPath();
    pts.forEach(function (p, i) { i ? c.lineTo(p[0],p[1]) : c.moveTo(p[0],p[1]); });
    c.stroke();
  }
  (DATA.anchors_used || []).forEach(function (a) {
    c.beginPath(); c.arc(a.px, a.py, 7, 0, Math.PI*2);
    c.fillStyle = '#ffd400'; c.fill();
    c.strokeStyle = '#fff'; c.lineWidth = 2; c.stroke();
  });
  const a = document.createElement('a');
  a.download = 'stitch_check.png';
  a.href = out.toDataURL('image/png');
  a.click();
};

document.getElementById('dlCanvas').onclick = function () {
  if (!DATA.stitched_url) { alert('没有拼接图'); return; }
  const a = document.createElement('a');
  a.download = 'stitched.png';
  a.href = DATA.stitched_url;
  a.click();
};

window.addEventListener('resize', draw);
</script>
</body>
</html>
"""


def render_stitch_check(result: dict[str, Any]) -> str:
    """把多图识别结果填进核对页。"""
    stitch = result.get("stitch") or {}
    payload = {
        "stitched_url": result.get("stitched_url"),
        "canvas_size": stitch.get("canvas_size"),
        "used": stitch.get("used") or [],
        "unused": stitch.get("unused") or [],
        "shots": stitch.get("shots") or [],
        "links": stitch.get("links") or [],
        "verify": stitch.get("verify") or [],
        "verify_score": stitch.get("verify_score"),
        "track_polyline_px": result.get("track_polyline_px") or [],
        "anchors_used": result.get("anchors_used") or [],
        "track_points": result.get("track_points"),
        "distance_m": result.get("distance_m") or 0,
    }
    return PAGE.replace("__STITCH_DATA__", json.dumps(payload, ensure_ascii=False))


def load_stitch_result(shot_id: str | None = None) -> dict[str, Any]:
    """读取最近一次**多图**识别结果。

    单独存一份 `last_stitch.json`——多图结果要被后续的单图识别冲掉，
    而核对页的价值恰恰在于「测完之后还能回看」。
    """
    import json
    from pathlib import Path

    from .config import DATA_DIR

    p = Path(DATA_DIR) / "last_stitch.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}
