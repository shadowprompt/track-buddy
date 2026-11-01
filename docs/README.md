# 文档与示例产物

这个目录存放识别结果的可视化产物。

## 为什么不入库

默认 `.gitignore` 排除了 `docs/*.png` 与 `docs/*.geojson`。这些图和轨迹文件
包含**真实骑行路线的位置信息**（起终点、途经街道），公开仓库等于公开位置历史。

`.gitignore` 里对 `samples/*.jpg`（原始截图）和根目录的 `trajectory*.geojson`
做了同样处理。

## 产出什么

跑一次识别会生成：

| 文件 | 内容 |
|---|---|
| `overlay_result.png` | 原截图 + 识别轨迹叠加（橙线=轨迹，白点=采样点，绿=起点，红=终点，青圈=POI 锚点） |
| `road_verify.png` | OSM 道路几何投影回截图 + 白色轨迹，用于验证绝对位置 |
| `trajectory_chart.png` | 独立轨迹图（米制等比例，标注路名与公里刻度） |
| `example_output.geojson` | GeoJSON 示例（轨迹主体 + 起点/终点） |

## 怎么生成

```bash
python -m app.cli samples/<你的截图>.jpg --out docs/
```

或启动网页服务后在浏览器里识别：

```bash
python run.py --port 8848     # 打开 http://127.0.0.1:8848
```

## 读图须知

- `overlay_result.png` 里 **橙色粗线是识别结果**，白点是采样点，绿色圆点=起点，
  红色圆点=终点，青色圆圈=参与配准的 POI 锚点。
  左下角若出现**红色虚线**，那是 App 自己画的规划路线，程序会正确排除它。
- `road_verify.png` 里 **白色粗线是识别轨迹**，彩色细线是 OSM 的道路几何。
  两者配色刻意区分——早期版本轨迹和「剑河路」都用黄色，导致无法分辨。
- `trajectory_chart.png` 里深绿线是轨迹，米色标签是自动归属的路名。
  路名来自 OSM，可能与 App 上的官方命名不一致。
