# 项目设计与交接文档

> **给接手者的第一句话**：这不是一个「图像识别」项目，是一个**图像配准**项目。
> GPS 坐标不在图片里。必须用图上印着的 POI 标注当锚点，把像素坐标映射到经纬度，
> 再用真实路网约束校正。理解这一点，其余所有设计选择都是它的推论。

本文档面向**接手继续开发的 AI / 工程师**。事实核对日期：2026-11-02，
对应 commit `369afd8`。代码唯一真源仓库：`D:\Dev\meituan-ride`
（GitHub: `https://github.com/shadowprompt/track-buddy`，public，MIT）。

---

## 目录

1. [项目是什么](#1-项目是什么)
2. [五分钟跑起来](#2-五分钟跑起来)
3. [整体架构](#3-整体架构)
4. [坐标系：全项目最重要的一节](#4-坐标系全项目最重要的一节)
5. [五个阶段的算法细节](#5-五个阶段的算法细节)
6. [模块职责速查表](#6-模块职责速查表)
7. [HTTP API 契约](#7-http-api-契约)
8. [命令行用法](#8-命令行用法)
9. [前端结构](#9-前端结构)
10. [关键参数表](#10-关键参数表)
11. [已知坑与排障手册](#11-已知坑与排障手册)
12. [测试体系](#12-测试体系)
13. [数据飞轮：本地词典](#13-数据飞轮本地词典)
14. [合规边界](#14-合规边界)
15. [WorkBuddy 技能包](#15-workbuddy-技能包)
16. [代码地图：改哪里](#16-代码地图改哪里)
17. [后续可做事项](#17-后续可做事项)
18. [决策记录](#18-决策记录)
19. [城市级低缩放截图的定位](#19-城市级低缩放截图的定位2026-11-新增)

---

## 1. 项目是什么

### 1.1 功能

输入一张骑行 App（美团骑行 / Keep / Strava 等）的**地图截图**，
输出一条带经纬度的 GPS 轨迹（GeoJSON），并给出：

- 轨迹在腾讯地图底图上的位置（网页可视化）
- 轨迹叠在原截图上的对照图（验证轨迹有没有抠错）
- 每一段轨迹经过的道路名
- 置信度与完整诊断信息

### 1.2 为什么难

| 难点 | 说明 |
|---|---|
| 坐标不在图里 | 截图只有像素，没有 GPS。要靠图上印的地名反推坐标 |
| 底图不统一 | 美团用高德系数据，OSM 用 WGS-84，两者同一 POI 可差 100–300 米 |
| 标注有偏移 | POI 文字中心 ≠ 真实点位，差 7–28 像素（35–150 米），排版决定，无法消除 |
| 截图尺度未知 | 同一张图可能是 0.22 m/px 也可能是 4.6 m/px，尺度必须估出来 |
| 深色/浅色底图 | 轨迹线颜色随主题变化，掩膜阈值要能自适应 |

### 1.3 不做什么

- **不做** 真实骑行 App 的逆向、接口破解
- **不做** 大模型推理（纯 CV + 几何 + 地理编码，无 LLM 参与）
- **不承诺** 亚米级精度。当前实测贴路中位距离 2–6 米，主要误差来自锚点基准

---

## 2. 五分钟跑起来

### 2.1 环境

```bash
# Python 3.13（开发版本；3.11+ 应该都能跑）
pip install -r requirements.txt
```

可选（仅用于真实浏览器端到端验证）：

```bash
pip install playwright && python -m playwright install chromium
```

### 2.2 三种用法

```bash
# ① 网页服务（主要用法）
python -m app.server --host 127.0.0.1 --port 8848
# 打开 http://127.0.0.1:8848

# ② 命令行
python -m app.cli samples/meituan_ride_shanghai.jpg --city 上海 --out out/
# 产出：trajectory_gcj02.geojson / trajectory_wgs84.geojson
#       overlay_result.png / trajectory_chart.png / road_verify.png
# stdout 输出一段 JSON 摘要

# ③ 作为 Python 库
from app.config import CONFIG
from app.pipeline import RidePipeline
from app.cv_utils import imread_unicode

res = RidePipeline(CONFIG).run(imread_unicode("shot.jpg"), city="上海")
print(res.confidence, res.track_points, res.distance_m)
print(res.geojson_gcj02)   # 国内底图用这个
print(res.geojson)         # GPS 设备 / Strava / GIS 用这个
```

### 2.3 地图密钥

| 密钥 | 用途 | 不配会怎样 |
|---|---|---|
| `AMAP_KEY` | 地理编码 + 路名归属（与美团底图同源，**精度最高**） | 退到词典 + OSM 冷启动，绝对精度受限 100–300m |
| `TMAP_KEY` | 地理编码 + 网页底图 | 底图回退到腾讯官方公开示例 key（仅本地代理内部使用） |
| `BAIDU_MAP_KEY` / `TIANDITU_KEY` | 地理编码备选 | 同上 |

配置方式：环境变量，或项目根目录 `.env` 文件。

```bash
AMAP_KEY=你的key
TMAP_KEY=你的key
```

**注意**：`tmap_gl_key`（底图用）与 `tmap_key`（地理编码用）是两回事。
底图可以用官方公开示例 key，地理编码必须自己申请，否则没配额。

---

## 3. 整体架构

### 3.1 数据流

```
截图 (BGR ndarray)
  │
  ├─① 轨迹提取  app/track.py
  │     色彩分割 → 形态学 → 连通域 → 骨架化 → 图路径追踪
  │     输出：像素折线 + 起终点徽章 + 掩膜 + 提取置信度
  │
  ├─② OCR       app/ocr_anchor.py
  │     RapidOCR（支持竖排/旋转）→ 文本分级 → 锚点候选
  │     输出：Anchor 列表（kind = icon / poi / road）
  │
  ├─③ 地理编码  app/geocode.py + app/coldstart.py
  │     优先级：用户确认 > 在线服务(高德/腾讯/百度/天地图)
  │            > 本地词典 > OSM 冷启动
  │     输出：AnchorPair 列表（像素 ↔ 经纬度，多候选全交 RANSAC）
  │
  ├─④ 配准      app/register.py
  │     RANSAC + 模型优先级 north_up → similarity → affine
  │     输出：像素→墨卡托变换矩阵 + 内点/离群 + RMSE
  │
  ├─④.5 绝对位置自校验  pipeline._self_check
  │     把锚点反投影回像素，检查与原锚点位置重合
  │
  ├─④.6 路网约束精修  app/roadfit.py
  │     拉真实路网 → 在保持尺度朝向前提下平移轨迹 → 贴路距离最小
  │     输出：平移量 + 精修前后贴路中位距离
  │
  ├─⑤ 折线投影  pipeline
  │     采样(5m) → 简化 → 像素→墨卡托→GCJ-02 → 按需转 WGS-84
  │     输出：GeoJSON（两份：GCJ-02 与 WGS-84）
  │
  └─⑥ 路名归属  app/roadname.py
        优先：路网几何判定 > 高德路径规划 > 截图自带标注
```

### 3.2 三层架构

```
┌─────────────────────────────────────────┐
│  app/server.py    FastAPI HTTP 服务       │  ← 网页前端
│  app/cli.py       命令行                 │  ← 脚本/批处理
├─────────────────────────────────────────┤
│  app/pipeline.py  RidePipeline 编排       │  ← 唯一的业务真相
│  （其余模块都是纯函数/无状态类）           │
├─────────────────────────────────────────┤
│  app/track.py     app/ocr_anchor.py       │
│  app/geocode.py   app/coldstart.py       │
│  app/register.py  app/roadfit.py         │
│  app/roadname.py  app/geo/coords.py      │
│  app/cv_utils.py  app/visuals.py         │
└─────────────────────────────────────────┘
```

**设计约束**：除 `pipeline.py` 外，所有模块都不应直接依赖彼此，
只依赖 `config.py` 的参数与 `geo/coords.py` 的坐标转换。
这样每个模块都能单独测试。

---

## 4. 坐标系：全项目最重要的一节

> **接手后第一件要确认的事**：你改的代码里，每个经纬度的基准是什么。

### 4.1 三个基准

| 基准 | 用在哪 | 说明 |
|---|---|---|
| **WGS-84** | GPS 设备、Strava、Garmin、GPX、GeoJSON 通用标准、**OSM API** | 国际标准 |
| **GCJ-02** | 高德/腾讯/百度/天地图（**含美团底图**） | 中国境内加密坐标 |
| **Web 墨卡托 (EPSG:3857)** | 配准的**中间空间** | 像素↔地理做线性变换用 |

中国境内 GCJ-02 相对 WGS-84 有 **100–700 米**系统性偏移（上海约东 450m / 南 200m）。

### 4.2 项目内部约定

**内部统一用 GCJ-02 工作**（与截图底图同源），输出时按需转换。
词典 `data/lexicon.json` 内部**一律存 GCJ-02**。

### 4.3 转换函数（`app/geo/coords.py`）

```python
wgs84_to_gcj02(lon, lat) -> (lon, lat)   # 一步
gcj02_to_wgs84(lon, lat) -> (lon, lat)   # 不动点迭代，8 次内收敛到亚米
gcj02_to_bd09 / bd09_to_gcj02            # 备用
lonlat_to_mercator / mercator_to_lonlat  # 配准中间空间
haversine(lon1, lat1, lon2, lat2)        # 球面距离（米），算里程用这个
```

### 4.4 ⚠️ 三个曾经咬人的坑

**坑 1：`_delta` 返回顺序**

```python
# 正确
dlat = _transform_lat(lon - 105.0, lat - 35.0)
dlon = _transform_lon(lon - 105.0, lat - 35.0)
```

早期版本两个方向都用了 `_transform_lat`，**经度偏差约 375 米**。
现在 `tests/test_e2e.py` 里有断言守着（天安门基准误差 0m）。

**坑 2：GCJ→WGS 的一步近似**

GCJ-02 的偏移量依赖于**原始 WGS-84 坐标**，但我们手上只有 GCJ-02，
这是个隐式方程。早期用「以 GCJ 为基准算偏移再减去」的一步近似，
往返会把点平移约 **347 米**，且里程凭空缩短 15%。
现在用不动点迭代（实测 4 次即收敛到亚米级）。

**坑 3：给 OSM API 喂了 GCJ-02**

`reg.pixel_to_lonlat()` 返回 **GCJ-02**，而 OSM API 的 bbox 期望 **WGS-84**。
直接传会让查询中心偏移约 450m（东）/200m（南），**拉回来的是别的街区的路网**。
`pipeline._road_refine()` 里必须先 `gcj02_to_wgs84` 再调 `fetch_roads`。

### 4.5 尺度换算

```python
地面距离 = 墨卡托距离 × cos(纬度)
```

`reg.scale` 是**墨卡托米/像素**。`diagnostics` 里两个字段都给了：

- `scale_mercator_m_per_px` — 原始值
- `scale_ground_m_per_px` — 已乘 cos(纬度)，是真实地面尺度

**算里程要用 `haversine`，不要用 `像素长 × scale` 估算**（会差 15%）。

---

## 5. 五个阶段的算法细节

### 5.1 ① 轨迹提取（`app/track.py`）

```python
extract_track(img, cfg) -> TrackResult
```

1. **起终点徽章检测** — HSV 阈值找绿色「起」和红色「终」，面积 1500–20000 px
2. **轨迹掩膜** — 低饱和 + 中等亮度（`track_max_value=200`, `track_max_saturation=90`），
   闭运算 11 跨断点，开运算 5 滤掉文字笔画（文字笔画宽 2–3px，轨迹线宽 8–10px）
3. **连通域筛选** — `_select_component` 用**像素级拓扑测距**选含起终点的那个。
   ⚠️ 早期用 `aspect_ratio > 1.6` 启发式，只适合横向路线，对角线样本直接失败。
4. **骨架化** — Zhang-Suen 细化，得到单像素中心线
5. **图路径追踪** — 8 邻接图 + Dijkstra，从起点走到终点
   ⚠️ 早期按投影排序（x 或 y），导致折线锯齿、里程虚增

**置信度**：起终点都在 0.9；缺一个 0.55；掩膜放宽时 0.35；
连通域与起终点距离 > 200px 时 ×0.7。

### 5.2 ② OCR 与锚点分级（`app/ocr_anchor.py`）

```python
ocr_texts(img, cfg) -> list[TextItem]     # RapidOCR，识别旋转/竖排
build_anchors(img, texts, cfg) -> list[Anchor]
```

OCR 侧的锚点分级与权重：

| kind | 权重 | 能否参与配准 |
|---|---|---|
| `icon` | 1.0 | ✅ POI 图标点，位置最准 |
| `poi` | 0.85 | ✅ 地标文字 |
| `road` | 0.5 | ❌ **仅用于区域校验** |

⚠️ **道路名不能作锚点**。路名文字中心可偏 2–5 公里，会让整个配准落到平行的
错误道路族上。实测用户实骑「新渔路/新渔东路/剑河路/金钟路」，
而 OSM 归属显示「天山路/北翟路」——就是路名污染导致的。

**地理编码侧会再分一次类**（`app/geocode.py` 的 `infer_kind`），
这是配准内点容差的依据：

| kind | 判定依据 | 标注偏移量级 |
|---|---|---|
| `district` | OSM `boundary=administrative`，或名称以「区/县/市/旗」结尾 | 28–93 像素（标签被挪到视野空白处） |
| `station` | OSM `train_station`/`subway_station` 等，或名称以「站/港/场」结尾 | 30–60 像素 |
| `transit` | 名称含「号线/地铁/联络线/环线/内圈/外圈」 | 弱参考 |
| `road` | 名称含「高速/高架/路/道/街/桥/隧道/匝道」 | 弱参考 |
| `poi` | 其余 | 30–62 像素 |

OSM 分类优先，名称形态兜底。**分类结果直接决定内点容差**，
详见 5.4.3。

### 5.3 ③ 地理编码（`app/geocode.py` + `app/coldstart.py`）

⚠️ **跨城同名地点必须过滤**。Nominatim 的 `viewbox` 只是**排序提示**，
不做强制过滤，同名异地地点照样返回。实测城市级截图里「3号线」匹配到
哈尔滨（126.60, 45.76）、「8号线」匹配到重庆、「1号线」匹配到成都——
这些点离真实轨迹上千公里，一旦被误当作内点，配准会整个跑偏。
`app/coldstart.py` 的 `_in_city()` 按城市 bbox（含 0.25° 余量）硬性裁剪，
且**在磁盘缓存命中之后也要执行**（缓存里可能存着过滤规则上线前写入的旧数据）。


**优先级（顺序不能改）**：

```
1. 用户确认过的坐标（source="user"）    ← 永远第一
2. 在线服务：高德 → 腾讯 → 百度 → 天地图   ← 与美团底图同源
3. 本地词典 data/lexicon.json             ← 离线可用
4. OSM 冷启动（app/coldstart.py）          ← 最后兜底
```

⚠️ 早期是「词典优先」，只要词典命中就永不调用在线服务。
**这是错的**：词典种子来自 OSM/Nominatim，而美团底图用国内官方 POI 数据，
同一 POI 两者可差 100–300 米。实测长宁新泾一例因此整体偏北约 245 米。

#### 冷启动的三阶段（`app/coldstart.py`）

换区域时词典没覆盖，必须靠 OSM 兜底。三个阶段缺一不可：

**阶段一 · 多查询变体** — Nominatim 索引对前缀敏感：

```
进才实验中学 上海       → 0 条
上海市进才实验中学      → 1 条 ✓
```

所以每个地名试 4 种写法：`{名} {市}`、`{市}{名}`、`{短名}{名}`、`{名}`。

**阶段二 · 名称相似度** — OCR 会截断地名，不能要求完全同名。
用 Jaccard 字符集 + 子串加权，阈值 **0.55**：

```
OCR「水清木华」 → OSM「水清木华公寓」      0.92 ✓
OCR「乡土田园区」→ OSM「何文田上海水配水庫」 0.08 ✗
```

⚠️ **放宽匹配不是答案**。放宽到 0.3 以下会把一堆无关地点放进来
（「乡土田园区」会匹配到深圳、湖南的同名地点）。必须同时有
**更严的约束**（相似度阈值 + 距离约束 + 互证）配套。

**阶段三 · 多锚点互证** — 单个地名无法自证位置：

```
「花园2期」能匹配到朗香/星宇/尼德兰/三湘盛世四个不同小区的「2期」
```

但多个**独立检索**的地名若彼此都落在 1 公里内，那片区域就是真的。
实测浦东：华丽家族花园 / 天安花园 / 当代清水园 / 水清木华公寓
四者互证，锁定簇心 `(121.5537, 31.2282)`，据此排除 164km 外的异地同名点。

> **因此冷启动必须批量调用**（`lookup_many()`），不能逐个 `geocode()`。
> `pipeline._resolve_anchors()` 里先把词典/在线能命中的处理掉，
> 剩下的 `pending` 一次性交给 `lookup_many()`。

互证阈值取「≥1 个**其他**地名在 800m 内」——要求 ≥2 个反而会让
只有两个地名能查到的区域定位失败。

判定代码里注意一个细节：

```python
corroborated = distinct | {ocr}   # 印证集合包含簇心自身的地名
```

簇心自身的地名也算印证的一环。这决定了簇心会落在**哪一个**候选点上——
若把自身排除，簇心会偏向「周围有最多邻居」的那个点，
而不是「被最多地名围绕」的那个点。

`_MAX_CENTER_DIST_M = 1500`：簇心 1.5 公里内的候选才保留。

⚠️ **viewbox 不是过滤器，只是排序提示**。对「上海市」这种大范围 bbox
几乎不起作用，同名异地地点照样返回。真正的过滤必须自己实现。

限速：Nominatim 使用政策要求最多 1 次/秒。命中 5 个锚点即停止继续查询。

### 5.4 ④ 配准（`app/register.py`）

```python
register(pairs, inlier_m=200.0, min_inliers=3) -> Registration
```

⚠️ **像素 Y 向下，墨卡托 Y 向上**。用 Umeyama 直接拟合会得到反射矩阵。
现在用显式矩阵形式处理。

#### 5.4.1 模型：按自由度分层降级

| 模型 | 自由度 | 采样点数 | 所需内点数 | 用途 |
|---|---|---|---|---|
| `north_up` | 3（尺度 + 平移 x/y） | 3 | 3 | 默认地图状态（正北朝上），精度最高 |
| `similarity` | 4 | 3 | 4 | 用户旋转过截图 |
| `affine` | 6 | **4** | 5 | 兜底 |
| `north_up_2pt` | 3 | 2 | 2 | **最后兜底**：可靠锚点只剩 2 个时 |

两条硬规则：

1. **采样点数必须 ≥ 模型自由度**。`north_up` 有 3 个自由度，用 2 点采样时
   尺度由两点距离比定出、平移被迫取均值，必然留下系统性偏差
   （实测在城市级截图上直接选出了偏差 10% 的错误尺度）。
   `affine` 有 6 自由度，用 3 点采样是**过参数化**——任何 3 个点都能被
   拟合成一个「完美」的仿射，于是模型能凭一两个错误锚点凑出更多内点，
   压过物理上正确的 `north_up`（实测旋转角被伪造成 -1.67°）。
2. **低自由度模型优先**。美团地图截图默认正北朝上无旋转，`north_up`
   是物理真值；高自由度模型拟合出的轻微旋转与被拉大的残差
   都是过参数化的产物，不是真实信息。`north_up` 成立时不再用高自由度
   模型覆盖它。

3. **2 点兜底路径**。可靠锚点只剩 2 个时（实测有样本只匹配到 3 个 POI
   锚点、其中 2 个自洽），3 点模型不成立，但两个一致锚点仍足以定出
   可用模型——后面还有路网精修兜底。⚠️ 此时**无法做留一验证**
   （留出 1 个只剩 1 个，拟合不出模型），评分退化为
   「内点锚点数 + 残差中位数」。

#### 5.4.2 锚点分组：同名候选不得刷票

Nominatim 对同一个地名常返回多个同名 POI（实测「外环高速」5 个、
「上海南站」5 个、「机场联络线」3 个），它们**屏幕像素完全相同**、
地理坐标彼此相差几十到几百米。

若把它们当成独立证据计入内点数，一个由两三个近距点撑起来的错误模型
就能「刷票」压过真正的解——这正是城市级截图配准到错误城区的直接原因。

处理：按屏幕像素位置分组，组内地理距离 < 200 米的候选合并，
每个锚点在评分时只保留残差最小的那一个代表。

#### 5.4.3 内点容差：按像素，不按米

**这是尺度自适应的关键**。地图标注的偏移是「渲染层面」的——标签避让
会把文字挪开若干像素，这个偏移量与截图缩放级别**无关**。

实测（城市级截图，15.2 米/像素）：

| 锚点 | 残差（米） | 残差（像素） |
|---|---|---|
| 上海站 | 495 | 32.6 |
| 上海西站 | 640 | 42.1 |
| 桃浦中央绿地 | 938 | 61.8 |
| 长宁区 | 436 | 28.7 |
| 静安区 | 1417 | 93.4 |

若用固定米数阈值：200 米只够 13 像素，可靠 POI 全被判成离群；
2500 米在街道级又太松，混进的离群点会污染模型。

所以容差 = `max(米制下限, 像素上限 × 当前模型尺度)`：

| kind | 像素上限 | 米制下限 |
|---|---|---|
| `station` / `poi` / `icon` | 60 px | 200 m |
| `district` | 110 px | 800 m |
| `road` / `transit` | 150 px | 800 m |

行政区上限更大：地图引擎会把「静安区」三个字挪到视野空白处，
与行政区的几何质心可差 1–2 公里。

#### 5.4.4 尺度投票：只用可靠锚点

可靠锚点（POI / 车站）由图标锚定真实点位，标注偏移小；
行政区锚点的偏移方向随机，混进采样会把尺度带偏几个百分点
（实测正确尺度 15.73 墨卡托米/像素，混入行政区后被拉成 14.5–14.7）。

做法：只用可靠锚点两两算「地理距离 / 像素距离」，取**一致性最高的簇**
（众数）的中位数。四处修正缺一不可：

1. 每个锚点对只贡献一个比值（当前代表之间的比值）——同名候选再多也不刷票；
2. **代表是迭代选出来的**——用当前尺度把每组里残差最小的候选选作代表，
   再重算比值。（固定用「组内中位数」会让组内第二候选把比值拉低 3.7%，
   这个误差在远距离锚点上会被放大成 4 公里的残差。）
3. **以「锚点对的自洽性」筛锚点**——只有参与了一致性最高那簇锚点对的
   锚点才可信。用残差筛是循环的：平移一旦被污染，坏锚点的某个候选
   反而会「达标」。
4. 迭代剔除后重新投票。

实测城市级截图：「上海南站」（标注偏 4 公里）、「外环高速」
（候选横跨 15 公里）都被剔除，尺度收敛到 **15.69**（真值 15.73）。

#### 5.4.5 留一交叉验证

模型可以通过「一个偏得离谱的锚点 + 一堆碰巧一致的锚点」刷出更多内点，
而这类点**只靠自己撑着**——拿掉它，其余点拟合不出同样的模型。

留一验证：每个锚点单独拿掉，用其余锚点重新拟合再预测它，
统计有多少锚点真的互相支撑。实测城市级样本：「每万佛讲寺」的标注位置
与其它锚点差了一个尺度，内点数看着最多（9 个），但留一后它自己撑不住，
真正的自洽锚点只有 8 个——去掉它反而得到更准的模型。

⚠️ 留一验证必须**固定尺度**（只重估平移）。尺度已经由投票确定，
验证的就应该是同一个模型；若在留一时重新放开尺度，验证的就不是它了。

#### 5.4.6 判定顺序

最终模型 = 尺度投票路径 与 枚举路径 两者中**留一分数更高**者。
两条路径的评分口径必须一致（都是「留一自洽锚点数」+「留一残差中位数」），
否则无法比较。

### 5.5 ④.6 路网约束精修（`app/roadfit.py`）

```python
fetch_roads(bbox, timeout, cache_path) -> list[(路名, [(lon, lat), ...])]
refine_with_roads(reg, polyline_px, roads) -> RoadFit
```

**为什么需要**：锚点是 POI 文字标注，标注位置与真实 POI 存在系统性偏差
（不同 POI 数据库之间 100–300 米），会让整条轨迹落到**平行的错误道路族**上。

**为什么可靠**：道路几何不存在这种问题——路网位置在所有地图上必须一致。

做法：在**保持尺度与朝向**的前提下平移轨迹，使贴路距离最小。
输出平移量 `shift_e_m` / `shift_n_m`，并入配准矩阵。

**缓存格式**（多块）：

```json
{"blocks": [{"bbox": [w, s, e, n], "roads": [[name, [[lon,lat],...]], ...]}]}
```

⚠️ **复用条件是「完全包含」而非「重叠」**。
重叠复用会把区域外的道路混进结果——这正是早期踩过的
「缓存里存着偏移 4.4 公里的路网」那个坑。实现时先写成重叠，
实测引入 **52 条越界道路**才发现。

**Overpass 拉路网要 20–40 秒**（浦东一带 41 秒），是识别慢的主因。
缓存命中后 0 秒。Overpass 公共实例不稳定（频繁 502/超时），
失败时退回 OSM API，再失败就用缓存凑合——**路网只是精修用的**，
没有它也有锚点配准的结果，不该让整个识别崩掉。

### 5.6 ⑤ 折线采样与简化

```python
meters_per_px = max(reg.scale, 1e-6)
step_px = min(max(1.0, 5.0 / meters_per_px), 4.0)   # 目标每点约 5 米
eps_px  = min(1.0 / meters_per_px, 1.0)             # 容差
```

⚠️ **两个上限都必须有**。高倍截图（scale=0.26 m/px）时：
步长会变成 38 像素、容差 7.7 像素，1990 像素的轨迹被削到只剩 **6 个点**。
步长上限 4 像素、容差上限 1 像素。

⚠️ **Douglas-Peucker 在近直线上失效**。经典 DP 用「点到端点连线的垂距」
作误差判据，而一条南北向长直线上任何微小横向抖动都会被这段长直线
放大成很大的垂距。实测轨迹（总长 2043px、端到端直线 1954px，仅 4.5% 蛇形）
简化后只剩 32 点，弯道全丢。

现在 `cv_utils.simplify_polyline()` 用**双判据**：

1. **弧长偏差**（主判据）— 折线比弦长多走了多少路，与形状无关，量纲是像素
2. **垂距**（辅助定位分裂点）— 捕捉整体横向平移

两个都超容差才算有实质弯曲。

### 5.7 ⑥ 路名归属（`app/roadname.py`）

**只做命名，不改坐标**。来源优先级：

1. **路网几何判定**（`source="road_network"`）— 最可靠，直接几何判定
2. **高德路径规划 API** — 与美团底图同源，官方路名
3. **截图自带标注** — 地图自己印的路名，命名与用户看到的一致

⚠️ OSM 等第三方源常保留旧名（上海的路网 2015 年后就少有更新）。

---

## 6. 模块职责速查表

| 文件 | 行数 | 职责 | 关键入口 |
|---|---|---|---|
| `app/config.py` | 152 | 全局参数、密钥读取 | `CONFIG`, `VisionConfig` |
| `app/pipeline.py` | 717 | 流水线编排（唯一业务真相） | `RidePipeline.run()` |
| `app/track.py` | 445 | 轨迹提取 | `extract_track()` |
| `app/ocr_anchor.py` | 329 | OCR + 锚点分级 | `ocr_texts()`, `build_anchors()` |
| `app/geocode.py` | 443 | 词典 + 在线地理编码 | `Geocoder.geocode()`, `Lexicon` |
| `app/coldstart.py` | 478 | OSM 冷启动三阶段 | `get_cold_start().lookup_many()` |
| `app/register.py` | 317 | RANSAC 配准 | `register()`, `apply_to_polyline()` |
| `app/roadfit.py` | 385 | 路网获取 + 精修 + 归属 | `fetch_roads()`, `refine_with_roads()` |
| `app/roadname.py` | 322 | 路名归属 | `attribute_by_amap()`, `attribute_by_shot_labels()` |
| `app/geo/coords.py` | 153 | 坐标转换与投影 | `wgs84_to_gcj02()`, `gcj02_to_wgs84()` |
| `app/cv_utils.py` | 200 | 图像 IO、几何工具 | `imread_unicode()`, `simplify_polyline()` |
| `app/visuals.py` | 382 | 三类可视化产出 | `build_overlay()`, `build_road_verify()`, `build_chart()` |
| `app/server.py` | 250 | FastAPI 服务 | `app` (ASGI) |
| `app/tmap_proxy.py` | 155 | 腾讯地图 GL JS 本地代理 | `create_proxy_router()` |
| `app/cli.py` | 117 | 命令行入口 | `main()` |
| `app/seed_lexicon.py` | 133 | 词典种子生成 | — |
| `app/snap.py` | 207 | ⚠️ **死代码**，无任何引用 | — |

> **`app/snap.py` 是历史遗留**：里面的 `RoadNetwork` 与 `roadfit.py` 同名但实现不同
> （走高德/腾讯路径规划而非 OSM 几何），且 `snap_to_roads()` 从未被调用。
> 可以删，但删之前确认没有外部脚本依赖。

---

## 7. HTTP API 契约

### `GET /api/health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "geocoding": {"amap": false, "tmap": false, "baidu": false,
                "tianditu": false, "online_available": false},
  "lexicon": {"entries": 43, "points": 45, "cities": ["上海"]}
}
```

### `POST /api/recognize`

multipart/form-data：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `file` | file | 必填 | 截图，≤20MB，jpg/png/webp |
| `city` | text | `""` | 城市提示，冷启动靠它限定检索范围 |
| `output_crs` | text | `"WGS-84"` | `WGS-84` 或 `GCJ-02` |
| `allow_online` | bool | `true` | 是否允许联网地理编码 |

响应体（关键字段）：

```jsonc
{
  "ok": true,
  "shot_id": "1793581569_148a669c",
  "shot_url": "/api/shots/1793581569_148a669c.jpg",
  "geojson": { /* WGS-84 */ },
  "geojson_gcj02": { /* GCJ-02，国内底图用这个 */ },
  "confidence": 0.879,
  "crs": "WGS-84",
  "track_points": 23,
  "distance_m": 840.0,
  "anchors_used":     [{"name": "...", "px": ..., "py": ..., "kind": "poi", ...}],
  "anchors_rejected": [...],
  "road_attribution": {"segments": [...], "roads": [...], "source": "road_network"},
  "road_fit": {"ok": true, "shift_e_m": 600, "shift_n_m": -230,
               "median_before_m": 14.0, "median_after_m": 3.0},
  "diagnostics": { /* 固定 16 个键，缺失为 null */ },
  "warnings": ["..."],
  "elapsed_ms": 14629,
  "track_polyline_px": [[x, y], ...],
  "image_size": [1260, 2720]
}
```

⚠️ **`diagnostics` 的键集是固定的**（`pipeline.DIAG_KEYS`）。
所有失败路径也必须返回完整结构，缺省值填 `null`。
早期失败路径只返回 `{"track_points": N}`，前端读到 `undefined` 再调
`.toFixed()` 直接把整页打崩，**后端 warnings 里的真正原因被整个吞掉**。

### `GET /api/shots/{name}`

回传原图，供前端叠加显示。

### `POST /api/feedback`

```json
{"name": "安龙小区", "lon": 121.3862, "lat": 31.2120,
 "city": "上海", "crs": "GCJ-02"}
```

用户纠正锚点 → 写回词典（`source="user"`, `boost=2.0`）→ 同区域后续识别直接命中。
若传的是 WGS-84，服务端会转成 GCJ-02 再存（词典内部统一 GCJ-02）。

### `GET /api/lexicon` / `GET /api/lexicon/export`

查看 / 导出本地词典。

---

## 8. 命令行用法

```bash
python -m app.cli <图片路径> [选项]
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--city` | `上海` | 城市提示（地理编码过滤用） |
| `--out` | `<项目>/out` | 输出目录 |
| `--online` | 关 | 允许调用在线地理编码（需已配密钥） |
| `--no-chart` | 关 | 不生成轨迹图 |
| `--no-road-verify` | 关 | 不生成路网验证图（省一次网络请求） |

产出：

| 文件 | 内容 |
|---|---|
| `trajectory_gcj02.geojson` | GCJ-02，腾讯/高德底图直接可用 |
| `trajectory_wgs84.geojson` | WGS-84，GPS 设备 / Strava / GIS 工具 |
| `<图片名>_overlay_result.png` | 原截图 + 识别轨迹叠加 |
| `<图片名>_trajectory_chart.png` | 独立轨迹图（米制等比例，标注路名） |
| `<图片名>_road_verify.png` | OSM 道路几何投影回截图 + 轨迹（绝对位置验证） |

GeoJSON 文件名固定；三个 PNG 带**输入图片的文件名前缀**（多图处理时不会互相覆盖）。

stdout 输出 JSON 摘要（`ok` / `confidence` / `track_points` / `distance_m` /
`roads` / `road_fit` / `diagnostics` / `warnings` / `files`），
退出码 0 = 成功，1 = 无轨迹，2 = 参数或图片错误。

> ⚠️ **识别失败时也必须输出可解析的 JSON**（`ok: false`），
> 而不是抛异常。上层脚本依赖这个契约来做批量处理。
> 早期 `_summary()` 直接 `res.geojson_gcj02["features"][0]`，失败路径上抛
> `KeyError: 'features'` —— CLI 崩溃，脚本拿不到任何结果。

---

## 9. 前端结构

单文件 `web/index.html`（735 行），无构建步骤。IIFE 封装。

### 9.1 右栏布局（重要）

**两个可视化面板上下堆叠、同时可见**：

```html
<div class="vis-block">
  <div class="vis-title"><b>识别结果</b><span class="hint">腾讯底图 · GCJ-02</span></div>
  <div id="map"></div>
</div>
<div class="vis-block">
  <div class="vis-title"><b>原始截图</b><span class="hint">绿色为识别出的轨迹，黄色为锚点</span></div>
  <div id="preview"><img id="prevImg"><canvas id="prevCanvas"></canvas></div>
</div>
```

⚠️ **历史上这里是「地图 / 原图叠加」标签页互斥切换**，
识别完成后停在「地图」页，原图整个消失——用户反馈
「右侧下方原始的怎么都不显示了」。

**用户的预期从头到尾都是「右侧上方是识别到图片，右侧下方是原始图」**，
两个要同时看到：地图用来确认位置，原图用来确认轨迹有没有抠错。
做成互斥后，用户校验时得反复来回切，甚至以为功能坏了。

> 交互设计要问的是「用户想**同时**看到什么」，而不是「怎么排版看起来整齐」。
> 当用户的描述和你的推测冲突时，先信用户的描述。

### 9.2 关键函数

| 函数 | 职责 |
|---|---|
| `drawMap(d)` / `drawMapInner(d)` | 绘制轨迹与起终点标记 |
| `drawPreview(d)` | 在原图上叠加绿色轨迹与锚点 |
| `applyPreviewVisibility()` | 原图面板显隐（恒为 block） |
| `ensureMapReady()` | 确保地图已创建、尺寸正确、视野对准轨迹 |
| `fitMapToTrack()` | 缩放到轨迹范围（带尺寸检查 + try-catch） |

### 9.3 ⚠️ 腾讯地图 GL JS 的三个坑

**坑 1：`map.removeLayer()` 传参**

```js
map.removeLayer(polyline);    // ✗ 传对象 → 抛 "Unhandled error. (undefined)"
map.removeLayer(polyline.id)  // ✓ 传图层 id 字符串
```

SDK 源码：`i.removeLayer = function (t) { var e = this._layers[t]; ... }`
—— `t` 必须是 id 字符串。

传对象 → 查不到 → 抛错 → **中断整个 `drawMapInner`** →
新轨迹没画、`lastPath` 没更新 → `fitBounds` 仍按旧路径定位。
外层 catch 只 `console.error`，于是**完全静默**：
左栏数据是新图的（那部分在 `drawMap` 之前就渲染完了），地图却停在上一张。

**坑 2：0 尺寸容器上做 `fitBounds`**

容器 `clientWidth === 0` 时腾讯地图会抛 `Unhandled error. (undefined)`
（内部算缩放级别时拿到 undefined）。所以：

- `drawMapInner` 里检查容器尺寸，为 0 就不画
- `fitMapToTrack()` 里检查尺寸 + try-catch
- `ensureMapReady()` 在识别完成后补画

**坑 3：鉴权从未成功过**

页面里的 `serviceHost` 占位符曾从未被替换，SDK 请求非法地址 →
`AUTH_ERROR.netError` → `'error'` 事件无监听器 → 抛
`Unhandled error. (undefined)`。

现在由 `app/tmap_proxy.py` 真实转发 `checkKey`/`oversea`/`pingd`，
**key 只存在于服务端**，浏览器永远拿不到。

### 9.4 调试钩子

```javascript
window.__ride = {
  getMap, getPolyline, getMarkers, getLastPath, getLastResult
}
```

自动化测试靠它观测内部状态——IIFE 里的变量外部取不到，
没有这个钩子就只能靠肉眼截图判断。

### 9.5 「识别」与「渲染」必须分开

```javascript
try {
  const r = await fetch('/api/recognize', ...);
  data = await r.json();
  if (!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
} catch (e) {
  showStatus('识别失败：' + (e && e.message ? e.message : e), 'err');
  return;                       // ← 接口失败才叫「识别失败」
}
try {
  render(data);
} catch (e) {
  showStatus('识别已完成（置信度 X%），但地图渲染出错：...<br>GeoJSON 已可正常下载。', 'err');
}
```

渲染失败**不能**显示成「识别失败」——识别其实成功了，数据也拿到了，
只是地图画不出来。混在一起会把底图问题说成识别问题。

---

## 10. 关键参数表

全部集中在 `app/config.py` 的 `VisionConfig` / `AppConfig`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `start_hsv_range` | `(35,90,90,85,255,255)` | 绿色「起」徽章 HSV |
| `end_hsv_range` / `end_hsv_range2` | `(0,...)` / `(170,...)` | 红色「终」徽章（色相环绕，两段） |
| `badge_min_area` / `badge_max_area` | 1500 / 20000 | 徽章面积阈值（px） |
| `track_max_value` | 200 | 轨迹掩膜最大亮度 |
| `track_max_saturation` | 90 | 轨迹掩膜最大饱和度 |
| `track_close_kernel` | 11 | 闭运算核（跨道路描边断点） |
| `track_open_kernel` | 5 | 开运算核（滤文字笔画） |
| `ocr_min_score` | 0.55 | OCR 置信度下限 |
| `anchor_icon_weight` | 1.0 | 图标锚点权重 |
| `anchor_place_weight` | 0.85 | 地标锚点权重 |
| `anchor_road_weight` | 0.5 | 路名权重（**不参与配准**） |
| `ransac_inlier_m` | 200.0 | RANSAC 内点阈值（米） |
| `ransac_min_inliers` | 3 | 最少内点数 |
| `road_fit_enabled` | True | 是否启用路网精修 |
| `road_fit_max_shift_m` | 800.0 | 最大允许平移量（米） |
| `http_timeout` | 12.0 | 单次 HTTP 超时（秒） |
| `max_upload_mb` | 20 | 上传大小限制 |
| `allow_online` | True | 是否允许联网 |

**采样参数在 `pipeline.py` 里**（不是 config）：`step_px = min(max(1.0, 5.0/meters_per_px), 4.0)`，
`eps_px = min(1.0/meters_per_px, 1.0)`。

---

## 11. 已知坑与排障手册

### 11.1 症状 → 根因速查表

| 症状 | 最可能的原因 | 查哪里 |
|---|---|---|
| 识别失败但后端日志全是 200 | 前端渲染阶段抛错被包装成「识别失败」 | 浏览器 console |
| `Cannot read properties of undefined (reading 'toFixed')` | 后端失败路径返回的 diagnostics 字段不全 | `pipeline.DIAG_KEYS` |
| 「所有锚点均未匹配到坐标」 | 词典无覆盖 + 无密钥 + 冷启动失效 | 城市提示是否填写；OSM 是否收录 |
| 轨迹落到**平行**的错误道路族 | POI 标注基准偏差（100–300m） | `road_fit` 是否生效 |
| 轨迹整体偏移约 350–450 米 | 坐标系混用 | `roadfit.py` 的 bbox 是否转了 WGS-84 |
| 轨迹点数少（<20） | 采样步长/简化容差没设像素上限 | `pipeline.py` 的 `step_px`/`eps_px` |
| 轨迹锯齿、里程虚增 | 按投影排序而非图路径追踪 | `track._trace_graph_path()` |
| 里程凭空缩短 15% | 用了「像素长 × 墨卡托scale」估算 | 应用 `haversine` |
| 地图不更新（停在上一张） | `removeLayer` 传了对象 | `web/index.html` `drawMapInner` |
| 原图面板消失 | 显隐被绑到标签页状态 | `applyPreviewVisibility()` |
| 原图下半部分被裁 | 容器有固定高度 | `#preview` 不能有 `height:xxx px` |
| 路网验证图是别的街区 | 给 OSM 喂了 GCJ-02 | `_road_refine` 里转 WGS-84 |
| 缓存的路网偏了几公里 | 缓存复用了「重叠」而非「完全包含」的块 | `_bbox_contains` |
| 识别特别慢（40s+） | Overpass 拉路网 | `data/cache/roads.json` 是否命中 |
| **CLI 崩溃，输出不是 JSON** | 失败路径上访问了不存在的键 | `cli._summary()` 的防御性读取 |
| CLI 识别不出新区域 | 没加 `--online`，冷启动不触发 | 加 `--online`（见第 2.3 节） |
| **路名归属完全对不上（差好几公里）** | 词典短残片条目劫持了正确候选 | `Lexicon.lookup()` 的子串匹配方向 |
| **路名归属对但锚点只有 2 个** | 冷启动簇心过滤丢弃了正确候选 | `coldstart.lookup_many()` 的簇心过滤 |
| 缩放级别高导致精度下降 | **不是这个原因** | 尺度与精度无关，看「有效内点数量」 |
| **城市级截图整体落到别的城区** | 配准被同名候选刷票 / 容差非尺度自适应 | 见第 19 章；`register.py` 的分组与容差 |
| 配准选出带旋转的模型（rot ≠ 0） | affine 用 3 点采样，过参数化 | `register.py` 的模型采样点数 |
| 尺度偏差几个百分点 | 行政区锚点混进了尺度估计 | `register.py` 的 `_estimate_scale_prior` |
| 「3号线」匹配到哈尔滨 | Nominatim 的 viewbox 不是强制过滤 | `coldstart.py` 的 `_in_city()` |
| 词典明明有 40+ 条目却只命中 5 个 | 「够用即停」的 break 把锚点截断了 | `pipeline.py` 的 `_resolve_anchors` |

### 11.1.1 「缩放程度高就不准」是个误解

实测全部样本，尺度越小反而越准：

| 尺度 m/px | 置信度 | 内点 | RMSE | 贴路后 |
|---|---|---|---|---|
| 0.221 | 0.791 | 4 | **8.6 m** | 2.3 m |
| 0.645 | 0.879 | 2 | 0.6 m | 3.3 m |
| 0.857 | 0.748 | **2** | **115.2 m** | 11.7 m |
| 4.083 | 0.885 | 4 | 96.8 m | 6.8 m |

尺度与精度**无关**。决定精度的是**有效内点数量**——
内点只有 2 个时，配准几乎等于两点定线，RMSE 必然上百米。

### 11.1.2 「词典命中就是对的」是个误解

词典优先级高于在线服务与冷启动，但**词典里的坐标可能是错的**。

词典里的短条目多半是 OCR 截断产生的**残片**——它的坐标属于另一个地点：

```
「上海世」（来自「上海西站」被截断） = 121.40071, 31.26275  ← 上海西站
查「上海世贸商城」→ 命中「上海世」    → 偏离真实位置 6.8 km
```

`Lexicon.lookup()` 的子串匹配因此必须是**有方向的**：

| 方向 | 安全性 | 规则 |
|---|---|---|
| 查询名包含词典 key（`key in name`） | 安全 | key ≥ 3 字即可 |
| 词典 key 包含查询名（反向） | **危险** | 需 `len(key)/len(name) ≥ 0.6` |

反向匹配的比例约束（`_PARTIAL_MIN_RATIO = 0.6`）是关键：
「上海世」/「上海世贸商城」= 0.5 < 0.6 → 拦下。

### 11.1.3 「冷启动该替 RANSAC 做筛选」是个误解

冷启动的互证机制用**地理距离**判断区域，簇心周围有别的地名候选就认为那一片是真的。
但**地理上更紧密 ≠ 方位正确**。

实测上海古北一例：

| 地名对 | 地理距离 | 像素 y | 真实纬度 |
|---|---|---|---|
| 虹桥基金小镇 ↔ 红松东路 | **791 m**（赢得簇心，错） | 2521 / 2122 | 31.1829 / 31.1875 |
| 上海世贸商城 ↔ 上海长风大悦城 | 2379 m（正确但被丢） | 1416 / 206 | 31.2019 / 31.2227 |

簇心被定到南边那片错误区域，于是按 1500m 阈值硬过滤时，
「上海世贸商城」（2399m）与「上海长风大悦城」（4471m）**被直接丢弃**——
而它们恰恰是唯一正确的两个。

**修法**：簇心过滤放宽为「**只排序、不丢弃**」，
挑选的正确性交给 RANSAC 的几何一致性。
实测同一组数据，RANSAC 直接用全部候选可得 **RMSE 11.7m、3 个内点**
（而旧逻辑是 RMSE 115.9m、2 个内点）。

> **教训**：当一个「预处理筛选」把正确候选过滤掉时，
> 先问「这个筛选比下游的算法更懂吗」。
> RANSAC 靠几何一致性判断哪一组候选自洽，
> 距离启发式做不到这件事。

### 11.2 三条最重要的元教训

**① 模拟测试只能验证「你假设的行为」。**

前两轮修复「连续识别报错」时，我用 Node 桩模拟腾讯地图行为，测试全绿——
**但桩是我自己写的**，我假设了「隐藏容器会抛错」这个前提，
于是测试验证的是我的假设，不是真实行为。

真实浏览器还当场抓到了我自己新引入的 bug（重构 try 块时 `const data` 作用域写错）。

> 涉及第三方 SDK、浏览器运行时、异步时序时，**必须上真浏览器**。
> Playwright 装上后 5 分钟定位，而之前两轮推断花了很久且都错了。

**② 双向验证必须有。**

「还原成有 bug 的版本 → 测试确实失败」这一步不能省。
第一版前端测试断言全是「有没有抛异常」，结果**bug 版本也能 11/11 通过**
——因为 `fitMapToTrack` 内部的 try-catch 把异常吞了。
改成断言「容器隐藏时有没有创建地图实例 / 有没有调用 fitBounds」才真正抓住。

**③ 遇到第三方 SDK 的怪异行为，去读它的源码。**

`removeLayer` 这个名字完全看不出要传 id，但压缩代码里
`this._layers[t]` 一目了然。

> 反例：我一开始想用 `try/catch` 把这个错误「绕过」——
> 但那只会让它更静默。**绕过不是修复。**

### 11.3 调试工具

```bash
# 真实浏览器端到端（连续识别三张，观测轨迹/视野/面板）
python tools/verify_browser.py

# 探测腾讯地图 SDK 图层 API 的正确用法
python tools/probe_tmap_layers.py

# 复现「连续识别轨迹不更新」
python tools/repro_map_update.py

# 路网验证（OSM 几何投影回截图）
python tools/verify_roads.py <图片> --out docs/
```

`tools/verify_browser.py` 依赖 Playwright + Edge/Chromium。
沙箱环境下 Playwright 自带的 Chromium 可能因 WebGL 限制启动失败，
脚本会自动回退到系统 Edge。

---

## 12. 测试体系

### 12.1 后端端到端（87 项）

```bash
python tests/test_e2e.py
```

覆盖：坐标系基准、轨迹提取、配准模型、路网精修、冷启动互证、
词典强化、GCJ→WGS 偏移量基准等。

第 17 段是 2026-11 新增的「城市级低缩放截图」回归项（12 项），
逐条固化 19.2 里那六个缺陷的修复：

| 用例 | 固化的是什么 |
|---|---|
| 17a | 同锚点的多个候选只算一个内点 |
| 17b | 内点容差随尺度自适应（像素制），行政区容差更大 |
| 17c | 尺度投票不被同名候选刷票 |
| 17d | 跨城同名候选被过滤（哈尔滨/重庆/上海三例） |
| 17e | 四个自洽锚点全部成为内点 |
| 17f | 锚点类型判定（district/station/transit/poi） |

⚠️ 构造测试数据时注意**墨卡托的各向异性**：x 方向 1° = 111320 m，
y 方向 1° = 130100 m；屏幕 y 向下对应地理向南（纬度减小）。
早期写反符号，测试用例本身不自洽，配准直接失败，误判成代码有 bug。

⚠️ **样本缺失时打印「样本缺失」并以非零码退出，不崩溃**。
把自己的截图放进 `samples/` 即可运行。

### 12.2 前端逻辑与代理（29 项）

```bash
python tests/test_frontend.py     # 需要 Node.js
```

把 `web/index.html` 的函数抽出来，配一个**会复现真实腾讯地图行为**的桩
（`removeLayer` 只接受 id、0 尺寸容器 `fitBounds` 抛同样的错），逐场景断言。

7 个场景：

1. 地图页连续识别两张
2. 容器尺寸为 0 时识别
3. `ensureMapReady` 补 `fitBounds`
4. 首次识别时容器尺寸为 0，布局就绪后补建
5. 预览容器样式（不裁切）
6. **连续识别必须换掉旧图层与旧视野**（`removeLayer` 传 id、旧图层无残留、`lastPath` 换新）
7. **地图与原图同时可见** + **互斥标签页已移除**

### 12.3 真实浏览器端到端

```bash
pip install playwright && python -m playwright install chromium
python tools/verify_browser.py
```

验证：连续识别三张不同截图 → 轨迹首点/点数/地图中心都变；
两个面板同时可见且地图在上方；原图未被裁切；无 JS 异常。

### 12.4 ⚠️ 断言写错会既掩盖真问题、也误报问题

实例：验证脚本里我一度把「原图未裁切」写成
`naturalHeight ≈ clientHeight`——**错的**。
`naturalHeight` 是原始像素（2720），`clientHeight` 是按宽度缩放后的显示高度（1606），
两者本来就不该相等。正确判据：

```
渲染高度 == naturalHeight × 渲染宽度 / naturalWidth
```

### 12.5 当前实测基线

| 指标 | 样本1（长宁东西向） | 样本2（普陀对角线） | 浦东（联洋） |
|---|---|---|---|
| 识别耗时 | 10 s | 12 s | 17 s |
| 配准模型 | `north_up` | `north_up` | `north_up` |
| 地面比例尺 | 4.596 m/px | 4.089 m/px | 0.645 m/px |
| 配准残差 RMSE | 120 m | 97 m | 0.6 m |
| 输出轨迹点 | 180 | 169 | 23 |
| 识别里程 | 4.00 km | 4.71 km | 0.84 km |
| 置信度 | 0.863 | 0.887 | 0.879 |
| **轨迹点落在渲染路网上** | **95.8%** | **95.5%** | — |
| **轨迹到真实道路中位距离** | 6 m | — | **2.3 m** |
| **路名归属** | 命中用户实骑 4/5 | — | **锦绣路→民生路→迎春路** |

---

## 13. 数据飞轮：本地词典

`data/lexicon.json` 结构：

```json
{
  "安龙小区": [
    {"lon": 121.38624, "lat": 31.21196, "weight": 44.0, "count": 22,
     "city": "上海", "source": "user", "address": ""}
  ]
}
```

| `source` | 含义 | 当前数量 |
|---|---|---|
| `verified` | 已核验的种子（`seed_lexicon.py` 生成） | 38 |
| `user` | 用户在网页上纠正的 | 1 |
| `osm` | OSM 冷启动检索到的 | 6 |

**共 43 条，全部在上海。**

### 写入路径

| 路径 | 触发 | `source` | boost |
|---|---|---|---|
| `seed_lexicon.py` | 手动跑一次生成种子 | `verified` | — |
| `POST /api/feedback` | 用户在地图上纠正锚点 | `user` | 2.0 |
| 冷启动检索 | 识别时自动写回 | `osm` | — |
| 在线地理编码 | 识别时自动写回 | `amap`/`tmap`/… | 0.5 |

`boost` 越高，词典排序越靠前。

### 词典会长大

每次识别都可能往词典里写新地名。随着使用，命中词典的锚点越来越多，
识别越准越快。这是「越用越准」的实现方式。

---

## 14. 合规边界

### 底图

**只用腾讯地图**（GL JS）。密钥不下发到浏览器（只在本地代理内部使用）。
不用 OpenStreetMap 直连瓦片、Mapbox、Google Maps、Leaflet + OSM。

### OSM 的使用边界

OSM 在本项目中**只提供坐标与几何参考**，不参与底图渲染，
也不作为首选地理编码来源：

| 位置 | 用途 |
|---|---|
| `app/roadfit.py` | 路网几何用于「轨迹必须走在真实道路上」的几何判定与精修 |
| `app/coldstart.py` | 地名检索用于词典未覆盖区域的冷启动兜底（优先级低于四家持牌服务） |
| `app/seed_lexicon.py` | 词典种子同样来自 OSM 名称检索（viewbox 限定） |

**底图始终是腾讯地图；坐标来源优先级见第 5.3 节。**

### 隐私

`.gitignore` 默认排除：

- `samples/*.jpg`（真实骑行截图，含位置历史）
- `docs/*.png`（轨迹可视化）
- `trajectory*.geojson`（原始轨迹）
- `data/cache/`（路网与冷启动缓存）
- `tmp_debug/`（开发时的临时目录）

`data/lexicon.json` **入库**——它只含地名坐标，不含原始 GPS。

---

## 15. WorkBuddy 技能包

项目同时封装成一个 WorkBuddy 技能：`ride-screenshot-georeference`。

```
.workbuddy/skills/ride-screenshot-georeference/
├── SKILL.md                      技能说明（触发条件、用法、解读方法、踩过的坑）
├── scripts/recognize.py          调用封装
├── references/methodology.md     方法论文档
└── assets/ride_core/             自包含识别内核（app 包 + 词典 + requirements）
```

### 同步流程

**代码只在本仓库改**，改完同步到各技能安装：

```bash
python tools/sync_skill.py \
  .workbuddy/skills/ride-screenshot-georeference \
  ~/.workbuddy/skills/ride-screenshot-georeference \
  "<workspace>/.workbuddy/skills/ride-screenshot-georeference"
```

同步后重新打包：

```python
# 用 Python zipfile（环境里没有 zip 命令）
# 详见对话记录，或直接复用 dist/ride-screenshot-georeference.zip 的生成逻辑
```

`dist/ride-screenshot-georeference.zip` 是**自包含**的——
解压到任意装了 Python 依赖的机器即可运行，不依赖 `D:\Dev\meituan-ride` 存在。

### ⚠️ 技能包不包含 web 前端与 tools

`assets/ride_core/` 只同步 `app/`、`data/lexicon.json`、`requirements.txt`。
`web/index.html`、`tools/`、`tests/` **不在技能包里**——
技能只提供「命令行识别」能力，不提供网页服务。
若要让技能也支持网页，需要扩展 `sync_skill.py`。

---

## 16. 代码地图：改哪里

| 我想改… | 去这个文件 |
|---|---|
| 轨迹提取的掩膜/连通域逻辑 | `app/track.py` |
| 锚点分级规则 | `app/ocr_anchor.py:classify_text()` |
| 地理编码优先级 | `app/geocode.py:Geocoder.geocode()` |
| 冷启动的匹配策略 | `app/coldstart.py`（`_variants` / `name_score` / `_find_region_center`） |
| 配准模型或 RANSAC | `app/register.py` |
| 路网精修的平移策略 | `app/roadfit.py:refine_with_roads()` |
| 路名归属来源顺序 | `app/pipeline.py:_attribute_roads()` |
| 采样密度 / 简化容差 | `app/pipeline.py` 阶段 5 |
| 置信度公式 | `app/pipeline.py:_score_confidence()` |
| 坐标系转换 | `app/geo/coords.py` |
| 所有可调参数 | `app/config.py` |
| HTTP 接口 | `app/server.py` |
| 网页布局与渲染 | `web/index.html` |
| 腾讯地图代理 | `app/tmap_proxy.py` |
| 命令行参数 | `app/cli.py` |

### 修改后的自检清单

```bash
python tests/test_e2e.py         # 后端 66 项
python tests/test_frontend.py    # 前端 29 项（改了 web/index.html 必须跑）
python tools/verify_browser.py   # 改了前端或 SDK 交互必须跑
```

⚠️ **改了 `web/index.html` 一定要跑前端测试**——
后端测试发现不了任何前端问题（后端日志全是 200，异常发生在渲染阶段）。

⚠️ **新增断言后要做双向验证**：把修复代码临时还原，确认新断言**确实会失败**。

---

## 17. 后续可做事项

按价值排序，供接手者参考。

### 高价值

1. **精度天花板：锚点基准偏差**
   当前贴路中位 2–6 米，主要误差来自 POI 标注与真实点位的
   100–300 米系统性偏差。路网精修已能纠正**整体平移**，
   但纠正不了**局部形变**（锚点各自偏差方向不同）。
   解法方向：用更多锚点做加权最小二乘而非 RANSAC 硬内点，
   或从多个 POI 源融合。

2. **旋转/倾斜截图**
   `similarity` 模型已实现但从未被真正触发过——
   因为 `register()` 是「按优先级依次尝试，north_up 成功就返回」。
   如果要支持用户旋转截图，需要让模型选择更智能
   （例如比较各模型的 RMSE 而不是按固定顺序）。

3. **多图连续骑行**
   现在一次识别一张截图。若用户一次截好几张，缺少跨图拼接
   （起点终点对齐、去重、合并）。

### 中价值

4. **冷启动覆盖更多城市**
   目前只在上海验证过。冷启动的多锚点互证逻辑本身与城市无关，
   但 `_variants()` 里的「市/省」前缀处理是硬编码的。

5. **OSM 冷启动的 Overpass 兜底**
   `tools/probe_tmap_layers.py` 之外，Overpass 作为 Nominatim 的
   第二数据源还没实现。Nominatim 对某些地名召回率低
   （如「进才实验中学」需要加「上海市」前缀才命中）。

6. **性能**
   冷启动 1 次/秒限速是 Nominatim 政策要求，无法绕过。
   但可以让**首次识别就并行查询**（现在是串行逐个），
   或用 Overpass 一次批量查完。

### 低价值

7. **GPX 输出** — GeoJSON 转 GPX 是纯格式转换。
8. **删除 `app/snap.py`** — 死代码。
9. **技能包支持网页** — 扩展 `sync_skill.py`。

---

## 18. 决策记录

按时间倒序，每条都标注了**为什么**和**放弃了什么方案**。

| # | 决策 | 放弃了什么 | 原因 |
|---|---|---|---|
| 1 | 两个可视化面板上下堆叠、同时可见 | 标签页互斥切换 | 用户的实际工作流是对照着看地图和原图 |
| 2 | `removeLayer(layer.id)` | `removeLayer(layer)` / try-catch 绕过 | SDK 内部是 `this._layers[t]`，参数必须是 id |
| 3 | 底图 key 用官方公开示例 key + 本地代理 | 要求用户必须申请 key | 底图渲染是必要功能，不能因为没 key 就不可用；key 不下发浏览器 |
| 4 | 「识别」与「渲染」分离，渲染失败不报「识别失败」 | 统一 try/catch | 底图问题说成识别问题，误导性极强 |
| 5 | 冷启动三阶段（多变体 + 相似度 + 互证） | 只用 Nominatim 精确匹配 | OCR 会截断地名；单锚点无法自证位置 |
| 6 | 冷启动必须批量调用 | 逐个 `geocode()` | 互证需要多个独立检索的地名 |
| 7 | 互证阈值「≥1 个其他地名在 800m 内」 | ≥2 个 | 浦东只有两个地名能查到，要求 ≥2 会定位失败 |
| 8 | 路网缓存多块 + **完全包含**才复用 | 重叠即复用 | 重叠会引入区域外道路（实测 52 条越界） |
| 9 | 路网获取失败降级继续 | 直接失败 | 路网只用于精修，没有它也有锚点配准结果 |
| 10 | 采样/简化都设像素上限 | 纯按米换算 | 高倍截图下容差会变 7.7 像素，把轨迹削成 6 个点 |
| 11 | Douglas-Peucker 改双判据（弧长 + 垂距） | 纯垂距 / 纯弧长 | 纯垂距在近直线上失效；纯弧长漏掉整体平移 |
| 12 | 字典在路名**之后** | 字典优先 | 词典种子来自 OSM，与国内底图有 100–300m 偏差 |
| 13 | 路名锚点不参与配准 | 全部锚点参与 | 路名文字中心可偏 2–5 公里 |
| 14 | 轨迹用图路径追踪（Dijkstra） | 按投影排序 | 投影排序导致锯齿与里程虚增 |
| 15 | 连通域用像素级拓扑测距 | `aspect_ratio > 1.6` 启发式 | 启发式只适合横向路线 |
| 16 | GCJ→WGS 用不动点迭代 | 一步近似 | 一步近似往返平移 347 米、里程短 15% |
| 17 | OSM 只做参考不做底图 | Leaflet + OSM 瓦片 | 合规要求 |

---

## 19. 城市级低缩放截图的定位（2026-11 新增）

### 19.1 问题现象

换了一张**骑行时间更长、缩放级别更低**的截图（1260×2720，一张图覆盖
整个上海城区，轨迹只占图中 454×263 像素），识别结果整条落到
「上海站 / 静安区」一带，与真实位置差了十几公里。

这是本项目迄今最难的一个 bug——它不是某个参数不对，而是**六个缺陷叠加**。

### 19.2 六层根因

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | `north_up` 有 3 个自由度却用 2 点采样 | 尺度系统性偏差 10% |
| 2 | 同名候选被逐个计入内点 | 内点虚增，伪解胜出 |
| 3 | 内点容差用绝对米数 | 低缩放下可靠 POI 全被判成离群 |
| 4 | 行政区标注点偏移 1–2 公里未被区别对待 | 正确锚点被误杀 |
| 5 | 跨城同名地点（「3号线」→哈尔滨） | 污染候选集 |
| 6 | 词典命中「够用即停」 | 43 条词典里 10 个可用锚点被丢掉 |

第 6 条尤其隐蔽：`targets` 里一旦匹配到 5 个名字就 `break`，
而词典里明明有「红星美凯龙」「花卉市场」「长风公」「阿拉城」「上海会聚」
等 10 个完全可用的锚点，却一个都没进入配准。放开后命中率从 5/38 涨到
23/38，内点从 3 个涨到 9 个，路名归属也恢复成「安龙路/剑河路/金钟路」。

### 19.3 修复后的效果

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 地理编码命中 | 10/30 | **29/30** |
| 配准模型 | affine（伪旋转 -1.67°） | **north_up**（无旋转） |
| 尺度 | 14.5 墨卡托米/像素（偏 8%） | **15.69**（真值 15.73，偏 0.3%） |
| 内点 | 3 个挤在错误区域 | **5 个，全部落在正确区域** |
| 落点 | 上海站 / 静安区 | 静安—徐汇—黄浦一带（与原图一致） |

修复过程中还发现并修掉一个**回归**：把内点门槛提高到 3 个不同锚点后，
一个只匹配到 2 个自洽锚点的样本从「conf 0.822、贴路 3m」变成了配准失败。
为此补了 2 点兜底路径（见 5.4.1 第 3 条），并让留一验证在锚点不足时
退化为直接评分。

### 19.4 这类截图的能力边界

低缩放截图的**绝对精度天然受限**，原因与处理：

| 事项 | 街道级截图 | 城市级截图 |
|---|---|---|
| 尺度 | 0.3–5 米/像素 | 10–30 米/像素 |
| 定位精度 | 十几米 | **几百米到一公里**（标注偏移决定） |
| 路网精修 | ✅ 有效 | ❌ 跳过（数据量过大 + 像素精度不足） |
| 路名归属 | 有意义 | 意义有限 |

因此**低缩放截图不应该追求「轨迹落在具体道路上」**，
能保证的是「落在正确的城区 / 正确的街区群」。置信度也会相应偏低
（实测 0.38 vs 街道级 0.85），这是诚实的表达，不是缺陷。

### 19.5 后续可做的更彻底方案

当前方案仍是「从 OCR 地名反推地理」。更彻底的方向是**直接用底图**：

- **底图图像匹配**：美团地图底图与高德同源。若能取到高德瓦片，
  把截图的底图区域与候选区域的瓦片做特征匹配（SIFT / 模板匹配），
  可**直接求出截图的地理 bbox**，完全绕过 OCR 地名这一环。
  这是最彻底的做法，工程量也最大（需要瓦片 API + 图像匹配管线）。
- **行政区多边形先验**：OSM 的 `boundary=administrative` 是多边形，
  取其质心只是近似；用多边形与轨迹的**形状**做约束，比单点约束强得多。
- **在线地理编码**：高德/腾讯的地理编码能直接返回「静安区」的中心，
  且与美团底图同源，基准一致。当前未配置密钥（`online_geocoding: false`）。
  配上密钥后，城市级截图的定位会有立竿见影的提升。
- **多候选返回 + 人工兜底**：低置信度时返回 Top-N 个候选区域，
  UI 上给「换个位置」按钮，让用户一键切换。这是数据飞轮的入口。


## 附：一句话速查

- **内部用 GCJ-02**，输出按需转 WGS-84；给 OSM 之前必须转。
- **道路名不能当锚点**；路名只用来校验区域。
- **冷启动必须批量 + 三阶段**，单锚点无法自证位置。
- **采样步长和简化容差都要有像素上限**。
- **`removeLayer` 传 id 不传对象**。
- **改了前端必须跑 `tests/test_frontend.py` 和 `tools/verify_browser.py`**。
- **新增断言要做双向验证**（还原 bug 版必须失败）。
- **用户说「A 和 B 都要看到」时，别做成互斥切换。**
