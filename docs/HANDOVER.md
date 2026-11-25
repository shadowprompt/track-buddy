# 项目设计与交接文档

> **给接手者的第一句话**：这不是一个「图像识别」项目，是一个**图像配准**项目。
> GPS 坐标不在图片里。必须用图上印着的 POI 标注当锚点，把像素坐标映射到经纬度，
> 再用真实路网约束校正。理解这一点，其余所有设计选择都是它的推论。

本文档面向**接手继续开发的 AI / 工程师**。事实核对日期：2026-11-16，
对应 commit `24d8d61`（工作区另有未提交改动，以实际仓库为准）。
代码唯一真源仓库：`D:\Dev\track-buddy`
（GitHub: `https://github.com/shadowprompt/track-buddy`，public，MIT）。

> **项目现状（2026-11-23）**：已从「美团骑行截图识别」通用化为
> **track-buddy 运动截图识别**，来源 App 由「输入源画像」参数化，见 §23；
> 配套微信小程序已上线（登录走 gateway，记录存腾讯 COS），网页版与 CLI
> 同步可用。已接入四个画像：**美团 / 高德 / 行者 / 小米运动健康**——行者画像含闭环赛道
> 管线（中国邮路追踪 + 断裂段桥接 + 成对徽章定位，见 §24.9/24.17/24.20）；
> 小米当前为首张真实户外跑步概览截图驱动的可测试初版，绝对位置仍需多样本验证。
> 地理编码已单源化（`MAP_GEOCODING_ORDER=amap`，本地经服务器中转复用
> 高额度 key，见 §24.12 / docs/amap_relay.md）。**下一个开发目标：FIT
> 格式导出；进一步提升绝对位置精度（锚点标注偏移仍是天花板）。**

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
20. [低缩放截图的定位：路网作校验器+命名器](#20-低缩放截图的定位路网作校验器--命名器2026-11-新增)
21. [低缩放的本质边界](#21-低缩放的本质边界大模型用户补充底图匹配各自能做什么2026-11)
22. [「轨迹跑到浦东」的根因与修复](#22-轨迹跑到浦东的根因与修复2026-11-03-新增)
    - [22.21 多图拼接：从「猜形状」到「量形状」](#2221-多图拼接从猜形状到量形状2026-11-05)
23. [新增截图识别来源：Source Profile 扩展手册（给下一个 AI）](#23-新增截图识别来源source-profile-扩展手册给下一个-ai2026-11-16新增)
24. [诚实失败闸门链 + 多源地理编码 + 概要里程尺度先验](#24-诚实失败闸门链--多源地理编码--概要里程尺度先验2026-11-17-新增)
    - [24.7 行者来源画像接入](#247-行者xingzhe来源画像接入2026-11-17-2100真实样本标定)

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

统一解释器为 **Python 3.11**（`D:\soft\Python311\python.exe`）。全新环境：

```bash
pip install -r requirements.txt
```

🔴 **已有环境只补缺包**：项目实际装的是 `opencv-python 4.13`（rapidocr 依赖），
而 `requirements.txt` 写的是 `opencv-python-headless`，两者都提供 `cv2`，
全量 `pip install -r` 会覆盖冲突。云记录还需 `cos-python-sdk-v5`，单独装：
`python -m pip install cos-python-sdk-v5`。未装 SDK / 未配云凭据时识别行为不变。

可选（仅用于真实浏览器端到端验证）：

```bash
pip install playwright && python -m playwright install chromium
```

### 2.2 三种用法

```bash
# ① 网页服务（主要用法）——用 run.py，不要用 python -m app.server
python run.py --host 127.0.0.1 --port 8848
# 打开 http://127.0.0.1:8848
# run.py 负责把 src/ 注入 sys.path（无需手动设 PYTHONPATH），
# 并在后台线程预热 OCR（不阻塞监听）；python -m app.server 会跳过这两步。

# ② 命令行（需 PYTHONPATH=src，run.py 方式不用）
PYTHONPATH=src python -m app.cli samples/meituan_ride_shanghai.jpg --city 上海 --out out/
# 产出：trajectory_gcj02.geojson / trajectory_wgs84.geojson（交付用 WGS-84）
#       overlay_result.png / trajectory_chart.png / road_verify.png
# stdout 输出一段 JSON 摘要
# 也可直接调技能包脚本，自带内核：
#   python ~/.workbuddy/skills/ride-screenshot-georeference/scripts/recognize.py <图> --out <目录>
```

③ 作为 Python 库（需 `PYTHONPATH=src` 或把 `src/` 加入 sys.path）：

```python
from app.config import CONFIG
from app.pipeline import RidePipeline
from app.cv_utils import imread_unicode

res = RidePipeline(CONFIG).run(imread_unicode("shot.jpg"), city="上海")
print(res.confidence, res.track_points, res.distance_m)
print(res.geojson_gcj02)   # 国内底图用这个
print(res.geojson)         # GPS 设备 / Strava / GIS 用这个
```

### 2.3 地图密钥与地理编码源

🔴 **高德配额已用尽，不再调用**（`AMAP_KEY` 保留仅为兼容旧配置，不要依赖它）。
识别走 `MAP_GEOCODING_ORDER` 指定的在线源，当前部署为百度 + 天地图；
配置驱动、可单源可多源，改 `.env` 即时生效、无需重启。

| 变量 | 用途 | 说明 |
|---|---|---|
| `MAP_GEOCODING_ORDER` | 在线地理编码顺序 | 如 `baidu,tianditu`；单源写一个，多源按顺序回退 |
| `TIANDITU_API_KEY` | 天地图 tk | 天地图单源时必配，只认这一个变量名 |
| `BAIDU_MAP_KEY` | 百度地理编码 | 当前部署的在线源之一 |
| `AMAP_KEY` | 高德（**已停用**） | 配额用尽，不要尝试调用或等恢复 |
| `BASEMAP_PROVIDER` / `AMAP_JS_KEY` / `AMAP_JS_SECURITY` | 网页底图 | 前端底图适配层，与地理编码 key 是两回事；默认 `tmap` 用公开示例 key |

配置方式：环境变量，或项目根目录 `.env` 文件（密钥只在服务端，绝不下发客户端）：

```bash
MAP_GEOCODING_ORDER=baidu,tianditu
TIANDITU_API_KEY=你的tk
BAIDU_MAP_KEY=你的key
```

未配置在线源时降级为纯本地词典 + OSM 冷启动，只能识别词典覆盖的区域，
绝对精度受限 100–300m。四类后端能力（地理编码/路名/道路几何/路线校验）
的可插拔装配与顺序变量见 README「配置」章节。

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
  ├─②.5 模板召回  app/route_template.py + app/template_fallback.py
  │     宽泛概览图 / 已沉淀的「像素折线↔地理折线」按画像隔离命中，直接兜底
  │
  ├─③ 地理编码  app/geocode.py + app/coldstart.py + app/maps/
  │     优先级：用户确认 > 在线(MAP_GEOCODING_ORDER 指定，当前 baidu,tianditu；高德已停用)
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
  ├─④.7 用户路名线约束  app/roadname_constraint.py
  │     用户提供的路名作为线约束重新定位（可选）
  │
  ├─⑤ 折线投影  pipeline
  │     采样(5m) → 简化 → 像素→墨卡托→GCJ-02 → 交付转 WGS-84
  │     有运动概要(ride_summary.py)时按里程比例插值时间戳 → GPX 有时间轴
  │     输出：GeoJSON（两份：GCJ-02 与 WGS-84）
  │
  ├─⑤.5 路名串路  app/roadchain.py + app/routebuild.py + app/seampath.py
  │     有序路名链 → 串出经过道路 → 补接缝
  │
  └─⑥ 路名归属 + 概要  app/roadname.py + app/ride_summary.py
        优先：路网几何判定 > 在线路径规划 > 截图自带标注
        概要读距离/时长/热量，与轨迹组合成完整运动记录
```

### 3.2 三层架构

```
┌─────────────────────────────────────────┐
│  run.py → app/server.py  FastAPI/HTTP    │  ← 网页 / 小程序
│  app/cli.py              命令行          │  ← 脚本 / 技能包
├─────────────────────────────────────────┤
│  app/pipeline.py  RidePipeline 编排       │  ← 唯一的业务真相（9 阶段）
│  app/source_profile.py  输入源画像        │  ← ContextVar 切换来源
│  （其余模块都是纯函数/无状态类）           │
├─────────────────────────────────────────┤
│  图像前端  track.py ocr_anchor.py         │
│           multishot.py stitch_backends/  │
│  地理配准  geocode.py maps/ register.py   │
│           roadfit.py roadname.py         │
│  护栏校准  anchor_quality.py eval.py      │
│           calibration.py waterguard.py   │
│  输出      exporters/ ride_summary.py     │
│           records/ visuals.py            │
└─────────────────────────────────────────┘
```

**设计约束**：除 `pipeline.py` 外，所有模块都不应直接依赖彼此，
只依赖 `config.py` 的参数与 `geo/coords.py` 的坐标转换。
这样每个模块都能单独测试。接新来源只改 `source_profile.py` 的画像配置；
换地图服务商只改 `.env` 的顺序变量，两者都不碰共享算法代码。

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

> ⚠️ **2026-11 重大升级**：低缩放截图现在也做路网精修，而且**尺度也参与搜索**。
> 详见 5.5.1 与第 20 章。

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

按运行流程分层（行数以 2026-11-17 源码为准，仅量级参考）。完整的文件→职责
一句话映射见 `README.md`「项目结构」章节；下表是**带关键入口/行数**的速查版。

**① 入口与编排**

| 文件 | 行数 | 职责 | 关键入口 |
|---|---|---|---|
| `run.py`（仓库根） | 50 | 启动入口：注入 path + 后台预热 OCR + 拉 uvicorn | `main()` |
| `app/server.py` | 1908 | FastAPI 服务，20+ 路由 | `app` (ASGI) |
| `app/cli.py` | 124 | 命令行入口 | `main()` |
| `app/pipeline.py` | 2835 | 流水线编排（唯一业务真相，9 阶段） | `RidePipeline.run()` |
| `app/config.py` | 555 | 全局参数 + `.env` 热读（mtime/size/sha256） | `CONFIG`, `VisionConfig` |
| `app/source_profile.py` | 193 | 输入源画像：换 App 只改这里 | `PROFILES`, `get_profile()` |

**② 图像前端（截图 → 像素折线 + 锚点）**

| 文件 | 行数 | 职责 | 关键入口 |
|---|---|---|---|
| `app/track.py` | 775 | 轨迹提取（掩膜、骨架化、图路径追踪） | `extract_track()` |
| `app/ocr_anchor.py` | 361 | OCR + 锚点分级（含 `warmup_ocr`） | `ocr_texts()`, `build_anchors()` |
| `app/multishot.py` | 1148 | 多图拼接（街道级路线专用） | `stitch()` |
| `app/posegraph.py` | 129 | 拼接位姿图优化（分摊累积漂移） | — |
| `app/stitch_backends/` | — | 拼接后端：`sift.py`（默认）、`llm.py`、`registry.py` | `stitch()` |
| `app/cv_utils.py` | 224 | 图像 IO、几何工具 | `imread_unicode()`, `simplify_polyline()` |
| `app/scale.py` | 180 | 图像几何尺度（像素阈值唯一来源） | — |
| `app/input_quality.py` | 191 | 输入质量独立校验 | — |
| `app/stitchcheck.py` | 255 | 多图拼接核对页 | — |

**③ 地理编码与配准精修（技术核心）**

| 文件 | 行数 | 职责 | 关键入口 |
|---|---|---|---|
| `app/geocode.py` | 934 | 地理编码编排（在线源经 maps/ 装配） | `Geocoder.geocode()` |
| `app/maps/` | — | 可插拔地图能力层：contracts / credentials / registry / http + geocoding / roadname / roads / routing / staticmap 各 provider | `registry.py` |
| `app/coldstart.py` | 698 | OSM 冷启动（词典无覆盖时兜底） | `get_cold_start().lookup_many()` |
| `app/register.py` | 1183 | RANSAC 配准 | `register()`, `apply_to_polyline()` |
| `app/roadfit.py` | 595 | 路网约束精修（消「平行路陷阱」） | `refine_with_roads()` |
| `app/roadname_constraint.py` | 707 | 用户路名线约束精修 | `refine_by_road_names()` |
| `app/mapmatch.py` | 182 | 地图匹配 HMM / Viterbi | — |
| `app/roadname.py` | 280 | 路名归属 | `attribute_by_*()` |
| `app/roadchain.py` | 375 | 有序路名链 DP 选择 | `select_chain()` |
| `app/routebuild.py` | 478 | 按路名几何串出路线 | `build_route()` |
| `app/seampath.py` | 154 | 接缝插入（Valhalla 填真实路网） | `insert_seams()` |
| `app/chainfit.py` | 364 | 路名骨架定尺度/位置/朝向 | — |
| `app/endpointfit.py` | 342 | 起终点锚定（解尺度） | `fit_by_endpoints()` |
| `app/shapefit.py` | 737 | 形状重建（旧方案，已被 routebuild 取代） | `fit_shape()` |
| `app/simfit.py` | 554 | 相似变换拟合实验 | — |
| `app/valhalla.py` / `app/osrm.py` | 319 / 197 | 路由客户端（bicycle/pedestrian / driving） | `*.route()` |
| `app/llm.py` / `app/llm_geo.py` | 221 / 183 | 可选语义层（需几何验证才采纳） | `review_road_names()` |
| `app/geo/coords.py` | 153 | 坐标转换与投影 | `wgs84_to_gcj02()`, `gcj02_to_wgs84()` |

**④ 护栏与校准（防「看起来准」）**

| 文件 | 行数 | 职责 | 关键入口 |
|---|---|---|---|
| `app/anchor_quality.py` | 312 | 分离 OCR 残片与系统性偏移 | — |
| `app/waterguard.py` / `app/waterdata.py` | 269 / 130 | 跨江河流护栏 / 地理数据 | `check_water_crossing()` |
| `app/calibration.py` | 322 | 置信度校准（自洽分→经验成功率） | — |
| `app/eval.py` | 373 | 准确率评测（可证伪指标） | — |
| `app/route_template.py` | 832 | 轨迹模板沉淀与复用（P0） | — |
| `app/template_fallback.py` | 124 | 宽泛概览图前置判据（P1） | — |
| `app/lexicon_guard.py` / `app/seed_lexicon.py` | 177 / 96 | 词典写入防毒 / 种子生成 | — |

**⑤ 输出层**

| 文件 | 行数 | 职责 | 关键入口 |
|---|---|---|---|
| `app/exporters/` | — | `base.py` 公共基座 + `gpx.py` + `tcx.py`（交付 WGS-84） | `export()` |
| `app/ride_summary.py` | 417 | 概要 OCR 解析（距离/时长/热量→时间轴） | — |
| `app/records/` | — | `auth.py`（微信登录+HMAC token）、`keys.py`（COS 键唯一构造处）、`store.py`（COS 存储）、`service.py`（保存编排） | `save_record()` |
| `app/visuals.py` | 409 | 三类可视化产出 | `build_overlay()`, `build_road_verify()`, `build_chart()` |
| `app/trajmap.py` / `app/tmap_proxy.py` | 209 / 196 | 轨迹地图对照页 / 腾讯地图 GL JS 本地代理 | — |

> ⚠️ `app/snap.py`（207 行）仍是**死代码**：`snap_to_roads()` 无任何调用，
> 且同名 `RoadNetwork` 与 `roadfit.py` 实现不同（走高德/腾讯路径规划而非 OSM 几何）。
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

### `POST /api/recognize_multi`（多图，2026-11-05 新增）

multipart/form-data：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `files` | file[] | 必填 | 多张截图（沿路线滑动截取，相邻约 1/3 重叠）|
| `city` | text | `""` | 同单图 |
| `output_crs` | text | `"WGS-84"` | 同单图 |
| `allow_online` | bool | `true` | 同单图 |
| `refine` | text | `"auto"` | 同单图 |
| `road_names` | text | `""` | 同单图 |

服务端会先用 `app/multishot.py` 把多张图拼成一张大图，再走单图的完整流程。
响应体 = 单图响应 + 以下字段：

| 字段 | 说明 |
|---|---|
| `stitch.ok` | 是否拼接成功 |
| `stitch.used` / `stitch.unused` | 用上了哪几张 / 哪几张没重叠被排除 |
| `stitch.canvas_size` | 拼接画布尺寸 `[宽, 高]` |
| `stitch.shots[]` | 每张原图的落位 `box`（画布坐标，供前端画定位框）|
| `stitch.links[]` | 每对图的重叠关系：`inliers` / `ncc` / `tx` / `ty` / `scale` |
| `stitch.verify[]` | **像素级复核**：每张图贴回画布后的相似度 |
| `stitch.verify_score` | 复核分中位数（<0.9 判接缝可能没对齐）|
| `stitched_url` | 拼接图 URL（PNG 无损，供前端叠加）|

⚠️ `image_size` 在多图模式下是**拼接画布**的尺寸，不是单张原图的——
前端的轨迹叠加必须按它换算，否则会错位。

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
| `overlay_result.png` | 原截图 + 识别轨迹叠加 |
| `trajectory_chart.png` | 独立轨迹图（米制等比例，标注路名） |
| `road_verify.png` | OSM 道路几何投影回截图 + 轨迹（绝对位置验证） |

GeoJSON 与三个 PNG 都用**固定文件名**；不同样本靠 `--out` 目录隔离，
输出命名不随输入文件名变化（通用性约定见 22.26）。

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

### 12.1.1 多图拼接（8 + 1 项，2026-11-05 新增）

```bash
python tests/test_multishot.py          # 8 项，纯离线
python tests/test_multishot_e2e.py      # 1 项，需服务在跑
```

`test_multishot.py` 覆盖：

| 用例 | 固化的是什么 |
|---|---|
| 两图拼接位移 | 位移必须等于真值 932px（容差 6px）|
| 静态 UI 不干扰 | 带 UI 时位移仍等于真值 |
| 三图链式拼接 | A↔C 偏移 = 两个单跳之和 |
| **内容落点方向回归** | 黑标落点偏移必须等于位置框报告的偏移（容差 20px）|
| **像素级复核分** | 每张图复核相似度 > 0.85 |
| 不重叠时正确排除 | 无重叠的图被报告为 `unused` 而不是乱拼 |
| 无关图优雅失败 | 不崩溃、给出原因 |
| 单图透传 | 单图时原样返回 |

⚠️ **方向回归那条最重要**：它锁住的是 22.21.5 那个 `warpPerspective`
方向坑——传反了内容会整体错位 500+ 像素，
而画面「看起来」正常，肉眼发现不了。

⚠️ **写这类测试的两个坑**：
- 用「精确颜色相等」找标记会被 JPEG 压缩坑到（边缘混色）→ 用黑块 + 灰度阈值 + 连通域；
- 标记可能被另一张图**覆盖**（正常的覆盖关系，不是 bug）→
  测试要区分「落点偏移不符」与「落点处没有标记」两种情况。

`test_multishot_e2e.py` 做**米级校验**：真值已知的合成数据，
断言起终点误差 < 60 米、里程偏差 < 25%。
它的合成参数由 `tools/pick_synth_route.py` 程序化挑选——
手工调经纬度太容易出错（高德静态图 zoom=17 只覆盖 523 米见方，
路线跨 900 米必然出界，起终点被裁 → 徽章自然检不出来）。

### 12.2 前端逻辑与代理（42 项）

```bash
node tests/frontend_hint_test.js web/index.html    # 需要 Node.js
python tests/test_frontend.py                      # 需要 Node.js
```

`frontend_hint_test.js` 从 `web/index.html` 里**抽取真实函数**执行
（不依赖浏览器——比截图验证更精准，且能覆盖全部分支）。

11 个场景：低置信度提示的 6 种组合（缩放过低 × 填没填路名 × 高低置信度）
+ 多图拼接信息块的 5 种（成功 / 失败 / 有图没拼上 / 复核分通过 / 复核分可疑）。

⚠️ **改了前端必须跑这个**。`drawPreview()` 的 canvas 定位逻辑依赖
`object-fit:contain` 后的实际显示矩形，
一旦改成用容器的 100% 就会整体偏移——这类问题肉眼很难发现。

`tests/test_frontend.py` 则把 `web/index.html` 的函数抽出来，
配一个**会复现真实腾讯地图行为**的桩
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
解压到任意装了 Python 依赖的机器即可运行，不依赖 `D:\Dev\track-buddy` 存在。

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
| **多图拼接（配准、UI 剔除、像素复核）** | `app/multishot.py` |
| **按路名串路（形状重建）** | `app/routebuild.py` |
| **接缝填充（Valhalla 真实路网）** | `app/seampath.py` |
| **起终点锚定（解尺度）** | `app/endpointfit.py` |
| **路名约束精修（可改尺度）** | `app/roadname_constraint.py` |
| **跨江检测护栏** | `app/waterguard.py` |
| **外部路网交叉验证** | `app/valhalla.py`（bicycle/pedestrian）、`app/osrm.py`（driving）|
| **可选语义层（需几何验证才采纳）** | `app/llm.py`、`app/llm_geo.py` |
| 拼图的落位框与复核分展示 | `web/index.html:stitchBlock()`、`drawPreview()` |

### 修改后的自检清单

```bash
python tests/test_e2e.py             # 后端端到端
python tests/test_multishot.py       # 多图拼接（8 项，纯离线）
python tests/test_multishot_e2e.py   # 多图端到端米级校验（需服务在跑）
python tests/test_routebuild.py      # 串路（11 项）
node tests/frontend_hint_test.js web/index.html   # 前端（42 项）
python tests/test_frontend.py        # 前端 + 腾讯地图桩
python tools/verify_browser.py       # 改了前端或 SDK 交互必须跑
```

⚠️ **改了 `web/index.html` 一定要跑前端测试**——
后端测试发现不了任何前端问题（后端日志全是 200，异常发生在渲染阶段）。

⚠️ **新增断言后要做双向验证**：把修复代码临时还原，确认新断言**确实会失败**。
（这条对 22.21.5 的方向回归尤其重要——
它锁住的是一个「画面看起来正常但内容错位」的 bug，
不双向验证就不知道断言本身有没有写错。）

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


## 20. 低缩放截图的定位：路网作「校验器 + 命名器」（2026-11 新增）

### 20.1 问题

第 19 章把城市级截图的落点从「完全错误的城区」修到了「正确的城区」，
但**精度仍是公里级**：锚点拟合出的尺度有 3-8% 误差，作用在 20 公里的
跨度上就是 600-1600 米的定位偏差；而路网精修当时对低缩放是**直接跳过**的。

早期跳过低缩放路网精修的理由是「一张图覆盖十几公里，路网拉不动，
贴路也没意义」。这个判断只对了一半：

- **拉不动是真的**：按整块 bbox 拉全量路网，Overpass 要 42 秒
  （实测 5634 条具名道路），用户等不起；
- **「贴路没意义」是错的**。低缩放下真正的问题不是「贴到哪条路」，
  而是**位置与尺度本身就不对**。

### 20.2 最终方案：路网按尺度切换两种角色

| 模式 | 条件 | 角色 | 行为 |
|---|---|---|---|
| **修正** | 尺度 < 5 m/像素（街道级） | 修正器 | 平移轨迹使贴路距离最小 + 路名归属 |
| **校验 + 命名** | 尺度 > 8 m/像素（城市级） | 校验器 + 命名器 | 只算贴路距离与路名归属，**不做任何微调** |

实测效果：

| 样本 | 修复前 | 修复后 |
|---|---|---|
| 城市级 置信度 | 0.404 | **0.502** |
| 城市级 贴路中位 | — | **58 m**（OSM 与高德底图的差异上限） |
| 城市级 路名归属 | 空 | 时光印记大道、打浦路隧道、通耀路、世博大道、龙腾大道 |
| 街道级 贴路中位 | 49 m | **5 m** |
| 街道级 路名归属 | 安龙路… | 安龙路、新渔东路、北虹路、新渔路、哈密路 |

### 20.3 为什么低缩放下**不做**位置微调

这一条是反复实测才定下来的，结论与直觉相反：

- **只优化贴路距离会跑偏**。低缩放下路网稀疏，轨迹整体平移几公里后
  照样能压在另一条平行的主干道上，目标函数甚至更优——
  **贴到路上 ≠ 位置正确**。
  实测：只优化贴路时落点比锚点已知的真实位置还偏 2.5-3 公里
  （「静安区」标签的真实位置就摆在那里）。
- **加入锚点正则也救不回来**。锚点项的权重无论取 0.1 还是 0.5，
  优化都会把尺度推到 ×1.08、锚点残差从 271 米涨到 1692 米，
  而贴路只从 58 米变到 54 米——**净亏**。
- **尺度变化必须绕参考点**。绕像素原点缩放 4%，在 1000 量级的像素
  坐标上就是 600 米的绝对漂移（把「尺度微调」变成「位置平移」）。
  绕轨迹质心缩放才只改形状、不动中心。

所以低缩放下路网的真正价值是：①**验证定位是否落在合理街区**
②**给出路名归属**。修正交给锚点，路网负责交叉检验。

### 20.4 工程上的五个坑

1. **数据源按面积选**：OSM API 适合小块（实测 3.5 秒），但它还有
   **节点数上限 5 万**——上海这种高密度区域 0.0054 平方度就已超限，
   请求直接 HTTP 400（而同一范围 Overpass 亚秒级返回 2028 条）。
   阈值定在 0.0015 平方度。
2. **Overpass 需要重试 + 镜像轮换 + 扩大范围**：主站繁忙时返回 504
   网关页而非 JSON（直接 `json.loads` 只会看到 JSONDecodeError），
   同一时刻其它镜像又常年不可用。实测主站查询要 40-46 秒，
   超时给 50 秒会「刚好差一点」判成失败。
3. **路网等级要排除细碎小路**。三个方向都实测过：
   只留主次干道 → 贴路 221 m（附近没主次干道）；
   留全量 → 轨迹「就近吸附」到旁边辅路，贴路 33 m；
   排除 service/footway、保留到居住路 → **贴路 56 m（城市级）/ 5 m（街道级）**。
4. **bbox 要对齐到固定网格**（0.005°）。否则分块边界随轨迹采样与配准的
   亚像素变化而微动，缓存块的「完全包含」判定永远不成立——实测每次
   识别都重新拉一遍路网（每次 220 秒），缓存形同虚设。
5. **缓存块不按道路等级区分**。覆盖更大区域的块可以复用给小 bbox，
   路网覆盖越大贴路越准（实测街道级贴路 33 m → 5 m）。

### 20.5 异步精修：不让识别请求干等

路网取数是网络密集操作（首次 50-130 秒，缓存后 <5 秒）。挂在识别主流程里
会让请求长时间挂起，因此改成：

- 主流程先出**粗结果**立即返回（实测 11.8 秒），响应里带
  `refine_pending: true`；
- 后台线程继续拉路网、算贴路距离、归属路名；
- 前端轮询 `GET /api/refine/{shot_id}`，拿到精修结果后自动更新地图。

前端实现见 `web/index.html` 的 `pollRefine()`。

### 20.6 仍然存在的本质限制

**低缩放下没有任何单一信息源能给出绝对位置**，这是实测数据支撑的结论：

| 信息源 | 粒度 | 低缩放下的误差 |
|---|---|---|
| POI 锚点（车站/场馆） | 粗 | 500-900 米 |
| 行政区锚点 | 粗 | 1.5-2.6 公里（标签被挪到视野空白处） |
| 路网几何 | 细 | 30-100 米（数据源差异）但**绝对位置不唯一** |
| 底图像素 | 最细 | 理论上像素级（见下） |

**路网只能定形状与尺度，绝对位置只能靠锚点，而锚点在低缩放下必然有
公里级偏差。** 要真正突破，需要引入第四种信息源——底图像素。

已验证**底图是同源的**（同一街区一一对应：高德瓦片的「徐汇区/中环路/
上海世博园/7 号线」与美团截图完全对应），因此「拿瓦片与截图做图像匹配
直接求地理 bbox」在原理上可行。当前卡在两处：

- **样式差异**：美团是米黄淡雅风，高德瓦片是冷白标准风，灰度相关仅 0.05；
- **文字干扰**：地名标注在两图位置不同，SIFT 只留下 65 对匹配。

可行的攻关方向：不做像素级 SIFT，而是提取两图的**道路骨架**（top-hat
变换，已验证能把美团淡雅底图里的白路提取出来）再做形状匹配；
或者等腾讯矢量瓦片的鉴权方案（`rt.map.gtimg.com/tile?type=vector` 现已
返回空白，需要 JS SDK 的签名参数）。


## 21. 低缩放的本质边界：大模型、用户补充、底图匹配各自能做什么（2026-11）

### 21.1 为什么要做这一章

第 19、20 章把城市级截图从「落到错误城区」修到「落在正确城区」，
但**精度停在公里级**，用户明确指出「这不是小修小补的问题，要本质的办法」。

本章做了三个**可复现的实验**，用数据回答「还有没有别的路」：

| 实验 | 脚本 | 结论 |
|---|---|---|
| 底图是否同源 | `tools/probe_basemap_match.py` | ✅ 同源，可对齐到像素 |
| 底图匹配在街道级 | 同上 | ✅ 可行，Chamfer 20 倍信噪比 |
| 底图匹配在城市级 | 同上 | ❌ **不可行**，信号无法定出唯一位置 |

### 21.2 实验一：底图确实同源

美团骑行 App 的底图与高德瓦片是**同一份数据**。

验证方式：取「上海动物园」（词典内坐标 121.3625, 31.1932），
在 z=15 的高德瓦片上按 XYZ 公式算位置并画红圈——红圈**精确落在
「上海动物园」标注上**。

推论：**底图像素是完全可用的信息源**，理论上可以绕开 OCR 直接求出
截图的地理 bbox。这是项目诞生时的设想，现在得到确认。

⚠️ 两个必须记住的坑：
- **瓦片 XYZ 索引直接用 GCJ-02 墨卡托**（高德瓦片本身就是 GCJ-02 渲染的），
  **不要**转 WGS-84——转了会偏 450 米（东）/ 200 米（南）。
- 瓦片抓取会限流，连续抓 135 片时大部分失败（返回空图）。
  分块小批量抓取 + 0.2-0.6 秒间隔才稳。

### 21.3 实验二：街道级——底图匹配可行

做法：top-hat 提取「浅色底图里的白色道路」（美团淡雅风、高德彩色风都适用）
→ Canny 成骨架 → Chamfer 距离（距离变换）比对形状。

结果（`samples/meituan_ride_shanghai.jpg`，600x600 区域）：

| 指标 | 数值 |
|---|---|
| 最优 Chamfer | **3.99 米** |
| 随机位置基线均值 | 79.2 米 |
| 信噪比 | **19.9 倍** |

结论：**街道级截图可以用底图匹配做高精度定位**（理论上能做到米级）。

### 21.4 实验三：城市级——底图匹配不可行（关键结论）

同样方法用在城市级截图（`data/uploads/1793589815_20da8357.jpg`，
15.73 米/像素，±1500 米搜索范围）：

| 指标 | 数值 |
|---|---|
| 最优 Chamfer | 3.52 米 |
| 配准原位 Chamfer | 4.29 米 |
| **改善倍数** | **仅 1.22 倍**（对比街道级的 20 倍） |
| 位置分布（最小/中位/最大） | 3.52 / 4.32 / 5.28 米 |
| **前 12 候选的分数差** | **3.52 vs 3.52 米（0.1%）** |
| **但这些候选的地理位置** | **分散在 900–1600 米范围内** |

**这就是核心问题**：分数几乎分不出高下，位置却相差一公里多。

低缩放下一张图覆盖 20 公里，道路骨架在 15 米/像素的尺度上高度**自相似**
——「几条平行主干道 + 几个路口」这种结构在整个城区到处都有。
所以匹配分数是一个**平缓的多峰曲面**，不是尖峰。

结论：**低缩放下没有任何单一信息源能给出精确绝对位置。**

### 21.5 三条路各自的真实能力边界

#### A. 大模型（语义层）——有用，但**不能生成坐标**

能做且值得做：

| 用途 | 具体做法 | 价值 |
|---|---|---|
| OCR 文本纠错 | 「每万佛讲寺」→「万佛讲寺」、「长风公」→「长风公园」 | 中高（直接提升锚点质量） |
| 地名类型判别 | 区分行政区 / 道路 / POI / 地铁线 | 高（决定内点容差） |
| 地名共现推理 | 「静安区 + 徐汇区 + 长宁区」→ 推断这是上海中心城区 | 中（提供区域先验，缩小搜索空间） |
| 多候选重排 | 结合周边地名的语义合理性给候选打分 | 中高 |
| 用户输入解析 | 用户说「我从家到公司」→ 提取约束 | 高（见下） |

**不能做**：
- 生成坐标。LLM 没有地理真值，输出坐标必然是幻觉。
- 判断像素级对齐。几何精度它给不了。

#### B. 用户补充（数据飞轮）——**这是最有效的手段**，但要设计对

用户能提供的**不是坐标**（他也不一定知道），而是**高信息量的弱约束**：

| 用户能给的 | 价值 | 怎么用 |
|---|---|---|
| 「这是XX区XX路附近」 | **极高** | 直接作为锚点进 RANSAC |
| 起终点的街区名 | **极高** | 轨迹两端各给一个强约束 |
| 「我骑过的路名」 | 高 | 路名 → OSM 路网几何 → 作为线约束 |
| 选「更像」的候选 | 高 | 多候选 UI 里一键选择 |
| 精确坐标 | 极高 | 已有 `/api/feedback` 纠正接口 |

⚠️ **关键设计原则**：不要问用户要坐标，要问**用户能答得上的问题**。
「这段轨迹大概在哪个区？」比「请给出精确经纬度」有效得多——而且
低置信度时给 Top-N 候选让用户**点一下**，比让用户打字更快。

当前的 `/api/feedback` + 前端「纠正」按钮已经是这个机制，
但只支持「选一个内点 → 点地图纠正」，覆盖面太窄。
应扩展为：**低置信度时把 Top-N 候选区域以列表形式呈现**，让用户选。

#### C. 底图匹配（几何层）——分场景用

| 场景 | 角色 | 依据 |
|---|---|---|
| 街道级（< 5 m/px） | **主力定位手段**，可做到米级 | Chamfer 20 倍信噪比 |
| 城市级（> 8 m/px） | **只能当合理性校验**，不能定位置 | 最优解只比原位好 1.22 倍 |

### 21.6 推荐的最终架构：三层叠加

```
用户输入（语义约束：区/路/起终点/选候选）
        ↓
大模型（把输入转成结构化约束 + OCR 纠错 + 候选语义重排）
        ↓
几何层（锚点配准 + 路网校验 + 街道级的底图匹配）
        ↓
低置信度 → 返回 Top-N 候选 → 用户点选 → 回流成词典
```

**每一层都在补另一层的短板**：
- 大模型补几何层的「不知道哪个候选对」
- 几何层补大模型的「给不出坐标」
- 用户补两层的「都不确定时定音」

### 21.7 低缩放下的能力上限（务必对用户诚实）

一张覆盖 20 公里的截图，能提供的**信息量**就是有限的：

| 想要 | 现实 |
|---|---|
| 精确到街道 | ❌ 不可能，取决于用户补充 |
| 精确到城区/街区群 | ✅ 可以（现在做到了） |
| 轨迹形状正确 | ✅ 可以 |
| 路名归属 | ⚠️ 有意义但有限 |

所以产品上的正确做法不是追求"永远精确"，而是：
**能精确时精确，不能精确时诚实 + 给用户选择权。**

### 21.8 已实现：低置信度的 Top-N 候选（数据飞轮的入口）

基于 21.4 的结论——低缩放下定位信号是**多峰**的——实现了「换个位置」：

#### 备选的两个来源

| 来源 | 做法 | 对应用户直觉 |
|---|---|---|
| **模型层备选** | north_up / similarity / affine 各自的结果 | 「换个拟合方式看看」 |
| **候选组合备选** | 固定 north_up，让每个多候选锚点组都取**第 k 个候选**重新拟合 | 「如果『上海站』指的是另一个同名地点呢」 |

⚠️ **候选组合备选才是真正有用的那个**。模型层备选往往给出相近的解
（街道级样本上三者相差不到 200 米），而真正的分歧来自
「这个地名指的是哪个地方」——Nominatim 对「上海站」返回 5 个、
「外环高速」返回 5 个，彼此相距几公里。

实现要点（`app/register.py` 的 `_combination_alternatives`）：
- 只用**有多候选**的锚点组（单候选的组换不了）；
- 组合出的新解必须**仍然自洽**（留一验证通过、内点数达标），
  否则就是噪声；
- 备选与最优解地理上必须**相距 > 300 米**——展示一堆相距 50 米的
  候选毫无意义，用户要选的是**不同的街区**，不是同一个街区的微调；
- 备选之间也要保持 300 米以上间隔。

#### 数据流

```
register()  →  Registration.alternatives（最多 4 个）
           →  pipeline._collect_alternatives()  →  每项带完整轨迹坐标（GCJ-02）
           →  API 响应的 "alternatives" 字段
           →  前端「位置存疑？试试其他候选」卡片
           →  点「用这个」→ 前端本地重画（不需要回服务端）
```

⚠️ 备选坐标用 **GCJ-02**（不是主 GeoJSON 的 WGS-84）——
前端地图底图是腾讯地图（GCJ-02），主绘制路径也优先用 `geojson_gcj02`。
若这里输出 WGS-84，切换候选后轨迹会整体偏移约 290 米。

#### 实测效果

城市级样本：**3 个备选**，锚点组合各不相同——

| 候选 | 模型 | 内点 | 锚点 |
|---|---|---|---|
| 1 | affine | 9 | 上海西站、外环高速、上海站、静安区、徐汇区、桃浦中央绿地 |
| 2 | similarity | 7 | 上海西站、上海站、静安区、徐汇区、普陀区、黄浦区 |
| 3 | north_up | 5 | 静安区、徐汇区、普陀区、黄浦区、长宁区 |

街道级样本：**0 个备选**——正确，街道级锚点唯一，没有歧义。

#### 为什么这比"让用户填地名"好

用户**答不上**「这段轨迹的经纬度是多少」，但**答得上**「是这三条里的哪一条」。
把 3 个候选摆在地图上让他点，比让他回忆路名更快也更准。

选中之后，再用「锚点明细 → 纠正」把正确位置固化进词典，
下一次同区域的识别就能直接命中——**这就是数据飞轮**。

### 21.9 实测：用户输入路名到底有多有效（用户提问后的专项验证）

用户问：「是不是增加一个让用户输入自己走了某某路，是只有这个办法能优化这类问题」。

用三个实验量化（`tmp_debug/road_constraint.py`、`road_line_v2.py`、
`road_verify.py`）：

#### 实验一：路名作为「点锚点」——很弱

全市范围查 OSM，看每个路名对应多少段几何：

| 路名 | 段数 | 总长 | 点锚点散布 |
|---|---|---|---|
| 龙东大道 | 13 | 8.2 km | 1.7 km |
| 打浦路隧道 | 8 | 6.8 km | 1.8 km |
| 延安高架路 | 36 | 31.8 km | 7.9 km |
| 内环高架路 | 117 | 101.4 km | 11.8 km |
| **世纪大道** | **154** | 34.6 km | **62.4 km** |

「世纪大道」在浦东有 154 段（跨多个区），点锚点散布 62 公里——
**这种路名当点锚点几乎没用**。

#### 实验二：路名作为「线约束」——有效

不要把路名变成一个点，而是让**整条轨迹必须贴着这条路的几何走**。
做法：查该路名的全部路段 → 平移搜索，最小化「轨迹到该路的距离」。

城市级样本实测：

| 约束 | 落点移动 | 轨迹到该路距离 |
|---|---|---|
| 单路名「时光印记大道」 | 2.09 km | 4411 → 2402 米 |
| 多路名联合（3 条） | 2.00 km | 3953 → 1968 米 |

#### 实验三：独立校验——**确实更准了**（关键）

用**全部路网**（3268 条道路）作独立裁判：这个度量与路名约束完全无关
（路名约束只用「用户说的那条路」，校验用「所有路」）。

| 策略 | 轨迹到全路网中位距离 |
|---|---|
| A 原落点（纯 OCR 锚点） | **191 米** |
| B 单路名约束 | **98 米** |
| C 多路名约束 | **100 米** |

**结论：用户输入路名让定位精度几乎翻倍（191 → 98 米）。**

⚠️ 注意「时光印记大道」在 OSM 里只有 1 段（世博地区 OSM 覆盖不全），
这反而是**好事**——用户说的路名越具体、OSM 里越唯一，约束越强。
这提示了产品设计：**应该引导用户输入更具体的路名**。

### 21.10 结论：三类手段的分工（回答用户的核心问题）

用户问「大模型 vs 用户补充路名，哪个是办法」。答案是**分工**，不是二选一：

| 手段 | 提供的约束 | 实测效果 | 谁来做 |
|---|---|---|---|
| **用户输入路名** | **线约束**（最强，可到百米级） | 191 → 98 米 | 用户 |
| 在线地理编码密钥 | 点约束（行政区/POI 中心） | 待接 | 密钥 |
| 底图匹配（街道级） | 像素级几何 | 信噪比 20 倍 | 算法 |
| 大模型 | 语义（把路名/口述转成约束、OCR 纠错） | 不产坐标 | LLM |

**最优组合**：用户输入 1-3 个路名 → 大模型把它们规范成标准路名并
去重 → 系统用「线约束 + 全路网校验」联合优化 → 出结果。

⚠️ **为什么不能只靠路名**：
- 路名必须在 OSM 里有几何（实测「通耀路」「世博大道」有时查不到）；
- 路名唯一性差异巨大（「世纪大道」154 段 vs「打浦路隧道」8 段）；
- 用户未必记得路名（他看到的是模糊的截图，不一定能说出路名）。

⚠️ **为什么不能只靠大模型**：LLM 没有地理真值，输出坐标必然是幻觉。

⚠️ **为什么不能只靠用户填坐标**：用户答不上经纬度，但答得上路名。

### 21.11 真实案例：用户口述 10 个路名（2026-11-02）

用户提供了该截图的真实骑行路名（骑行顺序）：
小木桥路 → 龙华中路 → 船厂路 → 中山南二路 → 东安路 → 肇嘉浜路 →
虹桥路 → 水城路 → 虹古路 → 安龙新村

#### 效果

| 指标 | 无路名 | 有路名 |
|---|---|---|
| 置信度 | 0.528 | **0.740** |
| 起点（GCJ-02） | 121.47778, 31.18140 | **121.46682, 31.19185** |
| 终点（GCJ-02） | 121.38011, 31.20851 | **121.36915, 31.21896** |
| 落点移动 | — | 西北 **1.8 公里** |
| 路名归属 | 时光印记大道、打浦路隧道、通耀路、世博大道… | **中山南二路、龙华中路、小木桥路、中山南二路、东安路、肇嘉浜路**… |

路名归属与用户口述**逐段吻合**（小木桥路/龙华中路/中山南二路/东安路/肇嘉浜路全部命中），
说明落点被拉到了正确的街区群。

#### 过程中修掉的四个坑

**坑 1：Overpass 的 UTF-8 精确匹配对中文失效**

`way["name"="龙华中路"]` 普遍返回空，必须用正则 `way["name"~"^龙华中路$",i]`。
这极易被误判成「OSM 里没这条路」——实测同一批路名，
精确匹配命中 3/10，正则匹配命中 7/10。

**坑 2：逐个查 Overpass 慢且不稳**

单次查询实测 **36–60 秒**（10 个路名 = 5–10 分钟），而且**结果有波动**：
同一个「虹桥路」上次返回 73 段、下次超时返回空。主站繁忙时返回
**HTTP 504 或部分结果**。

正解：**一次拉全量具名道路，然后本地按名字过滤**。
`fetch_roads` 的查询不带 name 过滤（只按 highway/name），完全绕开 UTF-8 问题，
一次查询覆盖整个区域，本地匹配任意路名都免费。
实测命中 **9/10**（只有「安龙新村」缺失——它是小区不是路）。

**坑 3：不能把路名当「点」约束，也不能等分分段**

一条 16 公里轨迹经过 10 条路，但**不可能同时贴着这 10 条路**——
只在每条路的**一小段**上。等分分段更是错的：每条路占的弧长比例不是 1/N。
实测等分后「小木桥路」被分到的段离它 **2855 米**（而整条轨迹上离它只有 27 米）。

正解：**DP 序列对齐**。把轨迹分成 30 段，求「路名序列 → 段序列」的
单调最优匹配（类似 DTW）。对齐在初始位置跑一次，搜索时固定。

**坑 4：分段约束的置信度不能用绝对值分档**

「分段目标值」量纲是「轨迹到各路名对应段的加权中位距离」，
16 公里轨迹经过 10 条路，每条只占一小段，目标值天然在几百米量级——
它衡量的是**整体贴合程度**，不是**位置误差**。
用绝对值分档会把「明明改善 42%」判成低质量（实测 1767 米 → 置信度反降）。
正确判据是**改善率**。

#### 后续修正（用户指出「确实是东安路然后到肇嘉浜路」）

用户反馈真实轨迹是「东安路 → 肇嘉浜路」，但归属里肇嘉浜路跑到了末尾。
排查出**四个问题**：

**问题 1：DP 回溯时把首段起点强行改成 0**

`pairs[0] = (0.0, pairs[0][1])` 这个 hack 破坏了 DP 结果——第一个路名
（当时是小木桥路）被拉伸成覆盖轨迹前 **47%**，其余 9 个退化成等分。
去掉后每个路名对应自己匹配的那段。

**问题 2：DP 转移只允许紧邻（j-1），不能跳段**

现实中两条路之间可能隔着好几条没被提到的路。改为前缀最小值转移，
允许任意 `j' < j`。

**问题 3：位置先验太强（150 米/段）**

位置先验假设「路名在弧长上均匀分布」，但用户口述「东安路然后到肇嘉浜路」
说明这两条路在轨迹上**紧挨着**，而它们在 10 条路里排第 5、6 位。
先验 150 米/段时把肇嘉浜路从几何最优的段 13 推到段 16。
降到 **30 米/段**——只用于打破平局，几何证据始终主导。

**问题 4：固定权重让「唯一指出错误方向的证据」被淹没**（最关键）

10 条路名里通常有 8-9 条已经贴合（残差 100–300 米），
只有 1 条明显对不上（残差 1200 米）。用固定权重时，那条「还在指错方向」
的证据被 8 条已贴合的淹没——目标函数已经很好了，没动力再往那边挪。

改为 **IRLS 残差自适应权重**：`wᵢ ∝ uniquenessᵢ · (1 + dᵢ / median(d))`。
残差大的路名自动获得更高权重。效果：

| 指标 | 固定权重 | IRLS 自适应 |
|---|---|---|
| 分段目标值 | 3534 → 1767 米（改善 50%） | **3534 → 390 米（改善 89%）** |
| 置信度 | 0.740 | **0.830** |
| 贴路中位 | 53.8 米 | **46.3 米** |
| 归属含东安路 | ✗ | ✓ |

**问题 5：`uniqueness` 只看段数、不看散布范围**

「肇嘉浜路」在 OSM 里有 **37 段**，散布在 2 公里内；只看段数会以为它
「有 37 段所以还算具体」。加上**散布因子**后唯一性从 0.38 降到 0.13——
「船厂路」(2 段) 与「肇嘉浜路」(37 段) 的权重差从 2.7 倍拉到 6 倍，
才符合物理直觉。

#### 这张图能达到的精度极限（务必对用户诚实）

排查中量到一个硬事实：**东安路、斜土路、肇嘉浜路是三条平行的南北向路**，
在 lat 31.185–31.200 带内：

| 路名 | 经度范围 |
|---|---|
| 东安路（东） | 121.446–121.455 |
| 斜土路（中） | 121.437–121.457 |
| 肇嘉浜路（西） | 121.434–121.441 |

**三者间距只有 300–800 米**，与低缩放截图的固有定位误差（几百米到一公里）
**同量级**。

所以：即使用户口述的路名完全正确、约束完全生效，
「我骑的是东安路还是肇嘉浜路」在低缩放下**本质上无法可靠区分**——
这是信息量限制，不是算法能突破的。

突破口在**数据源**：用户口述的是美团/高德的**官方路名**，
而 OSM 是第三方数据源，命名与几何都可能不一致（见第 21.5 节
「第三方路网命名与官方命名不一致」的坑）。**高德/腾讯的在线地理编码
能直接返回官方路名的准确坐标**——这正是「配密钥」这条建议的价值所在。

#### 匹配策略（`_match_by_name`）

| 优先级 | 规则 | 例 |
|---|---|---|
| 1 | 精确匹配 | 「水城路」 |
| 2 | 去小区后缀 | 「安龙新村」→「安龙路」 |
| 3 | 去「xxx弄」支弄后缀 | 「虹桥路」而非「虹桥路996弄」 |
| 4 | 去后缀加路尾 | 「安龙新村」→「安龙路」 |
| 5 | 前缀匹配 | 「虹桥」→「虹桥路」 |

### 21.12 尚未实现但已验证可行的下一步

按性价比排序：

1. **在线地理编码密钥**（高德/腾讯，与美团底图同源，基准一致）——
   行政区中心能直接查到，低缩放定位立竿见影地提升。改动最小。
2. **街道级启用底图匹配** —— 把街道级精度从十几米提到米级
   （21.3 已验证信噪比 20 倍，是当前最有潜力的精度提升手段）。
3. **大模型做 OCR 纠错与候选重排** —— 需要接 LLM API；
   21.5 已列出具体该让 LLM 做什么、不该做什么。
4. **让用户口述路线**（「从静安寺到中山公园」）→
   大模型转成结构化约束 → 几何层直接用。
   这是唯一能把城市级截图做到街道级精度的路径。


## 22. 「轨迹跑到浦东」的根因与修复（2026-11-03 新增）

### 22.1 现象

用户反馈：**「我根本没有去浦东啊，怎么识别的路线还有浦东」**。

实测确认：识别轨迹的**起点落在黄浦江东岸（浦东侧）**，
417 个轨迹点里有 59 个（14%）在浦东。路名归属因此报出
「打浦路隧道 / 时光印记大道 / 世博大道 / 龙腾大道 / 瑞宁路 / 东岸绿道」
——全是浦东的路。

而截图上看得清清楚楚：起点在**徐汇滨江**（7 号线龙耀路一带，
「上海世博园」标注在江对岸），灰色轨迹线全程在西岸。

### 22.2 根因：不是位置错，是**尺度错**

三把**互相独立**的标尺（都与算法无关）测出真实尺度：

| 标尺 | 依据 | 尺度 (m/px) |
|---|---|---|
| 1 | 上海西站 ↔ 上海站（截图 px 474） | 13.42 |
| 2 | 上海西站 ↔ 上海南站（截图 px 1129） | 12.58 |
| 3 | 上海站   ↔ 上海南站（截图 px 996） | 12.47 |
| — | **锚点配准给出的 scale** | **15.73** |

配准偏大 **26%**。轨迹被拉长到 15739 米（真实约 12500 米），
东端因此被甩过江。

**尺度为什么被污染**：低缩放截图只有 5 个锚点，全是远距离大范围地标
（上海西站、上海站、桃浦中央绿地、黄浦区、长宁区）。
地图引擎把地标的**文字标注**放在视野空白处，与真实点位差 100-2000 米，
而且偏移**有方向性**。尺度 = 地理距离 / 像素距离，两端同向偏移就放大尺度。

实测逐对锚点隐含尺度从 **12.57 到 19.86 m/px，极差 58%** —— 完全不可信。

### 22.3 为什么之前的路名约束救不回来

路名约束（`app/roadname_constraint.py`）只允许**平移**、不许改尺度。
尺度错 26% 时，轨迹整体被拉长，无论怎么平移，
总有一段会留在错误的位置——本例就是东端 14% 落在浦东。

**这是设计缺陷**：把尺度当成「几何上更可靠的锚点产物」，
但低缩放下它恰恰是最不可靠的量。

### 22.4 试过但失败的三条路

这三条都失败在同一个地方——**残差类目标函数对尺度不敏感**：

| 思路 | 实测结果 | 为什么失败 |
|---|---|---|
| 路网 ICP（`app/simfit.py`） | 尺度从 15.73 只降到 15.51 | 低缩放下路网密集、平行路多，尺度偏 26% 时轨迹仍能贴在某条路上 |
| 路名链骨架跨度（`app/chainfit.py`） | 尺度 14.35，仍偏大 | 「接触点」随位置漂移，逐段比值从 10.2 到 26.0 波动，中位数不可靠 |
| 投影覆盖长度 | 曲线不连续 | 链式 DP 选出的实例随尺度跳变 |

**结论（重要）**：低缩放下，「轨迹贴不贴路」几乎不携带尺度信息。
任何以「贴路距离」为目标函数的优化都无法修正尺度。

这不是调参能解决的——它反映了一个几何事实：
低缩放时路网的**间距**（几百米到一公里）和尺度误差（几公里）同量级，
所以「贴上某条路」这件事对尺度的区分度极低。

### 22.4.1 起终点锚定为什么能work

因为它用了一个**完全不同的信号**：不是「轨迹和路网像不像」，
而是「起终点徽章和两端路名的**距离**对不对」。

距离是尺度的一次齐次函数——它对「像不像」免疫。
两个点相隔 712.8 像素（精确，纯色块检测），
它们之间的真实地理距离只取决于尺度，与路网密度无关。

### 22.5 修复：起终点锚定（`app/endpointfit.py`）

突破口：**起终点徽章是纯色块检测出来的像素位置**，
它没有「文字标注偏移」这个问题（地标文字标注偏移实测可达 100-2000 米）。

而用户提供的路名列表给出语义锚点：
* 起点在列表**前几条**路上
* 终点在列表**后几条**路上

于是：起终点徽章的像素距离（精确，712.8 px）
+ 两端路名的真实地理距离 -> **直接定出尺度**，与地标标注完全无关。

**算法（两阶段）**：

1. **链式 DP 选实例**（`app/roadchain.py`）
   同名路全市有多个实例（「虹桥路」149 段、「东安路」跨 lon 121.06~121.83）。
   DP 的 transition 项要求相邻路名的实例空间连贯（骑行不会瞬移 5 公里），
   一次就能把实例选对。
2. **在选定实例内枚举起终点锚点**，两点定相似变换
   （4 自由度 4 方程，唯一闭式解），用固定实例的残差打分。
3. 迭代一轮「用新变换重跑链式 DP」，让实例与变换相互收敛。

关键：**阶段 2 不重跑链式 DP**，所以整轮 < 5 秒。
朴素做法（「先枚举所有锚点对再 DP」）要几分钟，不能上线。

### 22.6 修复效果（同一张截图）

| 指标 | 修复前 | 修复后 | 真值 |
|---|---|---|---|
| 尺度 (m/px) | 15.73 | **13.11** | 12.47~13.42 |
| 轨迹长度 | 15739 m | 13020 m | ~12500 m |
| 起点 lon | 121.4774（**浦东侧**） | **121.4586**（浦西侧） | 121.452~121.470 |
| 终点 lon | 121.3798 | 121.3772 | 121.374~121.400 |
| 旋转 | — | -0.11° | ≈0（正北朝上） |

### 22.7 兜底护栏：跨江检测（`app/waterguard.py`）

即使算法再出 bug，也不该输出一条横穿黄浦江的骑行轨迹。
从 Overpass 拉主要通航大河的**岸线**，判定轨迹是否有连续 3 点落在「江那一侧」。

命中时**大幅降低置信度**并在界面上明确提示「请核对起终点是否在江的同一侧」。

注意骑行**确实可能过桥**，所以这里是警告而非硬拒。

#### 22.7.1 三个必须记住的几何事实

这三点都是实测踩出来的，写错任何一个都会让护栏完全失效：

1. **OSM 的 `waterway=river` 几何是岸线，不是河心线**。
   实测黄浦江那条线距西岸艺术中心仅 391 米，而真正的河宽约 2600 米。
   OSM 里也**没有**黄浦江的水面多边形（`natural=water` 查询返回 0 个）。

2. **岸线是斜的，法向会随弯道翻转 180 度**。
   必须用「法向的东向分量」统一符号，否则同一条河会被判成
   「有的段在浦东、有的段在浦西」。

3. **单位**：KD 树返回的距离是**度**，阈值是**米**。
   直接比较会让 100% 的点都命中（黄浦江宽 500 米 ≈ 0.005 度）。
   而且经纬度两个方向的每度米数不同，不能简单缩放欧氏距离——
   要用最近点的经纬度单独算球面距离。

#### 22.7.2 坐标系

轨迹是 GCJ-02（与美团底图同源），OSM 是 WGS-84，**必须转换后再比较**。
这个坑在测试里也踩了一次：直接把 WGS 岸线当 GCJ 传，偏移 400 多米，
判定结果完全不对。

### 22.8 新增/修改的模块

| 模块 | 职责 |
|---|---|
| `app/endpointfit.py` | **起终点锚定**（核心修复）：两点定相似变换 + 路名链打分 |
| `app/roadchain.py` | 路名实例切分 + 链式 DP 选实例 + 骨架尺度估计 |
| `app/waterguard.py` | 跨江检测护栏（岸线 + 法向侧别判定） |
| `app/simfit.py` | 路网相似变换拟合（备选思路，实测对尺度不敏感，暂未接入主流程） |
| `app/chainfit.py` | 路名链迭代求解（备选思路，实测尺度仍偏大，暂未接入主流程） |
| `app/pipeline.py` | `_apply_road_names` 改为两级：起终点锚定 -> 平移精修；加跨江护栏与置信度惩罚 |
| `app/server.py` | **后台精修必须传 road_names**（见 22.8.1） |
| `web/index.html` | 展示「起终点锚定」效果（尺度变化）与「跨江检查」警告 |
| `tests/test_endpointfit.py` | 坐标系一致性测试（防南北镜像回归） |
| `tests/test_waterguard.py` | 跨江检测测试（防单位混用、防法向翻转、防误报） |

#### 22.8.1 后台精修丢 road_names 的坑

`/api/recognize` 主流程 `refine=False`（跳过路网精修，省 40-80 秒），
然后后台线程重跑整个 pipeline 做精修。

**坑**：后台重跑时若不传 `road_names`，用户的路名约束
（以及基于它的起终点锚定）**全部丢失**，精修后的结果反而比主流程更差——
尺度会退回被污染的 15.73，轨迹东端又甩过黄浦江。

修法：`_start_refine` / `_run_refine` 增加 `road_names` 参数并透传。

**教训**：凡是「后台重跑整个 pipeline」的地方，
必须把主流程用过的所有**用户输入**都带上——
否则后台任务算出来的是另一个问题。

### 22.9 遗留问题

* **无路名输入时结果不可信（已实测确认，2026-11-03）**。
  配好高德 Web服务 key 后做了「有路名 / 无路名」对照：

  | | 置信度 | 轨迹长 | 起点 lon | 路名归属 |
  |---|---|---|---|---|
  | 无路名（纯锚点配准）| **0.203** | 7864 m（偏短 30%）| 121.449 | 常熟路/武宁南路/普雄路（**全错，静安区**）|
  | 有路名（起终点锚定）| **0.776** | 11195 m ✅ | 121.458 | 枫林路/肇嘉浜路/虹桥路/虹古路 ✅ |

  **为什么纯锚点配准救不回来**：
  截图上标注的是**文字标签的锚点像素**，高德给的是 **POI 真实点位**，
  两者有系统偏移（地图引擎的注记避让），且偏移量因标注类型而异
  （车站标签、区名标签、绿地标签各自不同）。
  实测逐对隐含尺度做中位数也偏大 18%（长基线 ≥800px 时 15.43 vs 真值 12.47~13.42），
  主要污染源是**同名 POI 匹配到远处实例**（如「1号线」匹配到 5 公里外的另一条线，
  隐含尺度 5220 m/px）。

  所以这条不是「算法不够好」，而是**信息不足**：
  城市级截图的地名标注无法单独支撑精确定位。
  现在前端在置信度 < 45% 时会明确提示「填路名能显著提高精度」。

  下一步候选：把**行政区锚点降权或排除**（区中心点与文字标注的偏移最不可控），
  以及在高德返回多候选时用**像素几何**选对实例（`_filter_by_pixel_geometry` 已有雏形）。
* `app/simfit.py` 与 `app/chainfit.py` 两条备选路线**未接入主流程**，
  保留作为后续探索的参考实现。它们的思路有价值，但实测都不如起终点锚定。

### 22.11 高德 key 的平台类型（2026-11-03 实测）

用户配了 `AMAP_KEY` 到 `.env`，但接口返回 `USERKEY_PLAT_NOMATCH`。

**原因**：高德 key 绑定**平台类型**，不能跨类型使用：

| key 类型 | 能调的接口 |
|---|---|
| Web端(JS API) | 浏览器里的 JS 地图（**本项目用不了**）|
| **Web服务** | `restapi.amap.com/v3/...`（**本项目需要这个**）|
| Android / iOS | 各自的原生 SDK |

本项目用的是 `https://restapi.amap.com/v3/place/text`（Web 服务），
所以必须申请 **Web服务** 类型的 key。错误码 10009 就是这个意思。

**教训：`.env` 配好了不等于服务可用。**
原来 `/api/health` 只检查「key 是否存在」，于是「key 配错类型」看起来
和「在线服务不可用」一模一样。现在改成：

* `Geocoder.probe_online()`——用真实端点**主动打一次**，
  把错误码和对应的修复建议一起返回；
* `/api/health` 的 `geocoding.probe` 字段直接展示结果；
* `Geocoder._amap()` 遇到失败时记录 `_last_error`，不再静默 `return []`。

顺带修掉一个隐患：`load_dotenv()` 原本只在 `app/server.py` 里调用，
**CLI 与任何不经过 server 的调用路径都读不到 `.env`**。
现在 `config.py` 在模块导入时自己加载一次。

### 22.12 换上 Web服务 key 之后的实测效果（2026-11-03）

用户换成 Web服务 类型的 key 后，效果变化非常明显：

| 指标 | 只有 Nominatim 冷启动 | 换上高德 Web服务 key |
|---|---|---|
| POI 锚点匹配率 | 1/20 | **19/20** |
| 路名归属 | 打浦路隧道/时光印记大道/世博大道（**浦东**）| 枫林路/肇嘉浜路/虹桥路/古北路/虹古路/北虹路 ✅ |
| 识别耗时 | 慢（Nominatim 限速 1 次/秒）| 21.5 秒（无路名）/ 97.7 秒（含路名 OSM 查询）|

结论：**锚点坐标的来源决定了配准质量**。
Nominatim 给的是 OSM 的要素点位，与地图引擎在截图上的文字标签位置差得远；
高德给的是 POI 的商用点位，与用户实际认知的地点一致。

### 22.13 修掉一个「我说错了」的坑：改 `.env` 必须重启

我之前跟用户说「key 属性是 `@property`，每次访问都重读环境变量，
换成 Web服务 key 后不用重启」——**这句话是错的**。

真相：`load_dotenv()` 是在**模块导入时**把 `.env` 的内容灌进 `os.environ` 的。
property 重读的是 `os.environ`，**不是 `.env` 文件**。
所以改完 `.env` 文件，服务进程里的 `os.environ` 还是旧值，必须重启。
现象是「明明改了却不生效」，非常难查。

修法（`app/config.py`）：

* 每次取值都**重新读 `.env` 文件**（按 mtime 缓存，避免重复读盘）；
* 用 `_FILE_SEEDED_VALUES` 记录「哪些 `os.environ` 的值是 load_dotenv 灌进去的」，
  从而区分「用户显式 export 的」与「只是文件里的」：

  | 情况 | 取值 |
  |---|---|
  | 用户显式 export | `os.environ` 的值（**优先级最高**）|
  | 只在 `.env` 里 | **文件当前内容**（改完立即生效）|
  | 两者都有且不同 | `os.environ` 的值（视为显式覆盖）|

* 踩到的第二个坑：`load_dotenv()` 会被调用两次（config 导入时 + server 里），
  第二次会把第一次灌进去的值误判成「用户显式设置的」，导致文件改动永不生效。
  现在用 `_FILE_SEEDED_VALUES` 判定，重复加载不再有副作用。

新增 3 个测试守住：`tests/test_geocode_config.py`
（改文件立即生效 / 显式环境变量优先 / 重复加载无副作用）。

### 22.14 用户实测反馈「还是不好」——量化了根本限制（2026-11-03）

用户拿**同一张**城市级截图重测（md5 一致），结果仍然错。
这次把根因彻底量化了。

**先确认了用户的输入**：这次**没填路名**（`road_name_fit` 为空）。
于是走的是纯锚点配准，置信度 0.203，起点落到长风公园一带，
路名归属变成常熟路/武宁南路/普雄路（静安、普陀方向）。

**为什么纯锚点配准必然失败——用正确变换量化每个锚点的偏差**：

用**已验证正确**的变换（起终点锚定后的相似变换）把每个 OCR 锚点的
像素位置反投影成地理坐标，再与该 POI 的真实坐标比较：

| 锚点 | 文字标注位置 → 真实位置 偏差 |
|---|---|
| 14号线 / 3号线 / 上海西站 | 800 – 917 米 |
| 上海站 / 12号线 / 上海南站 | 1638 – 2301 米 |
| 静安区 / 徐汇区 / 4号线内圈 | 2681 – 3225 米 |
| 1号线 / 西岸自然 | 6143 – 16696 米 |

**每个锚点的偏移都是公里级，而整条轨迹只有 11 公里长**——
锚点偏移比轨迹本身还大。信息论上就不可解。

**根因**：这张截图是**城市级缩放**（覆盖整个上海），
底图在这个级别**根本不显示街道名**。OCR 出的 38 条文本只有
行政区、地铁线、大型 POI（详见 `app/ocr_anchor.py` 的分类逻辑），
没有一条街道名——所以「填路名」这条路必须走，因为路名是**用户补充的信息**。

**实现：自动识别「缩放级别过低」**（`RidePipeline._assess_zoom_level`）

判据（两个条件都要满足，避免误判）：
1. 地理编码后**一个街道级锚点都没有**（高速公路/国道/高架不算——
   实测「外环高速」偏差 3087 米，同样是几十公里级的大对象）；
2. **大范围地标占多数**（≥45%）。

命中时：
* warnings 明确说明「底图未显示街道名，仅认出 N 个大范围地标（静安区、徐汇区…），
  它们的位置误差可达数公里，无法定位到街区」；
* 置信度**封顶 0.2**（配准再"自洽"也是自洽在错误的位置上）；
* 前端显示醒目提示，并给出两个可执行动作：
  ① 填路名 ② 截一张放大到街道级的地图。

实测对比（同一张图）：

| | 置信度 | 起点 lon | 路名归属 |
|---|---|---|---|
| 无路名 | 0.200（封顶）| 121.449（长风公园，**错**）| 常熟路/武宁南路/普雄路（**错**）|
| 有路名 | 0.776 | 121.458（龙华中路，✅）| 枫林路/肇嘉浜路/虹桥路（✅）|

新增 `tests/test_zoom_level.py`（5 项）守住误判/漏判两个方向。

### 22.15 「位置」和「形状」是两个维度（2026-11-03 晚）

用户填了 10 个路名重测（这次填的是「安龙路」而非「安龙新村」）。
**服务端输出是对的**：起点 121.4579/31.1917（高德查：小木桥路与零陵路交叉口
121.4609/31.1931，差 300 米）、终点 121.3765/31.2145（安龙路一带）、
路名归属枫林路/肇嘉浜路/虹桥路/古北路/虹古路/北虹路/威宁路、置信度 0.776。

用户仍说「效果也不行」。用**像素级客观验证**（不是肉眼看图）排除了前端 bug：

1. OCR 识别用户结果截图上的底图地名 -> 得到 38 个地名的**精确像素位置**；
2. 高德查这些地名的**真实坐标**；
3. 拟合「像素 -> 墨卡托」仿射变换（残差中位 439 米——正是文字标注偏移）；
4. 提取紫色轨迹像素（2256 个），反投影回地理坐标。

结论：**紫线与服务端坐标一致**（最近距离中位 438 米，与标定残差同量级；
紫线右端对应服务端起点 837 米、左端对应终点）。前端渲染没问题。

**真正的差异在轨迹的「形状」**：

| | 街道级样本 | 用户这张（城市级）|
|---|---|---|
| lat 方向反转 | 8 次 | 5 次 |
| 真实路线 | 天山路→水城路→长宁路→北翟路→福泉北路 | 小木桥路→龙华中路→…→安龙路 |
| 路名归属 | **全是真街道名** ✅ | 枫林路/古北路/北虹路（推测出来的）|

真实骑行是 **Z 字形**（小木桥路往南 → 中山南二路往西 → 肇嘉浜路往北
→ 虹桥路往西 → 虹古路往北），而还原出来是一条斜线，**中段偏差 1–1.5 公里**。

**根因**：城市级缩放下 11 公里的骑行只占 ~680 像素，
美团画的那条线**本身就是直线**——街道锯齿在截图阶段就丢了。
路名能修正**位置**，但修不了**形状**。

**试过「用路名重建形状」，失败**（`tmp_debug/run/exp_rebuild*.py`）：
* v1：把路名所有 segment 拼成折线 -> 「虹桥路」拼出 202 公里，弧长坐标失去意义；
* v2：用 `roadchain.select_chain` 选出的**连贯实例** -> 残差从 28266 米
  变成 33913 米，**明显变差**。

失败的原因值得记住：**DP 对齐本身不可靠**——
它把 3233 米的轨迹段分给了只有 1289 米长的肇嘉浜路。
真实路径长度 ≥ 直线距离，所以「段长 vs 路名长度」必须作为对齐约束，
否则重建只会把轨迹拉歪。

**产品改动：把「位置」和「形状」拆成两个独立信号**
（`RidePipeline._assess_zoom_level`）：

| 字段 | 含义 | 影响 |
|---|---|---|
| `zoom_too_low` | 无街道级锚点（底图没显示路名）| 提示形状不可恢复 |
| `position_ok` | 有路名约束 **或** 有街道级锚点 | 位置是否可信 |

* 置信度封顶 0.2 只在 `position_ok == False` 时生效——
  填了路名时位置已经准到 300–900 米，封顶会把正确结果说成不可信；
* 填了路名时**仍然提示**，但话术改成
  「位置已修正到街区级，但街道级走向无法从这张图恢复」。

⚠️ **上一版的设计错误**：判据里带了「没填路名」的条件，
于是用户一填路名就不再提示——那会让用户以为拿到了完整结果。
实测证明**填路名只解决位置，不解决形状**。

### 22.16 形状重建：把城市级截图里的直线掰回沿路走向（2026-11-04）

用户第三次反馈「重新失败了，效果依然没有改进」。查服务端记录：4 次上传
**md5 完全一致**——一直用的是同一张没放大的城市级截图。
既然这张图要靠算法，就只能攻「形状」。

#### 22.16.1 三个关键发现（每个都踩了坑）

**① 旧的对齐把轨迹等分成 30 段，每条路名只匹配 1 段**

`_align_segments` 把轨迹等分成 30 段（13102/30 = 437 米），
每条路名匹配**恰好 1 段**，于是
* 中山南二路（实例长 4963 米）只拿到 437 米（0.09 倍）；
* 船厂路（实例长 350 米）拿到 437 米（1.25 倍，超出）。

真实跨度差异巨大，等分 30 段太粗。**改成允许路名匹配连续区间**后，
区间长度变成：中山南二路 3822 米、船厂路 328 米——与真实骑行完全一致。

**② 区间代价只罚「超长」不罚「太短」，DP 会作弊**

DP 于是给每条路名挑**最短的、恰好贴合**的一小段，
重建后轨迹几乎没变（残差 28266 → 28103）。
加上「目标覆盖长度」项（按各路名实例长**按比例**分配总长）
才解决。

**③ 长度模型错了：像素弧长是「直线距离」，真实路径更长**

城市级截图里轨迹是直线，所以弧长 = 直线距离；
而真实骑行沿路走**有绕行**，路径**更长**。
用「首末最近点之间的距离」当几何片段 → 重建必然把轨迹**压短**
（实测 10 段里 6 段被压短 25%+，物理上不可能）。
**必须以原段长为准**取几何片段。

#### 22.16.2 最终方案（`app/shapefit.py`）

1. `roadchain.select_chain` 选**空间连贯**的实例（同名路全市多实例，
   「虹桥路」149 段）；
2. **改进版 DP 对齐**（`align_by_length`）：
   * 路名可匹配连续区间；
   * 区间代价 = 区间内距离 + **覆盖长度偏差**（双向罚）；
   * **强制连续铺满**，某路名找不到对应段时占空区间（代价 3000 米）。
     ⚠️ 上一版给「跳过」设 25 米代价，DP 全选跳过，
     10 条路名全挤到段 12（覆盖率 10%）——跳过代价必须高于几何残差；
3. **投影 + 保序**（`_project_ordered`，map matching 标准做法）；
4. **双向尝试**：OSM 的 way 有方向，骑行方向可能相反
   （实测「船厂路」最近点索引是递减的 6,6,5,5,3,2,2,2）；
5. **不动点迭代**：重建后新弧长是「路径长度」，用它重新分配再重建。

#### 22.16.3 踩过的四个 bug（都是「结果看起来对但其实没生效」）

| bug | 现象 | 根因 |
|---|---|---|
| 弧长单位不一致 | 长度约束完全失效 | 像素弧长（992）vs 路名长度（米） |
| `prefix` 只记值不记位置 | DP 沿虚假最优路径收敛到同一索引，10 段里 5 段长度为 0 | 回溯链断裂 |
| `main_axis` 不去重 | 投影塌陷 | OSM 相邻 way 共享端点，22 点里只有 13 个唯一点 |
| notes 收集时序 | 重建成功了但 warnings 里一个字都没有 | 收集发生在阶段 4.5，重建在 5.5 |

#### 22.16.4 效果（同一张城市级截图）

| 指标 | 重建前 | 重建后 |
|---|---|---|
| lat 方向反转次数 | 5（直线）| **19**（锯齿）|
| 离首尾直线距离 | 0.067 | **0.121** |
| 轨迹长度 | 13102 m | 17553 m（真实路径本该更长 ✓）|
| 逐段残差 | 9333 m | **7021 m** |
| 起终点 | — | **完全不动** ✓ |
| 跨江检测 | 未横跨 | 未横跨 ✓ |

重建后的坐标已出现 Z 字形：20–50% 段往南偏 1.4–2.2 公里，
正是中山南二路的位置（高德查该路三个实例 lat 31.179–31.195）。

#### 22.16.5 安全阀（缺一不可）

* **起终点必须不动**——它们是用户给的锚，重建动了就说明整条轨迹被平移了；
* **每段长度不压缩**——真实路径 ≥ 直线距离，压缩说明映射错位
  （单段压缩则**跳过该段**，不整体放弃）；
* **重建后重跑跨江检查**——重建让轨迹变长变弯（13102 → 17553 米），
  可能触到江边，护栏必须反映**最终**输出；
* **逐段残差**作判据，不能用「到所有路名的残差合计」——
  重建后该段贴自己的路名（残差 0）但离其他路名更远，
  合计值反而变差（28201 → 31788），会误杀正确的重建。

⚠️ **重建是「基于路名的推断」，不是从截图直接读出的**。
原始轨迹保留在 `road_name_fit.shape_fit.original`，置信度**不上浮**
（宁可保守，不要让用户过度信任）。

### 22.17 借助大模型？——先做实测，再定分工（2026-11-04）

用户提问：「不要只局限于用代码实现，是不是可以借助大模型呢」。
这是个方向性问题，答案是「能，但不在坐标层」。以下都是**实测**出来的。

#### 22.17.1 先测外部路径规划 API：不行

高德有 Web 服务路径规划，本以为它能直接给出「沿真实道路的几何」，
从而一劳永逸解决形状问题。实测（`/v3/direction/walking`）：

* **骑行（cycling）服务该 key 未开通**：`SERVICE_NOT_AVAILABLE`；
* **步行（walking）可用**，但返回结构是 `route.paths`（新版），
  不是 `routes`——我一开始读 `routes`，误判成「没有路径」；
* **全程规划的结果几乎不是用户的路线**：

  | 路径走的路 | 用户填的路名 |
  |---|---|
  | 枫林路 → 肇嘉浜路辅路 → 高安路 → 建国西路 → 广元路 → 华山路 → 淮海西路 → 法华镇路 → 延安西路 → 天山路 → 北虹路辅路 | 小木桥路 → 龙华中路 → 船厂路 → 中山南二路 → 东安路 → 肇嘉浜路 → 虹桥路 → 水城路 → 虹古路 → 安龙路 |

  14 步里**只命中 1 步**（肇嘉浜路辅路）。而且 lat 方向反转 **0 次**
  （比直线的 5 次还少）——它是「先西北再正西」的简单折线。
* 改用「路名接触点」做途经点重试，**结果完全一样**（距离都是 9677 米）
  ——说明步行 API 把偏离最优走廊的途经点**静默丢弃**了。

**原因很清楚**：高德给的是「**步行最短路**」，而用户实际骑的是有小区便道、
有特定顺序的路线。服务不知道用户走了哪条路。

逐对连接相邻路名倒是**部分可用**（6/9 成功）：

| 连接 | 直线 | 路径 | 经由 |
|---|---|---|---|
| 小木桥路 → 龙华中路 | 804 m | 1028 m | 零陵路 ✓ |
| 龙华中路 → 船厂路 | 103 m | 154 m | 零陵路 ✓ |
| 船厂路 → 中山南二路 | 478 m | 574 m | 零陵路、宛平南路 ✓ |
| 虹桥路 → 水城路 | 56 m | 695 m | 虹古路 ✓ |
| 水城路 → 虹古路 | 649 m | 872 m | 虹古路、北虹路、仙霞路 ✓ |
| 虹古路 → 安龙路 | 344 m | 467 m | 仙霞路 ✓ |

失败的 3 对全是长段（东安路→肇嘉浜路 2206 m、肇嘉浜路→虹桥路 5282 m）
——步行路网在这些跨度上不连通。

#### 22.17.2 结论：大模型该放在「语义层」

| | 谁能做 | 为什么 |
|---|---|---|
| 精确坐标 | 代码 + OSM + 地理编码 | 大模型没有 OSM 级别的几何 |
| 用户的**实际**路线 | **只有用户本人** | 外部服务给的是「最优路」（实测 14 步命中 1 步） |
| 路名序列是否合理、顺序对不对、哪条记错了 | **大模型** | 这是语义推理，几何算不出来 |
| 截图的缩放级别、底图上还有什么 | **多模态大模型** | OCR 只能读文字，读不出图标与布局 |

**最强的信号其实是用户自己给的路名序列**——它精确描述了用户走的路，
比任何服务都准。外部服务只能**辅助**，不能替代。

#### 22.17.3 实现（可选增强，不配就完全不变）

* `app/llm.py`：极简 OpenAI 兼容客户端（只依赖 requests），支持多模态
  （本地图片转 data URL）；
* `app/llm_geo.py`：
  * `check_road_sequence()` —— 用地理常识校验路名序列；
  * `review_zoom_level()` —— 多模态复核缩放级别与漏读地标；
* `app/pipeline.py`：`_llm_review_road_names()` 在**起终点锚定之前**
  调用（重排后的顺序会被后续所有步骤采用）；
* 配置项全可选：`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`，
  不配置就**完全跳过**，现有行为一字不变。

#### 22.17.4 安全设计：LLM 的建议必须通过几何验证

模型也会判断错，所以**它只给建议**：

```
采纳新顺序  ⟺  几何代价(new) < 几何代价(old)
```

几何代价 =「各路名到轨迹的最近距离之和」（便宜、够用）。
模型判断错时，几何证据会把它挡下来。

另外三道防线：
* 模型**只能重排**，擅自新增/改写路名直接拒绝（元素集合必须完全一致）；
* 调用失败 / 返回非 JSON / 超时 → 序列不变，主流程继续；
* `warnings`（可能记错的路名）只提示，**不自动修改**。

新增 `tests/test_llm_geo.py`（6 项）守住这些。

### 22.18 OSM 有路径规划，而且比高德好用得多（2026-11-04）

用户追问：「不能用高德的规划 API，OSM 有类似的功能吗」。有，而且实测差距巨大。

**OSM 生态的路由引擎**：OSRM（最常用，官方 Demo 免费无 key）、
GraphHopper、Valhalla、BRouter。本地自建需要 Docker + 区域 PBF，
本项目环境不具备（无 Docker + 磁盘只剩 19G），所以直接用官方 Demo。

#### 22.18.1 实测对比（同一张城市级截图，同样的起点/终点/途经点）

| | 高德 walking | OSRM driving |
|---|---|---|
| 命中用户路名 | **1 / 14 步** | **15 / 51 步** |
| 途经点 | **静默丢弃**（结果与不传完全一样）| **生效** |
| 命中的路 | 只有肇嘉浜路辅路 | **中山南二路、东安路、肇嘉浜路、虹桥路、水城路、虹古路**（6 条）|
| lat 方向反转 | 0 次（比直线还少）| 21 次 |

OSRM 走的是「枫林路 → **中山南二路** → 宛平南路 → 零陵路 → **东安路** →
斜土路 → 双峰北路 → **肇嘉浜路** → **虹桥路** → 古北路 → **水城路** →
**虹古路** → 北虹路 → 仙霞路」——**确实就是用户那条路线**。

三个关键差异：
1. **途经点生效**。高德会把偏离最优走廊的途经点直接丢掉（实测改用接触点
   重试，距离仍是 9677 米，一模一样）；OSRM 会照着走。
2. **路网更完整**。OSM 里有 `bicycle=yes` 的小区内部道路，高德 walking
   只有人行可达的边。
3. **返回结构不同**。高德新版是 `route.paths`（不是 `routes`！）——
   我一开始读 `routes`，误判成「没有路径」。

#### 22.18.2 但 OSRM 的几何**不能**直接当轨迹

OSRM 路线长度 **24.7 公里**，而真实骑行约 12 公里——**偏长 2 倍**。
原因：Demo 只有 **driving profile**（bike/foot 退化成 car，骑车走的
非机动车道它不走），加上途经点有几百米偏移，绕路就多。

所以分工是：

* **形状**用几何重建（`app/shapefit.py`）：逐段精确、里程正确；
* **OSRM 只做交叉验证**——它只知道起终点，**完全不知道用户填了哪些路名**，
  它算出的路线能命中 5/10 条，说明用户的记忆与真实路网结构吻合。
  这是一份**独立于本项目算法**的第三方证据。

实测输出：「外部路网校验：OSRM 独立规划出的路线也经过中山南二路、
水城路、虹古路、虹桥路（5/10 条路名）」。

#### 22.18.3 踩的四个坑

| 坑 | 现象 | 根因 |
|---|---|---|
| 缓存永远不命中 | 端到端耗时从 150 秒涨到 **528 秒** | 用内置 `hash()` 做缓存键——Python 对 str 的 hash 受 PYTHONHASHSEED 随机化影响，**每次进程启动路径都不同**。改用 md5 |
| 缓存路径格式化错 | `TypeError: not all arguments converted` | `"osrm_%s.json" % abs(hash(key)) % (10**12)` 的 `%` 从左到右，先格式化再取模 |
| 几何长度暴涨 9 倍 | 17553 → 156377 米 | leg 级 geometry 缺失时的兜底切分**没有累积偏移**，每个 leg 都从几何开头切一段，累加后爆炸。教训：**切分必须带累积偏移**；现在直接用顶层 `route.geometry` |
| 对齐时把轨迹压短 | 17553 → 14998 米 | 用带缩放的相似变换对齐 OSRM 几何。物理约束：路网路径长度 ≥ 直线距离，缩放必然压短。改成**刚性变换**（只旋转+平移） |

#### 22.18.4 顺带修掉的两个「静默失败」

* `fit_shape` 的安全阀用**最后一轮**的残差比较，而不是**最好一轮**——
  不动点迭代里中间某轮可能更好，用最后一轮会误杀正确的重建；
* `_rebuild_shape` 只把含「跳过/未改善/放弃」的 note 传给 warnings，
  OSRM 失败原因（「未采用」）被**静默吞掉**，排查时完全看不到发生了什么；
  而且 note 收集的时序还早于 OSRM 调用——两个问题叠加，
  表现为「OSRM 明明能跑，却完全没有动静」。

### 22.19 OSM 除了 driving 还有别的模式——Valhalla 原生支持 bicycle/pedestrian

用户追问：「OSM 除了 driving profile，还能有其它模式吗」。有，而且差别很大。

#### 22.19.1 OSRM Demo 其实**只有** driving

实测传 7 个 profile 名（driving / bike / foot / motorcycle / truck /
cycling / bicycle），返回结果**完全一样**：11064 米、同样的路名序列。
Demo 服务器只 `.osrm` 编译了一份 driving 的图，URL 里的 profile 参数
被直接忽略。

所以「用 OSRM 的 bike profile」这条路在公共 Demo 上走不通。

#### 22.19.2 BRouter 有 bike profile，但**不返回路名**

| profile | 里程 | 几何点 | 路名 |
|---|---|---|---|
| trekking | 10.25 km | 154 | **空** |
| fastbike | 10.25 km | 154 | 空 |
| safety | 10.31 km | 155 | 空 |
| shortest | 9.70 km | 179 | 空 |
| moped | 10.54 km | 139 | 空 |

GeoJSON 的 `properties` 里只有 `track-length` / `total-time` / `cost`，
**没有 `legs`**——拿不到路名就没法做「命中用户路名」的交叉验证。

#### 22.19.3 Valhalla 是最优解

公共实例 `valhalla1.openstreetmap.de`，免费无需 key，
原生支持 `bicycle` / `pedestrian` / `auto` 三种 costing。

**实测对比（同一张城市级截图，同样的起点/终点/途经点）**：

| 方案 | 里程 | 命中用户路名 | 顺序一致 | lat 反转 | 几何点 |
|---|---|---|---|---|---|
| **Valhalla pedestrian** | **12.63 km** | **7/10** | **100%** | 27 | 269 |
| Valhalla auto | 16.46 km | 7/10 | 100% | 23 | 306 |
| OSRM driving | 16.72 km | 7/10 | 100% | - | - |
| Valhalla bicycle | 17.26 km | 4/10 | 100% | 19 | 336 |
| 高德 walking | 9.68 km | 1/14 | - | 0 | - |

两个关键观察：

1. **顺序一致率比命中数更有价值**。Valhalla 只拿到起终点（和途经点），
   **完全不知道用户填了哪些路名**，它自己算出的路线依次经过
   「中山南二路 → 东安路 → 肇嘉浜路 → 虹桥路 → 水城路 → 虹古路 →
   安龙路」，与用户记忆的顺序**完全吻合**。这是独立于本项目算法的
   第三方印证。

2. **`bicycle` 反而不如 `pedestrian`**。上海路网里自行车常被引到高架
   下面的机动车道，或因 `use_roads` 权重绕远——命中从 7/10 掉到 4/10、
   里程从 12.63 涨到 17.26 公里。`pedestrian` 走的是人行道+小路网，
   反而更贴近骑行者的实际路径。

#### 22.19.4 但**不能**用 Valhalla 的几何当轨迹

Valhalla 里程 10.5–12.6 公里，而几何重建是 17.5 公里。谁对？

**独立标尺**：用户给的 10 条路名，链式 DP 选出的实例主轴长度之和是
**17308 米**——真实骑行必然接近这个量级（沿路走，路径 ≥ 直线距离，
且不可能超过这些路的总长）。

| 方案 | 里程 | 与 17308 米的关系 |
|---|---|---|
| 直线轨迹（截图原始） | 13103 m | 偏短 |
| **几何重建** | **17552 m** | **吻合（+1.4%）** |
| Valhalla pedestrian | 10510 m | **偏短 39%** |

原因：Valhalla **跳过了起点那一片**——小木桥路 / 龙华中路 / 船厂路 /
中山南二路四条路的实例长度合计 8712 米，Valhalla 一条都没走到
（它的路线直接从起点跳到东安路）。所以它偏短不是「更准」，
而是「漏了一大段」。

结论不变：**形状用几何重建，Valhalla 只做交叉验证**。

#### 22.19.5 Valhalla 逐对连接 9/9 成功

把「相邻两条路名之间的过渡段」单独规划（而不是一次给 10 个途经点），
**9 对全部成功**，且路径长度都 ≥ 直线距离（物理正确）：

```
小木桥路 → 龙华中路   直线 696 米 | 路径  700 米 | 东安路、零陵路
船厂路   → 中山南二路 直线 1562 米 | 路径 1912 米 | 零陵路、漕溪北路
东安路   → 肇嘉浜路   直线 1920 米 | 路径 2979 米 | 宛平南路、肇嘉浜路
虹古路   → 安龙路     直线 294 米 | 路径  372 米 | 北虹路
```

这条能力留着做「段间连接」，但**暂未接入**——因为接入前要先解决
「哪些段该用连接段」的判定，目前逐段残差已足够好。

#### 22.19.6 踩的四个坑

| 坑 | 现象 | 根因 |
|---|---|---|
| `requests` 的 `params` 传 JSON | Valhalla 返回 `400 Failed to parse json request` | requests 会把 `{` `"` 编码成 `%7B` `%22`，Valhalla 拒绝。必须自己 `urllib.parse.quote` 拼 URL |
| location 数量超限 | HTTP 400 `error_code:150 Exceeded max locations: 10` | 公共实例最多 10 个 location。超了直接截断（首尾保留、中间均匀采样） |
| 途经点用了重建后的段末点 | 命中从 7/10 掉到 3/10 | 那批点已经变形了，会让引擎走完全不同的路线。必须用**路名接触点**（`chain.contact_geo`） |
| 接触点含 NaN | 请求直接失败，且无任何提示 | `contact_points` 在实例为空时写入 `(nan, nan)`。必须过滤；且失败时必须留 note |

#### 22.19.7 实现

* `app/valhalla.py` —— Valhalla 客户端（encoded polyline 解码、md5 稳定
  缓存键、location 上限截断）、`order_consistency()` 顺序一致性判定；
* `app/pipeline.py` —— `_external_verify()` 替换 `_osrm_verify()`，
  依次试 `pedestrian` / `auto` costing，取命中多 + 顺序一致的；
* 前端新增「外部路网校验」行，显示引擎名、命中路名、顺序一致率。

实测输出：「外部路网校验：Valhalla pedestrian 独立规划出的路线也经过
东安路、肇嘉浜路、虹桥路、水城路、虹古路（5/10 条路名，顺序一致 100%）」。

新增 `tests/test_valhalla.py`（14 项，不依赖网络）。

### 22.20 路线构建：把「掰弯直线」改成「按路名串路」（2026-11-04）

用户第三次反馈「还是不行」。这次用**客观几何判定**而非肉眼看图，
发现上一轮的方案（`app/shapefit.py` 把直线掰弯到路名上）**根本没生效**。

#### 22.20.1 先证明「渲染」是对的

服务端输出：起点 121.4579/31.1917（龙华中路）、终点 121.3765/31.2145（安龙路）。
但用户截图上起点画在长风公园、终点画在内环高架路——差 4-5 公里。

客观验证（不靠肉眼）：
1. OCR 用户结果截图上的腾讯底图地名 -> 40 个地名的精确像素位置
2. 高德查这些地名的真实坐标
3. 拟合「像素 -> 墨卡托」north_up 变换
4. **把服务端坐标反算成像素**

结果：服务端起点反算到 (633,347)、终点反算到 (195,205)；
截图上的红色标记在 (609,331)、绿色「终」标记在 (185,193)——
**完全吻合**。前端渲染没问题，服务端坐标本身也是对的。

（顺带排除一个坑：腾讯地图的「2 公里」比例尺，OCR 框宽度不是标尺线长，
我据此算出的 48.8 米/像素是错的；实际约 19.3 米/像素。）

#### 22.20.2 但「形状」是假的

用几何判定（轨迹点到各路名 OSM 几何的距离，60 米内算「经过」）：

| 路名 | 掰弯后接触点数 |
|---|---|
| 小木桥路 | **0**（完全没经过）|
| 肇嘉浜路 | **0**（完全没经过）|
| 中山南二路 | 131 |

**中间有 5.8 公里完全没贴任何路**——那还是原始直线。

⚠️ **旧指标「逐段残差 9333 -> 7021」与「覆盖 99%」都是假的**：
DP 强制把弧长铺满给每条路，所以覆盖率必然接近 100%，
它衡量的是「分配了多少弧长」，**不是「真的贴在路上」**。
这个教训值得记住：**指标必须能区分「真做对」和「假装做对」**。

#### 22.20.3 新方案：按路名串路

用户的路名列表本身就是**路线描述**——「从小木桥路出发，经龙华中路、
船厂路、中山南二路……到安龙路」。直接把它串成几何（`app/routebuild.py`）：

1. 链式 DP 选空间连贯的实例
2. 相邻两条路求**路口**
3. 每条路的「进入点」= 与上一条路的路口，「离开点」= 与下一条路的路口
4. 按主轴方向裁出 entry->exit 的子弧，串成完整路线
5. **枚举每条路的正/反方向**（2^10 = 1024 种），取「接触顺序与用户顺序最一致」的

#### 22.20.4 三个关键的坑

**坑 1：链式 DP 选错了实例**（`app/roadchain.py` 加 `length_penalty`）

DP 原本只看「实例到轨迹的最近距离」+「相邻实例间距离」，
结果选中了**最长**的实例：

```
中山南二路 4 个实例：
  121.45649,31.19608 -> 121.46785,31.20083  长 1413 米  离起点 590 米  ← 正确
  121.43160,31.18164 -> 121.42761,31.18348  长 4963 米  离起点 2676 米 ← 被误选
```

4963 米那条在西南 2.7 公里外，但因为「离相邻实例更近」而胜出——
结果「船厂路→中山南二路」「中山南二路→东安路」「东安路→肇嘉浜路」
三条接缝全是 1.5-1.8 公里的**假连接**（那两条路根本不相邻）。

修法：对长实例加惩罚（`length_penalty=0.5`），DP 就会优先选
「离轨迹近」的短实例。

**坑 2：路口要用 OSM 共同端点，不能用几何最近点**

OSM 的相邻 way **共享端点节点**，两条路在同一个路口相接时，
way 端点坐标**完全相同**。用共同端点判据，间距是 0，绝对可靠：

```
中山南二路 <-> 东安路   共同端点 3 个
东安路   <-> 肇嘉浜路   共同端点 2 个
水城路   <-> 虹古路     共同端点 1 个
```

**坑 3：接缝插入的下标必须用「原始路线」的**

最初边插边改锚点，导致后面用的下标是「已插入后的」，
而 `out` 长度也在变——实测区间下标飙到 `[271..272]`，
远超路线长度 129，插入后长度从 14905 暴涨到 **56047 米**。

正确做法：先把每个接缝在**原路线**上的区间算好，
再从后往前执行（后面的插入不影响前面的下标）。

另外 2 点路径时 `path[1:-1]` 是空的——那恰恰是「直线跳跃」，
是最需要替换的情况，之前直接 `continue` 导致所有接缝都被跳过。

#### 22.20.5 接缝用 Valhalla 的真实路径填充

相邻两条路之间是直线跳跃，但真实骑行必然经过别的路。
用 Valhalla 逐对规划（实测 9/9 成功，路径长度都 >= 直线距离）：

```
船厂路 -> 中山南二路  直线 2053 米 | 真实路径 2202 米
                     经由 零陵路、大木桥路、斜土路、合宝路
中山南二路 -> 东安路  直线 2550 米 | 真实路径 2786 米
虹桥路 -> 水城路      直线 3403 米 | 真实路径 3666 米
```

#### 22.20.6 最终效果

| | 掰弯直线（shapefit）| **按路名串路** |
|---|---|---|
| 路名经过数 | 8/10 | **10/10**（每条最小距离 0 米）|
| 顺序一致率 | — | **100%**（9/9 递增）|
| 里程 | 17553 m | **18843 m** |
| 独立标尺（10 条路 OSM 实例长之和）| 17308 m | 偏差 **+9%** |
| 端点偏移 | 0 m | 236 m / 188 m |

新增模块：
* `app/routebuild.py` —— 串路（共同端点定路口 + 方向枚举）
* `app/seampath.py` —— 接缝插入（下标定位 + 长度守门）
* `app/roadchain.py` —— 加 `length_penalty` 参数
* `app/pipeline.py` —— `_build_route()` 替换 `_rebuild_shape()`

新增 `tests/test_routebuild.py`（11 项）。

### 22.21 多图拼接：从「猜形状」到「量形状」（2026-11-05）

#### 22.21.1 问题从哪来

22.15 立下了一个判断：**「位置」和「形状」是两个维度**。

| | 城市级单图 | 街道级单图 |
|---|---|---|
| 位置 | ✅ 路名可修正到 300-900 米 | ❌ 尺度算成 16 m/px（差 8 倍）|
| 形状 | ❌ 锯齿在截图阶段就丢了 | ✅ 锯齿清晰可见 |
| 结论 | 位置能修，形状修不了 | 形状能看，定位失准 |

22.16-22.20 用「路名串路」从**路网几何**反推形状，把形状误差压到 +9%。
但那是**猜**出来的——靠路名约束在真实路网里猜一条合理的走法，
终究猜不出「当时到底拐在哪条街的哪个口」。

于是用户提出一个直击本质的问题：

> 我如果放到街道级的截图，一张图片截不下，怎么办呢

**11 公里的骑行，街道级截图确实盖不下。**
但这恰好说明：**只要能拍到，形状就不用猜了**。

#### 22.21.2 方案：沿路线滑动截图，拼成一张大图

用户沿路线滑动地图连续截 4-6 张有重叠的街道级图，
服务端用特征匹配求出图间位移、拼成一张大图，再走现有单图流程。

拼成大图顺带解决了**尺度**问题——
大图上的 POI 标注铺开在数公里范围内，
而单张街道级图的锚点挤在几百米内，尺度极不稳定。

#### 22.21.3 关键实验：三个否定结论 + 一个肯定结论

用**可复现的实验**决定方案，而不是猜。

**✅ 结论一：SIFT 拼接可行，且精度极高**

| 指标 | 实测值 |
|---|---|
| 双向匹配的位移互逆性 | tx=932 / -932（完全互逆）|
| 估计尺度 | 1.000002（真值 1）|
| 估计旋转 | -0.0006°（真值 0）|
| 与经纬度差的自洽性 | 0.005° 经度 = 476 米 = 932 像素 ✅ |

**✅ 结论二：静态 UI 不会干扰**（但差点翻车）

App 的状态栏、右侧按钮、底部数据卡片在每张截图里**位置完全相同**，
它们提供大量「位移=0」的对应。而**单位变换本身也是合法的相似变换**，
RANSAC 的「内点最多」准则会被 UI 骗过去。

实测：UI 对应 58 对 vs 地图内点 64 个——**只差 6 个，UI 一多就翻车**。

解法不靠猜 App 布局，而靠一个事实：
**静止 UI 在两张图的同一像素位置内容完全相同**，
而地图内容即使重叠、在同一像素位置也不同（那是不同的世界位置）。
所以「同位置内容相同」的区域就是 UI，直接从特征匹配里剔除。
实测带 UI 时位移仍是 933px（真值 932）。

**❌ 结论三：美团底图与高德底图不能互为参考**

盲搜 45 个「中心点 × 缩放级别」组合，最佳解的尺度算成 **0**（纯噪声）。
与 21.2 的结论**看似矛盾**——那次说「底图同源、可对齐到像素」。

区别在于：21.2 用的是**高德瓦片**（有真实数据、有真实道路骨架），
这次用的是**高德静态图 API**（渲染风格与美团 App 不同，注记布局也不同）。
⇒ **能当参考的只有用户自己那几张截图**。

**❌ 结论四：街道级单图确实不可用**

实测街道级样本：尺度 16.14 m/px、2 内点 / 46 离群、里程算出 **1898 万公里**。
⇒ 街道级必须**多图**，单图无解。

#### 22.21.4 实现

新增 `app/multishot.py`：

1. 逐对 SIFT 匹配 + RANSAC 拟合相似变换；
2. **剔除静止 UI 区域**（见 22.21.3 结论二）；
3. 并查集找最大连通块（不重叠的图会被排除并明确报告）；
4. BFS 求每张图到画布的变换；
5. 按 BFS 深度从小到大贴图，同一像素只留深度最小的——
   不做像素级混合，避免重影。

新增 `POST /api/recognize_multi`（多文件），`/api/recognize` 保持不变。

#### 22.21.5 踩过的坑：`warpPerspective` 的矩阵方向

**这是本次最隐蔽的 bug**，而且用户反馈「不知道拼得对不对」正是它导致的。

`cv2.warpPerspective(src, M, dsize)` 默认把 M 当作 **src→dst** 直接使用。
我原先传的是 `inv`（T⁻¹），导致**内容整体错位**：

```
实测：A 图左上角的黑标（src x=25）应落在画布 x=491，
      实际落在 x=983 —— 差 507 像素。
```

为什么难发现：画面「看起来」还行（重叠度高，肉眼难辨），
但**接缝处其实没对上**，轨迹叠加会完全错位。

同一处错误还污染了 `_ncc()`——它当时传 `inv` 却注释说「传 M⁻¹ 正好」，
所以那个「NCC 0.99」是**碰巧高**。

**验证方法（可复现）**：

```python
src = np.zeros((100,100,3), np.uint8); src[10:40,10:40] = (255,0,0)
T = np.array([[1,0,200],[0,1,0],[0,0,1]], float)   # src 的 x=25 应落到 dst 的 x=225
cv2.warpPerspective(src, T, (400,100))
# → 标记落在 x=210 ✅（传 T 正确）
```

#### 22.21.6 关键设计：像素级复核，回答「拼得对不对」

用户原话：「**我也不知道你拼的对不对**」。

光看接口返回的数字不直观，必须给**像素级**的客观判据。
`_verify_canvas()` 把每张图按算出的变换贴回画布，
与画布内容**逐像素比对**（剔除互相覆盖的重叠区）：

| | 第 1 张 | 第 2 张 | 第 3 张 | 中位数 |
|---|---|---|---|---|
| 复核相似度 | 1.0 | 0.9515 | 0.9395 | **0.9515** |

前端显示「拼接复核：通过（像素相似度 0.952），接缝对得上」，
低于 0.9 判「可疑：接缝可能没对齐」。

这个复核还有个更重要的作用：**自动抓住方向性错误**——
方向反了相似度会立刻掉下来，不需要靠肉眼。

另外前端在拼接图上用**黄色虚线框**标出每张原图的位置并编号，
框应当首尾相接、彼此略有重叠；哪个框飘到一边就是那张没对上图。

#### 22.21.7 最终效果

真值已知的合成数据（起点 121.459265,31.192059 → 终点 121.45643,31.191969，
长 307 米）：

| | 起误差 | 终误差 | 里程 | 偏差 |
|---|---|---|---|---|
| **多图拼接（3 张）** | **13 米** | **21 米** | **294 米** | **4.0%** |
| 单图（只截中间那段） | — | — | 931 米 | 204% |

对比城市级单图：位置误差 300-900 米、形状偏差 1-1.5 公里。

多图识别出的路名是「零陵路、小木桥路、中山南二路、枫林路」——
与用户真实路线吻合。

#### 22.21.8 新增测试

* `tests/test_multishot.py`（8 项）——含**方向回归测试**：
  黑标落点偏移必须等于位置框报告的偏移（容差 20px）。
  方向反了会差 500+ 像素，立刻暴露。
* `tests/test_multishot_e2e.py`——端到端**米级校验**（不看肉眼）。
* 前端测试 38 → **42 项**（复核分显示、复核分偏低判可疑）。

⚠️ **测试方法本身的坑**：用「精确颜色相等」找标记会被 JPEG 压缩坑到
（边缘混色，只命中中心几个像素）→ 改用黑块 + 灰度阈值 + 连通域。
另外标记可能被另一张图**覆盖**（正常的覆盖关系，不是 bug），
测试要区分「落点偏移不符」与「落点处没有标记」两种情况。

#### 22.21.9 用户操作指引

1. 在 App 里把地图放大到**能看到街道名**；
2. 沿路线滑动连续截图，**相邻两张留约 1/3 重叠**；
3. 一起选中上传（前端已支持多选）；
4. 看「拼接复核」分数与虚线框位置，确认拼对了再看轨迹。

滑动截图中途的那张**没有**起终点徽章是正常的
（只有第一张有「起」、最后一张有「终」），拼接后会自动对上。

#### 22.21.10 实测：重叠到底要多少？（用户提问后补做）

用户问「如果没保证相邻两张有 1/3 重叠呢」。实测结论：

| 重叠 | 内点 | NCC | 位移误差 |
|---|---|---|---|
| 25% | 278 | 1.0000 | 0.0 px |
| 15% | 159 | 1.0000 | 0.0 px |
| 10% | 89 | 1.0000 | 0.0 px |
| 5% | 36 | 1.0000 | 0.0 px |
| **3%** | **16** | **1.0000** | **0.0 px** |

**街道级下 3% 重叠（约 24 像素）就能精确配准。**
1/3 是保守建议，不是硬门槛——街道级底图的特征足够密集。

（测试：`tests/test_overlap_need.py`，脚本：`tools/exp_overlap_need.py`）

##### 实验设计上的三个坑（都踩过）

**坑一：不能用「同一张图切片」当两张截图**

切片出来的两张图重叠区**像素级完全相同**，
SIFT 反而分不清「同一位置」和「相似纹理」：
匹配对只有 225 对（独立渲染时 1000+），位移分布弥散（标准差 153px），
完全配不出真值。

⇒ 真实场景用户在 App 里滑动会**重新渲染**（有细微差异），
实验必须模拟这一点。

**坑二：不能用城市级底图测**

城市级只有行政区/地铁线/大 POI，**结构高度重复**：
实测任意两个视角的匹配位移中位数都是 0、离散度 200-500px —— 全是误匹配。
（与 §21.4「城市级底图匹配不可行」一致。）

**坑三：切片时算错了真实重叠宽度**

`a = band[:, 0:bw-dx]`、`b = band[:, dx:bw]` 的**真实重叠宽度是 `bw - 2*dx`**。
按这个公式，标称 30% 那档实际是**负重叠**——于是整组数据全是「配不上」，
差点得出「需要 60% 重叠」的错误结论。

正确做法：两片都取宽度 W，第 2 片右移 dx ⇒ 真实重叠 = W - dx。

#### 22.21.11 顺带修掉：底图重复纹理让 RANSAC 选错解

上海街道网格规则，底图里有大量**周期性重复结构**。实测（两图重叠 240px）：

* 真解 `dx=-560` → **393** 个匹配
* 假解 `dx=+449` → **509** 个匹配（更多！）

RANSAC 选「内点最多」的，于是选了假解——**位移误差 1024 像素**，
完全拼错。

⚠️ **光靠 NCC 也救不了**：用假解算出的 NCC 是 **0.9987**——
因为 NCC 是拿「RANSAC 选出的变换」去比，数学上自洽、语义上却错。

**正确做法**（`app/multishot.py:_pick_best_shift`）：
从**位移直方图**里找出几个候选峰，逐个算 NCC，取最高的。
实测真解 NCC=1.0000、假解 NCC=0.1866 —— 区分度极高。

另外加了两道防线：
* `_is_valid_similarity()`：拒绝退化矩阵
  （`estimateAffinePartial2D` 偶尔返回第一行全零的垃圾矩阵，
  **不报错**，而且会让所有匹配都变成「内点」——实测 617 个，
  内点数完全失去筛选作用）；
* 位移上限检查：超过两张图较短边的 75% 视为配到别处。

⚠️ 这类错误**不会报错、只会静默拼错**，
所以必须靠「已知真值 + 位移误差」来验，不能只看接口返回 200。

#### 22.21.12 用户实测踩到：底图色块被当成轨迹（2026-11-06）

用户拿 4 张**腾讯淡色底图**的截图测多图拼接，反馈「中间一大块绿色三角形」。

**拼接本身完全正确**（4 张严丝合缝、复核分 1.0），
问题出在轨迹提取：那块绿色是**腾讯底图里的绿地（公园）**。

| 现象 | 实测 |
|---|---|
| 绿地色块 | 色相 68、饱和度高、面积 1692 像素 |
| 轨迹掩膜取值区间 | 「低饱和 + 中等亮度」——绿地正好落在里面 |
| 结果 | 骨架化后变成实心三角形，还被当成轨迹画到地图上 |

⚠️ **腾讯淡色底图比高德底图更容易误判**：
高德底图饱和度高（绿地被饱和度条件排除），腾讯淡色模式的绿地饱和度低，
直接落进轨迹线的取值区间。

**排查中确认的两个事实**（都写进了测试）：

1. 那 4 张截图**里根本没有骑行轨迹线**——紫蓝像素 0；
   竖长条形状的「紫色轨迹」实际是**内环高架的紫色道路**（带白描边、双线）。
2. 腾讯底图的骑行路线是**按需渲染**的——
   只有骑行详情页才画，单纯在地图上浏览平移时**不画**。

⇒ 用户看到的紫色轨迹（上半部分那张腾讯地图）来自骑行详情页，
而他上传的 4 张是**浏览地图**时截的。

**修复**（`app/track.py`：形状合理性检查）：

骑行轨迹线的本质特征有两条，用它们拦住误判：

| 判据 | 误判样本 | 真实样本 |
|---|---|---|
| 不占满整幅图宽度 | 连通域宽 1150~1260 px（图宽 1260）| 只占一小部分 |
| 细长而非团块 | 骨架化后填充率高 | 填充率低、长宽比大 |

命中即**放弃提取并明确报错**：

> 轨迹形状不合理：连通域宽 3242px ≈ 整幅图宽（3353px）——
> 多半是底图色块（绿地/水面/建筑/文字）被误当成轨迹。
> 请确认截图里**有骑行轨迹线**（骑行详情页才有，单纯浏览地图不会画路线）。

新增 `tests/test_track_shape.py`（3 项）：
真实样本零误伤（硬约束）+ 用户那 4 张全部拦住 + 合成团块拦住。

#### 22.21.13 画布空白与重复图（同一批实测里发现的两个问题）

用户上传的 4 张图拼完后反馈「中间有白色」。查下来是**两件独立的事**：

**① 白色 = 画布空隙（几何必然，不是 bug）**

画布尺寸是所有图的**外接矩形**，而用户的 4 张图是**错位排列**的：

| 图 | 位置框 |
|---|---|
| 第 4 张 | (0, 0) |
| 第 2 张 | (1009, 467) |
| 第 1 张 | (2092, 992) |
| 第 3 张 | (2090, 991) |

覆盖率只有 **75.1%**，剩下 24.9% 是图与图之间的空隙——裁不掉，是几何事实。

**修法**：画布裁到「已覆盖区域的包围盒」（去掉**外侧**空白），
空隙填**中性灰 245** 而不是纯白，一眼能看出「这里没有图」，
并在 notes 里报告空洞比例。

实测：纯白像素从 **14.4% → 0.7%**，并新增提示
「图之间有 25% 的空洞（滑动步子太大或截图之间有遗漏）」。

⚠️ **测「有没有空白」不能用纯白像素比例**——
底图本身就有白色的道路（腾讯/高德都是白底白路），
裁剪后仍有 11.5% 纯白像素，但那些是**道路**不是空白。
正确判据是**画布四边**外侧不该有空白（空白必在包围盒外，道路不会）。

**② 重复图**

用户的 4 张里有 2 张是**同一视角的重复上传**（配准位移 (0, 0.3) 像素），
不增加覆盖范围，只会让画布出现「两张叠在同一处」的错觉。

**修法**：在「可信度过滤」**之前**做重复检测，判据用**位移接近 0**
（比 NCC 可靠——那对图的 NCC 只有 0.18，会被提前过滤掉）。

⚠️ 还有个坑：**完全相同的两张图，SIFT 反而配不上**——
描述子一一对应，ratio test 把它们全部落选。
所以要用**像素比对**兜底（`_is_near_duplicate`，平均差异 < 6）。

**顺带修掉一个位置框 bug**：`int()` 是**向零截断**，
坐标 -0.5 会变成 0（应为 -1）、0.5 变成 0（应为 1），
导致位置框与画布差一像素、甚至出现负数。
改用 `floor`/`ceil`，并把位置框钳制到画布范围内。

`tests/test_multishot.py` 8 → **10 项**（新增重复图检测 + 画布裁剪）。

#### 22.21.14 「明明有起终点，却说没检测到」——两个 bug 叠加

用户上传 4 张街道级截图（起点在第 1 张、终点在第 4 张），
结果页面却提示：

```
· 未检测到起点徽章（绿色「起」）
· 轨迹形状不合理：连通域宽 3242px ≈ 整幅图宽（3352px）
· 未能提取到有效轨迹线
· 第1、4 张是同一视角的重复，只保留一张
```

用户明确指出：**图里明明有起点和终点**。这是两个 bug 叠加的结果。

##### bug A：形状检查的 `too_wide` 判据在拼接画布上必然失效

第一版形状检查有两条**独立**的否决条件：

```python
too_wide = bw >= img_w * 0.75          # 横跨整幅图 → 底图元素
blob    = fill > 0.45 and ratio < 2.5  # 团块而非线条
if too_wide or blob:  →  判为底图色块，放弃提取
```

`too_wide` 隐含的假设是「骑行轨迹线不会横跨整幅图」。
这个假设在**单图**上成立，在**多图拼接画布**上必然失效——
11 公里的路线横跨整张拼接图，轨迹线的包围盒宽度**本来就该接近图宽**。

实测（用户这 4 张拼成的画布 4178×3864）：

| | 框宽 / 图宽 | 骨架填充率 |
|---|---|---|
| 真实轨迹 | 4016 / 4178 = **96%** | 0.0008 |
| 绿地图块（误判样本） | 1150 / 1260 = 91% | **0.0021 ~ 0.0022** |

⚠️ **填充率也区分不了**：绿地图块的填充率比真实轨迹还**高**。
原因是误判的绿地本身也是**细长的沿河绿地边界**，不是实心色块——
所以「填充率高 ⇒ 是色块」这条判据从一开始就不成立。

**最终判据：以拓扑为主导信号。**

真正的骑行轨迹**一定同时贴着「起」和「终」两个徽章**，底图元素不会。
实测真实轨迹的拓扑距离（连通域包围盒到两徽章的距离之和）只有 **39 像素**。
而且 `_select_component` 内部**已经**用像素级精确测距验证过这一点
（`best_cost > 260` 直接淘汰）——所以两个徽章都在时，
被选中的连通域必然是轨迹，不该再被形状否决。

改成：

```python
has_both = bool(start_pt and end_pt)
topo_ok  = has_both and (bbox 到两徽章距离之和) < 600
wide = bw >= img_w * 0.75
blob = fill > 0.45 and ratio < 2.5
if not topo_ok and (wide or blob):   →  判为底图元素
```

即：**有徽章且拓扑通过 → 一定放行**；缺徽章 → 才用形状兜底。

##### bug B：去重误判，把含起点的图丢掉了

旧版去重把第 1 张（含起点）与第 4 张判成「同一视角的重复」，
丢弃了含起点的那张 → 画布上找不到起点徽章 → 连带触发 bug A。

像素比对证实了这一点：08:07 那次生成的拼接图上，
标注为「第 1 张」的位置（2092, 992）里的内容
与**第 4 张原图逐像素完全相同**（平均差 0.00），
而与第 1 张差 12.74 —— 起点所在的图确实没进画布。

这一条已在 16:51 修好（去重改为在「可信度过滤」之前做、
丢弃较大索引并同步重映射），当前代码 4 张全部用上。

##### 为什么两个 bug 必须一起修

只修 bug A 不够：起点徽章丢了 → 没有两个徽章 → 拓扑无从谈起 →
仍然会落进形状兜底被拦。
只修 bug B 也不够：起点保住了，但轨迹线照样会被 `too_wide` 拦掉。

**修复后的实测结果**（同一批 4 张图）：

| | 修复前 | 修复后 |
|---|---|---|
| 拼接 | 3 张（第 1 张被丢弃） | **4 张全用上**，复核分 1.0 |
| 起终点徽章 | 未检测到起点 | 起 (78, 992)、终 (4044, 2903) |
| 轨迹 | 0 点 | **255 点**，置信度 0.576 |
| 里程 | 0 m | **9345 m** |
| 识别出的路名 | — | 安龙路、仙霞路、日晖东路、龙腾大道、内环高架路 |

##### 教训

1. **判据必须按「信息从哪来」分层**。
   几何判据（宽度、填充率）依赖图像的**呈现形式**，
   而拼接画布的呈现形式和单图完全不同——同一个判据在两层含义不同。
   拓扑判据依赖的是**内容语义**，跨场景稳定。
2. **「更严格的阈值」不等于「更可靠的判据」**。
   `too_wide` 单独用会误杀真实轨迹，`fill > 0.45` 单独用拦不住绿地。
   唯一稳的是**两个独立信号同时成立**（拓扑不通 且 形状可疑）。
3. **报错文案不能误导**。原来无论哪种失败都说
   「请确认截图里有骑行轨迹线」——而用户明明截到了。
   现在按「有没有徽章」分流：有徽章但没连上 → 说「没连上徽章」；
   无徽章 → 才说「确认是否有轨迹线」。

新增 `tests/test_user_multishot_real.py`（4 项），
直接用用户的真实截图做端到端校验：
起终点必须检出、轨迹点数 > 100、横跨整图的轨迹不得被误伤、
绿地样本必须仍被拦住、`samples/` 真实样本零误伤。

#### 22.21.15 倒序上传又失败：判重把「含起点的图」删了（2026-11-08）

**现象**：用户把同一批 4 张截图改名 `3-1.jpg ~ 3-4.jpg` 后重新上传，
浏览器按文件名排序，实际送入顺序是 **3-4、3-3、3-2、3-1（倒序）**。
页面又出现「未检测到起点徽章 + 轨迹形状不合理」，看起来像上一节白修了。

**先证伪「修复失效」**：对两批上传文件做 sha1，四张逐一完全一致，
只是顺序反了。于是用 `tools/exp_order_bug.py` 把正序/倒序都跑一遍：

| 顺序 | 终点图↔起点图 的 SIFT | 去重结果 | 画布 | 起 / 终 | 轨迹 |
|---|---|---|---|---|---|
| 正序 1-2-3-4 | inl 37 / ncc 0.02 / t=(183, 296) | 不判重 | 4178×3864 | (78,992) / (4044,2903) | 4794 点 |
| 倒序 4-3-2-1 | **inl 13 / ncc 0.18 / t=(0.0, 0.3)** | **误判重复，丢掉含起点的图** | 3352×3712 | **None** / (3218,2751) | 0 点 |

`_match_pair(a, b)` 不是对称的：同一对图换个方向，RANSAC 会选出一个
**位移几乎为 0 的假解**。旧去重只看「位移 < 6px」就认定同一视角，
于是把起点图删了，连锁触发「无徽章 → 形状判据否决」。

**第二个坑**：这条假匹配的 `ncc = 0.18` **高于默认可信阈值 0.15**，
就算不判重，它也会通过 1.6 的过滤，把起点图叠到终点图的位置上。
所以只改判重条件不够，必须把这条 link 一起断掉。

**修复**（`app/multishot.py`）：

1. `_is_near_duplicate()` 加**结构校验**：除平均像素差 < 6 外，
   还要求灰度 NCC > 0.90。实测这批图两两 NCC 只有 0.18~0.25，
   同一视角则接近 1.0 —— 淡色底图 90% 是浅色背景，
   单看平均差（实测 12.2~12.8）虽已能分开，但结构判据更稳。
2. 判重**只认像素/结构一致**，不再把「位移近 0」当充分条件。
3. 「位移近 0 但内容不同」的 link 判为错配，**直接从 `all_links` 删除**，
   既不判重也不参与拼接，并在诊断里明确写出原因。

**验证**：倒序 `used=[0,1,2,3]`，画布 4177×3864，
起 (78,991) / 终 (4043,2903)，轨迹 4790 点，与正序仅差 1 像素；
HTTP 倒序上传返回 `ok=true`、里程 9318.5 m、269 个轨迹点，
无「未检测到起点徽章 / 形状不合理」告警。
新增 2 项回归（倒序一致性、真重复图仍被去重），全量 **91 项测试通过**。

##### 教训

1. **判重是破坏性操作，判据必须是「内容一致」，不能是「变换一致」**。
   变换是算法算出来的，会错；内容是输入自带的，不会骗人。
2. **顺序无关性要显式测**。任何「顺序无所谓」的注释都需要一个
   `reversed()` 用例来兜底，否则只是口头承诺。
3. **一条错配可能同时污染两条路径**（去重 + 拼接图构建），
   修的时候要沿着数据流把下游都检查一遍。

#### 22.21.16 拆得更细的 12 张截图：三个叠加问题（2026-11-08 12:00）

用户把**同一条路线**从 4 张改成 12 张更细的截图，又失败了。
诊断里刷了 48 条「位移近 0 但内容不同，判为错配并断开」，
最后只剩 `used=[5..11]`，前 5 张（含起点）被整段排除，
`find_marker` 于是把截图顶部的红色 UI 圆点当成「终」，
拓扑约束永远满足不了 →「未找到轨迹连通域 / 掩膜为空」。

**根因是上一节的修复矫枉过正**，而且是三个问题叠在一起：

| # | 问题 | 实测数据 |
|---|---|---|
| 1 | 「位移近 0 ⇒ 删 link」误杀**真相邻对** | 图5↔图6 是唯一连接前后两块的桥，SIFT 却给出假解 `(0,0) NCC 0.4203`，而真解 `(1134,149) NCC 0.3956` 只低 0.02，被 NCC 择优选中假解；删掉后 12 张裂成 5+7 两块 |
| 2 | 12 张全用后画布超限 | `9897×6694 = 6625 万像素`，超过旧上限 6000 万，直接「画布过大，放弃」 |
| 3 | 接缝把轨迹切成两段，粗筛尺度假设失效 | 贴起点段 `(73,1105,3175,934)` d起=40 / d终=7238；贴终点段 `(3274,2107,6296,3706)` d起=3317 / d终=0。旧粗筛要求「两端距离之和 < 700」，两段都被淘汰 |

**修复**（三处，都在源头）：

1. `app/multishot.py::_pick_best_shift`：在**候选峰层面**排除 0 位移假解——
   两张**不同视角**的截图不可能配出 0 位移，除非它们像素级近重复
   （`_is_near_duplicate`，真重复图仍要能检出）。
   图5↔图6 于是自动改选真解 `(1134,149)`，48 条假匹配彻底消失。
2. 撤销上一节「事后按位移删 link」的规则：判重只认内容一致。
   删 link 是破坏性操作，绝不能拿一个**算法算出来的、会出错的变换**当判据。
3. `max_pixels` 60M → 120M（6625 万像素 RGB 约 190MB，实测可正常处理）。
4. `app/track.py::_select_with_markers`：有徽章时粗筛改为「**至少贴住一个**徽章」，
   若不存在单一连通域同时贴两端，就找贴起点段与贴终点段，
   两者都必须像素级贴住对应徽章（<260px）且空隙 <500px，
   用一条 9px 短直线补上接缝再合并。都不满足仍返回 None，绿地误判照旧被拦。

**验证**：12 张全用、画布 9896×6694、复核分 1.0、
起 `(125,1065)` / 终 `(9546,5607)`（真徽章面积 4186，UI 圆点只有 1840）、
轨迹 11188 点、置信度 0.95；HTTP 端到端 `ok=true`、里程 9097.8 m、
路名安龙路/仙霞路/日晖东路/龙腾大道/内环高架路，
无「掩膜为空 / 形状不合理 / 未检测到起点 / 画布过大」告警。
全量 91 项测试 + 新增 12 张回归用例全部通过。

##### 教训

1. **修一个 bug 时，别把「可疑现象」直接当成「判据」**。
   上一节看到「倒序假匹配位移≈0」，就在下游加了「位移≈0 就删」；
   但 12 张真相邻对也会配出 0 位移假解。正确做法是在**产生假解的地方**
   （候选峰择优）修，而不是在下游用另一个启发式兜底。
2. **任何基于「距离/尺度」的阈值都要注明它适用的画布尺度**。
   `两端距离之和 < 700` 在 4178×3864 的单批拼接上成立，
   在 9896×6694 上必然全灭——和 `too_wide` 是同一个错误类型。
3. **拼接画布上轨迹被接缝切断是常态**，不是异常。
   与其调颜色阈值硬凑单一连通域，不如显式补缝后再靠拓扑校验。

### 22.22 从「按样本打补丁」到「尺度自适应」（2026-11-08 12:40）

用户验收时的反馈：**「你应该改通用的结构，不要针对我传的图片定制化设置」**。
这是对 §22.21.14–16 三轮修复的准确批评：功能确实能跑了，但代码里塞满了
为某一批截图调出来的数字，注释里甚至写着那批图的坐标和实测值。

#### 问题清单（重构前）

| 位置 | 常数 | 来源 |
|---|---|---|
| `track._select_with_markers` | `area<800`、`d_near>400`、`ds/de<260`、`空隙>500`、`cv2.line(...,9)` | 按 4 张 / 12 张画布实测调 |
| `track` 形状校验 | `topo_ok < 600`（注释写着「粗筛 700、测距 260，取中间值」） | 按样本凑数 |
| `track` 其余 | `disk_kernel(13/5/3)`、`d<200`、`d>120` | 单图分辨率假设 |
| `multishot._pick_best_shift` | `abs(tx)<6 and abs(ty)<6` | 单图配准残差假设 |
| `multishot.stitch` | `max_pixels=120_000_000`（docstring 明写「12 张会拼出 6625 万，旧上限 6000 万会放弃」） | 按张数调 |

每换一批图就要重调一次，而且**修一个阈值常常在另一个尺度上制造新故障**——
§22.21.16 正是「位移近 0 ⇒ 删 link」修过头，导致 12 张图裂成两块。

#### 根因：两类尺度被混为一谈

* **画布尺度**（对角线 / 短边）随拼接张数增长 → 决定粗筛半径、拓扑距离、
  补缝容差、画布内存上限。
* **内容尺度**（徽章直径 → 轨迹线宽）由单张截图的缩放级别决定，
  **与拼了几张无关** → 决定贴合容差、面积下限、形态学核、补线宽度、
  0 位移容差。

12 张拼出 9896×6694 的画布，徽章仍然是几十像素。旧代码用
「两端距离之和 < 700」这种**画布尺度**的固定值去筛**内容尺度**的对象，
画布一大就必然全灭；反过来用固定 9px 补线宽度去补任意分辨率的接缝也不对。

#### 重构：`app/scale.py`

单一职责模块，只定义「无量纲系数 × 标尺」。`Scale.from_image(img, badge_d)`
里的 `badge_d` 由 `_detect_markers_ex()` 从检出的徽章连通域面积反推
（`√(4A/π)`，取两端较大者并 clamp），无徽章时用 UI 元素典型直径兜底。

| 派生量 | 定义 | 尺度类型 |
|---|---|---|
| `line_w` | `badge_d × 0.14` | 内容 |
| `touch_dist` / `coarse_dist` | `badge_d × 3` / `× 6` | 内容 |
| `min_area` | `(line_w × 3)²` | 内容 |
| `close/open/skeleton_kernel` | `line_w × 1.1 / 0.5 / 0.3`，取奇数 | 内容 |
| `bridge_width` / `marker_lead` | `line_w` / `badge_d × 2` | 内容 |
| `zero_shift_tol` | `max(2, 短边 × 0.005)` | 内容（配准残差） |
| `topo_dist` | `对角线 × 0.12` | 画布 |
| `bridge_gap(start,end)` | `min(路线跨度 × 0.08, badge_d × 12)`，下限 `badge_d × 4` | 路线跨度 |
| `canvas_pixel_budget(总输入像素)` | `min(512MB / 3B, 总输入像素 × 3)` | 画布内存 |

`track.py` / `multishot.py` 的公开签名（`extract_track` / `detect_markers` /
`stitch`）保持不变，内部一律经 `Scale` 取阈值；代码注释里的具体样本坐标
全部换成几何原理说明。`_is_near_duplicate` 的逐像素差 + 灰度结构相关、
`min_ncc`、填充率、长宽比这些**本来就是尺度无关**的量原样保留。

#### 验证（行为等价 + 跨尺度）

* 新增 `tests/test_scale.py` 11 项：锁住「内容阈值不随画布变化、
  画布阈值不随徽章变化」，以及预算/补缝/容差的单调性与边界。
* 真实样本脚本 7 项全过，且**结果与重构前完全一致**：
  4 张 4794 点 / 倒序 4790 点 / 12 张 11188 点，起终点坐标未变，
  绿地误判与 `samples/` 零回归。
* **缩放不变性**：同一批 4 张截图整体缩放到 0.85 / 1.0 / 1.25 倍，
  三者都识别成功，起终点坐标严格按倍数缩放（`78 → 67 → 98`），
  轨迹点数 4060 / 4794 / 5977。这条直接证明阈值不是给某个分辨率写死的。

#### 通用启示

1. **凡是「距离 / 面积 / 像素数」的阈值，先回答「它的标尺是什么」**。
   答不上来，就是潜在的场景绑定。
2. **修 bug 时警惕「新增一个启发式」**：在下游加规则看似快，但每个新
   阈值都是下一个尺度上的故障点。优先在产生错误信号的地方修，并统一量纲。
3. **重构的验收标准是「行为等价 + 跨尺度一致」**，不是「样例跑通」。
   同样的轨迹点数、同样的起终点坐标，才说明只是换了阈值来源。

### 22.10 这个案例的通用启示

1. **「贴路距离」几乎不携带尺度信息**。
   低缩放下路网间距（几百米到一公里）与尺度误差（几公里）同量级，
   所以任何以「轨迹贴不贴路」为目标函数的优化都无法修正尺度。
   要定尺度，必须用**与路网密度无关**的信号——
   比如两个已知地理点之间的距离。

2. **地标文字标注的位置不能用来定尺度**。
   地图引擎会把地标名放在视野空白处，偏移可达 100-2000 米且**有方向性**。
   逐对锚点隐含尺度的极差实测达 58%。
   但**纯色块检测的起终点徽章**没有这个问题——它们是精确像素位置。

3. **尺度错了，平移救不回来**。
   尺度错 26% 意味着 13 公里的轨迹被拉成 16 公里，
   两端各多出 1.5 公里——平移只能整体挪动，
   总有一段留在错误的位置。所以「路名约束」必须能改尺度。

4. **护栏要能独立于主算法工作**。
   跨江检测在「联网拉岸线」失败时用内置兜底折线，
   且兜底折线的控制点直接取自真实岸线（偏差 0 米）。

5. **「猜」和「量」要分清楚**。
   路名串路是从路网几何**猜**一条合理走法（误差 +9%），
   多图拼接是从截图里**量**出真实走法（误差 4.0%、起终点 13/21 米）。
   当「猜」的上限不够时，优先想办法把信息源做厚
   （让用户提供更多、更细的输入），而不是继续调「猜」的参数。

6. **给用户的判断依据要能落到像素级**。
   用户说「我不知道你拼得对不对」时，
   给出「内点数 863、NCC 0.94」是不够的——
   要给「把每张图贴回去逐像素比对，相似度 0.952」这种
   用户能理解、能自己复核的判据。

7. **先做可复现的实验，再决定方案**。
   本次三个否定结论（街道级单图不可用、静态 UI 会骗过 RANSAC、
   美团底图与高德静态图不同源）都是实验得出的，
   每一个都直接否掉了一条看起来可行的路。
   特别注意：21.2「底图同源」与 22.21「不同源」**并不矛盾**——
   差别是**高德瓦片** vs **高德静态图 API**，渲染风格不同。

8. **`warpPerspective` / `warpAffine` 默认把 M 当 src→dst**。
   传反了不会报错，只会**内容整体错位**，
   而且画面「看起来」还行——这类错误必须用**已知标记反推落点**来验，
   不能靠看图。
   护栏静默失效比护栏误报更危险。

9. **信息源不存在时，要明确说「没有」，不要硬编一个**。
   用户拿 4 张纯地图截图来测，程序却从底图绿地里「提取」出一条三角形轨迹。
   ——宁可报错，也不能给一个看起来像结果的东西。
   判据：**结果的形状必须与信息源的形状一致**；
   底图色块的形状不可能是骑行轨迹的形状。
   这条推广到整个项目：**每个输出都要能追溯到输入里真实存在的东西**。

10. **判据要按「信息从哪来」分层**。
    几何判据（宽度、填充率）依赖图像的**呈现形式**——
    同一张图在「单图」和「拼接画布」两种呈现下，同一个判据含义完全不同。
    拓扑判据（连不连得上起终点徽章）依赖的是**内容语义**，跨场景稳定。
    实测踩过：`too_wide`（宽度 ≥ 图宽 75%）在单图上能拦住绿地，
    在拼接画布上却把 11 公里的真实轨迹判成了色块。
    **选判据时先问：它依赖的是呈现形式，还是内容语义？**

11. **「更严格的阈值」不等于「更可靠的判据」**。
    实测三条判据在真实数据上的表现：

    | 判据 | 真实轨迹 | 绿地误判 | 能区分吗 |
    |---|---|---|---|
    | 宽度 ≥ 图宽 75% | 96%（误触发） | 91%（触发） | ❌ |
    | 骨架填充率 > 0.45 | 0.0008 | 0.0022 | ❌ |
    | 拓扑距离 < 600 像素 | **39 像素** | 无起点徽章 | ✅ |

    唯一可靠的做法是**两个独立信号同时成立**（拓扑不通 **且** 形状可疑）。
    而且阈值要**双向验证**：既不能放过误判，也不能误伤真实样本。

5. **后台重跑 pipeline 必须带上所有用户输入**。
   否则后台任务算出来的是另一个问题（见 22.8.1）。

---

## 附：一句话速查
- **内部用 GCJ-02**，输出按需转 WGS-84；给 OSM 之前必须转。
- **道路名不能当锚点**；路名只用来校验区域。
- **冷启动必须批量 + 三阶段**，单锚点无法自证位置。
- **采样步长和简化容差都要有像素上限**。
- **`removeLayer` 传 id 不传对象**。
- **改了前端必须跑 `tests/test_frontend.py` 和 `tools/verify_browser.py`**。
- **新增断言要做双向验证**（还原 bug 版必须失败）。
- **用户说「A 和 B 都要看到」时，别做成互斥切换。**
- **低缩放下「贴路距离」定不出尺度**；要用「两个已知地理点的距离」这类
  与路网密度无关的信号（见 22.10）。
- **地标文字标注有方向性偏移**，不能用来定尺度；
  纯色块检测的起终点徽章可以。
- **尺度错了平移救不回来**——路名约束必须能改尺度。
- **OSM 的 `waterway=river` 是岸线不是河心线**；法向会随弯道翻转，
  要用东向分量统一符号；KD 树距离是「度」不是「米」。
- **后台重跑 pipeline 要带上所有用户输入**（否则算的是另一个问题）。
- **`@property` 不等于「实时读文件」**——`load_dotenv` 在导入时就把值灌进
  `os.environ` 了；改 `.env` 要生效必须真的重读文件（见 22.13）。
- **「有 key」≠「key 能用」**：高德 key 分平台类型，Web端(JS API) 的 key
  调 Web 服务接口必然报 `USERKEY_PLAT_NOMATCH`。
- **城市级截图里底图不显示街道名**——能认出的只有行政区/地铁线/大 POI，
  它们的标注位置偏差公里级，比轨迹还大，无路名时原理上不可解（见 22.14）。
- **「位置」和「形状」是两个维度**：路名能修正位置（300-900 米），
  但修不了形状——城市级截图里的轨迹线本身就是直线，锯齿在截图阶段就丢了。
- **用路名重建形状会变差**：DP 对齐把 3233 米的轨迹段分给了 1289 米长的路，
  「段长 vs 路名长度」必须作为对齐约束（见 22.15）。
- **「猜」和「量」要分清楚**：路名串路是猜（误差 +9%），
  多图拼接是量（误差 4.0%、起终点 13/21 米）——
  猜的上限不够时，把信息源做厚，别继续调参（见 22.21）。
- **`warpPerspective` / `warpAffine` 默认把 M 当 src→dst**：
  传反了不报错，只是内容整体错位，画面还「看起来」正常——
  必须用已知标记反推落点来验（见 22.21.5）。
- **RANSAC 会被静止 UI 骗**：同一像素位置内容相同的区域就是 UI，
  它们的对应是「单位变换」，而单位变换也是合法相似变换（见 22.21.3）。
- **给用户的判据要能落到像素级**：「内点数 863」不够，
  要给「贴回去逐像素比对，相似度 0.952」（见 22.21.6）。
- **像素阈值必须有标尺**：画布尺度用对角线、内容尺度用徽章直径，
  两类不能混用；写死绝对像素等于把代码绑死在某一批截图上（见 22.22）。

### 22.23 「老单图样本变差了」的 A/B 排查，以及退化配准导致的 500

用户反馈「原来测试的那些单张截图，现在反而识别不对」。
在动代码之前先把变量拆干净（`tools/exp_single_regression.py`、
`tools/exp_ab_full.py` 可复跑）：

| 组别 | 代码 | 词典 | 结果 |
|---|---|---|---|
| A | HEAD | 当前（571 条） | 基准 |
| B | HEAD | 多图改造前（491 条） | 与 A **逐字段一致** |
| C | 重构前 `track.py` | 当前 | 与 A **逐字段一致** |
| D | `5713484` 整包（多图功能之前） | 当前 | 与 A **逐字段一致** |

四组对 `samples/meituan_ride_shanghai.jpg`、`samples/meituan_ride_2.jpg`
和用户新上传的城市级截图，置信度、里程、首尾坐标、内点/离群数、
RMSE、路名归属全部相同。**所以不是回归，也不是词典数据飞轮污染。**

观感落差的真实来源：最近几轮验收用的是**街道级多图**（起终点误差 13~21 米），
回头看**城市级单图**时，后者上限本来就是「大致片区」——
实测只有 4~10 个内点、31~41 个离群、RMSE 250~350 米，
因为截图里轨迹本身已被简化成直线/折线，POI 文字标注与真实点位差几百米。
这是 22.14 / 22.15 早就写明的**信息量上限**，不是算法退步。

排查中另外抓到一个一直存在、与本次重构无关的真 bug：
`samples/ride_motion.jpg` 走在线 `/api/recognize` 会 **HTTP 500**
（`LinAlgError: Singular matrix`，基线版同样触发）。
根因是兜底模型 `north_up_2pt` 在锚点地理坐标几乎重合时估出 `scale=0`、
矩阵全零却仍 `ok=True`：所有像素被映射到同一个经纬度（里程 0），
下游 `_self_check` 对矩阵求逆直接抛异常。

修复分两层（都是通用判据，与具体截图无关）：
1. `app/register.is_degenerate()` —— 矩阵有限、行列式非零、
   两轴尺度 >1e-3 米/像素才算配准成功；退化时在 `register()` 出口
   直接返回 `ok=False`，并给出「放大到街道级重截 / 用多图拼接」的可操作提示。
2. `app/pipeline._self_check()` —— 诊断逻辑对求逆做兜底，
   任何来源的退化矩阵都只跳过自校验，不打断整次识别。

验证：`tests/test_register_degenerate.py` 6 项；全量 pytest **109 passed**；
真实样本脚本 7 项全过；三张正常单图的结果与修复前逐字段一致；
`ride_motion.jpg` 在线识别不再 500，改为明确的「配准尺度退化 / 地理配准失败」。

### 22.24 「道路有偏移」——后台精修的触发阈值与算法分支互相矛盾

用户上传城市级偏大截图（`meituan_ride_shanghai.jpg`，4.58 米/像素），
看到轨迹在腾讯地图上整体压偏道路：置信度 52%、221 点、4.02 km、
配准残差 252 m、10 内点 / 31 离群，且
`road_fit = {"ok": false, "本次跳过路网精修（异步模式下由后台任务执行）"}`。

**根因不在配准算法，在「要不要精修」的判据**：

| 位置 | 判据 | 含义 |
|---|---|---|
| `pipeline._road_refine` | `scale > 8` → 校验+命名（**不平移**）<br>`scale < 8` → **修正模式**（平移轨迹贴路） | 8 米/像素是两种算法模式的**分界** |
| `server.recognize` 旧代码 | `scale > 8.0` 才启动后台精修 | 把分界值直接当成**触发门槛** |

于是 4.58 米/像素这类图恰好落在「算法里有修正能力、但永远不会被触发」的空档，
用户拿到的始终是 POI 文字标注造成的整体平移结果。这是**结构性矛盾**：
`_road_refine` 里 <8 才是能消除偏移的模式，触发条件却要求 >8。

实测同步精修（`refine=True`）对同一张图：

    平移 东 -55 m / 北 -320 m
    贴路中位 42.3 m → 4.5 m
    置信度 0.521 → 0.912
    路名归属 amap 泛化（威宁路/长宁路/北翟路/协和路）
              → 真实道路序列（新渔东路/威宁路/双流路/北虹路/新渔路/中环路…）

**修复**（`app/server.py`）：抽出 `_need_background_refine(result, want, sync)`，
判据只看**状态**——用户允许精修、本次未跑过、已配准出真实坐标、有轨迹点、
`road_fit` 尚未成功。精修耗时与采用哪种模式仍由 `_road_refine` 按尺度决定，
异步执行不阻塞主请求。另外后台任务完成后，仅当 `shot_id` 仍是最近一次识别
才把精修结果落盘，避免刷新页面退回粗结果、也避免旧任务覆盖新上传。

验证：`tests/test_server_refine.py` 7 项（含 4.58 / 0.5 / 15 米/像素三档、
配准失败、已精修、用户关闭）；全量 pytest **115 passed**；
真实 HTTP 上传 `refine_pending=true`，约 12 秒后 `/api/refine/{shot_id}`
返回 `done`：`road_fit.ok=true`、平移 (-55, -320)、贴路 42→4.5 m、置信度 0.912。

**通用启示**：一个阈值不能同时充当「算法分界」和「是否执行的门槛」；
「需不需要做」是状态判断，「怎么做」才是尺度判断，两者要分开。

### 22.25 「精修没提升、路线偏北」——指标更新了，地图没更新

承接 22.24：用户重测后状态条已是「已用真实路网精修位置（贴路中位 5 米）」、
置信度 91%，但腾讯地图上的轨迹**仍比实际路线偏北**，且认为「多图拼接之前是好的」。

**先证伪「多图改造改了坐标」**：把 `5713484`（多图功能之前）整包导出，
在 8849 起一个基线服务，用同一张 `meituan_ride_shanghai.jpg` 走完整 HTTP：

    基线 5713484 : conf 0.521 / 4016.7 m / GCJ-02 首点 121.3873326,31.2141934
                   road_fit.ok=false（scale 4.58 < 8，不触发后台精修）
                   路名归属 amap 泛化：威宁路、长宁路、北翟路、协和路
    当前粗配准  : 逐字段完全一致

也就是说多图改造前后，**服务端给出的初始位置一模一样**。
而 `app/roadfit.py` 开头的复盘早就写明：这张图的锚点全部来自 OSM POI，
粗配准「整体偏北约 245 米」，轨迹落到平行的错误道路族上
（用户实骑 新渔路/新渔东路/剑河路/金钟路，OSM 归属却是 天山路/北翟路）。
路网精修往南平移 320 米、把贴路中位从 42.3 m 降到 4.5 m，
**方向是对的**；精修后的路名序列也与用户实骑吻合。

后端也确实把精修结果算出来并送出去了（真实 HTTP 实测）：

    初次响应  geojson_gcj02 首点 121.3873326, 31.2141934，conf 0.521
    精修 done geojson_gcj02 首点 121.3868386, 31.2117349，conf 0.912
    差值：东 -47 m / 北 -272 m

**真正的断点在前端的「第二次渲染」**：`pollRefine()` 拿到 `done` 后调用
`render(payload)`，左栏数值、诊断、状态条全部换成精修后的值；但
`drawMapInner()` 删旧图层用的是 `map.removeLayer(polyline.id)`，
腾讯 GL JS 的 `removeLayer(t)` 内部按 `this._layers[t]` 查表，
`MultiPolyline` 实例未必对外暴露 `.id` —— 传 `undefined` 时旧图层没被删，
重建又可能直接抛 `"Unhandled error. (undefined)"`。而 `drawMap()` 把这个异常
**静默 catch 掉**，状态条照常报「已精修」。于是用户拿到的是
「精修后的指标 + 精修前的地图」，看起来自然是「没提升、还是偏北」。
另外 `render()` 只写 `currentResult`，没同步 `lastResult`，
`ensureMapReady()` 仍可能用旧的粗结果补画。

**修复**（`web/index.html`，与具体截图无关）：

1. 二次渲染**复用已有图层、用 `setGeometries()` 原地换几何**（轨迹 + 起终点标记），
   不依赖图层 id；SDK 不支持时才降级到 removeLayer + 重建。
2. `render()` 同步 `lastResult = d`，保证补画、导出、`getLastResult()` 都是最新结果。
3. `drawMap()` 返回成功与否，不再静默吞异常；`pollRefine()` 据此把状态条分成
   「地图已刷新 / 刷新失败 / 精修未采纳」三种，并写出
   **平移方向与量**（如「整体 东 -55 米 / 北 -320 米」）——用户能直接对照地图
   判断轨迹有没有动、往哪动了，而不是只看到一个抽象的「贴路中位 5 米」。

验证：新增 `tests/frontend_map_refresh_test.js`（最小 TMap 桩，跑两次
`drawMapInner()`，断言复用图层且几何换成新坐标）与
`tests/test_frontend_map_refresh.py`（接入 pytest），11 项全过；
全量 pytest **116 passed**、真实样本脚本 7 项全过；
HTTP 端到端确认初次与精修 done 的 GCJ-02 首点相差 272 米，
前端二次渲染确实有可移动的新几何。

**通用启示**：异步计算的结果要验证「**用户看到的那一份**」是否更新，
指标更新 ≠ 视图更新。把渲染异常静默吞掉，会把「视图没刷新」
伪装成「算法没效果」，排查方向从一开始就错了。

### 22.26 「不要根据图片定制」——去定制化审计与通用性契约

用户再次强调：改功能要走通用结构，不能针对某一次上传的截图调分支、
调常数、写名单。于是对整条链路做了一次静态审计。问题不在算法，
而在**代码里还留着四类样本痕迹**：

| 类别 | 审计发现的残留 | 处理 |
|---|---|---|
| 文件名分支 | `visuals._tag()` 对默认样本返回空前缀、其它样本加前缀 | 删除分支，三张 PNG 用固定名，样本隔离交给 `out_dir` |
| 样本专属名单 | `visuals.FIVE_ROADS`：某次实骑的 6 条路 + 固定配色 | 按识别结果 `road_attribution` 动态选路；按路名摘要稳定配色。任意城市、任意路名走同一套规则 |
| 地理数据内嵌 | `waterguard.WATER_SPECS` / `_FALLBACK`、`simfit.MAJOR_WATERS` 与半宽字典、Overpass 的上海 bbox、`seed_lexicon` 的 POI 坐标与 OCR 别名 | 外置到 `data/waters.json`、`data/seed_lexicon.json`；新增 `app/waterdata.py` 加载器，支持 `RIDE_WATERS_JSON` / `RIDE_SEED_LEXICON_JSON` 覆盖。换城市只换数据 |
| 裸数字散落 | 路网精修模式分界 `8.0` 写死在 `pipeline.py` 顶部；验证图连线阈值 `400px` | 分界改为配置项 `cfg.road_fit_low_zoom_m_per_px`（米/像素，物理单位）；连线阈值按图像短边派生 |

**通用性契约**（`tests/test_no_sample_customization.py` 把它变成可执行检查）：

1. **算法只吃数据**：核心模块的代码字面量里不许出现样本文件名、
   样本专属路名、固定经纬度；河流名单、POI 坐标、OCR 别名一律放
   `data/*.json`，可用环境变量整体替换。
2. **阈值分三类**：几何系数（无量纲 × 标尺，见 22.22 的 `Scale`）、
   物理量（米/像素、米，放 `AppConfig`）、统计样本量（与分辨率无关，
   如 NCC 最少像素数）。「实测这张图刚好多少」的数字只能出现在
   测试与排查记录里，不能进判据。
3. **输入文件名不影响行为**：命名、配色、筛选、触发条件只看内容
   或状态（是否已有坐标、是否已精修、贴路是否成功），不看文件名、
   固定偏移或预设道路名单。

要区分两类「实测」：文档与测试引用具体截图及其误差，是**验证用例**，
不构成代码分支；审计只检查代码字面量，不禁止排查记录。

验证：新增通用性测试 18 项（13 个模块的静态审计、水系/种子数据热替换、
任意路名配色与选路、配置项生效），全量 pytest **134 passed**；
水系数据迁移前后逐字段一致（3 条河、岸线 17/5/7 点、河心线 20/5/7 点、
半宽 1500/2500/250 与 200/800/60），POI 种子 29 条、别名 11 条不变；
多图拼接、轨迹提取、配准、精修、前端刷新等既有测试全部通过——
**识别算法的行为没有改动**，只把样本专属内容搬出了代码。

### 22.27 放宽后台精修判据后的过拟合回归：位移必须受轨迹尺度约束

**现象**：用户反馈「改过之后，多图拼接的不行了，有的单张图也不行了」。
用 `git archive HEAD` 导出改动前版本起 8849 对照服务，做逐字段 A/B：

| 对比 | 改动前 HEAD | 改动后工作区 |
|---|---|---|
| 多图（4 张）stitch / 画布 / 置信度 / 首末点 | ok / 4178×3865 / 0.575 | 完全一致（首末点差几米） |
| 6 张组拼接 notes（含「4% 空洞」） | 逐条一致 | 逐条一致 |
| 单图**主识别结果** | 逐字段一致 | 逐字段一致 |
| 单图 `refine=auto` 是否触发后台精修 | 否（旧判据要求尺度 > 8 m/px） | 是（22.24 改为状态判据） |
| 短轨迹图（763 m / 23 点 / 置信度 0.2）精修后 | 不触发，用户看到主结果 | **整体西移约 620 m**，`road_fit.ok=true` 但置信度仍 0.2 |
| 正常图（4017 m / 221 点 / 置信度 0.521）精修后 | 不触发 | 贴路 42→5 m、置信度 0.911，符合预期 |

**结论**：多图服务端没有回归；用户看到的那批 4 张图「28% 空洞」在改动前
就存在（相邻截图重叠不足，不是算法改动）。真正的回归是：22.24 把后台精修
判据放宽到「所有已配准出坐标的图」后，短轨迹、低置信度的图也被纳入精修，
而精修本身缺护栏，把轨迹搬到了另一条平行道路上。

**根因（通用结构缺陷，与具体截图无关）**：`refine_with_roads()` 用固定
±800 m 搜索半径 + 「贴路中位变小」作为唯一采纳判据。路网是稀疏的平行线族，
短轨迹整体搬几百米一样能压在**另一条**路上，判据全部满足。另外二维网格
搜索在长直道路上，「沿路方向」平移不改变贴路距离，旧的「严格小于才更新
最优」会挑到网格边界上的任意大 dx，凭空制造横向位移。

**修复**：

1. 位移上限取
   `min(road_fit_max_shift_m, road_fit_shift_span_ratio × 轨迹米制长度)`，
   比例是无量纲配置项 `AppConfig.road_fit_shift_span_ratio = 0.15`；
   搜索范围与采纳判据都用它，超限直接不采纳（`app/roadfit.py`）。
2. 候选择优改为「贴路距离在最优值 + 容差内时取欧氏位移最小者」，
   消除沿路方向的自由度。
3. 后台精修结果只在 `road_fit.ok` 为真时落盘，未采纳的诊断不覆盖
   用户当前看到的结果（`app/server.py`）。
4. 多图接口明确提示「不执行后台精修（大画布重跑成本过高）」，
   不再让用户误以为后台还在算。

**验证**：新增 `tests/test_roadfit_shift_guard.py` 3 项合成几何测试
（短轨迹不被搬走 / 长轨迹仍正常采纳 / 比例可配置且与分辨率无关）；
全量 pytest **137 passed**；HTTP 实测短轨迹从 620 m 变为北 30 m 微调，
正常图贴路 42→5 m、置信度 0.911；多图 HTTP 结果 stitch.ok=true、
4 张全用、画布 4178×3865、置信度 0.575。

**通用教训**：放宽「是否执行某优化」的触发条件时，必须同时检查该优化的
**适用前提**。触发判据只看状态是不够的——优化本身也要能拒绝「数据不足以
支撑这个结论」的输入。指标变好（贴路距离下降）不等于位置变对。

---

## 22.28 天地图接入：变量名、V2 接口格式与坐标系（本轮未提交）

> ⚠️ **本节是历史记录，其中的变量名已过时**：当时为兼容控制台不同写法引入了
> `TIANDITU_KEY` / `TIANDITU_API_KEY` / `TIANDITU_TK` 三个别名，顺序开关也叫
> `GEOCODE_PROVIDER_ORDER`。按「同一功能只用一个环境变量」的原则，现已在
> **§22.41 统一为 `TIANDITU_API_KEY` 与 `MAP_GEOCODING_ORDER` 两个单一变量名**，
> 其余全部下线。配置请以 §22.41 与 README 为准。

**背景**：用户提出「地名、路名各家地图都一样，只是 GPS 体系不同，是不是可以
换成天地图」，并在 `.env` 写入 `TIANDITU_API_KEY`；当时高德 key 配额超限。

**概念澄清（实测数据）**：地名文本相同 ≠ 坐标相同。坐标系差异是**系统性、
可换算**的；POI 点位差异来自各家数据库的采集与命名，**不可换算**。
上海实测：天地图 CGCS2000 经 `wgs84_to_gcj02` 消除约 **570 m** 系统偏移后，
与本地 amap 词典的同名点仍差 143 m（上海西站）、677 m（长风公园），部分 POI
在天地图甚至没有同名结果。所以天地图适合作为「高德不可用时的兜底 / 独立
校验」，不宜默认替代与美团底图同源的数据源；是否固定优先交给配置项。

**代码原本「支持」天地图，实际从未可用**：

1. 只读 `TIANDITU_KEY`，读不到用户写的 `TIANDITU_API_KEY`；
2. `/v2/search` 误用地理编码接口的 `ds={"keyword":...}` 参数，实测 HTTP 400；
3. 按 `result.items[].lon/lat` 解析；官方 V2 返回是 `pois[].lonlat` 字符串；
4. 未做 CGCS2000 → GCJ-02 转换，直接用会凭空整体偏约 570 m。

**修复（全部通用，无城市/图片硬编码）**：

- `app/config.py`：`TIANDITU_KEY_ALIASES` 兼容三种变量名；
  `tianditu_map_bound` 默认全球且只做格式校验；`online_provider_order`
  由 `GEOCODE_PROVIDER_ORDER` 配置，默认同源优先（高德→腾讯→百度→天地图）。
- `app/geocode.py`：`_tianditu` 改官方 V2 `postStr + type=query + tk`，
  解析 `pois[].lonlat` 并转 GCJ-02；同城候选优先但不过滤（多候选交给
  RANSAC）；`probe_online` 增加天地图真实探测；`geocode()` 按配置顺序遍历，
  上游返回空自动落到下一源。
- `tests/test_tianditu_geocode.py` 6 项离线 mock：别名兼容、V2 请求格式、
  坐标转换、顺序与自动回退、探测与错误码、视野配置，外加静态护栏防止
  `ds/keyword` 旧格式回流。
- README 配置段与「③ 地理编码」补充变量名、坐标转换与顺序开关说明。

**端到端 A/B 抓到的回归（必须先修才能启用该数据源）**：默认全球视野下，
天地图对 OCR 残片（「生活」「2号线」「花卉市场」）做「包含关键词」匹配，
返回苏州/昆山/内蒙古乃至新疆的同名泛类 POI；候选从 20 个膨胀到 66 个，
RANSAC 的 2 点模型被拉到 **682 km 里程、RMSE 58 km**，路名变成
「G16 丹锡高速」。只把视野收窄到城市范围后仍有 44.9 km / 崇明大道的错误
（市内同名点相距几十公里）。三层通用护栏：

1. `Lexicon.city_bounds(city)` 从词典里该城市已核验坐标派生搜索视野，
   中位数 ±3° 剔除跨省错条目，不内置任何城市边界；客户端再按坐标
   二次过滤（服务端 `mapBound` 未必严格过滤）。
2. `_tianditu` 只保留**同名**候选（复用 `Lexicon.normalize`，容忍标点差异）；
   泛匹配直接返回空，让下游走词典 / 冷启动兜底，与高德时代的行为一致。
3. `geocode()` 写回词典时同样只沉淀同名候选，避免泛匹配永久污染锚点和
   `city_bounds`。事故期间写入的 57 条 tianditu 条目（含内蒙古/新疆坐标
   却标着 city=上海）已用 `git show HEAD:data/lexicon.json` 整体恢复。

**验证**：用户 tk 有效，官方 V2 返回 10 条 POI；高德 probe 报
`USER_DAILY_QUERY_OVER_LIMIT` 时自动落到天地图。同一张反馈图
`1794134264_7cfc5be1.jpg`（`refine=off`）：

| 指标 | 事故态（全球视野） | 仅收窄视野 | 最终（+ 同名过滤） | 改动前纯词典 |
|---|---|---|---|---|
| 里程 | 682 km | 44.9 km | **1891.9 m** | 2048.2 m |
| 置信度 | 0.168 | 0.259 | **0.515** | 0.347 |
| 内点 / RMSE | 2 / 58 km | 2 / 488 m | **7 / 259 m** | 4 / 348 m |
| 路名 | G16 丹锡高速… | 崇明大道 / 陈海公路 | 安龙路 / 新渔东路 / 剑河路 | — |

最终路名与用户实骑一致，粗配准内点更多、残差更低。
`tests/test_tianditu_geocode.py` 8 项离线 mock（含城市视野、同名过滤、
词典写回护栏）；全量 pytest **148 passed**（2 个既有 PytestReturnNotNone
警告，与本轮无关）；服务重启后 `/api/health` 的 `probe.tianditu.ok=true`，
HTTP `/api/recognize` 200 且与离线结果逐字段一致，词典仅新增 1 条同城
同名候选（121.386, 31.183），无外省污染。**本轮改动未提交，等用户验收。**

---

## §22.29 锚点护栏误拒：在线锚点源降级引发的精修退化

**现象**：用户 21:01 上传 `1794142882_b0d04de1.jpg`（即 `samples/meituan_ride_shanghai.jpg`，221 点 / 4.02 km / 4.59 m/px），反馈「精修后不再符合真实路径」。

**链路事实**：后台精修确实执行（`/api/refine/...` 返回 `state=done`），但 `road_fit.ok=false`、shift 0/0；`server._run_refine` 只有 `road_fit.ok` 才落盘，所以页面停留在粗配准（conf 0.2、路名「天山路/北渔路/北翟路/协和路」）。拒绝原因是「可靠锚点残差 139.6m → 393.4m」。

**根因是两层叠加**：

1. 高德 `USER_DAILY_QUERY_OVER_LIMIT` → 在线锚点落到天地图 → 内点 9→7、锚点残差基线 201.9m→139.6m。
2. `refine_with_roads` 的锚点护栏第二条「相对恶化 >2.5 倍且 >100m」没有限定前提。基线 139.6m 时 `393.4 > 349` 触发拒绝；基线 201.9m 时 `407.5 < 505` 反而放行。**同一段合理平移，锚点基线越小越容易被误拒**。

物理上，中低缩放截图的 POI 文字是排版标注而非真实点位，锚点残差 100~300m 是常态，锚点本身没有资格否决绝对位置；这类图恰恰最需要路网几何纠正整体偏移。

**A/B**（同一张图、同一份路网，只切 `allow_online`）：

| | 在线（天地图锚点） | 离线（词典 amap 缓存） |
|---|---|---|
| 内点 | 7 | 9 |
| 锚点残差 before→after | 139.6 → 393.4 | 201.9 → 407.5 |
| 修复前护栏 | 拒绝 | 放行 |
| 平移 / 贴路 | — / 43→43 | -45/-330 / 49→5.3 |
| 路名 | 天山路·北渔路·北翟路 | 新渔东路·剑河路·金钟路·淞虹路 |

离线首点 `121.3869285,31.2117383` 与 §22.25 记录的精修后位置一致，说明「之前的正确效果」可复现，只是被护栏挡住。

**修复（通用，不调样本阈值）**：给相对恶化判据加「锚点本身可信（`before_anchor <= anchor_guard_m`）」前提。锚点不可信时仍由位移比例护栏（0.15×轨迹长度）与贴路改善判据兜底，防「搬到平行路」的能力不变。新增 `tests/test_roadfit_anchor_guard.py` 2 项：基线不可信时放行合理平移 / 基线可信时仍拒绝。

**验证**：护栏专项 8 passed；全量 pytest **150 passed**（2 个既有 PytestReturnNotNone 警告）。两张真实图在线模式 A/B：本图 `road_fit.ok=true`，平移 -40/-350m，贴路 43→5.7m，路名 `road_network` = 安龙路·新渔东路·北虹路·剑河路；§22.28 反馈图仍 `ok=false`（锚点 15.9→173.7m 被拒），护栏保护未失效。

**两点遗留**：① 本图置信度仍显示 0.2，来自 `_score_confidence` 末尾的缩放封顶（`zoom.position_ok=false → min(conf,0.2)`），与精修是否成功无关，本轮不动；② 后台精修是 `refine=True` **重跑整条 pipeline**、会重新取在线锚点，锚点源随配额波动会让同一张图的精修判定不稳定。后续 `app/maps/` 重构应改为复用主请求配准、只补算路网平移。

**本轮改动未提交**，连同 §22.28 天地图改动一起等用户验收。

---

## §22.30 地图能力模块化：阶段 1（骨架 + ①② 接入）

**背景与决策**：用户要求「把需要地图能力的部分模块化，方便拆分和替换服务商」。现状盘点发现六类能力散在 12+ 文件，只有地理编码有顺序回退雏形；CRS 转换靠注释和记忆（天地图漏转偏 570 m 就是这么来的）；`roadname._amap_key()` 自己读 .env 绕开 AppConfig；三份重复的 Overpass 镜像表。用户三项决策：**范围 = 后端四类能力**、**力度 = 完整抽包**、**节奏 = 先验收提交天地图那轮再分阶段**。验收通过后基线固化为 `04b81a0`。

### 阶段 1a：骨架（零行为变更）

新增 `app/maps/` 四个基础模块：`contracts.py`（CRS 枚举、`to_gcj02` 统一换算、`GeoPoint/PlaceCandidate/ProviderMeta/ProbeResult/MapProvider`）、`credentials.py`（唯一一处 env/.env 别名解析）、`registry.py`（按能力装配、`MAP_<CAP>_ORDER` 配置顺序、回退与 probe）、`http.py`（Session/UA/限流/重试/磁盘缓存）。不改任何现有调用点，新增 `tests/test_maps_skeleton.py` 16 项离线测试，全量 166 passed。

**关键设计**：provider 必须声明 `output_crs`，框架统一换算到 GCJ-02。坐标系进契约而不是进人脑，新增厂商不可能漏转。注册表索引改成 `(capability, name)`——高德同时提供地名与路名，只按名字索引会互相覆盖。

### 阶段 1b：①② 接入

**① 地理编码**（`app/maps/geocoding/providers.py`，403 行）：amap/tmap/baidu/tianditu 的 HTTP 请求与解析全部迁出 `app/geocode.py`；`Geocoder` 只保留 `_amap/_tmap/_baidu/_tianditu` 薄适配（兼容既有调用与测试注入）、四级优先级编排和词典。`geocode()` 不再手写厂商字典，顺序来自 `registry.order(GEOCODING)`；`probe_online()` 改为 `registry.probe(GEOCODING)`。四家分别声明 `GCJ02 / GCJ02 / BD09 / CGCS2000`，`PlaceCandidate.lon/lat` 自动是 GCJ-02。

**② 路名**（`app/maps/roadname/amap.py`）：高德路径规划的请求与 steps 解析迁出 `app/roadname.py`；`_amap_key()` 改走 `CredentialStore`；新增通用入口 `attribute_by_online(lonlat, cfg, timeout)`，`pipeline._attribute_roads` 改用它，调用方不再出现厂商名；`attribute_by_amap` 保留为兼容入口。

**health 去硬编码**：`server.health` 原先手写 4 个 `bool(CONFIG.xxx_key)`，现改为遍历 `geocoder.registry.names(GEOCODING)`，额外输出 `order` 与 `roadname` 能力状态。新增服务商时 health 自动多一个键。

**词典写回护栏（含一次自我纠错）**：第一版用 `Lexicon.city_bounds(city)`（min/max 经纬度 + 0.35° padding）校验，HTTP 验收发现**等于没校验**——该词典的视野被个别标错城市的条目撑到 `119.63~122.27°`，苏州/宜兴候选全部落在范围内，仍写回 38 条、其中 9 条标着 `city=上海`。根因是 min/max 对尾部离群点毫无抵抗力，而 `city_bounds` 本来就是为「搜索视野宁可宽」设计的，两种用途不该共用。

改为 `Lexicon.in_city_core()`：取该城市已核验点的中位数中心，算每个点到中心的米制距离，用 **95% 分位 × 1.6**（下限 20 km）作允许半径。分位数不受尾部错条目影响；实测上海核心半径约 68 km，上海动物园 8.5 km / 青浦 54 km 放行，苏州 88.5 km / 宜兴 130 km / 吴江 89 km 拦截。词典不足 8 条时返回 `None`、调用方放行，避免误杀新城市冷启动。回滚词典并重启后，同一张图 HTTP 识别写回的 tianditu 条目为 **0 条、跨市 0 条**。

### 完成判据与验证

判据是「`.env` 从 `tianditu,amap` 改成 `amap,tianditu`，**不改任何 Python 代码**即可切换并正确回退，同一批截图识别结果逐字段一致」。

- `tests/test_maps_geocoding_integration.py` 5 项：新环境变量切换顺序、上游返回空自动回退、旧名 `GEOCODE_PROVIDER_ORDER` 兼容（**该兼容用例已于 §22.41 移除，旧名不再支持**）、跨市同名不写回、四家 CRS 声明与 probe 形状。
- `tests/test_maps_roadname.py` 5 项：注册表按能力隔离同名 provider、`_amap_key` 走凭证层、`attribute_by_online` 走注册表且无 key 降级、health 静态护栏。
- 静态护栏随实现迁移：`test_old_ds_format_not_reintroduced` 从检查 `geocode.py::_tianditu` 改为检查 `providers.py::TiandituProvider`，额外要求 `CRS.CGCS2000` 声明与契约层换算。
- **A/B**：`git archive 04b81a0` 导出基线，同一张用户验收图、离线 `refine=False`，两边 `confidence / distance_m / track_points / GCJ-02 首末点 / model / inliers / outliers / rmse_m / scale_ground_m_per_px` 逐字段 **IDENTICAL**（0.2 / 4021.2 m / 221 点 / 121.3873328,31.2142736 / north_up / 9 / 10 / 195.86 m / 4.5802 m·px⁻¹）。
- 护栏改用 `in_city_core` 后新增 1 项分位数判据测试；最终全量 pytest **177 passed**（2 个既有 PytestReturnNotNone 警告）；`git diff --check` 通过。
- HTTP 端到端：health 返回 `order: [amap,tmap,baidu,tianditu]`、`roadname.order: [amap]`、`probe.tianditu.ok=true`；`refine=off` 识别 200；默认 `auto` 后台精修 `done` → `road_fit.ok=true`、平移 -40/-350 m、贴路 43→5.7 m、结果正确落盘。词典回滚重启后跨市写回为 0。

### 边界与遗留

- 阶段 1 只覆盖 ①②。**阶段 2**（③④ 道路几何与路线校验、合并三份 Overpass 镜像池 / OSRM·Valhalla 重复缓存并接入统一 HTTP 基座）、**阶段 3**（前端 `MapAdapter` 收敛 15 处 `new TMap.*`）、**阶段 4**（tools 静态地图）本轮一律不做。
- 路名目前只有一个在线实现，`MAP_ROADNAME_ORDER` 是扩展点而非「已有两家可切换」；完全关闭在线路名通过移除 `AMAP_KEY` 实现（凭证层统一）。
- `app/maps/http.py` 已就绪但 ①② 仍用各自 Session（为保持 A/B 逐字段一致），阶段 2 统一收敛。
- §22.29 的两点遗留未动：置信度缩放封顶、后台精修 `refine=True` 重跑整条 pipeline 会重新取在线锚点导致判定随配额波动。后者建议在阶段 2 改为复用主请求配准、只补算路网平移。

---

## 22.31 阶段 2：③ 道路几何 / ④ 路线校验接入，收敛重复基础设施

### ③ 道路几何：三份 Overpass 实现合成一份

过去 `roadfit.py`、`roadname_constraint.py`、`waterguard.py` 各维护一份
镜像列表与轮换逻辑：三份表不一致，`waterguard.py:50` 还有
`interparser` 拼写错误（同一个 private.coffee 实例被重复列两次），
改一处的超时策略另外两处不会跟着变。

新建 `app/maps/roads/`：

- `overpass.py` 是**唯一**镜像表 + 请求实现。提供 `overpass_raw()`
  （返回文本）与 `overpass_query()`（返回 dict）。主镜像拿完整超时，
  备用镜像拿 `max(10, timeout/镜像数)`；网关返回的 HTML 错误页与
  畸形 JSON 都在循环内被当作失败，继续换下一个镜像。需要「每个镜像
  都拿完整超时」的水系查询通过 `per_mirror_timeout` 显式指定。
- `provider.py` 的 `OsmRoadGeometryProvider` 声明
  `capability=ROAD_GEOMETRY` / `output_crs=WGS84`（OSM 经纬度不是
  GCJ-02，漏换算就是约 500 m 系统偏移）。公共 Overpass 无凭证，
  `probe()` 因此只做静态声明，**不在 `/api/health` 里打公共实例**，
  避免 health 被偶发超时拖住几十秒。
- 三个调用点改为委托：`roadfit._fetch_overpass` 用 `overpass_raw`
  （UA `meituan-ride/0.1`、失败不 sleep）；`roadname_constraint._overpass`
  用 `overpass_query`（失败等 0.8 s）；`waterguard._fetch_water` 逐镜像
  调用 `overpass_query(mirrors=(m,))`，保留「拿到 JSON 但岸线为空也要
  换下一个实例」的原有语义，拼写错误的镜像随之消失。

### ④ 路线校验：两份客户端共用一套基座

`osrm.py` 与 `valhalla.py` 原先各写了一遍 Session/UA、磁盘缓存、
md5 缓存键、polyline 解码、路名命中率与顺序一致性，评分口径容易漂移。

- `app/maps/routing/common.py` 收敛纯函数：`decode_polyline`、
  `match_ratio`、`polyline_length`、`order_consistency`、`stable_key`，
  以及 `RouteCache`。缓存**沿用旧文件名（`osrm_*` / `valhalla_*`）与
  字段格式**，避免重构后既有缓存一夜失效、把公共实例打爆；TS 字段
  缺失的历史 Valhalla 缓存视为有效，带 ts 的 OSRM 缓存仍按 7 天过期。
- `app/maps/http.py` 新增 `request_json()` 返回 `(status_code, JSON)`，
  让上层能区分「HTTP 错误」与「换实例」，同时复用同一套 Session / UA /
  限流 / 重试。两个客户端不再自建 `requests.Session()`；Valhalla 的
  0.35 s 限流改由 `HttpClient(min_interval=0.35)` 统一执行。
- `app/osrm.py` / `app/valhalla.py` 保留所有公共名称（数据类、客户端、
  四个工具函数）为薄导出，业务调用点与既有测试零改动；`_session`
  保留可注入 property，测试仍可替换假会话。
- `providers.py` 注册 `osrm` / `valhalla` 两个
  `Capability.ROUTE_VALIDATION` provider，`WGS84` 声明一致；
  `MAP_ROUTE_VALIDATION_ORDER` 可改顺序，
  `MAP_ROUTE_VALIDATION_ENABLED=0` 可整体关闭。

### health 与配置

`/api/health` 顶层新增 `maps.road_geometry` 与 `maps.route_validation`，
上报 `order` / `available`；两者只遍历注册表、不发网络请求。
新增环境变量见 README 配置段：`MAP_ROAD_GEOMETRY_ORDER`、
`MAP_ROUTE_VALIDATION_ORDER` 与两个 `*_ENABLED` 开关。

### 完成判据与验证

判据是「行为不变前提下去重复」：镜像轮换的超时分配、UA、失败等待、
缓存文件名与字段、公共函数签名全部逐一对照原实现；新增服务商只改
`.env` 顺序、不改 Python 代码。

- 新增 `tests/test_maps_stage2.py` 13 项离线测试：镜像表唯一且无拼写
  错误、网关 HTML / 畸形 JSON 轮换、全镜像失败抛错、三个调用方委托
  关系、provider 的 CRS 声明与开关、顺序由配置驱动、
  `request_json` 保留状态码、缓存旧格式兼容、两个客户端共享基座、
  工具函数导出一致。全部用假 opener / 假 session，不发网络。
- 专项 `test_maps_stage2.py + test_valhalla.py + test_maps_skeleton.py`
  **42 passed**；`compileall app` 通过。

### 边界与遗留

- ①② 的可注入 `session_getter` 保持原样：它是既有测试注入契约，且
  geocoding 基类的限流/UA 本就只有一份。HttpClient 已在 ③④ 落地，
  后续把 ①② 接上属于纯机械改动，但会改变请求侧行为，需要另做一次
  逐字段 A/B，本阶段不冒险动刚验收通过的链路。
- 后台精修 `refine=True` 重跑整条 pipeline、锚点源随配额波动的遗留
  未动：那是**行为修复**而非去重复，混进重构会破坏归因，留作独立改动。
- 阶段 3（前端 `MapAdapter` 收敛 `new TMap.*`）与阶段 4（tools 静态
  地图）仍按计划分开提交。

## 22.32 阶段 3：前端底图适配层（MapAdapter）

### 问题

腾讯地图 GL JS 的对象模型（`new TMap.Map` / `MultiPolyline` / `MultiMarker` /
`LatLngBounds`）与业务逻辑混在同一个 `drawMapInner()` 里。后果有三个：

1. 换底图服务商（高德 / 天地图 / Leaflet）必须改业务函数；
2. 测试只能整体 mock `TMap` 全局对象；
3. 厂商 API 一旦改名，业务代码跟着炸。

### 做法

新增独立文件 `web/js/basemap.js`（UMD，既能被页面 `<script>` 引入，
也能被 Node 测试 `require`）。它只暴露一个统一接口：

```
RideBasemap.create(providerName, { container, zoom })
  → adapter.setTrack(coords)      // coords: [{ lat, lon }]，GCJ-02
    adapter.fitBounds(padding)
    adapter.resize()
    adapter.onClick(cb)
    adapter.getPolyline() / getMarkers() / getLastCoords()
```

每个 provider 只回答 7 个问题：能不能用、怎么建地图、怎么建轨迹、
怎么更新轨迹、怎么删图层、怎么缩放视野、怎么监听点击。

已注册两个实现：

| provider | SDK | 更新几何 | 删图层 | 视野 |
|---|---|---|---|---|
| `tmap` | 腾讯 GL JS | `setGeometries` | `map.removeLayer(id)` | `fitBounds(bounds, {padding})` |
| `amap` | 高德 JS API 2.0 | `setPath` / `setPosition` | `map.remove(overlay)` | `setFitView(overlays, false, avoid)` |

### 关键设计决定

- **坐标结构统一为 `{ lat, lon }`（GCJ-02）**。腾讯回调给的是
  `evt.latLng`（`LatLng` 实例）、高德给的是 `evt.lnglat`，
  两者形状不同，必须在 provider 里抹平，否则业务代码要按厂商分支。
- **二次渲染仍是「原地更新几何」**。适配层 `setTrack()` 优先
  `updateTrack()`，失败才 `removeTrack()` + 重建——与阶段 3 之前的
  语义一致，异步精修仍能真正刷新地图。既有回归项「连续识别移除旧图层」
  因此改为断言「不再 removeLayer、图层数不堆积、几何确实换新」。
- **`lastPath` 从 `TMap.LatLng[]` 变成 `{lat,lon}[]`**：字段名由
  `lng` 变 `lon`，适配层内部再转回厂商类型。

### 服务端配合

- `app/config.py` 新增 `BASEMAP_PROVIDER` / `AMAP_JS_KEY` /
  `AMAP_JS_SECURITY`；`AppConfig.basemap_provider` 返回当前服务商名。
  **没配高德 JS key 时即使写 `BASEMAP_PROVIDER=amap` 也自动回退
  `tmap`**——高德 JS SDK 的鉴权模型就是「前端 key + securityJsCode」，
  缺 key 会直接白屏。
- `app/tmap_proxy.py` 的 `render_page()` 新增两个占位符：
  `__WB_BASEMAP_PROVIDER__`（服务商名）与 `__WB_AMAP_SDK__`
  （高德 SDK 脚本整段）。不用高德时后者替换成空串。
- `/api/health` 的 `maps` 段新增 `basemap: { provider, available }`，
  验收时可直接确认切换是否生效。

### 验证

- 新增 `tests/test_frontend_basemap.py` 13 项：结构层（业务代码不再有
  `new TMap.*`）、行为层（Node 跑**真实**适配层，只把 TMap 换成桩）、
  服务端层（占位符注入、缺 key 回退、health 契约）。
- 新增 `tests/frontend_basemap_test.js`：24 项断言，覆盖原地更新、
  标记同步、点击回调只绑一次、换 provider 后业务调用不变、
  provider 异常可诊断。
- 更新 `tests/frontend_map_refresh_test.js` 与 `tests/test_frontend.py`：
  两者改为加载真实适配层而非直接 mock `TMap`。
- `tests/test_frontend.py` 主流程 **25/25 通过**；
  前端相关 pytest **13 passed**。

### 边界

- 高德 JS SDK **未经真机验证**——本地没有高德 JS key，只验证了
  契约与代码路径。要启用需在 `.env` 配 `AMAP_JS_KEY` 与
  `AMAP_JS_SECURITY`，并确认 key 的「JS API」类型与域名白名单。
- `app/trajmap.py`（对比页）仍直接用 `TMap.*`，未纳入适配层——
  它是独立页面，改造收益低、风险高，留作独立任务。
- 阶段 4（tools 静态地图收敛）待做。

## 22.33 阶段 4：⑥ 静态地图素材收敛

### 问题

`tools/` 下三个实验脚本各自抄了一份「拼 URL + urllib 取图 + 解码」：

- `tools/exp_basemap_match.py`
- `tools/exp_e2e_multishot.py`
- `tools/exp_overlap_need.py`

三份的 URL 参数顺序、`size` 拼法（`1024*1024` vs `1024%2A1024`）、
错误处理（一个抛异常、一个返回 None）都不一样。改一次瓦片参数要动
三个文件，也很容易漏。

### 做法

新增 `app/maps/staticmap/`（能力⑥），只暴露两个函数：

```
fetch_static_map(lon, lat, zoom, size, provider="amap", key=..., markers=...)
  → StaticMap(data=bytes, provider=..., url=...)
amap_key(cfg=None) -> str | None
```

- 两个 provider：`amap`（`location` + `size=W*H`）与
  `tmap`（`center=lat,lon` + `width`/`height` 分开写）。
- 缺 key 或请求失败一律返回 `None`——实验脚本据此跳过，
  不该因为一次网络抖动丢掉整批实验。
- 三个实验脚本改为调用它，自己不再拼 URL。

### 边界

静态地图是**实验素材**，不是识别链路的依赖：删掉整个
`app/maps/staticmap/` 不影响 `app/server.py` 的可用性。
它没有走 `ProviderRegistry`——注册表面向的是「按能力装配 + 配置驱动
顺序 + 健康探测」，静态地图只有两个固定实现、只在实验脚本里用，
硬套注册表只会增加间接层。

### 验证

新增 `tests/test_maps_staticmap.py` 12 项：两个 provider 的 URL 形态、
缺 key 返回 None、请求失败返回 None、key 从配置读取、
三个脚本确实 import 统一 provider 且不再手写 URL、
仓库内不存在第二份静态地图 URL 拼接。

### 阶段 2–4 的共同边界

- ①② 的可注入 `session_getter` 仍保持原样（既有测试注入契约）。
- 后台精修 `refine=True` 重跑整条 pipeline、锚点源随配额波动的遗留
  未动——那是**行为修复**而非去重复，混进重构会破坏归因。
- `app/trajmap.py`（对比页）仍直接用 `TMap.*`，未纳入前端适配层。

### 22.33.1 阶段 3 的一个隐蔽 bug：占位符与 JS 变量名嵌套

阶段 3 验收时用真实 HTTP 拉页面，发现渲染结果是：

```js
window.tmap = 'tmap';   // ← 不是 window.__WB_BASEMAP_PROVIDER__ = 'tmap'
```

原因：页面里原本写的是

```js
window.__WB_BASEMAP_PROVIDER__ = '__WB_BASEMAP_PROVIDER__';
```

服务端 `render_page()` 做的是**朴素字符串替换**：

```python
html = html.replace(PLACEHOLDER_BASEMAP, basemap_provider)
```

于是第一处（变量名里的同名子串）也被替换成 `tmap`，
得到 `window.tmap = 'tmap'`——**不报任何错**，只是变量名被悄悄改掉。
当时因为默认值恰好也是 `tmap`，行为上看不出异常；一旦切到高德就会炸。

修法：变量名与占位符名脱钩（`window.__WB_MAP_PROVIDER__`），
并在 `tests/test_frontend_basemap.py` 加了两条回归：

1. 页面代码行（跳过注释）里不得出现 `window.__WB_BASEMAP_PROVIDER__`；
2. 渲染后的页面必须仍含 `window.__WB_MAP_PROVIDER__ = 'tmap'`，
   且代码行里不得出现 `window.tmap`。

**通用教训**：服务端渲染的占位符不能与前端任何标识符同名。
凡是「同一个字符串在页面里出现两次以上」的地方，都要检查
其中一处是否会被连带替换。

### 22.33.2 顺手修掉的一个真实缺陷：.env 热加载只看 mtime 会漏改

阶段 3 的全量回归里，`tests/test_geocode_config.py` 偶发失败
（`改 .env 后取值没变化`）。追下去发现**这不是测试的问题，是生产代码
的缺陷**：

`AppConfig._file_env_value()` 原先只看 `st_mtime` 判断「文件是否变了」。
但文件系统的 mtime 精度取决于时间刻度，**两次写入可能落在同一刻度内**——
实测同一秒内「复制 + 覆写」有 3/8 概率 mtime 完全相同：

```
0 float differs: False  ns differs: False
1 float differs: False  ns differs: False
2 float differs: True   ns differs: True
...
```

后果：用户改了 `.env`，服务**读不到新值**，而且不报任何错——
「明明改了却不生效」，正是这个项目最早踩过的坑。

修法：缓存指纹从 `st_mtime` 换成 `(st_mtime_ns, st_size, 内容 sha256)`：

- `st_mtime_ns` 抓「同一刻度内的两次写入」；
- `st_size` + `sha256` 兜底任何 mtime 精度问题。

代价是每次取值多读一次文件内容——`.env` 只有几十行，
而配置读取不在热点路径上，这个代价可以忽略。

## 22.34 目录结构重构：源代码收进 src/

### 新布局

```
<仓库根>/
  run.py           启动入口（自己设好 sys.path，不需要 PYTHONPATH）
  data/ samples/ docs/ tests/ output/     数据与文档留在根目录
  src/
    app/           Python 包（识别核心）
    web/           前端页面与底图适配层
    tools/         诊断与 A/B 实验脚本
```

`app/` → `src/app/`、`web/` → `src/web/`、`tools/` → `src/tools/`，
根目录 4 个 `trajectory*.geojson` 移到 `output/`（该目录已 gitignore）。

### 三个必须小心的点

**1. `config.py` 的 `ROOT` 不能用固定层数。**
本文件会被打包进技能包，vendor 布局是
`<skill>/assets/ride_core/app/config.py` + `<core>/data/`，
与仓库布局差两层。固定层数会让 vendor 版去读**技能包外面**的 data/，
直接读不到词典。改成向上查找「最近的、含 `data/` 目录的祖先」，
两种布局都对。`SRC_ROOT` 同理：仓库布局是 `ROOT/src`，
vendor 布局下 `app/` 直接在 `ROOT` 下。

**2. A/B 工具的语义分层。**
`src/tools/exp_ab_full.py` 会把历史版本的 `src` 导出到临时目录再导入。
基线的 data/ 与 .env 放在**导出目录根部**，所以 `PROJECT` 必须仍指仓库根，
`pkg_root` 则要指 `src/`（那里才有 app 包）。两个常量现在显式分开。

**3. 顺带修掉一批硬编码绝对路径。**
`tests/test_endpointfit.py` / `test_routebuild.py` / `test_valhalla.py` /
`test_waterguard.py` 曾把仓库根绝对路径（现为 `r"D:/Dev/track-buddy"`）写死在 `sys.path`，
`test_waterguard.py` 还有两处硬编码的数据/缓存路径——换台机器就崩。
全部改成从 `__file__` 推导。`src/tools/probe_basemap_match.py` 里的
`sys.path.insert(0, ".")` 也一样：它依赖当前工作目录，从别处调用就找不到包。

### 顺带引入的改进

- 新增 `tests/conftest.py`：把 `src/` 加进 `sys.path`，**所有测试自动生效**，
  不必每个文件各写一遍 `sys.path.insert`（那种写法容易漏、也容易在
  目录调整时全网改一遍）。
- 包内 2 处绝对导入 `from app.xxx import` 改成相对导入
  （`routebuild.py` / `visuals.py`）——包内用相对导入才不受包位置影响。
- `run.py` 自己把 `src/` 加进 `sys.path`，所以 `python run.py` 直接能跑，
  用户不需要手动设 `PYTHONPATH`。
- 技能脚本 `recognize.py` 现在**同时支持两种内核布局**：
  仓库模式 `<root>/src/app/` 与 vendor 模式 `<root>/app/`，
  并按布局设置对应的 `PYTHONPATH`。

### 新增回归

`tests/test_project_layout.py` 11 项，从**结构**上锁住布局：
三块源码在 src/ 下且根目录无残留、data/samples/docs/tests 仍在根、
根目录无游离 geojson、两种布局下 `ROOT`/`DATA_DIR`/`WEB_DIR` 都正确、
无硬编码绝对路径、无依赖 CWD 的 `sys.path.insert(0, ".")`、
`from __future__` 都在文件开头、`python run.py --help` 无需 PYTHONPATH 即可运行。

**教训**：目录重构最怕「路径错了但测试不报」——
所以测试必须从结构层面验证，而不是只跑一遍功能测试。

## 22.35 天地图同名过滤放宽 + 在线源「单源严格模式」

### 起因：一次实测暴露的三个问题

问「项目当前用高德还是天地图」，实测发现：

1. 高德当日配额用尽（`USER_DAILY_QUERY_OVER_LIMIT`），**每个地名都先打高德
   再打天地图**——既慢一倍（21.4s / 18 次请求），又让人误判实际用的是哪家；
2. 天地图作为兜底**基本失效**：接口明明返回了 10 个坐标正确的 POI，
   解析后剩 **0 个**；
3. 查「静安寺」返回的是**苏州常熟**的静安寺，查「外滩」返回**昆山**的
   「外滩8号」——异地同名污染。

### 问题一：天地图同名过滤过严

早期实现要求返回标题与查询词**完全相等**：

```python
if Lexicon.normalize(title) != Lexicon.normalize(name):
    continue
```

但天地图的搜索是「包含关键词」语义。查「东方明珠」返回的是
「东方明珠广播电视塔」「东方明珠-公交站」「东方明珠公园东南门」——
**一条标题恰好等于查询词的都没有**，于是全部被丢掉。
这个源等于从未可用。

改成两级判据（`providers.py::_name_matches`）：

1. **连续子串**：归一化后 `query in title`；
2. **分段包含**：按**原始**标题的分隔符切段，查询词完整命中其中一段。

第 2 条是必需的：「东方明珠-公交站」归一化成「东方明珠公交站」后，
「东方明珠」不是它的连续子串，但按原始分隔符切开第一段就是。
地图上最常见的标注恰恰都带分隔符（公交站 / 门 / 入口 / 地铁站）。

顺带修了两处同源问题：

- **地址过滤过严**：`address` 不含城市名就丢弃，但公交站/地铁站这类
  标注的 address 常常就是 POI 名本身（「东方明珠-公交站」的 address
  是「东方明珠」）。改成：地址不含城市名时，只要**标题**与查询词同名就放行。
- **视野被离群条目撑大**：原来用 `Lexicon.city_bounds`（min/max 派生），
  而上海词典里混进了经度 113.1（山西）、纬度 22.5（广东）的错条目，
  视野被撑到 `119.63~122.27°E / 30.47~32.19°N`，苏州、昆山全在视野内。
  改用「中位数中心 + 核心半径（距离分位数派生）」，
  实测视野收紧到 `121.10~121.77°E / 30.92~31.50°N`，
  「外滩」「静安寺」「上海大学」「陆家嘴」「世纪公园」全部命中上海本地 POI。

新增 `Lexicon.core_radius_m()` / `core_center()`，与既有的
`in_city_core()` 同源（都是「中位数中心 + 距离分位数」）。

### 问题二：不再静默兜底

新增 `Geocoder._online_search()`，语义改为**单源严格模式**：

| 配置 | 行为 |
|---|---|
| `MAP_GEOCODING_ORDER=tianditu` | 只用天地图，失败直接提示 |
| `MAP_GEOCODING_ORDER=amap,tianditu` | 显式多源，才依次回退 |
| 不配 | 默认顺序里**第一个可用**的源（当前高德），失败不换源 |

区分「显式配置」与「默认表」需要新方法
`ProviderRegistry.configured_order()`——`order()` 在没配置时返回
内置的四家默认表（amap/tmap/baidu/tianditu），若直接拿它判断
「是不是多源」，默认情况就会被当成「四家依次回退」。

失败原因通过两条通道暴露：

- `Geocoder._last_error` → `/api/health`（原有）；
- `Geocoder._online_failures` → **识别结果的 warnings**（新增），
  例如「在线地理编码不可用，已降级到本地词典与 OSM 冷启动：
  amap: 高德 USER_DAILY_QUERY_OVER_LIMIT。如需更换服务商，在 .env 设置
  MAP_GEOCODING_ORDER…」。

同时修正了 `diagnostics.online_geocoding` 的语义：它原本是
「配了 key」，现在改成「本次识别**真的用到**了在线源」——
单源模式下配的那家可能每次都失败，报 True 会让人以为在线源工作正常。

### 顺带修掉的两个「陈旧状态」问题

1. **provider 的 `last_error` 成功时不清空**。它是「最近一次错误」，
   而成功路径不写它。不清会出现「上一次配额超限、这次成功」却仍报故障。
   现在每个 `search()` 入口先 `_reset_error()`。
2. **「查不到」被当成故障**。OCR 残片（「支工」「丰河」）本来就查不到，
   报成失败会让人以为服务商坏了。现在归为 `misses`，
   只有真正的错误（配额/鉴权/异常）才进 `_online_failures`。
   并且「只是查不到」时会**清掉**陈旧的 `_last_error`——否则天地图模式下
   明明查得到「东方明珠」，health 里却仍显示上一次的「高德配额超限」。

### 验证

- 新增 `tests/test_geocode_single_source.py` 16 项：模糊名被接受、
  精确匹配排最前、前缀优先于任意包含、无关候选被排除、视野用核心中心
  派生、离群点被排除、单源不回退、多源才回退、失败被记录、
  「查不到」不算故障、成功后清空陈旧错误、`configured_order` 区分配置与默认。
- 更新 `tests/test_tianditu_geocode.py` 3 处旧语义断言
  （原本断言「候选恰好 1 个」「视野 == city_bounds」，都是旧规则）。
- 实测三种模式：
  - 默认 → 只打高德，失败记录 `amap: 高德 USER_DAILY_QUERY_OVER_LIMIT`，
    **不再落天地图**；
  - `tianditu` → 8 个上海本地候选，零失败记录；
  - `amap,tianditu` → 天地图接手，`source=tianditu` 标记正确。

## §22.36 多图精修链路：从「手动跑得通」到「默认可用」（2026-11-09）

### 起因：一次零改动实测

问「多图模式下能支持精修吗」，先不改代码，直接拿 11:09 上传的 12 张图调
`POST /api/recognize_multi?refine=on` 验证：

| 项 | 实测 |
|---|---|
| HTTP / 耗时 | 200 / **263.8 秒** |
| `road_fit.ok` | `true`（修正模式，尺度 0.887 m/px < 8 分界） |
| 平移 | 东 +25 m、北 −50 m（合 56 m） |
| 贴路中位 | 24.7 m → 8.7 m |
| 路名来源 | `amap`（10 段）→ `road_network`（OSM 几何，87 段） |
| `confidence` | 0.2 → 0.829 |

结论：**通路本来就跑得通**，但只有手动传 `refine=on` 才有，默认 `auto` 下
多图明确跳过。三个障碍：

1. 后台精修是 `_pipeline.run(img, refine=True)` **整条重跑**——大画布上
   OCR、轨迹提取、锚点地理编码、配准全部再来一遍，成本翻倍；
2. 拼接画布只有一个刚体 `reg`，12 张链式拼接 depth=10 的累积漂移会被
   当成「配准误差」，用 `min(800, 0.15×轨迹长)` 的平移量去纠正，
   容易压到平行主干道上；
3. 多图接口不返回 `refine_pending`，后台任务也不更新 `last_stitch.json`，
   前端那段轮询根本不会触发。

另外实测发现 `data/cache/roads.json` 时间戳没变——这次**命中了已有路网缓存**，
没有重新打 Overpass，264 秒基本花在画布级重算与 KD 树搜索上。

### 三个阶段（各自单独跑全量、单独提交）

| 提交 | 阶段 | 内容 |
|---|---|---|
| `b319d5d` | 1 | `PipelineResult.refine_context` + `RefineContext` + `refine_existing()`；后台不再重跑 pipeline |
| `5a341ae` | 2 | 多图接同一后台任务、`refine_pending` 轮询、同步 `last_stitch.json`、`verify` 档 |
| 本次 | 3 | 评估点抽样提速、任务耗时上报、前端「贴路校正」开关、文档 |

**阶段 1 的关键设计**：`run()` 成功配准后把 `reg / track / simplified /
anchors / pairs / zoom / texts / alternatives / road_name_fit` 留在
`RefineContext`（不进接口 payload）。后台只做三件事：补算平移 →
`apply_to_polyline` 重投影 → 重算置信度与路名归属。`reg` 用 `deepcopy`，
任务重试不会在上一次平移上再叠一层。用户路名约束生效时**直接跳过后台平移**
——它是会改尺度的强约束，叠一个贴路优化等于把轨迹从用户指定的路上拉走。
后台默认 `allow_online_roadname=False`，不再悄悄打一次商业接口，避免配额
波动让同一张图两次结果不一致。

**阶段 2 的关键设计**：`_road_refine(mode="verify")` 强制走低缩放档，
只算贴路距离 + 几何路名，坐标、置信度、里程一律保持主请求的值，并在
`road_fit.mode` 打标供前端区分文案。多图 `auto` 用 `verify`，显式
`refine=on` 仍是同步修正；只传 1 张图时仍用完整 `auto`。后台结果
通过 `extra` 带回 `stitch / n_shots / stitched_url`，前端刷新不丢拼接画布。

**两处竞态**（都是集成测试逼出来的）：
- 多图里 `_start_refine` 起初放在 `_persist_last_result` 之前，后台线程
  读 `last_result.json` 做 shot_id 保护时文件还没写，误判「用户换了新图」
  而不落盘 → 启动挪到两份结果都写完之后；
- `_run_refine` 原先先标 `done` 再落盘，前端轮询拿到结果立刻渲染，
  此时核对页文件还是旧的 → `done` 挪到落盘之后。

**阶段 3 的性能**：`refine_with_roads` 新增 `eval_budget`（默认 2000）。
位移候选数与轨迹点数相乘：1.1 万点 × 4225 候选 ≈ 4700 万次 KD 树查询。
轨迹点本来就是约 6 米等距采样，等距抽样不改变「中位距离」这个统计量；
**搜索空间、锚点护栏、位移限幅依据的轨迹长度（`span_m` 用完整折线算）
都不变**。`tests/test_roadfit_eval_budget.py` 用合成几何锁住
「抽样前后平移量与护栏判定一致」。

### 验证

- 全量：阶段 1 **251 passed**、阶段 2 **257 passed**、阶段 3 **265 passed**。
- 新增测试：`test_refine_reuse.py`（编排 / deepcopy / 强约束 / 不重跑 pipeline）、
  `test_multi_refine.py`（接口接线 / 落盘 / 三种 refine / elapsed_s）、
  `test_roadfit_eval_budget.py`、`test_frontend_refine_control.py`。
- **重启服务后的真实端到端验证**（同一批 12 张图、默认 `auto`、不传 `refine`）：

  | 项 | 结果 |
  |---|---|
  | 主请求 | HTTP 200 / **158.5 秒**，`refine_pending=true` |
  | 后台精修 | **3.0 秒**完成（路网缓存命中 + 评估点抽样） |
  | `road_fit` | `ok=true, mode=verify`，平移 `东0 / 北0`，贴路中位 25 m |
  | GeoJSON 首点 | 精修前后都是 `[121.3822816, 31.210834]`，**一字未动** |
  | 附带字段 | `stitch` / `n_shots=12` / `stitched_url` 都在后台结果里 |
  | 落盘 | `last_stitch.json` 的 `road_fit.mode=verify`，核对页不会退回粗结果 |

  对比改造前手动 `refine=on` 的 263.8 秒同步等待：现在粗结果先出，
  3 秒后自动补上路网校验与几何路名，且不会误平移拼接画布。
- 零改动实测与前后对照留在 `output/verify_*.json`（该目录已 gitignore）。

### 仍然存在的边界

- 能力③「道路几何」取数仍未经过 `app/maps/registry` 注册表，
  `MAP_ROAD_GEOMETRY_ORDER` 目前只在 `/api/health` 上报（见 §22.31）；
  公共 Overpass 镜像挂掉时也不会按「配置哪个就用哪个」明确报错。
- 能力②「路名归属」仍只有高德一个在线实现；不过精修成功时结果走
  `road_network` 几何判定，不依赖高德配额。
- 同步 `refine=on` 在大画布上依旧是分钟级，靠前端开关和状态提示告知用户，
  并没有把同步路径本身变快（只有后台路径复用了中间结果）。
- `app/trajmap.py` 仍直接使用 `TMap.*`，未走底图适配层。

## §22.37 未采纳的精修路线可视化（2026-11-09）

**问题**：三道护栏（锚点一致性 / 位移限幅 / 改善不足）否决平移时，旧代码
直接把 `shift_e_m/n_m` 归零，搜索出的最优解随之丢失。用户只能看到一句
「路网精修未采纳」，无从判断护栏拒得对不对（16:43 那次 6 图实测：可靠
锚点残差 95 m → 179 m 被拒）。

**做法**：

- `RoadFit` 增加 3 个可选字段；三道护栏统一走 `_reject()`，输出位移归零的
  同时保留候选位移与候选贴路距离。采纳成功时 `to_dict()` 键集不变，旧契约
  完全保持。
- `Pipeline._candidate_shift_px()` 按 `M⁻¹·d` 把米制候选位移换算成截图像素
  位移。配准是 `f(p) = M·p + t`，精修想让同一像素轨迹对应到 `f(p)+d`；以
  固定截图底图为参照，候选位置应当画在 `p + M⁻¹d`。前端无需知道尺度、
  旋转或 CRS，也不必多传一份折线；矩阵退化时返回空字典，前端不画线。
- 前端「原始截图」叠加层用橙色虚线 `#ff8a3d` 画候选轨迹 + 浮动图例；
  同步 `refine=on` 被拒时不会启动后台轮询，所以状态条与路网卡片也补了
  说明（此前这条信息只藏在 `warnings` 里，页面上完全看不出来）。
- 多图默认 `verify` 档不产生候选解，不会画线；换图时图例主动清空。

**验证**（重启服务后，同一批 6 张图、同步 `refine=on`，两次上传顺序不同）：

| 顺序 | 精修判定 | `road_fit` 关键字段 |
|---|---|---|
| A | **采纳** | `ok=true`，平移 东−20 / 北−175 m，贴路 64.5 → 9.3 m，**没有** candidate 字段 |
| B | **否决** | `ok=false`，输出位移 0，锚点残差 100 → 187 m；`candidate_shift_e_m=-20`、`candidate_shift_n_m=-165`、`candidate_median_after_m=9.2`、`candidate_shift_px=[-10.06, 82.96]` |

顺序 B 的像素偏移是西约 10 px / 南约 83 px，与配准矩阵 y 轴翻转一致；
结果已落盘到 `last_result.json` / `last_stitch.json`，前端刷新即可看到
橙色虚线。两次请求各约 73 秒，响应留在 `output/verify_candidate_line*.json`
（该目录已 gitignore）。**上传顺序会改变拼接与锚点残差，进而改变护栏
判定**——这也是「候选线」有实际价值的原因：同批图既可能采纳也可能否决。

新增 `tests/test_roadfit_candidate.py`（7 项：三道护栏保留候选、采纳时
键集不变、逆矩阵换算、退化矩阵降级、`_road_refine` 接线），
`tests/test_frontend_refine_control.py` 追加 2 项静态契约；全量
**274 passed**。

## §22.38 「轨迹横穿街区」的根因澄清与处置（2026-11-09）

### 起因

一张单图截图（1260×2720，2.38 m/px）识别后轨迹横穿同仁医院与街区：
置信度 53%、`RMSE 270.8m`、9 内点 / 30 离群，路网精修被护栏否决
（可靠锚点残差 52m → 222m），候选平移西 55m / 南 175m 能把贴路中位
**78.1m → 6.4m**。直觉上像是「配准没定准 + 护栏过于保守」。

### 离线复算：推翻这个直觉

用 `anchors_used` 重建 `north_up`（尺度 2.7385 m/px），再只用 4 个可信
POI/车站锚点重新拟合：尺度 2.7538（差 0.6%），中心点只差
**东 6.8m / 北 39.7m**。也就是说**配准的尺度与绝对位置基本是对的**。

候选平移后，4 个可信锚点残差一致涨到 199 / 207 / 254 / 334m；
几何路名从「威宁路、仙霞路、北虹路、中环路…」的杂乱就近吸附，变成
几乎整条「仙霞路」。这正是文档开头反复警告的假解：一条**笔直**轨迹整体
搬到另一条平行主干道上，贴路距离漂亮但街区错了。护栏拒得对。

把原图轨迹区域放大后看到了真正的原因：截图里的轨迹本身就是一条带白色
箭头的**起终点连线**（x 271→968、y 1295→1398，直线度约 0.97），
没有沿任何道路。缩放评估只看「米/像素」，2.38 m/px 属街道级，
`zoom_too_low=false`、`has_road_names=false`，完全抓不到「轨迹本身没有
形状信息」这种情况。所以这不是配准 bug，是输入信息量不足——算法无法
凭空恢复曲折走向。

另外 9 个内点里混进了 5 个 OCR 残片路名（「云岭东」「虹桥基」「水城」），
其中一条残差 616m 仍算内点，因为 `_KIND_TOLERANCE["road"]` 有 800m 的
米制下限容差；`RMSE 270.8m` 因此被拉高，诊断数字也不够干净。

### 处置（不放宽护栏）

1. **低信息量输入检测**：`_track_straightness()` = 端点弦长 / 折线总长，
   `_track_lacks_shape_info()` 要求三条件同时成立——直线度 ≥0.80、
   轨迹框长宽比 ≥4、且 `has_road_names=false`。命中时在 `warnings`
   明确说明「只能定位到街区级、无法还原沿路走向」，并给出三条可操作建议
   （重截完整轨迹曲线页 / 放大到能看到道路名 / 填写路名约束）。
   `diagnostics.track_straightness` 一并输出。

   **为什么不只看直线度**：端到端第一次验证暴露了口径问题——原始 731 点
   折线因提取抖动直线度只有 0.83（简化后 60 点是 0.859），阈值 0.93
   根本不触发；但 12 图拼接的真实轨迹直线度也有 0.869。所以补上
   「长宽比 ≥4」（起终点连线实测 7.5，12 图拼接 2.0）和
   「没有街道名锚点」两个条件，才把两类输入分开。
2. **候选位置交还用户**：状态条下方新增「在地图上查看未采纳的贴路位置」
   按钮，点击把候选轨迹画到底图上，再点切回。候选坐标由主轨迹 GCJ-02
   折线 + `candidate_shift_e_m/n_m` 的刚体平移一阶换算得到（几百米量级
   误差 <1m）。**只改显示层**，坐标 / GeoJSON / 里程 / 置信度都不变。
3. 顺带修掉状态条「路网精修未采纳（路网精修未采纳（…））」的嵌套重复：
   后端 `notes[0]` 已带该前缀，前端不再包第二层。

**刻意没做的事**：没有让配准置信度调制 `anchor_guard_m`、也没有自动采纳
候选。本例两类证据冲突时，自动采纳会把轨迹搬到错误的平行主干道；
把选择权交给用户比替他猜测更安全。若后续积累了「锚点系统性偏移」
的可信判据（例如残差向量方向高度一致），再重新评估护栏权重。

### 收尾修复：被否决的后台精修也要落盘

`server._run_refine` 早期只在 `road_fit.ok` 为真时才写 `last_result.json`
（当时理由是「未采纳就不必写诊断信息」）。候选路线可视化上线后这个前提
失效了：**护栏否决时** `candidate_shift_*`、拒因与形状提示只存在后台
payload 里，不落盘的话用户刷新页面，橙色虚线和「查看候选位置」按钮
会一起消失。现在去掉 `ok` 门槛，只保留 `shot_id` 一致性保护（防止
精修期间用户又上传新图后被旧任务覆盖）；否决时输出位移为 0，
落盘只是补齐诊断信息，不改坐标。新增
`tests/test_multi_refine.py::test_rejected_background_refine_is_persisted`。

### 验证

新增 `tests/test_track_straightness.py`（9 项：直线 / 转弯 / 缩放平移不变 /
退化输入 / 上界 / 三条件门槛 / 真实样本触发 / 主流程接线），
`tests/test_frontend_refine_control.py` 追加 3 项静态契约（去重文案、
候选切换按钮与换算），`test_multi_refine.py` 追加 1 项落盘；全量
**286 passed**。

真实单图端到端（`1794226850_77de82b4`，主请求 34.9 秒、后台 4 秒）：

| 项 | 实测 |
|---|---|
| `diagnostics.track_straightness` | `0.83`（原始 731 点折线） |
| `bbox_px` / `has_road_names` | `[269,1313,710,95]`（长宽比 7.5）/ `false` |
| 形状提示 | 主请求与后台结果**都带** 1 条「起终点连线 / 只能定位到街区级」 |
| 后台 `road_fit` | `ok=false`，候选 东−55 / 北−175 m，贴路 78.1 → **6.4 m**，`candidate_shift_px=[-19.73, 62.79]` |

候选字段齐全，前端「在地图上查看未采纳的贴路位置」按钮可正常构造候选坐标。
落盘修复后再次端到端（`1794227434_0877ed2d`，主请求 28.0 秒）：
`last_result.json` 的 `shot_id` 与本次一致、形状提示为 `true`、
`road_fit.ok=false` 且 `candidate_shift_px=[-19.73, 62.79]`、
候选 −55/−175 m、贴路 6.4 m —— **刷新页面仍能看到橙线与切换按钮**。
产物留在 `output/verify_straight_final.json`（已 gitignore）。

## §22.39 本轮收尾：提交清单、验收状态与运行副作用（2026-11-09 22:00）

§22.36–§22.38 逐项写了实现与实测，本节只把「现在到底停在哪、怎么验收、
哪些文件不该进提交」收口，避免下一个人接手时从 4600 行里翻。

### 提交清单与 HEAD

| 提交 | 内容 |
|---|---|
| `b319d5d` | 后台精修复用主请求的 `RefineContext`，不再重跑 OCR / 配准 / 在线锚点 |
| `5a341ae` | 多图接后台精修，默认 `verify` 档只校验 + 命名不平移；修两处落盘竞态 |
| `80f114f` | 位移搜索抽样提速、`elapsed_s` 上报、前端「贴路校正」开关 |
| `0e6c086` | 未采纳的精修路线用橙色虚线画到原始截图 |
| `c18d025` | 低信息量直线输入识别 + 候选位置显示层切换 + 拒因文案去重 |
| `3230c9a` | 判据改为三条件组合（直线度 / 长宽比 / 无路名锚点），端到端实测触发 |
| `445ce00` | 护栏否决的后台精修结果也要落盘，刷新后橙线不丢 —— **代码 HEAD** |

文档收尾（本节与 README 快速开始一节）单独提交，不夹带运行数据。

### 当前服务与验收入口

- 启动：`PYTHONPATH=src <Python312> -m app.server --host 127.0.0.1 --port 8848`；
  `MAP_GEOCODING_ORDER=tianditu`（单源严格模式），health 里天地图 probe ok；
  高德 `/v3/place/text` 仍报 `USER_DAILY_QUERY_OVER_LIMIT`，其余高德接口正常。
- **刷新页面即可验收最后一张单图，不用重新上传**：`warnings` 里有「起终点连线 /
  只能定位到街区级」提示，状态条下有「在地图上查看未采纳的贴路位置」按钮，
  原始截图叠加层有橙色虚线 `#ff8a3d` + 图例。
- `data/last_result.json` 关键字段：`diagnostics.track_straightness=0.83`、
  `bbox_px=[269,1313,710,95]`、`has_road_names=false`、`road_fit.ok=false`、
  候选 东 −55 / 北 −175 m、`candidate_shift_px=[-19.73, 62.79]`。
  候选只改显示层，坐标 / GeoJSON / 里程 / 置信度一律不变。

### 运行副作用清单（哪些不该进提交）

| 路径 | 状态 | 说明 |
|---|---|---|
| `data/lexicon.json` | **tracked，但按约定不提交** | 识别时地名词库自学习（权重 / 计数 / 城市）。收尾时工作区唯一 diff 就是它（+475 / −48 行），提交会夹带真实地址数据，并让工作区长期不干净 |
| `data/last_result.json`、`data/last_stitch.json`、`data/uploads/` | gitignore | 最近一次识别结果与上传原图；刷新页面靠它们恢复候选线与按钮 |
| `data/cache/` | gitignore | 路网 / 岸线 / 路线校验缓存，见下节 |
| `output/` | gitignore | 端到端验证响应、pytest 日志 |

### 新踩的坑：识别会覆写城市级岸线缓存，打挂水岸测试

22:02 全量测试从 **286 passed** 变成 `4 failed, 282 passed`，4 项全在
`tests/test_waterguard.py`（`KeyError: '黄浦江'` / 岸线几何断言）。**不是代码回归。**

原因：`get_waters()` 按城市名缓存 24 小时（`data/cache/waters_上海.json`）。
21:52 那次识别联网拉河时 Overpass 只返回「苏州河」，`_store_cached()` 把整份
缓存覆写成 `{"waters": {"苏州河": ...}}`。测试用
`get_waters(..., allow_network=False)` 命中这份「新鲜但不完整」的缓存，
既拿不到黄浦江，也不会走到 `data/waters.json` 的 `bank_fallback` 兜底。

恢复（已实测）：

```bash
mv data/cache/waters_上海.json data/cache/waters_上海.json.bak   # 别直接删，留证据
python -m pytest tests/test_waterguard.py -q                       # 6 passed（走兜底折线）
```

后续建议（本轮**未改代码**）：`_store_cached()` 按河名合并旧缓存、缺河时
不覆盖已有完整数据；`tests/test_waterguard.py` 改用 `tmp_path` 独立缓存目录，
不再与运行目录共享。在此之前，跑全量前先确认该缓存是否完整。

### 测试与遗留

- 测试命令：`/d/soft/Python311/python -m pytest -q`
  （WorkBuddy 内置 3.13 没装 pytest）；本轮三个提交时全量 **286 passed**。
- 文档收尾时移走被覆写的岸线缓存后**再跑全量仍是 286 passed**（263.96 秒），
  确认 4 项失败纯属运行缓存污染，日志留在
  `output/pytest_final_20261109.txt`（已 gitignore）。
- 遗留：③ 道路几何取数仍未接 `app/maps/registry`（`MAP_ROAD_GEOMETRY_ORDER`
  只在 health 展示）；② 在线路名归属仍只有高德实现；`app/trajmap.py` 仍直接用
  `TMap.*`；同步 `refine=on` 大画布仍是分钟级（只有后台路径复用了中间结果）；
  高德 JS 底图未经真机验证；`waterguard` 缓存覆写问题见上节。

---

## §22.40 准确率优化轮：从「自洽分」到「可证伪指标」+ 数据飞轮防毒（2026-11-10）

### 这轮为什么不是继续调 RANSAC

用户的问题是「这个功能能继续从哪些方向深挖吗，提高识别准确率吗」，
随后明确指令：「**那你按你罗列的开始一步一步优化吧，完成一步就提交下代码，
直到全部优化完成**」。

先纠正方向：配准本身在街道级截图上已比较准，继续调 RANSAC 迭代数、
KD 树阈值、贴路权重属于边际收益饱和，而且**无法证伪**。§22.38 的反例是：
把一条起终点直线整体平移到平行主干道后，贴路中位距离 78→6.4 m、
锚点残差和 confidence 一起变好，真实位置却更错。所有自洽指标都会
奖励这个假解。因此第一步不是改算法，是建**外部真值**。

### 八个阶段与提交清单

| 任务 | 提交 | 内容 |
|---|---|---|
| #64 | `ee892ce` | `src/app/eval.py`（起终点误差 / 离散 Fréchet / 里程相对误差 / 合成基准）、`src/tools/eval_accuracy.py`、`tests/test_eval_metrics.py`；产出 `output/eval_baseline_p0.json` |
| #65 | `30b77e0` | `src/app/anchor_quality.py`：剔除 OCR 残片 road 锚点、`rmse_reliable_m` 只由点状锚点计算、`systematic_bias()` 输出残差方向相干性 |
| #66 | `b48a3f8` | `GeocodeResult.source` / `AnchorPair.source` / 词典条目 source 贯通；跨源同名候选 >200 m 只降 `match_quality` 不删；`src/tools/check_lexicon_sources.py` 按源偏差报告（真实词典发现 11 组跨源冲突，near/far 分类） |
| #67 | `611fbc8` | `src/app/input_quality.py`：截图里程 vs 几何里程、外部路由轻权乘子、`assess_input_quality()`；缺失外部证据一律不惩罚 |
| #68 | `14a449b` | 前端上传前分辨率/大小预检、结果页展示尺度/路由/输入质量；`node --check` 校验内联脚本 |
| #69 | `580d315` | `src/app/mapmatch.py`：默认关闭的 HMM 备选显示层，低信息量 / 吸附距离 / 长度剧变三道护栏 |
| #70 | `f5cfedf` | `src/app/posegraph.py`：默认关闭的位姿图全局平差，用全部重叠边加权最小二乘替代 BFS 链式累积 |
| #71 | 本提交 | 置信度校准 + 词典防毒 + 文档与全量回归（详见下） |

### A/B 证据（同一 seed0，合成基准 P50）

| 指标 | p0 基线 | p1 内点清洗后 | p2 源统一后 | 方向 |
|---|---:|---:|---:|---|
| `center_error_m` | 13.813 | **11.505** | 11.505 | 越小越好 |
| `precision` | 0.778 | **1.000** | 1.000 | 越大越好 |
| `recall` | 1.000 | 1.000 | 1.000 | 持平 |
| `start_m` / `end_m` | 14.949 / 13.374 | **13.308 / 12.460** | 13.308 / 12.460 | 越小越好 |
| `frechet_m` | 20.054 | **17.898** | 17.898 | 越小越好 |
| `scale_rel_error` | 0.004 | 0.002 | 0.002 | 越小越好 |

p2 与 p1 完全一致，说明「锚点源贯通 + 跨源降权」在合成基准上**无退化**；
它的收益体现在真实词典的基准一致性诊断上，不是合成噪声能覆盖的。
`eval_accuracy.py --compare` 里 `precision/recall` 属于越高越好指标，
完全相等输出「持平」而不是「变差」。

### 阶段 #71-A：置信度校准（`src/app/calibration.py`）

- 每个真值样本 = `(raw_confidence, label)`；`label_from_metrics()` 只用
  `start_m / end_m / frechet_m / length_rel_error` 四个外部误差，
  门槛默认 300 m / 300 m / 400 m / 0.30。缺任一指标 = 0（宁可保守）。
- PAVA 保序回归得到单调不减的分段经验率，分段间线性插值、端点外常数外推；
  再与原始分做收缩混合 `w = n / (n + 20)`，样本少时校准影响自然变小。
- **安全降级是硬契约**：文件不存在 / 样本 < 8 / JSON 损坏 / version 不符 /
  折点非单调 → `load_calibration()` 返回 `None`，
  `apply_calibration()` **逐值原样返回**。没做过真值评测的部署，
  置信度与改动前完全一致，不会因缺文件被偷偷改写。
- `CalibrationStore` 按 `(mtime_ns, size)` 热加载，改模型不用重启服务。
- 接入点：`pipeline._score_confidence()` 最后一步；`diagnostics.
  confidence_calibration` 给出 `enabled / status / n_samples / path`。
- CLI `src/tools/calibrate_confidence.py`：`--case "pred.json|truth.gpx"`
  （用 `|` 分隔，避开 Windows 盘符里的冒号）或 `--cases` 清单；
  样本不足时不写文件并返回 1；`--show` 查看当前状态。
  `data/confidence_calibration.json` 已加进 `.gitignore`（由含真实轨迹的
  样本派生，属个人数据）。

### 阶段 #71-B：词典防毒（`src/app/lexicon_guard.py` + `Lexicon` 会话）

问题：在线地理编码 / OSM 冷启动在**配准之前**就写 `data/lexicon.json`，
那时还不知道这次识别准不准。原来的 `remember()` 只按坐标近似累加
`weight/count`，低置信错误结果会一次次加权，形成「越用越偏」。

设计（纯函数 `evaluate_write()` + `Lexicon` 会话）：

1. `pipeline.run()` 外层 `begin_capture()`；所有**自动**写入只排队不落盘，
   等 `_run_impl()` 返回后用最终 `confidence` 和自洽诊断 `flush_capture()`
   统一裁决；异常路径 `discard_capture()`。用户 `/api/feedback` 的
   `source="user"` 不排队、立即写、标记 `verified`。
2. 自洽判据 `_result_is_self_consistent()` 只用独立证据：
   `input_quality.level != reject`、`scale_check.consistent`、
   `anchor_bias.systematic == False`、低缩放且 `position_ok == False` 否决。
   **不看贴路距离/锚点残差**（§22.38 的假解上它们同样变好）。
3. 其余否决：置信度 < 0.35；地名不像地名（长度 < 2 / 无文字 /
   「3公里内免费」这类 UI 文案）；坐标非有限或越界；与已核验或高权重的
   **异源**条目相距 200 m~10 km（>10 km 视为同名异地，保留多候选）。
4. 未核验条目权重封顶 3.0；`lookup().quality = weight/5` 因此永远 < 1.0，
   自动候选盖不过用户核验点位。用户纠正把条目升级 `verified=true`
   并解除上限；自动写入命中已核验条目时不加权也不封顶拉低。
5. 非流水线调用（无最终置信度）不直接丢弃：boost 压到 0.2 且同样封顶，
   冷启动仍可积累但积累慢、权重有限。
6. `diagnostics.lexicon_write_guard` 记录 `pending / written / rejected /
   reasons / confidence / self_consistent / input_ok`；`stats()` 新增
   `verified_points / auto_points`。

> 兼容性代价：会话期间排队的坐标不会立刻出现在 `lookup()` 里，
> 同一识别内同名锚点可能多打一次在线请求。项目里 `targets` 按名字去重，
> 影响有限；换来的是「错误结果根本不进词典」，比回滚式写入更不容易出错。
>
> **存量词典迁移**：旧的 `data/lexicon.json` 条目没有 `verified` 字段，
> 一律按「未核验」看待（`stats().auto_points`），但已有 `weight` **不会被
> 回溯下调**；只有新的自动写入才受 3.0 封顶。用户在地图上纠正一次，
> 该条目即升级为 `verified`。存量里 weight ≥ 3 的条目在近距异源冲突判据里
> 按「高权重」处理，会拦下新候选——这是刻意的保守方向，宁可不沉淀也不要
> 引入基准不一致。

### 开关与诊断字段

| 开关 / 字段 | 默认 | 说明 |
|---|---|---|
| `MAP_MATCH_ENABLED` | 关 | HMM 备选显示层 `map_matched`，不改主结果 |
| `POSE_GRAPH_ENABLED` | 关 | 多图拼接全局平差，只改拼接画布平移 |
| `RIDE_CONFIDENCE_CALIBRATION` | 词典同目录 | 校准模型路径，文件缺失 = 未校准 |
| `diagnostics.confidence_calibration` | `uncalibrated` | 校准状态与样本数 |
| `diagnostics.lexicon_write_guard` | 每轮一份 | 本轮词典写入裁决审计 |

前两个开关走 `.env` 热读，默认 `False`，保证既有线上验收结果不变；
低信息量直线输入仍由三道护栏强制跳过。

### 本轮改动的文件

新增：`src/app/eval.py`、`src/app/anchor_quality.py`、
`src/app/input_quality.py`、`src/app/mapmatch.py`、`src/app/posegraph.py`、
`src/app/calibration.py`、`src/app/lexicon_guard.py`、
`src/tools/eval_accuracy.py`、`src/tools/calibrate_confidence.py`、
`src/tools/check_lexicon_sources.py`、
`tests/test_eval_metrics.py`、`tests/test_anchor_quality.py`、
`tests/test_anchor_sources.py`、`tests/test_input_quality.py`、
`tests/test_frontend_input_quality.py`、`tests/test_mapmatch.py`、
`tests/test_posegraph.py`、`tests/test_confidence_calibration.py`、
`tests/test_lexicon_guard.py`。
修改：`src/app/pipeline.py`、`src/app/register.py`、`src/app/geocode.py`、
`src/app/config.py`、`src/app/multishot.py`、`src/web/index.html`、
`README.md`、`.gitignore`、`docs/HANDOVER.md`。

### 运行副作用（提交纪律不变）

`data/lexicon.json`（tracked 但按约定不提交）、`data/last_result.json`、
`data/last_stitch.json`、`data/uploads/`、`data/cache/`、
`data/confidence_calibration.json`（本轮新增，已 gitignore）、`output/`。
每次只 `git add` 明确的源码 / 测试 / 文档路径，不 `git add -A`。

---

## §22.41 环境变量收敛：同一功能只用一个变量名（2026-11-10）

**动因**：用户指出「同一个功能不要使用超过一个的环境变量，太容易混乱了」。
排查发现两处历史遗留的双/多变量名，读代码时无法判断哪个才是真正生效的，
`.env` 里写错名字也不会报错，只是静默失效。

### 现状（唯一生效的两个变量名）

| 功能 | 唯一环境变量 | 当前 `.env` 取值 |
|---|---|---|
| 在线地理编码源顺序 | `MAP_GEOCODING_ORDER` | `tianditu` |
| 天地图 tk | `TIANDITU_API_KEY` | 已配置 |

其余 `GEOCODE_PROVIDER_ORDER`、`TIANDITU_KEY`、`TIANDITU_TK` **全部下线**，
写进 `.env` 不再有任何效果。其余能力（`MAP_ROADNAME_ORDER` /
`MAP_ROAD_GEOMETRY_ORDER` / `MAP_ROUTE_VALIDATION_ORDER`）本来就各只有一个名字，不变。

选择保留名的依据：`.env` 与全部既有测试实际写的就是 `MAP_GEOCODING_ORDER`
和 `TIANDITU_API_KEY`，改动零破坏性——不需要改 `.env`、不需要用户迁移配置。

### 改动清单

- `src/app/maps/registry.py`：`ORDER_ENV[GEOCODING]` 由
  `("MAP_GEOCODING_ORDER", "GEOCODE_PROVIDER_ORDER")` 收敛为单元素；节首注释删掉
  「兼容旧名」表述，明确「每个能力只对应一个环境变量」。
- `src/app/config.py`：删除 `GEOCODE_PROVIDER_ORDER_ENV`、`TIANDITU_KEY_ALIASES`、
  死属性 `online_provider_order`（全仓零引用）、死常量 `ONLINE_PROVIDERS`；
  `TIANDITU_KEY_ENV` 定为 `"TIANDITU_API_KEY"`，`tianditu_key` 由「遍历三个别名」
  简化为单次读取。
- `src/app/maps/geocoding/providers.py`：`TiandituProvider.credential_aliases`
  由三别名收敛为 `("TIANDITU_API_KEY",)`。
- 注释：`geocode.py`、`maps/geocoding/__init__.py` 去掉「兼容 GEOCODE_PROVIDER_ORDER」。
- 测试：4 个文件的 `_ENV_NAMES` 清理；删除 `test_legacy_order_env_still_honoured`、
  把 `test_credentials_resolve_alias_in_order` / `test_alias_and_defaults` 改为
  单变量名语义，registry 旧名用例改用新名；
  `test_tianditu_geocode.py::test_provider_order_and_fallback` 的 `extra_env` 改用新名。
- 文档：README 配置段与 `.env` 模板删除三别名/旧名兼容说明；§22.28、
  §22.31 加「历史记录、变量已过时」标注；本节为现行唯一依据。

### 验证

- 相关 4 个测试文件 48 项全过；`.py` 全仓 grep 旧变量名 0 命中。
- 服务重启后 `configured_order(GEOCODING) == ('tianditu',)`，amap 不被调用
  （本月配额已耗尽，本就不该再尝试）。

---

## §22.42 词典永不增长：置信度封顶与写入门槛的互斥死锁（2026-11-10）

### 现象

用户清空 `data/lexicon.json` 后连续跑了约 10 张截图，词典始终不增长。
最近一次 `data/last_result.json` 的 `diagnostics.lexicon_write_guard` 给出
直接证据：

```json
{"pending": 6, "written": 0, "rejected": 6,
 "reasons": {"low_confidence": 6}, "confidence": 0.2,
 "self_consistent": true, "input_ok": true}
```

候选**捕获到了**（6 个），是裁决环节全部拒绝，不是没有识别、也不是
`save()` 失败。

### 根因：两个独立设计的常量形成死锁

| 位置 | 行为 |
|---|---|
| `pipeline._score_confidence` | `if zoom and not zoom["position_ok"]: conf = min(conf, 0.2)` |
| `lexicon_guard.MIN_AUTO_CONFIDENCE` | 自动写入要求 `conf >= 0.35` |

`_assess_zoom_level()` 里 `position_ok = bool(has_road_names or street)`：
截图没有街道级锚点、用户也没填路名约束时就是 `False` —— 这是街区级
/城市级截图的**常态**，也是冷启动最常见的一类输入。于是 `0.2 < 0.35`
恒成立，这类截图永远写不进词典；空词典 → 无本地基准 → 更难有街道级
锚点 → `position_ok` 更容易为 False → 更写不进，形成冷启动死锁。

更根本的**逻辑错配**：`Lexicon.remember()` 沉淀的是在线地理编码
（tianditu / osm）返回的 **POI 坐标**，属于外部权威源数据；而 confidence
衡量的是「这一整条轨迹还原得准不准」。因为轨迹没配准好就丢弃天地图
给的正确地名坐标，把两件独立的事绑在了一起。

顺带发现 `_commit()` 在裁决**之前**就 `self._data.setdefault(key, [])`：
被拒绝的写入会在内存里留下空 key，使 `stats()["entries"]` 虚高、磁盘
文件却没变化（实测 `last_result` 里 entries=12 而 `/api/health` 只有 8），
看起来像「写进去了又消失」。

### 修复：锚点级证据通道（不放松防毒）

- `lexicon_guard.py`：新增 `MIN_ANCHOR_CONFIDENCE = 0.15` 与
  `evaluate_write(..., anchor_evidence=False)`。`conf < 0.35` 时，只有
  `anchor_evidence=True` 且 `conf >= 0.15` 才放行，且 `boost` 压到
  `UNKNOWN_CONFIDENCE_BOOST`(0.2)、`weight_cap` 仍为 `AUTO_WEIGHT_CAP`(3.0)，
  reason = `low_conf_anchor_capped`。崩塌/早退结果（conf 趋近 0 或 nan）
  仍一律拒绝；无证据的候选仍 `low_confidence` 拒绝。
- `pipeline.py`：`run()` 每轮重置 `_last_anchor_keys`；`_run_impl()` 在算出
  confidence 后把 `reg.inliers` 的地名规范化存入该集合；`flush_capture()`
 传入。早退路径没有内点证据 → 自动写入全部拒绝（保持安全）。
- `geocode.py`：`flush_capture(..., anchor_keys=None)` 按 key 逐条传
  `anchor_evidence`；report 增 `anchor_keys` 便于审计；`_commit()` 的
  `setdefault` 移到裁决通过之后，杜绝空 key 虚高。

**保留的不变量**：`source="user"` / `verified` 仍绕过一切门槛；名称合法性、
坐标合法性、`self_consistent`、`input_ok`、异源 200m~10km 冲突、未核验权重
封顶 3.0 全部不变。低置信通道只是让「确实参与配准的地名坐标」以最慢速度
冷启动，绝不会因为一次轨迹没配准好就写入垃圾。

### 实测（`data/uploads/` 用户真实截图，新代码）

| 截图 | confidence | guard | 结果 |
|---|---|---|---|
| `1794277357_ca2d0f79.jpg` | 0.20 | pending 6 → **written 2**，reasons `{low_confidence: 4}` | 死锁解除，仍拒 4 个无证据候选 |
| `1794276987_201446e0.jpg`（34 km 失败例） | 0.20 | pending 0 → written 0 | 防毒照常挡住 |
| `1794276937_8bb61def.jpg` | 0.79 | pending 6 → **written 6** | 正常高置信路径不受影响 |

磁盘词典 **8 → 10 条**。注意「增长」有两种形式：新增 key（`entries` 数）
与已有 key 加权（`weight`），后者不增加条目数但同样是积累。若想快速
建立高可信基准，用 `/api/feedback` 提交人工核验（`source="user"` 全额
boost、标 `verified`、绕过自动门槛）。

### 验证

- `tests/test_lexicon_guard.py` 22 项全过（新增 6 例：capped 冷启动、
  无证据仍拒、崩塌仍拒、被拒不建空 key、flush 按 key 裁决、pipeline 端到端）。
- 全量 pytest 见本节提交记录。

---

## §22.43 轨迹模板（P0）：概览图独立通道 + 形状沉淀

### 起因与判断

用户提出：多图场景额外上传一张「原始概览单图」（缩放等级高、细节看不清），
希望程序逐渐进化到「后续同类轨迹只放一张高缩放等级图也能识别」。

只读调研结论：

1. `/api/recognize_multi`（`server.py`）把所有 `files` 交给 `multishot.stitch`
   拼成一张画布，再走**一次**单图流程；概览图尺度差一个量级，不能混入。
2. OCR 为空 → `build_anchors` 返回 `[]` → `register()` 要求 ≥2 有效锚点
   （`register.py:881-887`）→ `pipeline.py:444-465` 早退，`confidence=0`；
   用户填 `road_names` 也救不回（`_apply_road_names` 在配准成功之后）。
3. `Lexicon` 的 key 是归一化地名、value 是点坐标；`lookup` 必须给名字，
   尺度每次由当次锚点重算（`_estimate_scale_prior`），历史点只派生
   `city_bounds / core_radius_m`。**所以「越用越准」只在同名地名再现时生效**，
   增加词典条目永远无法让无地名的图变准。
4. 全项目没有「保存成功轨迹几何供复用」的持久化；`data/last_result.json`
   只给核对页看且不进 pipeline，`RefineContext` 只在进程内。
5. §21.4 记录城市级底图 Chamfer 模板匹配失败（前 12 候选分数差 0.1%，
   位置差 900-1600 m），`simfit` ICP 同样失败。但「同类轨迹」的问题不是
   全城搜索，而是**在有限模板集合内对齐**，可行性高得多。

结论：需要第二种资产「轨迹模板」，lexicon 承担不了。

### 本轮改动

| 文件 | 改动 |
|---|---|
| `src/app/route_template.py`（新增） | `TemplateStore` / `match_shape` / 折线归一化 / 写入裁决 / 去重 |
| `src/app/server.py` | `overview` 独立字段（不进 `stitch_images`）；单图、多图成功后 `_capture_template`；`/api/health` 增 `templates` |
| `src/app/config.py` | `template_path = DATA_DIR / "route_templates.json"` |
| `.gitignore` | 忽略 `data/route_templates.json*`（含真实位置） |
| `tests/test_route_template.py`（新增） | 24 项 |
| `README.md` / 本节 | 数据飞轮从「三个资产」改为四个；新增用户侧说明 |

### 数据结构：为什么是「两条折线」而不是存配准矩阵

模板同时存 `px_points`（像素）与 `geo_points`（GCJ-02），各按弧长等距重采样
到 64 点、**一一对应**。这样模板本身就是一张「像素 → 地理」查找表：将来形状
对齐后按弧长插值即可出坐标，完全不必理解 `Registration` 的 `matrix/model`
格式；只匹配上其中一段时，也能只对那一段给高置信坐标。

其余字段：`shape`（质心原点 + bbox 对角归一）、`signature`（16 段有序方向角）、
`confidence / verified / hit_count / city / source / shot_id / length_px`。

### 形状匹配：同类但不完全一样

`shape_distance = 0.6·Chamfer + 0.4·逐点对应`。只用 Chamfer 会丢掉连接顺序
（反向变体形同虚设、交叉路线可能误判），只用逐点对应又对「中间多绕一段」
过敏。再叠三道：

* `signature_distance`：16 段有序方向角的平均环形差，`> 0.45 rad` 直接拒识；
* 滑窗：正反两个方向枚举连续子段（起点/长度各按 8 点步进，最短 60%），
  支持「新图只截一段」与「新图多绕一段」；
* `coverage`：匹配弧长占较短折线的比例，`< 0.60` 拒识，防止一小段碰巧相似。

窗口打分 `r + 0.03·(1-coverage) + 0.05·signature`，避免短子段归一化后
更容易拟合、把整条相同的轨迹也判成局部匹配。

实测（合成折线，120 点）：

| 变体 | residual | coverage | signature | 结论 |
|---|---|---|---|---|
| 同一条 | 8e-05 | 1.00 | 0.0001 | 命中 |
| 缩放 1.7 倍 + 平移 | 8e-05 | 1.00 | 0.0001 | 命中 |
| 反向 | 8e-05 | 1.00 | 0.0001 | 命中（`reversed=true`） |
| 只截中间 62.5% | 0.0288 | 0.7344 | 0.3197 | 命中 |
| 另一条不同波浪线 | 0.0751 | 0.9844 | 1.1250 | **拒识**（方向签名） |
| 只截 15% 碎片 | — | 过低 | — | **拒识**（覆盖度） |

注意：「另一条不同波浪线」的 residual 0.0751 原本低于 0.085 会被误收，
是方向签名把它挡住的——细长轨迹按 bbox 对角归一化会压扁纵向差异，
**单靠点集距离不够**。这是本轮实测踩到的坑，阈值不是拍脑袋定的。

### 写入裁决与去重

与 `lexicon_guard` 同源：`self_consistent`（复用 `_result_is_self_consistent`
的四项一票否决）、`input_ok`、`confidence ≥ 0.35`、坐标合法、点数 ≥ 10；
`verified / source="user"` 绕过自动门槛。形状几乎相同（去重 residual ≤ 0.045
且 coverage ≥ 0.92）不新增条目，只累加 `hit_count` 并在置信度更高时刷新折线；
上限 200 条，按 `(verified, confidence, hit_count)` 淘汰。

### P0 边界与后续

* **不改识别结果**：`find_similar` 只判定相似度、不产出坐标，主流程
  （`pipeline.py:444` 的无锚点早退）一字未动。服务由用户自行重启。
* **P1**：已在 §22.44 实施。注意真实概览图的失败点不是「配准失败」，而是
  「宽泛锚点主导」——所以兜底前移到在线地理编码**之前**，不是等配准失败。
* **P2**：概览图作为拼接画布的全局形状约束（挂 `_insert_seam_paths` /
  pose graph）；前端加概览图选择 UI（本轮只做后端通道）。
* **P3**：同类模板聚类去重、`/api/feedback` 核验提权、命中次数反哺置信度。

前提：概览图本身要能提取出 ≥10 点轨迹线（`pipeline.py:367`）；没有可读
比例尺时绝对尺度只能来自历史模板，结果只能标「参考级」；冷启动必须先
有若干次含地名的成功识别沉淀模板，因此是「逐渐进化」而非立刻生效。

### 验证

- `tests/test_route_template.py` 24 项：折线归一化、同形/缩放/反向/局部子段、
  异形与碎片拒识、写入裁决与去重、损坏文件降级、概览图不参与拼接、
  单图与多图沉淀、health 字段。
- 全量 pytest（Python 3.11）：**407 passed / 2 warnings（409.58 s）**，
  较 P0 前 383 passed 净增 24 项，无回归（2 个 warning 是既有前端测试写法提示）。
- 本轮**未启动、未重启 8848 服务**（按用户约定由用户自行重启）；接口验证
  只用进程内 `TestClient`，未触发真实 OCR / 在线地理编码。

---

## 22.44 P1：宽泛概览图的参考级定位（2026-11-10）

### 真实概览图实测：推翻了 §22.43 的两个假设

用户提供 `美团骑车截图8.jpg` / `9.jpg`（1260×2720，同一条徐汇滨江↔长宁
路线的**往返两个方向**）。全部为离线纯函数验证，未启动服务、未写运行期
模板库（临时库在 `tempfile`）。

| 检查项 | 结果 | 影响 |
|---|---|---|
| `extract_track` | 872 / 829 点，confidence 0.95，起终点徽章定位、拓扑校验通过 | ≥10 点门槛毫无问题，形状可用 |
| 离线 OCR | 42 / 46 条文本：行政区（普陀区/静安区/徐汇区…）、地铁线号（3/12/14号线、4号线内圈…）、高架高速（中环路/外环高速/南北高架路…）、大 POI（上海南站/大宁公园/西岸艺术中心…） | **不是 OCR 为空**，所以「无锚点早退」不是真实失败点 |
| 街道级路名 | 仅「龙吴路」1 条 | 宽泛锚点主导；这些标注位置与真实点位实测差 0.8-16 km（§21.4 表） |
| 两图形状互匹配（图8 作模板） | 原始全局 `shape_distance=0.2529`、`signature_diff=2.2963`；经反向+滑窗后 residual **0.06695**、coverage 0.9844、signature 0.3799、`reversed=True` → **命中** | 证明分段/反向设计必需；两图底图偏移、点数/bbox 都不同，属「同类但不完全一样」 |
| 跨缩放/跨路线假阳性 | 用另一次识别（2.04 km、61 点、2.56 m/px、`input_quality=reject` 的扁长直线）作模板查概览图8：residual **0.04381**、coverage 1.0、signature 0.3873 → **accepted** | 🔴 形状描述子对细长线区分度不足；不是「匹配不上」而是「太容易匹配上」 |

结论：概览图的问题不是没形状、也不是没地名，而是**地名锚点的空间精度
只有公里级**；继续在线地理编码就会把垃圾锚点喂进配准，重演尺度崩坏。
所以用户选定的方案是「**宽泛锚点直跳**」——在编码之前识别并绕开。

### 实现

* `src/app/template_fallback.py`（新）：`assess_broad_overview()` 纯文本判据。
  文本 ≥10、宽泛地名 ≥6 且占比 ≥0.35、街道路名 ≤2、用户未手填路名，
  五条同时成立才触发。`is_broad_name` 遵循既有约定**不把「大道」当大范围
  对象**（世纪大道仍是有效街道级参考）。`reference_confidence()` 封顶 0.45。
* `src/app/route_template.py`：
  - `ShapeMatch.mode` 区分 `query_in_template` / `template_in_query`——
    不区分就无法知道弧长投影该往哪个方向取段；
  - `project_to_geo()` 按弧长把新图 64 点插值到模板 GCJ-02 折线；
    `reversed` 换算回模板原方向取段，保证 GeoJSON 首点仍是起点；
    `template_in_query` 的子段外**夹到端点，绝不外推**；
  - `shape_has_enough_info()`：直线度 ≥0.80 且 bbox 长宽比 ≥4.0 的扁长直线
    禁用（与 `pipeline._track_lacks_shape_info` 同判据）；
  - `TemplateStore.locate()`：形状命中才产出坐标，双侧形状信息、模板里程
    100 m–200 km、`template_in_query` coverage ≥0.85、坐标合法性逐条裁决。
* `src/app/pipeline.py`：阶段 **2.5**（OCR 后、`_resolve_anchors` 前）插入
  兜底。`RidePipeline(cfg, *, templates=None)` 由 server 注入；**未命中
  继续原流程、未注入则整条通道关闭**，保证零回归。`_template_result()`
  产出 `source="template"` / `template_derived=True` / `template_match`
  诊断 / `input_quality.usable=False` / `_template_geojson()`
  （`quality=reference_only_unverified`）。
* `src/app/server.py`：`_templates` 提前到 `_pipeline` 之前创建并注入；
  `_capture_template` 拦截 `template_derived` → `template_derived_result`。

**为什么不会污染词典/模板/精修**：`usable=False` 让 `flush_capture` 拒绝
本轮全部自动写入（实测 `written=0`）；不设 `scale_ground_m_per_px` 让
`_need_background_refine` 直接返回 False；`template_derived` 让模板沉淀
直接短路。三道闸门避免「模板推导出坐标 → 又沉淀成模板」的自我强化。

### 端到端实测（真实图，OCR 文本用离线清单桩件替换，无联网）

```
confidence 0.414  track_points 64  crs WGS-84  source=template
template_match: residual 0.06695 / coverage 0.9844 / signature 0.3799
                reversed=True / span [0,63] / mode=query_in_template
input_quality.usable=False → lexicon_write_guard written=0
need_background_refine=False   refine_context=None
_capture_template → {"written": False, "reason": "template_derived_result"}
空模板库 → _resolve_anchors 确实被调用（回落），并给出明确 warning
```

### 边界与未做

* 🔴 **冷启动仍然存在**：`data/route_templates.json` 当时仍不存在；模板库
  为空时概览图只能回落原流程。需要先有若干次含街道名的成功识别。
* 参考级坐标**没有当次地理核验**，绝对尺度完全来自历史模板，只能当轮廓
  参考；接口必须靠 warnings / GeoJSON properties 明示，不能当精确轨迹。
* `/api/recognize_multi` 的 `overview` 仍是独立存档字段（P2 才用于拼接
  全局约束）；单图接口 `/api/recognize` 没有该字段，也不需要——P1 之后
  只把那一张概览图传到 `/api/recognize` 即可。
* 未做：概览图形状本身的模板变体（当前模板形状来自细节图/拼接画布）、
  旋转不变匹配、前端概览图选择 UI、模板 verified 人工核验提权（P3）。

### 验证

- 新增 `tests/test_template_fallback.py` **19 项**；P0 的
  `tests/test_route_template.py` 24 项全部保持通过（`mode` 字段为向后兼容
  的追加，未改变 `find_similar` 语义）。
- 全量 pytest（Python 3.11）：**426 passed / 2 warnings（248.10 s）**，
  较 P0 的 407 净增 19 项，无回归（2 个 warning 仍是既有前端测试写法提示）。
- **未启动、未重启 8848 服务**（用户自行重启）；未写项目运行期模板库。

---

## §22.45 P2：双上传入口、配对注册与形状域统一

### 要解决的问题

§22.44 交付后做的一次离线压力测试暴露了根因：**模板库的 key 与 query
不属于同一个形状域**。模板的 `px_polyline` 由 P0 的 `_capture_template()`
从细节图 / 拼接画布沉淀，而 `locate()` 只在「宽泛概览图」判据触发时被调用，
`query_px` 必然是概览图。拿 `data/last_stitch.json`（3.665 km、78 点、
bbox 长宽比 2.41、`input_quality=ok`）建模板查真实概览图9（829 点）：

```
accepted  residual 0.05357  coverage 0.8594  signature 0.2986
mode=template_in_query  → 产出 64 点 GCJ-02 坐标
```

两条**不是同一条路线**（北新泾局部 vs 市中心→上海南站十几公里）。
`shape_has_enough_info()` 只挡「直线度 ≥0.80 且长宽比 ≥4」的极端扁长直线，
aspect 2.41 顺利过关；coverage 0.8594 又刚好压在 0.85 之上。所以 P1 的
参考级结果存在把错误路线坐标套上去的风险，光调阈值治不了本。

### 设计：把形状域统一到概览图

上传语义反转——**概览单图是主入口，多图只在缩放过高时补传一次**：

| 分支 | 请求 | 行为 |
|---|---|---|
| 纯单图 | `file` | P1 路径一字未改；`template_capture` 照常、`paired_registration` 不存在 |
| 配对 | `file` + `details` | 细节图拼接配准拿真值 → 返回真值结果 → 把「概览形状 ↔ 真值」注册成 `overview` 模板 |
| 仅多图 | `/api/recognize_multi` | 旧端点保留，前端用「只用细节图」开关进入 |

之后形状接近（含反向、不同裁切）的概览图只传 `file` 即可命中，不必再带
多图——这就是用户要的「逐渐进化」飞轮。冷启动从「无限等待」缩短为
「每条路线首次补传一次多图」。

### 改动

* `src/app/route_template.py`
  - 常量 `SHAPE_DOMAIN_OVERVIEW` / `SHAPE_DOMAIN_DETAIL` / `MIN_PAIR_COVERAGE=0.85`；
  - `remember(..., shape_domain=...)` 写入模板，去重命中时会把旧 `detail`
    模板**升级**为 `overview`，否则它永远无法参与概览图定位；
  - `locate(..., shape_domain=None)` 指定域时只比对同域模板；旧模板缺字段按
    `detail` 处理（向后兼容，`find_similar` 语义不变）；
  - 新增 `TemplateStore.register_pair()`：把 `detail_px` 与 `geo_polyline`
    先按弧长重采样到 64 点，用 `match_shape(overview_px, truth)` 求对齐，
    再由 `project_to_geo()` 把真值投影到概览图的弧长参数化上，最后以
    「概览形状 ↔ 地理折线」沉淀。复用 P1 的投影逻辑，不重写几何。
  - `stats()` 增 `overview` 计数，`/api/health` 可直接看注册进度。
* `src/app/pipeline.py`：阶段 2.5 调用 `locate(shape_domain=overview)`，
  旧 `detail` 域模板不再参与概览图定位。
* `src/app/server.py`
  - `/api/recognize` 增 `details: list[UploadFile] | None = File(None)`；
    `file` 保持必填；新增 `_capture_paired_template()`；
  - 配对分支返回**多图真值结果**，`shot_url` 指向 `_stitched.png`（与
    `track_polyline_px` 同坐标系，避免前端叠加错位），单图放 `overview_url`，
    `payload` 增 `paired_registration` / `n_shots`；配对时**不**再沉淀 detail
    模板、**不**启动后台精修（result 来自画布而 `shot_path` 是单图，会串图；
    且拼接画布有链式累积漂移，本来就不该平移）；
  - 注册失败只写进 warnings，绝不影响识别结果。
* `src/web/index.html`：单个 `multiple` 控件拆成「① 概览单图（必选、单选）」
  +「② 街道级细节图（可选、多选）」两个入口，统一走 `/api/recognize`；
  新增「没有概览图，只用细节图」开关保留纯多图旧端点。`runPrecheck` 改为
  对两组文件合并检查。
* 测试：`tests/test_paired_registration.py` 18 项；`test_template_fallback.py`
  的 `_store()` fixture 默认按 `overview` 域建模板（P1 的查询本就是概览图）；
  前端两个测试文件的三处旧字符串断言同步更新为新语义。

### 几何正确性的两个坑

1. **临时真值模板必须先重采样**。`match_shape` 与 `project_to_geo` 都用
   「重采样后 64 点」的下标解释 `span`；直接喂原始 78 点 `detail_px` 会让
   `n_t` 与 span 下标错位。`register_pair()` 先做 `resample_by_arc`。
2. **不能写死 `mode == query_in_template`**。实测完全同形的两条折线，滑窗
   打分（带 `COVERAGE_PENALTY`）会偏好 `template_in_query`、span `[0,63]`
   、coverage 0.9844。写死 mode 会把正常配对全部拒成 `partial_route`。
   `coverage` 在两种 mode 下都表示「多图真值与概览图的覆盖是否一致」，
   单靠 `≥0.85` 就足以挡住只截局部的多图；`project_to_geo` 的子段外一律
   夹到端点、绝不外推。

### 实测

```
跨渲染（非均匀缩放 1.6x/1.25x + 抖动）弧长重采样 64 点：
  shape_distance 0.01435   signature_diff 0.1202
register_pair: written=True  residual 0.01565  coverage 1.0
  shape_domain=overview  source=paired
反向概览图 locate(domain=overview): reversed=True  coverage 0.9844
  residual 0.06695  → 64 点坐标
locate(domain=detail) 对同一条 overview 模板 → None（域隔离生效）
服务端 TestClient：纯单图 200 + template_capture / 无 paired_registration；
  配对 200 + shot_url=_stitched.png + overview_url + n_shots=2 +
  paired_registration.written=True + template_capture 不存在
```

### 边界与未做

* 🔴 参考级坐标**仍没有当次地理核验**：`source="template"`、confidence
  封顶 0.45、`quality=reference_only_unverified` 一条都没放松。配对分支
  本次返回的是多图真值（可靠），但以后纯单图命中的是模板投影（参考级）。
* 🔴 首次配对要求多图**覆盖整条路线**（coverage ≥0.85）；只截中间一段会
  返回 `partial_route` 并提示补截，否则模板地理折线不完整、子段外会被
  夹到端点，把错误固化。
* 🟡 配对注册沿用自动写入的四项防毒判据；`data/last_result.json`（19:11）
  与 `data/last_stitch.json`（10:55）分别因 `self_inconsistent=False` /
  `confidence=0.2 < 0.35` 都进不了模板库，这是冷启动偏严的真实原因。
* 未做：旋转不变匹配、`/api/feedback` 对模板的 verified 人工核验提权、
  同类模板聚类去重之外的置信度反哺、概览图给拼接画布做全局漂移约束
  （原 P2 计划，现让位给配对注册，顺延）。

### 验证

- 新增 `tests/test_paired_registration.py` **18 项**；P0 24 项 + P1 19 项
  保持通过（合计 61 项模板相关测试）。
- 全量 pytest（Python 3.11）：**444 passed / 2 warnings（335.13 s）**，
  较 P1 的 426 净增 18 项、无回归（2 个 warning 仍是既有前端测试写法提示）。
- **未启动、未重启 8848 服务**（用户自行重启）；所有验证为进程内
  TestClient / 纯函数级，运行期目录与模板库一律指向 `tmp_path`。

---

## §22.46 锚点纠正的可见反馈：提示浮层 + 点击标记（2026-11-11）

### 用户反馈与根因

「点了纠正，页面上没有任何提示」「在识别结果地图上点了，也没有任何交互反应」。
两处都是纯前端问题，服务端日志全是 200，看不出来：

1. `correctBox` 与 `#status` 都在**左栏**（锚点明细卡片 / 首卡片），
   而 `startCorrect()` 用 `$('map').scrollIntoView({block:'center'})` 把页面滚到
   右侧地图，左栏两个提示被一起滚出视野。
2. `submitCorrection()` 成功后只调 `showStatus()`（还是左栏），点击处没有任何
   标记、地图无状态变化，用户无法确认这一下是否生效；失败也只有一行文字。

### 改动

* `src/web/index.html`
  - `#map` 外包 `.map-shell`（`#mapShell`），同级新增 `#mapHint` 浮层，
    `scrollIntoView` 目标改成 `#mapShell`，保证提示与地图同时进入视野。
  - 纠正态：`.map-shell.correcting #map` 紫色 outline + 十字光标，
    并用 `.map-shell.correcting #map canvas { cursor:crosshair !important }`
    覆盖底图 SDK 的 grab 光标。
  - 新增 `setMapHint / hideMapHint / fmtLL / setCorrectingUI / cancelCorrect`；
    `startCorrect` 显示「请在地图上单击「X」的正确位置」，底图未就绪时改为
    错误提示。
  - `submitCorrection`：点击后**先** `basemap.showPin(lat, lon)` 落临时标记，
    浮层显示「正在保存… + GCJ-02 坐标」；成功显示「已记录 + 坐标 + 词典条目数
    + 下次识别生效」与「完成」按钮；失败显示错误并**重新进入纠正模式**，
    允许直接在地图上再点一次重试。点「完成」/「取消」/下一次「开始识别」
    都会清 pin 与浮层。
  - 提交语义不变：仍是「先点按钮、再点一次地图」，`/api/feedback` 的 body
    仍是 `{name, lon, lat, crs:'GCJ-02', city}`，`verified=true` 写入不变，
    本次轨迹不重算。
* `src/web/js/basemap.js`：两个 provider 各加 `showPin / removePin`
  （腾讯 `MultiMarker` + `removeLayer(id)`；高德 `Marker` + `map.remove`），
  `Adapter` 加 `showPin / clearPin` 与 `this.pin`；底图不可用时静默返回 false，
  不影响提交。
* 测试：`tests/test_frontend_correction_ui.py`（Node 驱动，执行 index.html 真实
  函数 + DOM/fetch 桩 + 两个 provider 的 pin 生命周期），覆盖进入纠正态、
  点击即时反馈、成功/失败/取消四态、pin 画与删、请求 body 与 GCJ-02 坐标。

### 边界与未做

* ✅ 当时仍没有撤销已写入点位的接口，已在 **§22.47** 补齐「一键清空全部
  人工纠正」。本次只解决「看得见」，数据语义到下一节才动。
* 🟡 成功提示需点「完成」关闭（或开始下一次识别），没有自动淡出——
  避免 `setTimeout` 拖住 Node 测试进程。
* 🟡 纠正不解锁轨迹模板：`position_ok` 只看街道名/手填路名，与用户点选坐标
  无关，§22.45 记录的 `low_confidence` 模板写入死锁仍在（P2 后续项）。

### 验证

- `node --check src/web/js/basemap.js`；主脚本 `new Function(...)` 编译通过。
- 前端组 `29 passed / 2 warnings`；新增测试单跑通过。
- 全量 pytest（Python 3.11）：**445 passed / 2 warnings（301.07 s）**，
  较 P2 基线 444 净增 1 项、无回归（2 个 warning 仍是既有前端测试写法提示）。
- **未启动、未重启 8848 服务**（用户自行重启）；未写入任何运行期词典/模板数据，
  `data/lexicon.json` 的用户运行期改动未纳入提交。

---

## §22.47 人工纠正清单、准确性判据与一键回滚（2026-11-11）

### 起因

用户连问两件事：①「既然纠正的不准危害很大，应该提供一个移除全部纠正的
入口吧」；②「我怎么知道纠正的准不准」。上一轮（§22.46）只解决了「看得见」，
当时明确记着「没有撤销已写入点位的接口」，本节把这条边界关掉。

**当前实况**（动手前读 `data/lexicon.json`）：22 词条 / 23 点位，全部
`tianditu` 自动源，`verified_points=0`，即**目前一条人工纠正都没有**，
用户此前并未成功提交过纠正。入口是给以后用的，本次没有清掉任何数据。

### 为什么「纠得多」不等于「越准」（代码依据）

| 现象 | 依据 |
|---|---|
| 一次错纠正会长期锁死该地名 | `geocode.py` `Geocoder.geocode()` 第 1 步 `lookup_by_source(name, city, ("user",))` 命中即 `return user`，在线源 / 本地兜底 / 冷启动全部跳过，天地图的门牌级正确坐标再也进不了候选池 |
| 写入几乎不设防 | `lexicon_guard.py:135-137` 对 `verified or source=="user"` 直接 `return WriteDecision(True, b, "user_verified", None)`，连自动写入要过的 200 m~10 km 异源冲突检查都不做；只保留名称合法性 / UI 文案 / 坐标范围（`:121-133`） |
| 点错不会消失 | `_commit()` 不同坐标走 `entries.append(...)` 新增一条同为 `weight=2.0` 的 user 点（`:564-579`）；两条 user 点 `source` 相同，`dedupe_cross_source_candidates()` 的 `len(sources)<2` 分支不剔除，两条一起进 RANSAC |
| 残片会扩散 | `name` 用 OCR 原文，`Lexicon.normalize()` 只去空格括号标点，`lookup()` 有包含匹配；短 key 会劫持别的查询（代码注释里「上海世」偏离 6.8 km 的实测事故） |
| 收益很快饱和 | `boost=2.0`、`quality=min(1, weight/5)`，同坐标第 3 次就到 1.0；唯一泛化收益是同城 ≥3/≥8 个点派生天地图搜索视野 |

### 怎么判断纠得准不准（写进页面说明）

最可靠的是**重新识别同一张图**：该地名出现在「内点」且内点数 / RMSE 没变差，
说明点被采纳且自洽；落入「离群」或整体指标变差就是点偏了。前端「人工纠正」
表格的「本次识别」列直接把这条结论标出来（用本次 `anchors_used` /
`anchors_rejected` 按归一化名称匹配）。辅助信号：

- `疑似 OCR 残片`：key ≤3 字或命中 UI 文案正则；
- `多个坐标（纠错过）`：同一 key 有 ≥2 条 user 点；
- `weight / count`：`2.0 / 1` 表示只纠过一次，多条同名说明纠错过；
- 手点坐标只有缩放级别精度（几十米），不要用去微调本来正确的自动点位。

### 实现

* `src/app/geocode.py`：`Lexicon.clear_user_points()`。只删
  `verified=True` 或 `source=="user"` 的点位，保留自动点位；空 key 一并删除；
  `RLock` 内改内存并 `save()`，无人工点时**不重写磁盘**（幂等）。返回
  `removed_points / removed_keys / remaining_points / upgraded_auto_points_lost`。
  ⚠️ 用户在**同一坐标**上纠正时 `_commit()` 会把原自动条目就地升级，
  无法还原；用 `last_confidence` 字段识别（用户提交不带 confidence），
  如实计入 `upgraded_auto_points_lost`。
* `src/app/server.py`：`POST /api/feedback/reset`，body 必须
  `{"confirm": true}`，否则 400；返回 `ok / 报告 / lexicon.stats()`。
  清空后**内存与磁盘同步，不需要重启服务**；轨迹模板库不受影响。
* `src/web/index.html`：左栏新增**独立卡片** `#correctionPanel`
  （`#correctionCount` / `#correctionRows` / `#clearCorrections`）。刻意不嵌进
  默认 `display:none` 的 `anchorCard`——否则页面刚打开、还没识别过时看不到
  回滚入口，无法清理历史错纠正。
  数据来自既有 `GET /api/lexicon`，在**页面初始化、每次 `render()`、
  纠正提交成功后**各刷新一次。清空按钮二次确认，成功后状态栏写明删除数、
  保留数与「无法还原」的损失；取消不发请求，失败有可见错误提示。
  新增 `.tag.warn/.tag.err` 与 `button.ghost.danger` 样式。
* 测试：`tests/test_feedback_reset.py`（纯函数 + TestClient，全部 tmp_path，
  覆盖只删 user、落盘、幂等、confirm 校验、接口联动）；
  `tests/test_frontend_feedback_reset.py`（Node 驱动真实前端函数，覆盖
  清单渲染/风险标记/二次确认/请求体/刷新/取消/失败）。
  `tests/test_frontend_correction_ui.py` 的 Node 沙箱补 `refreshCorrections`
  桩并新增「提交成功后刷新清单」断言（新调用曾让旧测试误入 catch）。

### 边界与未做

* 🟡 **只做「全部清空」，没有单条删除**。同名多点、只是某一处点错时仍要
  整体回滚；后续可加按 `name + 坐标` 的单条删除。
* 🟡 被就地升级的自动点位无法还原（见上），会一起删掉；清空调返回计数并
  在前端提示，不做「假装无损」的降级。
* 🟡 纠正仍不解锁轨迹模板：`position_ok` 只看街道名 / 手填路名，
  §22.45 的 `low_confidence` 模板写入死锁仍在（P2 后续项）。
* 🟡 纠正只影响**下一次**识别，本次轨迹不重算；清空同理。

### 验证

- 前端语法检查 `new Function(...)` 编译通过；`app.server` 导入正常。
- 新增后端测试 4 passed；新增前端测试 1 passed。
- 前端组 + 新增后端测试：**35 passed / 2 warnings**（2 个 warning 仍是
  既有 `test_frontend.py` 写法提示）。
- 全量 pytest（Python 3.11，最终版本）：**450 passed / 2 warnings（417.27 s）**，
  较 §22.46 的 445 净增 5 项（新增 4 个后端 + 1 个前端测试），无回归。
  期间曾有一轮全量在修复旧 Node 沙箱之前启动，报 `1 failed / 449 passed`，
  根因是新加的 `refreshCorrections()` 在旧测试的 `new Function` 沙箱里
  没有桩、同步抛 `ReferenceError` 让 `submitCorrection` 误入 catch；
  补桩后复跑全绿，该中间结果不作为交付依据。
- **未启动、未重启 8848 服务**（用户自行重启）；所有验证为进程内
  TestClient / 纯函数级 / Node 桩，运行期词典指向 `tmp_path`，
  `data/lexicon.json` 未改动、未纳入提交。

---

## §22.48 拼接画布保住起终点徽章（2026-11-11）

### 用户反馈与现象

用户 21:19 上传 1 张概览单图 + 7 张细节图，配对路径返回 `ok=false`，
warnings 首条是「未检测到起点徽章（绿色「起」）」，随后
「轨迹形状不合理：连通域宽 2883px ≈ 整幅图宽（3077px）」、
「未能提取到有效轨迹线」；`track_points=0`、
`paired_registration.reason=not_enough_points`。用户质疑
「明明有开始的截图，怎么没识别到」。

### 根因（逐像素复核）

* 概览单图 `1794403187_10f153a3.jpg` **确有**起 (293,1230) / 终 (968,1373)，
  但配对模式下它只用于注册模板、**不进入识别画布**（`server.py:531-600`
  的 `run_img` 来自 `details` 拼接）。识别跑的是 7 张细节图拼出的画布。
* 7 张细节图里只有第 7 张 `0789bbe3` 有「起」（绿色连通域 4270 px），
  第 1 张 `c98c910b` 有「终」，其余 5 张都没有徽章。
* 离线重跑 `stitch()` 与落盘画布逐像素一致（平均差 0.0）：
  `used=[0..6]` 全部保留，**没有被判重或判无重叠而排除**；画布
  3077×3818、17% 空洞、10 对配准不可信。画布上终点在 (2852,2189)，
  绿色最大连通域只剩 5 px。
* 复刻贴图 owner map：起点绿色块 warp 后落在画布 (693,1443)-(770,1532)，
  4270 个像素 **100% 由 index 2 的图提供**，而不是含徽章的 index 6。
  原因是 `multishot.py` 第 5 步按 `(depth, index)` 贴图、
  `sel = mask & ~filled` **先贴者胜**；index 2 与 index 6 同为 depth 2，
  index 2 先贴，把同区域的「无徽章版本」盖了上去。
* 对照组：18:45 那次成功的 5 图批次，画布 4177×3864 里
  起 (78,991) / 终 (4043,2903) 都在，所以能识别。**差别在覆盖顺序**，
  不是截图本身缺起点。

### 改动

* `src/app/multishot.py` 新增 `_badge_protection_items()`：用与
  `track.find_marker()` 相同的 HSV 区间 / 面积阈值找出各图的起/终连通域，
  每类只留面积最大的一枚（与画布检测语义一致，避免写回多枚矛盾的「起」）；
  闭运算填掉白色「起/终」字造成的孔洞、膨胀包住描边。
* 第 5 步贴图排序由 `(depth, index)` 改为 `(depth, 含徽章优先, index)`，
  并把每张图的变换存进 `transforms`。
* 新增第 6.5 步：画布裁剪后按各自变换把徽章掩膜 warp 回去，强制写入彩色
  像素（同一张图只 warp 一次），`notes` 记「已保护 N 枚起终点徽章」。
  同深度排序解决不了跨深度覆盖，强制写回才是兜底。
* 不改识别语义、不改接口、不动 `track.py`；`tests/test_stitch_badge_protection.py`
  为新增文件。

### 验证

* 新测试用自包含合成底图（随机纹理供 SIFT 配准），刻意构造「含徽章图
  depth 更大、必被先贴图覆盖」的最坏情形：修复前 `1 failed / 1 passed`
  （准确复现事故），修复后 `2 passed`。另有一条断言要求两枚徽章写回后
  横向间距仍是 460±6 px，防止裁剪偏移算错。
* 真实 7 图纯函数级复验：画布起 (732,1481)、终 (2852,2189)，
  `extract_track()` 从「0 点 / 失败」变成 **2431 点**并给出
  「轨迹连通域通过起终点拓扑校验」。
* 拼接 + 轨迹组 **35 passed（216.58 s）**；全量 pytest（Python 3.11）
  **452 passed / 2 warnings（263.89 s）**，较 §22.47 的 450 净增 2 项、
  无回归（2 个 warning 仍是既有前端测试写法提示）。
* **未启动、未重启 8848 服务**（用户自行重启）；未写运行期词典 / 模板数据，
  `data/lexicon.json` 未改动、未纳入提交。

### 边界与未做

* 🟡 徽章恢复只是**必要条件**，不是充分条件：这批画布仍有 17% 空洞、
  10 对配准不可信；完整流水线（地理配准 / 置信度 / 模板注册）未离线重跑，
  避免写运行期数据与消耗在线配额。用户重启后需自行复跑一次确认。
* 🟡 warnings 里「概览图未能注册（not_enough_points）：请确认细节图覆盖
  整条路线且识别自洽」归因仍不准（真实原因是本次识别失败 / 概览图轨迹
  点数不足），未纳入本轮。
* 🟡 徽章识别沿用 `find_marker()` 的 HSV + 面积判据（1500–20000 px），
  没有额外做圆形校验：底图里若恰有一块近圆形绿地 / 绿色 UI 落在该区间，
  仍可能被当成徽章写回。这与 `track.py` 现有判据同源，未扩大风险面，
  但值得后续一起加形状判据。
* 🟡 §22.45 的 `low_confidence` 模板写入死锁仍在；§22.47 的单条删除接口
  也仍待做。

## §22.49 失败归因：徽章守恒诊断与告警分层（2026-11-12）

### 起因

用户连问「这类错误之前是不是犯过，怎么优化」「细节图里面的起点，
程序没识别到吗」。核对本文档后确认：**同一失效模式已出现 4 次**——
§22.21.14（去重误判删掉含起点的图）、§22.21.15（倒序上传判重又删掉）、
§22.21.16（12 张整段排除前 5 张）、§22.48（重叠区贴图覆盖）。
四次都是「输入里有徽章、画布里没有」，四次都报同一句
「未检测到起点徽章」，四次都只修了**触发路径**。

逐张复核 21:19 / 21:22 两批上传（同一组 8 张图，后者是重传）：
细节图 `0789bbe3` / `6a66e9cc` 有绿「起」(163,1163) 4270 px、
`c98c910b` / `c2de14c5` 有红「终」(1035,1786) 4171 px，概览图另有一对；
两张画布都只剩「终」(2852,2189)。**程序不是「没认出」，而是从没在
原图上找过**：`find_marker()` 只出现在 `track.py::_detect_markers_ex()`，
输入永远是最终画布。21:22 重跑仍失败，是因为用户尚未重启服务，
`fd0292e` 未加载进进程。

### 为什么复核机制没抓住

`multishot._verify_canvas()` 为了避开「正常覆盖关系」，主动用
`own = mask & ~_covered_by_others(...)` **剔除重叠区**——而徽章只可能
丢在重叠区，所以复核分 1.0 也看不见语义信息丢失。§22.21.8 甚至写过
「标记可能被另一张图覆盖（正常的覆盖关系，不是 bug）」，当时没有区分
「普通像素」与「语义锚点」。

### 改动

* `src/app/multishot.py`
  * `StitchResult` 增 `badge_conservation`，`to_dict()` 一并输出。
  * `_badge_protection_items()` 额外返回**输入清单**（每类在哪些原图
    检出过），与保护动作共用一次检测，避免两处 HSV 阈值漂移。
    `scan_indices` 传入全部输入图：被判「无重叠」而整张排除的图上
    若有徽章，同样属于「输入有、画布没有」，不能悄悄略过；
    写回仍只对 `res.used`（它们才有变换矩阵）。
  * 新增 `_badge_conservation(canvas, inventory)`：只在「输入本来就有」
    的类别上要求画布也有，输出 `input / canvas / ok / missing`。
    中途截图没有徽章是正常现象，反向要求会制造误报。
  * `stitch()` 收尾写入诊断；不守恒时 `notes` 追加「拼接产物丢失了…；
    这是多图贴图阶段的信息丢失，不是截图里没有徽章」。
  * **不改 `stitch.ok`**：几何拼接确实是对的，只是语义锚丢了。
* `src/app/server.py`
  * `_explain_badge_loss(warnings, conservation)`：不守恒时把下游
    「未检测到起点/终点徽章」换成「拼接阶段丢失了…（输入第 N 张图有）」，
    并注明后续「轨迹形状不合理 / 未能提取到有效轨迹线」都是连锁结果；
    守恒或无诊断时**一字不改**（零回归）。`/api/recognize` 配对分支与
    `/api/recognize_multi` 两处 warnings 组装都接入。
  * `_paired_failure_note(reason)`：注册失败按 `paired_registration.reason`
    分流（not_enough_points / low_confidence / partial_route / deduped /
    shape_* / error:*），不再一律说「请确认细节图覆盖整条路线且识别自洽」。
  * 识别失败时不再宣称「本次结果为地理真值」，改为「已用 N 张细节图
    拼接，但本次识别未成功（原因见下方告警）」。
  * 配对分支补 `payload["stitch"]`，排障可直接读到守恒诊断。
* 未改 `track.py`：它只看自己拿到的那张图，措辞本身没有错，
  错在下游把「画布上没有」直接当成「截图里没有」。

### 验证

* 新增 `tests/test_badge_conservation.py`（9 项）：守恒正常返回、
  只报输入本来有的类别、被排除图上的徽章仍计入输入清单、
  `_explain_badge_loss` 替换/零回归、reason 分流，以及 TestClient
  端到端断言失败时不再出现「未检测到起点徽章（绿色「起」）」
  「本次结果为地理真值」。
* 相关组 **47 passed（9.42 s）**（新增 + 徽章保护 + 配对注册 +
  多图精修 + 拼接单测）。
* 真实 7 图纯函数级复验：`badge_conservation` =
  `input.start=[6] / input.end=[0] / canvas 全 True / ok=True`，
  `extract_track()` 2431 点、起 (732,1481)、终 (2852,2189)。
* 全量 pytest（Python 3.11）：**461 passed / 2 warnings（607.82 s）**，
  较 §22.48 的 452 净增 9 项（即本轮新增测试），无回归；
  2 个 warning 仍是既有 `test_frontend.py` 写法提示。
  中途有一轮全量在「守恒清单扩到全部输入图」之前启动，已停掉并以
  最终版本重跑，避免结果对不上交付代码。
* **未启动、未重启 8848 服务**（用户自行重启）；未写运行期词典 / 模板数据，
  `data/lexicon.json` 未改动、未纳入提交。

### 边界与未做

* 🟡 这是**最小版**守恒诊断：只守起/终徽章，没有做完整的
  `failure_stage`（input / stitch / track / register / geocode / template）
  分层字段，也没有把「输入有徽章 ⇒ 画布有徽章」做成跨任意排列 / 深度 /
  重叠比例的参数化性质测试。但下一次不管因什么原因丢徽章，
  至少会被同一条断言抓住，而不是等下游甩锅给截图。
* 🟡 §22.45 的 `low_confidence` 模板写入死锁、§22.47 的单条删除接口、
  「离群行的『文字』标签 + 纠正入口」三项仍待做。

---

## §22.50 位姿树按配准置信度选边（2026-11-12）

### 起因

用户重启服务后复跑 09:04 那批 8 张图，反馈「拼接的图片不对，
7 张图片没有全部拼接，识别到的路线也不对」。服务确已重启
（`run.py --host 127.0.0.1 --port 8848`，09:03 启动），所以
`fd0292e` / `4c4758f` 都已加载，不是旧代码残留。

### 现象与根因

**不是重启引入的回归**。11-11 21:19 与 11-12 09:04 落盘的两张
`_stitched.png` 都是 3077×3818、终点同为 (2852,2189)，逐像素
平均差仅 1.46，明显差异 5.07% 且集中在 (569,318)-(1475,3038)
（徽章写回 + 同深度贴图顺序调整）。说明位姿树折叠**一直存在**，
只是 11-11 那次先因「起」徽章被覆盖直接 `ok=false`，没走到
「看画布」这一步；徽章修好后，第二层问题才第一次暴露出来。

`data/last_result.json`（09:06:37）：`ok=true`、`n_shots=7`、
`stitch.used=[0..6]`、`badge_conservation.ok=true`、
`verify_score=1.0`，看起来全绿；但

* `stitch.canvas_size=3077×3818`，而 7 张 1260×2720 的图按 links
  里 ~900 px/张的位移应展开成约 6400 px 宽；
* `shots[].box` 全部挤在 x=0~1817、y=0~1089，图 4 被甩到 y=+686，
  画布上能看到 3 个重复状态栏与底部一条明显错位缝；
* `verify[5].similarity=0.3149`（其余 1.0）；
* 轨迹 `bbox_px=[709,1242,2174,1026]`、`distance_m=2834.7`，
  而概览图对应的完整路线约 9 km；`paired_registration.reason=
  partial_route`、`coverage=0.625`。

🔴 **根因在 `multishot.stitch()` 第 3 步的位姿树**：旧版 BFS 按
`link_map.items()` 的插入顺序「先到先得」，只保证跳数最少，
**完全不看配准质量**。本批 10 条可用边里，强链是
0↔1(314) → 1↔2(549) → 2↔3(721) → 3↔4(965) → 4↔5(725)
→ 5↔6(490)；但 BFS 从 root=0 先吃下 (0,1)、(0,5, 内点 25)，
cur=1 又吃下 (1,2)、(1,4, 内点 21)，cur=5 吃下 (3,5, 内点 46)、
(5,6)。于是图 3 定位到 x=-262（应为 -2720）、图 4 定位到
(-470, +686)（应为 -3363, -491），**6 条强边里 3 条被丢弃**，
7 张图折叠堆叠。`_verify_canvas()` 主动剔除重叠区，对这种
「整体落位错误」依旧给出 1.0（只有图 5 的独占区露馅）。
`POSE_GRAPH_ENABLED` 默认关闭，全局平差也没机会补救。

### 改动

* `src/app/multishot.py`
  * 新增 `_pose_tree_weight(link)`：权重口径与
    `posegraph.PoseEdge` 一致（`max(1, inliers) × max(0.05, ncc)`），
    避免选树与平差各算各的。
  * 新增 `_build_pose_tree(link_map, root, members)`：**Prim 式
    最大生成树**。每轮只在「已定位集合 ↔ 未定位集合」的跨接边里选
    权重最高的一条，等价于让每个节点的位姿来自最强瓶颈路径；
    权重相同按 `(-i, -j)` 排序保证可复现；只纳入 `members` 所在
    连通块；没有强边时弱边仍能把图连起来，不会罢工。
    返回 `(T, depth, edges)`，`edges=(parent, child, inliers, weight)`。
  * `stitch()` 第 3 步用该函数替换旧 BFS；`depth` 语义与贴图
    优先级不变，`notes` 增「位姿树按配准置信度选边：N 条，
    最弱边内点 X」；位姿树覆盖不全时安全返回 `ok=false`，
    不会在后续贴图阶段 KeyError。
  * **未改**像素贴图、徽章保护、守恒诊断、`track.py`、
    任何接口字段与识别语义。

### 验证

* 新增 `tests/test_pose_tree_selection.py`（7 项）：4 图强弱边竞争、
  内联**旧算法**复现事故成因、09:04 真实批次 10 条 link 参数重放、
  确定性 / 连通块边界 / 只有弱边 / 权重口径。修复前该模块因
  `_build_pose_tree` 不存在而 collection error，修复后 7 passed。
* 真实 7 图纯函数级复验（09:04 批次，按上传时间还原输入顺序）：
  画布 **3077×3818 → 6423×3373**；`shots[].box` 沿 x 从 5163
  依次递减到 0、y 漂移仅 0~653；`verify` **全 1.0**
  （原图 5 只有 0.3149）；`badge_conservation.ok=true`；
  `extract_track()` **2431 → 6663 点**，起 (163,1163)、
  终 (6198,2439)，轨迹连续横跨整张画布并通过起终点拓扑校验。
  缩略图见 `output/stitched_tree_fixed_thumb.png`（未提交）。
* 相关组 **77 passed（384.95 s）**：选树 + 拼接 + 多图 e2e +
  stitchcheck + 真实多图 + 徽章保护 + 守恒诊断 + 轨迹形状 +
  配对注册 + 多图精修。
* 全量 pytest（Python 3.11）：**468 passed / 2 warnings（450.72 s）**，
  较 §22.49 的 461 净增 7 项（即本轮新增测试），无回归；
  2 个 warning 仍是既有 `test_frontend.py` 写法提示。
* **未启动、未重启 8848 服务**（用户自行重启）；未写运行期词典 /
  模板数据；验证只用 pytest 与纯函数级调用，未调在线地理编码。

### 边界与未做

* 🟡 修的是**位姿树选边**，不是地理配准。09:04 那次结果本身只有
  3 个内点、7/20 个 POI 命中、置信度 0.35，还伴随
  「冷启动未找到可用坐标」；拼接展开后只是让轨迹形状与覆盖范围
  变对，**地理绝对位置是否更准要用户重启后实测**。完整流水线
  （地理配准 / 置信度 / 模板注册）未离线重跑，避免写运行期数据
  与消耗在线配额。
* 🟡 仍有约 18% 空洞与链式纵向漂移（y 累积 653 px）。
  `POSE_GRAPH_ENABLED=1` 可用全部重叠边做平移平差，默认仍关闭，
  本次未改配置（遵守「一个功能只用一个环境变量」）。
* 🟡 `3↔6` 那条 `rotation_deg=-25.9127` 的错配仍会通过可信度
  过滤进入 `res.links`（只是不再被选树采用）；角度异常剔除留待后续。
* 🟡 §22.45 的 `low_confidence` 模板写入死锁、§22.47 的单条删除
  接口、离群行标签与纠正入口、徽章圆度判据仍待做。

## §22.51 多图拼接后端可插拔：SIFT / 大模型 A/B 对比（2026-11-13）

### 起因

11-12 晚用户问「把多图拼接交给大模型做，效果会更好吗」。当时的回答是
**不会更好**：拼接要的是 3×3 变换矩阵，VLM 只能给离散语义判断，给不出
稳定的像素级坐标（§21.5、§22.17.2 已有实测；公开基准 CrossPoint-Bench
里 Gemini-2.5-Pro 从离散「对应判断」60.26% 掉到连续「坐标指认」16.41%，
人类 93.63% 而通用 VLM 只有 12.78%）。而且近期 4 次拼接事故全是
确定性算法缺陷，不是语义理解不足。

用户随后要求把这个结论做成**可实测的开关**：把拼接能力模块化，让用户
自己选「项目现用方式」或「大模型方式」，跑同一批截图对比效果。

### 设计：唯一变量是「边从哪来」

`multishot.stitch()` 拿到 `link_map` 之后的连通块、位姿树、贴图、
逐像素复核、徽章守恒与后端完全无关。所以只把第一步抽成可插拔后端，
三种方式的**下游计算逐字相同**，`verify` / `verify_score` /
`canvas_size` / `badge_conservation` 才具备可比性。

| 后端 | 边的来源 | 跳过 SIFT |
|---|---|---|
| `sift`（默认） | 21 对全组合 SIFT + RANSAC + 候选峰 NCC | 否 |
| `llm` | 多模态模型判断相邻关系与相对位移比例，直接注入 `PairLink` | 是 |
| `llm_guided` | 模型只圈定候选图对，变换仍由 SIFT 精配准 | 否 |

新增 `src/app/stitch_backends/`：`base.py`（`StitchPlan` / `BackendSpec` /
异常）、`registry.py`（注册 + `run_stitch()` + `list_backends()`）、
`sift.py`、`llm.py`、`__init__.py`。服务端统一由 `_stitch_with_backend()`
分派：默认 `sift` 保留 `server.stitch_images` 的调用形态（既有回归测试靠
monkeypatch 这个符号注入假拼接结果），其余后端走 `run_stitch()`；两条路
最终都调同一个 `multishot.stitch()`。新增后端只要实现 `plan()` 并
`register()`，前端由能力清单自动列出。

#### 纯大模型后端怎么给变换

不要求 VLM 输出像素坐标，只让它输出**离散方位 + 相对比例**：

```json
{"pairs": [{"i": 1, "j": 2, "dx_ratio": 0.75,
            "dy_ratio": 0.05, "confidence": 0.8}]}
```

缩略图统一缩到最长边 768px 并打 `#1..#N` 角标，降低 token 成本、
让模型能对应编号。代码校验编号范围、排除 0 位移假解与超范围值后，
转成 `matrix.tx = -dx_ratio × 图宽`。

⚠️ **方向用真实落盘结果校准过**：11-12 的结果里 `link(0→1).tx=+866`，
而画布上 `shots[1].x0` 比 `shots[0].x0` 小 867px——「j 在 i 右侧」
对应的矩阵平移是**负**号。写反了会得到镜像画布且 `verify` 全低。

#### 数值诚实性

语义边没有做过几何配准，所以 `inliers=0 / ncc=0`；这不是「匹配失败」，
而是「没有几何证据」。`PairLink` 新增 `weight`（模型置信度）与
`source`（`sift` / `llm`），`_pose_tree_weight()` 在 `weight` 非 None 时
改用它选树，位姿图平差权重同步改用该函数。前端对 `source=llm` 的边
显示「模型置信度」而不是「0 内点」，也不套 SIFT 的「重叠偏弱」判据。

### 🔴 一条硬规矩：绝不静默回退

用户选了大模型后端却因为「没配 key / 调用失败 / 返回空」偷偷用 SIFT
出一张正常画布，A/B 表里「大模型」这一栏就变成 SIFT 的成绩，而且看不出来
——比直接报错更糟。因此：

* 配置检查、模型失败、空输出一律走 `StitchPlan.error` →
  `StitchBackendError` → 服务端 `stitch.ok=false` + `backend_error=true`；
* 失败文案明写「未回退到其它拼接后端」，识别改用第 1 张原图继续；
* 前端从 `/api/stitch-backends` 拉清单，不可用项直接 `disabled`；
* 测试专门锁住这条（`test_stitch_backends.py` / `test_server_stitch_backend.py`）。

### 顺带修掉的误导

`recognize()` 的配对模式原来无论拼接是否成功都会写
「已用 N 张细节图拼接配准，本次结果为地理真值」。拼接失败时底下其实只是
第 1 张细节图的单图识别结果。现在 `stitch_failed=True` 时改写成
「拼接未成功，结果基于第 1 张图识别，不能作为地理真值」。

更严重的是同一分支还会调 `_capture_paired_template()`，把这份**单图结果**
当「多图地理真值」注册成模板——模板一旦写错会带偏以后所有形状接近的
概览图，属于错误自我强化。现在拼接失败直接返回
`{"written": False, "reason": "stitch_failed"}` 并跳过注册，
`_PAIRED_FAILURE_REASONS` 增加对应文案。这条与本次新增的「后端失败」
路径直接相关：以前拼接失败较少见，加了可切换后端后会更常出现。

### 接入点

* `multishot.py`：`stitch()` 新增 `links_override` / `pair_candidates` /
  `backend` 三个**仅关键字**参数；三者取默认值时原生 SIFT 路径一字不变。
  `StitchResult` 增加 `backend` / `backend_diagnostics` 并进 `to_dict()`。
* `server.py`：两个识别入口各加 `stitch_backend: Form(DEFAULT_BACKEND)`；
  新增 `GET /api/stitch-backends` 与共用的 `_stitch_with_backend()`。
* `web/index.html`：新增「多图拼接方式」下拉（仅多图生效）、结果卡片展示
  本次后端与模型诊断（模型名 / 耗时 / 采纳与剔除的关系数）。
* 配置**复用**已有的 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`，
  没有新增任何环境变量（遵守「一个功能只用一个环境变量」）。

### 一次真实回归：模块级符号也是接口

第一次全量 pytest 出现 **1 failed + 14 errors**，全是同一条：

```
AttributeError: module 'app.server' has no attribute 'stitch_images'
```

根因：接入后端注册表时把 `from .multishot import stitch as stitch_images`
删掉换成了 `run_stitch`，而 `test_multi_refine.py` /
`test_paired_registration.py` / `test_route_template.py` /
`test_badge_conservation.py` 共 15 处 fixture 靠 monkeypatch
`server.stitch_images` 注入假拼接结果来控制服务端行为。

修复：保留别名，默认 `sift` 路径直接调 `stitch_images(imgs)`，并在结果里
补上 `backend` / `backend_diagnostics`（计算与 `run_stitch(backend="sift")`
等价）；新增 `test_default_sift_path_keeps_legacy_patch_point` 锁住契约。
另外把 `test_paired_registration.py` 中「拼接返回 None 却期望模板注册
成功」的不真实桩改成成功画布，以配合新的拼接失败护栏。

**教训**：换调用入口之前先搜测试里的 monkeypatch 目标；模块级符号同样是
对外契约，不能当内部实现随手删。

### 验证

* 合成底图（图 2 是图 1 右移 200px 的裁剪）：注入语义边后画布 600×240、
  `shots.x0=[0,200]`、`verify` 全 1.0，方向与真实几何一致；
  monkeypatch `_match_pair` 抛异常仍成功 → 证明确实跳过 SIFT。
* 新增 23 项测试（后端 13 / 服务端 8 / 前端 2），覆盖注册表、输出解析、
  变换方向、失败不回退、接口 Form 参数、OpenAPI、前端禁用项与渲染文案、
  「拼接失败不得注册配对模板」以及「默认路径仍可被 monkeypatch」的契约。
* 全量 pytest（Python 3.11）：**491 passed / 5 warnings（279.27 s）**，
  较 §22.50 的 468 净增 23 项（即本轮新增测试），无回归。5 个 warning
  为既有 2 个 `test_frontend.py` 写法提示 + 3 个 tmap_proxy 的 OpenAPI
  重复 operation id（由新增的 OpenAPI 契约测试触发，非本次引入的问题）。
* **未启动、未重启 8848 服务**（用户自行重启）；未调真实大模型接口
  （当前 `.env` 只有 `AMAP_KEY` / `TIANDITU_API_KEY` / `MAP_GEOCODING_ORDER`，
  没有 LLM_*），所有 LLM 调用均用 StubClient。

### 边界与未做

* 🔴 **真实大模型效果尚未实测**：需要用户自行在 `.env` 配好
  `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 并重启服务，再用同一批
  截图分别跑 `sift` / `llm` / `llm_guided`。按设计 `llm` 的
  `verify_score` 大概率明显低于 SIFT，这个后端是量化差距用的，
  不是承诺更高准确率。
* 🟡 `llm` 一次最多 16 张图（VLM 上下文与 token 成本）；缩略图最长边
  固定 768px。超时与流式参数已在 §22.52 修正为 `LLM_TIMEOUT=600` +
  `stream=True` + `max_tokens=16384`，本节原先「沿用 25s」的说法作废。
* 🟡 语义边只建模**平移**，不处理缩放、旋转与透视；App 截图缩放级别
  不一致时误差会直接进入画布。
* 🟡 模型原始响应截断到 2000 字符放进 `backend_diagnostics.raw_response`，
  便于复盘但不落盘；如需长期留档要另做归档。
* 🟡 §22.45 的 `low_confidence` 模板写入死锁、§22.47 单条删除接口、
  离群行标签与纠正入口、徽章圆度判据仍待做。

---

### 22.52 大模型拼接超时事故：不是超时不够，是请求形式不对

#### 现象

用户配好 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 后连跑两次 7 图
`llm` 后端，结果一字不差：`ReadTimeout ... read timeout=25.0`、
`stitch.ok=false`、`backend_error=true`，接口 `ok=true` 只是第 1 张原图
的单图回退。用户用 Postman 打同一个端点却是 `200 OK / 1.22s`，
因此怀疑模型没生效。

#### 犯的错

1. **只凭一条 `read timeout=25.0` 就建议加大 `LLM_TIMEOUT`**。后来把
   超时放宽到 90s 实测，7 图请求在 30.88s 拿到 `HTTP 102` 空响应——
   加大本地超时根本救不了。分级探测：纯文本 1.82s/200、单图 2.88s/200、
   2 图 30.22s/`HTTP 102`、4 图 17.19s/200、7 图 45s 超时、
   7 图缩到 384px 仍 `HTTP 102`。**根因是中转网关对「非流式长请求」
   约 30 秒的截断**，不是 payload 大小，也不是本地超时。
2. **`max_tokens=1200` 对推理型模型远远不够**。流式实测 7 图任务
   `reasoning_tokens=2668 / completion_tokens=2794`，思考阶段就把预算
   耗光，`content` 为空，即使等到响应也解析不出 JSON。
3. **失败路径把诊断丢了**：`server._stitch_with_backend()` 的
   `StitchBackendError` 分支把 `backend_diagnostics` 硬设 `{}`，
   模型名 / 耗时 / HTTP 状态全看不到，只能靠翻 `last_result.json` 猜。

#### 实测对照（2026-11-13，同一批 7 张细节图，真实请求）

| 请求形式 | 结果 |
|---|---|
| 项目原参数：非流式 / `max_tokens=1200` / 25s | 25s `ReadTimeout`；90s 时 30.88s 返回 `HTTP 102` 空体 |
| `stream=True` + `max_tokens=16384` + `gpt-6.1-sol` / 600s | **58.97s 成功**，首 chunk 7.36s，132 分片，合法 JSON 3 对相邻关系 |
| 完全照搬 Postman：`gpt-6-sol(max)` + `thinking:true` + `temperature=1` | **600s 跑满仍 `content_len=0`**，118 分片全是 reasoning |

🔴 结论：**用户 `.env` 里的 `gpt-6.1-sol` 是对的**，Postman 那个
`gpt-6-sol(max)` 只是在纯文本短请求上验证过，配 `thinking:true` 跑多图
会一直思考不输出答案。只借它的 `stream:true` 与 `max_tokens`，不要抄
模型名和 `chat_template_kwargs`。

#### 改动

* `src/app/llm.py`：`chat_json()` 固定 `stream=True`、`max_tokens` 提到
  `DEFAULT_MAX_TOKENS=16384`、默认 `DEFAULT_TIMEOUT=600.0`；新增
  `_read_chat_content()` 统一解析 SSE 与普通 JSON，只收 `content` 分片，
  逐行按 UTF-8 解码（避免 `decode_unicode=True` 把中文解成乱码），
  忽略 `reasoning_content`；流式总时长同样受 `LLM_TIMEOUT` 约束。
  **`chat_json()` 签名不变**，`llm_geo.py` 与既有测试桩不受影响。
* `.env` 增加 `LLM_TIMEOUT=600`；`config.py` 的 `llm_timeout` 默认
  同步改 600.0（原先 25.0 会误导）。
* 新增 `tests/test_llm_client_stream.py`：SSE 分片拼接、中文编码、
  reasoning 不污染答案、`stream/max_tokens/timeout` 参数契约、
  非 200 报错、普通 JSON 端点兼容、流式响应贯通 `_semantic_pairs`。

#### 教训

* **超时错误先分层再调参**：本地 read timeout、上游网关截断、模型输出
  被 max_tokens 截断是三种不同故障。一次「加大超时」的建议如果没有
  分级实测支撑，会让用户按错误方向反复重试。
* **拿 Postman 验证端点时，要验证同一负载**：短文本 1.22s 成功不能证明
  7 图多模态请求能跑完；反过来，能出结果的模型名也可能不是 Postman
  里那个。
* **推理型模型的 token 预算要留给 reasoning**：只看答案长度设
  `max_tokens` 会静默拿到空 `content`。
* 本轮仍**未重启 8848 服务**，验证只用离线脚本 + pytest；重启交给用户。

---

### 22.53 `max_tokens` 超端点上限：一次「参数越大越好」的手调事故

#### 现象

09:27:38 用户重启服务后跑 7 图 `llm` 后端，09:28:24 落盘结果：

```
stitch.backend        = "llm"
stitch.ok             = false
stitch.backend_error  = true
notes[0] = "拼接后端失败：[大模型语义配准（纯 VLM）] 大模型调用失败：
            HTTP 400: {"error":{"code":"invalid_parameter",
            "message":"Field 'max_tokens' must be at most 65536", ...
```

和前两次不同：这次**不是超时**，`elapsed_ms` 只有 24569（整个识别），
模型请求是 0.44 秒就被拒了。接口 `ok=true` 仍是第 1 张原图的单图识别，
`paired_registration = {"written": false, "reason": "stitch_failed"}`
（§22.51 的保护继续生效）；`shot_url` 又指向像素与原图完全一致的
`_stitched.png`。

#### 根因

`src/app/llm.py` 存在一处**未提交的本地修改**，把上一节刚定下的
`DEFAULT_MAX_TOKENS = 16384` 改成了 `512000`（文件 mtime 09:27，与服务
启动时间吻合）。对端点做最小请求分级实测：

| max_tokens | 结果 |
|---|---|
| 512000 | **HTTP 400**，0.44s，`Field 'max_tokens' must be at most 65536` |
| 65536 | 200 SSE，1.85s |
| 16384 | 200 SSE，1.59s |

所以这是「想给模型更足预算」反而越过端点硬上限，与模型名、超时、
图片数量、流式与否都无关。用户随后在 09:33:49 自己把值改成了 65536
（合法上限）——排查期间出现了并发编辑，把本轮刚写入的钳制逻辑冲掉过
一次，已重新应用并保留用户选的数值。

#### 修复

* `app/llm.py`：保留 `DEFAULT_MAX_TOKENS = 65536`，新增
  `MAX_TOKENS_CEILING = 65536`，请求体改为
  `"max_tokens": min(DEFAULT_MAX_TOKENS, MAX_TOKENS_CEILING)`。即使以后
  再有人把常量手调成 512000，也只会发 65536，不会让整个后端必然失败。
* `app/llm.py`：非 200 的错误文本从 `r.text[:200]` 放宽到 `[:500]`。
  参数类错误的 `details` 常在 JSON 后半段，截到 200 字只剩半句话。
* `stitch_backends/base.py`：`StitchBackendError` 增加 `diagnostics` 属性；
  `registry.run_stitch()` 抛错时带上 `plan.diagnostics`；
  `server._stitch_with_backend()` 失败分支不再硬设 `{}`，并补 `error` 原文。
  前端 `stitchBlock()` 失败时显示「错误 …」。这样 §22.52 里「只能翻
  `last_result.json` 反推」的缺口关闭。
* `server.py`：新增 `_canvas_preview_id(shot_id, really_stitched)`，两个
  识别入口统一调用。真正拼出画布才叫 `_stitched.png`，回退到第 1 张
  原图时叫 `_fallback.png`。`stitched_url` 字段名不变，前端无需改。
* 新增测试：`test_llm_client_stream.py` +3（上限契约、常量被调大仍钳制、
  400 错误文本完整）、`test_server_stitch_backend.py` +3（registry 诊断透传、
  结果含诊断、预览命名）、`test_frontend_stitch_backend.py` 渲染失败原因。

#### 验证

* 受影响组 `60 passed / 3 warnings`；**全量 pytest `502 passed /
  5 warnings`（333.29s）**，基线 496 + 新增 6，无回归。5 个 warning
  全部是既有 `tmap_proxy` OpenAPI 重复 operation id，与本次无关。
* 改后生产代码端到端真实模型调用：`plan_llm(7 张真实细节图)` →
  **94.54s 成功**、`error=""`、`pairs_reported/accepted=3/3`、
  3 条 `source=llm` 边（权重 0.78/0.86/0.84）。**请求链路确认恢复**。
* 但 3 条边仍只形成 2~3 个连通块，7 张图无法全定位，`stitch()` 依旧
  「定位不全即放弃」。**`llm` 模式仍会 `ok=false`，原因不是故障，而是
  纯 VLM 给不全相邻关系**——这正是 A/B 要量化的差距。

#### 教训

* **「给足预算」不等于「把上限调到最大」**：外部端点的参数约束要先探测，
  再决定取值；超限时报错是即时 400，不是慢，很容易被误读成「模型又挂了」。
* **关键常量要运行时再校验一次**：只靠「提交里写对了」挡不住本地手调，
  尤其是未提交的工作区改动会随服务重启直接生效。
* **多人/多工具同时改同一文件时，先 `git diff` 再写**：本轮有一次写入被
  并发保存覆盖，靠测试收集报错才发现；改完要立刻复核 `git diff --stat`。
* **失败也要把诊断带出去**：`backend_diagnostics={}` 让每次事故都只能靠翻
  落盘 JSON 复盘，成本远高于在异常类上带一个 `dict`。

### 22.54 提示词 v2：让模型「按轨迹线识别一条链」，并量化识别结果（2026-11-13）

#### 触发

用户看到 09:47 纯 VLM 结果（7 张图折叠进约 2 张图宽的画布），质疑
「是不是没有告诉大模型按把各个图的轨迹线按实际情况连在一起来拼接啊」。
上一节收尾时只做了只读复盘、没改代码。用户随后明确要求改提示词，并说
要测「**识别这一步**交给大模型能做到什么程度」。

#### 根因（复盘结论落地）

v1 `_SYSTEM` 把「轨迹线」与「道路走向、地名、POI」并列成依据之一，
`_user_prompt` 只要求「输出确信有重叠的图对」，没有要求：
1. 相邻两图的轨迹线在画面边缘首尾接续（出边↔入边、方位互对）；
2. 输出一条覆盖全部 n 张图的链，而不是若干零散图对；
3. 沿链位移的主导轴符号一致（截图是单向滑动截的）；
4. 不能仅凭同名道路 / POI 判相邻；
5. 每条边说明依据，便于核错。

模型实际按「吴淞江、北翟路、天山西路、小区标签的连续性」凑边，
`dx_ratio` 序列 `-0.62/+0.78/-0.55/+0.55/-0.5/-0.52` 符号翻转，
落位后 7 张图挤进 2546×3155（SIFT 基线 6423×3373）。

#### 修复（提示词 / 解析 / 前端三层）

* `app/stitch_backends/llm.py` 新增 `PROMPT_VERSION = "trajectory-chain-v2"`：
  - `_SYSTEM` 明确「编号只是上传顺序」「轨迹线是**唯一主证据**」，给出
    出边↔入边方位互对、走向 / 所沿道路 / 路口形状连续、起终徽章各出现
    一次的判据；明令禁止只凭同名地名 / POI 凑边、禁止按编号硬连。
  - `_user_prompt(n)` 要求先输出 `order`（1..n 的排列），再**只**输出
    order 相邻两图的边（理想 n-1 条），`i` 在前 `j` 在后；写死「主导轴
    必须同号，若需 +0.78 / -0.55 交替说明 order 排错」；每条边必填
    `evidence`；附 4 条输出前自检。`_SCHEMA_HINT` 同步改为
    `order / pairs / notes`。
* 新增 `_parse_order()`：必须是 1..n 的排列，容错字符串数字；不合规
  返回空 + 原因，**不阻断**调用（格式不合规本身就是「识别能力」的结论）。
* 新增 `_chain_diagnostics()`：只读体检，不改边、不降权（保持 A/B 纯净）：
  - `order` / `order_valid` / `order_error`；
  - `chain_complete` / `connected_components` / `components` /
    `missing_images` / `pairs_expected` / `off_chain_pairs`；
  - 把边按 order 定向后比较主导轴（x/y 取 |dx| / |dy| 总和较大者）符号，
    小于 `MIN_CONFLICT_RATIO=0.12` 视为微调、多数票定参考方向，输出
    `direction_conflicts` / `direction_conflict_count`；
  - 收集 `evidence`（最多 8 条）供结果卡片展示。
  09:47 事故序列被复现为「x 轴、2 条方向冲突（+0.78、+0.55）」。
* `_chain_warnings()` 把体检结论翻成 ⚠ 告警追加进 `StitchPlan.notes`；
  `diag` 新增 `prompt_version`，失败路径也带上。
* `plan_llm_guided()` 候选不再只取模型给的 pairs：`order` 有效时把相邻
  图对一并交给 SIFT 验证（上次只给 3 条边，7 张图永远连不全），
  `diag["candidates_from_order"]` 记录补齐数量。
* `src/web/index.html` `stitchBlock()` 模型诊断行新增「识别顺序
  #a→#b…」「链条完整 / 不完整（N 块）」「方向冲突 N 条」「顺序无效」，
  并列出前 2 条「轨迹接续」依据；两个 `BackendSpec.description` 同步
  改成「按轨迹线识别」，下拉可见。

#### 刻意没做

* **校验只诊断、不拦截**：不因 order 非法 / 方向冲突就丢边、降权或判失败。
  用户这轮要测「识别能做到什么程度」，把模型原始判断如实出图 + 明确告警
  比提前掐掉更有信息量；降权还会污染 A/B 的唯一变量。
* **`stitch.ok` 产物级断言仍欠账**（上一节复盘建议 3）：`ok` 仍只看能否
  贴出画布，`verify_score` 用中位数会掩盖单点 0.03 的灾难。列为 P2，
  等 A/B 数据出来再决定阈值。
* 未重启服务、未起临时端口（约定：服务器由用户自己重启）。**代码需
  重启 8848 才生效**；`index.html` 每次请求读盘，重启后强刷页面即可。

#### 验证

* 新增 `tests/test_stitch_llm_prompt.py`（16 例）：提示词契约、请求确实
  带上新 system/user、order 解析、evidence 保留与方向归一化、事故序列
  符号冲突复现、覆盖不全 / off-chain / 无效 order、guided 补候选、
  老格式输出向后兼容、失败诊断含版本号。
* 更新 `tests/test_frontend_stitch_backend.py`：静态契约 + Node 渲染验证
  顺序 / 链条 / 方向冲突 / 轨迹接续依据出现在结果卡片。
* 受影响组 `50 passed / 3 warnings`；全量 `pytest -q` **518 passed /
  5 warnings（494.33s）**，基线 502 + 新增 16，无回归。5 个 warning
  全是既有 `tmap_proxy` OpenAPI 重复 operation id，与本次无关。
* 全量验证的两个环境坑（与代码无关）：默认 tmp 收尾清理旧
  `garbage-*` 会触发工具侧批量删除拦截、打不出 summary；改
  `--basetemp` 时必须用 **D 盘** `D:/Users/hxiong/AppData/Local/Temp/...`，
  写成 `C:/Users/...` 会让 119 个 `tmp_path` fixture 全部
  `FileNotFoundError`（本机用户目录在 D 盘）。
* 建议按 `llm_guided` → `llm` 顺序测：前者只把「顺序识别」交给模型，
  后者再叠加模型给的位移，两次相减可分离「顺序误差」与「坐标误差」。

#### 教训

* **「列了一个依据」≠「把它当约束」**：轨迹线必须写成唯一主证据 +
  明确禁止项 + 反例，否则模型会挑最省事的地名共现来凑答案。
* **让输出结构天然满足约束**：要求 `order`（排序）而不是 n-1 个独立
  `dx_ratio`，符号一致性才有可校验的锚点；VLM 擅长排序、不擅长坐标。
* **诊断优先于拦截**：A/B 阶段拆掉模型输出会让「模型差在哪」不可见；
  先把识别结果量化出来，再决定要不要加硬断言。

### 22.55 「拼开了却没轨迹」：纯 VLM 把斜向滑动当成水平平移（2026-11-13 14:05）

#### 现象

用户贴 14:05 结果截图：7 个黄色虚线框从左到右 `#7→#6→…→#1` 横向
铺开、起终徽章分列两端，比 09:47 的折叠团明显改进；但画面里看不到
绿色轨迹线，整体识别仍 `ok=false`。两次落盘同批截图可直接对比：

| | 11:05 纯 VLM（v2 提示词） | 14:05 SIFT 基线 |
|---|---|---|
| 画布 | **6854×2910** | **6423×3373** |
| 逐图 y0 | 框齐平（≈0 漂移） | 0 / 70 / 161 / 246 / 249 / 396 / **653** |
| 起/终徽章 | (162,1217) / (6629,1786) | (163,1163) / (6198,2439) |
| 轨迹 | `track_points=0`，「未找到轨迹连通域」「掩膜为空」 | `track_points=122`，`distance_m=3495.7`，`verify_score=1.0` |
| 整体 | `ok=false`，模板 `not_enough_points` | `ok=true`，但 `input_quality=reject`、模板 `degenerate_shape`（扁长直线、无街道名锚点，只能定到街区级） |

#### 根因

SIFT 画布高 3373 = 单图高 2720 + **653px 累积垂直漂移**（截图是斜向
滑动）；纯 VLM 画布高 2910 ≈ 单图高 2720 —— 模型给的所有 `dy_ratio`
都是 0，把斜向滑动当成纯水平平移。轨迹线在每张图的接缝处被垂直错开，
`track._select_component()` 要求一个连通域同时连起起终点，放宽掩膜后
仍找不到 → 「掩膜为空」。

**这不是顺序识别问题**（模型给的 `#7→…→#1` 与 SIFT 落位一致），
**也不是轨迹颜色 / 截图质量问题**，而是位移精度问题。

#### 修复（提示词 v3 + 失败归因，不改 A/B 语义）

* `app/stitch_backends/llm.py`：`PROMPT_VERSION=trajectory-chain-v3`；
  `_SYSTEM` 新增「垂直位移通常不是 0」硬约束——用重叠区同一地物
  （路口 / 河道拐弯 / 桥 / 小区边界 / 同一段高架）的上下错位估
  `dy_ratio`，典型 0.02~0.15、符号沿链一致；写明反例「全填 0 →
  轨迹线在接缝断开 → 整次识别失败」。`_user_prompt` 同步，自检增至 5 条。
* `_chain_diagnostics` 新增 `dy_values` / `dy_zero_count` /
  `dy_all_zero`（阈值 `DY_ZERO_EPS=0.005`）；`_chain_warnings` **只在
  `mode=="llm"` 时告警**，避免误伤 `llm_guided`（它的变换由 SIFT 求，
  模型 dy 根本不进矩阵）。前端结果卡片显示「垂直位移全为 0（接缝可能错位）」。
* `server.recognize()` 配对分支：`payload.ok=false` + `stitch.ok=true` +
  `backend=="llm"` 时，warnings 追加可操作归因「这是位移精度问题，不是
  轨迹颜色问题；建议改选 llm_guided 或 SIFT 用同一批截图复测」，不再只
  丢一句让人无从下手的「掩膜为空」。

#### 刻意没做 / 结论

* **没有让纯 VLM 变成几何配准**：dy 提示词只能让模型**尝试**给非零值，
  压不到像素级精度；这是 §21.5 的固有上限。要让接缝真正对齐，答案就是
  `llm_guided`（模型定 `order`，SIFT 求变换），它在 §22.54 已支持
  「order 相邻对补齐候选」。
* `stitch.ok` 产物级断言仍是 P2：本次仍是 `stitch.ok=true` + 整体
  `ok=false`，靠新增归因解释，没改 `ok` 语义（避免破坏既有流程与
  monkeypatch 契约）。
* ⚠️ 代码需**用户自行重启 8848** 才生效（约定：助手不重启服务）。

#### 验证

* 新增 4 例后端测试（v3 提示词契约、dy 全 0 检出、非零 dy 不误报、
  llm 告警而 guided 不告警）+ server 静态接线契约 + 前端 Node 渲染断言。
* 受影响组 `54 passed / 3 warnings`；全量 `pytest -q` **522 passed /
  5 warnings（467.72s）**，基线 518 + 新增 4，无回归。5 个 warning
  均为既有 `tmap_proxy` OpenAPI 重复 operation id，与本次无关。
* 建议复测顺序：先 `llm_guided`（预期接近 SIFT 基线 6423×3373 /
  `track_points≈122`），再 `llm` 看 v3 是否让 dy 非零、画布高是否
  从 2910 变高。

#### 教训

* **画布尺寸是免费的诊断信号**：653px 的高差直接暴露「模型忽略了垂直
  维度」，比读模型 notes 可靠得多。
* **「拼上了」≠「能识别」**：拼接层 `ok` 只看能否贴出画布；下游失败必须
  归因到上游，否则用户会去查颜色阈值、白折腾一轮。
* **别给不适用的后端加告警**：guided 不用模型 dy，同一条诊断必须按
  `mode` 分流，否则会误导正在做 A/B 的用户。

### 22.56 模板飞轮第一圈真正转起来：配对注册门槛重构 + 召回双通道（2026-11-14）

#### 背景与诉求

用户明确优先级：「重新查天地图不是最要紧，现在优先希望的是——之前用过多张
细节图拼接的路线，下次再有相同甚至类似的截图，可以不用再传细节图。」

这正是 P2 `register_pair()` / `locate()` 设计的能力，但落盘数据是铁证：
`data/route_templates.json` **根本不存在**，11:12 用户只传概览单图的测试
`conf=0.2` 街区级结果，模板通道全程未参与。两条路各有一个断点：

#### 断点一：注册路（用真图复盘逐门排查）

拿 11-13 14:05 的真实概览图（`1794549954_b9cb2d08.jpg`）与拼接画布
（`6423×3373`）重放 `register_pair` 每道门槛：

| 门槛 | 结果 |
|---|---|
| 概览↔画布 `match_shape` | **accepted，residual 0.0116、coverage 1.0** |
| `shape_has_enough_info(画布轨迹)` | **False（aspect≥4 扁长）→ `degenerate_shape`** |
| `confidence=0.2 < 0.35` | 会拒（死锁 A） |
| 画布 `input_quality=reject` | 会拒（死锁 C 连坐） |

即：**配对本身近乎完美，却被三道为「单图自动沉淀」设计的门槛连环拒杀**。
0.2 封顶是给「无街道名锚点的单图」的；`input_quality` 是画布 OCR 的属性；
画布扁长是「沿河近直路线」的客观形状——三者与「概览↔真值配对是否成立」
全部无关。

#### 断点二：召回路

概览单图 OCR 51 条文本，宽泛占比 0.373 < 0.5 → `assess_broad_overview`
False → 阶段 2.5 **根本不查模板库**。就算查了，库里也是空的。

#### 改动

* `route_template.register_pair()`：画布侧扁长否决移除（对应质量由
  `match_shape` 的残差/覆盖率/方向签名把关）；概览侧形状检查保留；新增
  地理真值里程区间校验（100m–200km，`geo_length_out_of_range`）。
* `TemplateStore.remember()`：门槛按来源分流——`auto` 维持三道防毒门；
  **`paired` 只留 `self_consistent`（多图拼接证据）**，置信度与
  input_ok 不再连坐。
* `locate()` / `find_similar()`：读侧置信度下限对 `paired` 模板豁免；
  `locate()` 命中累计 `hit_count`（`/api/health` 的 `templates.hits`
  可观察复用频率）。
* `server._capture_paired_template()`：证据门改为「payload ok +
  徽章守恒 + 逐像素复核 ≥0.5 + 拼接 ok」，`input_ok` 恒 True；新增
  `recognition_failed` 拒因；精修落盘合并保留 `paired_registration` /
  `template_capture`（修复后台精修覆盖丢键，11:12 那次诊断就是被它吃掉的）。
* `pipeline` 阶段 2.5 双通道触发：宽泛概览图照旧；**库里已有 overview
  模板时任何概览单图都探测**，非宽泛图要求近复刻（残差 ≤0.045 且
  coverage ≥0.90）才采信，否则静默回落。`_template_result` 告警按
  宽泛/近复刻分文案。
* 前端结果卡片新增「历史模板」行（模板 ID / 残差 / 覆盖 / 反向 / 里程）。

#### 真图闭环演练（临时模板库，未触碰运行期数据）

注册 `written=true`（conf 0.2 直接过门）→ 同图 locate residual **0.0021**、
coverage 1.0、`near_template_hit=True` → 反向变体命中。即上次那批截图
今后只传概览单图即可出参考级结果，参考级约束（≤0.45、`template_derived`
不沉淀）一条未放松。

#### 教训

* **门槛必须与证据同源**：给「单图 OCR 置信度」设计的门去拦「多图拼接
  证据」，等于用错尺子量人——三个死锁（§22.56 前的 A/C）在这条路上全部
  复现。
* **召回触发不能只依赖输入侧判定**：`assess_broad_overview` 是 OCR 启发式，
  概览图里街道名一多就失灵；库里有没有资产本身就是触发信号。
* **后台任务落盘要合并而不是覆盖**：精修 payload 天然没有主流程后加的
  字段，直接覆盖会把诊断证据抹掉，复盘时两头空。

### 22.57 飞轮第一圈用户实测闭环确认（2026-11-14）

§22.56 改完当天，用户用真实数据完成了完整闭环，两次实测把「演练结论」升级
为「实测事实」，也暴露出一个值得长期记住的观察点。

#### 实测一（11:43）：注册成功，单图连坐门槛确认解除

用户传「概览 + 7 张细节图」：SIFT 拼接画布 6423×3373、verify 1.0、徽章守恒
通过、`track_points=122` / 3495.7m；`paired_registration.written=true`，
**`data/route_templates.json` 项目史上第一次生成**（`tpl_1794627890_58d1f6e3`，
overview 域、64 点、上海）。关键在于这次是在 `input_quality=reject` +
`conf=0.2` 下过门的——旧代码必然拒收的三道单图门槛（画布扁长
`degenerate_shape`、置信度封顶连坐、`input_quality` 连坐）全部确认解除，
新证据门（拼接 ok + 徽章守恒 + verify ≥0.5）按设计工作。
`pair_match` residual **0.01164** 与 11-13 复盘演练的 0.0116 吻合。

#### 实测二（11:52）：单图复用命中，且双通道是必要条件

用户只传同一张概览图（165KB，无多图）：

| 项 | 实测值 |
|---|---|
| `diagnostics.template_match` | accepted、residual **0.00207**、coverage 1.0、未反向 |
| 输出 | 64 点弧长插值、GCJ 3406.1m（真值 3495.7m，差 ~2.6%）、conf 0.449 |
| 防自我强化 | `template_capture.written=false, reason=template_derived_result` |
| 复用计数 | `hit_count` 累计至 2（主流程 + 后台精修各一次 locate） |

坐标 121.349,31.223 → 121.381,31.217 落长宁天山一带，与真值路名归属一致。
结果卡片「历史模板」行正常显示（前端读 `dg.template_match`）。

**最重要的经验：双通道触发是必要条件，不是锦上添花。** 这张图的 OCR 宽泛
占比始终 0.373（<0.5），旧宽泛判据判定永远 false——如果没有「库里有 overview
模板就探测」这条通道，模板注册成功后也永远不会被召回，飞轮会在
「注册→召回」之间**静默断裂**：库文件存在、写入成功、`/api/health` 看着
正常，但没有任何一次识别实际受益。诊断这类「数据有了但系统没用上」的问题，
要看的是调用点而不是落盘文件。

#### 长期结论

* **门槛与证据同源 + 召回双通道**两条改造均经真图验证；「拼过的路线下次只
  传概览单图」目标落地。
* 参考级约束（confidence ≤0.45、`template_derived` 不回写、无当次核验
  标注）在实测中一条未放松，防自我强化设计有效。
* 下一个高价值人工动作：对命中结果的偏差走 `/api/feedback` 人工纠正，把
  模板 `verified` 提为 true——这是飞轮外部熵源第一次有真实数据可用。

### 22.58 输入源画像层：为高德/小米截图预留的模块化结构（2026-11-14）

#### 背景

用户决策按演化逻辑做模块化准备：① 为后续支持高德、小米等其它运动 App
截图预留结构，当前按输入源注册美团；② 之前接入的 LLM 多图拼接继续保留。

#### 架构判断（先想清楚再动手）

识别链路天然分两层，这决定了模块化的切口：

* **下游与来源无关**：多图拼接（SIFT 特征与轨迹颜色无关，`llm` /
  `llm_guided` 后端同样复用）、轨迹模板飞轮（按形状注册/召回，且形状
  域隔离机制现成）、地名 OCR、地理编码、路网精修、GeoJSON 输出。
  新增来源**零改动**。
* **上游与来源绑定**：轨迹线掩膜（美团浅荧光绿 `V>200/S<90`，高德
  亮蓝系互斥）、起/终徽章 HSV、形态学核、RANSAC 内点容差（底图 POI
  标注偏移决定）、里程 OCR 词表（「骑行/已骑/封顶」是美团文案）。
  每个 App 一套参数。

#### 改动

* 新增 `src/app/source_profile.py`：`SourceProfile(id, title, vision,
  distance_positive, distance_negative)` + 注册表 `PROFILES` +
  `resolve_source_profile()`（未知 id 报错并列出可用值，绝不静默回退）。
  当前只注册 `meituan`，参数值与重构前**逐字段一致**（测试逐项锁定）。
* `AppConfig` 新增 `source_profile_id` / `source` property：来源选择
  只有 `SOURCE_PROFILE` 一个环境变量（遵守「同一功能一个环境变量」
  约定），`.env` 改完即生效。切换是**全局**的（单进程单来源），
  逐请求自动检测截图来源属于拿到多样本后的后续演化，不提前设计。
* `input_quality.extract_distance_km()`：里程正/负词表改为跟随
  `CONFIG.source`（按词表元组缓存编译正则），词表唯一来源是画像，
  模块内不留副本。词表只影响**优先级排序**，裸数字兜底语义不变，
  现有无参调用与测试零改动。
* 诊断透传：`_base_diagnostics()` 增加 `source = {id, title}`；
  `GET /api/health` 增加 `source_profile` 字段——多来源时代排查
  「参数用错画像」的第一线索。
* `VisionConfig` docstring 标注美团归属：未来来源在画像层新建，
  不要改这里的默认值。

#### 明确不做的

* 不做逐请求来源自动检测（OCR 判 App 品牌）：没有多样本之前是空谈，
  且误判的代价是整条链路用错参数——宁可显式切换。
* 不在注册表放未标定占位条目：切换到盲参数等于全链路瞎跑。
* **LLM 拼接后端完整保留**：`stitch_backends/llm.py`、提示词 v3、
  诊断、前端下拉一律不动，默认后端仍 `sift`（11-13 已定性「实验位，
  不再投入」，但保留代码与可观测性）。

#### 验证

新增 `tests/test_source_profile.py` 7 例：默认画像参数逐字段锁定、
`CONFIG.source` 一致性、未知 id 报错文案、`SOURCE_PROFILE` 切换后
画像与词表立即跟随（含「词表只影响优先级、裸数字兜底」语义）、
诊断透传、注册表无占位。受影响组 89 passed；全量见提交信息。
未重启服务。

### 22.59 运动概要识别：订单详情页 → 完整运动记录（2026-11-14）

#### 诉求

此前只识别轨迹截图的经纬度；用户要求补「该次骑行的概要信息」——订单
详情页截图（美团：3.5km 本次骑行 / 骑行20分35秒 / 203kcal 本次消耗 /
172g 本次减碳 / 2026-11-11 20:20:26），与轨迹组合成完整运动记录。

#### 设计

* 新增 `src/app/ride_summary.py`：`parse_ride_summary(texts,
  geom_distance_m, ride_labels)` 纯函数解析。**防歧义按风险分级**：
  kcal / 时间戳 / 时长单位模式唯一直接提；`km` 多候选（页面同时有
  「1314km」累计与「3.5km」本次）用「本次骑行」标签的**文本块近邻绑定**
  （OCR 按阅读顺序输出、索引相邻），无标签取最小候选；减碳 `g` 极易
  误报，**必须**有「碳」标签近邻才采信（`_bind_labeled` 的单候选短路
  会绕过标签检查，碳字段单独判定——第一版在这里有 bug，被
  `test_carbon_requires_label_neighbor` 抓住）。
* 标签词表进 `SourceProfile.ride_labels`（美团：「本次骑行/本次消耗/
  本次减碳」），语义映射通用（含「骑行/里程」→距离、「消耗/卡」→
  卡路里、「碳」→减碳），文案随画像——画像层第一个真实复用点。
* 交叉校验：订单里程 vs 轨迹几何里程，比值 0.6–1.7 合理（复用
  input_quality 的尺度区间思想），不符出告警——订单值来自平台记账，
  偏差通常说明轨迹截图没截全。
* `server`：`recognize` / `recognize_multi` 均加可选 `summary: UploadFile`；
  `_attach_ride_summary()` 在 `_persist_last_result` 之前并入
  （顺序有契约测试锁定），解析失败只告警绝不影响轨迹识别。
* GeoJSON `properties["ride_summary"]`：非空字段并入 GCJ/WGS 两份
  FeatureCollection——输出即可直接导入运动记录库。
* 前端：第三个可选上传控件「③ 订单详情截图」+ 结果卡「运动概要」行
  （里程/时长/卡路里/减碳/开始时间/✓ 一致性）。

#### 测试坑

* `_bind_labeled` 单候选短路 vs 碳标签强校验的语义冲突（见上）。
* `SourceProfile` 加必填字段后，既有测试里的 fake profile 需要补
  `ride_labels`——dataclass 必填字段变更的连带面。

#### 验证

新增 `tests/test_ride_summary.py` 13 例（真实截图样例值 + 歧义 + 交叉
校验 + server 直测）与 `tests/test_frontend_ride_summary.py` 3 例；
全量见提交信息。未重启服务。

### §22.60 GPX 导出：轨迹 + 时间轴合成（2026-11-14）

**动机**：`ride_summary` 识别出来之后，里程/时长/开始时间只躺在
JSON 与 GeoJSON `properties` 里；运动平台（Strava / Garmin / 佳明
Connect）只认 GPX 的逐点 `<time>`——GeoJSON 没有时间轴，导入后时长、
配速全丢。把「轨迹 + 概要」合成 GPX 才是完整闭环。

**实现**（`src/app/gpx.py` + server 三处接线）：

* `build_gpx(coords, ride_summary=..., source_title=...)`：WGS-84
  `[lon, lat]` 序列 → GPX 1.1 文本。坐标必须取 `payload["geojson"]`
  （WGS-84），GCJ-02 进 GPX 会整体偏 ~290m。
* **时间插值**：按累计里程比例（haversine）分配逐点时刻
  `t_i = start_utc + cumdist_i/total × duration`；里程 0 退化为按
  序号均匀。**坑**：第一版在本地时间域插值完没换 UTC，
  `_utc_z` 只做格式化不做换算，整条时间轴冒充 UTC 偏 8 小时——
  被 `test_track_times_span_start_to_end` 抓住。
* **时区**：订单页时间是东八区本地时间，GPX 要求 UTC；固定 +8h
  换算（中国无夏令时），不引入 zoneinfo/tzdata（Windows 缺 IANA 库）。
* 概要字段写 `<trk><extensions>` 自定义命名空间 `mr:`（distanceKm /
  durationS / caloriesKcal / carbonG）；平台忽略未知扩展不影响导入。
  只有时长没有开始时间时不给逐点 `<time>`（绝对时刻给不出，宁缺勿错）。
* `server._attach_gpx(payload)`：`recognize` / `recognize_multi` 在
  `_attach_ride_summary` 之后、`_persist_last_result` 之前调用；
  无坐标时不产出。失败只告警，绝不影响识别。
* **精修路径**：`_run_refine` 的 payload 重新组装，原只合并回
  `paired_registration/template_capture`；现在 `ride_summary` 也进
  合并清单（概要属于本次运动，精修不改起止时间），并按精修后坐标
  重新合成 GPX。顺带修正了「merged 落盘但 last_stitch/响应 payload
  拿不到合并字段」的不一致。
* 前端：`payload["gpx"]` → Blob（`application/gpx+xml`）下载链接
  「↓ 下载 GPX（含时间轴与运动概要）」，无 GPX 时隐藏。

**验证**：新增 `tests/test_gpx.py` 17 例（结构/插值/UTC/扩展/转义/
退化轨迹/server 直测/前端契约）；全量 567 passed。测试坑：自己先
把 20:20:26+20分35秒 算错成 20:40:41（应为 20:41:01）——3 例失败
全是断言算术错，代码无误。未重启服务。

### §22.61 GPX 时间轴消失根因：OCR 吞空格（2026-11-14 15:48）

用户实测反馈「GPX 里连运动开始时间都没有」。复盘 `last_result.json`：
时长 1235s / 里程 3.5km / 卡路里 203 全部解析成功，唯独 `started_at`
为 null——`duration_text` 是 **"2026-11-1120:20:26骑行20分35秒"**：
OCR 把日期与时间之间的空格吞掉，整块粘连输出。旧时间戳正则
`(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):...` 分隔符必选，匹配失败。
GPX 规则是「开始时间+时长齐全才写逐点 <time>」，于是整条时间轴
跟着消失——用户看到的现象与数据完全对上。

**修复**（ride_summary.py）：

* 分隔符改 `[ T]?`，按位数切分（日/时各 2 位、月可 1-2 位）；
* 解析后做**范围校验**（月≤12、日≤31、时≤23、分/秒≤59），
  粘连错位读出的胡值跳过、`finditer` 继续找下一个候选；
* 时长有但开始缺时，notes 提示「GPX 导出将没有时间轴」；
* `_attach_gpx` 缺时间轴时往 warnings 写明缺的字段与原因
  （started_at/duration_s 实际值），不再让用户对着 GPX 文件猜。

**教训**：OCR 输出的粘连是常态不是异常，时间/数字类正则的分隔符
一律可选 + 结果范围校验。新增 4 例回归（粘连样例用真实落盘字符串）；
全量 571 passed。未重启服务。

### §22.62 导出层模块化（exporters/）+ 交付坐标系统一 WGS-84（2026-11-14 16:26）

用户需求：文件生成模块化，后续加 TCX / FIT；GeoJSON 交付只留 WGS-84。

**实现**（`src/app/exporters/` 包）：

* `base.py` 公共基座：WGS-84 输入约定、haversine 累计里程、按里程
  比例的时间插值（含 2026-11-14 踩过的「本地域插值没换 UTC」防回归
  注释）、东八区固定 +8h、XML 转义、started_at 解析。
* `gpx.py`：从 `app/gpx.py` 迁入（原文件删除），签名统一为
  `build_gpx(coords, *, ride_summary, source_title, geom_distance_m)`
  ——GPX 不用几何里程（避免与订单里程两套数字），参数只是为注册表
  统一签名。第一版漏加该参数，`build_exports` 里 TypeError 被兜底
  告警吞掉只表现为「gpx 缺席」，测试抓住。
* `tcx.py`（新）：TCX v2，`Activity Sport="Biking"` + Lap
  （TotalTimeSeconds/DistanceMeters/Calories/TriggerMethod 必填序列）
  + 逐 Trackpoint Time/Position。**关键契约**：TCX 规范要求逐点
  `<Time>`，缺时间轴返回 `None` 缺席——宁缺勿错不产半合规文件。
  DistanceMeters 订单里程优先、退化几何里程；Calories 缺失填 0
  （schema 必填，0 合法且诚实）。
* `__init__.py`：注册表 `_REGISTRY` + `build_exports()`（跑全部格式，
  None 缺席）+ `export_one()`（未知格式报错并说明 FIT 待实现）。
  **加新格式只需：实现 build_xxx + 注册表加一行**。
* server：`_attach_gpx` → `_attach_exports`，payload 加 `payload["tcx"]`
  （`gpx` 键不变）；缺时间轴告警文案覆盖 GPX/TCX 两格式。
* 前端：GeoJSON 下载收敛为单个 WGS-84 链接（GCJ-02 链接删除），
  新增 TCX 链接（无产出隐藏），说明文案改为「导出统一 WGS-84，
  地图显示自动换算对齐 GCJ-02 底图」。
* CLI：只产出 `trajectory_wgs84.geojson`（`_summary` JSON 里的
  start/end_gcj02 字段保留——程序化消费契约，不动）。

**边界明确**：GCJ-02 **没有**从识别流水线移除——它是锚点地理编码
（`maps/geocoding` 契约统一输出 GCJ-02）的原生产物，页面地图 /
对照页 / 核对页 / trajmap 全靠它对齐 GCJ-02 底图；收敛的只是
「交付下载」。pipeline 的 `geojson_gcj02` 字段与 payload 字段保留。

**验证**：`tests/test_gpx.py` → `tests/test_exporters.py`（25 例，
新增注册表分发/TCX 结构与时间/TCX 缺时间轴/距离与卡路里退化/
payload 双格式/源标题注入/前端链接契约）；全量 579 passed。
未重启服务。

### §22.63 微信小程序前端（现 `wxmin/`，原 `ridewx/`，2026-11-14 17:40）

用户决定：小米扩展暂缓，先做小程序前端对接本项目后端（全流程 v1，
开发者工具联调）。

**工程**（现 `wxmin/`，原 `ridewx/`，原生微信小程序，16 文件）：

* 结构：`pages/index`（上传页：①概览图必填 ②细节图≤9 ③订单详情
  截图 + 城市/路名）→ `pages/result`（指标卡 + map 轨迹预览 + 导出 +
  纠正）；结果经 `getApp().globalData.lastResult` 传递（payload 含
  GPX/TCX 大文本，不走页面参数）。
* **multipart 手工构造**（`utils/multipart.js`）：`wx.uploadFile`
  一次只能带一个文件，而 `/api/recognize` 契约是单请求同时传
  `file + details(n) + summary`。方案：`wx.getFileSystemManager()
  .readFile` 读 ArrayBuffer → 手拼 multipart（自实现 UTF-8 编码器，
  城市名/路名含中文）→ `wx.request` 发 ArrayBuffer。文件名统一
  ASCII（overview.jpg/detail_N.jpg/summary.jpg）规避转义问题。
* 地图：`map` 组件用 `geojson_gcj02`（组件坐标系就是 GCJ-02）画
  polyline + 起终点 circle 标记；中心/缩放按 bbox 估算。里程用
  WGS-84 geojson 客户端 haversine 计算，与订单里程并列展示。
* 精修轮询：`refine_pending` 时每 3s 轮询 `/api/refine/{id}`，
  running 显示 elapsed_s，done 用 `job.payload` 整体替换并重渲染。
* 导出：GeoJSON/GPX/TCX 写入 `USER_DATA_PATH` 后
  `wx.shareFileMessage` 转发（分享面板取消时文件其实已写好，toast
  兜底）；与网页版一致统一 WGS-84。
* 纠正：`wx.chooseLocation`（返回 GCJ-02，直接传 `crs=GCJ-02`）
  + 地名输入 → `/api/feedback`，对应后端「越用越准」数据入口。
* `config.js` 单点配置 `BASE_URL`（默认 `http://127.0.0.1:8848`，
  `project.config.json` 已设 `urlCheck: false`）；真机预览需手机开
  调试模式并改成局域网 IP；正式发布需 https 域名。
* appid 占位 `touristappid`——IDE 打开工程需替换为真实 appid。

**校验**：JSON 严格性 + `node --check` 全部 JS 语法 + WXML 无
`?.`/`??`/箭头函数 + multipart 构造功能级验证（UTF-8 中文回读、
边界计数、闭合检查）全部通过。脚手架工具两次 `template_unavailable`
（模板哈希不匹配），降级为按原生工程约束手工建工程。

**IDE 联调状态**（截至本文档写入）：本机已装开发者工具
（`D:/soft/微信web开发者工具`）但 NW 运行时低于兼容门槛
（`MIN_COMPATIBLE_VERSION=2.02.2607152`，诊断
`nw_runtime_incompatible`），Nightly `2.02.2611132`（2026-11-13）
安装包已下载到用户 Downloads，待覆盖安装后用
`check_wechatide_status` 完成门禁再编译预览。

### §22.64 输入源画像升级为接口入参（2026-11-14 18:10）

用户决策：来源选择从「服务级环境变量」升级为「接口入参」，为单进程
多来源（美团/高德并存）铺路。

**机制**（`source_profile.py` + `config.py`）：

* `active_source_profile()` ContextVar 上下文管理器：请求处理期间把
  显式画像挂到 `CONFIG.source` / `CONFIG.vision` /
  `CONFIG.source_profile_id` 的既有读取点上，退出自动还原。选
  ContextVar 而不是全链路显式传参的理由：`CONFIG.source` /
  `CONFIG.vision` 的调用时刻读取点散在 pipeline / visuals / multishot /
  input_quality / ride_summary / server 十几处，显式传参是一次大范围
  签名重构且极易漏改；ContextVar 对既有代码完全透明。
* FastAPI yield 依赖 `_active_source`：解析 `source` 表单字段
  （空 → 环境默认；未知 id → 400 带可用值清单，绝不静默回退），
  `with active_source_profile(profile): yield`——async 依赖与端点
  主体同 task，端点结束自动还原，不跨请求泄漏。
* `AppConfig.vision` 从 dataclass 字段改为 property（跟随当前画像）；
  测试均以 `AppConfig()` 无参构造，无字段注入用法，零回归。

**模板库按画像隔离**（`route_template.py`）：

* 模板新增 `profile` 字段；`remember`/`register_pair` 写入，
  `locate`/`find_similar` 按画像过滤召回，`_find_duplicate` 只在同
  画像内去重。历史模板无该字段 → 视为 `meituan`。
* pipeline 阶段 2.5 的 locate 调用传 `profile=self.cfg.source_profile_id`
  （请求期 ContextVar 生效）。

**后台精修线程**：`threading.Thread` 不继承 ContextVar——
`_start_refine` 把 `profile_id` 记进 ctx，`_run_refine` 入口用
`activate_profile_globally()` 重建（daemon 线程只跑单一任务，无需
reset）。

**前端**：

* 网页版：表单新增「输入源」下拉，选项从 `/api/health`
  （`source_profiles.available`）动态下发——新注册画像前端零改动；
  两个识别接口统一 `fd.append('source', ...)`。
* 小程序：首页 picker 同样从 health 拉清单；`utils/api.js`
  recognize 增加 `opts.source`。
* health：`source_profile` 增加 `available` 清单。

**边界**：词典（lexicon）**不**按画像隔离——它是「地名→坐标」的
地理事实，与截图来源无关，共享反而有利于多来源互相印证；隔离的只有
像素形状驱动的模板库。

**验证**：`tests/test_source_param.py` 13 例（上下文切换/还原、vision
身份、模板隔离与历史数据兼容、同画像去重、端点 400/200/环境默认、
multi 端点、health 清单）；`test_paired_registration.py` 的 4 个
locate 桩件签名同步补 `profile` 形参。全量 592 passed。未重启服务。

### §22.65 模板召回劫持真值路径事故（2026-11-14 18:30）

用户实测：概览图 + 5 张细节图，识别结果却是**之前那条历史轨迹**
（地图蓝色路线与细节图真实路线完全无关）。复盘 `last_result.json`：

* `diagnostics.source = "template"`、`template_derived = true`，
  `template_match` 显示 `tpl_1794629201_6d1a7832`（hit_count 已 11）
  以 `reversed=true`、`residual 0.0266`、`coverage 0.9844`、
  `signature 0.3632` 命中——但同一次 payload 的告警里还有
  「已用 5 张细节图拼接配准，本次结果为地理真值」。自相矛盾。

**根因两条**：

1. **裁决顺序漏洞**：阶段 2.5 模板召回在 pipeline 内部最先执行，而
   paired 模式（带细节图）的拼接真值在它之后。召回命中即 `return`，
   真值路径被整体短路。召回的本意是「免传细节图时复用历史真值」，
   但用户**传了**细节图时它不该有任何发言权。
2. **近复刻门槛被实测证伪**：旧假设「不同路线 residual 都在 0.05
   以上」不成立——两条不同的上海长路线（走向相近的带弯长条形）在
   64 点归一化形状空间里反向匹配 residual 仅 0.0266。形状距离度量
   不了「走法」，签名 0.3632（同类实测≈0.00）才是真正的异类证据。

**修复**（`pipeline.py` + `server.py`）：

* `run()`/`_run_impl()` 新增 `skip_template_recall` 旗标；server 两个
  识别入口在带细节图（paired/multi）时一律置位。原则一句话：**用户
  愿意传细节图，就一定走当次地理配准；模板召回只服务「免传细节图」
  的复用场景，绝不与真值路径竞争。**
* `_near_template_hit` 收紧为三道门：residual ≤0.045（反向再减半至
  0.0225）+ coverage ≥0.90 + 方向签名 ≤0.30。真复刻（正/反向）实测
  residual 都在 0.01 量级，不会误伤。

**遗留**：模板 `hit_count` 被假命中持续抬高（本条已 11），命中率
统计失真；后续可考虑在用户纠正/低分反馈时回退计数。假命中的那条
模板本身是真数据（曾经真实注册），无需删除。

**验证**：`tests/test_template_recall_guard.py` 11 例（事故原值回归
锁定、真假复刻分流、端点接线、`_run_impl` 行为级验证——skip 时
locate 一次都不调用）；全量 603 passed。未重启服务。

### §22.66 模板复用三态入参 + 结果展示（2026-11-14 18:50）

用户需求：① 提供选项「优先使用模板」；② 识别结果展示是否参考了模板。

**接口**（`template` 表单入参，两个识别端点统一）：

* ``auto``（默认）：没传细节图才允许召回；带细节图一律当次配准
  （§22.65 事故后的安全默认）。
* ``prefer``：优先模板——带细节图也先探测，近复刻命中直接返回
  参考级结果，未命中回落完整配准流程。注意：拼接在 pipeline 之前
  执行，prefer 命中省的是**配准**时间而非拼接时间。
* ``off``：禁用模板，每次独立配准。
* 未知取值 400 带可用值清单；**校验必须在 try 之外做**——
  HTTPException 在 `_pipeline.run` 参数求值里抛出会被
  `except Exception` 包成 500（测试抓住过）。
* payload 新增顶层 ``template_mode`` 回显本次生效策略。

**展示**：

* 网页版结果卡顶部 stat 新增「模板」卡：`⚠️ 参考历史模板`（参考级）
  vs `本次独立配准`；诊断区的「历史模板」明细行（id/残差/覆盖/反向）
  已有，保持不变。
* 小程序结果页指标卡同步新增「模板」行；上传页新增「模板复用」
  picker（auto/prefer/off），`utils/api.js` 传 `template` 字段。

**文案一致性**：prefer 命中时 paired 模式的提示语分流——模板插值
结果**绝不**配「本次结果为地理真值」（§22.65 事故的展示层教训），
改为「优先模板已命中……参考级（未做当次地理配准）；如需本次独立
配准请把模板复用改为自动后重试」。

**验证**：`tests/test_template_recall_guard.py` 扩到 19 例（prefer
覆盖 paired skip / off 强制跳过 / auto 默认 / 非法值 400 / multi
prefer / prefer 命中不宣称真值 / 网页与小程序静态契约）；全量
611 passed。未重启服务。

### §22.67 COS 云记录、部署、unionId 对齐、gateway 统一登录、小程序上线（2026-11-15）

§22.63 之后至小程序上线之间的整批改动（提交号以实际仓库为准，
截至本节写入 HEAD 为 `24d8d61`）：

**云存储与后端**（`5590615`、`2ae980e`）：

* 运动记录存腾讯 COS（`src/app/records/`：keys/auth/store/service），
  `uid = HMAC(RECORDS_TOKEN_SECRET, openid)[:32]` 分目录；带合法
  Bearer 识别后**响应后的后台任务**自动保存，不带 token 一次 COS
  调用都不发。7 项 `.env` 配置齐全才启用，未启用零回归。
* 部署 `https://track.daozhao.com`（pm2 + Nginx）。**504 根因是
  Nginx `proxy_read_timeout` 默认 60s**，与慢识别路径冲突；已按
  README「反代超时与冷启动」对齐 600s。OCR 预热进 `run.py`
  daemon 线程，COS 保存改 BackgroundTasks 并显式 Timeout。

**unionId 跨端身份对齐**（`24d8d61`，参考 gateway 项目）：

* uid 不换主键、零 COS 迁移；新增脱敏
  `union_uid = "u_" + HMAC(secret, "union:" + unionid)[:32]`，
  token claims 加可选 `un`（v 仍 1，旧 token 继续有效），meta 写
  `union_uid`，别名索引 `{prefix}aliases/{union_uid}.json` 可重建。
* `GET /api/records/identity` 只认 token 内 `un`，不接受客户端传参。
* 🔴 `RECORDS_TOKEN_SECRET` 同时决定 uid 与 union_uid，**不可轮换**。

**目录与文案**：仓库 `meituan-ride` → `track-buddy`、小程序目录
`ridewx/` → `wxmin/`、布局 WIP 定稿入库（`7110c24`/`e6c381f`）；
小程序字号全面上调（`7df1e6c`）、文案淡化「骑行」（`b444e87`）、
网页版同步（`7325f45`）。

**gateway 统一登录**（`gateway.daozhao.com.cn`，main 提交
`004ec6d`）：

* `wxService.js` 新增参数化 code2session + `TRACKBUDDY_WXMIN_*`
  独立凭据组 + `getTrackBuddyWxminOpenId`；`POST /wxminTrackBuddy/login`
  用镜像签发器 `trackBuddyRecordToken.js` 直发与 Python 侧**逐字节
  兼容**的三段式 HMAC 记录 token（Python↔JS 交叉向量回归锁定）。
* 小程序 `config.js` 加独立 `LOGIN_BASE_URL`（不动全局 BASE_URL），
  `wxlogin()` 优先 gateway、网络故障/5xx/404 降级回
  `/api/auth/wxlogin`，400/401 不降级。
* 🔴 两侧 secret 必须逐字节相同；gateway 自 2026-11-15 起**统一加载
  仓库根 `.env`**（不再按 NODE_ENV 区分 `.env.dev`/`.env.prod`）。

**小程序上线（2026-11-15 已发布）**前后的三处功能调整：

* 导出文件名以运动开始时间命名：`result.js` 新增 `exportBaseName`，
  产出 `fitmemo-YYYYMMDD-HHMMSS.<ext>`；概要缺失回退
  `trajectory*` 旧名（网页版/CLI 下载名不变）。
* 导出分享面板加固：`doExport` 改 `fs.writeFileSync` 同步写保住
  点击手势上下文（异步回调里调 `wx.shareFileMessage` 部分机型拒弹）；
  低版本基础库升级提示；fail 区分「用户取消」与真错误。导出到手机
  的唯一通道是分享面板发「文件传输助手」；文件实际存于
  `wx.env.USER_DATA_PATH` 沙箱，工具里不支持该 API 属正常。
* **轨迹纠正暂时下线**：`wx.chooseLocation` 需在微信公众平台额外
  申请——`result.wxml` 纠正卡片、`result.js` 三个处理函数、
  `app.json` 的 `requiredPrivateInfos`/`scope.userLocation` 全部
  移除（恢复步骤写在 result.js 注释里）；后端 `/api/feedback` 与
  词典逻辑、网页版纠正不受影响。

**上线验证**：登录（gateway 签发）、识别、云端记录列表/详情/删除、
GPX/TCX/GeoJSON 导出全链路真机可用。


---

## 23. 新增截图识别来源：Source Profile 扩展手册（给下一个 AI，2026-11-16 新增）

> 本章是**下一个开发目标的操作手册**：给小米 / Keep / Garmin / Coros
> 等运动 App 接入截图识别。读完本章 + §3（架构）+ §4（坐标系） +
> §11（排障）即可开工，不必通读全文。

### 23.1 核心判断：下游复用，上游每 App 一套

识别链路天然分两层（决策依据见 `src/app/source_profile.py` 模块
docstring 与 §22.64）：

| 层 | 内容 | 新来源要改吗 |
|---|---|---|
| **下游（与来源无关）** | 多图拼接（SIFT / llm / llm_guided）、轨迹模板飞轮、地名 OCR、地理编码、路网精修、GeoJSON/GPX/TCX 导出、COS 云记录 | **零改动复用** |
| **上游（与来源绑定）** | 轨迹线掩膜阈值、起/终徽章 HSV、形态学核、RANSAC 内点容差、里程/概要 OCR 词表 | **每个 App 一套参数**，互斥不能混用 |

例：美团轨迹是浅荧光绿（`V>200 且 S<90` 可提取），高德是亮蓝色系，
参数完全互斥——这正是 Source Profile 这一层存在的原因。

### 23.2 改动清单（以「新增 amap」为例，共一个文件 + 若干测试）

**唯一必须改的代码文件：`src/app/source_profile.py`**

1. 在 `PROFILES` 注册表加条目（⚠️ **只允许已标定的画像**，不放未
   标定占位——切到盲参数等于全链路瞎跑）：

```python
AMAP = SourceProfile(
    id="amap",                      # source 入参 / SOURCE_PROFILE 环境变量取值
    title="高德运动",                # 前端与诊断展示名
    vision=VisionConfig(...),       # 按真实截图标定，不要抄美团默认值
    distance_positive=(...),        # 里程正面词（命中则块内数字优先）
    distance_negative=(...),        # 里程干扰词（命中整块排除）
    ride_labels=(...),              # 运动概要页的「本次」字段标签
)
PROFILES["amap"] = AMAP
```

2. `resolve_source_profile` / `/api/health` 的
   `source_profile.available` / 网页版与小程序的来源选择器**自动
   生效，无需改动**（它们都读注册表）。

**`VisionConfig` 字段标定指南**（定义在 `config.py:99`，默认值是
美团标定结果，新画像在 `VisionConfig(...)` 里覆盖）：

| 参数 | 标定方法 | 美团参考值 |
|---|---|---|
| `track_max_value` / `track_max_saturation` | 对样本截图统计轨迹像素 HSV 分布，目标是「只提轨迹线，避开彩色虚线/文字/底图」 | V≤200 且 S≤90（浅荧光绿） |
| `start_hsv_range` / `end_hsv_range(+2)` | 起/终徽章图标色相区间，用真实徽章像素采样 | 起绿(35-85) / 终红(0-12, 170-179) |
| `badge_min_area` / `badge_max_area` | 徽章实际像素面积 × 安全系数（虚线碎片 150-230，徽章 4000+） | 1500–20000 |
| `track_close_kernel` / `track_open_kernel` | 闭运算跨断点、开运算滤细笔画；按轨迹线宽（美团 8-10px）调 | 11 / 5 |
| `ransac_inlier_m` | 底图 POI 文字标注中心与真实点位的固有偏移决定 | 200 米（偏移 7-28px ≈ 35-150 米） |
| `ocr_min_score` | OCR 置信度下限，按目标 App 字体清晰度 | 0.55 |

**OCR 词表**（`distance_positive/negative`、`ride_labels`）：
收集目标 App 的里程文案与**运营干扰文案**（累计值、广告、会员）。
`ride_summary` 的语义映射是通用的——含「骑行/里程」→ 本次距离、
「消耗/卡」→ 卡路里、「碳」→ 减碳克数；只有标签**文案**需要按
App 填（美团：「本次骑行/本次消耗/本次减碳」）。

### 23.3 标定样本要求（瓶颈所在，先收集再动代码）

- **5-10 张真实截图**：概览单图 + 沿路线滑动的细节图（多图模式）、
  不同缩放级别、至少一条**有对照真值**的路线（在真实地图上手动
  描一遍真轨迹，或用同路线 GPX）。
- **运动概要页截图**若干（测 `ride_labels` 与时间正则）。
- 评测命令：`python -m app.eval`（合成基准）+ CLI 单图跑真实样本，
  看 `overlay_result.png`（轨迹叠回截图）与 `road_verify.png`
  （路网校验）人工核对——这两个产出是标定是否成功的直接判据。

### 23.4 生效机制（不用改，但要懂）

- **优先级**：请求级入参 `source`（表单字段，server.py 的
  `_active_source` 依赖）> 环境变量 `SOURCE_PROFILE`（默认
  `meituan`）。未知 id 直接报错列出可用值，**绝不静默回退**。
- 画像通过 **ContextVar**（`active_source_profile`）在请求处理期间
  生效，`CONFIG.source` / `CONFIG.vision` 自动跟随——识别链路读取点
  零改动。单进程可同时服务多来源请求。
- ⚠️ **后台线程不继承 ContextVar**：路网精修线程（`server._run_refine`）
  已在入口显式重建画像上下文；新加后台线程时必须照做
  （`activate_profile_globally`）。
- **模板库按画像隔离**（`route_template.py` 的 `profile` 字段）：
  美团模板永远不会命中其它 App 截图；模板召回另有三道门
  （residual/coverage/方向签名，见 §22.65 事故与
  `test_template_recall_guard.py`），`skip_template_recall` 与
  `template=auto/prefer/off` 入参对所有画像统一生效。

### 23.5 验收清单（全部绿灯才算接入完成）

1. `tests/test_source_profile.py` / `test_source_param.py` 扩展：
   新画像解析、入参切换、未知 id 报错文案。
2. 真实样本单图识别：轨迹形状正确、`road_verify` 重合、置信度合理。
3. 多图模式（带细节图）+ 运动概要截图：`ride_summary` 各字段解析
   正确、里程交叉校验（0.6–1.7 区间）通过。
4. 模板隔离测试：新画像截图不命中美团模板（反之亦然）。
5. `/api/health` 的 `source_profile.available` 出现新 id；网页版
   与小程序来源选择器直接可选（零改动）。
6. 全量 `pytest -q` 无回归（当前基线 **654 passed / 5 warnings**，
   2026-11-15；沙箱拦 Temp 时用 `--basetemp=/tmp/wb_xxx_$(date +%s)`）。
7. 文档同步：本节加一行接入记录 + README「输入源画像」更新注册状态。

### 23.6 硬约束（违反即返工，详图见对应章节）

- **坐标系**（§4）：内部全程 GCJ-02，**交付文件（GeoJSON/GPX/TCX）
  必须 WGS-84**；底图同源地理编码才可收紧 RANSAC 容差。
- **地理编码单源天地图**（`MAP_GEOCODING_ORDER`、`TIANDITU_API_KEY`）：
  🔴 高德配额已尽，**不要**建议或尝试接入高德在线编码。
- **合规**（§14）：只用截图上印着的公开 POI 文案，不逆向地图瓦片。
- 概要时间解析：东八区固定 +8h 转 UTC（中国无夏令时），导出层契约
  不动；OCR 正则已兼容粘连文本（`2026-11-1120:20:26` 类）。

### 23.7 协作约定（与用户已确认的工作方式）

- **一次一项**：按清单逐步推进，完成一步提交一次；先完整分析再实施。
- 🔴 **服务器由用户自己重启**：不要 kill/start 8848、不要起临时端口
  做 health 验证；本地验证只用 pytest / 纯函数调用。
- 统一解释器 `D:/soft/Python311/python.exe`（3.11.4，已装
  rapidocr-onnxruntime 1.4.4 + opencv-python 4.13；**不要**在该环境
  `pip install -r requirements.txt`，cv2 会混装冲突）。
- `.env` 与运行期数据不提交；新增脚本一律放 `src/tools/`
  （`test_project_layout.py` 禁止仓库根出现 `tools/`）。
- 每日日志追加两处：`D:\Users\hxiong\Dropbox\memoHub\track-buddy\YYYY-MM-DD.md`
  与工作区 `.workbuddy/memory/YYYY-MM-DD.md`。
- 输出偏好：中文、结构清晰、标优先级（🔴🔴🟡）、要可直接部署的代码。

### 23.8 接入记录：amap（高德运动）初版（2026-11-16）

**状态：代码侧已接入，参数为初版估计值，待真实样本精标。**
本章 23.3 的「先收集样本再动代码」流程被用户决策调整为
「先注册初版画像 → 用户网页实测 → 样本回传精标」迭代模式，
故 amap 是注册表里第一个**非全标定**画像（代码注释与测试均已
显式标注，防止误当已精标）。

改动清单：

| 文件 | 内容 |
|---|---|
| `src/app/source_profile.py` | 注册 `AMAP`：区间式轨迹掩膜 `(95,120,150,120,255,255)`（高饱和亮蓝 #1677FF 系），徽章沿用起绿/终红假设，里程词表按高德文案估计 |
| `src/app/config.py` | `VisionConfig` 新增可选字段 `track_hsv_range`（None=美团上限式，区间式优先；默认值行为零变化） |
| `src/app/track.py` | `build_track_mask` 支持双模式：`cv2.inRange` 区间式 / 上限式 |
| `src/tools/calibrate_profile.py` | **新增**标定 CLI：全景 HSV 统计（`--hue` 色相带细分 / `--badges` 徽章候选 / `--profile` 掩膜覆盖率复核） |
| `tests/test_source_profile.py` | 锁定 amap 初版参数锚点 + 双掩膜模式行为测试；反例 id 从 amap 换成 keep |
| `tests/test_source_param.py` | 同上：400 反例换成 keep |

验收状态（§23.5）：①⑤⑥ 已过（pytest 全量无回归）；②③④（真实样本
识别 / 多图+概要 / 模板隔离实测）**待用户网页测试回传样本后执行**。

精标流程（拿到高德真实截图后）：

```bash
python src/tools/calibrate_profile.py samples/amap_1.jpg --hue 100 115
python src/tools/calibrate_profile.py samples/amap_1.jpg --badges
python src/tools/calibrate_profile.py samples/amap_1.jpg --profile amap
```

读 S/V 分位数（p5 留 10-20 余量做下限）回改 `AMAP` 的
`track_hsv_range` / 徽章 HSV / 面积阈值，再更新
`test_amap_profile_initial_params_locked` 锚点。

### 23.8b 概要先验兜底：主截图底部卡片（2026-11-24 网页端事故）

**「CLI 对、网页错」的同图分叉根因。** 同一张小米截图：CLI 传
`--summary` 落点正确；网页端没传概要图 → `distance_hint_km=None` →
外部尺度先验与时长-里程物理闸门全部失明 → 2 锚点（热情世界杯杨/姜庵）
模型的尺度爆炸无拦截，轨迹整体偏出约 40km（落进泗门镇一带）。而小米
的距离/时长**就印在主截图底部统计卡片上**——被 `map_crop_v` 裁掉后
pipeline 看不见，等于守着先验瞎跑。

修复（三层兜底，全部默认生效）：

- `ride_summary.parse_summary_from_region(img)`：对图像区域做概要
  解析的共用入口（OCR + `parse_ride_summary`，无字段返回 None）。
- server（单图/多图端点）：没传概要图且画像带 `map_crop_v` 时，保留
  裁剪前原图，对被裁掉的底部条带调用兜底解析（`_fallback_summary_from_main`），
  结果作为 `summary_pre` 参与尺度先验；`_attach_ride_summary` 改为
  接受「无概要文件但 pre_parsed 非空」——兜底概要照常进
  `payload["ride_summary"]`，里程交叉校验与物理速度闸门一并生效。
- pipeline（CLI/直跑）：`run()` 自行裁剪时同样保留底部条带，
  `distance_hint_km` 缺失时兜底解析补先验。

实测（out/xiaomi_v8_nosummary，不传概要图、默认 provider 顺序）：与传
`--summary` 的 v7 完全一致——交付 13.193km、scale 7.293 m/px、路网
归属鄞奉路/澄浪南路（奉化江沿岸）。回归 231 passed
（server/pipeline/visual/multishot/refine/xiaomi/xingzhe/source/
summary/export）。测试新增：`_attach_ride_summary` 接受无文件的
pre_parsed、`parse_summary_from_region` 空图安全。

### 23.9b 分辨率归一化：vision_ref_short_side 预缩放（2026-11-25）

**背景**：小米 1.25× 等比截图（1080×2400 vs 标定 864×1920）暴露了
「按标定分辨率实测的绝对像素参数」在全链路的分辨率依赖（§24.21 修了
抹除半径/容差/配对上限三处，但徽章**面积窗口** `badge_min/max_area`
（随分辨率²缩放）、行者 `scale_badge_d=39.2` 锁定值、行者
`endpoint_fragment_detect=(100,80)` 仍是绝对的；历史上传已有
2850×4945 / 3864×4177 / 6694×9896 等大图进过系统）。

**方案对比**：逐参数改相对（badge_d 比例系数） vs 输入预归一化。
选后者：一次 resize 让全部标定值原样生效（等价于「全参数相对化」但
零重标定）、改动集中单入口、行为可用合成缩放样本验证。纯相对化的
结构性缺陷：badge 检测的面积窗口是**入场筛选**而 badge_d 恰从检测
结果量出（先有鸡还是先有蛋），必须引入预估计且估计失败时无兜底——
`Scale.BADGE_D_MIN/MAX` clamp 因此保留为安全网（相对为主、绝对为界）。

**实现**：
- `VisionConfig.vision_ref_short_side`：meituan 1260 / xingzhe 640 /
  xiaomi 864（各自标定样本短边）；amap 标定样本分辨率未记录，None=
  维持现状，待下次高德样本补测。
- `pipeline.prepare_input_image`：裁剪（crop_v 相对原图高度）→ 等比
  缩放到参考短边（INTER_AREA 缩小 / CUBIC 放大，±2px 容差内跳过），
  返回 meta `{"crop_v", "prepared_hw"}`。
- `RidePipeline.run(..., prep_meta=)`：server 预处理路径把 meta 传回，
  diagnostics 下发 `vision_prep.prepared_hw`（`map_crop_applied` 兼容
  键不变）。**顺带修了一个潜在 bug**：server 单图端点此前裁剪不记录
  crop，overlay 反投影会整体画偏（裁掉多少偏多少）；且
  `_fallback_summary_from_main` 被用 3 参数调用（签名只有 2 个），
  无概要图的网页请求会 TypeError。
- `visuals._registration`：按 `vision_prep.prepared_hw` 精确复现缩放，
  overlay / road_verify / chart 反投影逐像素对齐。
- server 单图/多图/细节图入口统一走 `prepare_input_image`；多图拼接
  画布不适用逐图 meta（`prep_meta=None if paired_mode`）。
- `endpoint_fragment_detect_rel=(radius_per_badge, min_area_per_badge_d2)`：
  行者切换到相对形式 (2.55, 0.052)，绝对形式保留为回退。
- **死配置清理**：`track_close_kernel` / `track_open_kernel` 无任何代码
  读取（track.py 全用 scale 派生核），已删除。

**验证**：`tests/test_resolution_normalization.py` 8 项——把标定样本
合成 1.25×（行者/小米）/0.5×（美团）缩放图跑完整提取链，断言 prepared
尺寸只由画像决定、徽章定位 ≤8px、弧长 ≤8% 漂移；行者/小米/美团三样本
全过。端到端：1080×2400 小米图 CLI 复跑，提取弧长 1861px（864 原图
1857px），overlay 逐像素贴合，交付 13.234km vs 概要 13.09km（1.1%）。
回归：resolution/xiaomi/xingzhe/source/anchor/ride_summary/export
141 passed + server/pipeline/visual/multishot/refine/track/loop
116 passed。内核已同步两份技能目录。

**遗留**：~~高德画像待补测标定样本短边后启用归一化~~ → 已于 2026-11-25
补齐（见 §23.9c）：标定样本从 uploads 模板飞轮找回（amap_ride_shanghai.jpg，
1260×2720），裁剪后短边 1020，AMAP.vision_ref_short_side=1020 生效。
4K 级输入归一化后 OCR 的 POI 文字变小（与标定状态一致，理论上无回归），
首个 4K 真实样本出现时复核一次锚点数量即可。

### 23.9c amap 短边补测：vision_ref_short_side=1020（2026-11-25）

**状态：四画像全部启用分辨率归一化，amap 遗留项关闭。**

标定样本找回过程：`samples/ride_motion.jpg` 并非高德样本（实为户外跑步页，
花瓣地图水印、纯绿轨迹），2026-11-21 的 `out/amap_only_rerun/amap_run_3p73km`
只是拿它跑了 amap 画像。真样本通过 `data/route_templates.json` 的模板飞轮
找回：`tpl_1794799987`（profile=amap）→ shot `1794799494_c36f39d3.jpg`
（data/uploads，2026-11-16 上海 3.68km 骑行，高德地图水印）。已归档为
`samples/amap_ride_shanghai.jpg`（样本不入库原则不变，.gitignore 仍排除）。

实测：1260×2720（与美团标定机同档），map_crop_v=(0.065,0.44) 裁出
1260×1020 → 裁剪后短边 **1020**。标定分辨率输入裁剪后零缩放直通，
badge_min_area/max_area 等像素参数行为不变。基础提取验证：弧长 1070.9px
= 模板 length_px 1070.91（逐位吻合），起终点徽章均定位。

改动：AMAP.vision_ref_short_side=1020（source_profile.py）；
test_resolution_normalization.py 加 amap 参数化用例 + 1.25× 样本不变性
用例（10 项全过）；test_source_profile 快照断言更新；
test_track_shape / test_user_multishot_real 的 `test_real_samples_not_broken`
改为画像感知（_PROFILE_SAMPLES 映射：amap 样本走 prepare_input_image +
AMAP.vision，其余仍走默认 CONFIG）。

教训：**标定样本必须当时归档进 samples/**——这次靠模板飞轮的 px_points
存档才找回真样本，若模板也过期就只能重测。全量回归 783 项 passed。

### 23.9 接入记录：xiaomi（小米运动健康）闭环精修（2026-11-23 当日二次）

**状态：轨迹提取全对（形状/顺序/里程），绝对位置仍为 2 锚点近似。**

首测差评后二次解剖：开环模式只描出上行半程 805px，右侧下行段被 POI 文字
打断整段丢失；实测该路线是**闭环棒棒糖**——起点绿点(334,796)→走廊上行→
顶部菱形环→走廊原路返回→绿点→红徽章终点(324,929)。三项画像级修复：

- `loop_course=True + loop_bridge_fragments=True`：闭环邮路追踪描出全程。
  同时 `pair_loop_badges=False`——行者式「成对相邻」约束（gap≤图高 8%
  ≈83px）对小米不成立（起终点相距 133px=走廊 ~1.2km），会把走廊上的
  红色轨迹块配成「终点」(363,852)，污染端点拟合。
- `erase_marker_blobs_r=45`：绿定位点/红徽章颜色落在轨迹色带内，外围
  花瓣状 UI 晕圈半径 ~42px；只挖中心会留下「绕洞的环」，walk 绕环徘徊
  （底段 153px 直线走出 518px，弧长虚高 20%+）。45px 抹除 + 新函数
  `_reconnect_marker_blobs` 按环带像素分侧连弦补线，弧长 1857px 与
  几何估计 1810px 差 3%。
- `split_loop_delivery=True` + `_lollipop_order`：邮路奇度配对 100px
  上限不复制走廊（500px 超限），欧拉端点落在环底结点/尖端。按语义重排
  起点→走廊→环→走廊→起点→终点，端点精确对齐绿点/红徽章。

实测（余姚样本→宁波鄞奉路一带）：8/8 锚点、RMSE 261→183m、交付
13.193km vs 概要 13.09km（0.8%）、scale 7.293 m/px、路网归属沿奉化江
道路与截图走向一致。`visuals._delivered_coords` 顺带修掉配准失败时
`KeyError: 'features'` 的不可读报错。新增 config 字段
`pair_loop_badges / split_loop_delivery / erase_marker_blobs_r`
（默认全关，既有画像行为不变）。测试补 `_lollipop_order` 单元用例，
xiaomi/xingzhe/source/anchor 73 passed。

### 23.9a 接入记录：xiaomi（小米运动健康）可测试初版（2026-11-23）

**状态：已注册并可供网页/API 测试；不是绝对位置精标完成。**

首张真实样本为 864×1920 的「小米运动健康｜户外跑步」概览截图，标定到
`src/app/source_profile.py` 的 `XIAOMI`：

- 地图区约占整图上部 54%，使用 `map_crop_v=(0.0, 0.54)`；概要卡片
  仍由独立 OCR 通道读取，不参与轨迹视觉提取。
- 轨迹为红/橙/黄主体加绿色快速段，使用两条高饱和 HSV 色带并集：
  `H[0,40], H[40,85]`，`S>=140, V>=120`。
- 当前按开环处理：`loop_course=False`。样本中绿色定位点和红色点相距约
  134px，尚不足以证明起终点重合，不能套行者的闭环成对徽章逻辑。
- 当前徽章面积不作为硬失败条件，避免把小米轨迹同色的定位点/终点误当成
  美团大徽章；后续样本需继续标定起终点检测，尤其要区分地图 POI 红色图标。
- 概要词表覆盖「公里 / 户外跑步 / 运动 / 活动卡路里」，负向词覆盖累计、
  历史、平均配速等统计干扰。

回归测试为 `tests/test_xiaomi_profile.py`，当前只验证来源注册、地图裁剪、
渐变掩膜、里程词表和开环语义；不把单张样本的绝对地理位置当作已验证结论。
拿到更多小米概览/动态轨迹截图后，继续补 marker、锚点质量、路网验证和多图
拼接验收。

### 23.10 amap 真实样本精标 + 多图链路修复（2026-11-16 当日）

用户用网页版试了高德 8 图多图识别（上海 3.68km 骑行），失败现场
`data/last_result.json`（「轨迹提取失败：掩膜为空」）。逐层排查出
**三个根因**，全部修复：

1. **轨迹不是蓝色，是速度渐变线**。高德轨迹按速度着色：
   红（最慢）→橙→黄→绿（最快），详情页底部图例条就是同款渐变。
   初版假设的亮蓝（#1677FF）完全错误。实测标定（工具
   `tools/calibrate_profile.py` + 掩膜可视化人工核对）：
   - 主体橙红黄带 `H[8,30] S≥150`（S p25≈191；H[80-89] 的 15.5 万
     高饱和像素全是 UI 按钮，不是轨迹）；
   - 快速段绿带 `H[35,70] S≥180`（薄荷绿公园 S≈50-90 被排除）；
   - **终点红标 #E94340 在 H[0,7]**，与轨迹红段 H[8,30] 天然错开，
     徽章范围上界压到 7 即不串扰；洋红端 end_hsv_range2 必须关闭
     （H[170,179] 命中详情页红心装饰）。
   - `track_hsv_range` 因此扩展为**多色带并集**（6 元组的元组），
     `build_track_mask` 对各带分别 `inRange` 后取并。

2. **整屏截图的 UI 卡片与轨迹同色系**。高德截图地图区只占上部
   ≈40%，下方卡片里的速度图例条（同款渐变）、红心装饰、内嵌地图的
   视频缩略图全部污染掩膜/徽章/SIFT。修复：`VisionConfig` 新增
   `map_crop_v=(top,bottom)` 比例裁剪，`server` 在**拼接前逐图**裁出
   地图区（高德实测 `y∈[6.5%,44%]`）；概要页走独立通道不受影响；
   None=不裁（美团零变化）。

3. **里程牌 + 接缝把轨迹切成多段，旧「单一连通域」假设失效**。
   3KM/2KM/1KM 白底胶囊与拼接接缝把轨迹切成 3-5 段，旧逻辑只取
   最大段丢 30-40% 路线。修复：`track.py` 新增
   `_merge_track_fragments`（端点邻近串接：从徽章碎片/最大段出发，
   反复并入「距链端点 ≤ bridge_gap」的线状碎片，宁缺勿错接）；
   单徽章画布（细节图无起终点圆点）新增拓扑豁免——像素级贴住现有
   徽章且长宽比 ≥2.5 时跳过「宽≈图宽」误杀。
   ⚠️ 两个防回归闸门（腾讯截图实测教训，`test_user_tencent_shots_rejected`
   锁定）：双徽章时 `_select_with_markers` 的拒绝不能被串接翻案，
   除非串接链像素级贴住**两端**；单徽章豁免必须同时要求形状像线。

修复后离线复现（同 8 图）：裁剪 → SIFT 拼接 7.5s → 轨迹 5055 点
全程贴合（含红色终点标到 1KM 后的对角段）。待用户网页端验证完整
识别（OCR 锚点 → 配准 → 路网精修）。

**✅ 2026-11-16 14:37 网页复测通过**：8/8 图拼接，north_up 5 内点
RMSE 95.4m，路网精修平移 东+115m/北+20m 贴路中位 31m→4m，几何里程
3652.6m ≈ App 显示 3.68km，落点长宁区（新渔路/天山西路等）与路名
归属吻合，用户确认轨迹正确。amap 首次端到端成功。

排查时两个小插曲（记录给后来者）：
- 13:39 前一次失败（10:14）是**页面旧 + 画像没切**：请求 source 走了
  meituan 默认。诊断方法：`last_result.json` 的
  `diagnostics.source.id`（读请求级 ContextVar，准确）+ 拼接画布是否
  残留 UI 卡片（amap 画像会裁掉）。
- **里程牌污染里程 OCR**（已修）：拼接画布上带 km 的 OCR 文本只有
  轨迹里程牌「3KM/2KM/1KM」，`extract_distance_km` 取中位 = 2.0，
  与几何 3.65km 比值 1.826 误报、置信度被压到 0.2（结果本身是对的）。
  修复：`input_quality.py` 排除纯整数+KM 的无上下文标签
  （`_MILESTONE_RE`，真实里程总带小数或文案上下文），回归测试
  `test_extract_distance_ignores_track_milestone_labels`。另：
  `anchors_used` 在 payload **顶层**（不在 diagnostics 里），前端读的
  就是顶层字段——排查时别看错位置。

**⚠️ 配对注册画像漏传（2026-11-16 16:12 修复，重要经验）**：多图成功
后补传概览「下次单图复用」的飞轮第一次实测就断了——单图上传没命中
模板，走了完整识别还给出错误定位（north_up_2pt 2 内点 RMSE 455m，
落点偏到人民广场）。根因：`_capture_paired_template` 收了
`profile_id` 参数但调 `register_pair` 时**漏传** `profile`，空串被
`_norm_profile` 回落成默认 meituan——amap 真值注册的模板盖上美团
的章，召回时被画像隔离过滤（`find_similar` 的 profile 过滤）。
修复三层：① server.py 补传 `profile=profile_id`；② 数据修复：把
`tpl_1794799987_36aa98b9`（shot_id 1794799494_c36f39d3，amap 真值）
的 profile 改回 amap（备份
`route_templates.json.bak-20261116-profilefix`）；③ 回归测试
`test_paired_template_stamps_request_profile`（端到端：source=amap
注册 → 模板 profile=amap）+ `test_paired_template_recall_is_profile_
isolated`（召回侧隔离）。修复后离线验证：同张概览图召回命中
residual 0.0022 / coverage 1.0。
**教训**：新增「请求级参数 → 后台/旁路任务」的透传链时，链上每一跳
都要核对——ContextVar 只活在请求 task 里，显式传参的函数内部还要
继续传下去才算到位。

**经验**：接入新来源前先拿 1 张真实截图跑 `calibrate_profile.py`
统计 + 掩膜可视化人工核对，比按公开视觉特征估计靠谱得多——
「高德轨迹是蓝色」就是想当然的产物。

### 23.10 amap 运动概要页解析精标 + 标签绑定行感知升级（2026-11-16 17:00）

小程序上线高德后用户反馈「识别成功但没有运动概要信息」。现场：
概要页截图有被处理，但 `ride_summary.available=false`（连与来源无关
的 kcal/时间戳都提不出）。对真实概要页 OCR 实测，高德的文案/排版
与美团完全不同：

| 字段 | 美团（解析器原有假设） | 高德实测 |
|---|---|---|
| 开始时间 | `2026-11-11 20:20:26` 横杠 | `2026.11.1608:13:27` **点分** + 与时间粘连 |
| 时长 | `骑行20分35秒` | `00:14:14` **冒号三段式** |
| 热量 | `203kcal` | `79大卡`（「大卡」） |
| 里程 | `3.5km` 数字单位同块 | `3.68` + `干米` **分块**，且 OCR 把「千」误读成「干」 |
| 干扰 | 累计里程、运营文案 | 轨迹里程牌 `3KM/2KM/1KM`、比例尺 `1公里`、配速 `15千米/时`、状态栏时钟 `08:34` |

修复（`ride_summary.py`，全部向后兼容、美团行为零变化）：

1. 时间戳分隔符族谱扩展：`[.\-/年]` 点分/横杠/斜杠/年月日；
2. 时长新增冒号三段式 `HH:MM:SS`（跳过含时间戳的块防尾段
   `:13:27` 误配；**两段式 MM:SS 不采**——状态栏时钟 `08:34` 会被
   读成 8 分 34 秒）；
3. kcal 单位族扩展 `千卡/大卡`；
4. km 单位族扩展 `千米/公里`，负向环视排除配速（单位后跟 `/每`）；
   数字与单位分块时（纯数字块 + 下一块纯单位块）也构成候选；
   `_norm_token` 把 OCR 误读 `干米` 归一成 `千米`；
5. **标签绑定升级为行感知**（关键）：`_bind_labeled` 有几何框时改用
   「同一文本行内水平最近」策略——索引距离在高德这种地图区自带
   数字的页面上没有区分度（`骑行` 标签按块索引同时够到比例尺
   `1公里` 和概要卡 `3.68`，旧逻辑取最小值就选错成 1.0）。同行判定
   = 纵向重叠超较矮块一半 + 水平间距 < 8×字高。无几何框（纯字符
   串输入/测试）回落旧索引窗口路径；
6. AMAP `ride_labels` 精标为实测文案 `("骑行", "消耗热量")`（高德
   骑行页没有减碳字段，宁缺勿错）；
7. 提示文案去美团化：「订单详情页」→「运动概要页」。

真实页面端到端复现：3.68km（绑定「骑行」，5 个干扰候选全排除）/
854s / 79 kcal / 2026-11-16T08:13:27，交叉校验比值 0.982 一致——
started_at + duration 齐了，GPX 导出从此有时间轴。新增回归测试
4 例（真实布局几何框模拟），`test_ride_summary.py` 19 passed。

---

## §24 诚实失败闸门链 + 多源地理编码 + 概要里程尺度先验（2026-11-17 新增）

> **一句话**：352 米高德短骑行从「被识别成 51.5km 假轨迹」走到「322m 正确识别」，
> 靠的不是针对该案例的定制逻辑，而是三类**通用机制**：垃圾锚点进不来（泛词过滤）、
> 概要里程读得出来（OCR 分块组合）、配准尺度有物理约束（外部尺度先验 + 物理闸门）。
> 代码中出现的「352」「南天」等字样全部只存在于注释（案例记录），零硬编码逻辑。

### 24.1 问题链路（三因叠加）

1. **高德 key 月度配额耗尽**（USER_DAILY_QUERY_OVER_LIMIT，12/1 恢复），
   天地图单源 POI 覆盖弱（商场内业态如「荟聚Livat/诺富特」不收录），20 锚点只命中 1；
2. **泛词锚点组成 2 点退化模型**：「大厦」「诺富特」等泛词在地名库里同名要素遍地，
   相似变换无冗余校验，尺度直接爆炸（352m → 51.5km）；
3. **概要页「骑行352米」OCR 解析失败**：大字「352」（y=419）与单位「米」（y=453）
   被切成两个独立文本块，旧裸米分支只认同块组合。

### 24.2 修复清单（全部通用规则，含提交号）

| 修复 | 文件 | 提交 | 性质 |
|---|---|---|---|
| 泛词黑名单（16 个通用词单独出现不作锚点） | `ocr_anchor.py` | 8bd3675 | 通用文本规则 |
| 裸「米」单位解析 + 数字块与相邻「米」块组合 | `ride_summary.py` | 8bd3675 / 1ebd891 | 通用 OCR 分块 |
| 概要里程硬闸门（配准里程/概要里程 比值 0.33~3 之外拒绝） | `server.py` | 8bd3675 | 通用数量级校验 |
| 米候选「无标签宁缺勿错」（防爬升值被误当里程） | `ride_summary.py` | c98a8b5 | 通用保守策略 |
| 精修继承闸门裁决（`_inherit_refine_fields` 重跑闸门） | `server.py` | c98a8b5 | 流程正确性 |
| 地理编码诊断明细透传前端（console.table） | `pipeline.py` / `index.html` | f2c815b | 可观测性 |
| BaiduProvider 鉴权参数 bug（key→ak，此前从未真正可用）+ city_limit + 候选上限 3→5 | `maps/geocoding/providers.py` | a79aa84 | 配置驱动接入 |
| 物理速度闸门（识别里程/概要时长 > 20 m/s 拒绝） | `server.py` | 1a8e2f3 | 物理常识 |
| 概要图诊断落盘 `data/uploads/<ts>_summary.jpg` | `server.py` | 0bbf2d8 | 可观测性 |
| **外部尺度先验**（概要里程 ÷ 轨迹像素长度 = m/px，融合进 RANSAC） | `pipeline.py` / `register.py` | 9279c3f | 纯几何原理 |

### 24.3 尺度先验细节（本次核心突破）

- `server._preparse_ride_summary` 端点**前置解析**概要图：一次 OCR 同时供
  `distance_hint_km`（pipeline）与 ride_summary attach 复用；
- `pipeline`：配准前算「轨迹像素长度 ÷ 概要里程 = 地面米/像素」传 `register`；
- `register.external_scale_prior` 融合规则：与内部投票先验比对——
  投票失效或偏离 >35% → 用物理里程（枚举容差 30%）；一致 → 用投票先验（容差 12%）；
- 为什么必须：街区级特写（~0.5 m/px）下 POI 标签排版偏移 100-200m 相对小地面
  跨度方差巨大，纯像素容差 RANSAC 必崩；米制尺度约束把垃圾词自然排除。

**验证结果**：322m vs 352m（比值 0.915）、north_up、0.39 m/px、
4 内点全为荟聚真锚点、路名归属（金钟路/协和路）正确，683 passed。

### 24.4 ProviderRegistry 架构（此前已有，非本次新增）

多源地理编码自始就是模块化插件架构（`src/app/maps/geocoding/`）：

- 每个源一个 Provider 类（Amap/Tmap/Baidu/Tianditu），声明
  `meta.output_crs`（GCJ-02/BD-09/CGCS2000）与 `credential_aliases`；
- `ProviderRegistry` 统一调度，顺序由 `.env` 的 `MAP_GEOCODING_ORDER` 控制
  （当前 `baidu,tianditu`；**12 月 1 日高德恢复后改回 `amap,baidu,tianditu`**）；
- 单源严格模式 vs 多源回退（`MAP_GEOCODING_FALLBACK=multi`）；
- 热配置：改 `.env` 即时生效（指纹缓存 mtime_ns+size+sha256）。

### 24.5 OCR 与地图 POI 搜索的解耦关系（架构确认）

**两者完全解耦，边界是「锚点候选」数据结构：**

```
OCR 层（通用，与地图源无关）                地理编码层（可插拔）
┌─────────────────────────┐            ┌──────────────────────┐
│ ocr_anchor.py            │            │ maps/geocoding/       │
│ · 任意截图读出文字块+框   │  锚点候选   │  AmapProvider         │
│ · 分类 place/road/marker │ ─────────→ │  BaiduProvider        │
│ · 泛词黑名单过滤          │ (name,kind,│  TmapProvider         │
│ · 不关心谁去查坐标        │  px,py)    │  TiandituProvider     │
└─────────────────────────┘            └──────────────────────┘
```

- OCR 层只产出 `(name, kind, px, py)` 锚点候选，**不知道也不关心**
  坐标由谁提供——换百度/高德/天地图/OSM 零改动 OCR 代码；
- 地理编码层只接受「地名 + 城市提示」，**不知道坐标将来配什么图**；
- 因此同一套 OCR 能力天然适配任意来源画像（meituan/amap/…）与
  任意地图源；换源只动 `.env` 或 Provider 类；
- 唯一的跨层约定：`kind` 分类（place 参与配准 / road 只作区域校验）
  与 `output_crs` 归一（各源坐标由 registry 统一转 GCJ-02）。

### 24.6 遗留事项

- 置信度 0.2 偏保守（真实正确结果也被压低），可调但先保守观察；
- 荟聚Livat 在 pipeline 内 city_limit 下偶发 miss（provider 直测有候选）；
- 垃圾 OCR 词的百度模糊匹配过滤（百度什么都能配，天地图查不到会放弃）；
- 城市短道路路名作弱锚点需重新设计（取离轨迹最近点/路口），未实施。

### 24.7 行者（xingzhe）来源画像接入（2026-11-17 21:00，真实样本标定）

第三类版式：宁波 34.67km 骑行，**卫星影像底图 + 橙红轨迹 + 蓝色规划线 +
红色三角旗起点 / 白底红字水滴终点**（样本 `samples/xingzhe_ningbo.jpg`）。
与美团（浅荧光绿轨迹、普通底图）、高德（速度渐变线）都不同。

**为什么默认画像必败**：美团上限式掩膜 `V<200,S<90` 把整张卫星图吞进来
（实测 57 万像素 = 全图 79%），起终点徽章误检成底图色块，里程闸门因顶栏
「距离 34.67」无 km 单位读不到而失守——输出 8.5km 假轨迹（真实 34.67km）。

**标定结论**（全部按真实样本像素统计，非拍脑袋）：

| 参数 | 值 | 依据 |
|---|---|---|
| 轨迹掩膜 | 区间式 `H[5,22] S[150,255]` | 橙红轨迹 H≈7-9/S≈175-190；卫星底图/蓝规划线(H≈95)被色相排除 |
| 徽章检测 | `badge_min_fill=0.25` | 徽章是团块(fill≈0.37)、轨迹是细长线(fill≈0.06)，填充率区分同色两者 |
| 徽章区间 | 起终点都 `H[0,15] S[90,255]` + 排除起点区域找终点 | 起红旗 S255、终红字 S103，无法用饱和度分；用排除半径区分 |
| 形态学核 | close 7 / open 5 | 轨迹线宽 ~8px |

**两处通用代码增强**（其他画像零影响，默认参数保持历史行为）：

1. `cv_utils.find_marker` 新增 `min_fill` / `exclude_pt` / `exclude_radius`
   参数——填充率筛选团块徽章（默认 0 = 关闭）、检测终点时排除起点团块。
2. `track._trace_graph_path` 环形鲁棒性：最短路只覆盖 <70% 骨架时（环上抄
   近道漏段），改用「从起点出发到最远可达点」回溯。实测行者最短路 976 点
   /1145px 漏掉左半环，改进后 1736 点 /1960px 覆盖全程。

**成果**：轨迹掩膜单一连通域 13970px（干净），徽章起(42,315)/终(334,1079)
定位准确，折线覆盖全程。端到端 9 内点、26.3km、路名归属宁波江北/海曙区
真实道路（环城北路/姚江东路/永丰北路），定位正确。

**遗留**：①里程 26.3km vs 真实 34.67km（偏低 24%），配准尺度待优化；
②顶栏「距离 34.67」无单位，概要正则读不到，里程闸门对行者不生效；
③置信度 0.2 为保守封顶。全量 687 passed（新测试 test_xingzhe_profile.py 4 例）。

### 24.8 终点「跑两遍」折返段：能检测、暂不接入（2026-11-17 排查结论）

用户观察：宁波样本底部有一段轨迹「跑了两遍」，识别结果里没有。

**根因（已用像素级实测确认）**：那段是终点徽章下方的 U 形折返（去一回一），
被**白色「终」徽章 + 蓝色定位点物理遮挡**，露出的橙红像素碎成一个独立小
连通域（area≈364px，bbox≈29×38，质心≈(311,1079)）。`_select_component`
只取最大连通域（主轨迹 13970px），这段因此被当噪声丢弃。
**不是颜色问题**——它的颜色本就在主掩膜色带内且已开运算去噪。

**能可靠做到**：从主掩膜里分离终点 100px 邻域内、不属于最大连通域的碎片，
干净剔出这个 U 形（无徽章红字/底图黄线污染，因为不做颜色重提）。已实现为
`track.detect_endpoint_fragment`（画像配置 `endpoint_fragment_detect=(100,80)`），
在 notes 里报告像素量：*"检测到终点邻域折返碎片 364px……形状不可靠，未纳入轨迹"*。

**暂不接入 polyline 的原因（多种方案实测均失败）**：主掩膜骨架有 **708 个
度≥3 分叉点**（自交 + 形态学毛刺），环形轨迹的「骑行顺序」无法由拓扑唯一确定：

| 尝试 | 结果 |
|---|---|
| 起点→终点最短路 | 漏掉整个左环（975 点 / 1118px，终点对但形状错） |
| 起点→最远点（已提交版） | 覆盖左环但漏末段（1736 点 / y只到930） |
| 起点→最远点→终点（拼接） | 里程虚高 25km→40km（重叠段重复计入） |
| 测地距离差 ds−de 排序 | 1098 处大跳（分叉处交替排列） |
| 贪心方向跟踪 | 死胡同卡住（180 点） |
| 去重拼接 p1+extra | arc=2604px（≈34km 吻合！）但有 1 处 532px 跳跃直线 |

最后一种最接近：去重后弧长 2604px 与图上 34.67km 高度吻合，但在最远点与
末段衔接处留一条穿越地图的直线，无法作为有效轨迹交付。**结论**：要正确还原
这类环形自交轨迹的顺序，需要在骨架上去毛刺 + 分叉点按曲率消歧（研究级问题），
不适合作为补丁上线。当前采取「检测 + 如实报告、不伪造轨迹」的诚实策略。

置信度 0.2 与里程偏低 27% 也与此同源：环形配准 + 顶栏里程无单位（未进尺度闸门）。

### 24.9 闭环赛道：起点=终点 + 中国邮路排序（2026-11-18，问题彻底解决）

**用户提供的关键线索**：「这个图其实终点和起点是一样的」。由此推翻 §24.8 的
开放路径假设，整个排序问题变成有强约束的闭环问题，此前 6 种方案失败的根因
一并揭开。

**用户线索的像素级验证**：
1. 左边缘 (42,315) 的红白大箭头是**导航指北针**（旧起点检测误检）；
2. 真**起点是「终」徽章旁的蓝色定位圆点** (365,1102)，与终点 (334,1079)
   仅相距 38.6px——闭环确认（HSV：H≈104 S≈162 V≈220，area 775/fill 0.67）；
3. 图左下角比例尺实测 **77px = 1km → 12.99 m/px**（此前里程换算无标尺）。

**失败方案追加记录**（都在「像素级图直接做邮路」的路上）：

| 尝试 | 结果 | 教训 |
|---|---|---|
| 像素级 8 邻接图直接邮路 | walk 5023 点弧长 5637px=73km | 524 个奇度节点全是骨架伪影，配对复制 2650px |
| 短弧收缩（删像素） | 收缩后总长 1786px=23km | 把「跑两遍」的双线合并成单线，丢一份重复长度 |
| 簇层邮路（union 抽象节点） | 展开错乱 11773px | 抽象根节点无物理像素锚，展开方向失效 |
| 焊接奇度近邻（merge into a） | 奇度 524→133 几乎不减 | 焊接级联产生新奇度点 |
| 手写 Guo-Hall 细化 | odd 1362 更糟 | 实现质量不如现有 Zhang-Suen |
| 线宽判别「跑两遍」段 | 无效 | 宽段散布全图，与重合段不相关 |

**根因（本轮最重要的发现）**：Zhang-Suen 细化对 **45° 斜线留下 2px 厚的
锯齿双列带**（算法已知缺陷，斜向光栅化无法收敛到单像素）。实测 20× 放大：
骨架斜线段是双列平行像素链，每个阶梯转折都是度3节点——**641 个「分叉」
的绝大多数是细化伪影，不是轨迹拓扑**。下游任何图算法都在为这个劣质骨架买单。

**三步骨架清理（实测 junc 641→19、奇度 524→28、单连通 comps=1）**：

1. `_press_skeleton`：删除 2px 带冗余像素。判据=像素有 4-邻接邻居且删除后
   其 8 邻域仍连通。**必须逐删+即时重查**——批量标记统一删会把骨架切成
   128 片（先删者破坏后删者的局部连通）；
2. `_weld_skeleton_parts`：分量间最近点对桥接（gap≤20px）；
3. `_bridge_skeleton_breaks`：奇度端点「物理近距(≤15px)但图距远(>50步)」
   的掩膜断口直线桥（宁波样本左中部：gap 8px / 图距 2166px，POI 遮挡）。

**中国邮路闭环追踪 `_trace_loop_path`**：
- 奇度节点最短路贪心配对（成本≤100px），复制配对路径 → 全偶多重图。
  **「跑两遍」的重合往返段天然被计两遍**——这正是往返段在骨架上只出现
  一次的补偿机制；
- 成本 >100px 的一对视为**自然开放端**（起终点徽章遮挡处）——起点=终点
  语义下欧拉路径从一个端头走到另一个端头，不复制 126px 飞线；
- Hierholzer 栈式遍历输出 walk（每步 1-1.4px 无跳步），剩余未走边 >2%
  即放弃回退开环。

**终点碎片接入**：`detect_endpoint_fragment` 重构为 `_endpoint_fragment_mask`
（返回碎片掩膜），闭环下桥接进主连通域（gap 33px）再统一骨架化。

**产品管线与实测**（弧长 2560px = **33.25km，误差 4.1%**；旧开环 25.2km
误差 27%；碎片 364px 已接入；conf 0.95）：

| 环节 | 配置/函数 |
|---|---|
| 起点检测 | `start_hsv_range=(95,100,150,120,255,255)` 蓝色定位点 |
| 终点检测 | 闭环时=起点邻域(15%图高)内面积最大红色团块（轨迹末端 195px 团块比真徽章 452px 更近起点，最近优先会错） |
| 起终点互斥 | 仅 `start_hsv_range == end_hsv_range` 时排除（蓝点距终点 39px，旧逻辑会把终点一起排除） |
| 尺度标尺 | `scale_badge_d=39.2` 显式锁定（起点换蓝点后自动反推 31.4 会把 close_kernel 7→5，主域 13970→9690） |
| 连通域 | `_select_loop_component`：终点贴住的最大连通域（双徽章相近时拓扑选择会误选 6043px 碎片） |
| 排序 | `_trace_loop_path`（压薄→焊接→桥接→邮路） |

**遗留（诚实边界）**：剩余 ~4% 里程差 = 压薄删除的 185px 双列带（≈2.4km）。
无法从图像区分「单线锯齿双列」与「跑两遍真双线」——若强行保留双列长度，
单线锯齿段会虚增。4.1% 在截图转换场景可接受；精确里程以 App OCR 为准。

**移植教训（写给下一个改这段代码的人）**：
- `connectedComponentsWithStats` 的组件像素要从**返回的 labels** 取，
  不是从输入二值掩膜取（`frag==i` 对 i≥2 恒空——实测让碎片接入静默失效）；
- `_select_component` 在双徽章相近（闭环）时会选中徽章旁小碎片；
- 行者样本的 `XINGZHE.vision` 是 module 级共享实例，extract_track 不得
  就地修改 cfg 与 img。

### 24.10 闭环管线端到端：地理配准核验 + 顶栏里程按列解析 + GPX 导出（2026-11-19）

闭环轨迹（§24.9）接入完整管线后的三个补强，全部由真实样本端到端验证。

**① 地理配准核验（虚惊一场 → 三个可信锚点 250-700m 内）**

初步看 CLI 摘要曾误判「定位偏 8km」——把新起点的经纬度与旧运行的
罗盘误检像素（图像北端）对比了。用在线编码真值核验：市中医院
(120,326)、江北区政府 (351,364)、江东区政府 (468,530) 投影误差
250-700m（结构性锚点偏差范围内），起点外推 (121.56, 29.80) 正确。
overlay/road_verify 图确认青色轨迹全程贴合。诊断脚本
`tmp_debug/georef_diag.py`（hook register dump 全部备选解）。

**② 顶栏无单位里程按列解析（本会话最初诊断的落地）**

行者顶栏「距离/速度/时间」三列，数字行无单位且跨列粘连（OCR 成
「34.67新=江18.29」），`_DISTANCE_RE`（要求单位）漏掉。硬补「无单位
也认」会把均速 18.29 当里程——必须按列归属：

- `extract_distance_column(img, texts)`（input_quality.py）：找正面词
  标签块 → 列带裁切（标签左缘-10 到右邻块左缘-6）→ 2× 放大二次 OCR
  → 取第一个合理数字。TextItem 只有块级 bbox 没有字级坐标，粘连块
  只能回原图裁。实测精确解出 **34.67**（均速列不吞）。
- **接入配准尺度先验**：pipeline 阶段 2.6 计算 `ocr_distance_km`
  （单位正则 → 按列兜底），`external_prior = (distance_hint_km or
  ocr_distance_km) × 1000 / 轨迹像素长`。概要页里程（服务端）仍优先。
- **接入 scale_check**：几何里程 33.94km vs OCR 34.67km，比值 0.979 ✓
  （此前 unavailable）。尺度 13.637 m/px 由物理里程锁定。

**③ CLI 补齐运动文件导出（server 已有，CLI 一直缺失）**

`cli.main` 调 `build_exports`（WGS-84 坐标），输出 `track.gpx` /
`track.tcx` 进 files 字典。无概要页时 GPX 纯轨迹导出、TCX 因缺时间轴
合规性缺席并出 warning（宁缺勿错契约）。

**端到端实测**（SOURCE_PROFILE=xingzhe，--online）：conf 0.2（绝对位置
自校验 583m>400m 结构性封顶，诚实报告）、11 内点/10 锚点 LOO、
33.94km（误差 2.1%）、GPX 1804 点。测试 +1
（`test_topbar_distance_column_parse`），全量 690 passed。

**遗留**：conf 封顶来自锚点反投影中位 583m 的结构性偏差（多源基准
不一致），需要锚点源统一策略才能突破——见 anchor_bias 诊断里的
「内点混用 3 个坐标源」警告。

### 24.11 路网温和吸附 + 里程弧长修复 + 尺度强制校准（2026-11-18 下午）

用户反馈「最终画出来的路线好多都压根不在马路上」，诊断出三个独立机制
并全部修复：

**① 里程 -12% 的真凶：RDP 弧长坍缩（意外发现）**

原始 2273 点描迹的地面长度 = **34.9km，与 App 记录 34.67km 只差 0.6%**——
之前交付 30.4km（-12%）是 `simplify_polyline` 的弧长判据把近直路段
连锁坍缩削掉的（eps=0.075px 看似极小，但弧长判据对「近乎完美直线」
的长段会一路坍缩到端点）。修复：吸附交付路径跳过 RDP，改用
`resample_polyline(step_px=2.0)` 等距重采样（弧长保真，~1300 点）。
**教训：eps 再小的 DP 也可能在长直段上连锁坍缩；弧长敏感场景必须
用「重采样」而不是「简化」。**

**② 全局 HMM+Dijkstra 吸附：实验后废弃**

先实现了完整方案（HMM 选状态 + 相邻状态间路网 Dijkstra，way 级链式
边 + 3m 路口连接），实测全部配置（max_points 600-1200 × k 4-6）长度
虚增 **29-42%**、百余处跨街跳段——绝对位置偏差 ~300-500m 是结构性的，
HMM 的发射项（σ=25m）在 78-82m 平均吸附距离下完全被转移项淹没，
状态选进错误的平行路族，Dijkstra 只能绕路连接。**在锚点源统一之前，
全局重匹配是在跟系统性偏差搏斗，注定失败**——代码删除，教训记录。

**③ 温和吸附（gentle_snap，采纳方案）**

轨迹本身已大致贴路（描迹到 OSM 路网中位 52m），不需要路网先验重塑
形状：离路 ≤60m 的点投影到路上（54% 点，吸附距离中位 21m），离路
超容差的点**保持原样**（诚实：吸附修不了绝对位置偏差，不伪造）。
两道防抖机制：窗口多数表决（孤点吸附多为平行路误吸，回退）、跳变
回退（吸附翻转处相邻点 >100m 时回退为原始坐标，不产生跨街直线）。
前端底图为腾讯/高德（GCJ-02），OSM-in-GCJ 与之对齐（差异=OSM 精度），
吸附显示一致。

**④ 里程尺度强制校准（阶段 5.7）**

OCR/概要里程与几何里程偏差 >8% 时绕轨迹质心缩放模型（×先验/几何），
把混源内点拟合的尺度误差拉回 ±3% 内。必须**先校准再吸附**——吸附点
lying on roads 是相对固定路网的，先吸附再校准会让吸附点漂离路网。
实测宁波样本弧长修复后 +0.7%，校准不触发（不需要）；坏拟合场景
（13:22 运行 -12.4%）会触发。

**管线顺序**（阶段 5.8a → 5.7 → 5.8b）：准备（弧长保真重采样）→
尺度校准 → 温和吸附执行。

**端到端实测**（宁波闭环，--online）：1281 点、**35.36km（+2.0%）**、
scale_check ratio 1.02、687/1281 点上路、4 处跳变回退、max_step 95m
（无跨街直线）、overlay 青线沿街平滑闭合。测试 +2
（`test_road_snap_delivery` / `test_gentle_snap_unit`），全量
634+18+40=692 passed。

**遗留**：46% 的点离路 >60m 保持原样——绝对位置偏差的根治仍需统一
锚点源（🔴 配置级：MAP_GEOCODING_ORDER 单源化）。

---

### 24.12 高德 API 中转服务：本地开发复用服务器高额度 key（2026-11-18）

**问题**：拿到一个高额度高德 Web key，但只对服务器出口 IP 开放（IP 白名单），
本地开发直连必被拒；而真实 key 不该复制到本地。

**方案**：服务器上跑独立轻量中转（不挂主服务），本地所有高德请求改走中转，
由中转在服务器侧注入 key 后转发。完整部署与鉴权细节见 `docs/amap_relay.md`。

| 组件 | 职责 | 运行位置 |
|---|---|---|
| `run_relay.py` + `ecosystem.relay.config.js` | 启动入口 / pm2（`amap-relay`，8849） | 服务器 |
| `src/app/amap_relay/server.py` | 中转：令牌鉴权 fail-closed + 固定上游 + key 注入 | 服务器 |
| `src/app/amap_relay/client.py` | 本地请求 URL 改写（未配置时原样返回=零回归） | 本地 |

四个高德调用点已接入：地理编码 provider、路名归属、静态地图、路网吸附。
提交 `d569f70`；配套修复 `e494644`（scale setter 崩溃 + 测试隔离）。

**🔴 排障教训（三条，都是真踩过的坑）**：
1. **本地代理伪造故障**：开发机有 `HTTP_PROXY=127.0.0.1:61284`，curl/requests
   全被劫持，8849 被代理报 `upstream connect failed`，一度误诊为服务器 nginx
   错误。**排障先加 `curl --noproxy '*'`**，项目侧用 `.env` 的 `NO_PROXY` 绕行；
2. **端口放行是两层**：腾讯云 Lighthouse 控制台防火墙 + 服务器 OS 的
   firewalld（OpenCloudOS 默认只开 22/80/443）。`tcpdump` 有 In SYN 无 SYN-ACK
   = OS 层拦，一个包都没有 = 控制台层拦；
3. **测试隔离**：`config.py` 导入时 `load_dotenv()` 把 .env 灌进 `os.environ`，
   开发者本地开了 relay 后，用临时 env_path 的 AppConfig 仍会经 `_env_value`
   回退读到 relay 配置——测试必须显式 `delenv` 或替换 `client._default_cfg`。

### 24.13 单次地理编码名额可配置（`GEOCODE_TARGET_LIMIT`，提交 `3f54e68`）

「超出单次地理编码名额」是项目自己的护栏（按词典命中/权重截断锚点），
**与高德配额无关**。原先硬编码 20，现改为 `.env` 热读：
`GEOCODE_TARGET_LIMIT=50`（范围 1-200，默认 20），无需重启。
42 锚点的截图建议设 50；注意 provider 的 0.35s 限流会线性拉长耗时。

### 24.14 overlay 展示层模型与交付层对齐（提交 `e93e138`）

**现象**：结果页绿线（识别轨迹反投影）相对橙线（原图轨迹）整体偏 1–2km，
黄点锚点位置却正确。

**根因**：`visuals.build_overlay` 不走管线交付的模型，自己重新拟合，且硬编码
`city=上海` / `allow_online=False`（美团上海时代遗留）——同一张上海样本拟出另一个
错误模型。**交付 GeoJSON 本身没错，是「展示层」与「交付层」用了两个模型。**

**修复**：`pipeline.py` 在 diagnostics 透传 `model_matrix` / `model_translation`；
`visuals._registration` 优先用交付模型构建 `Registration`，重拟合只作旧数据兜底
（城市取 `res` 真实值）。

> 🔴 **通用教训**：展示层永远不要重新推导交付层已有的结果——"复现 ≠ 一致"，
> 条件稍异即漂移。`res` 应携带足够信息让所有下游渲染零重算。

### 24.15 前端原图叠加在布局变化后整体错位（提交 `a3f9771`）

**现象**：原图预览上的 canvas 叠加层相对红轨迹左偏约 60px，缩放窗口或卡片撑高后出现。

**定位方法（先量化再定性）**：掩膜-polyline 的 KDTree 距离中位 **0px**（数据层完美
重合），说明偏差只在显示层——排除识别问题。

**根因**：canvas 是 `drawPreview` 时刻的静态绝对定位；布局变化后 `<img>` 重排而
canvas 不动。

**修复**：绘制逻辑抽成可重入的 `renderPreviewOverlay`，挂 resize/ResizeObserver
去抖重绘，并处理缓存图 `complete` 时补画。

### 24.16 OCR 文本有效性过滤层（可插拔，2026-11-21）

**背景**：OCR 会把大量非地名文本（UI 统计栏、数值、状态栏、乱码碎片、泛词）
读成独立文本块。这些若占用 `GEOCODE_TARGET_LIMIT` 名额或混进 RANSAC，
轻则降低精度，重则让整张图定不出尺度（行者宁波长截图 0 轨迹点）。

**分层设计**（见 `app/anchor_filter.py` 模块 docstring）：
① 规则层（默认，零成本）→ ② 地理编码反馈 → ③ LLM 语义（可选）→ ④ RANSAC 几何裁决。

**接口与扩展**：
- 协议 `AnchorFilter.reject(text) -> bool`（True=丢弃）。
- 默认实现 `RuleFilter`；`register_filter(name, factory)` 注册新实现；
  `get_filter(name)` 工厂，未知名字/构造失败回落默认，绝不抛异常。
- 切换优先级：环境变量 `ANCHOR_FILTER` > `AppConfig.anchor_filter` > `"rule"`。
  例如接 LLM 过滤器：实现协议 → 注册为 `"llm"` → `.env` 设 `ANCHOR_FILTER=llm`。
- 流水线在 `__init__` 取一次实例（`self.anchor_filter`）复用，`build_anchors`
  从 `cfg.anchor_filter` 读取；`ocr_anchor.is_ui_noise` 等旧函数保留为兼容别名。

> 🔴 **设计约束**：过滤层「宁可漏放，不可误杀」。被误杀的真实地名直接减少
> 配准锚点（实测挡掉「迪士尼/东柳」会让整图定不出尺度）。默认实现只拦强特征
> 噪声，模糊情况（如残片还原 `杭甬高→杭甬高速`）留给可选 LLM，且最终由几何
> 一致性裁决。改动后需同步两个便携技能目录（`python src/tools/sync_skill.py <dir>`）。

### 24.17 长截图版式两处识别 bug（2026-11-21）

用户新传 631×1830 长截图（上半地图 + 下半详情面板），里程与绝对位置明显错。
两个独立根因，均已修复并加回归测试。

**① 速度文本被当里程（里程短 29%）**

- 现象：截图顶栏/详情面板均为 34.63 公里，系统 OCR 先验却是 25.15，
  `scale_check` 拿几何 24.4km 对比后判「一致」（ratio 0.971），尺度校准不触发。
- 根因：`input_quality._DISTANCE_RE` 只匹配到 `15.67km` 就停，详情面板的
  `15.67km/h`、`13.95km/h`、`39.53km/h` 全被当成里程，与真值取中位数
  `(15.67+34.63)/2 = 25.15`。
- 修：正则加负向前瞻排除 `单位 + /h|/小时|/s|/秒|/min`，并在
  `extract_distance_km` 里整块跳过含速度单位的 OCR 文本（双保险）。
  结果：先验回到 34.63，几何里程 37.0km，ratio 1.069。

**② 起终点误检到详情面板/指北针**

- 现象：闭环模式把起点定在 (318,1117)（详情面板蓝色标签）、终点 (99,309)
  （地图左上角红色指北针），相距 837px，而闭环起终点实测仅相距 39-47px。
- 根因：`_detect_markers_ex` 闭环分支按「面积最大」挑蓝/红团块。长截图里
  UI 元素面积碾压真徽章（面板蓝 2464 vs 真起点 768；指北针 1329 vs 真终点 327）。
- 修：新增 `_pick_adjacent_badge_pair()`——闭环改为枚举蓝×红候选所有配对，
  间距上限取 `2.5 × scale_badge_d`（内容尺度，不用图高比例，否则 1830px 长图
  的 15% = 274px 会放进无关团块），评分 `(面积和)/(1 + gap/gap_hi)`。
  真徽章对 (389,586)/(391,539) 距 47px 胜出；旧 640×1136 样本
  (365,1102)/(334,1079) 距 38.6px 不受影响。

**端到端效果**：里程 24.4→37.0km、点数 940→1407、置信度 0.502→0.553、
起终点拓扑校验通过。（关于「绝对位置偏移 597m」的调查结论见 §24.18——
配准位置其实是对的，属自校验误报，非锚点混源。）

### 24.18 绝对位置自校验误报修复（2026-11-21）

**先纠正 §24.17 遗留的判断**：曾以为 597m 反投影偏移源于「锚点混源」，
建议 `MAP_GEOCODING_ORDER` 单源化。调查后确认**该判断错误**：
`.env` 早已是 `MAP_GEOCODING_ORDER=amap`，参与配准的 6 个锚点也全是
amap 单源（`anchor_sources.mixed=false`）。

**实证结论：配准整体位置是正确的**。逐点反投影诊断（宁波站 405/668/747m、
东柳 347m、董州区政府 1370m、钟公庙村 526m）显示偏移集中在 OCR 错字
地名上；把可靠锚点的真实坐标相对方位与截图像素逐一对照——东柳在宁波站
右侧且同纬度、钟公庙村在其西南、鄞州区政府在东南——**全部吻合**。
597m 是 POI 文字标注中心相对真实点位的固有偏移，不是整体平移。

**真正的 bug：`pipeline._self_check` 的容差自相矛盾**。它用硬编码 400m
判定整体平移，但项目自身：
- `register._KIND_TOLERANCE` 给 POI/station 内点容差下限 200m、
  transit/road 800m；
- HANDOVER 多处记录标注偏移 35-150m、几百米到一公里、残差 439m「正是
  文字标注偏移」。

城市级截图（14.9 m/px）下 400m 仅 27px，把「少数 transit 标注偏移大」
误判成「整体平移」，还把置信度打 0.45 折。

**修法**：
- `_self_check` 改为**逐点**与各自类型容差比较（复用 `_KIND_TOLERANCE`
  同式：`max(m_floor, px_cap × scale)`），超容差内点比例 >0.6 且内点 >=3
  才判整体平移。整体平移错误 → 多数内点同时超容差；个别标注偏移大 →
  仅少数超，不误报。
- 置信度降权改用 `_self_check` 返回的 notes 作为同一判据，去掉独立的
  400m 硬阈值；未判平移但中位偏移仍偏大时温和降 0.9（原 0.8）。

### 24.19 地理编码输入质量：词库去重与高德错字纠错拦截（2026-11-21）

§24.18 调查中记下两处「影响有限、暂不动」的副产物，本轮一并修掉。

**缺陷 1：词库同址多源坐标重复占位**。`Lexicon.lookup()` 的去重用
`round(lon/lat, 6)` 作签名（精度约 0.1m），跨源几十米的定位抖动全部漏过。
实测「宁波站」沉淀 baidu/amap/tianditu 三源，彼此 30-70m，被当成 3 个
独立候选进入配准，挤占内点名额、稀释可靠锚点权重。

修法：改为 **150m 空间聚类**，同址近邻只保留权重最高的一条（通常是
用户确认或反复核验过的源）。阈值远小于同名异地 POI 间距（通常 >1km），
不会误合并真正的同名不同地点。真实词库验证「宁波站」3→1（amap 胜出）。

**缺陷 2：高德 provider 照单全收错字纠错结果**。天地图/Tmap 早有
`_name_matches()` 包含校验，唯独 `AmapProvider.search()` 把返回的 POI
全部接受。OCR 错字「董州区政府」被高德「智能纠错」成「鄞州区政府」，
坐标偏离真值 1370m，作为内点把整个配准模型拉偏。

修法：给高德补同类校验，但**用包含关系而非字符相似度**——这样才能
区分两类纠错：

| 类型 | 例子 | 判据 | 处置 |
|---|---|---|---|
| 有益：残片补全 | 杭甬高→杭甬高速 | `query ⊂ title` | 放行，quality 0.8 |
| 有益：截断还原 | 波站→宁波站 | `query ⊂ title` | 放行，quality 0.8 |
| 有益：带后缀 POI | 宁波站→宁波站-公交站 | 分段包含 | 放行，quality 0.8 |
| 有害：错字替换 | 董州区政府→鄞州区政府 | 无包含关系 | 剔除 |

一刀切要求严格相等会把残片补全全杀掉——那是配准的主力锚点。
注意 `AmapProvider.search()` 内需延迟 `from ...geocode import Lexicon`
（顶层导入会形成包初始化环）。

**端到端效果**（同一张 631×1830 长截图）：

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 置信度 | 0.553 | **0.604** |
| 跨源冲突候选 / 涉及地名 | 7 / 4 | **6 / 3**（「董州区政府」消失） |
| 配准内点 / 不同锚点 | 6 / 4 | **8 / 6** |
| 里程 / 轨迹点 | 37.0km / 1407 | 36.6km / 1407 |

模型从 similarity 换为 north_up、RMSE 757→809m：内点更多、类型容差
不同，不是精度劣化。新增 `tests/test_geocode_dedup.py` 6 项；相关 64 passed，
全量 753 passed；两个便携技能目录已同步。

**验证**：行者宁波误报消失；新增 2 项合成测试锁住「少数大偏移不误报」
「多数内点平移仍报警」；`test_e2e/test_input_quality/test_register_degenerate`
15 passed，全量 753 passed。

> 📌 **调查副产物（未改，影响有限）**：① 词库「宁波站」沉淀了 baidu/amap/
> tianditu 三源坐标（彼此 <100m），会占 3 个内点名额稀释权重；② 高德会把
> OCR 错字「董州区政府」纠错匹配到鄞州区政府坐标。两者暂不动，若后续要
> 提升绝对精度可从这两处入手。

> 📌 **通用教训**：① OCR 里程正则必须排除速度单位后缀，否则详情面板的多列
> 速度会污染中位数；② 闭环徽章定位用**相对几何约束**（起终点相邻距离）比
> 绝对面积稳健——长截图的 UI 元素面积不可控，但「起终点必相邻」是语义硬约束。

---

> **2026-11-19 地名口径备注**：工作日志中该闭环样本的坐标
> （lon 121.49–121.52, lat 31.26–31.32）经复核指向**上海**，
> 与旧日志里写的「宁波」不一致；样本文件名 `xingzhe_ningbo.jpg` 系历史命名。
> 涉及具体里程时以当次运行的 OCR 概要读数为准（日志中出现过 32.17km 与
> 34.67km 两种口径，疑为不同运行/裁切所致）。此处只备注口径差异，
> 不改历史章节的原始记录。

### 24.20 闭环轨迹断裂段桥接：修「识别路线比原图少」（2026-11-21）

**先记一条方法论教训**：用户反馈「紫色轨迹比原图少一截、右上角和左下角
缺失」时，前几轮我改的是置信度、锚点过滤、词库去重、自校验容差——
这些都不改变轨迹几何形状，所以视觉上必然没有变化。用户说「没改进」
时，第一步应当核对改动是否触及他看到的那一层，而不是拿置信度数字当改进。

**真根因在像素提取阶段**。`track._select_loop_component` 用
`max(area)` 只保留**一个**连通域。行者长截图（631×1830）的轨迹掩膜
被 POI 文字、比例尺、UI 图标打断成 31 个连通域：

| 连通域 | 面积 | bbox | gap | 位置 | 原处置 |
|---|---|---|---|---|---|
| #1 主域 | 8097 | (106,258,258,641) | — | 主体 | 保留 |
| #2 | 962 | (368,294,111,119) | 54px | 右上角（东柳附近） | **丢弃** |
| #3 | 721 | (144,666,61,95) | 4px | 左下角 | **丢弃** |
| #4 | 605 | (165,762,37,86) | 91px | 左下角 | **丢弃** |
| #5 | 530 | (72,296,48,30) | 103px | 左上角引线 | **丢弃** |

原图绿线实际覆盖 x=18–614、y=250–908，而提取 bbox 只到 x=364/y=899
——右上角和左下角被整段切掉。

**修法**：新增 `_merge_line_fragments(scale)`，闭环模式下把断裂段逐段
桥接回主域。判据（宁可漏接不可错接）：

1. 面积 ≥ 主域 1.5%（滤抗锯齿小点）；
2. **线状判据用填充率**：真轨迹 fill 0.07–0.20；底图图例/指北针虽有
   长边但 fill 高（旧样本误接段 area575/bbox50×33 → fill 0.35）。
   规则：`fill > 0.25 && !(fill <= 0.42 && ratio >= 3)` 排除；
3. 到主域最近距离 ≤ 4×badge_d（实测最大断口 131px < 157px）；
4. 迭代时记录已并入的 seg_id，避免同一段被重复桥接（早期 bug：
   同一段连桥 12 次）；每轮重算主域，支持链式桥接。

配置 `loop_bridge_fragments`（默认 False，行者画像 True），开环的
美团/高德完全不受影响。

**效果**（同一张长截图）：

| 指标 | 桥接前 | 桥接后 |
|---|---|---|
| 轨迹 bbox 右边界 | x=364 | **x=507** |
| 弧长 | 2812px | **4372px** |
| 桥接段数 | 0 | **5** |
| 里程校准因子 | ×1.34（放大补里程） | **×0.88**（不再硬拉） |
| 交付里程 | 36.61km | **36.12km**（真值 34.63） |

旧样本（640×1136）bbox 与里程（34.89km）完全不变——图例段被 fill
判据正确排除。新增 `xingzhe_ningbo_long.jpg` 回归样本 + 2 项测试
（断裂段接回 / 图例不误接）；行者画像 17 passed。
