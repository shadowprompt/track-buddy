---
name: ride-screenshot-georeference
description: 从运动 App 的地图截图还原带经纬度的轨迹，并导出 GeoJSON / GPX / TCX。当用户发来美团骑行、高德运动等 App 的运动记录截图，并希望「识别里面的路线」「导出轨迹」「转成 GPX / GeoJSON」「看看骑了哪几条路」「提取开始时间做运动记录」「这条轨迹准不准」时使用本技能。技能封装图像配准（像素→经纬度）、轨迹提取、路网约束精修、运动概要识别与多格式导出的完整能力，支持多来源画像与多图拼接，也用于排查「识别出的轨迹不在道路上」「坐标整体偏移」这类问题。
agent_created: true
---

# 运动截图轨迹识别与地理配准

## 概述

把一张运动 App 的地图截图还原成带经纬度的轨迹文件，并可选附带运动概要
（距离、时长、卡路里、开始时间）导出为可直接导入 Strava / Garmin / GIS 的
运动文件。

这不是「图像识别」，而是**图像配准**：GPS 坐标不在图片里，要用图上印的 POI
标注当锚点做像素→经纬度映射，再用真实路网约束校正。

本技能提供**两条调用路径**，按需求选择：

| 路径 | 能力 | 前置条件 | 何时用 |
|---|---|---|---|
| **A. 便携 CLI**（`scripts/recognize.py`） | 单图 → GeoJSON + 对照图；自带识别内核，复制到任意机器可跑 | 仅需 Python 依赖 | 快速识别单张截图、离线、无密钥 |
| **B. 本地服务 API**（`scripts/recognize_api.py`） | 多来源、多图拼接、运动概要、GPX/TCX 导出、云记录 | 需先起本地服务（用户自启） | 完整运动记录、多图、要 GPX/TCX |

两条路径共享同一套识别内核与判据。默认优先用路径 A；用户要多图、GPX/TCX、
运动概要或指定来源时走路径 B。

## 何时使用

- 用户发来运动记录截图，要求导出轨迹、GeoJSON、GPX、TCX
- 用户问「我骑的是哪几条路」「这条路线在什么位置」
- 用户提供运动概要截图，希望带上开始时间/时长做成完整运动记录
- 用户反馈「轨迹不在道路上」「坐标偏了」「终点跑到别的街区」
- 截图有多张（沿路线滑动），需要拼接成长轨迹

## 路径 A：便携 CLI 单图识别

```bash
python scripts/recognize.py <图片路径> --city 上海 --out <输出目录>
```

常用参数：
- `--city` 城市提示（默认上海）
- `--out` 输出目录（默认内核目录下的 out）
- `--summary` 运动概要截图（可选）。**轨迹与数据可分离**：只传轨迹图也能
  拿到完整的一等公民产物（GeoJSON + 纯轨迹 GPX）；传了概要且能解析出
  「开始时间+时长」时，GPX 补上时间轴并额外产出 TCX
- `--source` 来源画像 id（`meituan` / `amap` / `xingzhe`，默认 `meituan`；通过
  `SOURCE_PROFILE` 环境变量传给 CLI）
- `--online` 允许在线地理编码（需已配置地理编码密钥，未配置时自动降级
  为本地词典模式）
- `--no-chart` / `--no-road-verify` 跳过轨迹图 / 路网验证图

脚本把 CLI 输出的 JSON 摘要原样打印到 stdout。内核定位顺序：
`TRACK_BUDDY_HOME` 环境变量 → 技能自带 `assets/ride_core/`（默认，便携）→
仓库根目录 → 旧环境变量 `MEITUAN_RIDE_HOME`（兼容）。也可用 `--project` 显式指定。

首次使用需装依赖：

```bash
pip install -r <技能目录>/assets/ride_core/requirements.txt
```

**CLI 路径的产出**：
- `trajectory_wgs84.geojson` — WGS-84 轨迹（唯一交付坐标系）
- `track.gpx` — GPX 1.1。无概要时为无 `<time>` 的纯轨迹（合法且可直接
  导入）；传 `--summary` 且概要含开始时间+时长时为带时间轴版本
- `track.tcx` — 仅传了概要且能解析出时间轴时产出（TCX 规范要求逐点时间）
- `overlay_result.png` — 原图 + 识别轨迹叠加（橙线=轨迹，白点=采样点，
  绿=起点，红=终点，青圈=锚点）
- `trajectory_chart.png` — 独立轨迹图（米制等比例，标路名与公里刻度）
- `road_verify.png` — OSM 路网投影回截图 + 白色轨迹（验证绝对位置）

## 路径 B：本地服务 API（完整能力）

前置：本地服务需**由用户自己启动**（默认 `http://127.0.0.1:8848`，
`python run.py --host 127.0.0.1 --port 8848`）。脚本只检测并调用服务，
**不会代为启动服务**。未起服务时给出清晰提示。

```bash
# 单图 + 指定来源 + 运动概要截图
python scripts/recognize_api.py <轨迹图> --source amap \
    --summary <运动概要图> --city 上海 --out <输出目录>

# 多图拼接（沿路线滑动的 2-9 张）
python scripts/recognize_api.py <图1> <图2> <图3> --source meituan --out <输出目录>
```

脚本调用 `POST /api/recognize`（单图）或 `/api/recognize_multi`（多图），
打印 JSON 摘要，并把返回的 GeoJSON / GPX / TCX 写入输出目录。可用值见
`GET /api/health` 的 `source_profile.available`。

**服务层额外产出**（CLI 没有）：
- `payload["ride_summary"]` — 运动概要（距离/时长/卡路里/减碳/开始时间）
- `payload["gpx"]` / `payload["tcx"]` — GPX 1.1 / TCX v2 运动文件文本
  （含按里程比例插值的时间轴；概要缺失时 TCX 因规范缺 `<Time>` 会缺席，
  宁缺勿错）

## 来源画像（Source Profile）

当前注册两个画像，默认 `meituan`：

| id | 标题 | 轨迹特征 | 状态 |
|---|---|---|---|
| `meituan` | 美团骑行 | 浅荧光绿单色线，绿色「起」/红色「终」徽章 | 已精标（默认值即美团参数） |
| `amap` | 高德运动 | 速度渐变线（红→橙→黄→绿）、整屏截图需裁地图区 | 初版接入，参数随真实样本迭代 |

来源选择：请求级 `source` 入参（服务 API）优先 > `SOURCE_PROFILE` 环境变量
（CLI / 直跑 pipeline）> 默认 `meituan`。未知 id 直接报错并列出可用值，
绝不静默回退。模板库按画像隔离，美团模板不会命中其它 App 截图。

**坐标与地理编码硬约束**：内部识别用 GCJ-02，**对外交付文件（GeoJSON /
GPX / TCX）统一 WGS-84**。地理编码只允许天地图单源
（`MAP_GEOCODING_ORDER` / `TIANDITU_API_KEY`）；高德在线配额已尽，
不要建议或尝试接入高德在线编码。只用截图上印着的公开 POI 文案，
不逆向地图瓦片。

## 结果解读：怎么判断这次识别可不可信

摘要关键字段：

| 字段 | 含义 | 判读 |
|---|---|---|
| `confidence` | 综合置信度 | < 0.6 属「识别成功但建议核对」，交付时说明 |
| `diagnostics.anchor_backproj_px` | 锚点真值反投影回像素的偏移中位 | 越小越好；> 100 px 说明绝对位置存疑 |
| `diagnostics.source` | 当前生效画像 | 确认与用户预期来源一致 |
| `diagnostics.model` | 配准模型 | `north_up` 最可靠；`affine` 说明锚点噪声大 |
| `road_fit.median_before_m` → `median_after_m` | 路网校正前后贴路距离 | 明显改善说明系统性偏差已被纠正 |
| `road_fit.shift_e_m` / `shift_n_m` | 路网校正平移量 | 超过 300 m 务必人工核对 |
| `ride_summary.cross_check` | 概要里程与轨迹里程一致性 | 0.6–1.7 为合理区间 |
| `warnings` | 全部告警 | 必须逐条读，含锚点匹配率、精修否决等关键信息 |

**交付前必做**：打开 `overlay_result.png`，确认轨迹贴着 App 自己画的那条
轨迹线。数字漂亮不等于结果对——内部自洽不能证明绝对位置正确。

## 硬性规则（违反就会出错）

1. **绝不重复换算基准**。交付 GeoJSON 已是 WGS-84；画叠加图只能用配准
   矩阵的逆，不可再套一次坐标转换。本项目踩过两次「轨迹明显偏离原图」
   都源于此。
2. **验证绝对位置要用第三方证据**。最硬的判据是「截图自己渲染的路网」：
   把 OSM 路网投影回截图统计命中率，而不是只看内部残差。
3. **单一指标变好 ≠ 位置更准**。平行道路（相距 100–300 m）会骗过「到最近
   道路距离最小」。需正交证据（App 路名标签 ↔ OSM 同名道路中心线距离）。
4. **不要用「横向延展」筛选连通域**。对角线/南北向路线会整条漏掉。判据
   应是拓扑：连通域须同时靠近「起」「终」两个徽章。
5. **路名不能当锚点**：地图把路名沿道路走向排布，文字中心不对应固定地理
   位置（实测偏移中位 2.5 km）。路名只用于区域一致性校验与归属。

## 排查指引

用户说结果不对时，按顺序查（详见 `references/methodology.md`）：

1. 加载的是 WGS-84 还是 GCJ-02？→ 坐标系错位（最常见）
2. `diagnostics.source` 是否与截图来源一致？→ 画像选错
3. 看 `overlay_result.png`，轨迹是否与 App 画的线重合？→ 像素提取问题
4. 看 `road_verify.png`，OSM 路网是否压在 App 渲染路网上？→ 基准/数据源
5. 看 `road_fit.shift_*`，平移量是否异常大？→ 锚点系统性偏差（词典问题）
6. 看 `warnings` 锚点匹配率 → 词典覆盖不足（换城市可能失败）

## 扩展新来源 / 深入开发

新增小米 / Keep / Garmin / Coros 等来源的完整操作手册在**项目仓库**
`docs/HANDOVER.md` §23（Source Profile 扩展手册）：唯一必改代码文件
`src/app/source_profile.py`，含 VisionConfig 标定指南、7 项验收清单、
硬约束与协作约定。识别链路下游零改动复用，瓶颈在标定样本不在代码。
该手册服务于开发场景，不在本便携技能包内。

## 维护约定

代码唯一真源在 `D:\Dev\track-buddy`。改完执行：

```bash
python src/tools/sync_skill.py <各技能目录>
```

把最新 app 内核同步进各份技能安装，不要直接改技能目录里的内核副本。
同一份技能装在三处（内容一致）：用户级 `~/.workbuddy/skills/`、
仓库级 `<仓库>/.workbuddy/skills/`、工作区级。

## 资源

- `scripts/recognize.py` — 便携单图 CLI 封装（自带内核，离线可用）
- `scripts/recognize_api.py` — 本地服务 API 封装（多图/来源/概要/GPX/TCX）
- `assets/ride_core/` — 打包的识别内核（app 包 + 词典 + 依赖清单）
- `references/methodology.md` — 完整方法论、判据设计、踩坑记录，深入排查时读
