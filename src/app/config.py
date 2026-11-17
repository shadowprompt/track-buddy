"""全局配置。所有可调参数集中在此，便于调参与回归测试。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

# 目录布局（2026-11 起）：
#
#   <repo>/            仓库根：data/ samples/ docs/ tests/ .env
#     src/             源代码
#       app/           Python 包（识别核心）
#       web/           前端页面与静态资源
#       tools/         实验与诊断脚本
#
# SRC_ROOT = src/，ROOT = 仓库根。
#
# ⚠️ **ROOT 必须是「含 data/ 的那一层」，不能用固定的 parent.parent**：
# 本文件会被复制进技能包，vendor 布局是
# ``<skill>/assets/ride_core/app/config.py`` + ``<core>/data/``，
# 与仓库布局差了两层。固定层数会让 vendor 版去读**技能包外面**的
# data/，直接读不到词典。
# 所以改成向上查找「最近的、含 data/ 目录的祖先」，两种布局都对。
_APP_DIR = Path(__file__).resolve().parent


def _find_root(start: Path) -> Path:
    """向上找最近的、含 ``data/`` 目录的祖先目录。

    仓库布局：``<repo>/src/app/config.py`` → ``<repo>``（有 data/）
    vendor 布局：``<core>/app/config.py`` → ``<core>``（有 data/）

    找不到时退回 ``start.parent.parent``（等价于旧的固定层数）。
    """
    for cand in (start, *start.parents):
        if (cand / "data").is_dir():
            return cand
    return start.parent.parent


ROOT = _find_root(_APP_DIR)
# src/ 根：仓库布局下是 ROOT/src；vendor 布局下 app/ 直接在 ROOT 下。
SRC_ROOT = ROOT / "src" if (ROOT / "src" / "app").is_dir() else ROOT

SAMPLES_DIR = ROOT / "samples"
DATA_DIR = ROOT / "data"

# 地图服务配置（合规：仅腾讯/高德/百度/天地图）
# 优先级：环境变量 > .env 文件 > 本地词典（离线兜底）
AMAP_KEY_ENV = "AMAP_KEY"
TMAP_KEY_ENV = "TMAP_KEY"
BAIDU_KEY_ENV = "BAIDU_MAP_KEY"
# 天地图 tk 的环境变量名（单一，与 AMAP_KEY / TMAP_KEY 保持一致风格）。
TIANDITU_KEY_ENV = "TIANDITU_API_KEY"
TIANDITU_SEARCH_URL = "https://api.tianditu.gov.cn/v2/search"
TIANDITU_MAP_BOUND_ENV = "TIANDITU_MAP_BOUND"
TIANDITU_MAP_BOUND_DEFAULT = "-180,-90,180,90"
# 在线地理编码源的默认顺序（注册表 DEFAULT_ORDER）由 MAP_GEOCODING_ORDER 覆盖。

# 网页底图（前端适配层）配置。
# BASEMAP_PROVIDER 决定前端用哪个底图实现（tmap / amap）；
# 高德 JS SDK 需要**前端可用**的 key 与安全密钥，与 WebService 的
# AMAP_KEY 是两套凭据。
BASEMAP_PROVIDER_ENV = "BASEMAP_PROVIDER"
AMAP_JS_KEY_ENV = "AMAP_JS_KEY"
AMAP_JS_SECURITY_ENV = "AMAP_JS_SECURITY"

# 腾讯地图 WebService 端点
TMAP_PLACE_SEARCH = "https://apis.map.qq.com/ws/place/v1/search"
TMAP_GEOCODER = "https://apis.map.qq.com/ws/geocoder/v1/"
# 高德 WebService 端点
AMAP_PLACE_SEARCH = "https://restapi.amap.com/v3/place/text"

# 运动记录云存储（腾讯 COS + 微信登录）的必需配置。
# 单一用途、无别名；缺任一项时 records_enabled() 自动降级，
# 识别端点保持零回归。密钥只在服务端 .env，绝不下发客户端。
RECORDS_REQUIRED_ENV = (
    "COS_SECRET_ID", "COS_SECRET_KEY", "COS_BUCKET", "COS_REGION",
    "WX_APPID", "WX_APP_SECRET", "RECORDS_TOKEN_SECRET",
)

# 腾讯地图 GL JS 的官方公开示例 key。
#
# 用途只有一个：**渲染网页底图**。它被局限在本地代理内部使用，
# 浏览器永远拿不到它（页面里只有随机生成的代理标记）。
#
# 为什么需要它：GL JS 加载时必须完成鉴权（checkKey），
# 鉴权失败会触发 EventEmitter 的 'error' 事件而无监听器，
# 于是抛出没有任何上下文的 "Unhandled error. (undefined)"——
# 表现为「识别失败」，但实际上识别是成功的（服务端返回 200）。
#
# 用户配置了自己的 TMAP_KEY 时优先用用户的；没配就用这个。
TMAP_PUBLIC_KEY = "OB4BZ-D4W3U-B7VVO-4PJWW-6TKDJ-WPB77"


@dataclass
class VisionConfig:
    """图像处理参数。

    ⚠️ 这套默认值是**美团骑行截图**的标定结果（轨迹浅荧光绿、绿色「起」
    红色「终」徽章、POI 标注偏移 7-28 像素）。它现在归属于输入源画像
    ``SourceProfile("meituan")``（见 ``app/source_profile.py``）；未来
    高德/小米等来源在此处**之外**新建 profile，不要直接改这里的默认值。
    """

    # 起/终徽章检测：绿色「起」、红色「终」
    start_hsv_range: tuple = (35, 90, 90, 85, 255, 255)
    end_hsv_range: tuple = (0, 90, 90, 12, 255, 255)
    end_hsv_range2: tuple = (170, 90, 90, 179, 255, 255)
    # 徽章面积阈值（像素）。虚线碎片面积约 150-230，徽章约 4000+
    badge_min_area: int = 1500
    badge_max_area: int = 20000
    # 徽章最小填充率（连通域面积/包围盒面积）。默认 0 = 不启用（历史行为）。
    # 给「徽章与轨迹同色」的来源用：团块徽章 fill≈0.3-0.4，细长轨迹 fill≈0.06，
    # 用它排除被误当成最大连通域的整条轨迹（行者 2026-11-17 实测）。
    badge_min_fill: float = 0.0

    # 轨迹线掩膜：低饱和 + 中等亮度，避免选中彩色虚线与文字
    track_max_value: int = 200
    track_max_saturation: int = 90
    # 可选的**区间式**轨迹掩膜。两种形态：
    #   单色带 (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi)；
    #   多色带（6 元组的元组，各带取并集）——服务于「速度渐变线」
    #   （高德运动：红→橙→黄→绿，一条轨迹横跨多个色相带）。
    # OpenCV HSV 取值域（H 0-179 / S 0-255 / V 0-255）。为 None 时走上面的
    # 上限式掩膜（美团浅荧光绿：低饱和高亮）。两种模式互斥：区间式优先。
    # 字段归属画像参数，新增来源在 SourceProfile 里覆盖。
    track_hsv_range: tuple | None = None

    # 可选的**地图区垂直裁剪** (top_frac, bottom_frac)：保留整图高度的
    # 该行区间。高德等 App 的地图截图是整屏截图，下方 UI 卡片与轨迹
    # 同色系（速度图例条/红心/内嵌地图缩略图），会在拼接前逐图裁掉，
    # 见 ``cv_utils.crop_map_region``。None = 不裁（美团默认，零变化）。
    map_crop_v: tuple | None = None
    # 闭运算核：跨过道路描边造成的断点
    track_close_kernel: int = 11
    # 开运算核：细长笔画（文字笔画宽约 2-3px）被滤掉，轨迹线宽约 8-10px
    track_open_kernel: int = 5

    # OCR
    ocr_min_score: float = 0.55

    # 配准
    # 锚点可信度阈值：POI 图标锚点 > 独立地标 > 路名文字
    anchor_icon_weight: float = 1.0
    anchor_place_weight: float = 0.85
    anchor_road_weight: float = 0.5
    # RANSAC 内点阈值（米）。
    # 实测：美团地图的 POI 文字标注中心与其真实点位存在 7-28 像素
    # （约 35-150 米）的固有偏移，这是标注排版决定的，无法靠算法消除。
    # 因此阈值必须放宽到能容纳该偏移，否则正确锚点会被误判为离群点。
    # 用高德/腾讯地理编码（与美团底图同源）时可收紧到 60-80 米。
    ransac_inlier_m: float = 200.0
    ransac_min_inliers: int = 3

    # 终点折返段检测：终点徽章+定位点会遮挡其下方「跑两遍」的折返末段，
    # 使其碎成独立连通域而被主轨迹提取丢弃。闭环赛道（loop_course=True）
    # 时该碎片会被**桥接进主轨迹**；关闭闭环时仅检测并在 notes 报告
    # （开环骨架上强行接入会乱线/里程虚高，见 HANDOVER §24.8）。None = 关闭。
    #   结构：(radius_px, min_area_px)
    endpoint_fragment_detect: tuple | None = None

    # 闭环赛道：起点与终点为同一位置（行者等 App 的环线骑跑常见形态）。
    # 启用后 extract_track 走专用管线：
    #   1. 连通域按「终点锚 + 最大」选取（双徽章相近时拓扑选择会误选
    #      徽章旁小碎片）；
    #   2. 终点邻域碎片桥接进主轨迹（被徽章遮挡的进出段）；
    #   3. 骨架做 2px 带压薄（Zhang-Suen 对 45° 斜线留双列锯齿带，产生
    #      数百个伪分叉，见 HANDOVER §24.9）+ 分量焊接 + 断口桥接；
    #   4. 排序用「中国邮路」闭环追踪：奇度节点最短路配对（自动把跑两遍
    #      的往返段计两遍）+ Hierholzer 欧拉回路，闭合无跳步。
    # 默认 False：开环来源（美团/高德）行为完全不变。
    loop_course: bool = False

    # 尺度标尺覆盖（可选）：显式指定 badge 直径（px），不走「起终点徽章
    # 面积反推」。行者的内容尺度（close_kernel=7 等）是对着罗盘箭头
    # （area 1205 → 直径 39.2）标定的；起点改为蓝色定位圆点（775 → 31.4）
    # 后若沿用自动反推，核会缩水、掩膜断口激增（实测主连通域 13970→9690）。
    # None = 沿用自动反推（美团/高德不受影响）。
    scale_badge_d: float | None = None

    # 路网吸附
    snap_radius_m: float = 60.0
    snap_enabled: bool = True

    max_upload_mb: int = 20


@dataclass
class AppConfig:
    # ⚠️ ``vision`` 不再是字段：视觉参数跟随输入源画像（请求级入参或
    # ``SOURCE_PROFILE`` 环境变量），见下方 ``vision`` property。把画像
    # 参数写成字段会导致「切换画像后视觉参数不跟着变」的两套真相。
    # 词典库：本地地名词典，随使用积累
    lexicon_path: Path = DATA_DIR / "lexicon.json"
    # 轨迹模板库：成功识别的「像素折线 ↔ 地理折线」，含真实位置，不入库
    template_path: Path = DATA_DIR / "route_templates.json"
    feedback_path: Path = DATA_DIR / "feedback.json"
    # 是否启用在线地理编码（无 key 时自动降级为纯本地词典模式）
    allow_online: bool = True
    # --- 可选的大模型（OpenAI 兼容协议）---
    # 不配置就完全跳过 LLM 相关逻辑，现有行为一字不变。
    # 职责边界见 `app/llm.py`：大模型只做**语义层**
    # （路名序列合理性、缩放级别复核），坐标层仍由 OSM/地理编码负责。
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # 推理型多模态模型的 7 图拼接实测约 59 秒；与 llm.py 的默认值保持一致。
    llm_timeout: float = 600.0
    llm_enabled: bool = True
    # 单次请求超时（秒）
    http_timeout: float = 12.0
    # 路网数据本地缓存目录（避免每次识别都重新拉取）
    road_cache_dir: Path = DATA_DIR / "cache"
    # 路网约束精修：用「轨迹必须走在真实道路上」纠正绝对位置。
    # 锚点来自 POI 标注，标注点位与真实 POI 在不同 POI 数据库之间可差
    # 100-300 米；道路几何没有这个问题，因此用它做最终校正。
    road_fit_enabled: bool = True
    road_fit_max_shift_m: float = 800.0
    # 允许的平移量还受「轨迹自身空间尺度」约束：位移超过轨迹总长度的
    # 这个比例，就不是「让轨迹贴上它本来的路」，而是把它整体搬到了
    # 另一条平行路上（路网稀疏时贴路距离一样会变小）。无量纲系数，
    # 与截图分辨率、城市、样本都无关。实际上限取
    # min(road_fit_max_shift_m, road_fit_shift_span_ratio × 轨迹长度)。
    road_fit_shift_span_ratio: float = 0.15
    # 路网精修的模式分界（米/像素，**物理单位**，不是针对某张图调的像素数）：
    # 低于它时贴路平移的像素精度足够，可做「整体平移纠正」；高于它只做
    # 校验与路名归属——城市级概览图里主干道稀疏，整体平移几公里照样能
    # 压在另一条平行主干道上。部署方可按底图精度调整。
    road_fit_low_zoom_m_per_px: float = 8.0
    # 当可靠锚点已经自洽时，不允许路网优化把轨迹整体搬走。
    # 这是通用的「证据优先级」护栏：路网只能修正明显的系统性偏差，
    # 不能用更低的 OSM 贴路距离覆盖已经较好的截图锚点。
    road_fit_anchor_guard_m: float = 100.0
    road_fit_anchor_worsen_m: float = 80.0
    road_fit_anchor_worsen_ratio: float = 2.5
    # 位移搜索时用来估计「中位贴路距离」的轨迹点上限。
    # 12 图拼接画布的轨迹有约 1.1 万个像素点，粗搜 4225 个位移候选
    # 就是 4700 万次 KD 树查询（实测一次同步精修 264 秒）。轨迹点本身
    # 已是约 6 米等距采样，等距抽到 2000 点不会改变中位距离的统计量，
    # 但能把查询量降一个数量级。设 0 表示关闭降采样。
    road_fit_eval_budget: int = 2000
    # .env 文件位置（改完不用重启服务就能生效，见 _env_value）
    env_path: Path = ROOT / ".env"
    # .env 解析结果的缓存（按内容指纹失效）
    _env_cache: dict[str, str] = field(
        default_factory=dict, repr=False, compare=False)
    _env_cache_fingerprint: tuple = field(
        default=(), repr=False, compare=False)

    def _file_env_value(self, name: str) -> str | None:
        """读 .env 文件里当前的 key 值（按 mtime 缓存，避免重复读盘）。

        ⚠️ **为什么要实时读文件**：`load_dotenv()` 在模块导入时把 .env
        的内容灌进了 ``os.environ``。若只读 ``os.environ``，用户改了
        .env 文件后**必须重启服务**才能生效——而 key 属性是
        ``@property``，看起来像是「实时读取」，容易误判。

        这里每次都重新读文件（指纹没变就用缓存），
        改完 .env 立即生效，不用重启。

        ⚠️ **为什么不能只看 mtime**：文件系统的 mtime 精度取决于
        时间刻度，两次写入**可能落在同一刻度内**——实测同一秒内的
        「复制 + 覆写」有 3/8 概率 mtime 完全相同。
        只看 mtime 会把「文件已改」误判成「没改」，
        用户改了 .env 却读不到新值，而且**不报任何错**。
        所以指纹用 (mtime_ns, size, 内容 sha256) 三者组合：
        mtime_ns 抓「同一刻度内的两次写入」，sha256 兜底。
        """
        try:
            st = self.env_path.stat()
            raw = self.env_path.read_bytes()
        except OSError:
            return None
        fingerprint = (
            st.st_mtime_ns,
            st.st_size,
            hashlib.sha256(raw).hexdigest(),
        )
        if fingerprint != self._env_cache_fingerprint:
            self._env_cache = _read_env_file(self.env_path)
            self._env_cache_fingerprint = fingerprint
        val = self._env_cache.get(name, "").strip()
        return val or None

    def _env_value(self, name: str) -> str | None:
        """取配置值。

        优先级：**显式环境变量 > .env 文件当前内容 > 其他环境变量**。

        「显式环境变量」= 用户自己 export / 用系统环境变量设置的，
        即 ``os.environ[name]`` 与 load_dotenv 灌进去的值不同。
        """
        file_val = self._file_env_value(name)
        env_val = os.environ.get(name, "").strip() or None
        seeded = _FILE_SEEDED_VALUES.get(name)
        if env_val and seeded is not None and env_val == seeded:
            # 还是 load_dotenv 灌进去的那个值（用户没改过）
            # -> 以 .env 文件**当前内容**为准（改文件立即生效）
            return file_val or env_val
        if env_val:
            return env_val
        return file_val

    @property
    def pose_graph_enabled(self) -> bool:
        """多图拼接是否启用位姿图全局平差（默认关闭）。

        ``POSE_GRAPH_ENABLED=1`` 时把链式累积漂移按全部重叠边分摊，
        只改拼接画布内各图的平移，不改地理配准结果。
        """
        raw = (self._env_value("POSE_GRAPH_ENABLED") or "").strip().lower()
        return raw in ("1", "true", "yes", "on")

    @property
    def source_profile_id(self) -> str:
        """当前输入源画像 id。

        优先级：**请求级显式指定**（``/api/recognize`` 等端点的
        ``source`` 表单入参，经 ``app.source_profile`` 的 ContextVar
        生效）> ``SOURCE_PROFILE`` 环境变量 > 默认 ``meituan``。
        遵守「同一功能只用一个环境变量」的约定：环境级来源选择只有
        ``SOURCE_PROFILE`` 这一个开关；请求级由入参控制。
        """
        from .source_profile import current_active_profile

        act = current_active_profile()
        if act is not None:
            return act.id
        raw = (self._env_value("SOURCE_PROFILE") or "").strip().lower()
        return raw or "meituan"

    @property
    def source(self) -> "SourceProfile":
        """当前输入源画像（含视觉参数与里程词表）；未知 id 直接报错。

        请求级入参 > 环境变量，优先级见 ``source_profile_id``。
        """
        from .source_profile import current_active_profile, resolve_source_profile

        act = current_active_profile()
        if act is not None:
            return act
        return resolve_source_profile(self.source_profile_id)

    @property
    def vision(self) -> VisionConfig:
        """当前生效的视觉提取参数（跟随输入源画像，调用时刻读取）。

        美团画像的参数值与 2026-11-14 模块化之前的字段默认值完全一致；
        切换画像（请求入参 / 环境变量）后这里自动换成对应画像的参数。
        """
        return self.source.vision

    @property
    def map_match_enabled(self) -> bool:
        """是否输出 HMM 地图匹配的**备选显示层**（默认关闭）。

        开启后不改变主 GeoJSON / 置信度 / 里程，只在结果里多一份
        ``map_matched``。低信息量直线输入仍由护栏强制跳过。
        配置 ``MAP_MATCH_ENABLED=1`` 即可，无需重启（.env 热读）。
        """
        raw = (self._env_value("MAP_MATCH_ENABLED") or "").strip().lower()
        return raw in ("1", "true", "yes", "on")

    @property
    def confidence_calibration_path(self) -> Path:
        """置信度校准模型文件（由真值样本离线拟合，默认不存在）。

        ``RIDE_CONFIDENCE_CALIBRATION`` 可改路径；文件缺失时
        ``app.calibration.apply_calibration()`` 会原样返回原置信度。
        与词典同目录，便于部署方把「识别产物数据」集中管理。
        """
        raw = (self._env_value("RIDE_CONFIDENCE_CALIBRATION") or "").strip()
        return Path(raw) if raw else self.lexicon_path.parent / "confidence_calibration.json"

    @property
    def amap_key(self) -> str | None:
        return self._env_value(AMAP_KEY_ENV)

    @property
    def tmap_key(self) -> str | None:
        return self._env_value(TMAP_KEY_ENV)

    @property
    def baidu_key(self) -> str | None:
        return self._env_value(BAIDU_KEY_ENV)

    @property
    def tianditu_key(self) -> str | None:
        """天地图 tk（环境变量 ``TIANDITU_API_KEY``）。"""
        return self._env_value(TIANDITU_KEY_ENV)

    @property
    def tianditu_map_bound_custom(self) -> str | None:
        """用户显式配置的天地图搜索视野；未配置或格式非法时返回 None。"""
        raw = (self._env_value(TIANDITU_MAP_BOUND_ENV) or "").strip()
        if not raw:
            return None
        parts = raw.split(",")
        if len(parts) != 4:
            return None
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            return None
        if not (vals[0] < vals[2] and vals[1] < vals[3]):
            return None
        return raw

    @property
    def tianditu_map_bound(self) -> str:
        """天地图搜索视野 minx,miny,maxx,maxy（默认全球）。

        只校验格式，不内置任何城市坐标；业务区域可由部署方配置，
        或由 geocoder 从词典里该城市的已核验坐标自动派生。
        """
        return self.tianditu_map_bound_custom or TIANDITU_MAP_BOUND_DEFAULT

    @property
    def has_online_key(self) -> bool:
        return bool(
            self.amap_key or self.tmap_key or self.baidu_key or self.tianditu_key
        )

    @property
    def tmap_gl_key(self) -> str:
        """网页底图用的腾讯地图 GL JS key。

        优先用用户自己配置的 ``TMAP_KEY``；没配则回退到官方公开示例 key。

        这个 key **只用于渲染底图**，且只存在于本地代理内部——
        浏览器拿不到它（页面里只有随机生成的代理标记）。
        它与地理编码用的 ``TMAP_KEY`` 是两回事：
        地理编码仍需用户自己申请，否则没有配额。
        """
        return self.tmap_key or TMAP_PUBLIC_KEY

    # ---------- 网页底图（前端适配层） ----------

    @property
    def basemap_provider(self) -> str:
        """前端底图适配层使用的服务商名。

        前端 ``web/js/basemap.js`` 注册了 ``tmap`` 与 ``amap`` 两个实现，
        这里只决定**默认用哪个**。改 ``.env`` 里的
        ``BASEMAP_PROVIDER`` 即可切换，业务代码零改动。

        高德 JS SDK 需要**前端可用的 JS key 与安全密钥**，
        没有配 ``AMAP_JS_KEY`` / ``AMAP_JS_SECURITY`` 时即使选了 amap
        也加载不出来——此时自动回退到 tmap，避免页面直接白屏。
        """
        raw = (self._env_value(BASEMAP_PROVIDER_ENV) or "").strip().lower()
        if raw in ("amap", "a_map", "gaode"):
            if not self.amap_js_key:
                return "tmap"
            return "amap"
        return raw if raw in ("tmap", "qq", "tencent") else "tmap"

    @property
    def amap_js_key(self) -> str | None:
        """高德地图 **JS API** 的 key（与 WebService 的 ``AMAP_KEY`` 不同）。"""
        return self._env_value(AMAP_JS_KEY_ENV)

    @property
    def amap_js_security(self) -> str | None:
        """高德 JS API 的安全密钥（``securityJsCode``）。"""
        return self._env_value(AMAP_JS_SECURITY_ENV)

    # --- 运动记录云存储（腾讯 COS / 微信登录）--------------------------
    # 与地图 key 同一套读取范式：@property + _env_value（改 .env 立即生效）。
    @property
    def cos_secret_id(self) -> str | None:
        return self._env_value("COS_SECRET_ID")

    @property
    def cos_secret_key(self) -> str | None:
        return self._env_value("COS_SECRET_KEY")

    @property
    def cos_bucket(self) -> str | None:
        return self._env_value("COS_BUCKET")

    @property
    def cos_region(self) -> str | None:
        return self._env_value("COS_REGION")

    @property
    def wx_appid(self) -> str | None:
        return self._env_value("WX_APPID")

    @property
    def wx_app_secret(self) -> str | None:
        return self._env_value("WX_APP_SECRET")

    @property
    def records_token_secret(self) -> str | None:
        """HMAC 签名密钥，同时用于 openid → uid 派生，**必须稳定不轮换**。"""
        return self._env_value("RECORDS_TOKEN_SECRET")

    @property
    def cos_http_timeout(self) -> float:
        """COS SDK 单次请求超时（秒），默认 15，上限 120。

        SDK 默认会重试且底层超时很长；云存储抖动时不能让识别响应一直
        挂在保存记录上。单一用途变量 ``COS_HTTP_TIMEOUT``，无别名。
        """
        try:
            raw = (self._env_value("COS_HTTP_TIMEOUT") or "").strip() or 15
            value = float(raw)
        except (TypeError, ValueError):
            return 15.0
        return max(3.0, min(value, 120.0))

    @property
    def cos_prefix(self) -> str:
        """对象根前缀，便于 prod/dev 共用一个桶；末尾恒带一个 ``/``。"""
        raw = (self._env_value("COS_PREFIX") or "").strip().strip("/")
        return f"{raw}/" if raw else "ride-records/"

    @property
    def records_token_ttl(self) -> int:
        """token 有效期（秒），默认 30 天，下限 60 秒。"""
        try:
            ttl = int((self._env_value("RECORDS_TOKEN_TTL") or "").strip()
                      or 2592000)
        except ValueError:
            return 2592000
        return max(60, ttl)

    @property
    def records_config_ready(self) -> bool:
        """7 项必需配置是否齐全（SDK 是否可导入由 records 包另行判断）。"""
        return all(self._env_value(name) for name in RECORDS_REQUIRED_ENV)


def _read_env_file(env_path: Path) -> dict[str, str]:
    """读取 .env 文件内容（纯文本解析，忽略注释与空行）。"""
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


# .env 里出现过的 key
_FILE_ENV_KEYS: set[str] = set()
# load_dotenv **实际写入** os.environ 的值。
# 用它区分「这个环境变量是用户显式设置的」还是「只是 .env 灌进去的」：
#   os.environ[key] == _FILE_SEEDED_VALUES[key] -> 还是文件灌的那个值
#                                            -> 以文件当前内容为准（改文件立即生效）
#   os.environ[key] != _FILE_SEEDED_VALUES[key] -> 用户显式改过 -> 用它的值
_FILE_SEEDED_VALUES: dict[str, str] = {}


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 加载，避免引入额外依赖。已存在的环境变量优先。"""
    env_path = path or (ROOT / ".env")
    values = _read_env_file(env_path)
    for key, val in values.items():
        if key not in os.environ:
            os.environ[key] = val
            _FILE_SEEDED_VALUES[key] = val
    _FILE_ENV_KEYS.update(values.keys())


# ⚠️ **必须在模块导入时加载**（而不是只在 `app/server.py` 里调用）。
#
# 实测踩过：`load_dotenv()` 原本只在 server.py 里被调用，
# 于是**任何不经过 server 的路径都读不到 `.env`**——
# CLI（`python -m app.cli`）、测试脚本、直接 import app.pipeline 的工具，
# 全部拿不到 AMAP_KEY，看起来就像「key 没配」。
#
# 放在这里最稳：`AppConfig` 的 key 属性是 `@property`（每次访问都重读
# `os.environ`），所以加载顺序不影响取值。
load_dotenv()

CONFIG = AppConfig()
