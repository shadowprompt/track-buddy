"""全局配置。所有可调参数集中在此，便于调参与回归测试。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
DATA_DIR = ROOT / "data"

# 地图服务配置（合规：仅腾讯/高德/百度/天地图）
# 优先级：环境变量 > .env 文件 > 本地词典（离线兜底）
AMAP_KEY_ENV = "AMAP_KEY"
TMAP_KEY_ENV = "TMAP_KEY"
BAIDU_KEY_ENV = "BAIDU_MAP_KEY"
TIANDITU_KEY_ENV = "TIANDITU_KEY"

# 腾讯地图 WebService 端点
TMAP_PLACE_SEARCH = "https://apis.map.qq.com/ws/place/v1/search"
TMAP_GEOCODER = "https://apis.map.qq.com/ws/geocoder/v1/"
# 高德 WebService 端点
AMAP_PLACE_SEARCH = "https://restapi.amap.com/v3/place/text"

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
    """图像处理参数。"""

    # 起/终徽章检测：绿色「起」、红色「终」
    start_hsv_range: tuple = (35, 90, 90, 85, 255, 255)
    end_hsv_range: tuple = (0, 90, 90, 12, 255, 255)
    end_hsv_range2: tuple = (170, 90, 90, 179, 255, 255)
    # 徽章面积阈值（像素）。虚线碎片面积约 150-230，徽章约 4000+
    badge_min_area: int = 1500
    badge_max_area: int = 20000

    # 轨迹线掩膜：低饱和 + 中等亮度，避免选中彩色虚线与文字
    track_max_value: int = 200
    track_max_saturation: int = 90
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

    # 路网吸附
    snap_radius_m: float = 60.0
    snap_enabled: bool = True

    max_upload_mb: int = 20


@dataclass
class AppConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    # 词典库：本地地名词典，随使用积累
    lexicon_path: Path = DATA_DIR / "lexicon.json"
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
    llm_timeout: float = 25.0
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
    # .env 文件位置（改完不用重启服务就能生效，见 _env_value）
    env_path: Path = ROOT / ".env"
    # .env 解析结果的缓存（按 mtime 失效）
    _env_cache: dict[str, str] = field(
        default_factory=dict, repr=False, compare=False)
    _env_cache_mtime: float = field(default=-1.0, repr=False, compare=False)

    def _file_env_value(self, name: str) -> str | None:
        """读 .env 文件里当前的 key 值（按 mtime 缓存，避免重复读盘）。

        ⚠️ **为什么要实时读文件**：`load_dotenv()` 在模块导入时把 .env
        的内容灌进了 ``os.environ``。若只读 ``os.environ``，用户改了
        .env 文件后**必须重启服务**才能生效——而 key 属性是
        ``@property``，看起来像是「实时读取」，容易误判。

        这里每次都重新读文件（mtime 没变就用缓存），
        改完 .env 立即生效，不用重启。
        """
        try:
            mtime = self.env_path.stat().st_mtime
        except OSError:
            return None
        if mtime != self._env_cache_mtime:
            self._env_cache = _read_env_file(self.env_path)
            self._env_cache_mtime = mtime
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
        return self._env_value(TIANDITU_KEY_ENV)

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
