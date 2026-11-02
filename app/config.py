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
    # 单次请求超时（秒）
    http_timeout: float = 12.0
    # 路网数据本地缓存目录（避免每次识别都重新拉取）
    road_cache_dir: Path = DATA_DIR / "cache"
    # 路网约束精修：用「轨迹必须走在真实道路上」纠正绝对位置。
    # 锚点来自 POI 标注，标注点位与真实 POI 在不同 POI 数据库之间可差
    # 100-300 米；道路几何没有这个问题，因此用它做最终校正。
    road_fit_enabled: bool = True
    road_fit_max_shift_m: float = 800.0

    @property
    def amap_key(self) -> str | None:
        return os.environ.get(AMAP_KEY_ENV) or None

    @property
    def tmap_key(self) -> str | None:
        return os.environ.get(TMAP_KEY_ENV) or None

    @property
    def baidu_key(self) -> str | None:
        return os.environ.get(BAIDU_KEY_ENV) or None

    @property
    def tianditu_key(self) -> str | None:
        return os.environ.get(TIANDITU_KEY_ENV) or None

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


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 加载，避免引入额外依赖。已存在的环境变量优先。"""
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


CONFIG = AppConfig()
