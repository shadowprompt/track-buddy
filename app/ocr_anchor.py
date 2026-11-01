"""OCR 与地理锚点提取。

锚点质量直接决定最终精度，因此这里做三层分级：

  icon   —— POI 图标锚点。图标锚点就在真实位置，误差仅 2-3 像素（最优）
  place  —— 独立地标文字（上海动物园、西郊宾馆…）。文字中心即标注点
  road   —— 道路名文字。文字中心与真实路口可能相差上百米，仅作校验

关键点：POI 图标在文字左侧一个小方块。把它定位出来，就能拿到比文字
中心精确得多的锚点像素位置。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import VisionConfig
from .cv_utils import disk_kernel

# 道路/水系类后缀。这些标注的名称沿要素走向排布，文字中心不对应
# 任何固定地理位置——实测反投影偏移中位 458px（≈2.5 公里），最大 1637px。
# 因此它们**不能作为配准锚点**，只用于区域一致性校验。
ROAD_SUFFIXES = (
    "路", "街", "大道", "公路", "高架路", "高架", "隧道", "桥", "弄", "巷",
    "环线", "辅路", "匝道", "立交", "广场", "公园", "中心", "河", "港", "江",
    "湖", "溪", "浜", "塘", "桥", "门", "苑", "园",
)
# 交通设施（地铁站等），图标锚点最可靠
TRANSIT_SUFFIXES = ("号线", "地铁", "火车站", "高铁站", "机场", "客运站", "码头")

# 这些是数字/符号类，不是地名
_NON_NAME = re.compile(r"^[\W\d_]+$")


@dataclass
class TextItem:
    """一个 OCR 文本块。"""

    text: str
    score: float
    cx: float
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float
    # 文本框宽高，用于判断是否竖排
    width: float
    height: float

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width * 1.5

    @property
    def is_rotated(self) -> bool:
        """非水平排版（沿道路方向旋转）。"""
        return self.height > self.width * 1.15

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Anchor:
    """一个候选地理锚点。"""

    name: str
    px: float  # 锚点像素 X
    py: float  # 锚点像素 Y
    kind: str  # icon / place / road
    weight: float
    candidates: list = field(default_factory=list)  # 候选经纬度
    matched: tuple[float, float] | None = None  # 最终选定的经纬度
    icon_box: tuple[int, int, int, int] | None = None

    @property
    def pixel(self) -> tuple[float, float]:
        return (self.px, self.py)


class OcrEngine:
    """RapidOCR 封装，单例复用模型。"""

    def __init__(self) -> None:
        self._engine = None

    def _get(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def run(self, img: np.ndarray) -> list[TextItem]:
        engine = self._get()
        res, _ = engine(img)
        items: list[TextItem] = []
        for entry in res or []:
            try:
                box, text = entry[0], entry[1]
                score = entry[2] if len(entry) > 2 else 0.0
            except (IndexError, TypeError):
                continue
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = 0.0
            text = (text or "").strip()
            if not text:
                continue
            pts = np.asarray(box, dtype=float)
            if pts.size < 8:
                continue
            x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            items.append(
                TextItem(
                    text=text,
                    score=float(score),
                    cx=float(pts[:, 0].mean()),
                    cy=float(pts[:, 1].mean()),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    width=x1 - x0,
                    height=y1 - y0,
                )
            )
        return items


_OCR = OcrEngine()


def ocr_texts(img: np.ndarray, cfg: VisionConfig) -> list[TextItem]:
    """执行 OCR 并过滤低质量结果。"""
    items = _OCR.run(img)
    return [it for it in items if it.score >= cfg.ocr_min_score]


def classify_text(text: str) -> str:
    """判断文本类型。

    返回值：
      "poi"    —— 独立 POI（公园、宾馆、广场、车站…）。可作配准锚点
      "road"   —— 道路/水系/通道名。**不可作锚点**，只作区域校验
      "transit"—— 地铁线路等交通线。可作弱锚点
      "other"  —— 无法判断
    """
    if any(s in text for s in TRANSIT_SUFFIXES):
        return "transit"
    # 竖排文字几乎必然是沿道路/河流走向排布的道路名
    if text.endswith(ROAD_SUFFIXES):
        return "road"
    return "poi"


def find_icon_anchor(
    img: np.ndarray,
    item: TextItem,
    search_radius: float = 34,
) -> tuple[float, float, tuple[int, int, int, int]] | None:
    """在文字左侧寻找 POI 图标，返回图标中心。

    美团地图的 POI 标注形如  [图标][文字]，图标是 12-30px 的彩色方块，
    紧贴文字左缘。找到它就能拿到比文字中心精确得多的锚点。

    注意事项：
    - 搜索半径限制在 34px。放宽会把道路编号牌（S20/G2）误当成 POI 图标。
    - 必须与文字左缘有明确间隔（gap <= 12px），否则会命中文字自身笔画。
    """
    h, w = img.shape[:2]
    left = int(max(0, item.x0 - search_radius))
    right = int(max(0, item.x0 + 2))
    top = int(max(0, item.cy - item.height * 0.75))
    bottom = int(min(h, item.cy + item.height * 0.75))
    if right - left < 4 or bottom - top < 4:
        return None

    region = img[top:bottom, left:right]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # 图标是饱和色块。阈值 40 足以覆盖灰色系 POI 图标，
    # 同时排除地图底色（低饱和）。
    mask = ((sat > 40) & (val > 60)).astype(np.uint8)
    if mask.sum() < 12:
        return None

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best_area, best = 0, -1
    for i in range(1, count):
        area = int(stats[i][4])
        bx, by, bw, bh = stats[i][0], stats[i][1], stats[i][2], stats[i][3]
        # 图标尺寸约束：接近方形、边长 8-30px
        if not (8 <= bw <= 30 and 8 <= bh <= 30):
            continue
        if bw / max(bh, 1) > 2.0 or bh / max(bw, 1) > 2.0:
            continue
        # 必须紧邻文字左缘（允许少量间隙）
        gap = item.x0 - (left + bx + bw)
        if gap < -2 or gap > 14:
            continue
        # 垂直方向需与文字中心大致对齐
        icon_cy = top + by + bh / 2.0
        if abs(icon_cy - item.cy) > max(item.height, 14) * 0.9:
            continue
        if area > best_area:
            best_area, best = area, i
    if best < 0:
        return None

    cx = left + float(centroids[best][0])
    cy = top + float(centroids[best][1])
    bx, by, bw, bh = stats[best][:4]
    box = (left + int(bx), top + int(by), int(bw), int(bh))
    return (cx, cy, box)


def build_anchors(
    img: np.ndarray, items: list[TextItem], cfg: VisionConfig
) -> list[Anchor]:
    """从 OCR 结果构建锚点候选。

    重要：只有「独立 POI」类锚点可以参与地理配准。

    实测（同一张截图，13 个 POI 锚点 vs 11 个道路锚点）：
      - POI 类锚点反投影偏移中位 32px（约 170 米），13/14 在 45px 内
      - 道路类锚点反投影偏移中位 458px（约 2.5 公里），最大 1637px

    原因是地图把路名沿道路/河流走向排布，文字中心不对应任何固定地理位置。
    早期版本把道路名也当锚点，直接让配准整体偏移 3 公里。
    """
    anchors: list[Anchor] = []
    for it in items:
        text = it.text
        if _NON_NAME.match(text):
            continue
        # 过滤状态栏、比例尺等非地图内容
        if text in {"美团", "美团地图"}:
            continue
        if re.fullmatch(r"[\d:.]+", text):
            continue

        cat = classify_text(text)

        # 竖排文字几乎必然是沿要素走向的道路名，直接排除
        if it.is_vertical and cat != "transit":
            continue

        if cat == "road":
            # 道路名不参与配准，但仍记录为「区域校验」用
            anchors.append(
                Anchor(name=text, px=it.cx, py=it.cy, kind="road",
                       weight=cfg.anchor_road_weight)
            )
            continue

        if re.fullmatch(r"[GS]\d+", text):
            # 高速编号（G2/S20）——路网设施，弱锚点
            anchors.append(
                Anchor(name=text, px=it.cx, py=it.cy, kind="poi",
                       weight=cfg.anchor_place_weight * 0.5)
            )
            continue

        icon = find_icon_anchor(img, it)
        if icon is not None:
            px, py, ibox = icon
            anchors.append(
                Anchor(
                    name=text, px=px, py=py, kind="icon",
                    weight=cfg.anchor_icon_weight, icon_box=ibox,
                )
            )
            continue

        anchors.append(
            Anchor(name=text, px=it.cx, py=it.cy, kind="poi",
                   weight=cfg.anchor_place_weight)
        )
    return anchors


def merge_adjacent(items: list[TextItem], gap: float = 14.0) -> list[TextItem]:
    """合并因相邻而粘连的文本框。

    RapidOCR 常把「百联中环购物广场」拆成两块或粘成一块。
    这里把水平方向紧邻且字号相近的框合并，便于整体匹配到 POI 名。
    """
    if not items:
        return []
    horizontals = [it for it in items if not it.is_vertical]
    horizontals.sort(key=lambda t: t.x0)
    merged: list[TextItem] = []
    for it in horizontals:
        if merged:
            prev = merged[-1]
            same_line = abs(prev.cy - it.cy) < max(prev.height, it.height) * 0.6
            close = it.x0 - prev.x1 < gap
            if same_line and close:
                nx0 = min(prev.x0, it.x0)
                ny0 = min(prev.y0, it.y0)
                nx1 = max(prev.x1, it.x1)
                ny1 = max(prev.y1, it.y1)
                merged[-1] = TextItem(
                    text=prev.text + it.text,
                    score=min(prev.score, it.score),
                    cx=(nx0 + nx1) / 2,
                    cy=(ny0 + ny1) / 2,
                    x0=nx0, y0=ny0, x1=nx1, y1=ny1,
                    width=nx1 - nx0, height=ny1 - ny0,
                )
                continue
        merged.append(it)
    verticals = [it for it in items if it.is_vertical]
    return merged + verticals
