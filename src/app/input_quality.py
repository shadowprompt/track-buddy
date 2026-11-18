"""输入质量的**独立校验**：不依赖配准自身的自洽指标。

流水线里多数质量分都来自同一批锚点（残差、贴路距离、置信度），
它们会一起被某个错误基准带偏。本模块用两个与配准无关的外部信号
做交叉验证：

1. **截图上的里程**（OCR 文本里的「X 公里 / X km」）对比还原轨迹的
   几何长度——尺度算错一个量级时，这个比值会立刻暴露；
2. **外部路由引擎**（Valhalla / OSRM，见 pipeline._external_verify）
   独立规划出的路线是否命中用户路名、顺序是否一致——只做轻权
   修正，因为公共实例不稳定，缺失结果绝不能变成惩罚。

另外 ``assess_input_quality()`` 把上传前就能判断的信息（分辨率、
直线度、路名可见性）汇总成可操作的重截建议。
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Iterable

import cv2

_DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(公里|千米|km|KM|Km)")
# 轨迹里程牌（高德轨迹自带的「3KM / 2KM / 1KM」剩余里程标记）：纯整数
# + km 的**无上下文**短标签。真实里程读数总带小数精度或文案上下文
# （「3.68千米」「里程 4.2 公里」「本次 10.5公里」）；把里程牌当总里程
# 读进来，多个标记取中位数会被拖偏——实测 1/2/3 取中位 = 2.0，触发
# 比值 1.826 误报、置信度被无谓压到 0.2。纯标签一律排除，任何来源
# 通用（美团等底图若有同款标记同样受益）。
_MILESTONE_RE = re.compile(r"^\d+\s*(?:km|KM|Km)$")
# 里程上下文词表跟随输入源画像（``CONFIG.source``，唯一来源见
# ``app/source_profile.py``）：美团文案是「骑行 / 已骑 / 封顶」，高德等
# 来源的页面文案不同，各画像自备一套。
_KEYWORD_RE_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]],
                        tuple[re.Pattern[str], re.Pattern[str]]] = {}
_MIN_REASONABLE_KM = 0.1
_MAX_REASONABLE_KM = 300.0
# 还原轨迹长度 / 截图里程 的合理区间。取得较宽：轨迹简化、截图里
# 用起终点连线、OCR 里程含绕路都会带来偏差，只有量级错误才该报警。
_SCALE_RATIO_MIN = 0.6
_SCALE_RATIO_MAX = 1.7

# 外部路由校验的置信度乘子。刻意温和：公共实例超时/无结果时不惩罚。
_ROUTE_BAD_FACTOR = 0.90
_ROUTE_GOOD_FACTOR = 1.03


def _keyword_patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    """当前输入源画像的（干扰词正则, 正面词正则），按词表元组缓存。"""
    from .config import CONFIG  # 延迟 import：config 不依赖本模块，无循环

    src = CONFIG.source
    keywords = (tuple(src.distance_negative), tuple(src.distance_positive))
    compiled = _KEYWORD_RE_CACHE.get(keywords)
    if compiled is None:
        neg_re = re.compile("|".join(map(re.escape, keywords[0])))
        pos_re = re.compile("|".join(map(re.escape, keywords[1])))
        compiled = (neg_re, pos_re)
        _KEYWORD_RE_CACHE[keywords] = compiled
    return compiled


def extract_distance_km(texts: Iterable[Any]) -> float | None:
    """从 OCR 文本块里读出截图显示的骑行里程（公里）。

    兼容 ``TextItem`` 对象与纯字符串。多个命中取中位数，避免
    「3 公里内免费」之类的运营文案盖过真实里程。里程上下文词表
    来自当前输入源画像（``SOURCE_PROFILE``）。
    """
    neg_re, pos_re = _keyword_patterns()
    found: list[tuple[int, float]] = []
    for item in texts or []:
        s = item if isinstance(item, str) else str(getattr(item, "text", "") or "")
        if _MILESTONE_RE.match(s.strip()):
            continue  # 里程牌（3KM/2KM/1KM），不是骑行总里程
        if neg_re.search(s):
            continue
        priority = 1 if pos_re.search(s) else 0
        for num, _unit in _DISTANCE_RE.findall(s):
            try:
                val = float(num)
            except ValueError:
                continue
            if _MIN_REASONABLE_KM <= val <= _MAX_REASONABLE_KM:
                found.append((priority, val))
    if not found:
        return None
    best = max(prio for prio, _ in found)
    return float(statistics.median(v for prio, v in found if prio == best))


# --- 顶栏无单位里程：按列切分 -------------------------------------------------
#
# 行者等 App 的顶栏是「距离 / 速度 / 时间」三列布局，数字行**没有单位**
# （实测宁波样本 OCR：标签「距离」「速度」「时间」三个独立块，数字粘连成
# 「34.67新=江18.29」一个块）——``_DISTANCE_RE`` 要求单位，直接漏掉。
# 硬补「无单位数字也认」会把均速 18.29 当成里程，所以必须**按列归属**：
# 在「距离」标签的正下方裁一条列带，二次 OCR 后取第一个数字。
#
# 粘连块横向跨两列，无法从块内切（TextItem 只有块级 bbox，没有字级坐标），
# 只能回到原图裁剪。列带右边界 = 右侧下一个文本块的左缘（留 6px 余量），
# 覆盖本列数字的完整宽度且不吞右列的数字。
_COLUMN_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_COL_CROP_PAD_L = 10.0
_COL_CROP_PAD_R = 6.0
_COL_CROP_PAD_Y0 = 8.0
_COL_CROP_PAD_Y1 = 60.0
_COL_REOCR_SCALE = 2.0


def extract_distance_column(img: Any, texts: Iterable[Any]) -> float | None:
    """顶栏「距离」标签正下方的无单位数字解析（按列裁切 + 二次 OCR）。

    调用契约：仅在 ``extract_distance_km`` 返回 ``None``（全文无单位数字）
    之后作为兜底调用。找到画像正面词标签（距离/里程…）后：

    1. 找与标签行相邻（y 带重叠）、x 范围与标签列相交的数字块；
    2. 若数字块完全落在本列（右缘 < 右邻块左缘），直接解析块内数字；
    3. 粘连块（跨列）则从原图裁「标签左缘-10 到右邻块左缘-6」的列带，
       2× 放大后二次 OCR，取第一个数字。

    返回公里数；找不到标签 / 无数字 / 数值超出合理区间均返回 ``None``。
    """
    items = [t for t in (texts or []) if hasattr(t, "x0")]
    if img is None or not items:
        return None
    _, pos_re = _keyword_patterns()
    labels = [t for t in items if pos_re.search(str(t.text)) and len(str(t.text)) <= 4]
    if not labels:
        return None
    neg_re, _ = _keyword_patterns()
    for lab in sorted(labels, key=lambda t: t.x0):
        # 右边界：y 带与标签重叠、且在标签右侧的第一个文本块的左缘
        boundary = None
        for t in items:
            if t is lab or t.x0 <= lab.x1:
                continue
            if t.y0 <= lab.y1 + _COL_CROP_PAD_Y1 and t.y1 >= lab.y0:
                boundary = t.x0 if boundary is None else min(boundary, t.x0)
        if boundary is None:
            boundary = lab.x1 + 200.0  # 无右邻：给一列的默认宽度

        # 数字块：与标签 y 带相邻、x 与列带相交、无干扰词、无单位
        cands = []
        for t in items:
            s = str(t.text)
            if neg_re.search(s) or _DISTANCE_RE.search(s):
                continue
            if not _COLUMN_NUMBER_RE.search(s):
                continue
            if t.y1 < lab.y0 - _COL_CROP_PAD_Y0 or t.y0 > lab.y1 + _COL_CROP_PAD_Y1:
                continue
            if t.x1 <= lab.x0 - _COL_CROP_PAD_L or t.x0 >= boundary:
                continue
            cands.append(t)
        if not cands:
            continue

        # 完全在本列内的块：直接解析，省一次 OCR
        tidy = [t for t in cands if t.x1 < boundary - 2.0]
        if tidy:
            val = _first_reasonable_km(str(tidy[0].text))
            if val is not None:
                return val

        # 粘连/跨列块：裁列带二次 OCR
        x1 = max(lab.x0 - _COL_CROP_PAD_L, 0)
        x2 = min(boundary - _COL_CROP_PAD_R, img.shape[1])
        y1 = max(lab.y0 - _COL_CROP_PAD_Y0, 0)
        y2 = min(
            max(t.y1 for t in cands) + 6.0,
            lab.y1 + _COL_CROP_PAD_Y1,
            img.shape[0],
        )
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        big = cv2.resize(
            crop, None, fx=_COL_REOCR_SCALE, fy=_COL_REOCR_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        from .ocr_anchor import _OCR  # 延迟 import：单例复用模型

        for it in _OCR.run(big):
            val = _first_reasonable_km(str(it.text))
            if val is not None:
                return val
    return None


def _first_reasonable_km(s: str) -> float | None:
    """取字符串里第一个落在合理里程区间的数字。"""
    m = _COLUMN_NUMBER_RE.search(s)
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    if _MIN_REASONABLE_KM <= val <= _MAX_REASONABLE_KM:
        return val
    return None


def scale_consistency(
    geom_distance_m: float, ocr_km: float | None
) -> dict[str, Any]:
    """几何里程与截图里程是否量级一致。"""
    base = {
        "available": False,
        "consistent": True,
        "ocr_km": ocr_km,
        "geom_distance_m": round(float(geom_distance_m), 1),
        "ratio": None,
        "note": "截图未识别到里程文字，跳过尺度独立校验",
    }
    if ocr_km is None or ocr_km <= 0 or geom_distance_m <= 0:
        return base
    ratio = float(geom_distance_m) / (ocr_km * 1000.0)
    consistent = _SCALE_RATIO_MIN <= ratio <= _SCALE_RATIO_MAX
    base.update({
        "available": True,
        "consistent": consistent,
        "ratio": round(ratio, 3),
        "note": ("几何里程与截图里程一致" if consistent else
                 "几何里程与截图里程差异过大，配准尺度或轨迹形状可能有问题"),
    })
    return base


def route_validation_factor(road_name_fit: dict[str, Any] | None) -> dict[str, Any]:
    """把外部路由校验结果折算成**温和**的置信度乘子。

    外部引擎不可用是常态（公共实例限流），所以「没有结果」必须返回
    factor=1.0；只有确实算出路线且明显不命中时才轻罚。
    """
    r = road_name_fit or {}
    verify = r.get("external_verify") or r.get("osrm_verify") or {}
    if not verify:
        return {"available": False, "factor": 1.0,
                "note": "外部路由校验未取得结果，不参与置信度"}
    match_ratio = float(verify.get("match_ratio") or 0.0)
    order = float(verify.get("order_consistency") or 0.0)
    hits = len(verify.get("hits") or [])
    if match_ratio < 0.3 or order < 0.5:
        factor = _ROUTE_BAD_FACTOR
        note = "外部路由规划与还原轨迹的路名/顺序不一致，置信度小幅下调"
    elif match_ratio >= 0.6 and order >= 0.8:
        factor = _ROUTE_GOOD_FACTOR
        note = "外部路由规划独立复现了还原轨迹的路名与顺序，置信度小幅上调"
    else:
        factor = 1.0
        note = "外部路由校验结果中性"
    return {
        "available": True,
        "factor": factor,
        "engine": verify.get("engine"),
        "hits": hits,
        "match_ratio": round(match_ratio, 3),
        "order_consistency": round(order, 3),
        "note": note,
    }


def assess_input_quality(
    image_size: tuple[int, int] | None = None,
    straightness: float | None = None,
    bbox_px: tuple[float, float, float, float] | None = None,
    has_road_names: bool = False,
    n_ocr_texts: int | None = None,
) -> dict[str, Any]:
    """上传前/识别初期的输入信息量评估，给出可操作的重截建议。"""
    reasons: list[str] = []
    suggestions: list[str] = []
    if image_size:
        w, h = int(image_size[0]), int(image_size[1])
        if min(w, h) < 600:
            reasons.append(f"截图短边仅 {min(w, h)}px，文字与道路细节不足")
            suggestions.append("用原图（不要压缩/缩略图）重新截图")
    ratio = None
    if bbox_px and len(bbox_px) >= 4 and float(bbox_px[3]) > 0:
        ratio = float(bbox_px[2]) / max(float(bbox_px[3]), 1e-6)
    low_shape = bool(
        straightness is not None and straightness >= 0.80
        and ratio is not None and ratio >= 4.0
        and not has_road_names
    )
    if low_shape:
        reasons.append("轨迹是一条扁长直线且没有街道名锚点，多为起终点连线")
        suggestions.append("截取包含完整轨迹曲线的页面，或放大到能看到道路名")
    if n_ocr_texts is not None and n_ocr_texts < 3:
        reasons.append(f"只识别到 {n_ocr_texts} 条文字，定位约束不足")
        suggestions.append("放大地图让 POI / 路名标注显示出来再截图")

    level = "reject" if low_shape else ("warn" if reasons else "ok")
    return {
        "level": level,
        "usable": level != "reject",
        "straightness": (round(float(straightness), 3)
                         if straightness is not None else None),
        "bbox_aspect": round(ratio, 2) if ratio is not None else None,
        "has_road_names": bool(has_road_names),
        "reasons": reasons,
        "suggestions": suggestions,
    }
