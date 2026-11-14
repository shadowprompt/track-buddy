"""运动记录概要解析：从「订单详情页」截图的 OCR 文本里读出一次骑行的
概要字段，与轨迹识别结果组合成完整的运动记录。

**输入是什么**：美团骑行 App 的「订单详情」页（不是轨迹截图）——顶部
地图缩略图下面有一排卡片：「203kcal 本次消耗 | 172g 本次减碳 | 3.5km
本次骑行」，再往下是「2026-11-11 20:20:26 骑行20分35秒」。用户把它与
轨迹截图一起上传，识别结果即可携带完整的运动元数据。

**解析策略**（按歧义风险从低到高）：

1. ``kcal`` / 时间戳 / 时长：单位模式几乎唯一，直接提取；
2. ``km``：页面同时有「累计骑行 1314km」和「本次骑行 3.5km」——用
   「本次骑行」标签的**文本块近邻绑定**（OCR 输出保持阅读顺序，
   标签与数值块索引相邻）选本次值；无标签时取最小候选（累计恒大于
   本次）；
3. ``g``（减碳克数）：单位太容易误报，**必须**有「减碳/碳」标签近邻
   才采信，否则宁缺勿错。

词表（标签）来自输入源画像（``SourceProfile.ride_labels``）——美团
订单页是「本次骑行/本次消耗/本次减碳」，其它 App 的标签文案不同，
各画像自备。数值单位正则（km/分秒/kcal/g）是通用的，不进画像。

交叉校验：轨迹识别的几何里程（``distance_m``）与订单里程应该是
同一件事的两种度量，量级不一致时输出告警——订单值来自平台记账，
以它为基准报告偏差方向。
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Iterable

from .config import CONFIG

# --- 通用单位正则（与来源无关，不进画像） ---
_RE_TIMESTAMP = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?")
_RE_DURATION = re.compile(r"(?:(\d+)\s*小时)?\s*(\d+)\s*分(?:\s*(\d+)\s*秒)?")
_RE_KCAL = re.compile(r"(\d+(?:\.\d+)?)\s*k?cal", re.IGNORECASE)
# km 需词边界防「3.5kmX」误读；单独的 g 需排除 kg
_RE_KM = re.compile(r"(\d+(?:\.\d+)?)\s*km\b", re.IGNORECASE)
_RE_GRAMS = re.compile(r"(\d+(?:\.\d+)?)\s*g\b", re.IGNORECASE)

# 标签文本块与数值文本块允许的最大索引距离（OCR 按阅读顺序输出，
# 订单页标签与数值在同一横排卡片，索引相邻）
_LABEL_WINDOW = 2

_SUMMARY_FIELDS = (
    "distance_km", "duration_s", "calories_kcal", "carbon_g", "started_at")


def _text_of(item: Any) -> str:
    s = getattr(item, "text", None)
    if s is None and isinstance(item, (tuple, list)) and item:
        s = item[0]
    return str(s or "").strip()


def _bind_labeled(
    tokens: list[str],
    candidates: list[tuple[int, float]],
    label_patterns: Iterable[re.Pattern[str]],
) -> tuple[float | None, str | None]:
    """从数值候选里按标签近邻挑出「本次」值。

    返回 ``(值, 标签文本)``：优先取窗口内有标签命中的候选（多者取小，
    防同一横排出现两个带标签数值）；无标签命中且候选多个时取最小值
    （订单页累计值恒大于本次值）；单候选直接采信。
    """
    if not candidates:
        return None, None
    if len(candidates) == 1:
        idx, val = candidates[0]
        return val, None
    labeled: list[tuple[int, float, str]] = []
    for idx, val in candidates:
        for j in range(max(0, idx - _LABEL_WINDOW),
                       min(len(tokens), idx + _LABEL_WINDOW + 1)):
            if j == idx:
                continue
            for pat in label_patterns:
                m = pat.search(tokens[j])
                if m:
                    labeled.append((idx, val, m.group(0)))
    if labeled:
        idx, val, label = min(labeled, key=lambda x: x[1])
        return val, label
    # 无标签：取最小（累计 > 本次是订单页的固定关系）
    idx, val = min(candidates, key=lambda x: x[1])
    return val, None


def parse_ride_summary(
    texts: Iterable[Any],
    *,
    geom_distance_m: float | None = None,
    ride_labels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """解析订单详情页 OCR 文本，返回结构化运动概要。

    缺什么字段置 None，绝不用猜测值填充；``available`` 表示至少提出
    一个字段。``geom_distance_m`` 传入轨迹识别的几何里程时做交叉校验。
    """
    if ride_labels is None:
        ride_labels = tuple(CONFIG.source.ride_labels)

    tokens = [_text_of(t) for t in (texts or []) if _text_of(t)]

    # 标签按语义分三组（标签文案随画像，语义映射通用）
    _dist_pat = [re.compile(re.escape(l)) for l in ride_labels if "骑行" in l or "里程" in l]
    _cal_pat = [re.compile(re.escape(l)) for l in ride_labels if "消耗" in l or "卡" in l]
    _carbon_pat = [re.compile(re.escape(l)) for l in ride_labels if "碳" in l]

    out: dict[str, Any] = {k: None for k in _SUMMARY_FIELDS}
    out["duration_text"] = None
    notes: list[str] = []

    # --- 开始时间 ---
    for s in tokens:
        m = _RE_TIMESTAMP.search(s)
        if m:
            y, mo, d, h, mi, sec = m.groups()
            out["started_at"] = (f"{y}-{mo}-{d}T{h}:{mi}:"
                                 + (sec or "00"))
            break

    # --- 时长（「骑行20分35秒」，可能整块或含「小时」） ---
    for s in tokens:
        m = _RE_DURATION.search(s)
        if m and ("分" in s or "小时" in s):
            h, mi, sec = m.groups()
            out["duration_s"] = (int(h or 0) * 3600 + int(mi) * 60
                                 + int(sec or 0))
            out["duration_text"] = s
            break

    # --- 卡路里（kcal 单位唯一，直接取） ---
    kcal_vals = []
    for s in tokens:
        m = _RE_KCAL.search(s)
        if m:
            kcal_vals.append(float(m.group(1)))
    if kcal_vals:
        out["calories_kcal"] = statistics.median(kcal_vals)

    # --- 本次骑行距离（km 多候选需标签绑定） ---
    km_cands: list[tuple[int, float]] = []
    for idx, s in enumerate(tokens):
        for m in _RE_KM.finditer(s):
            try:
                km_cands.append((idx, float(m.group(1))))
            except ValueError:
                continue
    if km_cands and _dist_pat:
        val, label = _bind_labeled(tokens, km_cands, _dist_pat)
        out["distance_km"] = val
        if label:
            out["distance_label"] = label
        if len(km_cands) > 1:
            notes.append(
                f"页面有 {len(km_cands)} 个公里数候选"
                f"（{[v for _, v in km_cands]}），"
                + ("已按「本次」标签绑定" if label else "无标签，取最小值"))
    elif km_cands:
        out["distance_km"] = min(v for _, v in km_cands)

    # --- 减碳克数（必须有「碳」标签近邻，宁缺勿错） ---
    g_cands: list[tuple[int, float]] = []
    for idx, s in enumerate(tokens):
        for m in _RE_GRAMS.finditer(s):
            try:
                g_cands.append((idx, float(m.group(1))))
            except ValueError:
                continue
    if g_cands and _carbon_pat:
        # 「g」单位太容易误报，**必须**有碳标签近邻才采信——这里不能用
        # _bind_labeled（它的单候选短路会绕过标签检查），单独判定：
        def _has_carbon_label(idx: int) -> bool:
            for j in range(max(0, idx - _LABEL_WINDOW),
                           min(len(tokens), idx + _LABEL_WINDOW + 1)):
                if j == idx:
                    continue
                if any(p.search(tokens[j]) for p in _carbon_pat):
                    return True
            return False

        labeled_vals = [val for idx, val in g_cands if _has_carbon_label(idx)]
        out["carbon_g"] = min(labeled_vals) if labeled_vals else None
        if not labeled_vals:
            notes.append(
                f"发现 {len(g_cands)} 个克数候选但页面无减碳标签，宁缺勿错不采信")
    elif g_cands:
        notes.append(
            f"发现 {len(g_cands)} 个克数候选但页面无减碳标签，宁缺勿错不采信")

    found = [k for k in _SUMMARY_FIELDS if out.get(k) is not None]
    out["fields_found"] = found
    out["available"] = bool(found)
    if not found:
        notes.append("订单详情页未识别到任何概要字段——确认传的是订单"
                     "详情页截图（含里程/时长/卡路里卡片）")

    # --- 与轨迹几何里程交叉校验 ---
    cc: dict[str, Any] = {"available": False, "consistent": True,
                          "ratio": None, "note": ""}
    if out["distance_km"] and geom_distance_m:
        ratio = float(geom_distance_m) / (out["distance_km"] * 1000.0)
        cc.update({"available": True, "ratio": round(ratio, 3)})
        if not (0.6 <= ratio <= 1.7):
            cc["consistent"] = False
            cc["note"] = (
                f"轨迹几何里程 {geom_distance_m:.0f}m 与订单里程 "
                f"{out['distance_km']}km 量级不符（比值 {ratio:.2f}）——"
                "订单值来自平台记账，请检查轨迹截图是否完整覆盖本次行程")
        else:
            cc["note"] = "几何里程与订单里程量级一致"
    out["cross_check"] = cc
    out["notes"] = notes
    return out


def summary_properties(summary: dict[str, Any]) -> dict[str, Any]:
    """提取可并入 GeoJSON properties 的字段（仅非空值）。"""
    props: dict[str, Any] = {}
    for k in _SUMMARY_FIELDS:
        if summary.get(k) is not None:
            props[k] = summary[k]
    return props
