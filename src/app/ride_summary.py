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
# ⚠️ OCR 常把日期与时间之间的空格吞掉：「2026-11-11 20:20:26」粘连成
# 「2026-11-1120:20:26」（2026-11-14 实测，duration_text 里可见）。
# 分隔符必须可选，日期/时/分秒用位数切分；解析后还要做范围校验，
# 防粘连错位读出 92 小时之类的胡值（见 test）。
# 分隔符族谱：美团用横杠，高德用点分且日期与时间粘连更狠
# 「2026.11.1608:13:27」（2026-11-16 实测）——正则靠回溯切位，
# 日期吃 2 位后时间才能对上冒号，无需专门处理粘连。
_RE_TIMESTAMP = re.compile(
    r"(\d{4})[.\-/年](\d{1,2})[.\-/月](\d{1,2})日?[ T]?(\d{1,2}):(\d{2})(?::(\d{2}))?")
_RE_DURATION = re.compile(r"(?:(\d+)\s*小时)?\s*(\d+)\s*分(?:\s*(\d+)\s*秒)?")
# 高德的时长是冒号三段式「00:14:14」（HH:MM:SS）。两段式 MM:SS 不采——
# 概要页顶部状态栏时钟「08:34」会被读成 8 分 34 秒。
_RE_DURATION_COLON = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
_RE_KCAL = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:k?cal|千卡|大卡)", re.IGNORECASE)
# km 需词边界防「3.5kmX」误读；单位族谱含中文（高德「3.68千米」）；
# 单位后紧跟 / 或 每 的是配速（「15千米/时」），必须排除
_RE_KM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:千米|公里|km)\b(?![每/])", re.IGNORECASE)
# 数字与单位被 OCR 切成两个文本块（高德「骑行 3.68 干米」）：数字块
# 的相邻块是纯单位块时也算候选。「干米」是 OCR 对「千米」的常见误读。
_RE_UNIT_TOKEN = re.compile(r"^(?:千米|公里|km)$", re.IGNORECASE)
_RE_PURE_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")
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


def _norm_token(s: str) -> str:
    """单位层面的 OCR 误读归一（只影响单位判定，不动原文本）。"""
    return s.replace("干米", "千米")


def _box_of(item: Any) -> tuple | None:
    """取文本块的几何框 (x0, y0, x1, y1)；无几何信息时返回 None。"""
    try:
        return (float(item.x0), float(item.y0),
                float(item.x1), float(item.y1))
    except AttributeError:
        return None


def _row_gap(a: tuple, b: tuple) -> float | None:
    """两个文本块的同行水平间距；非同行或相距过远返回 None。

    「同一行」= 纵向区间重叠超过较矮块高度的一半（概要卡片的标签与
    数值横排共线）；水平间距上限按字高相对化，防跨卡误配。
    """
    ha, hb = a[3] - a[1], b[3] - b[1]
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    if overlap <= 0.5 * min(ha, hb):
        return None
    gap = max(a[0], b[0]) - min(a[2], b[2])
    if gap > 8 * max(ha, hb):
        return None
    return max(gap, 0.0)


def _bind_labeled(
    tokens: list[str],
    candidates: list[tuple[int, float]],
    label_patterns: Iterable[re.Pattern[str]],
    boxes: list[tuple | None] | None = None,
) -> tuple[float | None, str | None]:
    """从数值候选里按标签挑出「本次」值。

    返回 ``(值, 标签文本)``。两档绑定策略：

    * **行感知**（有几何框时）：候选与标签在同一文本行内水平最近者
      胜出。索引距离在高德这类「地图区自带数字」的页面上没有区分度
      （比例尺「1公里」与里程牌「3KM」和概要卡片标签的块索引可能
      一样近），几何同行是唯一可靠的信号；
    * **索引窗口**（无几何框，纯字符串测试/旧调用方）：±2 块内找
      标签，多者取小，防同一横排出现两个带标签数值。

    无任何标签命中且候选多个时取最小值（累计恒大于本次）；单候选
    直接采信。
    """
    if not candidates:
        return None, None
    if len(candidates) == 1:
        idx, val = candidates[0]
        return val, None
    if (boxes is not None and len(boxes) == len(tokens)
            and all(b is not None for b in boxes)):
        best: tuple[float, float, str] | None = None  # (gap, val, label)
        for idx, val in candidates:
            cb = boxes[idx]
            for j, tok in enumerate(tokens):
                if j == idx:
                    continue
                for pat in label_patterns:
                    m = pat.search(tok)
                    if not m:
                        continue
                    gap = _row_gap(cb, boxes[j])
                    if gap is None:
                        continue
                    if best is None or gap < best[0]:
                        best = (gap, val, m.group(0))
        if best is not None:
            return best[1], best[2]
        # 同行无标签命中：回落索引窗口（页面排版怪异时的兜底）
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
    # 几何框与 tokens 对齐（纯字符串输入时全是 None → 走索引窗口回退）
    _raw = [t for t in (texts or []) if _text_of(t)]
    boxes = [_box_of(t) for t in _raw]

    # 标签按语义分三组（标签文案随画像，语义映射通用）
    _dist_pat = [re.compile(re.escape(l)) for l in ride_labels if "骑行" in l or "里程" in l]
    _cal_pat = [re.compile(re.escape(l)) for l in ride_labels if "消耗" in l or "卡" in l]
    _carbon_pat = [re.compile(re.escape(l)) for l in ride_labels if "碳" in l]

    out: dict[str, Any] = {k: None for k in _SUMMARY_FIELDS}
    out["duration_text"] = None
    notes: list[str] = []

    # --- 开始时间（可能多候选/粘连，逐个校验取第一个合法的） ---
    for s in tokens:
        for m in _RE_TIMESTAMP.finditer(s):
            y, mo, d, h, mi, sec = m.groups()
            mo_i, d_i = int(mo), int(d)
            h_i, mi_i, sec_i = int(h), int(mi), int(sec or 0)
            if not (1 <= mo_i <= 12 and 1 <= d_i <= 31
                    and h_i <= 23 and mi_i <= 59 and sec_i <= 59):
                continue  # 粘连错位/OCR 误读出的胡值，跳过看下一个
            out["started_at"] = (
                f"{int(y):04d}-{mo_i:02d}-{d_i:02d}T"
                f"{h_i:02d}:{mi_i:02d}:{sec_i:02d}")
            break
        if out["started_at"]:
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
    # 高德族：冒号三段式「00:14:14」。必须跳过含时间戳的块
    #（时间戳尾段的「:13:27」不是时长）。
    if out["duration_s"] is None:
        for s in tokens:
            if _RE_TIMESTAMP.search(s):
                continue
            m = _RE_DURATION_COLON.search(s)
            if m:
                h, mi, sec = (int(g) for g in m.groups())
                if h <= 23 and mi <= 59 and sec <= 59:
                    out["duration_s"] = h * 3600 + mi * 60 + sec
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
        norm = _norm_token(s)
        for m in _RE_KM.finditer(norm):
            try:
                km_cands.append((idx, float(m.group(1))))
            except ValueError:
                continue
        # 数字与单位被 OCR 切成两块（高德「3.68」+「干米」）：数字块的
        # 下一块是纯单位块时也构成候选（标签绑定逻辑不变，仍按数字块
        # 的近邻找标签）。
        if (_RE_PURE_NUMBER.match(norm) and idx + 1 < len(tokens)
                and _RE_UNIT_TOKEN.match(_norm_token(tokens[idx + 1]))):
            try:
                km_cands.append((idx, float(norm)))
            except ValueError:
                continue
    if km_cands and _dist_pat:
        val, label = _bind_labeled(tokens, km_cands, _dist_pat, boxes)
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
        notes.append("运动概要页未识别到任何概要字段——请确认上传的是运动"
                     "概要/详情页截图（含里程/时长/卡路里卡片）")

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

    # GPX 时间轴需要「开始时间 + 时长」两个端点：只有时长给不出绝对
    # 时刻序列（GPX 要求 UTC 绝对时间），提前说明缺什么。
    if out["duration_s"] is not None and out["started_at"] is None:
        notes.append("识别到时长但没有开始时间，GPX 导出将没有时间轴")
    out["notes"] = notes
    return out


def summary_properties(summary: dict[str, Any]) -> dict[str, Any]:
    """提取可并入 GeoJSON properties 的字段（仅非空值）。"""
    props: dict[str, Any] = {}
    for k in _SUMMARY_FIELDS:
        if summary.get(k) is not None:
            props[k] = summary[k]
    return props
