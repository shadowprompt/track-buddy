"""词典写入防毒：决定一条「地名 → 坐标」能不能沉淀进本地词典。

问题
----
``data/lexicon.json`` 原本是「越用越准」的数据飞轮，但写入没有门槛：
在线地理编码和 OSM 冷启动会在**配准之前**就把结果写进去，那时还不知道
这次识别准不准。低置信、起终点直线、锚点基准不一致的识别结果会一次次
给错误坐标加权；下次再用同名候选时它权重最高，错误于是被自我强化。

交接文档 §22.38 的反例说明为什么不能用自洽指标当门槛：把直线平移到
平行主干道后，贴路距离、锚点残差、confidence 会一起变好。因此门槛只
接受**外部证据**：最终置信度、输入信息量、尺度独立校验、锚点残差方向
相干性（由 pipeline 在结果出来后统一裁决）。

策略
----
* ``source="user"`` / ``verified=True``：用户在地图上显式纠正，永远放行，
  且写入后该条目升级为已核验、解除自动权重上限。
* 自动写入（在线源 / OSM）：
  1. 地名要像地名（长度、含文字、不是「X 公里内免费」这类 UI 文案），
     坐标必须有限且在经纬度范围内；
  2. 置信度低于 ``MIN_AUTO_CONFIDENCE`` 一律拒绝；
  3. 低信息量输入、尺度不一致、系统性锚点偏移一律拒绝；
  4. 与已有**异源且已核验/高权重**条目相距 200 m ~ 10 km 时拒绝，
     避免把「基准不一致」写成「同名候选」；超过 10 km 视为同名异地，
     本来就该保留多个候选（与 §22.40 的 near/far 分类一致）；
  5. 通过也只能把未核验条目的权重累积到 ``AUTO_WEIGHT_CAP``。
     ``lookup()`` 的 quality = weight/5，所以未核验条目永远达不到 1.0，
     不会盖过用户核验过的点位。

没有置信度证据（例如脚本单独调 ``Geocoder``）时**不直接丢弃**，
而是把 boost 压到 ``UNKNOWN_CONFIDENCE_BOOST`` 并同样封顶 —— 冷启动
仍然能积累，但积累速度慢、权重有限。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .geo.coords import haversine

MIN_AUTO_CONFIDENCE = 0.35
AUTO_WEIGHT_CAP = 3.0
UNKNOWN_CONFIDENCE_BOOST = 0.2
MIN_NAME_LEN = 2
CONFLICT_MIN_M = 200.0
CONFLICT_MAX_M = 10_000.0
USER_SOURCE = "user"

# 截图上的运营/统计文案，不是地名。它们来自 OCR 残片，一旦入词典
# 就会变成「看起来有权重的锚点」，是词典中毒最常见的入口。
_UI_TEXT = re.compile(
    r"公里|千米|分钟|小时|免费|累计|已骑|里程|本次|套餐|优惠|"
    r"起点|终点|导航|分享|km", re.IGNORECASE)
_HAS_LETTER = re.compile(r"[\u4e00-\u9fffA-Za-z]")


@dataclass(frozen=True)
class WriteDecision:
    """一次写入的裁决结果（写进 diagnostics 便于审计）。"""

    allowed: bool
    boost: float
    reason: str
    weight_cap: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "boost": round(float(self.boost), 4),
            "reason": self.reason,
            "weight_cap": self.weight_cap,
        }


def _reject(reason: str) -> WriteDecision:
    return WriteDecision(False, 0.0, reason)


def evaluate_write(
    *,
    key: str,
    lon: float,
    lat: float,
    source: str,
    boost: float,
    entries: Sequence[dict[str, Any]],
    confidence: float | None = None,
    self_consistent: bool = True,
    input_ok: bool = True,
    verified: bool = False,
) -> WriteDecision:
    """纯函数：给定已有条目与本次证据，判断能不能写。

    ``key`` 是 ``Lexicon.normalize(name)`` 的结果，由调用方传入，
    避免本模块反向依赖 ``geocode.Lexicon`` 造成循环导入。
    """
    # --- 基础合法性（对所有来源生效，用户纠正也要挡住脏数据）---
    if len(key) < MIN_NAME_LEN or not _HAS_LETTER.search(key):
        return _reject("name_not_place")
    if _UI_TEXT.search(key):
        return _reject("name_is_ui_text")
    try:
        lo, la, b = float(lon), float(lat), float(boost)
    except (TypeError, ValueError):
        return _reject("invalid_value")
    if not (math.isfinite(lo) and math.isfinite(la) and math.isfinite(b)):
        return _reject("invalid_value")
    if not (-180.0 <= lo <= 180.0 and -90.0 <= la <= 90.0) or b <= 0:
        return _reject("invalid_coordinate")

    # --- 用户显式核验：最高信任，不受任何自动门槛约束 ---
    if verified or source == USER_SOURCE:
        return WriteDecision(True, b, "user_verified", None)

    # --- 自动写入：先看外部证据 ---
    if not self_consistent:
        return _reject("self_inconsistent")
    if not input_ok:
        return _reject("low_information_input")
    if confidence is None:
        # 没有最终置信度（非流水线调用）：冷启动仍可用，但只给最小 boost
        return WriteDecision(
            True, min(b, UNKNOWN_CONFIDENCE_BOOST),
            "no_confidence_capped", AUTO_WEIGHT_CAP)
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    if not math.isfinite(conf) or conf < MIN_AUTO_CONFIDENCE:
        return _reject("low_confidence")

    # --- 异源冲突：近处不同坐标不新增，远处同名异地保留 ---
    for e in entries or []:
        if str(e.get("source") or "") == source:
            continue
        try:
            d = haversine(lo, la, float(e["lon"]), float(e["lat"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not (CONFLICT_MIN_M < d < CONFLICT_MAX_M):
            continue
        if bool(e.get("verified")) or float(e.get("weight", 0) or 0) >= AUTO_WEIGHT_CAP:
            return _reject("cross_source_conflict")

    return WriteDecision(True, b, "ok", AUTO_WEIGHT_CAP)
