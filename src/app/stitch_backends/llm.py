"""大模型驱动的拼接后端。

## 两种模式，区别只在「变换矩阵由谁给」

- ``llm``：**纯语义配准**。多模态模型看图，直接判断哪些图两两相邻、
  j 相对 i 在画布上的位移（以图宽/高为 1 的比例），代码把它转成
  3×3 平移矩阵后跳过 SIFT。这是「交给大模型拼」的字面实现。
- ``llm_guided``：模型只圈定**候选图对**，变换矩阵仍由 SIFT+RANSAC
  在这些对上求。测的是「语义先验能不能缩小搜索空间」。

两者之后都走同一套连通块 / 位姿树 / 贴图 / 逐像素复核 / 徽章守恒，
所以 `verify` 等客观指标可直接横向比较。

## 为什么不要求模型输出像素坐标

公开基准与本项目 HANDOVER §21.5 的结论一致：VLM 能做离散语义判断，
给不出稳定的像素级坐标。因此这里只让它输出**相对比例**，且把
「连续坐标预测」降级成「方位 + 一个粗比例」，误差如实体现在
`verify_score` 上——这正是 A/B 要测的东西，不是实现缺陷需要藏起来。

## 数值诚实性

纯语义边没有做过几何配准，所以 `inliers=0 / ncc=0`，位姿树改用
模型给的 `confidence` 作为权重（`PairLink.weight`）。结果里会明确
标注来源，绝不能把这些 0 当成「SIFT 匹配失败」。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..multishot import PairLink
from .base import BackendSpec, StitchPlan
from .registry import register

THUMB_MAX_EDGE = 768
MAX_IMAGES = 16
MAX_SHIFT_RATIO = 1.5
MIN_SHIFT_RATIO = 0.02

_SYSTEM = """你是地图截图拼接助手。用户给你同一条骑行路线沿路滑动截取的
多张连续截图，每张图左上角有 #1、#2… 的编号，图片按编号顺序给出。

任务：判断哪些图两两之间存在重叠/相邻关系，并估计第 j 张相对第 i 张
在拼接画布上的位移。只依据画面里的道路走向、轨迹线、地名与 POI 的
连续性判断，不要臆测看不清的图对。"""


def _user_prompt(n: int) -> str:
    return f"""共 {n} 张截图，编号 #1 到 #{n}。请输出一个 JSON 对象：

{{"pairs": [{{"i": 1, "j": 2, "dx_ratio": 0.75, "dy_ratio": 0.05,
             "confidence": 0.8}}], "notes": "简要说明"}}

字段含义：
- i、j 是图编号（1 开始），i != j。
- dx_ratio / dy_ratio 是「第 j 张相对第 i 张」的位移，以一张图的
  宽 / 高为 1。j 在 i 右侧且两图重叠约 1/4 宽 → dx_ratio=+0.75；
  j 在 i 左侧 → dx_ratio 为负；j 在 i 下方 → dy_ratio 为正。
- confidence 是你对这条相邻关系的把握，0 到 1。

规则：只输出你确信有重叠的图对；不要给 dx_ratio 和 dy_ratio 同时为 0
的图对（那意味着两图完全重合）；不要输出 Markdown 或解释文字。"""


_SCHEMA_HINT = "只输出一个 JSON 对象，键为 pairs 与 notes；不要代码块。"


def _write_thumbs(images: list[np.ndarray], folder: Path) -> list[str]:
    """缩放并加序号角标，降低 token 成本、让模型能对应上编号。"""
    paths: list[str] = []
    for k, im in enumerate(images, start=1):
        h, w = im.shape[:2]
        scale = min(1.0, THUMB_MAX_EDGE / max(h, w))
        small = cv2.resize(
            im, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.rectangle(small, (0, 0), (92, 46), (255, 255, 255), -1)
        cv2.rectangle(small, (0, 0), (92, 46), (25, 25, 25), 2)
        cv2.putText(small, f"#{k}", (9, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (25, 25, 25), 2, cv2.LINE_AA)
        p = folder / f"shot_{k:02d}.jpg"
        ok, buf = cv2.imencode(
            ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            p.write_bytes(buf.tobytes())
            paths.append(str(p))
    return paths


def _parse_pairs(data: dict[str, Any], n: int) -> tuple[list[dict], int, list[str]]:
    """把模型输出规范化成 0-based 的 (i<j, dx, dy, confidence)。"""
    raw = data.get("pairs")
    if not isinstance(raw, list):
        raw = data.get("relations") if isinstance(data.get("relations"), list) else []
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    rejected = 0
    # 模型实测会给字符串（提示词里 notes 就是字符串）；直接迭代字符串
    # 会把「依据北翟路…」拆成 ['依','据','北','翟']，污染诊断卡片。
    raw_notes = data.get("notes")
    if isinstance(raw_notes, str):
        notes = [raw_notes] if raw_notes.strip() else []
    elif isinstance(raw_notes, list):
        notes = [str(x) for x in raw_notes if x][:4]
    else:
        notes = []
    for item in raw:
        if not isinstance(item, dict):
            rejected += 1
            continue
        try:
            i = int(item.get("i"))
            j = int(item.get("j"))
            dx = float(item.get("dx_ratio", item.get("dx", 0)) or 0)
            dy = float(item.get("dy_ratio", item.get("dy", 0)) or 0)
            conf = float(item.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            rejected += 1
            continue
        if not (1 <= i <= n and 1 <= j <= n) or i == j:
            rejected += 1
            continue
        if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(conf)):
            rejected += 1
            continue
        if abs(dx) > MAX_SHIFT_RATIO or abs(dy) > MAX_SHIFT_RATIO:
            rejected += 1
            continue
        if math.hypot(dx, dy) < MIN_SHIFT_RATIO:
            # 与几何层一致：两张不同视角的图不可能配出 0 位移
            rejected += 1
            continue
        a, b = i - 1, j - 1
        if a > b:
            a, b = b, a
            dx, dy = -dx, -dy
        if (a, b) in seen:
            rejected += 1
            continue
        seen.add((a, b))
        out.append({
            "i": a, "j": b, "dx": dx, "dy": dy,
            "confidence": max(0.01, min(1.0, conf)),
        })
    return out, rejected, notes


def _semantic_pairs(
    images: list[np.ndarray], client: Any, cfg: Any = None,
) -> tuple[list[dict], dict[str, Any], str]:
    """调一次多模态模型，返回规范化的相邻关系与诊断。

    失败一律通过第三个返回值（error）上报，绝不回退到 SIFT。
    """
    n = len(images)
    if n > MAX_IMAGES:
        return [], {"images": n}, (
            f"一次最多处理 {MAX_IMAGES} 张图，本次收到 {n} 张；"
            "请减少截图数量或改用 SIFT 后端")
    diag: dict[str, Any] = {
        "strategy": "多模态模型语义相邻判断",
        "images": n,
        "thumbnail_max_edge": THUMB_MAX_EDGE,
    }
    with tempfile.TemporaryDirectory(prefix="wb_stitch_llm_") as td:
        paths = _write_thumbs(images, Path(td))
        if len(paths) != n:
            return [], diag, "缩略图生成失败，无法调用大模型"
        res = client.chat_json(
            system=_SYSTEM, user=_user_prompt(n), images=paths,
            schema_hint=_SCHEMA_HINT,
        )
    diag.update({
        "model": getattr(res, "model", "") or getattr(client, "model", ""),
        "elapsed_ms": getattr(res, "elapsed_ms", 0),
        "images_sent": len(paths),
        "raw_response": (getattr(res, "text", "") or "")[:2000],
    })
    if not getattr(res, "ok", False):
        err = getattr(res, "error", "") or "模型调用失败"
        diag["error"] = err
        return [], diag, (
            f"大模型调用失败：{err}。本次未拼接，"
            "也没有回退到 SIFT（避免污染 A/B 对比）")
    data = getattr(res, "data", None) or {}
    pairs, rejected, notes = _parse_pairs(data, n)
    diag.update({
        "pairs_reported": len(data.get("pairs") or [])
        if isinstance(data.get("pairs"), list) else 0,
        "pairs_accepted": len(pairs),
        "pairs_rejected": rejected,
        "model_notes": notes,
    })
    if not pairs:
        return [], diag, (
            "大模型没有给出任何可用的相邻关系（可能看不出重叠或输出格式不对），"
            "本次未拼接，也没有回退到 SIFT")
    return pairs, diag, ""


def _links_from_pairs(
    images: list[np.ndarray], pairs: list[dict],
) -> dict[tuple[int, int], PairLink]:
    """把「j 相对 i 的位移比例」转成 multishot 的 3×3 变换。

    `PairLink.matrix` 的语义是「把 i 的像素坐标映射到 j 的像素坐标」。
    若 j 在画布上比 i 右移 dx*w，则 i 中的点在 j 坐标系里要 **减** 去
    这段位移，所以矩阵平移取负号。方向用 2026-11-12 的真实落盘结果
    校准过：link(0→1).tx=+866，而画布上图 1 比图 0 左移 867px。
    """
    links: dict[tuple[int, int], PairLink] = {}
    for p in pairs:
        i, j = p["i"], p["j"]
        h, w = images[i].shape[:2]
        matrix = np.array([
            [1.0, 0.0, -p["dx"] * w],
            [0.0, 1.0, -p["dy"] * h],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        links[(i, j)] = PairLink(
            i=i, j=j, matrix=matrix,
            inliers=0, good=0, ncc=0.0, ui_px=0,
            weight=p["confidence"], source="llm",
        )
    return links


def plan_llm(
    images: list[np.ndarray], cfg: Any = None, client: Any = None,
) -> StitchPlan:
    pairs, diag, err = _semantic_pairs(images, client, cfg)
    diag["mode"] = "llm"
    diag["sift_skipped"] = True
    if err:
        return StitchPlan(error=err, diagnostics=diag)
    links = _links_from_pairs(images, pairs)
    return StitchPlan(
        links=links,
        diagnostics=diag,
        notes=[
            f"大模型语义配准：采纳 {len(links)} 条相邻关系，"
            "未做 SIFT 逐对配准；links 里的内点数为 0 是正常值，"
            "位姿树按模型置信度选边",
        ],
    )


def plan_llm_guided(
    images: list[np.ndarray], cfg: Any = None, client: Any = None,
) -> StitchPlan:
    pairs, diag, err = _semantic_pairs(images, client, cfg)
    diag["mode"] = "llm_guided"
    diag["sift_skipped"] = False
    if err:
        return StitchPlan(error=err, diagnostics=diag)
    candidates = sorted({(p["i"], p["j"]) for p in pairs})
    return StitchPlan(
        pair_candidates=candidates,
        diagnostics=diag,
        notes=[
            f"大模型给出 {len(candidates)} 对候选，SIFT 只在这些对上求精确变换；"
            "模型判断错的图对会在内点/相似度过滤阶段被剔除",
        ],
    )


register(BackendSpec(
    id="llm",
    label="大模型语义配准（纯 VLM）",
    description=(
        "多模态模型直接判断图间相邻关系与相对位移，跳过 SIFT；"
        "需配置 LLM_BASE_URL / LLM_API_KEY。用于量化语义粗配准与几何精配准的差距。"),
    plan=plan_llm,
    requires_llm=True,
    skips_sift=True,
))

register(BackendSpec(
    id="llm_guided",
    label="大模型候选 + SIFT 精配准",
    description=(
        "模型只圈定可能重叠的图对，变换仍由 SIFT+RANSAC 求；"
        "测的是语义先验能否缩小搜索空间。需配置大模型服务。"),
    plan=plan_llm_guided,
    requires_llm=True,
    skips_sift=False,
))
