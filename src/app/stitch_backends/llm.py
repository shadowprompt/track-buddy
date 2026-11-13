"""大模型驱动的拼接后端。

## 两种模式，区别只在「变换矩阵由谁给」

- ``llm``：**纯语义配准**。多模态模型看图，识别图间相邻关系与全局顺序、
  估计 j 相对 i 在画布上的位移（以图宽/高为 1 的比例），代码把它转成
  3×3 平移矩阵后跳过 SIFT。这是「交给大模型拼」的字面实现。
- ``llm_guided``：模型只负责**识别**哪些图相邻、按什么顺序，变换矩阵
  仍由 SIFT+RANSAC 在这些对上求。测的是「语义先验能不能缩小搜索空间」。

两者之后都走同一套连通块 / 位姿树 / 贴图 / 逐像素复核 / 徽章守恒，
所以 `verify` 等客观指标可直接横向比较。

## 提示词 v2：识别的证据优先级（2026-11-13）

v1 只说「依据道路走向、轨迹线、地名与 POI 的连续性判断」，模型实际
**只凭吴淞江 / 北翟路 / 天山西路 / 小区名重复出现**就凑边，7 张图给
3 条边、`dx_ratio` 符号 -0.62 / +0.78 / -0.55 / +0.55 / -0.5 / -0.52
来回翻转，画布折叠成一小团。根因是提示词没把「按轨迹线首尾接续」写成
硬约束，也没要求输出一条覆盖全部截图的链。

v2 的四条硬约束：编号只是上传顺序；轨迹线的出边↔入边是唯一主证据；
必须给出 1..n 排列的 `order` 且只输出相邻边；沿链主导轴符号必须一致、
每条边必填 `evidence`。解析层同步做 order / 连通性 / 符号一致性体检，
结果写进 `backend_diagnostics`，这样「识别这一步模型做到了什么程度」
可以直接从结果卡片读出来，而不是靠猜一张折叠画布。

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
# 方向一致性判定：主导轴上小于这个比例的位移算微调，不算「符号翻转」
MIN_CONFLICT_RATIO = 0.12
MAX_EVIDENCE_CHARS = 240
# 写进诊断，A/B 复盘时能分清是哪一版提示词产出的结果
PROMPT_VERSION = "trajectory-chain-v2"

_SYSTEM = """你是地图截图拼接助手。输入是同一条骑行路线沿路滑动截取的多张
地图截图，每张图左上角有 #1、#2… 的编号。**编号只是上传顺序，不是
路线先后**；图片按编号顺序给你，但真实先后顺序必须由你自己判断。

## 唯一的主证据：轨迹线

画面里的彩色骑行轨迹线（一条连续粗线，两端可能有「起」「终」圆形徽章）
是判断相邻关系的**唯一主证据**。对每张图依次做：

1. 找出轨迹线从画面哪条边进入、从哪条边离开（上/下/左/右），以及
   出/入点在边上的大致位置（例如「右侧边偏上 1/3 处」）。
2. 记下轨迹在出/入点附近的走向（朝东/西/南/北、沿哪条路或河道、
   有没有拐弯），以及紧邻的路口、桥、河道弯曲等形状特征。
3. 两张图能判定相邻，当且仅当：一张图的**出边**与另一张图的**入边**
   方位互为对侧（右出↔左入、下出↔上入…），且轨迹走向、所沿道路/河道、
   路口形状能连成**同一条连续轨迹**。
4. 整条链上「起点」徽章只应出现一次、「终点」徽章只应出现一次，分别
   在链的两端。含起点徽章的图排在 order 的一端，含终点徽章的图在另一端。

## 明确禁止

- **禁止只因为两图出现相同地名 / POI / 小区名 / 路名就判定相邻**。
  同一条路、同一个片区会反复出现在多张截图里，这是最常见的误判来源。
  地名只能用来佐证轨迹线接续，不能单独当作相邻依据。
- 禁止按 #1→#2→#3 的编号顺序硬连；禁止凭「看起来都像同一片区」凑边。
- 看不清、接不上的图不要硬塞进链；但仍要给它一个位置，并在 notes
  里写明「#k 接不上」及原因。

## 输出必须是一条链，不是若干零散图对

先定 `order`（1..n 的一个排列，沿轨迹线从一端到另一端的真实截图顺序），
再**只输出 order 中相邻两张图之间的边**（理想为 n-1 条）。不要给分叉、
不要给跨过中间图的边、不要留孤立图。

`pairs` 里 i 必须是 order 中靠前的那张、j 是紧随其后的那张；
`dx_ratio` / `dy_ratio` 是「j 相对 i 在画布上的位移」。截图是沿路线
朝同一方向滑动截的，所以**所有边的主导轴（dx 或 dy 中绝对值更大的
那个）必须同号**；如果你发现必须一会儿 +0.78、一会儿 -0.55 才能接上，
说明 order 排错了，请重排，不要输出自相矛盾的边。
"""


def _user_prompt(n: int) -> str:
    # 示例放在普通字符串里，避免 f-string 的花括号转义把 JSON 弄坏。
    # 示例按 5 张图给出，只示意格式；实际 order 必须是 1..n 的排列。
    example = """{
  "order": [3, 1, 5, 2, 4],
  "pairs": [
    {"i": 3, "j": 1, "dx_ratio": 0.78, "dy_ratio": 0.04,
     "confidence": 0.85,
     "evidence": "#3 轨迹从右侧边偏上出画、沿吴淞江向东；#1 轨迹从左侧边同高度入画、同一条江继续向东"}
  ],
  "notes": "整体走向，以及哪张图接不上、原因"
}"""
    return f"""共 {n} 张截图，编号 #1 到 #{n}。请只输出一个 JSON 对象
（示例按 5 张图给出，仅示意格式，不是答案）：

{example}

字段说明：
- order：1..{n} 的**排列**，按轨迹线从一端到另一端的实际先后顺序，
  共 {n} 个数、不重复、不遗漏。
- pairs：只给 order 里相邻两图的边，理想为 {n - 1} 条。i 是 order 里
  靠前的编号，j 是紧跟其后的编号。
- dx_ratio / dy_ratio：j 相对 i 的画布位移，以一张图的宽 / 高为 1。
  j 在 i 右侧 → dx_ratio 为正；左侧 → 为负；j 在 i 下方 → dy_ratio
  为正；上方 → 为负。例：j 在 i 右侧且两图重叠约 1/4 宽 → dx_ratio=+0.75。
- confidence：0~1。只有轨迹线确实接得上的边才给 >0.6；仅凭地名
  猜测的给 <0.4。
- evidence：**每条边必填**，写清「#i 从哪条边出 → #j 从哪条边入，
  靠哪条路 / 河道 / 路口形状接上」。没有 evidence 的边会被视为臆测。
- notes：说明你定的整体走向、哪张图接不上及原因。

输出前逐条自检：
1. order 是 1..{n} 的排列，没有重复或遗漏；
2. pairs 只连接 order 中相邻的编号，且 i 在前、j 在后；
3. 所有边的主导轴符号一致，没有 + / - 来回翻转；
4. 每条边都有 evidence，依据是轨迹线接续而不是地名相同。"""


_SCHEMA_HINT = (
    "只输出一个 JSON 对象，键为 order、pairs、notes；"
    "不要 Markdown 代码块，不要解释文字。"
)


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


def _parse_order(data: dict[str, Any], n: int) -> tuple[list[int], str]:
    """解析「沿轨迹线的全局截图顺序」，必须是 1..n 的排列。

    不合法时返回空列表 + 原因，**不阻断**本次调用：用户正在测「识别能
    做到什么程度」，格式不合规本身就是结论之一，要如实写进诊断。
    """
    raw = data.get("order")
    if not isinstance(raw, list):
        return [], "模型未输出 order（无法判断全局顺序）"
    vals: list[int] = []
    for item in raw:
        try:
            vals.append(int(item))
        except (TypeError, ValueError):
            return [], "order 含非整数项"
    if sorted(vals) != list(range(1, n + 1)):
        return [], (
            f"order 不是 1..{n} 的排列（收到 {vals[:16]}，"
            f"共 {len(vals)} 项）")
    return vals, ""


def _parse_pairs(data: dict[str, Any], n: int) -> tuple[list[dict], int, list[str]]:
    """把模型输出规范化成 0-based 的 (i<j, dx, dy, confidence)。

    `evidence` 原样带出（限长），供结果卡片展示模型的接续依据。
    """
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
        ev = item.get("evidence")
        if isinstance(ev, str):
            evidence = ev.strip()[:MAX_EVIDENCE_CHARS]
        elif isinstance(ev, (int, float)):
            evidence = str(ev)[:MAX_EVIDENCE_CHARS]
        else:
            evidence = ""
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
            "evidence": evidence,
        })
    return out, rejected, notes


def _chain_diagnostics(
    pairs: list[dict], order: list[int], n: int, order_error: str = "",
) -> dict[str, Any]:
    """对「识别」结果做只读体检，不修改任何边。

    量三件事：order 本身合不合规、所有图是否连成一条链、沿链前进时
    位移主导轴的符号是否自相矛盾。2026-11-13 折叠事故的 dx 序列
    (-0.62/+0.78/-0.55/+0.55/-0.5/-0.52) 应当被这里直接检出。
    """
    # 调用方若已判定 order 解析失败，即使顺手传了内容也不能算有效顺序
    valid = bool(order) and not order_error
    pos = {k - 1: idx for idx, k in enumerate(order)} if valid else {}
    oriented: list[tuple[int, int, float, float]] = []
    for p in pairs:
        a, b, dx, dy = p["i"], p["j"], p["dx"], p["dy"]
        if pos:
            pa, pb = pos.get(a, -1), pos.get(b, -1)
            if 0 <= pb < pa:
                # 统一换成「沿 order 前进」方向的位移，才能比符号
                a, b, dx, dy = b, a, -dx, -dy
        oriented.append((a, b, dx, dy))

    sx = sum(abs(dx) for _, _, dx, _ in oriented)
    sy = sum(abs(dy) for _, _, _, dy in oriented)
    axis = "x" if sx >= sy else "y"
    vals = [(dx if axis == "x" else dy) for _, _, dx, dy in oriented]
    votes = [1 if v > 0 else -1 for v in vals if abs(v) >= MIN_CONFLICT_RATIO]
    ref_sign = -1 if sum(votes) < 0 else 1
    conflicts: list[dict[str, Any]] = []
    for (a, b, _, _), v in zip(oriented, vals):
        if abs(v) >= MIN_CONFLICT_RATIO and (1 if v > 0 else -1) != ref_sign:
            conflicts.append({
                "i": a + 1, "j": b + 1, "axis": axis,
                "value": round(v, 3), "expected_sign": ref_sign,
            })

    off_chain: list[list[int]] = []
    if pos:
        for p in pairs:
            pa, pb = pos.get(p["i"], -1), pos.get(p["j"], -1)
            if pa < 0 or pb < 0 or abs(pa - pb) != 1:
                off_chain.append([p["i"] + 1, p["j"] + 1])

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        ra, rb = find(p["i"]), find(p["j"])
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k + 1)
    covered = {k for p in pairs for k in (p["i"], p["j"])}
    missing = [k + 1 for k in range(n) if k not in covered]
    evidence = [
        {"i": p["i"] + 1, "j": p["j"] + 1, "text": p["evidence"]}
        for p in pairs if p.get("evidence")
    ][:8]

    return {
        "order": list(order),
        "order_valid": valid,
        "order_error": order_error,
        "pairs_expected": max(0, n - 1),
        "off_chain_pairs": off_chain,
        "dominant_axis": axis,
        "direction_conflicts": conflicts,
        "direction_conflict_count": len(conflicts),
        "connected_components": len(groups),
        "components": [sorted(v) for v in groups.values()],
        "missing_images": missing,
        "chain_complete": valid and len(groups) == 1 and not missing,
        "evidence": evidence,
    }


def _semantic_pairs(
    images: list[np.ndarray], client: Any, cfg: Any = None,
) -> tuple[list[dict], dict[str, Any], str]:
    """调一次多模态模型，返回规范化的相邻关系与诊断。

    失败一律通过第三个返回值（error）上报，绝不回退到 SIFT。
    """
    n = len(images)
    if n > MAX_IMAGES:
        return [], {"images": n, "prompt_version": PROMPT_VERSION}, (
            f"一次最多处理 {MAX_IMAGES} 张图，本次收到 {n} 张；"
            "请减少截图数量或改用 SIFT 后端")
    diag: dict[str, Any] = {
        "strategy": "多模态模型按轨迹线识别全局链条",
        "images": n,
        "thumbnail_max_edge": THUMB_MAX_EDGE,
        "prompt_version": PROMPT_VERSION,
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
    order, order_error = _parse_order(data, n)
    pairs, rejected, notes = _parse_pairs(data, n)
    diag.update({
        "pairs_reported": len(data.get("pairs") or [])
        if isinstance(data.get("pairs"), list) else 0,
        "pairs_accepted": len(pairs),
        "pairs_rejected": rejected,
        "model_notes": notes,
    })
    diag.update(_chain_diagnostics(pairs, order, n, order_error))
    if not pairs:
        extra = f"（{order_error}）" if order_error else ""
        return [], diag, (
            "大模型没有给出任何可用的相邻关系（可能看不出轨迹线接续或"
            f"输出格式不对）{extra}，本次未拼接，也没有回退到 SIFT")
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


def _chain_warnings(diag: dict[str, Any]) -> list[str]:
    """把识别体检的结论翻成用户能直接看懂的告警。"""
    out: list[str] = []
    if diag.get("order_error"):
        out.append(f"⚠ 全局顺序无效：{diag['order_error']}")
    if not diag.get("chain_complete"):
        out.append(
            f"⚠ 链条不完整：{diag.get('connected_components')} 个连通块，"
            f"未被任何边覆盖的图 {diag.get('missing_images')}；"
            f"本次只采纳 {diag.get('pairs_accepted')}/"
            f"{diag.get('pairs_expected')} 条应有相邻边")
    k = int(diag.get("direction_conflict_count") or 0)
    if k:
        out.append(
            f"⚠ {k} 条边的位移方向与链条主方向（{diag.get('dominant_axis')} 轴）"
            "相反，画布很可能折叠；这是模型输出自相矛盾，不是几何配准失败")
    off = diag.get("off_chain_pairs") or []
    if off:
        out.append(f"⚠ 有 {len(off)} 条边不在全局顺序的相邻位置上：{off[:4]}")
    return out


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
        ] + _chain_warnings(diag),
    )


def plan_llm_guided(
    images: list[np.ndarray], cfg: Any = None, client: Any = None,
) -> StitchPlan:
    pairs, diag, err = _semantic_pairs(images, client, cfg)
    diag["mode"] = "llm_guided"
    diag["sift_skipped"] = False
    if err:
        return StitchPlan(error=err, diagnostics=diag)
    direct = {(p["i"], p["j"]) for p in pairs}
    candidates = set(direct)
    # 模型「识别」出的全局顺序比零散图对更有价值：即使它只给了 3 条边，
    # 也把 order 的相邻图对交给 SIFT 验证，否则 7 张图永远连不全。
    order = diag.get("order") or []
    for a, b in zip(order, order[1:]):
        i, j = a - 1, b - 1
        if i != j:
            candidates.add((min(i, j), max(i, j)))
    candidates = sorted(candidates)
    diag["pair_candidates"] = len(candidates)
    diag["candidates_from_order"] = len(candidates) - len(direct)
    return StitchPlan(
        pair_candidates=candidates,
        diagnostics=diag,
        notes=[
            f"大模型识别出 {len(pairs)} 条边、全局顺序覆盖 {len(order)}/"
            f"{len(images)} 张；共 {len(candidates)} 对候选送 SIFT 求精确变换"
            f"（其中 {len(candidates) - len(direct)} 对由 order 补齐）；"
            "模型判断错的图对会在内点/相似度过滤阶段被剔除",
        ] + _chain_warnings(diag),
    )


register(BackendSpec(
    id="llm",
    label="大模型语义配准（纯 VLM）",
    description=(
        "多模态模型按轨迹线首尾接续识别全局顺序与相对位移，跳过 SIFT；"
        "需配置 LLM_BASE_URL / LLM_API_KEY。用于量化语义粗配准与几何精配准的差距。"),
    plan=plan_llm,
    requires_llm=True,
    skips_sift=True,
))

register(BackendSpec(
    id="llm_guided",
    label="大模型候选 + SIFT 精配准",
    description=(
        "模型按轨迹线识别相邻顺序、圈定候选图对，变换仍由 SIFT+RANSAC 求；"
        "测的是语义先验能否缩小搜索空间。需配置大模型服务。"),
    plan=plan_llm_guided,
    requires_llm=True,
    skips_sift=False,
))
