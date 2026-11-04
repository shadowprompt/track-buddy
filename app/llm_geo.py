"""大模型的语义层：地理常识推理。

分工（实测得出，见 docs/HANDOVER.md 22.17）
--------------------------------------------
* **坐标层**（代码 + OSM + 地理编码服务）：精确几何、DP 对齐、起终点锚定。
  大模型提供不了 OSM 级别的几何。
* **语义层**（大模型）：路名序列合不合理、顺序对不对、哪条路可能记错、
  截图的缩放级别、底图上还有哪些地名。

一个实测结论决定了这里的做法：

    高德「步行路径规划」规划全程，14 步里只命中用户路名 **1 步**
    （它走枫林路/建国西路/华山路/延安西路/天山路——那是「步行最短路」，
    不是用户实际骑的路线）。

所以**不能**用外部路径规划代替用户的语义输入；反过来，
用户的路名序列恰恰是最强的语义信号——它是「用户本人」提供的，
比任何服务都准。

安全设计
--------
LLM 的输出**只作建议**，必须再通过几何验证（DP 对齐总代价）才被采纳：

    采纳新顺序  ⟺  几何代价(new) < 几何代价(old)

这样即使模型判断错，几何证据会把它挡下来。
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable

from .llm import LLMClient, LLMResult

# 路名语义校验的提示词
_SYSTEM = """你是城市路网的地理常识专家，熟悉中国的街道命名与路网结构。
你的任务：判断用户口述的骑行路名序列是否合理。

严格遵守：
1. **只能重排用户给出的路名**，绝不新增或改写路名。
2. 不要臆造路名是否存在——只根据你对路网结构的常识判断。
3. 你的判断是**建议**，下游有几何算法验证，错了会被挡下。
4. 只输出 JSON，不要任何解释文字。"""

_SCHEMA = """输出 JSON，结构：
{
  "order": ["路名1", "路名2", ...],   // 重排后的完整序列（元素与输入完全一致）
  "changed": true/false,             // 是否与输入顺序不同
  "notes": ["简短说明，每条一句"],
  "warnings": ["可能记错或不在该城市的路名"],
  "direction": "从东南到西北"           // 整条路线的走向判断（若能判断）
}"""


def check_road_sequence(
    client: LLMClient,
    names: list[str],
    city: str = "",
    cost_fn: Callable[[list[str]], float] | None = None,
) -> dict[str, Any]:
    """用大模型校验路名序列（可选增强）。

    参数
    ----
    names     用户给出的路名（骑行顺序）
    city      城市名
    cost_fn   几何代价函数：给定一个顺序，返回 DP 对齐的总代价。
              提供时，LLM 建议的顺序必须**代价更低**才会被采纳。

    返回
    ----
    {
      "ok": bool,            # 是否拿到可用结果
      "applied": bool,       # 是否真的改了顺序
      "names": [...],        # 最终采用的顺序
      "notes": [...],
      "warnings": [...],
      "llm": {...},          # 原始响应摘要
    }
    """
    out: dict[str, Any] = {
        "ok": False, "applied": False, "names": list(names),
        "notes": [], "warnings": [], "llm": {},
    }
    if not names or not client.available:
        return out

    user = (
        f"城市：{city or '未知'}\n"
        f"用户口述的骑行路名（按他记忆的顺序）：\n"
        + "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))
        + "\n\n请判断："
        "\n1. 这个顺序在地理上合理吗（例如是否方向反了、"
        "是否有明显跳跃）"
        "\n2. 给出你认为更合理的顺序（只能重排，不能新增/改写）"
        "\n3. 哪些路名可能记错了或不在这个城市"
    )
    res: LLMResult = client.chat_json(_SYSTEM, user, schema_hint=_SCHEMA)
    out["llm"] = {"ok": res.ok, "error": res.error,
                  "elapsed_ms": res.elapsed_ms, "model": res.model}
    if not res.ok or not res.data:
        return out

    data = res.data
    order = data.get("order")
    notes = [str(x) for x in (data.get("notes") or [])][:6]
    warns = [str(x) for x in (data.get("warnings") or [])][:6]
    out["notes"] = notes
    out["warnings"] = warns
    out["ok"] = True

    # 校验：必须是同一组路名的重排
    if not isinstance(order, list) or len(order) != len(names):
        return out
    if sorted(map(str, order)) != sorted(map(str, names)):
        return out

    if list(map(str, order)) == list(map(str, names)):
        return out          # 没改动

    # 必须通过几何验证才采纳
    if cost_fn is None:
        return out
    try:
        old_cost = float(cost_fn(list(names)))
        new_cost = float(cost_fn([str(x) for x in order]))
    except Exception:  # noqa: BLE001
        return out
    out["llm"]["old_cost"] = round(old_cost, 1)
    out["llm"]["new_cost"] = round(new_cost, 1)
    if new_cost < old_cost:
        out["applied"] = True
        out["names"] = [str(x) for x in order]
        out["notes"].append(
            f"按大模型建议调整了路名顺序（几何代价 "
            f"{old_cost:.0f} → {new_cost:.0f}）")
    else:
        out["notes"].append(
            f"大模型建议的顺序几何代价更高（{old_cost:.0f} → {new_cost:.0f}），"
            f"已保留原顺序")
    return out


def review_zoom_level(
    client: LLMClient,
    image_path: str,
    ocr_names: list[str],
    city: str = "",
) -> dict[str, Any]:
    """让多模态模型看截图，复核缩放级别与遗漏地标（可选增强）。

    OCR 只能读文字，读不出图标与布局；多模态模型能判断
    「这是一张覆盖整个市区的图」这类整体语义，也能补出漏读的小字。
    """
    out: dict[str, Any] = {"ok": False, "notes": [], "extra_names": [],
                           "zoom": "unknown"}
    if not client.available:
        return out
    user = (
        f"这是一张地图 App 的骑行截图（城市：{city or '未知'}）。\n"
        f"OCR 读到的文字：{'、'.join(ocr_names) if ocr_names else '（无）'}\n\n"
        "请判断：\n"
        "1. 地图的缩放级别（city / district / street）——"
        "city=覆盖整个城区，street=能看到街道名与沿街细节\n"
        "2. 图上还有哪些 OCR 漏掉的地名（路名、地铁站名、地标名）\n"
        "3. 骑行轨迹的起点和终点分别靠近哪个标注"
    )
    res = client.chat_json(
        "你是地图截图分析专家，只输出 JSON。",
        user, images=[image_path],
        schema_hint=('输出 JSON：{"zoom": "city|district|street", '
                     '"extra_names": ["..."], "notes": ["..."]}'))
    out["llm"] = {"ok": res.ok, "error": res.error,
                  "elapsed_ms": res.elapsed_ms, "model": res.model}
    if not res.ok or not res.data:
        return out
    out["ok"] = True
    out["zoom"] = str(res.data.get("zoom") or "unknown")
    out["extra_names"] = [str(x) for x in (res.data.get("extra_names") or [])][:20]
    out["notes"] = [str(x) for x in (res.data.get("notes") or [])][:6]
    return out
