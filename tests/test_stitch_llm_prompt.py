"""提示词 v2：把「按轨迹线识别全局链条」写成硬约束后的回归测试。

背景（2026-11-13 09:47 事故）：v1 提示词只说「依据道路走向、轨迹线、
地名与 POI 的连续性」，模型实际只凭吴淞江 / 北翟路 / 天山西路 /
小区名重复出现凑边，7 张图给 3 条边，dx 符号 -0.62/+0.78/-0.55/
+0.55/-0.5/-0.52 来回翻转，画布折叠成一小团。

这组测试锁住三件事：
1. 发给模型的提示词确实把轨迹线接续、全局 order、符号一致、evidence
   写成了硬约束，而不是一句泛泛的「依据画面连续性」；
2. 模型输出无论好坏，都能被解析层量化成「识别做到了什么程度」的诊断；
3. `llm_guided` 会把识别出的全局顺序转成 SIFT 候选，而不是只试模型
   零散给出的那几条边。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.llm import LLMResult
from app.stitch_backends.llm import (
    PROMPT_VERSION,
    _SYSTEM,
    _SCHEMA_HINT,
    _chain_diagnostics,
    _parse_order,
    _parse_pairs,
    _user_prompt,
    plan_llm,
    plan_llm_guided,
)

CFG = SimpleNamespace(llm_enabled=True)


class StubClient:
    """假的多模态客户端：只记录请求，不联网、不需要 API key。"""

    available = True
    model = "stub-vlm"

    def __init__(self, payload: dict, ok: bool = True, error: str = "") -> None:
        self.payload = payload
        self.ok = ok
        self.error = error
        self.calls: list[dict] = []

    def chat_json(self, system, user, images=None, schema_hint=""):
        self.calls.append({
            "system": system, "user": user,
            "images": list(images or []), "schema_hint": schema_hint,
        })
        r = LLMResult(ok=self.ok, model=self.model, elapsed_ms=12)
        if self.ok:
            r.data = self.payload
            r.text = "{}"
        else:
            r.error = self.error
        return r


def _shots(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.integers(0, 256, (48, 72, 3), dtype=np.uint8)
            for _ in range(n)]


def _accident_payload() -> dict:
    """复刻折叠事故：order 完整，但 dx 主导轴符号来回翻转。"""
    dxs = [-0.62, 0.78, -0.55, 0.55, -0.5, -0.52]
    return {
        "order": [1, 2, 3, 4, 5, 6, 7],
        "pairs": [
            {"i": k + 1, "j": k + 2, "dx_ratio": dx, "dy_ratio": 0.02,
             "confidence": 0.8,
             "evidence": f"#{k + 1}→#{k + 2} 依据同名道路凑边"}
            for k, dx in enumerate(dxs)
        ],
        "notes": "按地名凑的链",
    }


def _edge_payload(order: list[int], linked: list[tuple[int, int]]) -> dict:
    return {
        "order": order,
        "pairs": [
            {"i": i, "j": j, "dx_ratio": 0.75, "dy_ratio": 0.0,
             "confidence": 0.8, "evidence": f"#{i} 右出 → #{j} 左入"}
            for i, j in linked
        ],
        "notes": "",
    }


# ------------------------------------------------------------ 提示词契约


def test_prompt_makes_trajectory_chain_a_hard_constraint() -> None:
    system = _SYSTEM
    assert "编号只是上传顺序" in system
    assert "唯一的主证据：轨迹线" in system
    assert "出边" in system and "入边" in system
    assert "禁止只因为两图出现相同地名" in system
    assert "起点" in system and "终点" in system
    assert "必须同号" in system

    user = _user_prompt(7)
    assert "编号 #1 到 #7" in user
    assert "1..7 的**排列**" in user
    assert "理想为 6 条" in user
    assert "evidence" in user and "每条边必填" in user
    assert "输出前逐条自检" in user
    assert "order、pairs、notes" in _SCHEMA_HINT
    assert PROMPT_VERSION == "trajectory-chain-v2"


def test_system_prompt_is_actually_sent_to_model() -> None:
    client = StubClient(_edge_payload([1, 2], [(1, 2)]))
    plan_llm(_shots(2), cfg=CFG, client=client)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert "轨迹线" in call["system"]
    assert "evidence" in call["user"]
    assert len(call["images"]) == 2


# ------------------------------------------------------------ order 解析


def test_parse_order_accepts_permutation_and_tolerates_strings() -> None:
    assert _parse_order({"order": [3, 1, 2]}, 3) == ([3, 1, 2], "")
    assert _parse_order({"order": ["2", "1"]}, 2)[0] == [2, 1]


def test_parse_order_rejects_missing_duplicate_and_absent() -> None:
    order, err = _parse_order({}, 3)
    assert order == [] and "未输出" in err
    order, err = _parse_order({"order": [1, 1, 3]}, 3)
    assert order == [] and "排列" in err
    order, err = _parse_order({"order": [1, 2]}, 3)
    assert order == [] and "排列" in err
    order, err = _parse_order({"order": [1, "x", 3]}, 3)
    assert order == [] and "非整数" in err


# ------------------------------------------------------------ 边的解析


def test_parse_pairs_keeps_evidence_and_normalizes_direction() -> None:
    data = {"pairs": [
        {"i": 2, "j": 1, "dx_ratio": 0.8, "dy_ratio": 0.0,
         "confidence": 0.9, "evidence": "  右侧出→左侧入  "},
    ], "notes": "ok"}
    pairs, rejected, notes = _parse_pairs(data, 2)
    assert len(pairs) == 1 and rejected == 0 and notes == ["ok"]
    assert (pairs[0]["i"], pairs[0]["j"]) == (0, 1)
    assert pairs[0]["dx"] == pytest.approx(-0.8)
    assert pairs[0]["evidence"] == "右侧出→左侧入"


def test_parse_pairs_without_evidence_is_empty_string() -> None:
    pairs, _, _ = _parse_pairs(
        {"pairs": [{"i": 1, "j": 2, "dx_ratio": 0.75}]}, 2)
    assert pairs[0]["evidence"] == ""


# ------------------------------------------------------------ 识别体检


def test_chain_diagnostics_flags_the_real_sign_flip_accident() -> None:
    pairs, rejected, _ = _parse_pairs(_accident_payload(), 7)
    assert rejected == 0 and len(pairs) == 6
    info = _chain_diagnostics(pairs, [1, 2, 3, 4, 5, 6, 7], 7)
    assert info["dominant_axis"] == "x"
    assert info["direction_conflict_count"] == 2
    assert {c["value"] for c in info["direction_conflicts"]} == {0.78, 0.55}
    # 边是齐的、图是连的：问题纯粹出在模型给的位移方向自相矛盾
    assert info["chain_complete"] is True
    assert info["connected_components"] == 1
    assert info["missing_images"] == []
    assert len(info["evidence"]) == 6


def test_chain_diagnostics_reports_incomplete_coverage() -> None:
    pairs, _, _ = _parse_pairs(
        _edge_payload([1, 2, 3, 4, 5, 6, 7], [(1, 2), (4, 5)]), 7)
    info = _chain_diagnostics(pairs, [1, 2, 3, 4, 5, 6, 7], 7)
    assert info["chain_complete"] is False
    assert info["connected_components"] == 5
    assert info["missing_images"] == [3, 6, 7]
    assert info["pairs_expected"] == 6


def test_chain_diagnostics_flags_off_chain_pairs() -> None:
    pairs, _, _ = _parse_pairs(_edge_payload([1, 2, 3, 4], [(1, 3)]), 4)
    info = _chain_diagnostics(pairs, [1, 2, 3, 4], 4)
    assert info["off_chain_pairs"] == [[1, 3]]
    assert info["order_valid"] is True


def test_chain_diagnostics_marks_invalid_order() -> None:
    pairs, _, _ = _parse_pairs(_edge_payload([1, 2, 3, 4], [(1, 2)]), 4)
    info = _chain_diagnostics(pairs, [], 4, "order 含非整数项")
    assert info["order_valid"] is False
    assert info["order_error"] == "order 含非整数项"
    assert info["chain_complete"] is False
    assert info["off_chain_pairs"] == []


def test_chain_diagnostics_with_consistent_chain_has_no_conflict() -> None:
    pairs, _, _ = _parse_pairs(
        _edge_payload([3, 1, 2], [(3, 1), (1, 2)]), 3)
    info = _chain_diagnostics(pairs, [3, 1, 2], 3)
    assert info["direction_conflict_count"] == 0
    assert info["chain_complete"] is True
    assert info["off_chain_pairs"] == []


# ------------------------------------------------------------ 两种后端


def test_plan_llm_exposes_recognition_diagnostics_and_warnings() -> None:
    client = StubClient(_accident_payload())
    plan = plan_llm(_shots(7), cfg=CFG, client=client)
    assert plan.error == "" and plan.links is not None and len(plan.links) == 6
    d = plan.diagnostics
    assert d["prompt_version"] == PROMPT_VERSION
    assert d["order"] == [1, 2, 3, 4, 5, 6, 7]
    assert d["order_valid"] is True
    assert d["direction_conflict_count"] == 2
    assert d["chain_complete"] is True
    assert d["sift_skipped"] is True
    assert any("方向" in n and "⚠" in n for n in plan.notes)
    assert len(client.calls[0]["images"]) == 7


def test_plan_llm_without_order_still_works_but_warns() -> None:
    """老格式输出（只有 pairs）不能被新校验打死，但要点明顺序缺失。"""
    payload = {"pairs": [
        {"i": 1, "j": 2, "dx_ratio": 0.75, "dy_ratio": 0.0,
         "confidence": 0.8},
    ]}
    plan = plan_llm(_shots(2), cfg=CFG, client=StubClient(payload))
    assert plan.error == "" and len(plan.links) == 1
    assert plan.diagnostics["order_valid"] is False
    assert any("全局顺序无效" in n for n in plan.notes)


def test_plan_llm_guided_fills_candidates_from_recognized_order() -> None:
    """模型只识别顺序、没给全边时，guided 必须把相邻对交给 SIFT。"""
    payload = _edge_payload([1, 2, 3, 4], [(1, 2)])
    plan = plan_llm_guided(_shots(4), cfg=CFG, client=StubClient(payload))
    assert plan.error == ""
    assert plan.pair_candidates == [(0, 1), (1, 2), (2, 3)]
    assert plan.diagnostics["candidates_from_order"] == 2
    assert any("由 order 补齐" in n for n in plan.notes)


def test_plan_llm_guided_without_order_keeps_original_behavior() -> None:
    payload = {"pairs": [
        {"i": 1, "j": 2, "dx_ratio": 0.75, "dy_ratio": 0.0,
         "confidence": 0.8},
    ]}
    plan = plan_llm_guided(_shots(2), cfg=CFG, client=StubClient(payload))
    assert plan.pair_candidates == [(0, 1)]
    assert plan.diagnostics["candidates_from_order"] == 0


def test_model_failure_keeps_prompt_version_in_diagnostics() -> None:
    plan = plan_llm(_shots(2), cfg=CFG,
                    client=StubClient({}, ok=False, error="HTTP 401"))
    assert "HTTP 401" in plan.error
    assert plan.diagnostics["prompt_version"] == PROMPT_VERSION
    assert plan.links is None
