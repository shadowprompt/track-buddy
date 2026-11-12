"""可插拔拼接后端的回归测试（A/B 对比的基础设施）。

三条不变量：

1. **默认路径不变**：不传新参数时仍是原来的全组合 SIFT；
2. **后端差异只在「边的来源」**：连通块 / 位姿树 / 贴图 / 逐像素复核 /
   徽章守恒共用同一套代码，所以 `verify` 可直接横向比较；
3. **绝不静默回退**：没配大模型、模型失败、输出为空，都必须明确报错，
   不能偷偷用 SIFT 出一张看起来正常的画布污染对比数据。

大模型调用全部用 StubClient，测试不依赖网络、不需要 API key。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import app.multishot as ms
from app.llm import LLMResult
from app.multishot import PairLink, _pose_tree_weight, stitch
from app.stitch_backends import (
    StitchBackendError,
    UnknownStitchBackend,
    get_backend,
    list_backends,
    run_stitch,
)
from app.stitch_backends.llm import (
    _links_from_pairs,
    _parse_pairs,
    plan_llm,
    plan_llm_guided,
)

CFG = SimpleNamespace(llm_enabled=True)
SHIFT = 200
W, H = 400, 240


class StubClient:
    """假的多模态客户端：只记录调用，不联网。"""

    available = True
    model = "stub-vlm"

    def __init__(self, payload=None, ok=True, error=""):
        self.payload = payload or {}
        self.ok = ok
        self.error = error
        self.calls: list[dict] = []

    def chat_json(self, system, user, images=None, schema_hint=""):
        self.calls.append({
            "images": list(images or []),
            "user": user, "system": system, "schema_hint": schema_hint,
        })
        r = LLMResult(ok=self.ok, model=self.model, elapsed_ms=12)
        if self.ok:
            r.data = self.payload
            r.text = '{"pairs": []}'
        else:
            r.error = self.error
        return r


def _two_shots() -> list[np.ndarray]:
    """图 2 是图 1 向右平移 200px 后的视野（同一张底图的两个裁剪）。"""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, (H, W + SHIFT, 3), dtype=np.uint8)
    return [base[:, 0:W].copy(), base[:, SHIFT:SHIFT + W].copy()]


def _payload() -> dict:
    return {"pairs": [
        {"i": 1, "j": 2, "dx_ratio": SHIFT / W,
         "dy_ratio": 0.0, "confidence": 0.8},
    ], "notes": ["右侧相邻"]}


def _links(images) -> dict[tuple[int, int], PairLink]:
    M = np.array([[1.0, 0.0, -SHIFT], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return {(0, 1): PairLink(
        i=0, j=1, matrix=M, inliers=0, good=0, ncc=0.0,
        weight=0.8, source="llm",
    )}


# ------------------------------------------------------------ 注册表


def test_registry_lists_three_backends_in_order() -> None:
    specs = list_backends(cfg=CFG, client=StubClient())
    assert [b["id"] for b in specs] == ["sift", "llm", "llm_guided"]
    assert specs[0]["available"] is True
    assert specs[1]["skips_sift"] is True
    assert specs[2]["skips_sift"] is False
    assert get_backend("SIFT ").id == "sift"


def test_unknown_backend_raises() -> None:
    with pytest.raises(UnknownStitchBackend):
        get_backend("magic")


def test_llm_unavailable_raises_without_calling_model() -> None:
    class NoLLM:
        available = False
        model = ""

        def chat_json(self, *a, **k):
            raise AssertionError("不可用时不应调用模型")

    with pytest.raises(StitchBackendError) as ei:
        run_stitch(_two_shots(), backend="llm", cfg=CFG, client=NoLLM())
    assert "LLM_BASE_URL" in str(ei.value)


# ------------------------------------------------------------ 语义输出解析


def test_parse_pairs_normalizes_and_rejects_bad_items() -> None:
    data = {"pairs": [
        {"i": 2, "j": 1, "dx_ratio": 0.75, "dy_ratio": 0.0,
         "confidence": 0.9},                       # i>j 要交换并反号
        {"i": 1, "j": 2, "dx_ratio": 0.0, "dy_ratio": 0.0,
         "confidence": 0.9},                       # 0 位移假解
        {"i": 1, "j": 9, "dx_ratio": 0.5, "confidence": 0.8},  # 越界
        {"i": 1, "j": 2, "dx_ratio": 9.0, "confidence": 0.8},  # 超范围
        "garbage",                                  # 非对象
    ], "notes": ["ok"]}
    pairs, rejected, notes = _parse_pairs(data, 2)
    assert len(pairs) == 1 and rejected == 4
    assert (pairs[0]["i"], pairs[0]["j"]) == (0, 1)
    assert pairs[0]["dx"] == pytest.approx(-0.75)
    assert pairs[0]["confidence"] == 0.9
    assert notes == ["ok"]


def test_links_matrix_direction_matches_canvas_convention() -> None:
    """j 在 i 右侧 SHIFT 像素 → matrix.tx 必须是 -SHIFT。

    方向用 2026-11-12 真实落盘结果校准：link(0→1).tx=+866，
    画布上图 1 比图 0 左移 867px。
    """
    imgs = _two_shots()
    links = _links_from_pairs(imgs, [
        {"i": 0, "j": 1, "dx": SHIFT / W, "dy": 0.0, "confidence": 0.8},
    ])
    link = links[(0, 1)]
    assert link.matrix[0, 2] == pytest.approx(-SHIFT)
    assert link.matrix[1, 2] == 0.0
    assert link.inliers == 0 and link.ncc == 0.0
    assert link.weight == 0.8 and link.source == "llm"


# ------------------------------------------------------------ 两种 LLM 后端


def test_plan_llm_produces_semantic_links_and_diagnostics() -> None:
    client = StubClient(_payload())
    plan = plan_llm(_two_shots(), cfg=CFG, client=client)
    assert plan.error == ""
    assert plan.links is not None and len(plan.links) == 1
    assert plan.diagnostics["pairs_accepted"] == 1
    assert plan.diagnostics["pairs_rejected"] == 0
    assert plan.diagnostics["sift_skipped"] is True
    assert len(client.calls) == 1 and len(client.calls[0]["images"]) == 2
    assert "编号 #1 到 #2" in client.calls[0]["user"]


def test_plan_llm_failure_never_falls_back() -> None:
    client = StubClient(ok=False, error="HTTP 401")
    plan = plan_llm(_two_shots(), cfg=CFG, client=client)
    assert "HTTP 401" in plan.error
    assert "回退到 SIFT" in plan.error
    assert plan.links is None


def test_plan_llm_empty_pairs_is_an_error_not_a_sift_fallback() -> None:
    client = StubClient({"pairs": [], "notes": []})
    plan = plan_llm(_two_shots(), cfg=CFG, client=client)
    assert "没有给出任何可用的相邻关系" in plan.error
    assert plan.links is None


def test_plan_llm_guided_only_narrows_search() -> None:
    client = StubClient(_payload())
    plan = plan_llm_guided(_two_shots(), cfg=CFG, client=client)
    assert plan.error == ""
    assert plan.links is None
    assert plan.pair_candidates == [(0, 1)]
    assert plan.diagnostics["sift_skipped"] is False


# ------------------------------------------------------------  stitch 集成


def test_run_stitch_llm_end_to_end_matches_known_geometry() -> None:
    imgs = _two_shots()
    st = run_stitch(imgs, backend="llm", cfg=CFG, client=StubClient(_payload()))
    assert st.ok, st.notes
    assert st.backend == "llm"
    assert st.backend_diagnostics["pairs_accepted"] == 1
    assert st.canvas is not None and st.canvas.shape[1] == W + SHIFT
    assert sorted(s.x0 for s in st.shots) == [0, SHIFT]
    assert all(v["similarity"] == 1.0 for v in st.verify)
    assert st.badge_conservation["ok"] is True
    assert st.links[0].source == "llm"
    d = st.to_dict()
    assert d["backend"] == "llm" and "backend_diagnostics" in d
    assert any("模型语义置信度" in n for n in st.notes)


def test_links_override_really_skips_sift(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("注入边时不应调用 SIFT 逐对配准")

    monkeypatch.setattr(ms, "_match_pair", boom)
    st = stitch(_two_shots(), links_override=_links(_two_shots()),
                backend="llm")
    assert st.ok, st.notes
    assert any("跳过 SIFT" in n for n in st.notes)


def test_default_sift_path_is_unchanged() -> None:
    one = stitch([_two_shots()[0]])
    assert one.ok and one.backend == "sift"
    assert one.notes == ["只有一张图，无需拼接"]
    # pair_candidates=[] 意味着「一幅重叠关系都不试」，必须如实失败
    none = stitch(_two_shots(), pair_candidates=[], backend="llm_guided")
    assert not none.ok
    assert any("找不到任何可用的重叠关系" in n for n in none.notes)


def test_pose_tree_weight_uses_semantic_confidence() -> None:
    m = np.eye(3)
    semantic = PairLink(0, 1, m, inliers=0, good=0, ncc=0.0,
                        weight=0.8, source="llm")
    geometric = PairLink(0, 1, m, inliers=314, good=0, ncc=0.5)
    assert _pose_tree_weight(semantic) == pytest.approx(0.8)
    assert _pose_tree_weight(geometric) == pytest.approx(157.0)
