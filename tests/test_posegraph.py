"""位姿图全局平差：把链式累积漂移分摊到各图。

合成 6 张图的链式观测（每步带噪）+ 一条首尾闭环边，验证：
优化后边残差下降、末端绝对误差小于链式初值、根节点被固定。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import CONFIG
from app.posegraph import PoseEdge, optimize_translations

ROOT = Path(__file__).resolve().parent.parent
MULTISHOT = ROOT / "src" / "app" / "multishot.py"


def _chain_case(n=6, noise=2.5, seed=7):
    rng = np.random.default_rng(seed)
    truth = np.asarray([[100.0 * i, 40.0 * i] for i in range(n)], dtype=float)
    step = np.asarray([truth[i + 1] - truth[i] + rng.normal(0, noise, 2)
                       for i in range(n - 1)])
    edges = [PoseEdge(i, i + 1, float(d[0]), float(d[1]), weight=3.0)
             for i, d in enumerate(step)]
    # 冗余闭环边：首尾也重叠，用来抵消累积漂移
    loop = truth[-1] - truth[0] + rng.normal(0, noise, 2)
    edges.append(PoseEdge(0, n - 1, float(loop[0]), float(loop[1]), weight=5.0))
    init = np.zeros((n, 2), dtype=float)
    for i, d in enumerate(step):
        init[i + 1] = init[i] + d
    return truth, edges, init


def test_global_adjustment_reduces_drift():
    truth, edges, init = _chain_case()
    pg = optimize_translations(len(truth), edges, init, root=0)
    assert pg.ok
    assert pg.n_edges == len(edges)
    assert pg.residual_after < pg.residual_before
    assert pg.deltas[0] == (0.0, 0.0), "根节点必须固定以消除 gauge 自由度"

    fixed = init + np.asarray(pg.deltas)
    before_tail = float(np.linalg.norm(init[-1] - truth[-1]))
    after_tail = float(np.linalg.norm(fixed[-1] - truth[-1]))
    assert after_tail < before_tail * 0.6
    assert float(np.mean(np.linalg.norm(fixed - truth, axis=1))) < \
        float(np.mean(np.linalg.norm(init - truth, axis=1)))


def test_insufficient_edges_or_nodes_are_noop():
    assert not optimize_translations(2, [PoseEdge(0, 1, 1, 1)]).ok
    assert not optimize_translations(4, [PoseEdge(0, 1, 1, 1)]).ok
    r = optimize_translations(4, [PoseEdge(0, 1, 1, 1)])
    assert r.deltas == [(0.0, 0.0)] * 4


def test_weighted_solution_prefers_reliable_edges():
    edges = [
        PoseEdge(0, 1, 100.0, 0.0, weight=1.0),
        PoseEdge(1, 2, 100.0, 0.0, weight=1.0),
        PoseEdge(2, 3, 100.0, 0.0, weight=1.0),
        PoseEdge(0, 3, 320.0, 0.0, weight=50.0),
    ]
    pg = optimize_translations(4, edges, np.zeros((4, 2)), root=0)
    assert pg.ok
    fixed = np.asarray(pg.deltas)
    # 链式三边合计 300，高权重闭环边给 320 → 解应明显偏向闭环边
    assert fixed[3, 0] > 310.0


def test_invalid_init_positions_are_rejected():
    edges = [PoseEdge(0, 1, 1, 0), PoseEdge(1, 2, 1, 0),
             PoseEdge(0, 2, 2, 0)]
    r = optimize_translations(3, edges, [[0, 0], [1, 0]], root=0)
    assert not r.ok and "维度" in r.reason


def test_to_dict_and_default_switch(monkeypatch):
    monkeypatch.delenv("POSE_GRAPH_ENABLED", raising=False)
    truth, edges, init = _chain_case()
    d = optimize_translations(len(truth), edges, init).to_dict()
    assert set(d) >= {"ok", "deltas", "residual_before",
                      "residual_after", "n_edges", "n_nodes"}
    assert CONFIG.pose_graph_enabled is False


def test_switch_can_be_enabled_by_env(monkeypatch):
    monkeypatch.setenv("POSE_GRAPH_ENABLED", "1")
    assert CONFIG.pose_graph_enabled is True


def test_stitch_wiring_is_optional_and_reported():
    src = MULTISHOT.read_text(encoding="utf-8")
    assert "pose_graph: bool | None = None" in src
    assert "from .posegraph import PoseEdge, optimize_translations" in src
    assert "res.pose_graph = pg.to_dict()" in src
    assert '"pose_graph": self.pose_graph' in src
    assert "use_pg = CONFIG.pose_graph_enabled if pose_graph is None" in src
