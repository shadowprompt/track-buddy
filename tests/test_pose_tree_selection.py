"""位姿树选边回归：强弱边竞争时，不能让弱配准抢先决定图的落位。

事故背景（2026-11-12 09:04 实跑）：7 张细节图本应沿 x 依次展开成
0→1→2→3→4→5→6 的强链（inliers 314/549/721/965/725/490），但修复前的
BFS 按 ``link_map`` 的插入顺序「先到先得」，先用了 (0,5)/(1,4)/(3,5) 这些
inliers 只有 25/21/46 的弱边，把图 3、4、5 定位到错误区域，7 张图折叠堆在
3077px 宽的画布里，轨迹只提取到局部 2.8 km。

这里把选树逻辑抽成纯函数后做不变量测试：
**连接已定位集合与未定位集合的边，必须选置信度最高的那条。**
同时内联复现旧算法，用来证明「事故确实由选边顺序造成」。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.multishot import (  # noqa: E402
    PairLink,
    _build_pose_tree,
    _pose_tree_weight,
)


def _link(
    i: int,
    j: int,
    tx: float,
    ty: float = 0.0,
    inliers: int = 100,
    ncc: float = 0.6,
    rot_deg: float = 0.0,
) -> PairLink:
    a = np.radians(rot_deg)
    m = np.array(
        [
            [np.cos(a), -np.sin(a), tx],
            [np.sin(a), np.cos(a), ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return PairLink(i=i, j=j, matrix=m, inliers=inliers,
                    good=0, ncc=ncc, ui_px=0)


def _legacy_order_tree(link_map, root, members):
    """修复前的实现：按 dict 插入顺序 BFS，先到先得（只看跳数，不看质量）。"""
    T = {root: np.eye(3)}
    depth = {root: 0}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for (i, j), link in link_map.items():
            if i == cur and j not in depth:
                depth[j] = depth[cur] + 1
                T[j] = T[cur] @ np.linalg.inv(link.matrix)
                queue.append(j)
            elif j == cur and i not in depth:
                depth[i] = depth[cur] + 1
                T[i] = T[cur] @ link.matrix
                queue.append(i)
    return T, depth


# --- 1. 最小复现：4 张图 + 2 条抢先的弱边 --------------------------------

def _four_shot_links():
    # 与 itertools.combinations 一致的插入顺序：弱边排在强链前面
    return {
        (0, 1): _link(0, 1, 900.0, inliers=400),
        (0, 2): _link(0, 2, 80.0, inliers=20),       # 弱边：几乎重叠的假解
        (0, 3): _link(0, 3, -400.0, 700.0, inliers=21),
        (1, 2): _link(1, 2, 900.0, inliers=420),
        (2, 3): _link(2, 3, 900.0, inliers=430),
    }


def test_weak_edges_must_not_win() -> None:
    lm = _four_shot_links()
    T, depth, edges = _build_pose_tree(lm, root=0, members=[0, 1, 2, 3])

    # 只用 3 条强链边，弱边一条都不进生成树
    assert {(p, c) for p, c, _n, _w in edges} == {(0, 1), (1, 2), (2, 3)}
    assert min(w for _p, _c, _n, w in edges) > 200

    # 4 张图沿 x 依次展开，而不是折叠堆叠
    assert abs(T[1][0, 2] - (-900.0)) < 1e-6
    assert abs(T[2][0, 2] - (-1800.0)) < 1e-6
    assert abs(T[3][0, 2] - (-2700.0)) < 1e-6
    assert depth == {0: 0, 1: 1, 2: 2, 3: 3}


def test_legacy_order_tree_reproduces_the_bug() -> None:
    """同一组输入下，旧算法确实会被弱边带偏——锁住事故成因。"""
    lm = _four_shot_links()
    T, depth = _legacy_order_tree(lm, root=0, members=[0, 1, 2, 3])

    # 旧 BFS 从 root 依次吃下 (0,1) / (0,2) / (0,3)
    assert depth == {0: 0, 1: 1, 2: 1, 3: 1}
    assert abs(T[2][0, 2] - (-80.0)) < 1e-6       # 应为 -1800
    assert T[3][1, 2] < -600                     # 应为 0，被甩到下方


# --- 2. 真实事故批次的 link 参数重放 -------------------------------------

# 来自 data/last_result.json（2026-11-12 09:04，7 张细节图）的 stitch.links。
# 内联成固定样本，测试不依赖任何运行期文件。
REAL_LINKS = [
    (0, 1, 866.2, 256.9, 314, 0.4964, 0.0036),
    (0, 5, 341.6, 14.9, 25, 0.3249, 0.0153),
    (1, 2, 950.6, 146.3, 549, 0.7599, -0.0019),
    (1, 4, -392.7, -947.4, 21, 0.2125, 0.1875),
    (2, 3, 902.8, 2.8, 721, 0.5909, 0.0),
    (3, 4, 642.9, 85.1, 965, 0.3274, 0.0),
    (3, 5, 80.2, -19.6, 46, 0.5363, -0.0037),
    (3, 6, 203.2, 249.3, 38, 0.5049, -25.9127),
    (4, 5, 893.9, 91.3, 725, 0.6092, -0.0002),
    (5, 6, 905.8, 70.7, 490, 0.5354, -0.0011),
]


def _real_link_map():
    return {
        (i, j): _link(i, j, tx, ty, inliers, ncc, rot)
        for i, j, tx, ty, inliers, ncc, rot in REAL_LINKS
    }


def test_real_batch_selects_the_strong_chain() -> None:
    lm = _real_link_map()
    T, depth, edges = _build_pose_tree(lm, root=0, members=list(range(7)))

    chosen = {(p, c) for p, c, _n, _w in edges}
    assert chosen == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)}, chosen
    # 旋转 -25.9° 的错误匹配与所有 inliers<50 的弱边都没被选中
    assert min(n for _p, _c, n, _w in edges) >= 314

    xs = [T[k][0, 2] for k in T]
    assert max(xs) - min(xs) > 5000, f"画布仍被折叠：x 跨度 {max(xs) - min(xs):.0f}"
    # y 方向只允许小幅漂移，不能像事故里那样甩出 700px
    ys = [T[k][1, 2] for k in T]
    assert max(ys) - min(ys) < 900, f"y 方向漂移过大：{ys}"


def test_real_batch_legacy_tree_is_folded() -> None:
    lm = _real_link_map()
    T, _depth = _legacy_order_tree(lm, root=0, members=list(range(7)))
    xs = [T[k][0, 2] for k in T]
    ys = [T[k][1, 2] for k in T]
    # 修复前实测画布 3077×3818：x 跨度约 1817，y 被甩出约 1089
    assert max(xs) - min(xs) < 2500, f"旧算法应当折叠，实际 {xs}"
    assert max(ys) - min(ys) > 600, f"旧算法应当有明显纵向错位，实际 {ys}"


# --- 3. 边界：确定性、弱边唯一时仍要连通、其它连通块不纳入 --------------

def test_deterministic_and_group_bounded() -> None:
    lm = _real_link_map()
    # 混入一组与 root 完全不连通的边，不能被拉进树里
    lm[(7, 8)] = _link(7, 8, 100.0, inliers=999)

    a = _build_pose_tree(lm, root=0, members=list(range(7)))
    b = _build_pose_tree(lm, root=0, members=list(range(7)))
    assert [tuple(np.round(a[0][k].ravel(), 6)) for k in sorted(a[0])] == [
        tuple(np.round(b[0][k].ravel(), 6)) for k in sorted(b[0])
    ]
    assert set(a[0]) == set(range(7))
    assert {(p, c) for p, c, _n, _w in a[2]} == {
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)
    }


def test_only_weak_edges_still_connect() -> None:
    """没有强边可用时不能罢工：弱边也必须把图连起来。"""
    lm = {
        (0, 1): _link(0, 1, 300.0, inliers=20, ncc=0.2),
        (1, 2): _link(1, 2, 300.0, inliers=21, ncc=0.2),
    }
    T, depth, edges = _build_pose_tree(lm, root=0, members=[0, 1, 2])
    assert set(T) == {0, 1, 2}
    assert len(edges) == 2
    assert abs(T[2][0, 2] - (-600.0)) < 1e-6
    assert depth[2] == 2


def test_weight_matches_posegraph_formula() -> None:
    """权重口径必须与位姿图平差一致，避免两处各算各的。"""
    lk = _link(0, 1, 100.0, inliers=314, ncc=0.4964)
    assert abs(_pose_tree_weight(lk) - 314 * 0.4964) < 1e-9
    weak = _link(0, 1, 100.0, inliers=1, ncc=0.01)
    assert abs(_pose_tree_weight(weak) - 1 * 0.05) < 1e-9
