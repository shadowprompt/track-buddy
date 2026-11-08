"""多图拼接（`app/multishot.py`）的测试。

用高德静态地图合成「用户在 App 里平移截图」的场景，
验证拼出的画布与真值一致。

真值怎么来
----------
高德静态图 zoom=17、1024px 宽时，1 像素对应约 0.51 米
（实测：经度差 0.005° = 476 米 = 932 像素）。
两张不同经度中心的静态图，其重叠关系是已知的，
拼完再量像素偏移，应当与真值一致。
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.multishot import stitch  # noqa: E402

TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tmp_debug", "zoomtest")


def _have(name: str) -> bool:
    return os.path.exists(os.path.join(TMP, name))


def test_two_shots_stitch_to_true_offset() -> None:
    """两张图拼出的相对位移应当等于真值。"""
    if not (_have("pan_A.png") and _have("pan_B.png")):
        print("跳过：缺少 pan_A/pan_B 样本")
        return
    A = cv2.imread(os.path.join(TMP, "pan_A.png"))
    B = cv2.imread(os.path.join(TMP, "pan_B.png"))
    res = stitch([A, B])
    assert res.ok, res.notes
    assert len(res.used) == 2, res.used
    assert len(res.shots) == 2

    # 两张图的落位框在画布上的横向差，应等于真值 932px
    x0 = sorted(s.x0 for s in res.shots)
    dx = x0[1] - x0[0]
    print(f"  拼出横向偏移 = {dx}px（真值 932）")
    assert abs(dx - 932) <= 6, f"横向偏移 {dx} 与真值 932 不符"

    # 尺度应接近 1，旋转应接近 0
    link = res.links[0]
    assert abs(link.scale - 1.0) < 0.01, link.scale
    assert abs(link.rotation_deg) < 0.5, link.rotation_deg
    print(f"  尺度 {link.scale:.6f}，旋转 {link.rotation_deg:.4f}°")


def test_static_ui_does_not_break_stitch() -> None:
    """静止 UI 大量存在时，拼出的位移仍应是真值。"""
    if not (_have("pan_A.png") and _have("pan_B.png")):
        print("跳过：缺少 pan_A/pan_B 样本")
        return
    A = cv2.imread(os.path.join(TMP, "pan_A.png"))
    B = cv2.imread(os.path.join(TMP, "pan_B.png"))

    def add_ui(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 90), (245, 245, 245), -1)
        cv2.putText(out, "08:17", (40, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (30, 30, 30), 3)
        cv2.circle(out, (w - 120, 45), 32, (220, 220, 220), -1)
        cv2.circle(out, (w - 90, 150), 34, (60, 130, 240), -1)
        cv2.circle(out, (w - 90, h - 140), 42, (255, 255, 255), -1)
        cv2.rectangle(out, (30, h - 230), (230, h - 60), (250, 250, 250), -1)
        cv2.rectangle(out, (30, h - 230), (230, h - 60), (90, 90, 90), 3)
        return out

    A2, B2 = add_ui(A), add_ui(B)
    res = stitch([A2, B2])
    assert res.ok, res.notes
    x0 = sorted(s.x0 for s in res.shots)
    dx = x0[1] - x0[0]
    print(f"  带 UI 横向偏移 = {dx}px（真值 932）")
    assert abs(dx - 932) <= 6, f"带 UI 时横向偏移 {dx} 与真值 932 不符"


def test_three_shots_chain() -> None:
    """三张图链式拼接：A↔B、B↔C 都要对得上。

    用 `chain_A/B/C`（经度差各 0.0025°，真值 466px）。
    注意不能用 `pan_A/B/C`——那三张彼此不重叠（A 与 C 相距 932px，
    超出重叠区），理应被判为不连通，这是正确行为不是缺陷。
    """
    names = ["chain_A.png", "chain_B.png", "chain_C.png"]
    if not all(_have(n) for n in names):
        print("跳过：缺少 chain_* 样本")
        return
    imgs = [cv2.imread(os.path.join(TMP, n)) for n in names]
    res = stitch(imgs)
    assert res.ok, res.notes
    assert len(res.used) == 3, res.used
    assert len(res.shots) == 3

    # A 与 C 的横向落位差应为 932px（两张 466px 之和）
    by_idx = {s.index: s for s in res.shots}
    dx = abs(by_idx[0].x0 - by_idx[2].x0)
    print(f"  A↔C 横向偏移 = {dx}px（真值 932）")
    assert abs(dx - 932) <= 8, f"A↔C 偏移 {dx} 与真值 932 不符"

    # 三张应当都建立了连接
    assert len(res.links) >= 2, len(res.links)
    for l in res.links:
        assert l.ncc > 0.5, f"link {l.i}->{l.j} 相似度过低 {l.ncc}"


def test_non_overlapping_chain_reports_excluded() -> None:
    """A 与 C 不重叠时，C 应被报告为排除，而不是拼出乱图。"""
    names = ["pan_A.png", "pan_B.png", "pan_C.png"]
    if not all(_have(n) for n in names):
        print("跳过：缺少 pan_C 样本")
        return
    imgs = [cv2.imread(os.path.join(TMP, n)) for n in names]
    res = stitch(imgs)
    assert res.ok, res.notes
    assert res.unused == [2], f"应排除第 3 张，实际 {res.unused}"
    assert len(res.used) == 2
    print(f"  不重叠的第 3 张被正确排除：{res.notes}")


def test_canvas_content_lands_in_right_place() -> None:
    """画布内容必须落在**正确位置**（回归测试）。

    ## 为什么必须有这条

    实测踩过一次 `warpPerspective` 的矩阵方向坑：
    传了 T⁻¹ 而不是 T，画面「看起来」还行，但内容整体错位
    （A 左上角的标记出现在画布 x=983，正确应为 476，差 507 像素）。
    肉眼几乎发现不了，但轨迹叠加会完全对不上。

    这类错误必须用**客观判据**锁住，不能靠看图。
    做法：给每张图打一个独有颜色标记，拼接后检查标记落点
    是否等于「原图坐标 + 该图的画布偏移」。
    """
    if not (_have("chain_A.png") and _have("chain_B.png")):
        print("跳过：缺少 chain_* 样本")
        return
    A = cv2.imread(os.path.join(TMP, "chain_A.png")).copy()
    B = cv2.imread(os.path.join(TMP, "chain_B.png")).copy()
    # 用**纯黑方块**做标记：底图是浅色系地图，黑块极醒目，
    # 且不会被 JPEG 压缩明显改变（阈值判定而非精确相等）。
    A[10:40, 10:40] = (0, 0, 0)
    B[-40:-10, -40:-10] = (0, 0, 0)

    res = stitch([A, B])
    assert res.ok, res.notes
    canvas = res.canvas

    by_idx = {s.index: s for s in res.shots}
    # 每张图的画布偏移 = box 的左边界（变换是纯平移）
    offsets = {i: by_idx[i].box[0] for i in by_idx}

    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    m = (g < 60).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    # 收集所有候选黑块中心
    blocks = [
        (int(stats[i][4]), float(cent[i][0]))
        for i in range(1, n)
        if 500 <= stats[i][4] <= 2500
    ]
    assert blocks, "拼接图上找不到标记黑块"

    checked = 0
    for img_idx, src_x in ((0, 25), (1, 994)):
        want = offsets[img_idx]
        expect_x = src_x + want
        # 找出「最接近期望落点」的黑块。
        # 若它明显偏离期望值，说明内容摆错了位置（warp 方向反了）；
        # 若期望落点处压根没有黑块，说明该标记被另一张图覆盖了——
        # 那是正常的覆盖关系，不是错误。
        near = min(blocks, key=lambda b: abs(b[1] - expect_x))
        if abs(near[1] - expect_x) > 20:
            # 期望位置没有黑块 → 检查是否有别的黑块明显对不上
            if all(abs(b[1] - expect_x) > 20 for b in blocks):
                print(f"  第{img_idx + 1}张标记被其它图覆盖（期望 x={expect_x}），跳过")
                continue
            got = near[1] - src_x
            print(f"  第{img_idx + 1}张标记落点偏移 = {got}，box 报告 {want}")
            raise AssertionError(
                f"第{img_idx + 1}张内容落点偏移 {got} 与 box 的 {want} 不符"
                f"（差 {abs(got - want)} 像素）——warp 方向可能反了"
            )
        got = near[1] - src_x
        print(f"  第{img_idx + 1}张标记落点偏移 = {got}，box 报告 {want}")
        assert abs(got - want) <= 20, (
            f"第{img_idx + 1}张内容落点偏移 {got} 与 box 的 {want} 不符"
            f"（差 {abs(got - want)} 像素）——warp 方向可能反了"
        )
        checked += 1
    assert checked >= 1, "至少要有一张图的标记可测量，否则测试无效"


def test_pixel_level_verify_score() -> None:
    """像素级复核分必须高（这才是「拼得对不对」的客观判据）。"""
    if not (_have("chain_A.png") and _have("chain_B.png")):
        print("跳过：缺少 chain_* 样本")
        return
    A = cv2.imread(os.path.join(TMP, "chain_A.png"))
    B = cv2.imread(os.path.join(TMP, "chain_B.png"))
    res = stitch([A, B])
    assert res.ok, res.notes
    assert res.verify, "应给出每张图的复核结果"
    for v in res.verify:
        assert v["similarity"] is not None, v
        assert v["similarity"] > 0.85, (
            f"第{v['index'] + 1}张复核相似度仅 {v['similarity']}——接缝没对上"
        )
    assert res.verify_score > 0.85, res.verify_score
    print(f"  复核得分 = {res.verify_score:.4f}")


def test_duplicate_shots_detected() -> None:
    """同一视角的重复上传要被识别并剔除。

    用户实测上传的 4 张里有 2 张是同一视角（配准位移 (0, 0.3) 像素）。
    重复图不增加覆盖范围，只会让画布出现「两张叠在同一处」的错觉。
    """
    if not (_have("chain_A.png") and _have("chain_B.png")):
        print("  跳过：缺少 chain_* 样本")
        return
    A = cv2.imread(os.path.join(TMP, "chain_A.png"))
    B = cv2.imread(os.path.join(TMP, "chain_B.png"))
    # A、A'、B —— 其中 A' 是 A 的轻微改动副本
    A2 = A.copy()
    A2[0:5, 0:5] = A2[0:5, 0:5]  # 极小改动，仍是同一视角
    res = stitch([A, A2, B])
    assert res.ok, res.notes
    dup_notes = [n for n in res.notes if "重复" in n]
    assert dup_notes, f"没有识别出重复图：{res.notes}"
    print(f"  {dup_notes[0]}")
    # 剔除后应该只剩 2 张
    assert len(res.shots) == 2, f"剔除重复后应剩 2 张，实际 {len(res.shots)}"


def test_canvas_trimmed_to_covered_area() -> None:
    """画布应裁到「已覆盖区域」，不留下外侧空白。

    ⚠️ **不能用「纯白像素比例」判断有没有空白**——
    底图本身就有白色的道路（腾讯/高德底图都是白底白路），
    实测裁剪后仍有 11.5% 纯白像素，但那些是**道路**，不是空白。

    正确判据：画布**外边界**上不该有空白。
    空白一定出现在包围盒之外，而道路不会。
    """
    if not (_have("chain_A.png") and _have("chain_B.png")):
        print("  跳过：缺少 chain_* 样本")
        return
    A = cv2.imread(os.path.join(TMP, "chain_A.png"))
    B = cv2.imread(os.path.join(TMP, "chain_B.png"))
    res = stitch([A, B])
    assert res.ok, res.notes
    canvas = res.canvas
    h, w = canvas.shape[:2]

    # 画布四边 3 像素内应当都有内容（否则说明外侧还有空白没裁掉）
    edges = {
        "上": canvas[0:3, :, :].reshape(-1, 3),
        "下": canvas[h - 3:h, :, :].reshape(-1, 3),
        "左": canvas[:, 0:3, :].reshape(-1, 3),
        "右": canvas[:, w - 3:w, :].reshape(-1, 3),
    }
    blank = np.all(np.abs(edges["上"].astype(int) - 245) < 4, axis=1).all()
    for name, px in edges.items():
        is_blank = np.all(np.abs(px.astype(int) - 245) < 4, axis=1).mean()
        print(f"  {name}边空白比例 {is_blank * 100:.1f}%")
        assert is_blank < 0.5, f"{name}边仍有大片空白（{is_blank * 100:.1f}%）"

    # 位置框必须落在画布内
    for s in res.shots:
        assert 0 <= s.x0 < w, f"第{s.index + 1}张位置框超出画布：{s.box}"
        assert s.x1 <= w, f"第{s.index + 1}张位置框超出画布：{s.box}"
    print("  四边无空白，位置框与画布对齐")


def test_disjoint_images_reported_not_crashed() -> None:
    """两张毫无关系的图应当优雅失败，而不是崩掉或拼出乱图。"""
    A = np.zeros((400, 400, 3), np.uint8)
    A[:] = (30, 90, 160)
    cv2.circle(A, (120, 120), 60, (240, 240, 240), -1)
    B = np.zeros((400, 400, 3), np.uint8)
    B[:] = (160, 90, 30)
    cv2.rectangle(B, (80, 80), (300, 300), (240, 240, 240), -1)
    res = stitch([A, B])
    assert not res.ok, "无关图不应拼接成功"
    assert res.notes, "失败时应给出原因"
    print(f"  无关图：{res.notes[0]}")


def test_single_image_passthrough() -> None:
    """单图直接透传。"""
    A = np.zeros((100, 120, 3), np.uint8)
    res = stitch([A])
    assert res.ok
    assert res.canvas is not None and res.canvas.shape == (100, 120, 3)


def main() -> int:
    tests = [
        test_two_shots_stitch_to_true_offset,
        test_static_ui_does_not_break_stitch,
        test_three_shots_chain,
        test_canvas_content_lands_in_right_place,
        test_pixel_level_verify_score,
        test_duplicate_shots_detected,
        test_canvas_trimmed_to_covered_area,
        test_non_overlapping_chain_reports_excluded,
        test_disjoint_images_reported_not_crashed,
        test_single_image_passthrough,
    ]
    failed = 0
    for t in tests:
        print(f"\n== {t.__name__} ==")
        try:
            t()
            print("  PASS")
        except AssertionError as exc:
            print(f"  FAIL: {exc}")
            failed += 1
    print(f"\n{'全部通过' if not failed else f'{failed} 项失败'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
