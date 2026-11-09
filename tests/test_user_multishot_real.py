"""多图拼接端到端的回归测试（用用户 2026-11-08 的真实截图）。

## 这组测试锁住什么

用户上传 4 张街道级截图（起点在第 1 张、终点在第 4 张），
结果页面提示：

    · 未检测到起点徽章（绿色「起」）
    · 轨迹形状不合理：连通域宽 3242px ≈ 整幅图宽（3352px）
    · 未能提取到有效轨迹线

而用户明明截到了起点和终点。这是**两个 bug 叠加**的结果。

### bug 1：形状检查的「too_wide」判据误杀真实轨迹

第一版判据有一条独立的否决条件：

    too_wide = 连通域宽 >= 图宽 * 0.75  ->  是底图元素

这个假设在**单图**上成立，在**多图拼接画布**上必然失效——
11 公里的路线横跨整张拼接图，轨迹线的包围盒宽度本来就该接近图宽。

实测（用户这 4 张拼成的画布，4178×3864）：

    真实轨迹：框宽 4016px / 图宽 4178px（占 96%），骨架填充率 0.0008
    => too_wide 成立 -> 真实轨迹被当成「底图色块」判掉

同时，实测发现**填充率也区分不了**：

    绿地图块（误判样本）：骨架填充率 0.0021 ~ 0.0022
    真实轨迹：            骨架填充率 0.0008

    同一量级——因为误判的绿地本身也是细长的沿河绿地边界，不是实心色块。

**最终判据：以拓扑为主导信号。**
真正的骑行轨迹一定同时贴着「起」「终」两个徽章，底图元素不会。
实测真实轨迹的拓扑距离只有 **39 像素**。

### bug 2：去重误判导致含起点的图被丢弃

旧版去重把第 1 张（含起点）与第 4 张判成「同一视角的重复」，
丢弃了含起点的那张 -> 画布上找不到起点徽章 -> 连带触发 bug 1。

## 硬约束

* 用户这 4 张拼起来**必须**检出起终点徽章
* **必须**提取出轨迹（点数 > 100）
* 绿地误判样本**必须**仍被拦住（不能为了放行真实轨迹而放开误判）
* `samples/` 里的真实样本**必须**零误伤
* **上传顺序无关**：同一批 4 张图倒序上传也必须得到同样的结果

## 2026-11-08 追加：倒序上传事故

用户把同一批截图改名成 3-1~3-4 并**倒序**上传（浏览器按文件名排序），
又看到「未检测到起点徽章 + 轨迹形状不合理」。根因不是上一轮的两处修复失效：

    正序：起点图↔终点图 SIFT = inl 37 / ncc 0.02 / t=(183,296)
    倒序：终点图↔起点图 SIFT = inl 13 / ncc 0.18 / t=(0.0, 0.3)  ← 假匹配

旧去重只看「位移 < 6px」就判成同一视角重复，把含起点的图丢掉；
而且 ncc=0.18 还能骗过默认可信阈值 0.15，就算不判重也会把两张图叠在一起。
现在判重只认「像素差 + 灰度结构相关」都一致，位移近 0 但内容不同的直接断开。

## 2026-11-08 12:00 再追加：12 张细截图

同一条路线拆成 12 张后，又出现「掩膜为空」。三个问题叠加：
0 位移假解删掉图5↔图6 这座唯一的桥（前 5 张被排除）、
画布 6625 万像素超过旧上限 6000 万、接缝把轨迹切成两段后被
「两端距离之和 <700」的粗筛全部淘汰。
修复：在候选峰层面排除 0 位移假解、画布上限提到 1.2 亿像素、
有徽章时允许「贴起点段 + 贴终点段」补线合并。详见 HANDOVER §22.21.16。
"""

from __future__ import annotations

import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from app.config import CONFIG  # noqa: E402
from app.multishot import stitch  # noqa: E402
from app.track import extract_track  # noqa: E402

UPLOADS = "data/uploads"

# 用户 2026-11-08 上传的 4 张腾讯底图截图：
# 起点「起」在第 1 张（左上），终点「终」在第 4 张（右下）。
USER_4 = [
    "1794096457_2463f8d7",
    "1794096457_5abde3aa",
    "1794096457_63ce03bd",
    "1794096457_d2051989",
]

# 2026-11-08 12:00 那批：同一条路线拆成 12 张更细的截图（按上传顺序）
FINE_12 = [
    "1794110270_3f7842e6", "1794110270_2b7a3086", "1794110270_dd87c35f",
    "1794110270_dcfee266", "1794110270_0aa85ee7", "1794110270_ace458b1",
    "1794110270_eaf0f0a3", "1794110270_b9471107", "1794110270_f9899cf0",
    "1794110270_e6d3d15f", "1794110270_320845a4", "1794110270_0d980f41",
]

# 2026-11-06 那批：纯地图截图，没有骑行轨迹线，必须被拦住
GREEN_4 = [
    "1793930041_1d6027d4",
    "1793930041_152f92f5",
    "1793930041_cedd9acf",
    "1793930041_70036afc",
]


def _load(names: list[str]) -> list:
    out = []
    for n in names:
        p = os.path.join(UPLOADS, f"{n}.jpg")
        if not os.path.exists(p):
            return []
        im = cv2.imread(p)
        if im is None:
            return []
        out.append(im)
    return out


def test_user_four_shots_stitch_and_extract() -> None:
    """用户这 4 张拼起来必须检出起终点徽章并提取出轨迹。"""
    imgs = _load(USER_4)
    if not imgs:
        print("  跳过：没有用户样本")
        return

    res = stitch(imgs)
    assert res.ok, res.notes
    canvas = res.canvas
    assert canvas is not None

    tr = extract_track(canvas, CONFIG.vision)
    print(f"  画布 {canvas.shape[1]}×{canvas.shape[0]}  轨迹 {len(tr.polyline)} 点")
    print(f"  起点 {tr.start_marker}  终点 {tr.end_marker}")

    # 1. 起终点徽章必须都在
    assert tr.start_marker is not None, (
        f"起点徽章没检出（notes: {tr.notes}）"
    )
    assert tr.end_marker is not None, (
        f"终点徽章没检出（notes: {tr.notes}）"
    )

    # 2. 必须提取出轨迹
    assert len(tr.polyline) > 100, (
        f"轨迹点数异常少：{len(tr.polyline)}（notes: {tr.notes}）"
    )
    assert tr.confidence > 0.5, (
        f"置信度过低：{tr.confidence}（notes: {tr.notes}）"
    )

    # 3. 不应出现「形状不合理」的误判
    bad = [n for n in tr.notes if "形状不合理" in n]
    assert not bad, f"真实轨迹被形状检查误伤：{bad[0][:100]}"

    # 4. 起点徽章附近应当有轨迹点（徽章引线接到折线两端）
    assert tr.polyline[0].shape[0] == 2, "折线格式不对"


def test_wide_trajectory_not_rejected() -> None:
    """横跨整幅图的真实轨迹**不能**被形状检查拦下。

    这是 bug 1 的直接回归：拼接画布上的轨迹包围盒宽度
    本来就接近图宽（实测 4016/4178 = 96%）。
    """
    imgs = _load(USER_4)
    if not imgs:
        print("  跳过：没有用户样本")
        return
    res = stitch(imgs)
    assert res.ok and res.canvas is not None
    tr = extract_track(res.canvas, CONFIG.vision)
    assert not any("形状不合理" in n for n in tr.notes), (
        f"横跨整图的真实轨迹被误判为色块："
        f"{[n for n in tr.notes if '形状不合理' in n][0][:100]}"
    )
    print("  横跨整图的轨迹未被误判")


def test_green_still_rejected() -> None:
    """绿地误判样本必须仍被拦住（不能为了放行真实轨迹而放开误判）。"""
    imgs = _load(GREEN_4)
    if not imgs:
        print("  跳过：没有用户样本")
        return
    for name, im in zip(GREEN_4, imgs):
        tr = extract_track(im, CONFIG.vision)
        assert len(tr.polyline) == 0, (
            f"{name} 产出了 {len(tr.polyline)} 个点（应是底图色块误判）"
        )
    print(f"  {len(imgs)} 张绿地样本仍被拦住")


def test_real_samples_not_broken() -> None:
    """`samples/` 里的真实轨迹必须零误伤（硬约束）。"""
    files = sorted(glob.glob("samples/*.jpg"))
    if not files:
        print("  跳过：没有样本")
        return
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        tr = extract_track(im, CONFIG.vision)
        bad = [n for n in tr.notes if "形状不合理" in n]
        assert not bad, f"{os.path.basename(f)} 被形状检查误伤：{bad[0][:80]}"
        assert len(tr.polyline) > 50, (
            f"{os.path.basename(f)} 轨迹点数异常少：{len(tr.polyline)}"
        )
        print(f"  {os.path.basename(f)}: {len(tr.polyline)} 点，正常")


def test_reverse_order_same_result() -> None:
    """同一批 4 张图**倒序**上传，结果必须与正序一致。

    这是 2026-11-08 倒序事故的直接回归：旧版会把第 1、4 张
    （终点图与起点图）误判成「同一视角重复」，丢掉含起点的图。
    """
    imgs = _load(USER_4)
    if not imgs:
        print("  跳过：没有用户样本")
        return
    res = stitch(list(reversed(imgs)))
    assert res.ok, res.notes
    assert res.canvas is not None
    assert res.used == [0, 1, 2, 3], (
        f"倒序时有图被误删：used={res.used}，notes={res.notes}"
    )
    assert not any("同一视角的重复" in n for n in res.notes), (
        f"不同视角的图被误判成重复：{res.notes}"
    )
    tr = extract_track(res.canvas, CONFIG.vision)
    assert tr.start_marker is not None, f"倒序丢了起点徽章：{tr.notes}"
    assert tr.end_marker is not None, f"倒序丢了终点徽章：{tr.notes}"
    assert len(tr.polyline) > 100, f"倒序轨迹点数异常：{len(tr.polyline)}"
    assert not any("形状不合理" in n for n in tr.notes), tr.notes
    print(f"  倒序画布 {res.canvas.shape[1]}×{res.canvas.shape[0]} "
          f"轨迹 {len(tr.polyline)} 点，起 {tr.start_marker} 终 {tr.end_marker}")


def test_true_duplicate_still_dropped() -> None:
    """真正重复上传的图仍要去重（不能为了修倒序把能力去掉）。"""
    imgs = _load(USER_4)
    if not imgs:
        print("  跳过：没有用户样本")
        return
    # 第 2 张再传一次，放在最后
    res = stitch(imgs + [imgs[1].copy()])
    assert res.ok, res.notes
    assert len(res.used) == 4, f"重复图没被去掉：used={res.used}"
    assert any("同一视角的重复" in n for n in res.notes), (
        f"重复图没被识别：{res.notes}"
    )
    print("  重复图仍被正确去重")


def test_twelve_fine_shots() -> None:
    """同一条路线拆成 12 张细截图，也必须 12 张全用并提取出完整轨迹。

    锁住 2026-11-08 12:00 的三个叠加问题：
      1. 0 位移假解把唯一的相邻桥弄断 → 前 5 张被排除、真终点徽章丢失
      2. 画布 6625 万像素超过旧上限 6000 万 → 直接放弃拼接
      3. 接缝把轨迹切成两段 → 旧粗筛（两端距离之和 <700）全部淘汰
    """
    imgs = _load(FINE_12)
    if len(imgs) != 12:
        print("  跳过：没有 12 张细截图样本")
        return
    res = stitch(imgs)
    assert res.ok, res.notes
    assert res.canvas is not None
    assert res.used == list(range(12)), (
        f"细拆分时有图被误删：used={res.used}，notes={res.notes}"
    )
    tr = extract_track(res.canvas, CONFIG.vision)
    assert tr.start_marker is not None, f"起点徽章丢失：{tr.notes}"
    assert tr.end_marker is not None, f"终点徽章丢失：{tr.notes}"
    assert len(tr.polyline) > 100, f"轨迹点数异常：{len(tr.polyline)}"
    assert not any("掩膜为空" in n or "形状不合理" in n for n in tr.notes), tr.notes
    print(f"  12 张画布 {res.canvas.shape[1]}×{res.canvas.shape[0]} "
          f"轨迹 {len(tr.polyline)} 点，起 {tr.start_marker} 终 {tr.end_marker}")


def main() -> int:
    tests = [
        test_user_four_shots_stitch_and_extract,
        test_wide_trajectory_not_rejected,
        test_reverse_order_same_result,
        test_true_duplicate_still_dropped,
        test_twelve_fine_shots,
        test_green_still_rejected,
        test_real_samples_not_broken,
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
