"""多图拼接核对页（`app/stitchcheck.py`）的测试。

## 为什么要有这个页面

用户反馈「我也不知道你拼的对不对」。光给数字不够——
必须把「拼接图 + 每张原图的位置框 + 轨迹」按**同一坐标系**叠在一起，
用户才能一眼看出哪里不对。

所以这个测试锁三件事：
1. 页面能从 `last_stitch.json` 读到最近一次多图结果；
2. 三样东西用的是**同一套坐标**（画布坐标，不是各自原图坐标）；
3. 位置框、轨迹、锚点三类叠加都画得出来。

第 2 条最容易出错——之前前端就因为底图用原图、轨迹用画布坐标而整体错位。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from app.stitchcheck import load_stitch_result, render_stitch_check  # noqa: E402


def test_render_contains_all_layers() -> None:
    """页面必须包含三类叠加：位置框、轨迹、锚点。"""
    payload = {
        "stitch": {
            "ok": True,
            "canvas_size": [828, 1024],
            "used": [0, 1, 2],
            "unused": [],
            "shots": [
                {"index": 0, "box": [0, 0, 700, 1024]},
                {"index": 1, "box": [190, 0, 890, 1024]},
                {"index": 2, "box": [380, 0, 1080, 1024]},
            ],
            "verify": [{"index": 0, "similarity": 1.0, "pixels": 100}],
            "verify_score": 0.9515,
        },
        "stitched_url": "/api/shots/x_stitched.png",
        "track_polyline_px": [[10, 20], [30, 40], [50, 60]],
        "anchors_used": [{"name": "东安路", "px": 100, "py": 200}],
        "track_points": 3,
        "distance_m": 294.0,
    }
    html = render_stitch_check(payload)
    assert "__STITCH_DATA__" not in html, "占位符没替换"
    assert "DATA.shots" in html, "没有画位置框"
    assert "track_polyline_px" in html, "没有画轨迹"
    assert "anchors_used" in html, "没有画锚点"
    assert "stitched_url" in html, "没有引用拼接图"
    # 三类数据都要进到页面里
    assert "东安路" in html, "锚点名没进去"
    assert "0.9515" in html, "复核分没进去"
    print("  三类叠加都在")


def test_shots_boxes_are_in_canvas_coords() -> None:
    """位置框必须是**画布坐标**，不是各自原图坐标。

    这是之前踩过的坑：底图用原图、轨迹用画布坐标 → 整体错位。
    位置框与轨迹必须能画在同一张画布上。
    """
    payload = {
        "stitch": {
            "canvas_size": [828, 1024],
            "shots": [{"index": 0, "box": [467, 1, 1491, 1025]}],
            "used": [0],
            "verify_score": 1.0,
        },
        "track_polyline_px": [[491, 25], [600, 300]],
    }
    html = render_stitch_check(payload)
    # 位置框与轨迹都用 sx/sy 从各自的原始坐标换算，
    # 页面里必须存在这两个换算变量
    assert "sx" in html and "sy" in html, "缺少画布坐标换算"
    assert "s.box[0] * sx" in html, "位置框没有按画布坐标换算"
    assert "p[0]*sx" in html, "轨迹没有按画布坐标换算"
    print("  位置框与轨迹用同一套换算")


def test_load_stitch_result_reads_file() -> None:
    """能从 last_stitch.json 读到最近一次多图结果。

    ⚠️ 这个测试会**写** `last_stitch.json`（用临时内容再还原），
    否则与 `test_multishot_e2e.py` 并发跑时会被对方覆盖，导致偶发失败。
    """
    import shutil
    from app.config import DATA_DIR

    p = os.path.join(str(DATA_DIR), "last_stitch.json")
    backup = None
    if os.path.exists(p):
        backup = p + ".bak"
        shutil.copy2(p, backup)
    try:
        payload = {
            "stitch": {
                "ok": True,
                "canvas_size": [828, 1024],
                "used": [0, 1, 2],
                "unused": [],
                "shots": [
                    {"index": 0, "box": [0, 0, 700, 1024]},
                    {"index": 1, "box": [190, 0, 890, 1024]},
                    {"index": 2, "box": [380, 0, 1080, 1024]},
                ],
                "verify": [{"index": 0, "similarity": 1.0, "pixels": 100}],
                "verify_score": 0.9515,
            },
            "stitched_url": "/api/shots/x_stitched.png",
            "track_polyline_px": [[10, 20], [30, 40]],
            "track_points": 2,
            "distance_m": 294.0,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        got = load_stitch_result(None)
        assert got, "没读到结果"
        assert "stitch" in got, "结果里没有 stitch"
        assert len(got["stitch"]["shots"]) == 3, "位置框数量不对"
        print(f"  读到 {len(got['stitch']['shots'])} 张图的位置框")
    finally:
        # 还原
        if backup and os.path.exists(backup):
            shutil.move(backup, p)
        elif os.path.exists(p):
            os.remove(p)


def test_missing_result_returns_empty() -> None:
    """没有结果时返回空 dict，而不是抛异常。"""
    got = load_stitch_result("__nonexistent_shot_id__")
    assert isinstance(got, dict)
    print("  无结果时返回空 dict")


def main() -> int:
    tests = [
        test_render_contains_all_layers,
        test_shots_boxes_are_in_canvas_coords,
        test_load_stitch_result_reads_file,
        test_missing_result_returns_empty,
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
