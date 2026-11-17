"""运动截图 → 轨迹 GeoJSON：便携 CLI 调用封装。

用法::

    python scripts/recognize.py <图片路径> [--city 上海] [--out 输出目录]
                                [--source meituan|amap] [--online]
                                [--no-chart] [--no-road-verify]

脚本自动定位识别内核并调用其 CLI，把 JSON 摘要原样打印到 stdout。
单图路径只产 GeoJSON + 对照图；多图 / 运动概要 / GPX / TCX 走 recognize_api.py。

内核定位顺序：
  1. 环境变量 TRACK_BUDDY_HOME（开发时指向代码仓库；旧名 MEITUAN_RIDE_HOME 兼容）
  2. **技能自带的内核**（assets/ride_core/）—— 默认走这条，复制到任何机器可跑
  3. 技能目录上溯三级 —— 装在 <仓库>/.workbuddy/skills/ 下时正好是仓库根
  4. 常见安装位置 D:\\Dev\\track-buddy

也可用 --project 显式指定。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

_SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLED = os.path.join(_SKILL_DIR, "assets", "ride_core")

_CANDIDATES = [
    os.environ.get("TRACK_BUDDY_HOME", ""),
    os.environ.get("MEITUAN_RIDE_HOME", ""),   # 旧名兼容
    _BUNDLED,
    os.path.join(_SKILL_DIR, "..", "..", ".."),  # 仓库根目录
    r"D:\Dev\track-buddy",
    "/d/Dev/track-buddy",
]


def find_project() -> str | None:
    """返回含 app/cli.py 的目录，找不到返回 None。"""
    for cand in _CANDIDATES:
        if not cand:
            continue
        path = os.path.abspath(cand)
        if os.path.isfile(os.path.join(path, "app", "cli.py")):
            return path
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="recognize.py",
        description="从运动 App 截图识别轨迹并输出 GeoJSON（WGS-84）")
    ap.add_argument("image", help="截图路径")
    ap.add_argument("--city", default="上海", help="城市提示")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--source", default=None,
                    help="来源画像 id（meituan / amap），默认 meituan")
    ap.add_argument("--online", action="store_true", help="允许在线地理编码")
    ap.add_argument("--no-chart", action="store_true", help="不生成轨迹图")
    ap.add_argument("--no-road-verify", action="store_true", help="不生成路网验证图")
    ap.add_argument("--project", default=None, help="底层项目根目录")
    args = ap.parse_args()

    project = os.path.abspath(args.project) if args.project else find_project()
    if not project or not os.path.isfile(os.path.join(project, "app", "cli.py")):
        print(json.dumps({
            "ok": False,
            "error": "找不到识别内核（track-buddy）。"
                     "请用 --project 指定，或设置环境变量 TRACK_BUDDY_HOME。",
            "searched": _CANDIDATES,
        }, ensure_ascii=False))
        return 2

    if not os.path.exists(args.image):
        print(json.dumps({"ok": False, "error": "图片不存在：%s" % args.image},
                         ensure_ascii=False))
        return 2

    # 必须用运行本脚本的解释器：裸 `python` 可能指向另一个没装识别依赖
    # （cv2 / rapidocr）的环境，导致自带内核 import 失败。
    python = sys.executable
    cmd = [python, "-m", "app.cli", os.path.abspath(args.image),
           "--city", args.city]
    if args.out:
        cmd += ["--out", os.path.abspath(args.out)]
    if args.online:
        cmd.append("--online")
    if args.no_chart:
        cmd.append("--no-chart")
    if args.no_road_verify:
        cmd.append("--no-road-verify")

    # CLI 没有 --source 参数：来源画像通过 SOURCE_PROFILE 环境变量选择
    # （CONFIG.source_profile_id 读取，见 config.py）。
    env = os.environ.copy()
    if args.source:
        env["SOURCE_PROFILE"] = args.source

    proc = subprocess.run(cmd, cwd=project, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    # CLI 正常时 stdout 末行是 JSON 摘要
    summary = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if summary is None:
        print(json.dumps({
            "ok": False,
            "error": "CLI 未返回 JSON 摘要",
            "returncode": proc.returncode,
            "stdout": out[-2000:],
            "stderr": err[-2000:],
        }, ensure_ascii=False))
        return proc.returncode or 1

    summary["project"] = project
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
