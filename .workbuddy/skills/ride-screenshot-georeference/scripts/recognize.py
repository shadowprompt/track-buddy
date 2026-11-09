"""骑行截图 → GeoJSON 轨迹：调用封装。

用法::

    python scripts/recognize.py <图片路径> [--city 上海] [--out 输出目录]
                                [--online] [--no-chart] [--no-road-verify]

脚本会自动定位底层项目并调用其 CLI，把 CLI 输出的 JSON 摘要原样打印到 stdout。

项目定位顺序：
  1. 环境变量 MEITUAN_RIDE_HOME（开发时指向代码仓库）
  2. **技能自带的内核**（assets/ride_core/）—— 技能包自带识别代码，
     复制到任何机器都能独立运行
  3. 技能目录上溯三级 —— 技能装在 <仓库>/.workbuddy/skills/ 下时，
     上溯三级正好是仓库根目录（那里就有 src/app/）
  4. 常见安装位置 D:\\Dev\\meituan-ride

也可用 --project 显式指定。

⚠️ **两种内核布局**：
  * 仓库模式   —— app 包在 ``<root>/src/app/``，需把 ``<root>/src`` 加进 PYTHONPATH；
  * 技能自带   —— app 包在 ``<root>/app/``（vendor 时被拍平），直接用 ``<root>``。
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
    os.environ.get("MEITUAN_RIDE_HOME", ""),
    _BUNDLED,
    os.path.join(_SKILL_DIR, "..", "..", ".."),      # 仓库根目录
    r"D:\Dev\meituan-ride",
    "/d/Dev/meituan-ride",
]


def find_project() -> str | None:
    """返回含 app/cli.py 的目录，找不到返回 None。

    兼容两种布局：仓库模式 ``<root>/src/app/cli.py`` 与
    技能自带的 vendor 模式 ``<root>/app/cli.py``。
    """
    for cand in _CANDIDATES:
        if not cand:
            continue
        path = os.path.abspath(cand)
        if os.path.isfile(os.path.join(path, "app", "cli.py")):
            return path
        if os.path.isfile(os.path.join(path, "src", "app", "cli.py")):
            return path
    return None


def _cli_layout(project: str) -> tuple[str, str]:
    """返回 (app 包所在目录, 需要加进 PYTHONPATH 的目录)。

    仓库模式：app 在 ``<root>/src/app``，PYTHONPATH 加 ``<root>/src``。
    vendor 模式：app 在 ``<root>/app``，PYTHONPATH 加 ``<root>``。
    """
    if os.path.isfile(os.path.join(project, "app", "cli.py")):
        return project, project
    return os.path.join(project, "src"), os.path.join(project, "src")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="recognize.py",
        description="从骑行 App 截图识别轨迹并输出 GeoJSON")
    ap.add_argument("image", help="截图路径")
    ap.add_argument("--city", default="上海", help="城市提示")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--online", action="store_true", help="允许在线地理编码")
    ap.add_argument("--no-chart", action="store_true", help="不生成轨迹图")
    ap.add_argument("--no-road-verify", action="store_true", help="不生成路网验证图")
    ap.add_argument("--project", default=None, help="底层项目根目录")
    args = ap.parse_args()

    project = os.path.abspath(args.project) if args.project else find_project()
    has_cli = (project and os.path.isfile(os.path.join(project, "app", "cli.py")))
    if not has_cli and project:
        has_cli = os.path.isfile(os.path.join(project, "src", "app", "cli.py"))
    if not project or not has_cli:
        print(json.dumps({
            "ok": False,
            "error": "找不到底层项目（meituan-ride）。"
                     "请用 --project 指定，或设置环境变量 MEITUAN_RIDE_HOME。",
            "searched": _CANDIDATES,
        }, ensure_ascii=False))
        return 2

    if not os.path.exists(args.image):
        print(json.dumps({"ok": False, "error": "图片不存在：%s" % args.image},
                         ensure_ascii=False))
        return 2

    python = shutil.which("python") or sys.executable
    _, pypath = _cli_layout(project)
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

    # app 包在 src/app/（仓库模式）或 app/（vendor 模式），
    # 都要把它所在的目录放进 PYTHONPATH 才能 `python -m app.cli`。
    env = dict(os.environ)
    env["PYTHONPATH"] = pypath + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

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
