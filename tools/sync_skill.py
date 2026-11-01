"""把识别核心同步（vendor）进某个技能目录，使技能可独立运行。

用法::

    python tools/sync_skill.py <目标技能目录> [<目标技能目录> ...]

每个目标目录下会生成::

    assets/ride_core/app/...        识别核心（app 包）
    assets/ride_core/data/lexicon.json  本地地名词典
    assets/ride_core/requirements.txt   依赖清单

技能脚本 `scripts/recognize.py` 优先使用打包进来的核心，找不到才回退到
`D:\\Dev\\meituan-ride`。这样技能包复制到任何机器都能直接用。

维护约定：**代码只在本仓库改**，改完跑一次本脚本同步到各技能目录。
"""
from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_NAME = "ride_core"

SKIP_DIRS = {"__pycache__", ".git", ".idea", "out", "dist", "cache", "uploads"}
SKIP_FILES = {".DS_Store"}


def _ignore(_dir, names):
    return [n for n in names if n in SKIP_DIRS or n in SKIP_FILES]


def sync_one(target: str) -> str:
    core = os.path.join(target, "assets", CORE_NAME)
    os.makedirs(core, exist_ok=True)

    # 1) app 包
    app_dst = os.path.join(core, "app")
    if os.path.isdir(app_dst):
        shutil.rmtree(app_dst)
    shutil.copytree(os.path.join(ROOT, "app"), app_dst, ignore=_ignore)

    # 2) 词典
    data_dst = os.path.join(core, "data")
    os.makedirs(data_dst, exist_ok=True)
    lex_src = os.path.join(ROOT, "data", "lexicon.json")
    if os.path.exists(lex_src):
        shutil.copy(lex_src, os.path.join(data_dst, "lexicon.json"))

    # 3) 依赖清单
    shutil.copy(os.path.join(ROOT, "requirements.txt"),
                os.path.join(core, "requirements.txt"))

    n = sum(len(f) for _, _, f in os.walk(app_dst))
    return "%s  (%d 个源文件)" % (core, n)


def main(argv: list[str]) -> int:
    targets = argv[1:]
    if not targets:
        print(__doc__)
        return 2
    for t in targets:
        path = os.path.abspath(t)
        if not os.path.isdir(path):
            print("跳过（目录不存在）：%s" % path)
            continue
        print("已同步 %s" % sync_one(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
