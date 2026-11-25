"""目录结构回归测试：锁住 `src/` 布局的约定。

2026-11 把 ``app/`` ``web/`` ``tools/`` 移到 ``src/`` 下。这类重构
最怕的是「**路径错了但测试不报**」——比如某个脚本仍按旧层数拼路径，
在别的机器上才崩。

这组测试从**结构**上验证：

1. 三块源代码都在 ``src/`` 下，且不含残留的旧位置；
2. ``src/app/config.py`` 的 ``ROOT`` / ``SRC_ROOT`` 在**两种布局**
   （仓库模式 / 技能 vendor 模式）下都指向正确目录；
3. 没有任何源码里残留硬编码的绝对路径；
4. ``run.py`` 能直接跑起来（不需要手动设 ``PYTHONPATH``）。
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


# ------------------------------------------------------------ 结构布局


def test_source_dirs_moved_to_src() -> None:
    """三块源代码必须在 src/ 下，且根目录不再有残留。"""
    assert (SRC / "app" / "__init__.py").exists(), "app 包不在 src/ 下"
    assert (SRC / "web" / "index.html").exists(), "前端不在 src/ 下"
    assert (SRC / "tools").is_dir(), "tools 不在 src/ 下"

    for old in ("app", "web", "tools"):
        assert not (ROOT / old).exists(), \
            f"根目录仍残留 {old}/（应已移到 src/ 下）"


def test_data_and_docs_stay_at_root() -> None:
    """数据与文档不是代码，必须留在根目录。"""
    for name in ("data", "samples", "docs", "tests"):
        assert (ROOT / name).exists(), f"{name}/ 应留在仓库根"


def test_geojson_outputs_moved_to_output() -> None:
    """根目录不应再有游离的轨迹产物文件。"""
    leftovers = list(ROOT.glob("trajectory*.geojson"))
    assert not leftovers, f"根目录仍有轨迹产物: {[p.name for p in leftovers]}"


# ------------------------------------------------------- 路径常量正确性


def test_config_root_in_repo_layout() -> None:
    """仓库布局：ROOT = 仓库根，SRC_ROOT = src/。"""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import app.config as c;"
        "print(c.ROOT); print(c.SRC_ROOT); print(c.DATA_DIR)"
        % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-400:]
    root, src_root, data_dir = out.stdout.strip().splitlines()
    assert Path(root).resolve() == ROOT.resolve()
    assert Path(src_root).resolve() == SRC.resolve()
    assert Path(data_dir).resolve() == (ROOT / "data").resolve()


def test_config_root_in_vendor_layout(tmp_path: Path) -> None:
    """技能 vendor 布局：app/ 与 data/ 同级，ROOT 必须指向那一层。

    这是打包进技能包后的实际布局。若 ROOT 用固定层数推导，
    vendor 版会去读**技能包外面**的 data/，直接读不到词典。
    """
    core = tmp_path / "ride_core"
    (core / "data").mkdir(parents=True)
    (core / "state").mkdir()
    (core / "app").mkdir()
    # 只复制 config.py 与它依赖的 geo 包
    import shutil

    shutil.copytree(SRC / "app" / "geo", core / "app" / "geo")
    shutil.copy(SRC / "app" / "__init__.py", core / "app" / "__init__.py")
    shutil.copy(SRC / "app" / "config.py", core / "app" / "config.py")
    (core / "state" / "lexicon.json").write_text("{}", encoding="utf-8")

    code = (
        "import sys; sys.path.insert(0, %r);"
        "import app.config as c;"
        "print(c.ROOT); print(c.DATA_DIR)"
        % str(core)
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-400:]
    root, data_dir = out.stdout.strip().splitlines()
    assert Path(root).resolve() == core.resolve(), \
        "vendor 布局下 ROOT 应指向 core/（含 data/ 的那层）"
    assert Path(data_dir).resolve() == (core / "data").resolve()


def test_server_web_dir_points_to_src_web() -> None:
    """静态挂载目录必须在 src/web 下，否则 /static 会 404。"""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import app.server as s;"
        "print(s.WEB_DIR)"
        % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-400:]
    assert Path(out.stdout.strip()).resolve() == (SRC / "web").resolve()


# --------------------------------------------------------------- 硬编码


@pytest.mark.parametrize("base", [SRC, ROOT / "tests"])
def test_no_hardcoded_absolute_paths(base: Path) -> None:
    """源码里不得残留机器相关的绝对路径。"""
    offenders: list[str] = []
    this_file = Path(__file__).name
    for f in base.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        # 本文件自身包含这些模式字面量（用来检测），跳过
        if f.name == this_file:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for marker in ("D:\\Dev\\", "C:\\Users\\", "/d/Dev/"):
            if marker in text:
                # 注释里提到历史路径不算问题
                for ln in text.splitlines():
                    if marker in ln and not ln.strip().startswith("#"):
                        offenders.append(f"{f.relative_to(ROOT)}: {ln.strip()[:80]}")
    assert not offenders, "存在硬编码绝对路径:\n" + "\n".join(offenders[:10])


def test_no_relative_syspath_dot() -> None:
    """``sys.path.insert(0, ".")`` 依赖当前工作目录，不可靠。"""
    offenders: list[str] = []
    this_file = Path(__file__).name
    for base in (SRC, ROOT / "tests"):
        for f in base.rglob("*.py"):
            if "__pycache__" in f.parts or f.name == this_file:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, ln in enumerate(text.splitlines(), 1):
                if 'sys.path.insert(0, ".")' in ln and \
                        not ln.strip().startswith("#"):
                    offenders.append(f"{f.relative_to(ROOT)}:{i}")
    assert not offenders, f"存在依赖 CWD 的 sys.path: {offenders}"


def test_future_imports_first() -> None:
    """``from __future__`` 必须位于文件开头，否则是 SyntaxError。"""
    offenders: list[str] = []
    for base in (SRC, ROOT / "tests"):
        for f in base.rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                offenders.append(f"{f.relative_to(ROOT)}: {exc}")
    assert not offenders, "语法错误:\n" + "\n".join(offenders)


# ------------------------------------------------------------------ 入口


def test_run_py_needs_no_pythonpath() -> None:
    """``python run.py --help`` 必须能跑，且不依赖外部 PYTHONPATH。"""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        capture_output=True, text=True, timeout=60, cwd=str(ROOT), env=env,
    )
    assert out.returncode == 0, f"run.py 启动失败: {out.stderr[-500:]}"
    assert "--port" in out.stdout
