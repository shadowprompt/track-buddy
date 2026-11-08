"""通用性约束：识别与产出逻辑不得绑定某一次上传的截图。

把「不要根据图片定制」变成可执行检查：

1. 核心模块的**代码字面量**里不出现样本文件名、样本专属路名、
   固定经纬度坐标（注释与 docstring 里的排查记录不算代码）；
2. 可视化产出不按输入文件名分支，道路配色/筛选对任意路名都成立；
3. 水系、POI 种子等地理参考数据放在代码之外，且可整体替换；
4. 物理判据以配置项形式存在，而不是散落在业务代码里的裸数字。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from app import seed_lexicon, visuals, waterdata
from app.config import CONFIG

ROOT = Path(__file__).resolve().parent.parent

CORE_MODULES = (
    "multishot", "track", "scale", "register", "pipeline", "server",
    "roadfit", "waterguard", "simfit", "visuals", "seed_lexicon",
    "waterdata", "config",
)

# 一旦代码里重新出现这些名字，说明又针对某批截图开了特例。
SAMPLE_MARKERS = (
    "meituan_ride_shanghai",
    "Clipboard_Screenshot",
    "FIVE_ROADS",
    "安龙路", "新渔东路", "新渔路", "剑河路", "金钟路", "北翟路",
)


def _code_literals(module: str) -> list[object]:
    """收集代码里的字面量，排除注释与 docstring（历史排查记录）。"""
    path = ROOT / "app" / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and ast.get_docstring(node):
                doc_ids.add(id(body[0].value))
    return [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and id(n) not in doc_ids
    ]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_no_sample_specific_literals(module):
    lits = _code_literals(module)
    for marker in SAMPLE_MARKERS:
        assert not any(isinstance(v, str) and marker in v for v in lits), (
            f"app/{module}.py 里出现了样本专属标记 {marker!r}")
    # 固定经纬度只应存在于 data/*.json，不应回到业务代码
    for v in lits:
        if isinstance(v, float) and (120.0 < v < 122.0 or 30.0 < v < 32.0):
            pytest.fail(f"app/{module}.py 里出现固定经纬度字面量 {v}")


def test_output_names_do_not_depend_on_filename():
    src = (ROOT / "app" / "visuals.py").read_text(encoding="utf-8")
    assert "_tag(" not in src
    assert "os.path.basename" not in src
    for name in ("overlay_result.png", "road_verify.png",
                 "trajectory_chart.png"):
        assert f'"{name}"' in src
    assert not hasattr(visuals, "_tag")


def test_road_colors_and_selection_work_for_arbitrary_names():
    # 任意路名都要能取到颜色，且同名跨进程稳定（用摘要而非内置 hash）
    for name in ("随便一条路", "Rue de Test", "长安街", ""):
        col = visuals._road_color(name)
        assert isinstance(col, tuple) and len(col) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in col)
    assert visuals._road_color("甲路") == visuals._road_color("甲路")

    raw = [("甲路", [(0, 0), (1, 1)]),
           ("乙路", [(2, 2)]),
           ("丙路", [(3, 3), (4, 4), (5, 5)])]
    by_result = visuals._roads_to_draw(
        {"road_attribution": {"roads": ["乙路"]}}, raw)
    assert list(by_result) == ["乙路"]
    # 归属名与路网命名对不上时，退化为按出现次数选，且不超过展示上限
    fallback = visuals._roads_to_draw(
        {"road_attribution": {"roads": ["外地别名"]}}, raw)
    assert fallback and len(fallback) <= visuals.MAX_VERIFY_ROADS


def test_water_data_is_external_and_replaceable(tmp_path, monkeypatch):
    custom = {
        "version": 1,
        "search_bbox": [10.0, 20.0, 11.0, 21.0],
        "rivers": [{
            "name": "测试河", "query": "Test River",
            "bank_half_width_m": 123.0, "east_side": False,
            "center_half_width_m": 45.0,
            "bank_fallback": [[20.5, 10.5], [20.6, 10.6]],
            "centerline": [[20.5, 10.5], [20.6, 10.6]],
        }],
    }
    f = tmp_path / "waters.json"
    f.write_text(json.dumps(custom, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv(waterdata.WATERS_FILE_ENV, str(f))
    waterdata.reload()
    try:
        assert waterdata.water_specs() == {
            "测试河": ("Test River", 123.0, False)}
        assert waterdata.center_half_widths() == {"测试河": 45.0}
        assert waterdata.search_bbox() == (10.0, 20.0, 11.0, 21.0)
        assert waterdata.fallback_polylines()["测试河"] == [
            (20.5, 10.5), (20.6, 10.6)]
    finally:
        monkeypatch.delenv(waterdata.WATERS_FILE_ENV)
        waterdata.reload()


def test_seed_lexicon_is_external_and_replaceable(tmp_path):
    custom = {
        "verified": {"测试公园": [100.0, 30.0, "测试市"]},
        "aliases": {"测试公": "测试公园"},
        "excluded": {"某路": "道路名"},
    }
    f = tmp_path / "seed.json"
    f.write_text(json.dumps(custom, ensure_ascii=False), encoding="utf-8")
    data = seed_lexicon.load_seed(str(f))
    assert data["verified"]["测试公园"][2] == "测试市"
    # 默认数据文件存在，且不在代码里写坐标
    assert seed_lexicon.DEFAULT_SEED_FILE.endswith("seed_lexicon.json")
    assert Path(seed_lexicon.DEFAULT_SEED_FILE).is_file()


def test_physical_threshold_is_a_config_item():
    assert CONFIG.road_fit_low_zoom_m_per_px > 0
    pipeline = (ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
    assert "_ROAD_FIT_MAX_M_PER_PX" not in pipeline
    assert "self.cfg.road_fit_low_zoom_m_per_px" in pipeline
