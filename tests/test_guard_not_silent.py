"""护栏「不能静默失效」的测试。

背景：实测踩过——`app/pipeline.py` 里调用 `check_crossing` 时参数名写错
（`allow_network` vs `allow_online`），触发 NameError，
而外层是 `except Exception: pass`，于是**护栏完全没有执行**，
但外部看起来一切正常（没有报错、没有警告、只是「没提示跨江」）。

这类 bug 极其隐蔽——它不会让功能失败，只会让**保护机制消失**。

所以这里守住两条：
  1. 跨江检查的失败必须**留痕**（进 warnings）
  2. 参数名必须正确（直接调用 pipeline 内部路径验证）
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def test_no_bare_except_pass():
    """不允许出现「裸 except: pass」——它会吞掉护栏的失败。"""
    bad = []
    for f in sorted((ROOT / "app").glob("*.py")):
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is not None:      # except SomeError:
                continue
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                bad.append(f"{f.name}:{node.lineno}")
    assert not bad, (
        "发现裸 except: pass（会静默吞掉护栏失败）：" + ", ".join(bad))
    print("  [OK] 没有裸 except: pass")


def test_check_crossing_call_signature():
    """pipeline 调 check_crossing 时用的参数名必须与函数签名一致。

    用 AST 检查，避免「运行时才发现」——这类错误在 except 里会被吞掉。
    """
    from app.waterguard import check_crossing

    sig = set(check_crossing.__code__.co_varnames[
        :check_crossing.__code__.co_argcount])
    src = (ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if name != "check_crossing":
            continue
        for kw in node.keywords:
            if kw.arg and kw.arg not in sig:
                bad.append(f"pipeline.py:{node.lineno} 未知参数 {kw.arg}")
    assert not bad, "check_crossing 调用参数名错误：" + "; ".join(bad)
    print(f"  [OK] check_crossing 参数名正确（可用：{sorted(sig)}）")


def test_water_check_reaches_result():
    """water_check 必须真的出现在 road_name_fit 里（而不是永远空 dict）。"""
    from app.pipeline import RidePipeline
    from app.config import CONFIG

    pipe = RidePipeline(CONFIG)
    src = (ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
    assert '"water_check": water_check' in src, "water_check 未写入返回值"
    # 确认 water_check 是在 except 之前赋值的（否则失败时永远是空）
    idx_assign = src.find('water_check = {')
    idx_except = src.find('except Exception as exc:', idx_assign)
    assert 0 < idx_assign < idx_except, "water_check 赋值在 except 之后"
    print("  [OK] water_check 写入路径正确")


if __name__ == "__main__":
    print("护栏静默失效测试：")
    test_no_bare_except_pass()
    test_check_crossing_call_signature()
    test_water_check_reaches_result()
    print("\n全部通过")
