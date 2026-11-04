"""大模型语义层的测试。

核心规则只有一条：**LLM 的建议必须再通过几何验证才被采纳**。
模型判断错的时候，几何证据要能把它挡下来。

其余测试覆盖「LLM 没配 / 调用失败 / 返回非法结果」时，
主流程必须**一字不变**地继续（LLM 是可选增强，不是依赖）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import _extract_json
from app.llm_geo import check_road_sequence


class _StubClient:
    """假的 LLM 客户端。"""

    def __init__(self, payload, ok=True):
        self.available = True
        self.payload = payload
        self.ok = ok
        self.calls = 0

    def chat_json(self, system, user, images=None, schema_hint=""):
        from app.llm import LLMResult
        self.calls += 1
        r = LLMResult(ok=self.ok, model="stub")
        if self.ok:
            r.data = self.payload
            r.text = "stub"
        return r


def test_no_llm_means_no_change():
    """没配 LLM -> 序列原样返回，applied=False。"""
    class _None:
        available = False
    out = check_road_sequence(_None(), ["A路", "B路", "C路"], "上海")
    assert out["names"] == ["A路", "B路", "C路"]
    assert not out["applied"]
    assert not out["ok"]
    print("  [OK] 未配置 LLM 时序列不变")


def test_llm_suggestion_blocked_by_geometry():
    """几何代价更高时，LLM 的建议必须被挡下。"""
    client = _StubClient({
        "order": ["C路", "B路", "A路"],
        "changed": True,
        "notes": ["顺序反了"],
        "warnings": [],
    })
    # 代价函数：A 最优，逆序最差
    def cost(order):
        return {"A路": 1.0, "B路": 2.0, "C路": 3.0}[order[0]]

    out = check_road_sequence(client, ["A路", "B路", "C路"], "上海", cost)
    assert out["ok"], "应拿到模型结果"
    assert not out["applied"], "几何代价更高却采纳了"
    assert out["names"] == ["A路", "B路", "C路"], "顺序被改了"
    assert any("几何代价更高" in n for n in out["notes"]), out["notes"]
    print("  [OK] 几何代价更高时挡下 LLM 建议")


def test_llm_suggestion_accepted_when_geometry_agrees():
    """几何代价更低时，采纳 LLM 的建议。"""
    client = _StubClient({
        "order": ["C路", "B路", "A路"],
        "changed": True,
        "notes": ["顺序反了"],
        "warnings": ["A路可能记错"],
    })

    def cost(order):
        return {"A路": 9.0, "B路": 2.0, "C路": 1.0}[order[0]]

    out = check_road_sequence(client, ["A路", "B路", "C路"], "上海", cost)
    assert out["applied"], "几何代价更低却没采纳"
    assert out["names"] == ["C路", "B路", "A路"]
    assert out["warnings"] == ["A路可能记错"]
    print("  [OK] 几何证据支持时采纳 LLM 建议")


def test_illegal_order_rejected():
    """模型擅自新增/改写路名 -> 直接拒绝。"""
    client = _StubClient({
        "order": ["A路", "B路", "X路"],       # 多了个 X路
        "changed": True, "notes": [], "warnings": [],
    })
    out = check_road_sequence(client, ["A路", "B路", "C路"], "上海",
                              lambda o: 1.0)
    assert not out["applied"], "含有非法路名的顺序被采纳了"
    assert out["names"] == ["A路", "B路", "C路"]
    print("  [OK] 模型擅自新增路名时被拒绝")


def test_failed_call_keeps_order():
    """调用失败 -> 序列不变，主流程继续。"""
    client = _StubClient(None, ok=False)
    out = check_road_sequence(client, ["A路", "B路"], "上海",
                              lambda o: 1.0)
    assert not out["ok"]
    assert not out["applied"]
    assert out["names"] == ["A路", "B路"]
    print("  [OK] LLM 调用失败时序列不变")


def test_extract_json_tolerates_fences():
    """模型输出带 ```json 包裹或前后废话时也能解析。"""
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json('好的，结果如下：\n{"a": 3}\n希望有用') == {"a": 3}
    assert _extract_json("不是 JSON") is None
    assert _extract_json("") is None
    print("  [OK] JSON 提取能容忍各种包装")


if __name__ == "__main__":
    print("大模型语义层测试：")
    test_no_llm_means_no_change()
    test_llm_suggestion_blocked_by_geometry()
    test_llm_suggestion_accepted_when_geometry_agrees()
    test_illegal_order_rejected()
    test_failed_call_keeps_order()
    test_extract_json_tolerates_fences()
    print("\n全部通过")
