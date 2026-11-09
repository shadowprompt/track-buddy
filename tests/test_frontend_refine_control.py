"""前端「贴路校正」开关与长任务提示的静态契约测试。

这些是纯前端行为：服务端日志全 200 也发现不了控件没接线。用静态断言
锁住三件事——开关只在多图生效、勾选才发 ``refine=on``、轮询能识别
后端返回的 ``elapsed_s`` 与 verify 档。
"""

from __future__ import annotations

from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "src" / "web" / "index.html"


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_road_fit_checkbox_exists() -> None:
    h = _html()
    assert 'id="roadFit"' in h
    assert 'type="checkbox" id="roadFit"' in h
    assert "多图拼接只做" in h and "不平移" in h


def test_switch_only_affects_multi_shot() -> None:
    h = _html()
    assert "const syncFit = !!(multi && $('roadFit') && $('roadFit').checked);" in h
    assert "if (syncFit) fd.append('refine', 'on');" in h


def test_poll_handles_verify_mode_and_long_tasks() -> None:
    h = _html()
    assert "rf.mode === 'verify'" in h
    assert "Number(j.elapsed_s || 0) > 180" in h
    assert "refineWarned" in h
    # verify 分支必须排在通用 rf.ok 分支之前，否则永远走不到
    assert h.index("rf.mode === 'verify'") < h.index("} else if (rf.ok) {")


def test_rejected_candidate_is_drawn_in_preview() -> None:
    """未采纳的精修路线必须用另一种颜色画在原图叠加层。"""
    h = _html()
    assert 'id="fitLegend"' in h
    assert "candidate_shift_px" in h
    assert "const cs = (!rf.ok && rf.candidate_shift_px) || null;" in h
    assert "ctx.strokeStyle = '#ff8a3d';" in h
    assert "ctx.setLineDash([9, 6]);" in h
    assert "(p[0] + cs[0]) * sx" in h
    # 换图时必须清掉上一张结果的候选线与图例
    assert "legend.style.display = 'none'" in h
    assert "橙色虚线 = 未采纳的路网精修位置" in h


def test_rejected_candidate_is_explained_outside_preview() -> None:
    """同步 refine=on 被拒时不启动轮询，状态条与卡片也要说明。"""
    h = _html()
    assert "橙色虚线为未采纳的路网精修位置" in h
    assert "} else if (rf.candidate_shift_px) {" in h
    assert "const fitHint = (!rf0.ok && rf0.candidate_shift_px)" in h
    assert "路网精修未采纳，候选位置已在原始截图上用橙色虚线标出" in h


def test_reject_note_is_not_double_wrapped() -> None:
    """后端 notes 已带「路网精修未采纳（…）」，状态条不能再套一层。"""
    h = _html()
    assert "const rawNote = text((rf.notes || [])[0], '路网精修未采纳（位置未改动）');" in h
    assert "/路网精修未采纳/.test(rawNote)" in h
    # 旧写法会把同一句话嵌套两次
    assert "'路网精修未采纳（' +\n              esc(text((rf.notes || [])[0]" not in h


def test_candidate_position_can_be_toggled_on_map() -> None:
    """护栏否决时把选择权交给用户：默认不切，点按钮才画候选并可切回。"""
    h = _html()
    assert 'id="candToggle"' in h
    assert "function buildCandidatePath(d)" in h
    assert "function setupCandidateToggle(d)" in h
    assert "setupCandidateToggle(d);" in h
    assert "candidateShown = !candidateShown;" in h
    assert "basemap.setTrack(candidateShown ? candidatePath : lastPath);" in h
    # 候选只在「未采纳 + 有位移」时出现；采纳成功不能残留按钮
    assert "if (rf.ok || !rf.candidate_shift_px) return null;" in h
    assert "btn.style.display = 'none';" in h
    # 刚体平移换算：东向用 cos(纬度)，北向用固定每度米数
    assert "const dlon = de / (111320 * cosLat);" in h
    assert "const dlat = dn / 110540;" in h
    # 切换只影响地图显示，结果坐标与下载文件不变
    assert "结果与下载文件仍是初次识别位置" in h
