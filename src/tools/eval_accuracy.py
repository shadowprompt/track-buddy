"""准确率评测 CLI：合成基准 + 真实 (识别结果, 真值轨迹) 对比。

用法（仓库根执行）::

    # 合成真值基准：20 个带噪/离群/OCR 残片的配准样本
    python src/tools/eval_accuracy.py --synthetic 20 --out output/eval_baseline.json

    # 真实样本：识别结果 JSON 与 GPX / GeoJSON 真值
    python src/tools/eval_accuracy.py --pred data/last_result.json \
        --truth samples/truth.gpx

    # 与上一版对比（同一 seed 才有意义）
    python src/tools/eval_accuracy.py --synthetic 20 \
        --compare output/eval_baseline.json

改动前后跑同一 ``--seed0``，比 ``center_error_m`` /
``scale_rel_error`` / ``precision`` / ``recall`` 的 P50-P90，
不要只比「贴路距离」——后者会奖励把轨迹搬到平行道路上的假解。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.eval import evaluate_files, run_synthetic_benchmark  # noqa: E402


def _print_summary(report: dict) -> None:
    if "aggregate" not in report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    agg = report["aggregate"]
    print(f"样本 {report['n_ok']}/{report['n_cases']} 配准成功")
    print(f"{'指标':<22}{'P50':>12}{'P90':>12}{'均值':>12}")
    for key, val in agg.items():
        print(f"{key:<22}{val['p50']:>12.3f}{val['p90']:>12.3f}{val['mean']:>12.3f}")


def _compare(report: dict, baseline_path: Path) -> dict:
    if not baseline_path.exists():
        return {"compare_error": f"基线文件不存在：{baseline_path}"}
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    out: dict = {"baseline": str(baseline_path), "delta": {}}
    for key, val in report.get("aggregate", {}).items():
        old = base.get("aggregate", {}).get(key, {}).get("p50")
        if old is None:
            continue
        new = val["p50"]
        out["delta"][key] = {
            "p50_before": old,
            "p50_after": new,
            "change": new - old,
            "improved": new < old,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="识别准确率评测")
    ap.add_argument("--synthetic", type=int, default=20,
                    help="合成基准样本数（0 关闭）")
    ap.add_argument("--seed0", type=int, default=0, help="首个随机种子")
    ap.add_argument("--pred", help="识别结果 JSON / GeoJSON")
    ap.add_argument("--truth", help="真值 GPX / GeoJSON / 坐标数组 JSON")
    ap.add_argument("--compare", help="与既有基准 JSON 比较 P50")
    ap.add_argument("--out", help="把完整报告写到该路径")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args(argv)

    report: dict
    if args.pred and args.truth:
        report = evaluate_files(args.pred, args.truth)
    elif args.synthetic > 0:
        report = run_synthetic_benchmark(args.synthetic, args.seed0)
    else:
        ap.error("需要 --pred/--truth 或 --synthetic > 0")
        return 2

    if args.compare:
        report["comparison"] = _compare(report, Path(args.compare))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)
        c = report.get("comparison")
        if c and "delta" in c:
            print("\n与基线相比（P50，越小越好）：")
            for key, d in c["delta"].items():
                flag = "改善" if d["improved"] else "变差"
                print(f"  {key:<22}{d['p50_before']:>10.3f} → "
                      f"{d['p50_after']:>10.3f}  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
