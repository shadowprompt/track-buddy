"""置信度校准 CLI：用 (识别结果, 真值轨迹) 拟合经验成功率曲线。

用法（仓库根执行）::

    # 单组样本可重复给；用 | 分隔，避免 Windows 盘符里的冒号
    python src/tools/calibrate_confidence.py \
        --case data/last_result.json|samples/truth_a.gpx \
        --case output/result_b.json|samples/truth_b.gpx

    # 样本多时用清单文件： [{"pred": "...", "truth": "..."}, ...]
    python src/tools/calibrate_confidence.py --cases output/calib_cases.json \
        --out data/confidence_calibration.json

    # 看当前模型状态（不改文件）
    python src/tools/calibrate_confidence.py --show

**没有真值就没有校准**：样本数不足 ``--min-samples`` 时不写模型文件，
服务端 ``apply_calibration()`` 会原样返回原置信度。真值必须来自
``eval.load_truth()`` 支持的 GPX / GeoJSON / 坐标数组，不能用流水线
自己输出的贴路距离、锚点残差等自洽指标当标签。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.calibration import (  # noqa: E402
    MIN_SAMPLES, SHRINK_K, default_path, fit_calibration,
    label_from_metrics, load_calibration, save_calibration, store_for,
)
from app.eval import evaluate_tracks, load_prediction, load_truth  # noqa: E402


def _pred_confidence(pred_path: Path) -> float | None:
    """从识别结果 JSON 里取原始置信度。"""
    try:
        data = json.loads(pred_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "confidence" not in data:
        return None
    try:
        val = float(data["confidence"])
    except (TypeError, ValueError):
        return None
    return val if 0.0 <= val <= 1.0 else None


def _parse_case(raw: str) -> tuple[str, str] | None:
    for sep in ("|", "::"):
        if sep in raw:
            a, b = raw.split(sep, 1)
            if a.strip() and b.strip():
                return a.strip(), b.strip()
    return None


def _load_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for raw in args.case or []:
        parsed = _parse_case(raw)
        if parsed is None:
            print(f"跳过无法解析的 --case：{raw}（需要 pred|truth）", file=sys.stderr)
        else:
            cases.append(parsed)
    if args.cases:
        data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        rows = data.get("cases", []) if isinstance(data, dict) else data
        for row in rows:
            if isinstance(row, dict):
                cases.append((str(row["pred"]), str(row["truth"])))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                cases.append((str(row[0]), str(row[1])))
    return cases


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def collect_samples(cases: list[tuple[str, str]]) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    samples: list[tuple[float, float]] = []
    details: list[dict[str, Any]] = []
    for pred, truth in cases:
        pp, tp = _resolve(pred), _resolve(truth)
        item: dict[str, Any] = {"pred": pred, "truth": truth}
        if not pp.exists() or not tp.exists():
            item["skipped"] = "文件不存在"
            details.append(item)
            continue
        conf = _pred_confidence(pp)
        if conf is None:
            item["skipped"] = "识别结果里没有 confidence 字段"
            details.append(item)
            continue
        try:
            metrics = evaluate_tracks(load_prediction(pp), load_truth(tp))
        except Exception as exc:  # noqa: BLE001 — 单个坏样本不该中断整批
            item["skipped"] = f"真值/预测解析失败：{exc}"
            details.append(item)
            continue
        label = label_from_metrics(metrics)
        item.update({
            "confidence": conf,
            "label": label,
            "start_m": round(float(metrics.get("start_m", float("inf"))), 1),
            "end_m": round(float(metrics.get("end_m", float("inf"))), 1),
            "frechet_m": round(float(metrics.get("frechet_m", float("inf"))), 1),
            "length_rel_error": round(float(metrics.get("length_rel_error", float("inf"))), 3),
        })
        details.append(item)
        samples.append((conf, float(label)))
    return samples, details


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="基于真值样本校准识别置信度")
    ap.add_argument("--case", action="append", default=[],
                    help="一组样本，格式 pred.json|truth.gpx（可重复）")
    ap.add_argument("--cases", help="样本清单 JSON：数组或 {'cases': [...]}")
    ap.add_argument("--out", help="模型输出路径（默认 data/confidence_calibration.json）")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                    help="少于该样本数不写模型（默认 8）")
    ap.add_argument("--shrink-k", type=float, default=SHRINK_K,
                    help="收缩强度，越大越靠近原始分（默认 20）")
    ap.add_argument("--show", action="store_true", help="只显示当前模型状态")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else default_path()
    if not out.is_absolute():
        out = ROOT / out
    if args.show:
        status = store_for().status()
        print(json.dumps(status, ensure_ascii=False, indent=2)
              if args.json else
              f"校准状态：{status['status']}，样本 {status['n_samples']}，"
              f"模型 {status['path']}")
        return 0

    cases = _load_cases(args)
    if not cases:
        ap.error("需要 --case pred|truth 或 --cases 清单")
        return 2
    samples, details = collect_samples(cases)
    skipped = [d for d in details if d.get("skipped")]
    model = fit_calibration(samples, min_samples=args.min_samples,
                            shrink_k=args.shrink_k)
    report = {
        "n_cases": len(cases),
        "n_samples": len(samples),
        "n_positive": sum(1 for _, y in samples if y >= 0.5),
        "n_skipped": len(skipped),
        "min_samples": args.min_samples,
        "model_written": model is not None,
        "out": str(out),
        "details": details,
    }
    if model is None:
        report["note"] = ("有效真值样本不足，未写出模型；服务端保持未校准，"
                          "置信度原样返回")
    else:
        save_calibration(model, out)
        report["knots"] = model["knots"]
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"样本 {report['n_samples']}/{report['n_cases']} 可用"
              f"（命中 {report['n_positive']}，跳过 {report['n_skipped']}）")
        for d in skipped:
            print(f"  跳过 {d['pred']}：{d['skipped']}")
        if model is None:
            print(f"样本不足 {args.min_samples} 组，未写出模型 —— 保持未校准")
        else:
            print(f"模型已写入 {out}")
            for k in model["knots"]:
                print(f"  raw≈{k['x']:.3f} → 经验成功率 {k['rate']:.3f}"
                      f"（n={k['n']}）")
    return 0 if model is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
