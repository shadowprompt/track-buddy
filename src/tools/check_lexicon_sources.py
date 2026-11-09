"""词典跨源一致性检查：找出同名词条里坐标基准可能不一致的条目。

用法::

    python src/tools/check_lexicon_sources.py \
        --path data/lexicon.json --out output/lexicon_sources.json

为什么需要它
------------
词典是多年累积的：同一个地名可能同时存着 amap / tianditu / osm /
verified / user 的坐标。在线侧的 CRS 已统一换算到 GCJ-02，但**历史条目**
不一定；同名跨源坐标若相距几百米，配准时就会出现「候选很多但基准不一致」
的系统性差。这份报告只**列出可疑条目供人工复核**，不自动改数据。

同名多源不等于错（同名异地本来就存在），所以判据是「不同源之间的最大
地表距离」超过阈值，而不是「来源不唯一」。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.geo.coords import haversine  # noqa: E402


def inspect_lexicon(
    data: dict, sep_m: float = 200.0, near_max_m: float = 10_000.0
) -> dict:
    """统计来源分布与跨源冲突。

    冲突分两类，处置完全不同：
      * ``near``（≤10km）：疑似坐标基准 / 版本不一致，需要人工复核；
      * ``far``（>10km）：多半是**同名异地**（不同城市的同名 POI），
        本来就该有多个候选，不是脏数据。
    """
    counts: Counter = Counter()
    multi_source = 0
    conflicts: list[dict] = []
    for name, entries in data.items():
        if not isinstance(entries, list):
            continue
        pts = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            src = str(e.get("source") or "unknown")
            counts[src] += 1
            try:
                lon, lat = float(e["lon"]), float(e["lat"])
            except (KeyError, TypeError, ValueError):
                continue
            pts.append((src, lon, lat))
        sources = {s for s, _, _ in pts}
        if len(sources) >= 2:
            multi_source += 1
        sep = 0.0
        for (sa, lon_a, lat_a), (sb, lon_b, lat_b) in combinations(pts, 2):
            if sa == sb:
                continue
            sep = max(sep, haversine(lon_a, lat_a, lon_b, lat_b))
        if sep > sep_m:
            conflicts.append({
                "name": name,
                "sources": sorted(sources),
                "max_cross_source_sep_m": round(sep, 1),
                "n_entries": len(pts),
                "category": "near" if sep <= near_max_m else "far",
            })
    conflicts.sort(key=lambda c: -c["max_cross_source_sep_m"])
    near = [c for c in conflicts if c["category"] == "near"]
    return {
        "n_names": len(data),
        "n_entries": int(sum(counts.values())),
        "source_counts": dict(counts.most_common()),
        "n_multi_source_names": multi_source,
        "sep_threshold_m": sep_m,
        "near_max_m": near_max_m,
        "n_conflicts": len(conflicts),
        "n_near_conflicts": len(near),
        "n_far_conflicts": len(conflicts) - len(near),
        "conflicts": conflicts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="词典跨源一致性检查")
    ap.add_argument("--path", default=str(ROOT / "data" / "lexicon.json"))
    ap.add_argument("--sep", type=float, default=200.0,
                    help="跨源坐标距离阈值（米）")
    ap.add_argument("--out", help="写完整报告 JSON")
    ap.add_argument("--top", type=int, default=15, help="打印前 N 条冲突")
    args = ap.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"词典文件不存在：{path}")
        return 2
    report = inspect_lexicon(json.loads(path.read_text(encoding="utf-8")),
                             sep_m=args.sep)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"地名 {report['n_names']} / 条目 {report['n_entries']}")
    print("来源分布：" + ", ".join(
        f"{k}={v}" for k, v in report["source_counts"].items()))
    print(f"同名多源 {report['n_multi_source_names']} 组，"
          f"跨源距离 > {args.sep:.0f}m 的冲突 {report['n_conflicts']} 组"
          f"（疑似基准不一致 {report['n_near_conflicts']} / "
          f"同名异地 {report['n_far_conflicts']}）")
    for c in report["conflicts"][:args.top]:
        tag = "待复核" if c["category"] == "near" else "同名异地"
        print(f"  [{tag}] {c['name']:<16}"
              f"{c['max_cross_source_sep_m']:>9.1f}m  "
              + "/".join(c["sources"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
