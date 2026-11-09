"""由地理种子数据生成初始词典（`data/lexicon.json`）。

代码与数据分离
--------------
经核验的 POI 坐标、OCR 别名、剔除清单都是**地理/业务参考数据**，
放在 `data/seed_lexicon.json`（也可用环境变量 `RIDE_SEED_LEXICON_JSON`
指向别处）。本模块只负责：

  1. WGS-84 → GCJ-02 转换；
  2. 名称规范化；
  3. 别名展开。

筛选原则是通用的：只收录 **POI 类点位**（公园/宾馆/车站/商场…）。
道路名与水系名沿要素走向排布，文字中心不对应任何固定地理位置，
把它们当锚点会让配准整体平移数公里；异地同名 POI 同样不可用。

换城市时替换数据文件即可，不需要改这里的逻辑。
"""
import json
import os
import sys

# 本模块既能被 `python -m app.seed_lexicon` 调用，也能被直接当脚本跑。
# 两种方式下包路径不同：前者靠父包引导，后者需要把 src/ 加进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import ROOT  # noqa: E402
from app.geo.coords import wgs84_to_gcj02  # noqa: E402

SEED_FILE_ENV = "RIDE_SEED_LEXICON_JSON"
DEFAULT_SEED_FILE = os.path.join(ROOT, "data", "seed_lexicon.json")


def load_seed(path: str | None = None) -> dict:
    """读种子数据；文件缺失时返回空结构，由调用方决定如何处理。"""
    p = path or os.environ.get(SEED_FILE_ENV, "").strip() or DEFAULT_SEED_FILE
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


_SEED = load_seed()

# 保留模块级变量名以兼容外部引用，但内容全部来自数据文件。
VERIFIED: dict[str, tuple[float, float, str]] = {
    str(name): (float(v[0]), float(v[1]), str(v[2]))
    for name, v in (_SEED.get("verified") or {}).items()
    if isinstance(v, (list, tuple)) and len(v) >= 3
}
EXCLUDED: dict[str, str] = dict(_SEED.get("excluded") or {})
ALIASES: dict[str, str] = dict(_SEED.get("aliases") or {})


def normalize(name: str) -> str:
    n = name.strip()
    for ch in " \t　()（）·・,，.。-—_":
        n = n.replace(ch, "")
    return n


def main() -> None:
    if not VERIFIED:
        print(f"种子数据为空或不可读：{os.environ.get(SEED_FILE_ENV) or DEFAULT_SEED_FILE}")
        return

    out: dict[str, list[dict]] = {}
    for name, (lon, lat, city) in VERIFIED.items():
        glon, glat = wgs84_to_gcj02(lon, lat)
        out.setdefault(normalize(name), []).append({
            "lon": round(glon, 7),
            "lat": round(glat, 7),
            "weight": 1.0,
            "count": 1,
            "city": city,
            "source": "verified",
            "address": "",
            "kind": "poi",
        })

    for alias, target in ALIASES.items():
        src = out.get(normalize(target))
        if src:
            out.setdefault(normalize(alias), []).extend(dict(e) for e in src)

    path = os.path.join(ROOT, "data", "lexicon.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} keys -> {path}")
    print(f"数据文件中标记不可用 {len(EXCLUDED)} 条（道路名/水系/异地同名）")


if __name__ == "__main__":
    main()
