"""水系护栏的地理数据加载器：代码与数据分离。

设计约定
--------
判定「轨迹是否横穿大河」是**通用几何问题**；「哪几条河、岸线在哪、
河有多宽」是**地理参考数据**。两者必须分开：

* 本模块只负责把 `data/waters.json` 读成结构化数据；
* 护栏与相似度筛选只按读进来的条目循环，不认识任何具体河名；
* 换城市、换河流、或接入自己的水系数据，只需替换 JSON 文件
  （也可用环境变量 `RIDE_WATERS_JSON` 指向别处），不需要改代码。

数据文件缺失或格式错误时返回空集合：护栏会自然降级为「不检查」，
而不是让识别流程因为一份参考数据报错。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WATERS_FILE = ROOT / "state" / "waters.json"
WATERS_FILE_ENV = "RIDE_WATERS_JSON"


def _data_path() -> Path:
    override = os.environ.get(WATERS_FILE_ENV, "").strip()
    return Path(override) if override else DEFAULT_WATERS_FILE


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    try:
        raw = json.loads(_data_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def reload() -> None:
    """清缓存（切换数据文件后调用，主要供测试与运行时热替换使用）。"""
    _load.cache_clear()


def rivers() -> list[dict[str, Any]]:
    """返回配置里的河流条目；每一项都是普通 dict，字段自解释。"""
    items = _load().get("rivers")
    if not isinstance(items, list):
        return []
    return [r for r in items if isinstance(r, dict) and r.get("name")]


def _points(value: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(value, list):
        return out
    for p in value:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                out.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                continue
    return out


def water_specs() -> dict[str, tuple[str, float, bool]]:
    """河名 -> (Overpass 查询名, 岸线半宽米, 是否把东侧算作过江)。"""
    specs: dict[str, tuple[str, float, bool]] = {}
    for r in rivers():
        try:
            specs[str(r["name"])] = (
                str(r.get("query") or r["name"]),
                float(r.get("bank_half_width_m", 0.0)),
                bool(r.get("east_side", True)),
            )
        except (TypeError, ValueError):
            continue
    return specs


def fallback_polylines() -> dict[str, list[tuple[float, float]]]:
    """河名 -> 岸线兜底折线（联网与缓存都不可用时使用）。"""
    return {
        str(r["name"]): pts
        for r in rivers()
        if (pts := _points(r.get("bank_fallback")))
    }


def centerlines() -> dict[str, list[tuple[float, float]]]:
    """河名 -> 河心线折线（相似度筛选里做粗粒度横穿检查）。"""
    return {
        str(r["name"]): pts
        for r in rivers()
        if (pts := _points(r.get("centerline")))
    }


def center_half_widths() -> dict[str, float]:
    """河名 -> 河心线判定的半宽（米）。"""
    out: dict[str, float] = {}
    for r in rivers():
        try:
            out[str(r["name"])] = float(r.get("center_half_width_m", 0.0))
        except (TypeError, ValueError):
            continue
    return out


def search_bbox() -> tuple[float, float, float, float]:
    """Overpass 查询范围（south, west, north, east）。

    这是**可替换的地理数据**，不是算法常量：换城市时改 JSON 即可。
    """
    raw = _load().get("search_bbox")
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            s, w, n, e = (float(v) for v in raw[:4])
            if s < n and w < e:
                return (s, w, n, e)
        except (TypeError, ValueError):
            pass
    # 没配范围时退化为全球查询（会慢，但不会把代码绑死在某个城市）。
    return (-90.0, -180.0, 90.0, 180.0)
