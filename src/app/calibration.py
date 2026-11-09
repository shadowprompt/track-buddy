"""置信度校准：把「自洽分」映射到「有真值支撑的经验成功率」。

动机
----
``confidence`` 是启发式加权和（配准残差、锚点数、贴路距离……），它只回答
「结果自洽吗」，不回答「结果对吗」。交接文档 §22.38 记录过反例：把一条
起终点直线整体平移到平行主干道后，贴路距离、锚点残差和 confidence 会
**一起变好**，真实位置却更错。只调权重系数永远无法证伪这类假解。

本模块用外部真值样本 ``(识别结果, GPX/GeoJSON)`` 拟合一个**单调**映射：
给定原始 confidence，预测「这次识别达到误差门槛」的经验概率。

拟合方法
--------
1. 每个样本 = ``(raw_confidence, label)``；``label`` 由
   ``eval.evaluate_tracks()`` 的起终点误差 / Fréchet 距离 / 里程相对误差
   共同判定（``label_from_metrics()``），不用任何自洽指标。
2. PAVA（pool-adjacent-violators）做保序回归，得到单调不减的分段经验率，
   避免小样本上出现「0.9 分的成功率反而低于 0.7 分」。
3. 与原始分做**收缩混合**：``w = n / (n + SHRINK_K)``，样本越少越靠近
   原分；分段之间线性插值，端点外做常数外推。

安全降级（最重要的一条）
------------------------
没有真值样本、样本数低于 ``MIN_SAMPLES``、模型文件不存在/损坏/版本不符时，
``load_calibration()`` 返回 ``None``，``apply_calibration()`` **原样返回**
原始置信度。也就是说：没做过真值评测的部署，行为与校准前逐字节一致，
不会因为缺文件而把用户看到的分数偷偷改掉。

校准模型默认写在 ``data/confidence_calibration.json``（已 gitignore，
它从含真实轨迹的数据派生，不该入库），可用
``RIDE_CONFIDENCE_CALIBRATION`` 覆盖路径；见
``src/tools/calibrate_confidence.py``。
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# 少于这个样本数就不拟合：8 个点做保序回归只会把噪声写成规律。
MIN_SAMPLES = 8
# 收缩强度。n=8 时 w≈0.29，n=40 时 w≈0.67，样本越多越信任经验率。
SHRINK_K = 20.0
MODEL_VERSION = 1

# 「这次识别算不算对」的外部判据，单位与 eval.evaluate_tracks() 一致。
# 取得比项目标称 10-30m 精度宽：校准关注的是「整条轨迹是不是同一个行程」，
# 不是米级配准精度；过严会把所有样本标成 0，校准无从谈起。
DEFAULT_THRESHOLDS: dict[str, float] = {
    "start_m": 300.0,
    "end_m": 300.0,
    "frechet_m": 400.0,
    "length_rel_error": 0.30,
}

CALIBRATION_PATH_ENV = "RIDE_CONFIDENCE_CALIBRATION"
CALIBRATION_FILENAME = "confidence_calibration.json"


def default_path(cfg: Any = None) -> Path:
    """校准模型路径：显式环境变量 > .env 当前值 > 词典同目录。"""
    raw = (os.environ.get(CALIBRATION_PATH_ENV) or "").strip()
    if raw:
        return Path(raw)
    if cfg is not None:
        path = getattr(cfg, "confidence_calibration_path", None)
        if path is None:
            lex = getattr(cfg, "lexicon_path", None)
            path = Path(lex).parent / CALIBRATION_FILENAME if lex else None
        if path is not None:
            return Path(path)
    from .config import CONFIG  # 延迟导入，避免模块加载期的循环依赖

    return Path(CONFIG.confidence_calibration_path)


def label_from_metrics(
    metrics: dict[str, Any] | None,
    thresholds: dict[str, float] | None = None,
) -> int:
    """把轨迹级真值指标转成 0/1 标签。

    任一指标缺失、非有限或超门槛即判 0 —— 真值不完整时宁可保守，
    不能把「没测到」当成「测过了且正确」。
    """
    th = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if not metrics:
        return 0
    for key, limit in th.items():
        val = metrics.get(key)
        if val is None:
            return 0
        try:
            v = float(val)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(v) or v > float(limit):
            return 0
    return 1


def _clean_samples(
    samples: Iterable[Sequence[float]],
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in samples or []:
        if item is None or len(item) < 2:
            continue
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return out


def fit_isotonic(
    samples: Iterable[Sequence[float]],
) -> list[dict[str, float]]:
    """PAVA 保序回归，返回单调不减的折点列表。

    相同的 raw confidence 先合并成 (命中数, 样本数) 的块，再不断把
    「前块经验率 > 后块经验率」的相邻块合并，直到整体单调。
    """
    groups: dict[float, list[float]] = {}
    for x, y in _clean_samples(samples):
        key = round(x, 6)
        g = groups.setdefault(key, [0.0, 0.0])
        g[0] += y
        g[1] += 1.0
    blocks: list[list[float]] = []
    for key in sorted(groups):
        s, n = groups[key]
        blocks.append([key, key, s, n])
        while len(blocks) >= 2:
            prev, cur = blocks[-2], blocks[-1]
            if prev[2] / prev[3] <= cur[2] / cur[3] + 1e-12:
                break
            merged = blocks.pop()
            blocks[-1][1] = merged[1]
            blocks[-1][2] += merged[2]
            blocks[-1][3] += merged[3]
    return [
        {
            "x": round((b[0] + b[1]) / 2.0, 6),
            "rate": round(b[2] / b[3], 6),
            "n": int(b[3]),
        }
        for b in blocks
    ]


def fit_calibration(
    samples: Iterable[Sequence[float]],
    min_samples: int = MIN_SAMPLES,
    shrink_k: float = SHRINK_K,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """从 (raw_confidence, label) 拟合模型；样本不足返回 ``None``。"""
    clean = _clean_samples(samples)
    knots = fit_isotonic(clean)
    if len(clean) < max(1, int(min_samples)) or not knots:
        return None
    return {
        "version": MODEL_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(clean),
        "n_positive": int(sum(1 for _, y in clean if y >= 0.5)),
        "min_samples": int(min_samples),
        "shrink_k": float(shrink_k),
        "thresholds": dict(DEFAULT_THRESHOLDS if thresholds is None
                           else thresholds),
        "knots": knots,
    }


@dataclass(frozen=True)
class ConfidenceCalibrator:
    """已加载的校准模型；``apply()`` 输出 0~0.99 的校准后置信度。"""

    knots: tuple[tuple[float, float], ...]
    n_samples: int
    shrink_k: float = SHRINK_K
    path: str = ""

    def _empirical(self, x: float) -> float:
        pts = self.knots
        if len(pts) == 1:
            return pts[0][1]
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                if x1 - x0 < 1e-9:
                    return max(y0, y1)
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return pts[-1][1]

    def apply(self, raw: float) -> float:
        try:
            x = min(1.0, max(0.0, float(raw)))
        except (TypeError, ValueError):
            return float(raw) if isinstance(raw, (int, float)) else 0.0
        if not math.isfinite(x):
            return 0.0
        w = self.n_samples / (self.n_samples + max(self.shrink_k, 1e-6))
        out = (1.0 - w) * x + w * self._empirical(x)
        return round(min(0.99, max(0.0, out)), 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "status": "calibrated",
            "n_samples": self.n_samples,
            "n_knots": len(self.knots),
            "shrink_k": self.shrink_k,
            "path": self.path,
        }


def save_calibration(model: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model, ensure_ascii=False, indent=2),
                 encoding="utf-8")


def load_calibration(path: str | Path) -> ConfidenceCalibrator | None:
    """读模型；任何异常/样本不足/结构不合法都返回 ``None``（安全降级）。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("version", -1)) != MODEL_VERSION:
            return None
        knots = data.get("knots") or []
        pts = tuple(
            (float(k["x"]), float(k["rate"]))
            for k in knots if "x" in k and "rate" in k
        )
        n = int(data.get("n_samples") or 0)
        min_n = int(data.get("min_samples") or MIN_SAMPLES)
        if not pts or n < max(1, min_n):
            return None
        rates = [y for _, y in pts]
        if any(not math.isfinite(y) for y in rates):
            return None
        if any(rates[i] > rates[i + 1] + 1e-9 for i in range(len(rates) - 1)):
            return None  # 非单调模型一律不信任
        return ConfidenceCalibrator(
            knots=pts, n_samples=n,
            shrink_k=float(data.get("shrink_k") or SHRINK_K),
            path=str(p),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


class CalibrationStore:
    """按文件 mtime/size 热加载模型，改完校准文件不用重启服务。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stamp: tuple[int, int] | None = None
        self._model: ConfidenceCalibrator | None = None

    def _refresh(self) -> None:
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = (-1, -1)
        if stamp == self._stamp:
            return
        self._stamp = stamp
        self._model = load_calibration(self.path)

    def get(self) -> ConfidenceCalibrator | None:
        self._refresh()
        return self._model

    def status(self) -> dict[str, Any]:
        model = self.get()
        if model is None:
            return {"enabled": False, "status": "uncalibrated",
                    "n_samples": 0, "path": str(self.path),
                    "note": "无真值校准样本，置信度按原分返回"}
        return model.as_dict()


_STORES: dict[str, CalibrationStore] = {}


def store_for(cfg: Any = None) -> CalibrationStore:
    p = default_path(cfg)
    key = str(p)
    if key not in _STORES:
        _STORES[key] = CalibrationStore(p)
    return _STORES[key]


def apply_calibration(raw: float, cfg: Any = None) -> tuple[float, dict[str, Any]]:
    """流水线入口：返回 ``(校准后置信度, 诊断状态)``。

    没有可用模型时第一项严格等于 ``raw``（不做任何四舍五入以外的事）。
    """
    store = store_for(cfg)
    model = store.get()
    status = store.status()
    if model is None:
        return raw, status
    return model.apply(raw), status
