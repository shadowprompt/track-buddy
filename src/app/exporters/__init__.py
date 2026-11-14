"""运动文件导出层：把「轨迹坐标 + 运动概要」合成为各平台通用的
运动文件格式（GPX / TCX，后续 FIT）。

**为什么独立成包**：文件生成的共同逻辑（WGS-84 输入约定、时间插值、
UTC 换算、XML 转义、数据不足时的行为）是跨格式复用的——每加一种
格式只需要实现一个 ``build_xxx(coords, *, ride_summary, source_title)
-> str | None`` 并注册进 ``_REGISTRY``，server 与前端不用改。

**格式契约**（新格式必须遵守）：

- 坐标输入一律 **WGS-84** ``[lon, lat]``；
- 概要字段缺失为 ``None``，**绝不猜测填充**；
- 数据不足以产出**合规**文件时返回 ``None``（如 TCX 无时间轴），
  宁缺勿错——不输出半合规的坏文件；
- 纯轨迹也有价值（GPX）的格式可以接受无概要输出。

**FIT 为什么还没实现**：FIT 是 Garmin 的二进制格式，规范带 CRC 与
复杂消息结构，需要 FIT SDK（``fitparse``/Garmin 官方 SDK）或完整的
自研编解码器，工作量与上面两种文本格式不在一个量级。注册表已为它
留好位置：实现 ``build_fit`` 后在 ``_REGISTRY`` 加一行即可。
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .gpx import build_gpx
from .tcx import build_tcx

# 格式注册表：fmt -> 构建函数。新增格式（如 FIT）在这里加一行。
_REGISTRY: dict[str, Callable[..., str | None]] = {
    "gpx": build_gpx,
    "tcx": build_tcx,
}


def supported_formats() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def export_one(
    fmt: str,
    coords: Sequence[Sequence[float]],
    *,
    ride_summary: dict[str, Any] | None = None,
    source_title: str = "骑行",
    geom_distance_m: float | None = None,
) -> str | None:
    """按格式名导出单个文件；未注册的格式（如尚未实现的 fit）报错。"""
    fn = _REGISTRY.get(fmt)
    if fn is None:
        raise ValueError(
            f"不支持的导出格式：{fmt}（当前支持 {supported_formats()}；"
            "FIT 为二进制格式待实现）")
    return fn(coords, ride_summary=ride_summary, source_title=source_title,
              geom_distance_m=geom_distance_m)


def build_exports(
    coords: Sequence[Sequence[float]],
    *,
    ride_summary: dict[str, Any] | None = None,
    source_title: str = "骑行",
    geom_distance_m: float | None = None,
) -> dict[str, str]:
    """跑全部已注册导出器，返回 ``{fmt: 内容}``。

    产出 ``None``（数据不足）的格式**缺席**而不是给空串——调用方
    （server → 前端）据此决定隐藏哪个下载链接。
    """
    out: dict[str, str] = {}
    for fmt in _REGISTRY:
        content = export_one(
            fmt, coords, ride_summary=ride_summary,
            source_title=source_title, geom_distance_m=geom_distance_m)
        if content:
            out[fmt] = content
    return out
