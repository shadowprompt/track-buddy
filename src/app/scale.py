"""图像几何尺度：所有像素阈值的唯一来源。

为什么需要
----------
同一套代码要处理尺度差异极大的输入：单张手机截图是千像素量级，
多图拼接画布可达数千乃至上万像素见方，而且张数、缩放级别都不固定。
早期代码里散落着 260 / 400 / 500 / 600 / 700 / 800
这类**绝对像素阈值**，它们只在某一个尺度上成立：画布一大，粗筛就把
真实轨迹整段淘汰；截图一换分辨率，形态学核又对不上线宽。

两类尺度必须分开
----------------
* **画布尺度**（对角线 / 短边）：随拼接张数增长。决定包围盒粗筛半径、
  接缝补线容差、形状校验里的拓扑距离、画布内存上限。
* **内容尺度**（徽章直径 → 轨迹线宽）：由**单张截图的缩放级别**决定，
  与拼了几张无关。拼接张数再多、画布再大，徽章仍是几十像素、轨迹线
  仍是十像素量级。它决定「贴住徽章」的容差、最小连通域面积、形态学核、
  补线宽度和 0 位移判据的容差。

把这两类混用，就是「换一批截图要重调一次数字」的根源：
用画布尺寸去量内容，或用内容尺寸去量画布，都会失效。

约定
----
本模块只定义**无量纲系数 × 标尺**的派生量，不出现任何
「针对某次上传实测调出来的」常数。调用方一律通过 `Scale` 取阈值，
不允许再在业务代码里写裸像素数字。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# 无徽章可测时的内容尺度兜底：地图 App 的起终点徽章是 UI 元素，
# 物理尺寸由设备像素密度决定，典型直径几十像素。它是**内容尺度**，
# 不随画布张数变化，因此用一个绝对值反而比用画布比例更准确。
DEFAULT_BADGE_D = 64.0
BADGE_D_MIN = 20.0
BADGE_D_MAX = 200.0

# 画布内存预算（RGB 三通道）。上限由「可用内存」和「输入总像素」共同决定，
# 而不是拍一个固定像素数：单张图不该占用多图预算，多图也不会因张数多被误杀。
BYTES_PER_PIXEL = 3
MAX_CANVAS_BYTES = 512 * 1024 * 1024
CANVAS_INPUT_FACTOR = 3.0   # 斜向阶梯排列时画布面积可达各图面积之和的 2~3 倍


def odd_kernel(value: float, minimum: int = 3) -> int:
    """把任意尺度换算成不小于 minimum 的奇数形态学核。"""
    k = max(minimum, int(round(value)))
    return k if k % 2 else k + 1


def canvas_pixel_budget(total_input_px: int) -> int:
    """按内存预算与输入规模推导画布像素上限，与具体张数/分辨率无关。"""
    memory_cap = MAX_CANVAS_BYTES // BYTES_PER_PIXEL
    input_cap = int(max(1, total_input_px) * CANVAS_INPUT_FACTOR)
    return min(memory_cap, input_cap)


@dataclass(frozen=True)
class Scale:
    """一张图像（原图或拼接画布）的几何尺度。"""

    width: int
    height: int
    badge_d: float

    # ---- 无量纲系数：按几何关系定义，与任何具体截图无关 ----
    LINE_PER_BADGE: ClassVar[float] = 0.14      # 轨迹线宽 / 徽章直径
    TOUCH_PER_BADGE: ClassVar[float] = 3.0      # 「贴住徽章」容差
    COARSE_PER_BADGE: ClassVar[float] = 6.0     # 包围盒粗筛半径
    AREA_PER_LINE: ClassVar[float] = 3.0        # 最小面积 = (k·线宽)^2
    CLOSE_PER_LINE: ClassVar[float] = 1.1       # 闭运算核 / 线宽
    OPEN_PER_LINE: ClassVar[float] = 0.5        # 开运算核 / 线宽
    SKELETON_PER_LINE: ClassVar[float] = 0.3    # 骨架前闭运算核 / 线宽
    CONF_PER_BADGE: ClassVar[float] = 3.0       # 置信度奖励的贴合距离

    TOPO_PER_DIAG: ClassVar[float] = 0.12       # 形状校验拓扑距离 / 画布对角线
    BRIDGE_PER_ROUTE: ClassVar[float] = 0.08    # 补缝上限 / 起终点跨度
    BRIDGE_PER_BADGE: ClassVar[float] = 12.0    # 补缝上限的绝对上界（内容尺度）
    BRIDGE_MIN_PER_BADGE: ClassVar[float] = 4.0 # 补缝下限，避免极短路线无处可补
    ZERO_PER_SHORT: ClassVar[float] = 0.005     # 0 位移容差 / 图短边
    ZERO_MIN: ClassVar[float] = 2.0

    @classmethod
    def from_image(
        cls, img: np.ndarray, badge_d: float | None = None
    ) -> Scale:
        """从图像尺寸 + 实测徽章直径构造尺度。"""
        h, w = img.shape[:2]
        d = DEFAULT_BADGE_D if badge_d is None else float(badge_d)
        d = min(BADGE_D_MAX, max(BADGE_D_MIN, d))
        return cls(width=int(w), height=int(h), badge_d=d)

    # ---- 标尺 ----
    @property
    def short(self) -> float:
        return float(min(self.width, self.height))

    @property
    def diag(self) -> float:
        return float(math.hypot(self.width, self.height))

    @property
    def line_w(self) -> float:
        """轨迹线宽（内容尺度）。"""
        return max(3.0, self.badge_d * self.LINE_PER_BADGE)

    # ---- 内容尺度派生量 ----
    @property
    def touch_dist(self) -> float:
        """连通域像素到徽章的最大距离，算「真的贴住」。"""
        return self.badge_d * self.TOUCH_PER_BADGE

    @property
    def coarse_dist(self) -> float:
        """包围盒粗筛：候选连通域至少要靠近某一端。"""
        return self.badge_d * self.COARSE_PER_BADGE

    @property
    def min_area(self) -> float:
        """连通域面积下限：小于若干个线宽见方的都是噪点。"""
        return (self.line_w * self.AREA_PER_LINE) ** 2

    @property
    def close_kernel(self) -> int:
        return odd_kernel(self.line_w * self.CLOSE_PER_LINE)

    @property
    def open_kernel(self) -> int:
        return odd_kernel(self.line_w * self.OPEN_PER_LINE)

    @property
    def skeleton_kernel(self) -> int:
        return odd_kernel(self.line_w * self.SKELETON_PER_LINE)

    @property
    def bridge_width(self) -> int:
        """接缝补线宽度 = 轨迹线宽，不随画布放大。"""
        return max(1, int(round(self.line_w)))

    @property
    def conf_touch(self) -> float:
        return self.badge_d * self.CONF_PER_BADGE

    @property
    def marker_lead(self) -> float:
        """徽章中心到轨迹端点的引线长度上限（内容尺度）。"""
        return self.badge_d * 2.0

    @property
    def zero_shift_tol(self) -> float:
        """「位移为 0」的容差：相对图短边，兼容不同分辨率。"""
        return max(self.ZERO_MIN, self.short * self.ZERO_PER_SHORT)

    # ---- 画布尺度派生量 ----
    @property
    def topo_dist(self) -> float:
        """形状校验中允许的两端距离之和，随画布对角线缩放。"""
        return self.diag * self.TOPO_PER_DIAG

    def bridge_gap(
        self, start_pt: tuple[int, int], end_pt: tuple[int, int]
    ) -> float:
        """允许补上的接缝空隙上限。

        依据是「空隙必须远小于整条路线的跨度」，因此随路线尺度自适应；
        同时用内容尺度给出上下界，避免极长/极短路线退化。
        """
        route = math.hypot(
            start_pt[0] - end_pt[0], start_pt[1] - end_pt[1]
        )
        gap = route * self.BRIDGE_PER_ROUTE
        gap = max(gap, self.badge_d * self.BRIDGE_MIN_PER_BADGE)
        return min(gap, self.badge_d * self.BRIDGE_PER_BADGE)
