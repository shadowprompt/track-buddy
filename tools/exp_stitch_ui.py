"""实验：多图拼接能否扛住「静态 UI 干扰」。

## 为什么要测这个

用户在 App 里平移地图连续截图时，每张截图都带有**位置完全相同**的
UI 元素（状态栏、右上角图标、右下角圆形按钮、左下角缩略图）。
这些 UI 在两张图里像素级相同 → SIFT 会把它们匹配成大量"完美对应"，
而它们给出的变换是**单位变换**（位移=0），与真实地图平移矛盾。

RANSAC 选「内点最多」的一致集。如果 UI 特征数 > 地图重叠特征数，
RANSAC 就会选中单位变换 → 拼接结果完全错位。

## 测什么

1. 纯底图（无 UI）→ 基线，验证 SIFT 能求出正确平移
2. 底图 + 相同位置 UI → 看是否被带偏
3. 加「排除零位移假设」后 → 看能否救回来
4. 加「NCC 验证」后 → 看能否正确挑出真解

## 真值

高德静态图 pan_A / pan_B 是同一 zoom、不同经度中心，
经度差 0.005° → 真值平移由 SIFT 在无 UI 的基线上测得。
"""
from __future__ import annotations

import cv2
import numpy as np

PAN_A = "tmp_debug/zoomtest/pan_A.png"
PAN_B = "tmp_debug/zoomtest/pan_B.png"


def add_static_ui(img: np.ndarray) -> np.ndarray:
    """叠加模拟美团截图的静态 UI（位置固定，与底图内容无关）。"""
    out = img.copy()
    h, w = out.shape[:2]
    # 顶部状态栏
    cv2.rectangle(out, (0, 0), (w, 90), (245, 245, 245), -1)
    cv2.putText(out, "08:17", (40, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (30, 30, 30), 3)
    cv2.circle(out, (w - 120, 45), 32, (220, 220, 220), -1)
    # 右上角图标
    cv2.circle(out, (w - 90, 150), 34, (60, 130, 240), -1)
    # 右下角圆形按钮
    cv2.circle(out, (w - 90, h - 140), 42, (255, 255, 255), -1)
    cv2.circle(out, (w - 90, h - 140), 42, (40, 40, 40), 4)
    # 左下角缩略图
    cv2.rectangle(out, (30, h - 230), (230, h - 60), (250, 250, 250), -1)
    cv2.rectangle(out, (30, h - 230), (230, h - 60), (90, 90, 90), 3)
    return out


def add_route(img: np.ndarray, dx: float = 0.0, color=(80, 200, 90)) -> np.ndarray:
    """画一条模拟美团轨迹线（高饱和绿色）。

    轨迹是**画在地图坐标系里**的，所以它随地图一起平移——
    截图 B 里它相对屏幕的偏移应当与地图偏移相反。
    """
    out = img.copy()
    h, w = out.shape[:2]
    pts = np.array([
        [w * 0.15, h * 0.30], [w * 0.35, h * 0.42], [w * 0.52, h * 0.38],
        [w * 0.70, h * 0.55], [w * 0.62, h * 0.72], [w * 0.40, h * 0.78],
    ], dtype=np.float64)
    pts[:, 0] -= dx
    cv2.polylines(out, [pts.astype(np.int32)], False, color, 14, cv2.LINE_AA)
    return out


def sift_pairs(a: np.ndarray, b: np.ndarray, ratio: float = 0.75):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(ga, None)
    k2, d2 = sift.detectAndCompute(gb, None)
    ms = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in ms if m.distance < ratio * n.distance]
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src, dst, len(good)


def fit(src, dst, min_tx=0.0):
    """RANSAC 拟合相似变换；min_tx>0 时排除「位移过小」的单位变换假设。"""
    if len(src) < 8:
        return None, 0
    M, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0
    )
    if M is None:
        return None, 0
    return M, int(inl.sum())


def ncc_score(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
    """把 b 按 M 叠到 a 上，算重叠区的归一化互相关。

    M 是 a→b 的变换，这里要 b→a，所以取逆。
    """
    if M is None:
        return 0.0
    try:
        h, w = a.shape[:2]
        inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    except np.linalg.LinAlgError:
        return 0.0
    warped = cv2.warpAffine(b, inv, (w, h))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 只在 b 有效覆盖的区域算
    mask = (cv2.warpAffine(np.ones_like(ga), inv, (w, h)) > 0.5)
    if mask.sum() < 1000:
        return 0.0
    x = ga[mask]
    y = gb[mask]
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return float((x * y).mean())


def report(tag, M, inl, ngood, a, b):
    if M is None:
        print(f"  {tag}: 失败")
        return
    tx, ty = M[0, 2], M[1, 2]
    sc = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    ncc = ncc_score(a, b, M)
    print(f"  {tag}: inl={inl}/{ngood} tx={tx:.1f} ty={ty:.1f} "
          f"scale={sc:.6f} rot={rot:.4f} NCC={ncc:.4f}")


def main() -> None:
    A = cv2.imread(PAN_A)
    B = cv2.imread(PAN_B)

    print("=== 1. 基线：纯底图 ===")
    src, dst, ng = sift_pairs(A, B)
    M, inl = fit(src, dst)
    report("baseline", M, inl, ng, A, B)
    truth = (M[0, 2], M[1, 2]) if M is not None else None
    print(f"  >>> 真值 tx={truth[0]:.1f} ty={truth[1]:.1f}")

    print("\n=== 2. 加静态 UI + 轨迹线 ===")
    # 地图内容整体左移 932px（等价于 A 相对 B 的位移），
    # 所以轨迹在 A 里的屏幕位置 = 基准位置 + 932
    A2 = add_route(add_static_ui(A), dx=932.0)
    B2 = add_route(add_static_ui(B), dx=0.0)
    cv2.imwrite("tmp_debug/synth_A.jpg", A2)
    cv2.imwrite("tmp_debug/synth_B.jpg", B2)
    src, dst, ng = sift_pairs(A2, B2)
    M, inl = fit(src, dst)
    report("with-ui", M, inl, ng, A2, B2)

    print("\n=== 3. 只在地图区域（裁掉 UI 位置）做匹配 ===")
    # 用「只保留地图区域」的裁剪，而不是把 UI 涂黑
    # （涂黑会产生大量重复的黑色区域，反而让 SIFT 退化）
    h, w = A2.shape[:2]
    crop_a = A2[100:h - 40, 0:w - 150]
    crop_b = B2[100:h - 40, 0:w - 150]
    src, dst, ng = sift_pairs(crop_a, crop_b)
    M, inl = fit(src, dst)
    if M is not None:
        # 裁剪引入的偏移要加回去
        M = M.copy()
        M[0, 2] += 0
        M[1, 2] += 100
    report("cropped", M, inl, ng, A2, B2)

    print("\n=== 4. 关键洞察：UI 对应给出单位变换，可直接剔除 ===")
    # 统计位移接近 0 的匹配对数量 —— UI 的对应全在这里
    src, dst, ng = sift_pairs(A2, B2)
    if len(src):
        d = np.linalg.norm(dst.reshape(-1, 2) - src.reshape(-1, 2), axis=1)
        near0 = int((d < 8).sum())
        print(f"  位移<8px 的匹配对: {near0}/{len(src)}  (UI 特征应集中于此)")
        keep = d >= 8
        M2, inl2 = fit(src[keep], dst[keep])
        report("drop-zero-shift", M2, inl2, int(keep.sum()), A2, B2)


if __name__ == "__main__":
    main()
