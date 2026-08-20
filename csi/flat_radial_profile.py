# NOTE: Phase 5 (2D smoothed) 移行後、このモジュールで実際に使用されるのは
# `_compute_bayer_luminance` のみ。他の関数 (_detect_optical_center,
# _compute_radial_profile, _smooth_radial_profile, _build_radial_gain_map)
# は互換性のため残置している。
"""Radial profile parametric flat field gain map.

レンズ vignetting の物理モデル (radial 対称) に基づく色非依存 gain map を生成する。
scipy 非依存 (numpy.polyfit のみ使用)。

主な関数:
- `_compute_bayer_luminance`: Bayer 生画像から半解像度 luminance を計算
- `_detect_optical_center`: luminance 重心を光軸中心として検出 (フォールバックあり)
- `_compute_radial_profile`: radial distance でビン化し median profile を計算
- `_smooth_radial_profile`: numpy.polyfit で profile を平滑化
- `_build_radial_gain_map`: radial profile から 2D gain map (Bayer 原寸) を構築
"""

import numpy as np


def _compute_bayer_luminance(flat_clean: np.ndarray, bayer_pattern) -> np.ndarray:
    """Bayer 生画像から半解像度の luminance 画像を計算する。

    2×2 Bayer ブロック単位で (B + G1 + G2 + R) / 4 を計算し、
    半解像度 `(H/2, W/2)` の luminance を返す。

    Args:
        flat_clean: Bayer 原寸の flat 画像 (dark 減算済み)。
        bayer_pattern: `BayerPattern` enum。`channel_slices()` メソッドを持つ。

    Returns:
        半解像度 `(H/2, W/2)` の luminance (float64)。
    """
    _sl = bayer_pattern.channel_slices()
    bayer_slices = [_sl["B"], _sl["G1"], _sl["G2"], _sl["R"]]
    h, w = flat_clean.shape
    h_half, w_half = h // 2, w // 2
    accumulator = np.zeros((h_half, w_half), dtype=np.float64)
    for row_sl, col_sl in bayer_slices:
        ch = flat_clean[row_sl, col_sl][:h_half, :w_half].astype(np.float64)
        accumulator += ch
    return accumulator / 4.0


def _detect_optical_center(
    lum_img: np.ndarray,
    valid_mask_half: np.ndarray,
) -> tuple[float, float]:
    """Luminance の輝度重み付き重心を optical center として検出する。

    `valid_mask_half` 内の luminance 値を重みとして重心 `(cx, cy)` (半解像度座標) を計算する。
    重心が画像中心から画像サイズの 20% 以上ずれた場合は画像中心にフォールバックする。

    Args:
        lum_img: 半解像度の luminance 画像。
        valid_mask_half: 半解像度の有効領域マスク (bool)。

    Returns:
        `(cx, cy)` 半解像度座標。
    """
    h, w = lum_img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    masked_lum = np.where(valid_mask_half, lum_img, 0.0)
    total = float(masked_lum.sum())
    if total <= 0.0:
        print("⚠ optical center: valid_mask が空 - 画像中心にフォールバック")
        return (w / 2.0, h / 2.0)
    cx = float((masked_lum * xs).sum() / total)
    cy = float((masked_lum * ys).sum() / total)

    # フォールバック判定: 画像中心から画像サイズの 20 % 以上ずれたら画像中心を使う
    center_x, center_y = w / 2.0, h / 2.0
    threshold_x = w * 0.20
    threshold_y = h * 0.20
    if abs(cx - center_x) > threshold_x or abs(cy - center_y) > threshold_y:
        print(
            f"⚠ optical center ({cx:.1f}, {cy:.1f}) が画像中心 "
            f"({center_x:.1f}, {center_y:.1f}) から外れすぎ - 中心にフォールバック"
        )
        return (center_x, center_y)
    return (cx, cy)


def _compute_radial_profile(
    lum_img: np.ndarray,
    valid_mask_half: np.ndarray,
    cx_lum: float,
    cy_lum: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """半解像度 luminance から radial profile を計算する。

    `(cx_lum, cy_lum)` からの radial distance でピクセルを **等面積** ビンに分け、
    各 bin の median と std/mean (= 非 radial 残差比) を計算する。

    Args:
        lum_img: 半解像度の luminance 画像。
        valid_mask_half: 半解像度の有効領域マスク (bool)。
        cx_lum: 半解像度座標での optical center x。
        cy_lum: 半解像度座標での optical center y。
        n_bins: ビン数。

    Returns:
        タプル `(r_centers, profile_median, residual_ratio)`:
          - `r_centers`: 各 bin の代表 radial distance (float64, shape `(n_bins,)`)
          - `profile_median`: 各 bin の luminance の median (float64)
          - `residual_ratio`: 各 bin の std/mean (float64)
    """
    h, w = lum_img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - cx_lum) ** 2 + (ys - cy_lum) ** 2)

    # valid pixels の r と luminance を抽出
    valid = valid_mask_half.astype(bool)
    r_valid = r[valid]
    lum_valid = lum_img[valid]

    if r_valid.size == 0:
        # 空の場合はフォールバック
        return (
            np.zeros(n_bins, dtype=np.float64),
            np.zeros(n_bins, dtype=np.float64),
            np.zeros(n_bins, dtype=np.float64),
        )

    # 等面積 (面積比例) ビンエッジ: r² を等分する
    r_max = float(r_valid.max())
    r2_edges = np.linspace(0.0, r_max ** 2, n_bins + 1)
    r_edges = np.sqrt(r2_edges)

    r_centers = np.zeros(n_bins, dtype=np.float64)
    profile_median = np.zeros(n_bins, dtype=np.float64)
    residual_ratio = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        r_lo = r_edges[i]
        r_hi = r_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (r_valid >= r_lo) & (r_valid <= r_hi)
        else:
            in_bin = (r_valid >= r_lo) & (r_valid < r_hi)
        bin_values = lum_valid[in_bin]
        r_centers[i] = (r_lo + r_hi) / 2.0
        if bin_values.size == 0:
            profile_median[i] = np.nan
            residual_ratio[i] = np.nan
            continue
        profile_median[i] = float(np.median(bin_values))
        mean_val = float(np.mean(bin_values))
        std_val = float(np.std(bin_values))
        residual_ratio[i] = std_val / mean_val if mean_val > 1e-12 else np.nan

    return r_centers, profile_median, residual_ratio


def _smooth_radial_profile(
    r_centers: np.ndarray,
    profile_values: np.ndarray,
    deg: int = 5,
):
    """numpy.polyfit で radial profile を平滑化する (scipy 非依存)。

    nan を除いた (r, value) ペアに対して `numpy.polyfit(deg)` を実行し、
    `numpy.poly1d` 評価関数を返す。

    Args:
        r_centers: 各 bin の代表 radial distance。
        profile_values: 各 bin の luminance value (nan を含んでよい)。
        deg: 多項式の次数 (4-6 推奨、デフォルト 5)。

    Returns:
        `numpy.poly1d` 評価関数。`evaluator(r)` で smoothed value を返す。
    """
    mask = ~np.isnan(profile_values)
    r_valid = r_centers[mask]
    v_valid = profile_values[mask]
    if r_valid.size < deg + 1:
        # データ不足時は 0 次 (平均) にフォールバック
        mean_val = float(np.mean(v_valid)) if v_valid.size > 0 else 1.0
        return np.poly1d([mean_val])
    coeffs = np.polyfit(r_valid, v_valid, deg)
    return np.poly1d(coeffs)


def _build_radial_gain_map(
    bayer_shape: tuple[int, int],
    cx_lum: float,
    cy_lum: float,
    profile_eval,
    r_max: float,
) -> np.ndarray:
    """Radial profile から Bayer 原寸の 2D gain map を構築する。

    1. 半解像度 `(H/2, W/2)` のグリッドを生成する。
    2. 各画素の r を半解像度座標で計算する。
    3. `gain_half(x,y) = profile_max / profile_eval(r(x,y))` を計算する。
    4. `r > r_max` の画素は `gain = 1.0` に固定する (外挿しない)。
    5. `np.repeat` で Bayer 原寸 shape へ upsample する。

    Args:
        bayer_shape: Bayer 原寸の shape `(H, W)`。
        cx_lum: 半解像度座標の optical center x。
        cy_lum: 半解像度座標の optical center y。
        profile_eval: `_smooth_radial_profile` が返す `numpy.poly1d` 評価関数。
        r_max: 半解像度座標での有効な最大 r。

    Returns:
        Bayer 原寸 shape の gain map (float32)。全 Bayer チャネルに同じ値を持つ。
    """
    h_full, w_full = bayer_shape
    h_half, w_half = h_full // 2, w_full // 2
    ys, xs = np.mgrid[0:h_half, 0:w_half]
    r = np.sqrt((xs - cx_lum) ** 2 + (ys - cy_lum) ** 2)

    # profile の最大値 (profile_max) を r=0 付近で取得
    r_eval = np.linspace(0.0, r_max, 100)
    profile_samples = profile_eval(r_eval)
    profile_max = float(np.max(profile_samples))

    # gain 計算: profile_max / profile(r)
    profile_at_r = profile_eval(r.ravel()).reshape(r.shape)
    # profile が 0 以下の場合のガード
    safe_profile = np.where(profile_at_r > 1e-8, profile_at_r, profile_max)
    gain_half = profile_max / safe_profile

    # 外挿領域 (r > r_max) は gain = 1.0 に固定
    gain_half = np.where(r > r_max, 1.0, gain_half)

    # 半解像度 → Bayer 原寸 shape へ upsample
    gain_full = np.repeat(np.repeat(gain_half, 2, axis=0), 2, axis=1)

    # 念のため shape を合わせる (奇数 shape の場合の切り詰め)
    gain_full = gain_full[:h_full, :w_full]
    return gain_full.astype(np.float32)
