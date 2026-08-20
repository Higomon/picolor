"""Per-pixel luminance inverse flat field gain map.

Phase 7 以降、gain map は **per-pixel luminance 逆数方式** で構築される:

    gain(x, y) = global_mean_lum / lum_flat(x, y)

ここで `global_mean_lum` は `valid_mask_half` 内 luminance の平均、
`lum_flat` は半解像度 Bayer luminance `(B + G1 + G2 + R) / 4` である。

設計要点:
- **平滑化を行わない** — 低周波 vignetting のみでなく、高周波不均一も含めて
  pixel 単位で厳密に補正し、同一背景でのピクセル値を完全に均一化する。
- 色非依存 (luminance-only gain を全 Bayer ch に同一適用) — 全 ch に同一
  scalar を乗ずるため R/G/B 比率は保存され CCM 行列は不変。
- scipy 非依存。
- Ref anchor 正規化ロジックは Phase 6 と同一 (overlap 50 % 検査・
  near-zero mean skip・`ref_anchor_gain=None` 返却による calibration 失敗
  シグナル)。

運用上の留意 (Phase 7 固有):
- flat 画像の単フレーム per-pixel ノイズが gain に直接入る。`capture_flat_field`
  は複数フレーム平均 (実機 64 枚) を取ることで SNR を確保する。
- flat 取得時の dust / scratch は gain に焼き付くため、target 撮影時も同じ
  光学系位置で運用する前提。

主な関数:
- `_build_smoothed_gain_map`: per-pixel luminance 逆数で Bayer 原寸 gain map を構築
- `_smooth_luminance_2d_masked`: 本モジュール内では未使用。normalized convolution
  の参考実装として残置 (将来の再導入備え)。
"""

import cv2
import numpy as np


def _smooth_luminance_2d_masked(
    lum_img: np.ndarray,
    mask: np.ndarray,
    sigma_px: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """半解像度 luminance を **normalized convolution** で平滑化する。

    標準的な Gaussian blur は有効領域外 (黒縁、padding) の 0 値を
    そのまま畳み込むため、`valid_mask` 内部の境界近傍で輝度が
    不当に引き下げられ、後段で `smoothed_max / smoothed` を計算すると
    境界近傍で gain が過補正される (Codex 実測で 1.38 まで上昇)。

    normalized convolution (aka Knutsson-Westin 法) では以下を計算する:

        num   = Gauss( lum * mask )
        denom = Gauss( mask )
        smoothed = num / denom     (denom > eps)
                 = 0                (denom <= eps)

    分母で mask 面積を正規化するため、mask 外の 0 値が入り込んでも
    bias は打ち消され、有効領域内は「有効画素だけの局所平均」に収束する。
    astronomical image smoothing で一般的な手法。

    `cv2.GaussianBlur` の `ksize=(0, 0)` 指定で sigma から自動的に
    カーネルサイズを決定し、`BORDER_REFLECT` で端点を扱う。

    Args:
        lum_img: 半解像度 luminance `(H_half, W_half)`。float32/float64 いずれでも可。
        mask: 有効領域マスク (bool, `(H_half, W_half)`)。
        sigma_px: Gaussian 標準偏差 (半解像度座標系の画素単位)。
        eps: 分母の下限 (これ以下の領域は smoothed=0 にする)。

    Returns:
        平滑化済み luminance (float32)。shape は入力と同じ。mask ほぼ 0 の
        領域では 0 が入るが、呼び出し側で `valid_mask_half` により最終的に
        gain=1.0 に固定されるため問題ない。

    注: Phase 7 (per-pixel luminance 逆数方式) への移行により、
        本関数は本モジュール内で呼び出されない。normalized convolution
        の参考実装として将来の再導入に備えて残置している。
    """
    lum_f32 = lum_img.astype(np.float32)
    mask_f32 = mask.astype(np.float32)
    lum_masked = lum_f32 * mask_f32

    num = cv2.GaussianBlur(
        lum_masked,
        ksize=(0, 0),
        sigmaX=float(sigma_px),
        sigmaY=float(sigma_px),
        borderType=cv2.BORDER_REFLECT,
    )
    denom = cv2.GaussianBlur(
        mask_f32,
        ksize=(0, 0),
        sigmaX=float(sigma_px),
        sigmaY=float(sigma_px),
        borderType=cv2.BORDER_REFLECT,
    )

    smoothed = np.where(
        denom > eps,
        num / np.maximum(denom, eps),
        0.0,
    )
    return smoothed.astype(np.float32)


def _build_smoothed_gain_map(
    bayer_shape: tuple[int, int],
    lum_img: np.ndarray,
    valid_mask_half: np.ndarray,
    ref_rect_half: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, float | None]:
    """半解像度 luminance から Bayer 原寸の per-pixel luminance gain map を構築する。

    Phase 7 方式 (per-pixel luminance 逆数):

        global_mean_lum = mean( lum_img[valid_mask_half] )
        gain_half(x, y) = global_mean_lum / max(lum_img(x, y), 1e-6)

    平滑化は行わず、pixel 単位で厳密に補正する。全 Bayer ch に同じ scalar
    を乗ずるため R/G/B 比率は保存される。

    処理ステップ:
        1. `valid_mask_half` 内 luminance の平均を `global_mean_lum` として計算
           (valid_mask が空の場合は全体 mean にフォールバック)。
        2. `gain_half = global_mean_lum / max(lum_img, 1e-6)` を計算
           (per-pixel 逆数)。
        3. `valid_mask_half` 外は `gain = 1.0` に固定。
        4. `ref_rect_half` 指定時、Ref 矩形の gain 平均で `gain_half` を除算し、
           Ref ROI 位置での gain を 1.0 に正規化 (cross-substrate 再現性補強)。
        5. `np.clip(gain_half, 0.25, 4.0)` で gain を最終レンジに収める
           (Cycle 2 C4: anchor 後に clip して暗縁の gain 爆発を防止)。
        6. anchor 有効時、post-clip Ref 平均が 1.0 ± 1% 外なら
           `RuntimeError` を送出 (clip により anchor 契約が崩壊した検知)。
        7. `valid_mask_half` 外を再度 1.0 に固定 (anchor 除算で
           1/ref_mean に動いた 1.0 を戻す)。
        8. `np.repeat` で Bayer 原寸 shape へ upsample (2×2 block で同一値)。
        9. `bayer_shape` で trim して return。

    Args:
        bayer_shape: Bayer 原寸の shape `(H, W)`。
        lum_img: 半解像度 luminance `(H/2, W/2)`。
        valid_mask_half: 半解像度の有効領域マスク (bool, `(H/2, W/2)`)。
        ref_rect_half: Ref ROI を半解像度座標で表した `(rx, ry, rw, rh)`。
            指定時は Ref 矩形の gain 平均で全体を除算し、Ref 位置での
            gain を 1.0 に正規化する (Ref anchor 正規化)。`None` の場合は
            正規化を行わない (後方互換)。

    Returns:
        `(gain_full, ref_anchor_gain)` の tuple:
          - gain_full: Bayer 原寸 shape の gain map (float32)。
            全 Bayer チャネルに同じ値を持つ。`ref_rect_half` 指定時は
            Ref 矩形の gain 平均が 1.0 に正規化済み。
          - ref_anchor_gain: 正規化前の Ref 矩形 gain 平均 (float)。
            `ref_rect_half` が `None` または正規化スキップ時は `None`。
    """
    lum_f32 = lum_img.astype(np.float32)
    if valid_mask_half.any():
        global_mean_lum = float(lum_f32[valid_mask_half].mean())
    else:
        global_mean_lum = float(lum_f32.mean())
    if global_mean_lum <= 0.0:
        global_mean_lum = 1.0
    gain_half = global_mean_lum / np.maximum(lum_f32, 1e-6)
    gain_half = np.where(valid_mask_half, gain_half, 1.0).astype(np.float32)

    # Ref anchor 正規化: 半解像度 gain_half 段で Ref 矩形平均を 1.0 にする。
    # 全 Bayer ch を同一スカラで除算するため RGB 比率は不変 → CCM 行列は不影響。
    #
    # Cycle 2 C4 対策: clip を anchor の **後** に移動する。Cycle 1 では
    # clip を anchor 前に置いていたが、anchor `gain /= ref_mean` で
    # `ref_mean < 1` (Ref が明るく global_mean_lum / lum < 1 になる
    # 普通のケース) だと最終 gain が再拡大して 4.0 を超えてしまう
    # (Codex Cycle 2 実測: 4x4 で max gain=5.16)。clip を anchor 後に
    # 戻すことで「最終 gain は必ず [0.25, 4.0]」という不変条件を回復し、
    # Ref 平均が 1.0 から崩れた場合は post-clip ガード (下記) で検出する。
    ref_anchor_gain: float | None = None
    ref_slice_indices: tuple[int, int, int, int] | None = None
    ref_mask_slice: np.ndarray | None = None
    if ref_rect_half is not None:
        h_half, w_half = gain_half.shape
        rx, ry, rw, rh = ref_rect_half
        # 半解像度座標範囲 [0, w_half] / [0, h_half] にクリップ
        rx0 = int(max(0, min(int(rx), w_half)))
        ry0 = int(max(0, min(int(ry), h_half)))
        rx1 = int(max(0, min(int(rx) + int(rw), w_half)))
        ry1 = int(max(0, min(int(ry) + int(rh), h_half)))
        ref_slice = gain_half[ry0:ry1, rx0:rx1]
        if ref_slice.size == 0:
            print(
                "⚠ ref_rect_half 正規化スキップ: empty slice "
                f"(rect={ref_rect_half}, shape={(h_half, w_half)})"
            )
        else:
            # valid_mask_half と Ref rect の重なりを検査 (C2 対策)。
            # 重なりが空、または ref 面積の 50 % 未満しか有効でない場合、
            # ref_slice は invalid 領域の 1.0 固定値に強く引きずられて
            # 正規化が no-op 化するため、明示的に skip シグナル (None) を返す。
            mask_slice = valid_mask_half[ry0:ry1, rx0:rx1]
            overlap_count = int(mask_slice.sum())
            ref_area = int(ref_slice.size)
            if overlap_count == 0:
                print(
                    "⚠ ref_rect_half 正規化スキップ: no overlap with valid_mask "
                    f"(rect={ref_rect_half}, valid_overlap=0/{ref_area})"
                )
            elif overlap_count < ref_area * 0.5:
                print(
                    "⚠ ref_rect_half 正規化スキップ: insufficient valid coverage "
                    f"(rect={ref_rect_half}, valid_overlap="
                    f"{overlap_count}/{ref_area} < 50%)"
                )
            else:
                # valid_mask_half 内の画素のみで平均 (R1: Ref が黒縁に落ちる対策)
                ref_mean_preanchor = float(ref_slice[mask_slice].mean())
                if ref_mean_preanchor < 1e-6:
                    print(
                        "⚠ ref_rect_half 正規化スキップ: near-zero mean "
                        f"(ref_mean={ref_mean_preanchor:.6g})"
                    )
                else:
                    gain_half = (gain_half / ref_mean_preanchor).astype(np.float32)
                    ref_anchor_gain = ref_mean_preanchor
                    # post-clip ガードで再評価するため、Ref 矩形の座標と
                    # mask スライスを保持する。
                    ref_slice_indices = (ry0, ry1, rx0, rx1)
                    ref_mask_slice = mask_slice

    # Cycle 2 C4 対策: clip を anchor の後に適用し、最終 gain が必ず
    # [0.25, 4.0] に収まることを保証する。
    gain_half = np.clip(gain_half, 0.25, 4.0).astype(np.float32)

    # Cycle 2 C4 対策: post-clip Ref 平均ガード。anchor が効いている
    # ケースで、clip により Ref ROI が境界 (0.25 / 4.0) に張り付くと
    # 「Ref 平均 = 1.0」という anchor 契約が崩れる。post-clip Ref 平均が
    # 1.0 ± 1% 外であれば、clip と anchor の両立が pathological
    # (Ref ROI 内に dust / scratch が多数で clip がほぼ全画素に当たる)
    # ため、calibration を失敗扱いにして RuntimeError を送出する
    # (沈黙の破綻を防止)。
    if ref_anchor_gain is not None:
        assert ref_slice_indices is not None and ref_mask_slice is not None
        ry0, ry1, rx0, rx1 = ref_slice_indices
        ref_slice_postclip = gain_half[ry0:ry1, rx0:rx1]
        ref_mean_postclip = float(ref_slice_postclip[ref_mask_slice].mean())
        if abs(ref_mean_postclip - 1.0) > 0.01:
            raise RuntimeError(
                "Ref anchor が clip [0.25, 4.0] により崩壊しました: "
                f"post-clip Ref 平均={ref_mean_postclip:.4f} "
                f"(1.0 から {abs(ref_mean_postclip - 1.0) * 100:.2f}% 乖離 > 1%, "
                f"pre-clip ref_anchor_gain={ref_anchor_gain:.4f}, "
                f"rect={ref_rect_half})。"
                "Ref ROI 内の dust / scratch が多数で clip 境界に張り付いて "
                "いる可能性があります。Ref 位置の見直しまたは flat 再取得を "
                "推奨します。"
            )

    # anchor 除算は valid_mask 外の 1.0 固定値も 1/ref_mean に動かして
    # しまうため、mask 外は再度 1.0 に固定し直す。clip だけのケースでも
    # 1.0 は [0.25, 4.0] 内なので no-op になるが、契約を揃えるため一律実施。
    gain_half = np.where(valid_mask_half, gain_half, 1.0).astype(np.float32)

    gain_full = np.repeat(np.repeat(gain_half, 2, axis=0), 2, axis=1)
    h_full, w_full = bayer_shape
    gain_full = gain_full[:h_full, :w_full]
    return gain_full.astype(np.float32), ref_anchor_gain
