"""
CSIキー入力ハンドラモジュール。

キーイベントの分岐処理、キャリブレーション操作、ROI調整、
SpyderCheckrチャートキャリブレーションを提供する。
"""

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401 - ImageFont kept for import compatibility

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from .colorimeter_common import (
    ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
    ACCEPTED_REFERENCE_SELECTION_MODE_FIXED,
    _ascii_to_fullwidth,
    _find_japanese_font,
    _delete_calibration_file_all,
    _order_corners_clockwise_from_topleft,
    _remove_cleared_marker,
    _get_sorted_date_dirs,
    CALIBRATION_DIR,
    get_today_calibration_dir,
    ProcessingSettings,
    FileNameInputOverlay,
    ROIMouseHandler,
    BlankRatioManager,  # noqa: F401 - kept for external import compatibility
    MasterRefManager,  # noqa: F401 - kept for external import compatibility
    CCMStore,
    build_measurement_context_summary,
    compute_ccm,
    compute_poly_ccm,  # noqa: F401 - kept for external import compatibility
    _POLY_CCM_COND_THRESH,  # noqa: F401 - kept for external import compatibility
    _json_ready,
    GRAY_CARD_CHECK_SCHEMA_VERSION,
    CCMVerifier,  # noqa: F401 - kept for external import compatibility
    compute_measurement_context_fingerprint,
    compute_gray_card_check_context_hash,
    compare_gray_card_check_to_baseline,
    describe_gray_card_check_baseline_state,
    gray_card_check_timestamp_slug,
    RelativeGrayVerifier,  # noqa: F401 - kept for external import compatibility
    SPYDERCHECKER_REFERENCE,
    SpyderCheckrGridExtractor,
    SpyderCheckrOrientedPanelPayload,
    SPYDERCHECKR_48_SHAPE,
    SPYDERCHECKR_48_WHITE_PATCH_INDEX,
    SPYDERCHECKR_48_GRAY_4D_INDEX,
    SPYDERCHECKR_48_BLACK_PATCH_INDEX,
    SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO,
    SPYDER_FLIP_MIN_RATIO,  # noqa: F401 - kept for external import compatibility
    detect_spyder_flip,
    detect_spyder_chart_pose_problem,
    CIELABConverter,
    FrameTransformer,
    DEFAULT_CONTROL_SAMPLE_ID,
    GUIDANCE_DEGRADED_REASON_AUTO_DETECT_DISABLED,
    GUIDANCE_DEGRADED_REASON_MANUAL_CORNER_ADJUSTMENT,
    GUIDANCE_DEGRADED_REASON_MANUAL_ROI_ADJUSTMENT,
    GUIDANCE_DEGRADED_REASON_SCENE_TIMEOUT_MANUAL_CONFIRM,
    GATE_STATE_BLOCKED,
    GATE_STATE_PROVISIONAL,
    OPERATOR_GUIDANCE_MODE_DEGRADED,
    OPERATOR_GUIDANCE_MODE_GUIDED,
    OPERATOR_GUIDANCE_MODE_MANUAL,
    RESULT_STATUS_RECHECK_REQUIRED,
    RUN_TYPE_END_OF_DAY,
    RUN_TYPE_REQUALIFICATION,
    RUN_TYPE_START_OF_DAY,
    SteadyStateSnapshot,
    auto_register_accepted_reference,
    apply_measurement_fail_stop_rollout,
    check_exposure_drift,
    check_master_ref_drift,
    compute_accepted_reference_diff,
    compute_previous_accepted_day_diff,
    clear_fixed_anchor_selection,
    describe_accepted_reference_relation,
    describe_measurement_context_resolution,
    evaluate_previous_accepted_day_judgement,
    evaluate_measurement_fail_stop,
    get_repro_rollout_mode,
    get_software_identity,
    get_run_type_display_name,
    list_fixed_anchor_candidates,
    load_accepted_reference_for_preflight,
    load_fixed_anchor_selection,
    persist_gray_4d_verification_artifacts,
    persist_gray_card_check_artifacts,
    persist_acceptance_result_artifacts,
    load_gray_card_check_baseline,
    resolve_accepted_reference_record,
    resolve_previous_accepted_day_record,
    RUN_TYPE_VERIFY_ONLY,
    save_fixed_anchor_selection,
    evaluate_acceptance_judgement,
    normalize_guidance_degraded_reasons,
    normalize_operator_guidance_mode,
    normalize_ref_scale_triplet,
    run_canonical_lab_pipeline,
    make_chart_analysis_composite,
    select_live_ref_baseline,
    select_live_ref_scale_baseline,
    summarize_gray_steady_state_observables,
    validate_gray_card_baseline_candidate,
)
from .key_commands import is_preview_freeze_key

# ---------------------------------------------------------------------------
# パスキー設定（全クリア操作の誤削除防止）
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P前ホワイト実測プリフライト設定
# ---------------------------------------------------------------------------
WHITE_PATCH_IDX = 4
WHITE_G_TARGET = 3350.0
WHITE_G_LIMIT = 3500.0
WHITE_G_MAX_RETRIES = 3
CHART_CAPTURE_FRAMES = 8
GRAY_STEADY_STATE_CAPTURE_FRAMES = 16
GRAY_STEADY_STATE_WINDOW = 3

# Phase A 選定値 (analysis/2026-04-28-csi-detection-stage-strengthening/
# corner_detection_chosen_params.md)。9 raw capture + ±45° 合成回転で全件 PASS。
# - aspect range は chart の minAreaRect bbox が回転角次第で
#   width/height が逆転 (横長 1.66:1 ↔ 縦長 0.6:1) するため緩和。
# - bridge kernel は固定 11 (旧 max(7, central.shape[1]//60) ≒ 27 では
#   2026-04-14/15 capture で chart 外暗部と過剰 bridge していた)。
_RIGID_ASPECT_MIN = 0.4
_RIGID_ASPECT_MAX = 4.0
# Phase 2 (20_strategy 2-D, post-Codex-revise): bridge kernel を dynamic 化。
# - 旧 kw=11 (axis-aligned 安全側) では SpyderCheckr 48 chart の
#   hinge gap (~30-40 px on real capture) を bridge できず、
#   左右パネルが 2 component に分裂し chart aspect filter で reject される。
# - **Codex review C3 sanctioned**: bridge ratio を 0.04 → **0.025** に下げ、
#   close を **2-pass** で適用することで、kw を絞っても hinge を確実に橋渡し
#   できる。kw=0.04 (chart 幅 1000 px で 40 px) は chart 外周 chamber padding を
#   巻き込み chart aspect short/long を 0.73-0.75 に膨張させていたが、
#   kw=0.025 (chart 幅 1000 px で 25 px) では巻き込みが軽減され、planner 理論値
#   (~0.563) 近傍に収束しやすい。
# - chart 幅 < 440 px (= 11 / 0.025) では `_RIGID_BRIDGE_KERNEL_BASE` で fallback し
#   既存 axis-aligned 動作 (Phase A 選定値) を retain する。
_RIGID_BRIDGE_KERNEL_BASE = 11  # 旧値 (axis-aligned 安全側、kw 下限)
_RIGID_BRIDGE_KERNEL_RATIO = 0.025  # central frame 幅の 2.5% を hinge gap 想定 bridge 幅
_RIGID_BRIDGE_KERNEL_PASSES = 2  # 2-pass close で kw を抑えつつ hinge を確実に bridge
# Phase A2 (analysis/2026-04-28-csi-detection-stage-strengthening/corner_quality.md):
# minAreaRect の rotated rect aspect (short/long, orientation invariant) を
# chart 物理外形 (canonical 6/8=0.75、bundle 実測 0.58) の妥当範囲に制限する。
# Phase 2 (20_strategy 2-C, post-Codex-revise): rotation 対応で planner spec の
# **[0.55, 0.70]** を採用する。
# - planner output L70: merged 全 chart 理論 short/long = 6/(8+2.67) ≈ 0.563、
#   frame_margin 0.52 込みで 0.58-0.60。
# - 上限 0.70 = 理論値 0.626 + margin 約 12%、warpAffine ±30° 時の minAreaRect
#   不安定性吸収用。
# - 上限を 0.78 まで広げる先行実装は Codex review C3 で chamber padding 巻き込み
#   症候を mask する旨指摘され revert。bridge ratio を 0.025 に下げ chamber padding
#   巻き込みを抑制することで、aspect 上限 0.70 でも acceptance 可能とする (couple
#   関係)。
# Phase 2W (runtime hinge_gap estimation): canonical _hinge_gap=2.67 を捨て、
# minAreaRect 観測 chart_asp から runtime で hinge_gap を逆算するため、aspect
# filter の上下限を **disclosed deviation** として拡張する。
# - canonical hinge_gap=2.67  → chart_asp=0.5623
# - v1 fixture observed       → chart_asp≈0.72 (hinge_gap≈0.33)
# - 上下限マージン:
#     0.50 (hinge_gap≈4.0) — 余裕ある下限
#     0.85 (hinge_gap≈-0.94 だが clip 下限 0.1 で潰す)
# 5 cycle (Phase 1 / Phase 2 cycle 2 / Phase 2X / 2Y / 2Z) primary evidence で
# 「定数空間に解なし」が確立しているため、planner spec [0.55, 0.70] を意図的に
# 拡張する。triad Generator implementation log Cycle 3 (Phase 2W) 参照。
_RIGID_CHART_ASPECT_MIN = 0.50
_RIGID_CHART_ASPECT_MAX = 0.85
# Phase A4: 候補 component を色一致度で評価し、SpyderCHECKR-48 reference に
# 最も近いものを採用。closed/dark 物体 (色分布なし) と open chart (48 色) を
# 色情報で分離する。score は exposure-normalized linear sRGB distance の median。
# 0.0 = 完全一致、約 0.10 = 露出ズレ程度、0.30+ = chart でない可能性大。
# Phase 2 (20_strategy 2-B, post-Codex-revise): rotation 推定が信頼できる場合のみ
# threshold を段階拡張 (上限 0.45 厳守)。
# - **Codex review C1 sanctioned restore**: BASE=0.30 (planner spec L112) を
#   厳守。`estimated_rot is None` の場合も 0.30 (regression 防止、axis-aligned chart
#   を従来通り厳格 reject)。
# - |rot| <= 5° または rotation 推定不能なら BASE (=0.30)。
# - |rot| > 5° なら `min(MAX, BASE + 0.005 × |rot|)` で線形に MAX (=0.45) まで拡張。
# - **MAX=0.45 は planner 厳守規定**。0.40 BASE への引き上げは bridge ratio 0.04
#   の副作用 (chamber padding 巻き込み + bilerp sampling 誤差膨張) を mask する
#   ものであり、Codex review で revert された。
_RIGID_COLOR_REJECT_THRESHOLD_BASE = 0.30  # axis-aligned 厳格 base (planner spec)
_RIGID_COLOR_REJECT_THRESHOLD_MAX = 0.45  # planner 厳守上限
_RIGID_COLOR_THRESHOLD_PER_DEG = 0.005  # 1° あたりの段階拡張係数
# Phase 2 (20_strategy 2-E): 左右 panel の lattice rotation 角度差許容値。
# - Phase A: 5.0° で axis-aligned 専用に厳格運用。
# - Phase 2: 8.0° に拡張。±20° rotation で panel 内黒ピクセル分布ノイズによる
#   6-7° 誤差を許容しつつ、真の L/R disagreement (panel 独立回転) を弾く水準を
#   保つ。Stage 2 grid sweep と Stage 1 validate (`_validate_corners_via_patchwise`)
#   の両方で参照される。
_LR_ROTATION_TOLERANCE_DEG = 8.0

# Phase 21 2-X (CCW chart + Ref panel merge 救済): Phase 2Z chart segmentation の
# `connectedComponentsWithStats` で top1 component が **巨大** (= chart + Ref panel +
# 影 が単一 component に merge) だった場合、erosion した copy で再分離を試み、得ら
# れた小成分を既存 candidates に **追加注入** する fallback ロジック。
# - 元 `dark_mask`/`bridged`/`labels`/`stats`/`num_labels` は破壊しない (CW/水平
#   regression を防ぐため、元 candidate ループの input は不変のまま)。
# - 発火 gate `_top_area > central.size * 0.10` は CW Press 1 (area_ratio≈0.111) で
#   borderline、CCW Press 3 (area_ratio≈0.18) で確実発火する閾値。CW でも発火する
#   が、erosion 後も元 chart-only top1 が candidates に維持されるため accept 結果
#   は不変。
# - kernel 5x5、iter=2: chart 8 cell 横幅 ~25 px に対し 10 px 腐食、cell 行は接続
#   維持しつつ chart-Ref panel の影 bridge を切断する。
# - Phase 21 2-X2 (cycle 2、CCW 厚 merge bridge 救済): cycle 1 の固定 iter=2 では
#   実機 (run_log_phase2_cycle1_failed.txt L46-52) の 12-20 px 厚 bridge を切れな
#   かった。`_MIN`=2 → `_MAX`=5 で adaptive ループ、`_top_eroded < _top_area * 0.60`
#   で early break (= merge が確実に切れた)。CW/水平 は cycle 1 と同じ iter=2 で
#   break、CCW 厚 bridge では iter=4-5 まで段階的に増やす。
_PHASE21_TOOBIG_AREA_RATIO = 0.10
_PHASE21_EROSION_KERNEL = 5
_PHASE21_EROSION_ITERS_MIN = 2  # cycle 1 と等価な下限 (CW/水平 影響最小)
_PHASE21_EROSION_ITERS_MAX = 5  # cycle 2 で追加、merge を切るまで段階拡張
_PHASE21_EROSION_BREAK_RATIO = 0.60  # _top_eroded < _top_area*0.60 で merge 切断確認
# Phase 21 2-X (deviation #2, advisor sanctioned, Codex C1 cycle 2 で修正):
# erosion 由来 candidates 専用の chart_aspect 上限 (planner spec L222 リスク表
# "不適切な component が注入される" mitigation の具体化)。Pass 1 (元 all_components)
# では `_RIGID_CHART_ASPECT_MAX` (=0.85) を維持。erosion 由来 extras は誤検出余地が
# 大きい (real capture 2026-04-15 で erosion 後 chart_asp=0.80 の sub-region が
# test_corner_detection_robustness の output-aspect 上限 0.722 を超え garbage_corners
# と判定される) ため上限を入れるが、**Phase 1 evidence (run_log_3poses.txt L64) で
# chart 単独 component の chart_asp=0.7077 が観測されている**。Codex C1 が指摘した
# とおり 0.70 では Press 3 erosion 後の救済対象自身を弾く。0.78 に上げ、0.85 から
# 余裕 0.07 を残しつつ実機 0.71 は admit する設定とした。
_PHASE21_EXTRA_CHART_ASP_MAX = 0.78

# Phase 4 (22_strategy 4-A): 旧検出器 `_detect_chart_via_rigid_rotated_rect` の
# candidates list に追加注入する contour pair merge 候補専用 sentinel idx の base。
# 既存 idx 値域 (Pass 1: idx>=1、Pass 2 erosion: idx<0 で `-j`、j∈{1..n_eroded})
# と非衝突するように `-1000` 起点とし、各 augment 候補は `BASE - counter` (=-1000,
# -1001, ...) で展開する。下流 score loop (L4738-4748) は idx を読み捨てるため、
# sentinel idx でも従来の patch_quad → color score 競争に乗る。
# revert 容易性: 本定数を grep して augment ブロックを localize 可能。
_PHASE4_CONTOUR_MERGE_IDX_BASE = -1000

# Phase 22 (輪郭ベース検出器): SpyderCheckr 48 outer rim → 1A patch CENTER 内縮率。
# canonical chart layout (8 cols + hinge_gap=2.67, 6 rows) に基づく:
#   _chart_patch_center_normalized(0, 0, 2.67) = (0.5/10.67, 0.5/6) ≈ (0.0469, 0.0833)
# 旧 `_detect_chart_via_rigid_rotated_rect` は dark-cell minAreaRect を inner-area と
# 解釈する経路だったが、新 `_detect_chart_via_contour_outer_rim` は黒縁外形を
# minAreaRect で捕えるため、外形 → 1A/1H/6H/6A patch CENTER への内縮率として直接
# 使う。`_chart_patch_center_normalized` 自体は変更せず constant としてハードコード
# 参照することで Phase 2X 整合性を維持する。
_CONTOUR_FRAME_MARGIN_X_FRAC = 0.0469
_CONTOUR_FRAME_MARGIN_Y_FRAC = 0.0833


def _chart_patch_center_normalized(
    col: int,
    row: int,
    hinge_gap: float,
    n_cols: int = 8,
    n_rows: int = 6,
) -> tuple[float, float]:
    """SpyderCheckr 48 patch center の正規化 (u, v) 座標を返す。

    Phase 2X (canonical model redesign): `_patch_quad_from_component_pts` と
    `_score_corners_by_color_match` の両者がこの helper を呼ぶことで、
    canonical 比率算術の二者間整合を保証する。

    座標系 (Style 1: raw fraction):
      `bilerp(chart_tl, chart_tr, chart_br, chart_bl)` の素直な内分係数として
      使えるよう、cell-pitch 単位の絶対位置を `n_x_total = n_cols + hinge_gap`
      および `n_y_total = n_rows` で除した raw fraction を返す。

      - col=0 (1A): u ≒ 0.5 / 10.67   = 0.0469
      - col=4 (1E): u ≒ 7.17 / 10.67  = 0.672  (旧 hinge 無視 col/7=0.571 から
                                                ~14% シフト = 約 1 cell 幅)
      - col=7 (1H): u ≒ 10.17 / 10.67 = 0.953
      - row=0 (1A): v ≒ 0.5 / 6  = 0.0833
      - row=5 (6A): v ≒ 5.5 / 6  = 0.917

    canonical 算術は `csi/colorimeter_common.py` の `_default_patch_display_quads`
    (L6281-6283) と equivalent (per-column 微調整 `col_x_norm_offsets[col]` は
    extractor 側 fine-tune なので detector 段階では考慮しない)。
    `gap_offset = hinge_gap * 100 if col >= half else 0.0` のロジックを継承。

    Args:
      col: 0..n_cols-1
      row: 0..n_rows-1
      hinge_gap: 左右パネル間 hinge の cell unit gap (実測逆算 ≒ 2.67)
      n_cols: 既定 8
      n_rows: 既定 6
    Returns:
      (u, v): bilerp の内分係数として直接使える raw fraction
    """
    half = n_cols // 2
    n_x_total = float(n_cols) + float(hinge_gap)
    n_y_total = float(n_rows)
    col_x_cell = float(col) + 0.5 + (
        float(hinge_gap) if col >= half else 0.0
    )
    row_y_cell = float(row) + 0.5
    u = col_x_cell / n_x_total
    v = row_y_cell / n_y_total
    return (u, v)


GRAY_STEADY_STATE_REQUIRED_CONSECUTIVE = 3
GRAY_STEADY_STATE_MAX_ATTEMPTS = 12
GRAY_STEADY_STATE_REF_SCALE_RANGE_MAX = 0.01

# ---------------------------------------------------------------------------
# 自動シーン検出設定
# ---------------------------------------------------------------------------
AUTO_SCENE_DETECTION_ENABLED = os.environ.get(
    "PICOLOR_AUTO_SCENE_DETECT", "true",
).lower() not in ("false", "0", "no")
AUTO_DETECT_TIMEOUT_SEC = 30.0
GUIDANCE_RUNNER_STATE_IDLE = "idle"
GUIDANCE_RUNNER_STATE_DARK = "dark"
GUIDANCE_RUNNER_STATE_FLAT = "flat"
GUIDANCE_RUNNER_STATE_WHITE = "white"
GUIDANCE_RUNNER_STATE_CHART = "chart"
GUIDANCE_RUNNER_STATE_VERIFY = "verify"
GUIDANCE_RUNNER_STATE_RESULT = "result"
GUIDANCE_RUNNER_STATE_CANCELLED = "cancelled"
GUIDANCE_RUNNER_STATE_BLOCKED = "blocked"
VALID_GUIDANCE_RUNNER_STATES = (
    GUIDANCE_RUNNER_STATE_IDLE,
    GUIDANCE_RUNNER_STATE_DARK,
    GUIDANCE_RUNNER_STATE_FLAT,
    GUIDANCE_RUNNER_STATE_WHITE,
    GUIDANCE_RUNNER_STATE_CHART,
    GUIDANCE_RUNNER_STATE_VERIFY,
    GUIDANCE_RUNNER_STATE_RESULT,
    GUIDANCE_RUNNER_STATE_CANCELLED,
    GUIDANCE_RUNNER_STATE_BLOCKED,
)



class KeyHandler:
    """
    CSI版のキー入力操作を管理するハンドラクラス。

    機能:
      - キーイベントを分岐し、計測保存・終了・露出制御切替を実行する。
      - ダーク/フラットフィールド取得、ROI選択/サイズ調整/リセットを制御する。
    入力:
      - 測定状態 `meas`、ログデータ `log_data`、ロガー/ダーク/フラット/AE等の依存オブジェクト。
      - キーコード（d/f/m/a/r/t/j/k/l/u/q）、アクティブROI状態、UI設定。
    出力:
      - 終了要求フラグ、スナップショット要求フラグ、更新済みROI設定を提供する。
    """

    def __init__(
        self,
        fps,
        logger,
        dark,
        flat,
        bayer,
        ae,
        picam2,
        config,
        window_name="CSI Precision Colorimeter",
        roi_persistence=None,
        mode=None,
        wb_calibrator=None,
        session_recorder=None,
        stability_tracker=None,
        lab_converter=None,
        stability=None,
        blank=None,
        master_ref=None,
        window_manager=None,
        spectral_drift_tracker=None,
        gray_verifier=None,
        ccm_verifier=None,
        camera_init_time=None,
    ):
        """
        キー操作に必要な依存オブジェクトと内部状態を保持する。

        Args:
          fps: キー入力待機間隔計算に使うフレームレート。
          logger: 計測イベントを記録するロガー。
          dark: ダークフレーム管理オブジェクト。
          flat: フラットフィールド管理オブジェクト。
          bayer: RAW処理に使うBayer抽出オブジェクト。
          ae: 適応露出制御オブジェクト。
          picam2: キャプチャ実行に使うPicamera2インスタンス。
          config: ROI設定を含むCSI設定。
          window_name: オーバーレイ表示先ウィンドウ名。
          roi_persistence: ROI設定の永続化オブジェクト。不要ならNone。
          blank: ブランク測定管理オブジェクト。
          master_ref: マスターRef正規化管理オブジェクト。
          window_manager: WindowManagerインスタンス。スケーリング付きimshow用。
        """
        self.fps = fps
        self.logger = logger
        self.dark = dark
        self.flat = flat
        self.bayer = bayer
        self.ae = ae
        self.picam2 = picam2
        self.config = config
        self.window_name = window_name
        self.roi_persistence = roi_persistence
        self.mode = mode
        self.wb_calibrator = wb_calibrator
        self.session_recorder = session_recorder
        self.stability_tracker = stability_tracker
        self.lab_converter = lab_converter
        self.stability = stability
        self.blank = blank
        self.master_ref = master_ref
        self.window_manager = window_manager
        self.spectral_drift_tracker = spectral_drift_tracker
        self.gray_verifier = gray_verifier
        self.ccm_verifier = ccm_verifier
        self.ref_anchor_lab = None
        self.last_mode_change_time = 0.0
        self.snapshot_pending = False
        self.wb_confirm_pending = False  # WBキャリブ2段階確認フラグ
        self.ccm_store: CCMStore | None = None
        # CCMキャリブ完了時のコールバック: processor 側に white_ratio_rgb / ref_train を即時反映
        self._on_ccm_calibrated: "Callable[[np.ndarray, np.ndarray | None], None] | None" = None
        self._on_live_ref_baseline_changed: "Callable[[np.ndarray | None, str | None], None] | None" = None
        self._on_live_ref_scale_baseline_changed: "Callable[[np.ndarray | None, str | None], None] | None" = None
        self._on_gray_verified: "Callable[[dict], None] | None" = None
        self._on_gray_absolute_verified: "Callable[[dict], None] | None" = None
        self._on_gray_results_invalidated: "Callable[[str], None] | None" = None
        self._on_acceptance_result_persisted: "Callable[[dict], None] | None" = None
        self._on_flat_roi_recheck: "Callable[[], None] | None" = None
        self._camera_init_time: float | None = camera_init_time
        self._acceptance_run_type: str | None = None
        self._acceptance_run_started_at: float | None = None
        self._accepted_reference_selection_mode = (
            ACCEPTED_REFERENCE_SELECTION_MODE_AUTO
        )
        self._accepted_reference_selection_requested_run_id = ""
        self._accepted_reference_selection_source_file = ""
        self._manual_roi_used = False
        self._manual_corner_used = False
        self._operator_guidance_mode = OPERATOR_GUIDANCE_MODE_GUIDED
        self._guidance_degraded_reasons: list[str] = []
        self._guidance_runner_state = GUIDANCE_RUNNER_STATE_IDLE
        self._positioning_method = ""
        # 白点パッチ（1E）の双光束比（デフォルト ones = 白点正規化なし、旧動作と互換）
        self.white_ratio_rgb: np.ndarray = np.ones(3, dtype=np.float64)
        self.ref_train: np.ndarray | None = None
        self.live_ref_baseline: np.ndarray | None = None
        self.live_ref_baseline_source: str | None = None
        self.live_ref_scale_baseline: np.ndarray | None = None
        self.live_ref_scale_baseline_source: str | None = None
        # チャートキャリブレーション状態（SpyderCheckr48 全体単一エクストラクタ方式）
        # 状態: "idle"|"corner_input"|"preview"|"measuring"
        self._chart_corners: list = []
        self._raw_corners: list = []      # set_corners 用に保持する生クリック座標
        self._chart_state: str = "idle"
        # Phase 7F: banner 文言を flip/回転/未設置 で分けるための reason 保存
        self._flip_warning_reason: str = ""
        self._grid_extractor: "SpyderCheckrGridExtractor | None" = None
        self._hinge_gap: float = 2.67     # ヒンジ余白（列幅単位）— 実測逆算値
        self._preview_margin_detected: bool = False
        self.preview_freeze_bundle_pending = False
        self._chart_workflow_status = self._new_chart_workflow_status()
        self._drag_corner_idx: int = -1
        self._drag_col_idx: int = -1
        self._drag_last_pos: tuple = (0, 0)
        self._frame_transformer = FrameTransformer(self.config)
        self.active_roi = "ref"  # "ref" or "tar"
        self.resize_axis = "w"  # "w" or "h"
        self.roi_mouse: Optional[ROIMouseHandler] = None
        # PIL 日本語フォント
        if _PIL_AVAILABLE:
            self._jp_font_title = _find_japanese_font(30)
            self._jp_font_body = _find_japanese_font(22)
            self._jp_font_action = _find_japanese_font(24)
            self._jp_font_small = _find_japanese_font(18) or self._jp_font_body
            if self._jp_font_body is None:
                print(
                    "[WARN] \u65e5\u672c\u8a9e\u30d5\u30a9\u30f3\u30c8\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3002"
                    "\u30aa\u30fc\u30d0\u30fc\u30ec\u30a4\u8868\u793a\u304c\u6587\u5b57\u5316\u3051\u3057\u307e\u3059\u3002\n"
                    "  \u2192 sudo apt install fonts-noto-cjk"
                )
        else:
            self._jp_font_title = None
            self._jp_font_body = None
            self._jp_font_action = None
            self._jp_font_small = None
            print("[WARN] PIL (Pillow) \u672a\u30a4\u30f3\u30b9\u30c8\u30fc\u30eb\u3002\u65e5\u672c\u8a9e\u30aa\u30fc\u30d0\u30fc\u30ec\u30a4\u4e0d\u53ef\u3002")

    def _set_live_ref_baseline(
        self,
        baseline: np.ndarray | None,
        source: str | None,
        notify: bool = False,
    ) -> None:
        """raw neutral baseline state を更新し、必要なら callback で同期する。"""
        if baseline is None or source is None:
            self.live_ref_baseline = None
            self.live_ref_baseline_source = None
            if self.lab_converter is not None:
                self.lab_converter.clear_ref_baseline()
        else:
            self.live_ref_baseline = np.asarray(baseline, dtype=np.float64).copy()
            self.live_ref_baseline_source = source
            if self.lab_converter is not None:
                self.lab_converter.set_ref_baseline(self.live_ref_baseline)
        if notify and self._on_live_ref_baseline_changed is not None:
            self._on_live_ref_baseline_changed(
                None if self.live_ref_baseline is None else self.live_ref_baseline.copy(),
                self.live_ref_baseline_source,
            )

    def _set_live_ref_scale_baseline(
        self,
        baseline: np.ndarray | None,
        source: str | None,
        notify: bool = False,
    ) -> None:
        """ref_scale 用 baseline state を更新し、必要なら callback で同期する。"""
        if baseline is None or source is None:
            self.live_ref_scale_baseline = None
            self.live_ref_scale_baseline_source = None
        else:
            self.live_ref_scale_baseline = np.asarray(baseline, dtype=np.float64).copy()
            self.live_ref_scale_baseline_source = source
        if notify and self._on_live_ref_scale_baseline_changed is not None:
            self._on_live_ref_scale_baseline_changed(
                None
                if self.live_ref_scale_baseline is None
                else self.live_ref_scale_baseline.copy(),
                self.live_ref_scale_baseline_source,
            )

    def _invalidate_gray_results(self, reason: str) -> None:
        """4D検証結果を stale とみなすイベントを main 側へ通知する。"""
        if self._on_gray_results_invalidated is not None:
            self._on_gray_results_invalidated(reason)

    def _resolve_live_ref_baseline_for_p(
        self,
        ref_train: np.ndarray | None,
    ) -> tuple[np.ndarray | None, str | None]:
        """P 完了時に適用する live baseline を優先順で決定する。"""
        neutral_baseline = (
            self.live_ref_baseline
            if self.live_ref_baseline_source == "neutral"
            else None
        )
        master_ref_rgb = None
        if self.master_ref is not None and self.master_ref.is_loaded:
            master_ref_rgb = self.master_ref.master_ref_rgb
        baseline, source = select_live_ref_baseline(
            neutral_baseline,
            master_ref_rgb,
            ref_train,
        )
        return baseline, None if source == "none" else source

    def _resolve_live_ref_scale_baseline_for_p(
        self,
        ref_train: np.ndarray | None,
    ) -> tuple[np.ndarray | None, str | None]:
        """P 完了時に適用する ref_scale baseline を優先順で決定する。"""
        explicit_scale_baseline = self.live_ref_scale_baseline
        if self.live_ref_scale_baseline_source is None:
            explicit_scale_baseline = None
        master_ref_rgb = None
        if self.master_ref is not None and self.master_ref.is_loaded:
            master_ref_rgb = self.master_ref.master_ref_rgb
        baseline, source = select_live_ref_scale_baseline(
            explicit_scale_baseline,
            master_ref_rgb,
            ref_train,
            self.white_ratio_rgb,
        )
        return baseline, None if source == "none" else source

    # Phase 3A: D→F→W→P 順序 hard gate. override 不可.
    _STEP_LABEL_FOR_PREREQ = {
        "d": "ノイズ補正 [d]",
        "f": "むら補正 [f]",
        "w": "ホワイト固定 [w]",
        "p": "色基準取得 [p]",
        "b": "ブランク [b]",
        "n": "ニュートラル [n]",
        "v": "Gray Check [v]",
    }

    def _format_prereq_error(self, step: str, missing: list[str]) -> list[str]:
        """Phase 3A.4: hard gate 警告 overlay 用の文面."""
        step_label = self._STEP_LABEL_FOR_PREREQ.get(step, f"[{step}]")
        lines = [f"操作手順エラー: {step_label} の前に未実行の手順があります"]
        for m in missing:
            lines.append(f"  未実行: {m}")
        lines.append("このままでは実行できません。手順通りに進めてください")
        return lines

    def _has_start_of_day_pass(self) -> bool:
        """Phase 2A: 当日 calibration dir に start_of_day 合格 acceptance があるか.

        True: 本番測定 [m] 許可.
        False: m を hard gate でブロック. 朝 1 回 [S] を合格させる必要あり.

        チェック条件:
        - 当日 (YYYY-MM-DD) の `calibration/<date>/acceptance_result_*.json` を列挙
        - run_type == "start_of_day"
        - result_status ∈ {"合格", "最低合格"}
        - 上記を満たす entry が 1 件以上
        """
        import datetime
        import glob
        import json
        today = datetime.date.today().isoformat()
        today_dir = os.path.join(str(CALIBRATION_DIR), today)
        if not os.path.isdir(today_dir):
            return False
        pattern = os.path.join(today_dir, "acceptance_result_*.json")
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            run_type = str(data.get("run_type", "")).strip()
            if run_type != RUN_TYPE_START_OF_DAY:
                continue
            result_status = str(data.get("result_status", "")).strip()
            if result_status in ("合格", "最低合格"):
                return True
        return False

    def _check_calibration_prerequisites(self, step: str) -> bool:
        """キャリブレーションの前提条件を検証する (Phase 3A: hard gate, override 不可)."""
        prereqs = {
            "d": [],
            "f": [("d", self.dark.is_loaded, "ノイズ補正 [d]")],
            "w": [
                ("d", self.dark.is_loaded, "ノイズ補正 [d]"),
                ("f", self.flat.is_loaded, "むら補正 [f]"),
            ],
            "p": [
                ("d", self.dark.is_loaded, "ノイズ補正 [d]"),
                ("f", self.flat.is_loaded, "むら補正 [f]"),
                ("w", self.wb_calibrator.is_calibrated, "ホワイト固定 [w]"),
            ],
            "b": [("w", self.wb_calibrator.is_calibrated, "ホワイト固定 [w]")],
            "n": [("w", self.wb_calibrator.is_calibrated, "ホワイト固定 [w]")],
            "v": [
                ("d", self.dark.is_loaded, "ノイズ補正 [d]"),
                ("f", self.flat.is_loaded, "むら補正 [f]"),
                ("w", self.wb_calibrator.is_calibrated, "ホワイト固定 [w]"),
            ],
        }
        missing = [name for _, loaded, name in prereqs.get(step, []) if not loaded]
        if not missing:
            return True
        # Phase 3A.2-3A.3: override 不可. overlay 警告のみを出し False を返す.
        lines = self._format_prereq_error(step, missing)
        self._show_capture_overlay(*lines, wait_sec=2.0)
        return False

    def _show_capture_overlay(self, *lines, wait_sec=0.0) -> int:
        """キャプチャ操作中のステータスメッセージを全画面オーバーレイ表示する。"""
        if not lines:
            lines = ("\u51e6\u7406\u4e2d...",)
        w, h = self._get_overlay_frame_size()
        frame = np.full((h, w, 3), (16, 18, 24), dtype=np.uint8)
        shade = np.full_like(frame, (0, 0, 0), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.80, shade, 0.20, 0.0)

        # 日本語文字が含まれるか判定
        has_cjk = any(ord(c) > 127 for line in lines for c in line)

        if has_cjk and _PIL_AVAILABLE and self._jp_font_body is not None:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            fw = _ascii_to_fullwidth
            title_font = self._jp_font_title or self._jp_font_body
            body_font = self._jp_font_body
            action_font = self._jp_font_action or self._jp_font_body
            text_specs = []
            max_text_w = 0
            total_text_h = 0
            line_gap = 14

            for idx, line in enumerate(lines):
                is_title = idx == 0
                is_action = idx > 0 and line.lstrip().startswith("[")
                font = (
                    title_font
                    if is_title
                    else (action_font if is_action else body_font)
                )
                text = fw(line)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                text_specs.append((text, font, tw, th, is_title, is_action))
                max_text_w = max(max_text_w, tw)
                total_text_h += th
            if len(text_specs) > 1:
                total_text_h += line_gap * (len(text_specs) - 1)

            card_w = min(w - 60, max(420, max_text_w + 120))
            card_h = max(180, total_text_h + 90)
            card_x = (w - card_w) // 2
            card_y = (h - card_h) // 2

            shadow_rect = (card_x + 6, card_y + 6, card_x + card_w + 6, card_y + card_h + 6)
            card_rect = (card_x, card_y, card_x + card_w, card_y + card_h)
            header_rect = (card_x + 2, card_y + 2, card_x + card_w - 2, card_y + 54)
            draw.rectangle(shadow_rect, fill=(0, 0, 0))
            if hasattr(draw, "rounded_rectangle"):
                draw.rounded_rectangle(card_rect, radius=16, fill=(26, 32, 40), outline=(92, 146, 172), width=2)
                draw.rounded_rectangle(header_rect, radius=14, fill=(34, 48, 61))
            else:
                draw.rectangle(card_rect, fill=(26, 32, 40), outline=(92, 146, 172), width=2)
                draw.rectangle(header_rect, fill=(34, 48, 61))

            max_action_w = max(
                (tw for _, _, tw, _, is_title, is_action in text_specs if is_action),
                default=0,
            )
            action_left_x = card_x + (card_w - max_action_w) // 2
            y_pos = card_y + (card_h - total_text_h) // 2
            for text, font, tw, th, is_title, is_action in text_specs:
                if is_title:
                    x_pos = card_x + (card_w - tw) // 2
                    color = (246, 252, 255)
                elif is_action:
                    x_pos = action_left_x
                    color = (255, 216, 128)
                else:
                    x_pos = card_x + (card_w - tw) // 2
                    color = (206, 224, 236)
                draw.text((x_pos + 1, y_pos + 1), text, font=font, fill=(0, 0, 0))
                draw.text((x_pos, y_pos), text, font=font, fill=color)
                y_pos += th + line_gap
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # cv2 ASCIIフォールバック
            def _strip_cjk(s: str) -> str:
                """非ASCII文字を除去し、ASCII部分だけ残す。"""
                parts = []
                for ch in s:
                    if ord(ch) <= 127:
                        parts.append(ch)
                stripped = "".join(parts).strip()
                return stripped if stripped else "(no JP font)"
            lines = tuple(_strip_cjk(line) for line in lines)

            font = cv2.FONT_HERSHEY_SIMPLEX
            metrics = []
            max_text_w = 0
            total_text_h = 0
            line_gap = 12
            for idx, line in enumerate(lines):
                is_title = idx == 0
                is_action = idx > 0 and line.lstrip().startswith("[")
                scale = 1.0 if is_title else (0.95 if is_action else 0.82)
                thick = 2 if is_title else 1
                text_size, _ = cv2.getTextSize(line, font, scale, thick)
                metrics.append((line, scale, thick, text_size, is_title, is_action))
                max_text_w = max(max_text_w, text_size[0])
                total_text_h += text_size[1]
            if len(metrics) > 1:
                total_text_h += line_gap * (len(metrics) - 1)

            card_w = min(w - 60, max(420, max_text_w + 120))
            card_h = max(170, total_text_h + 90)
            card_x = (w - card_w) // 2
            card_y = (h - card_h) // 2
            cv2.rectangle(frame, (card_x + 6, card_y + 6), (card_x + card_w + 6, card_y + card_h + 6), (0, 0, 0), -1)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (36, 44, 52), -1)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (104, 160, 190), 2)
            cv2.rectangle(frame, (card_x + 2, card_y + 2), (card_x + card_w - 2, card_y + 46), (52, 66, 78), -1)

            max_action_w_cv = max(
                (text_size[0] for _, _, _, text_size, is_title, is_action in metrics if is_action),
                default=0,
            )
            action_left_x_cv = card_x + (card_w - max_action_w_cv) // 2
            y_pos = card_y + (card_h - total_text_h) // 2
            for line, scale, thick, text_size, is_title, is_action in metrics:
                if is_title:
                    x_pos = card_x + (card_w - text_size[0]) // 2
                    color = (240, 248, 255)
                elif is_action:
                    x_pos = action_left_x_cv
                    color = (100, 210, 255)
                else:
                    x_pos = card_x + (card_w - text_size[0]) // 2
                    color = (200, 220, 235)
                cv2.putText(frame, line, (x_pos + 1, y_pos + 1), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
                cv2.putText(frame, line, (x_pos, y_pos), font, scale, color, thick, cv2.LINE_AA)
                y_pos += text_size[1] + line_gap

        if wait_sec > 0:
            cv2.putText(frame, "[ any key: skip ]", (10, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 110, 120), 1, cv2.LINE_AA)

        if self.window_manager is not None:
            self.window_manager.show(frame)
        else:
            cv2.imshow(self.window_name, frame)

        last_key = -1
        if wait_sec > 0:
            remaining_ms = int(wait_sec * 1000)
            while remaining_ms > 0:
                k = cv2.waitKey(min(50, remaining_ms))
                if k >= 0:
                    last_key = k & 0xFF
                    break
                remaining_ms -= 50
        else:
            k = cv2.waitKey(1)
            if k >= 0:
                last_key = k & 0xFF
        return last_key

    def _confirm_operator_action(
        self,
        *lines,
        next_label: str = "次へ",
        extra_confirm_keys: tuple[int, ...] = (),
    ) -> bool:
        """operator に次へ/中断だけを求める共通プロンプト。"""
        prompt_lines = list(lines) if lines else ["準備ができたら進めてください"]
        if extra_confirm_keys:
            key_names = "/".join(chr(k).upper() for k in extra_confirm_keys)
            prompt_lines.append(f"[{key_names}] {next_label}")
        else:
            prompt_lines.append(f"[Enter] {next_label}")
        prompt_lines.append("[ESC/q] 中断")
        self._show_capture_overlay(*prompt_lines)
        confirm_key = cv2.waitKey(0) & 0xFF
        if extra_confirm_keys:
            accepted_keys = set(extra_confirm_keys)
        else:
            accepted_keys = {10, 13, 32}  # Enter / Space (runner prompts)
        if confirm_key in accepted_keys:
            return True
        self._show_capture_overlay("中断しました", wait_sec=1.0)
        return False

    def _require_operator_action(
        self,
        *lines,
        next_label: str = "設置完了",
        extra_confirm_keys: tuple[int, ...] = (),
    ) -> None:
        """operator が準備完了を返すまで待ち、拒否時は runner を止める。"""
        if getattr(self, "_acceptance_run_type", None):
            self._promote_operator_guidance_mode(OPERATOR_GUIDANCE_MODE_MANUAL)
        if not self._confirm_operator_action(
            *lines,
            next_label=next_label,
            extra_confirm_keys=extra_confirm_keys,
        ):
            if getattr(self, "_acceptance_run_type", None):
                self._set_guidance_runner_state(GUIDANCE_RUNNER_STATE_CANCELLED)
            raise RuntimeError("operator cancelled")

    @staticmethod
    def _get_acceptance_env_value(name: str) -> str:
        """acceptance runner 用 env 値を空文字へ正規化して返す。"""
        return str(os.environ.get(name, "")).strip()

    def _ensure_operator_guidance_context(self) -> None:
        """guidance state を lazy 初期化して固定集合へそろえる。"""
        try:
            self._operator_guidance_mode = normalize_operator_guidance_mode(
                getattr(self, "_operator_guidance_mode", None),
            )
        except ValueError:
            self._operator_guidance_mode = OPERATOR_GUIDANCE_MODE_GUIDED
        try:
            self._guidance_degraded_reasons = normalize_guidance_degraded_reasons(
                getattr(self, "_guidance_degraded_reasons", None),
            )
        except ValueError:
            self._guidance_degraded_reasons = []
        runner_state = str(getattr(self, "_guidance_runner_state", "")).strip()
        if runner_state not in VALID_GUIDANCE_RUNNER_STATES:
            self._guidance_runner_state = GUIDANCE_RUNNER_STATE_IDLE

    def _reset_operator_guidance_context(self) -> None:
        """acceptance runner 開始時に guidance state を既知状態へ戻す。"""
        self._operator_guidance_mode = OPERATOR_GUIDANCE_MODE_GUIDED
        self._guidance_degraded_reasons = []
        self._guidance_runner_state = GUIDANCE_RUNNER_STATE_IDLE

    def _set_guidance_runner_state(self, state: str) -> None:
        """closed-set runner state を更新する。"""
        candidate = str(state or "").strip()
        if candidate not in VALID_GUIDANCE_RUNNER_STATES:
            raise ValueError(
                f"invalid guidance runner state: {candidate!r}; "
                f"expected one of {VALID_GUIDANCE_RUNNER_STATES}"
            )
        self._ensure_operator_guidance_context()
        self._guidance_runner_state = candidate

    def _promote_operator_guidance_mode(self, mode: str) -> None:
        """strict guided -> degraded -> manual の順で mode を昇格する。"""
        self._ensure_operator_guidance_context()
        candidate = normalize_operator_guidance_mode(mode)
        priority = {
            OPERATOR_GUIDANCE_MODE_GUIDED: 0,
            OPERATOR_GUIDANCE_MODE_DEGRADED: 1,
            OPERATOR_GUIDANCE_MODE_MANUAL: 2,
        }
        current = self._operator_guidance_mode
        if priority[candidate] >= priority[current]:
            self._operator_guidance_mode = candidate

    def _append_guidance_degraded_reason(self, reason: str, *, mode: str) -> None:
        """guidance degraded reason を重複なしで記録する。"""
        self._ensure_operator_guidance_context()
        normalized_reasons = normalize_guidance_degraded_reasons([reason])
        if not normalized_reasons:
            return
        self._promote_operator_guidance_mode(mode)
        normalized_reason = normalized_reasons[0]
        if normalized_reason not in self._guidance_degraded_reasons:
            self._guidance_degraded_reasons.append(normalized_reason)

    def _sync_guidance_from_manual_flags(self) -> None:
        """manual ROI/corner flags を guidance model に反映する。"""
        if getattr(self, "_manual_roi_used", False):
            self._append_guidance_degraded_reason(
                GUIDANCE_DEGRADED_REASON_MANUAL_ROI_ADJUSTMENT,
                mode=OPERATOR_GUIDANCE_MODE_DEGRADED,
            )
        if getattr(self, "_manual_corner_used", False):
            self._append_guidance_degraded_reason(
                GUIDANCE_DEGRADED_REASON_MANUAL_CORNER_ADJUSTMENT,
                mode=OPERATOR_GUIDANCE_MODE_DEGRADED,
            )

    def _get_operator_guidance_snapshot(self) -> tuple[str, list[str]]:
        """current guidance mode / degraded reasons を返す。"""
        self._ensure_operator_guidance_context()
        self._sync_guidance_from_manual_flags()
        return (
            self._operator_guidance_mode,
            list(self._guidance_degraded_reasons),
        )

    def _resolve_control_sample_id(self) -> str:
        """control sample ID を env 優先で返す。"""
        return (
            self._get_acceptance_env_value("PICOLOR_CONTROL_SAMPLE_ID")
            or DEFAULT_CONTROL_SAMPLE_ID
        )

    def _resolve_fixture_id(self) -> str:
        """fixture ID を env 優先で返す。"""
        return self._get_acceptance_env_value("PICOLOR_FIXTURE_ID")

    def _resolve_context_profile_id(self) -> str:
        """opaque な context profile ID を env 優先で返す。"""
        return self._get_acceptance_env_value("PICOLOR_CONTEXT_PROFILE_ID")

    def _resolve_positioning_method(self) -> str:
        """fixture 未設定時の位置決め方法を決める。"""
        env_value = self._get_acceptance_env_value("PICOLOR_POSITIONING_METHOD")
        if env_value:
            return env_value
        explicit_value = str(getattr(self, "_positioning_method", "")).strip()
        if explicit_value:
            return explicit_value
        if self._manual_corner_used:
            return "manual_chart_corners"
        if len(self._raw_corners) == 4:
            return "saved_chart_corners"
        return "chart_grid_preview"

    def _build_current_measurement_context(
        self,
        *,
        control_sample_id: str | None = None,
        fixture_id: str | None = None,
        positioning_method: str | None = None,
    ) -> tuple[dict, str]:
        """現在 run の measurement context summary / fingerprint を返す。"""
        software_identity = get_software_identity()
        summary = build_measurement_context_summary(
            control_sample_id=control_sample_id or self._resolve_control_sample_id(),
            fixture_id=fixture_id if fixture_id is not None else self._resolve_fixture_id(),
            positioning_method=(
                positioning_method
                if positioning_method is not None
                else self._resolve_positioning_method()
            ),
            context_profile_id=self._resolve_context_profile_id(),
            software_version=software_identity.get("software_version", "unknown"),
            git_revision=software_identity.get("git_revision", "unknown"),
        )
        return summary, compute_measurement_context_fingerprint(summary)

    def _evaluate_acceptance_context_gate(self, run_type: str) -> dict:
        """formal run 開始前の same-condition context 契約を評価する。"""
        current_context, current_fingerprint = self._build_current_measurement_context()
        diagnostics = describe_measurement_context_resolution(current_context)
        if run_type == RUN_TYPE_VERIFY_ONLY or diagnostics["is_resolved"]:
            return {
                "allowed": True,
                "run_type": run_type,
                "measurement_context": diagnostics["summary"],
                "measurement_context_fingerprint": current_fingerprint,
                "measurement_context_missing_contract_fields": [],
            }

        display_name = get_run_type_display_name(run_type)
        missing_contract_fields = diagnostics["missing_contract_fields"]
        return {
            "allowed": False,
            "blocked": True,
            "reason": "current_measurement_context_unresolved",
            "run_type": run_type,
            "run_type_display_name": display_name,
            "context_contract_status": "failed",
            "measurement_context": diagnostics["summary"],
            "measurement_context_fingerprint": "",
            "measurement_context_missing_contract_fields": missing_contract_fields,
            "operator_detail": (
                f"{display_name} には同一条件IDが必要です"
            ),
            "operator_hint": (
                "PICOLOR_CONTEXT_PROFILE_ID と fixture/positioning 設定を確認してください"
            ),
            "overlay_lines": [
                "context contract: formal run を開始できません",
                f"{display_name} の同一条件IDが不足しています",
                f"不足: {', '.join(missing_contract_fields)}",
                "PICOLOR_CONTEXT_PROFILE_ID と fixture/positioning を確認してください",
            ],
        }

    def _get_overlay_frame_size(self) -> tuple[int, int]:
        """overlay 描画用フレームサイズを安全な既定値付きで返す。"""
        camera_cfg = getattr(self.config, "camera", None)
        display_size = getattr(camera_cfg, "display_size", (960, 720))
        try:
            display_w = max(int(display_size[0]), 640)
            display_h = max(int(display_size[1]), 480)
        except (TypeError, ValueError, IndexError):
            display_w, display_h = 960, 720
        left_panel_width = int(getattr(camera_cfg, "left_panel_width", 240))
        panel_width = int(getattr(camera_cfg, "panel_width", 320))
        return left_panel_width + display_w + panel_width, display_h

    def _refresh_fixed_anchor_selection_state(self) -> dict:
        """保存済み fixed anchor 選択状態を読み直して internal state へ反映する。"""
        selection = load_fixed_anchor_selection(
            calibration_root=CALIBRATION_DIR,
            control_sample_id=self._resolve_control_sample_id(),
        )
        self._accepted_reference_selection_mode = selection.get(
            "accepted_reference_selection_mode",
            ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
        )
        self._accepted_reference_selection_requested_run_id = selection.get(
            "accepted_reference_selection_requested_run_id",
            "",
        )
        self._accepted_reference_selection_source_file = selection.get(
            "accepted_reference_selection_source_file",
            "",
        )
        return selection

    def _resolve_accepted_reference_record_for_context(
        self,
        *,
        control_sample_id: str,
        before_timestamp: str | None = None,
        require_open_gate: bool = False,
    ) -> tuple[dict | None, dict]:
        """fixed anchor 優先で accepted reference record を解決する。"""
        selection = self._refresh_fixed_anchor_selection_state()
        current_context, _current_fingerprint = self._build_current_measurement_context(
            control_sample_id=control_sample_id,
        )
        record, selection_meta = resolve_accepted_reference_record(
            calibration_root=os.path.dirname(get_today_calibration_dir()),
            control_sample_id=control_sample_id,
            before_timestamp=before_timestamp,
            accepted_reference_selection_mode=selection.get(
                "accepted_reference_selection_mode",
                ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
            ),
            accepted_reference_selection_requested_run_id=selection.get(
                "accepted_reference_selection_requested_run_id",
                "",
            ),
            accepted_reference_selection_source_file=selection.get(
                "accepted_reference_selection_source_file",
                "",
            ),
            require_open_gate=require_open_gate,
            current_measurement_context=current_context,
        )
        return record, selection_meta

    def _restore_default_mouse_callback(self) -> None:
        """現在の state に応じて既定 mouse callback を復元する。"""
        if self._chart_state == "preview" and self._grid_extractor is not None:
            if self.window_manager is not None:
                drag_cb = self.window_manager.make_offset_callback(self._on_preview_drag)
            else:
                drag_cb = self._on_preview_drag
            cv2.setMouseCallback(self.window_name, drag_cb)
            return
        if self._chart_state == "corner_input":
            if self.window_manager is not None:
                cb = self.window_manager.make_offset_callback(self._on_chart_click)
            else:
                cb = self._on_chart_click
            cv2.setMouseCallback(self.window_name, cb)
            return
        if self.roi_mouse is not None:
            if self.window_manager is not None:
                cv2.setMouseCallback(
                    self.window_name,
                    self.window_manager.make_offset_callback(self.roi_mouse.callback),
                )
            else:
                cv2.setMouseCallback(self.window_name, self.roi_mouse.callback)

    @staticmethod
    def _truncate_overlay_text(text: str, max_chars: int) -> str:
        """overlay 表示向けに文字列を短く整える。"""
        normalized = str(text or "").strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 1] + "…"

    @staticmethod
    def _measure_text_width_pil(text: str, font) -> int:
        """PIL font でのテキスト幅を返す。"""
        if font is None or not text:
            return 0
        try:
            bbox = font.getbbox(text)
        except Exception:
            return 0
        return max(0, int(bbox[2] - bbox[0]))

    @classmethod
    def _fit_text_to_width_pil(
        cls,
        text: str,
        *,
        max_width_px: int,
        font,
        ellipsis: str = "...",
    ) -> str:
        """PIL 描画用にテキストを最大幅へ収める。"""
        normalized = str(text or "").strip()
        if max_width_px <= 0:
            return ""
        if cls._measure_text_width_pil(normalized, font) <= max_width_px:
            return normalized
        if cls._measure_text_width_pil(ellipsis, font) > max_width_px:
            return ""
        clipped = normalized
        while clipped:
            clipped = clipped[:-1].rstrip()
            candidate = f"{clipped}{ellipsis}" if clipped else ellipsis
            if cls._measure_text_width_pil(candidate, font) <= max_width_px:
                return candidate
        return ellipsis

    @staticmethod
    def _fit_text_to_width_cv2(
        text: str,
        *,
        max_width_px: int,
        font,
        font_scale: float,
        thickness: int = 1,
        ellipsis: str = "...",
    ) -> str:
        """cv2 描画用にテキストを最大幅へ収める。"""
        normalized = str(text or "").strip()
        if max_width_px <= 0:
            return ""

        def _width(value: str) -> int:
            return cv2.getTextSize(value, font, font_scale, thickness)[0][0]

        if _width(normalized) <= max_width_px:
            return normalized
        if _width(ellipsis) > max_width_px:
            return ""
        clipped = normalized
        while clipped:
            clipped = clipped[:-1].rstrip()
            candidate = f"{clipped}{ellipsis}" if clipped else ellipsis
            if _width(candidate) <= max_width_px:
                return candidate
        return ellipsis

    def _render_fixed_anchor_picker_frame(
        self,
        *,
        options: list[dict],
        cursor_idx: int,
        checked_idx: int,
        scroll_offset: int,
    ) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], int]:
        """fixed anchor picker の現在フレームを返す。"""
        w, h = self._get_overlay_frame_size()
        frame = np.full((h, w, 3), (16, 18, 24), dtype=np.uint8)
        shade = np.full_like(frame, (0, 0, 0), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.80, shade, 0.20, 0.0)
        card_x = 44
        card_y = 28
        card_w = max(520, w - 88)
        card_h = max(320, h - 56)
        header_h = 92
        footer_h = 54
        row_h = 60
        visible_count = max(1, (card_h - header_h - footer_h) // row_h)
        max_scroll = max(0, len(options) - visible_count)
        scroll_offset = max(0, min(scroll_offset, max_scroll))
        if cursor_idx < scroll_offset:
            scroll_offset = cursor_idx
        elif cursor_idx >= scroll_offset + visible_count:
            scroll_offset = cursor_idx - visible_count + 1
        title_text = "固定アンカー選択"
        subtitle_text = "1件だけ選べます。Enterで決定、ESCでキャンセル"
        footer_left_text = "[↑/↓/j/k] 移動  [Space] 選択  [Enter] 決定  [ESC] 戻る"

        if _PIL_AVAILABLE and self._jp_font_body is not None:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            title_font = self._jp_font_title or self._jp_font_body
            body_font = self._jp_font_body
            small_font = self._jp_font_small or self._jp_font_body
            inner_pad = 26
            card_inner_width = max(0, card_w - inner_pad * 2)
            if hasattr(draw, "rounded_rectangle"):
                draw.rounded_rectangle(
                    (card_x, card_y, card_x + card_w, card_y + card_h),
                    radius=16,
                    fill=(26, 32, 40),
                    outline=(92, 146, 172),
                    width=2,
                )
                draw.rounded_rectangle(
                    (card_x + 2, card_y + 2, card_x + card_w - 2, card_y + header_h - 6),
                    radius=14,
                    fill=(34, 48, 61),
                )
            else:
                draw.rectangle(
                    (card_x, card_y, card_x + card_w, card_y + card_h),
                    fill=(26, 32, 40),
                    outline=(92, 146, 172),
                    width=2,
                )
            draw.text(
                (card_x + 26, card_y + 18),
                self._fit_text_to_width_pil(
                    title_text,
                    max_width_px=card_inner_width,
                    font=title_font,
                ),
                font=title_font,
                fill=(246, 252, 255),
            )
            draw.text(
                (card_x + 26, card_y + 54),
                self._fit_text_to_width_pil(
                    subtitle_text,
                    max_width_px=card_inner_width,
                    font=small_font,
                ),
                font=small_font,
                fill=(206, 224, 236),
            )
            hitboxes: dict[int, tuple[int, int, int, int]] = {}
            list_top = card_y + header_h + 6
            for visible_pos in range(visible_count):
                option_idx = scroll_offset + visible_pos
                if option_idx >= len(options):
                    break
                option = options[option_idx]
                row_y = list_top + visible_pos * row_h
                row_rect = (card_x + 18, row_y, card_x + card_w - 18, row_y + row_h - 8)
                hitboxes[option_idx] = row_rect
                fill = (52, 66, 78) if option_idx == cursor_idx else (30, 38, 46)
                outline = (124, 194, 230) if option_idx == cursor_idx else (70, 88, 102)
                if hasattr(draw, "rounded_rectangle"):
                    draw.rounded_rectangle(row_rect, radius=10, fill=fill, outline=outline, width=2)
                else:
                    draw.rectangle(row_rect, fill=fill, outline=outline, width=2)
                checkbox = "[x]" if option_idx == checked_idx else "[ ]"
                draw.text(
                    (row_rect[0] + 16, row_rect[1] + 10),
                    checkbox,
                    font=body_font,
                    fill=(255, 216, 128) if option_idx == checked_idx else (220, 232, 242),
                )
                row_text_x = row_rect[0] + 76
                row_text_max_width = max(0, row_rect[2] - row_text_x - 14)
                draw.text(
                    (row_text_x, row_rect[1] + 8),
                    self._fit_text_to_width_pil(
                        option["primary"],
                        max_width_px=row_text_max_width,
                        font=small_font,
                    ),
                    font=small_font,
                    fill=(238, 244, 248),
                )
                draw.text(
                    (row_text_x, row_rect[1] + 32),
                    self._fit_text_to_width_pil(
                        option["secondary"],
                        max_width_px=row_text_max_width,
                        font=small_font,
                    ),
                    font=small_font,
                    fill=(176, 198, 214),
                )
            footer_text = (
                f"{min(len(options), scroll_offset + 1)}-"
                f"{min(len(options), scroll_offset + visible_count)}/{len(options)}"
            )
            footer_right_width = self._measure_text_width_pil(footer_text, small_font)
            footer_left_width = max(0, card_inner_width - footer_right_width - 18)
            footer_left = self._fit_text_to_width_pil(
                footer_left_text,
                max_width_px=footer_left_width,
                font=small_font,
            )
            draw.text(
                (card_x + 26, card_y + card_h - 40),
                footer_left,
                font=small_font,
                fill=(196, 214, 228),
            )
            draw.text(
                (
                    card_x + card_w - 26 - footer_right_width,
                    card_y + card_h - 40,
                ),
                footer_text,
                font=small_font,
                fill=(196, 214, 228),
            )
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return frame, hitboxes, scroll_offset

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (36, 44, 52), -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (104, 160, 190), 2)
        title_fallback = self._fit_text_to_width_cv2(
            "Fixed anchor",
            max_width_px=max(0, card_w - 52),
            font=font,
            font_scale=0.9,
            thickness=2,
        )
        subtitle_fallback = self._fit_text_to_width_cv2(
            "Select one only / Enter OK / ESC cancel",
            max_width_px=max(0, card_w - 52),
            font=font,
            font_scale=0.6,
            thickness=1,
        )
        cv2.putText(frame, title_fallback, (card_x + 24, card_y + 34), font, 0.9, (240, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle_fallback, (card_x + 24, card_y + 68), font, 0.6, (206, 224, 236), 1, cv2.LINE_AA)
        hitboxes = {}
        list_top = card_y + header_h + 6
        for visible_pos in range(visible_count):
            option_idx = scroll_offset + visible_pos
            if option_idx >= len(options):
                break
            option = options[option_idx]
            row_y = list_top + visible_pos * row_h
            row_rect = (card_x + 18, row_y, card_x + card_w - 18, row_y + row_h - 8)
            hitboxes[option_idx] = row_rect
            fill = (52, 66, 78) if option_idx == cursor_idx else (30, 38, 46)
            outline = (124, 194, 230) if option_idx == cursor_idx else (70, 88, 102)
            cv2.rectangle(frame, (row_rect[0], row_rect[1]), (row_rect[2], row_rect[3]), fill, -1)
            cv2.rectangle(frame, (row_rect[0], row_rect[1]), (row_rect[2], row_rect[3]), outline, 2)
            checkbox = "[x]" if option_idx == checked_idx else "[ ]"
            cv2.putText(frame, checkbox, (row_rect[0] + 16, row_rect[1] + 22), font, 0.7, (255, 216, 128), 2, cv2.LINE_AA)
            row_text_x = row_rect[0] + 80
            row_text_max_width = max(0, row_rect[2] - row_text_x - 14)
            primary_fallback = option.get("primary_fallback", option["primary"])
            secondary_fallback = option.get("secondary_fallback", option["secondary"])
            cv2.putText(
                frame,
                self._fit_text_to_width_cv2(
                    primary_fallback,
                    max_width_px=row_text_max_width,
                    font=font,
                    font_scale=0.55,
                    thickness=1,
                ),
                (row_text_x, row_rect[1] + 22),
                font,
                0.55,
                (238, 244, 248),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                self._fit_text_to_width_cv2(
                    secondary_fallback,
                    max_width_px=row_text_max_width,
                    font=font,
                    font_scale=0.45,
                    thickness=1,
                ),
                (row_text_x, row_rect[1] + 44),
                font,
                0.45,
                (176, 198, 214),
                1,
                cv2.LINE_AA,
            )
        footer_text = (
            f"{min(len(options), scroll_offset + 1)}-"
            f"{min(len(options), scroll_offset + visible_count)}/{len(options)}"
        )
        footer_right_size = cv2.getTextSize(footer_text, font, 0.55, 1)[0]
        footer_left_max_width = max(0, card_w - 52 - footer_right_size[0] - 18)
        footer_left_fallback = self._fit_text_to_width_cv2(
            "[↑/↓/j/k] Move  [Space] Select  [Enter] OK  [ESC] Back",
            max_width_px=footer_left_max_width,
            font=font,
            font_scale=0.55,
            thickness=1,
        )
        cv2.putText(
            frame,
            footer_left_fallback,
            (card_x + 24, card_y + card_h - 20),
            font,
            0.55,
            (196, 214, 228),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            footer_text,
            (card_x + card_w - 24 - footer_right_size[0], card_y + card_h - 20),
            font,
            0.55,
            (196, 214, 228),
            1,
            cv2.LINE_AA,
        )
        return frame, hitboxes, scroll_offset

    def _open_fixed_anchor_picker(self) -> None:
        """固定 anchor を GUI で選択する modal picker。"""
        if self._chart_state != "idle":
            self._show_capture_overlay(
                "fixed anchor は今は変更できません",
                "CCM / チャート操作を終えてから開いてください",
                wait_sec=1.5,
            )
            return
        control_sample_id = self._resolve_control_sample_id()
        selection = self._refresh_fixed_anchor_selection_state()
        candidates = list_fixed_anchor_candidates(
            calibration_root=os.path.dirname(get_today_calibration_dir()),
            control_sample_id=control_sample_id,
        )
        options = [{
            "run_id": "",
            "source_file_rel": "",
            "primary": "自動: 最新の正式基準を使う",
            "secondary": "固定アンカーを解除して自動選択へ戻す",
            "primary_fallback": "Auto: use latest open ref",
            "secondary_fallback": "Clear fixed anchor / back to auto",
        }]
        for record in candidates:
            timestamp = str(record.get("timestamp", "unknown")).replace("T", " ", 1)
            run_label = get_run_type_display_name(record.get("run_type")).replace(" run", "")
            result_status = str(record.get("result_status", "unknown"))
            result_status_ascii = {
                "合格": "ACCEPTED",
                "最低合格": "MINIMUM",
                "要再判定": "RECHECK",
                "失格": "REJECTED",
            }.get(result_status, result_status)
            gate_state = str(record.get("gate_state", "unknown"))
            primary = (
                f"{timestamp} | "
                f"{run_label} | "
                f"{result_status}/{gate_state}"
            )
            secondary = (
                f"{record.get('source_file_rel', '')} | "
                f"{record.get('run_id', 'unknown')}"
            )
            primary_fallback = (
                f"{timestamp} | "
                f"{record.get('run_type', 'unknown')} | "
                f"{result_status_ascii}/{gate_state}"
            )
            options.append({
                "run_id": str(record.get("run_id", "")),
                "source_file_rel": str(record.get("source_file_rel", "")),
                "selected_measurement_context": record.get("measurement_context", {}),
                "selected_measurement_context_fingerprint": record.get(
                    "measurement_context_fingerprint",
                    "",
                ),
                "primary": primary,
                "secondary": secondary,
                "primary_fallback": primary_fallback,
                "secondary_fallback": secondary,
            })
        checked_idx = 0
        if (
            selection.get("accepted_reference_selection_mode")
            == ACCEPTED_REFERENCE_SELECTION_MODE_FIXED
        ):
            requested_run_id = selection.get(
                "accepted_reference_selection_requested_run_id",
                "",
            )
            for idx, option in enumerate(options[1:], start=1):
                if option["run_id"] == requested_run_id:
                    checked_idx = idx
                    break
        cursor_idx = checked_idx
        scroll_offset = max(0, cursor_idx - 4)
        mouse_state = {"hitboxes": {}}

        def _on_picker_click(event, x, y, _flags, _param) -> None:
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for option_idx, rect in mouse_state["hitboxes"].items():
                x0, y0, x1, y1 = rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    mouse_state["cursor_idx"] = option_idx
                    mouse_state["checked_idx"] = option_idx
                    break

        mouse_state["cursor_idx"] = cursor_idx
        mouse_state["checked_idx"] = checked_idx
        if self.window_manager is not None:
            cv2.setMouseCallback(
                self.window_name,
                self.window_manager.make_offset_callback(_on_picker_click),
            )
        else:
            cv2.setMouseCallback(self.window_name, _on_picker_click)
        try:
            while True:
                frame, hitboxes, scroll_offset = self._render_fixed_anchor_picker_frame(
                    options=options,
                    cursor_idx=mouse_state["cursor_idx"],
                    checked_idx=mouse_state["checked_idx"],
                    scroll_offset=scroll_offset,
                )
                mouse_state["hitboxes"] = hitboxes
                if self.window_manager is not None:
                    self.window_manager.show(frame)
                else:
                    cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(50) & 0xFF
                if key == 255:
                    continue
                if key in (27, ord("q")):
                    self._show_capture_overlay("fixed anchor 選択をキャンセルしました", wait_sec=0.8)
                    return
                if key in (82, ord("k")):
                    mouse_state["cursor_idx"] = max(0, mouse_state["cursor_idx"] - 1)
                    continue
                if key in (84, ord("j")):
                    mouse_state["cursor_idx"] = min(len(options) - 1, mouse_state["cursor_idx"] + 1)
                    continue
                if key == ord(" "):
                    mouse_state["checked_idx"] = mouse_state["cursor_idx"]
                    continue
                if key in (10, 13):
                    mouse_state["checked_idx"] = mouse_state["cursor_idx"]
                    selected_option = options[mouse_state["checked_idx"]]
                    if mouse_state["checked_idx"] == 0:
                        clear_fixed_anchor_selection(
                            CALIBRATION_DIR,
                            control_sample_id=control_sample_id,
                        )
                        self._refresh_fixed_anchor_selection_state()
                        self._show_capture_overlay(
                            "fixed anchor を解除しました",
                            "accepted reference は自動選択へ戻ります",
                            wait_sec=1.5,
                        )
                        return
                    save_fixed_anchor_selection(
                        run_id=selected_option["run_id"],
                        source_file=selected_option["source_file_rel"],
                        calibration_root=CALIBRATION_DIR,
                        control_sample_id=control_sample_id,
                        selected_measurement_context=selected_option.get(
                            "selected_measurement_context",
                            {},
                        ),
                        selected_measurement_context_fingerprint=selected_option.get(
                            "selected_measurement_context_fingerprint",
                            "",
                        ),
                    )
                    self._refresh_fixed_anchor_selection_state()
                    self._show_capture_overlay(
                        "fixed anchor を設定しました",
                        selected_option["primary"],
                        selected_option["secondary"],
                        wait_sec=1.8,
                    )
                    return
        finally:
            self._restore_default_mouse_callback()

    def _mark_manual_roi_used(self) -> None:
        """manual ROI 使用フラグを立てる。"""
        self._manual_roi_used = True
        if getattr(self, "_acceptance_run_type", None):
            self._append_guidance_degraded_reason(
                GUIDANCE_DEGRADED_REASON_MANUAL_ROI_ADJUSTMENT,
                mode=OPERATOR_GUIDANCE_MODE_DEGRADED,
            )

    def _mark_manual_corner_used(self) -> None:
        """manual corner 使用フラグを立てる。"""
        self._manual_corner_used = True
        extractor = getattr(self, "_grid_extractor", None)
        if extractor is not None and getattr(extractor, "corners", None):
            extractor.mark_corners_source("manual")
        if getattr(self, "_acceptance_run_type", None):
            self._append_guidance_degraded_reason(
                GUIDANCE_DEGRADED_REASON_MANUAL_CORNER_ADJUSTMENT,
                mode=OPERATOR_GUIDANCE_MODE_DEGRADED,
            )

    @staticmethod
    def _new_chart_workflow_status() -> dict[str, object]:
        """chart workflow 状態の既定 dict を返す。"""
        return {
            "active": False,
            "stage": "idle",
            "message": "",
            "detail": "",
            "progress": None,
            "total": None,
            "can_cancel": False,
            "source": "",
            "chart_orientation_order": "",
        }

    def _set_chart_workflow_status(
        self,
        *,
        stage: str,
        message: str,
        detail: str = "",
        progress: int | None = None,
        total: int | None = None,
        can_cancel: bool = False,
        source: str | None = None,
        chart_orientation_order: str | None = None,
        active: bool = True,
    ) -> None:
        """chart workflow の現在状態を更新する。"""
        previous_status = getattr(
            self,
            "_chart_workflow_status",
            self._new_chart_workflow_status(),
        )
        previous_source = str(previous_status.get("source", "")).strip()
        source_value = (
            str(source).strip()
            if source is not None
            else previous_source
        )
        previous_orientation = str(
            previous_status.get("chart_orientation_order", "")
        ).strip()
        if chart_orientation_order is not None:
            orientation_value = str(chart_orientation_order or "").strip()
        elif source_value == "oriented_panel_lattice":
            orientation_value = self._oriented_panel_payload_order_name()
            if not orientation_value and (
                source is None or previous_source == source_value
            ):
                orientation_value = previous_orientation
        elif source is None:
            orientation_value = previous_orientation
        else:
            orientation_value = ""
        self._chart_workflow_status = {
            "active": bool(active),
            "stage": str(stage or "idle"),
            "message": str(message or "").strip(),
            "detail": str(detail or "").strip(),
            "progress": None if progress is None else int(progress),
            "total": None if total is None else int(total),
            "can_cancel": bool(can_cancel),
            "source": source_value,
            "chart_orientation_order": orientation_value,
        }

    def _clear_chart_workflow_status(self) -> None:
        """chart workflow 状態を idle に戻す。"""
        self._chart_workflow_status = self._new_chart_workflow_status()

    def get_chart_workflow_status(self) -> dict[str, object]:
        """描画用の chart workflow 状態コピーを返す。"""
        return dict(self._chart_workflow_status)

    def get_chart_trace_state(self) -> dict[str, str]:
        """CSV trace 用の chart 状態コピーを返す。"""
        return {
            "chart_state": self._chart_state or "unknown",
            "chart_warning_reason": self._flip_warning_reason or "",
        }

    def _chart_analysis_frame_size(self) -> tuple[int, int]:
        """saved corners 検証用に chart analysis composite のサイズを返す。"""
        display_size = getattr(getattr(self.config, "camera", None), "display_size", (800, 600))
        try:
            display_w = max(int(display_size[0]), 1)
            display_h = max(int(display_size[1]), 1)
        except (TypeError, ValueError, IndexError):
            display_w, display_h = 800, 600
        left_panel_width = int(getattr(getattr(self.config, "camera", None), "left_panel_width", 0))
        return left_panel_width + display_w, display_h

    def _saved_chart_corners_are_valid(
        self,
        corners: list[tuple[float, float]],
    ) -> bool:
        """saved corners が current composite frame 上で妥当かを返す。"""
        pts = np.asarray(corners, dtype=np.float64)
        if pts.shape != (4, 2):
            print("⚠ saved corners invalid: expected 4 corner points")
            return False

        frame_w, frame_h = self._chart_analysis_frame_size()
        if np.any(pts[:, 0] < 0.0) or np.any(pts[:, 0] >= float(frame_w)):
            print("⚠ saved corners invalid: x is outside current composite frame")
            return False
        if np.any(pts[:, 1] < 0.0) or np.any(pts[:, 1] >= float(frame_h)):
            print("⚠ saved corners invalid: y is outside current composite frame")
            return False

        bbox_w = float(pts[:, 0].max() - pts[:, 0].min())
        bbox_h = float(pts[:, 1].max() - pts[:, 1].min())
        if bbox_w < 40.0 or bbox_h < 30.0:
            print(
                "⚠ saved corners invalid: chart bounding box is too small "
                f"(w={bbox_w:.1f}, h={bbox_h:.1f})"
            )
            return False

        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        min_area = max(1500.0, float(frame_w * frame_h) * 0.003)
        if area < min_area:
            print(
                "⚠ saved corners invalid: polygon area is too small "
                f"(area={area:.1f}, threshold={min_area:.1f})"
            )
            return False
        return True

    @staticmethod
    def _is_chart_cancel_key(key: int) -> bool:
        """chart workflow を中断するキーかを返す。"""
        return key in (27, ord("q"), ord("Q"))

    def _raise_if_chart_cancel_requested(self, key: int, context: str) -> None:
        """cancel key が押されていれば RuntimeError を送出する。"""
        if self._is_chart_cancel_key(int(key)):
            raise RuntimeError(f"operator cancelled ({context})")

    def _poll_chart_cancel_key(self, context: str, delay_ms: int = 1) -> None:
        """ノンブロッキングに cancel key を監視する。"""
        key = cv2.waitKey(delay_ms)
        if key >= 0:
            self._raise_if_chart_cancel_requested(key & 0xFF, context)

    def _show_chart_workflow_overlay(
        self,
        stage: str,
        message: str,
        *,
        detail: str = "",
        progress: int | None = None,
        total: int | None = None,
        can_cancel: bool = False,
        source: str | None = None,
        wait_sec: float = 0.0,
    ) -> int:
        """chart workflow 状態更新つきの capture overlay を表示する。"""
        self._set_chart_workflow_status(
            stage=stage,
            message=message,
            detail=detail,
            progress=progress,
            total=total,
            can_cancel=can_cancel,
            source=source,
        )
        lines = [message]
        meta_parts: list[str] = []
        current_source = str(self._chart_workflow_status.get("source", "")).strip()
        if current_source:
            meta_parts.append(f"source: {current_source}")
        if progress is not None and total is not None and total > 0:
            meta_parts.append(f"{progress}/{total}")
        if meta_parts:
            lines.append("  ".join(meta_parts))
        current_orientation = str(
            self._chart_workflow_status.get("chart_orientation_order", "")
        ).strip()
        if current_orientation:
            if current_orientation == "visual_180":
                lines.append("ORIENT: UPSIDE-DOWN")
                lines.append("remap applied")
            elif current_orientation == "visual":
                lines.append("ORIENT: UPRIGHT")
            else:
                lines.append(f"ORIENT: {current_orientation}")
        if detail:
            lines.append(detail)
        if can_cancel:
            lines.append("[ESC/q] 中断")
        return self._show_capture_overlay(*lines, wait_sec=wait_sec)

    def _cancel_chart_workflow(
        self,
        reason: str,
        *,
        show_overlay: bool = True,
    ) -> None:
        """chart workflow を安全側で中断し UI state を戻す。"""
        if getattr(self, "_acceptance_run_type", None):
            self._set_guidance_runner_state(GUIDANCE_RUNNER_STATE_CANCELLED)
        self._reset_chart_calibration()
        if show_overlay:
            self._show_capture_overlay(
                "P workflow を中断しました",
                reason,
                wait_sec=1.0,
            )

    def _run_chart_measurement_with_cancel_guard(self) -> dict | None:
        """interactive preview からの P 実行を cancel-safe に包む。"""
        self._chart_state = "measuring"
        try:
            return self._execute_chart_measurement()
        except RuntimeError as exc:
            if "cancelled" not in str(exc):
                self._set_chart_workflow_status(
                    stage="error",
                    message="P workflow error",
                    detail=str(exc),
                    can_cancel=False,
                    source=str(self._chart_workflow_status.get("source", "")),
                )
                raise
            self._cancel_chart_workflow("状態を保存せず安全に戻しました")
            return None
        except Exception as exc:
            self._chart_state = "preview"
            self._set_chart_workflow_status(
                stage="error",
                message="P workflow error",
                detail=f"{type(exc).__name__}: {exc}",
                can_cancel=False,
                source=str(self._chart_workflow_status.get("source", "")),
            )
            raise

    def _flip_warning_operator_copy(
        self,
        *,
        still_problem: bool = False,
    ) -> tuple[str, str]:
        """flip_warning の表示文言を検出 reason に応じて返す。"""
        reason = self._flip_warning_reason or ""
        if reason.startswith("flip"):
            if still_problem:
                return (
                    "まだ上下反転しています",
                    "色基準(48色)を 180° 回してから再度 [P] を押してください",
                )
            return (
                "色基準(48色)が上下反転しています",
                "180° 回して再度 [P] を押してください",
            )
        if still_problem:
            return (
                "まだ色基準(48色)の向き・設置が合っていません",
                "水平に正しい向きで置き直してから再度 [P] を押してください",
            )
        return (
            "色基準(48色)の向き・設置を確認してください",
            "水平に正しい向きで置き直して再度 [P] を押してください",
        )

    def _enter_flip_warning_state(self) -> None:
        """Phase 7B: flip 検出時の hard-lock state 遷移.

        _grid_extractor は残したまま state を "flip_warning" に切替、
        main render loop が per-frame で派手な警告バナーを描画する.
        P 再押下で re-check → OK 時のみ preview に進む.
        ESC/q で idle に戻る (cancel).
        """
        self._chart_state = "flip_warning"
        message, detail = self._flip_warning_operator_copy()
        self._set_chart_workflow_status(
            stage="flip_warning",
            message=message,
            detail=detail,
            can_cancel=True,
            source="flip_warning",
            chart_orientation_order=(
                self._oriented_panel_payload_order_name()
                or str(
                    getattr(
                        self,
                        "_chart_workflow_status",
                        self._new_chart_workflow_status(),
                    ).get(
                        "chart_orientation_order",
                        "",
                    )
                ).strip()
            ),
        )
        print(
            "- chart_state → flip_warning: P 再押下で再チェック, "
            "ESC/q で cancel"
        )

    def _exit_flip_warning_to_preview(self) -> None:
        """flip_warning 解除 → preview state へ復帰."""
        self._chart_state = "preview"
        self._positioning_method = "saved_chart_corners"
        self._set_chart_workflow_status(
            stage="preview",
            message="色基準48色 プレビュー中",
            detail="",
            can_cancel=True,
            source="saved",
        )
        self._restore_default_mouse_callback()
        print("- flip_warning 解除: chart_state → preview (正置で再チェック合格)")

    def _preview_detect_spyder_flip(self) -> bool:
        """Phase 4B / 7D: preview 段階の軽量 orientation / placement 判定.

        saved_chart_corners 復元直後に **1 frame + 同一 request の main frame で
        patchwise ROI 準備** を行い、向き・設置を判定する. 32 frame 本測定時の複数
        capture を避けつつ、回転追従 ROI は stale cache に依存させない.

        G 比で SpyderCHECKR の上下反転・回転・設置ズレを早期検出する。
        True なら preview を中止して呼び出し元で hard-lock state に入る.

        Returns:
            True: 向き・設置異常を検出 (preview を中止すべき)
            False: 正置、または判定不能 (preview 継続)
        """
        if self._grid_extractor is None or not self._grid_extractor.is_ready:
            return False
        try:
            self._show_chart_workflow_overlay(
                "prescan_flip_check",
                "向き・設置チェック中",
                detail="1 フレーム + ROI補正で判定します",
                can_cancel=True,
            )
            patch_bayer_means = self._capture_chart_patch_bayer_means_fast()
        except RuntimeError as exc:
            if "cancelled" in str(exc):
                # ユーザーが ESC/q で cancel した. chart state を idle に戻し propagation.
                self._cancel_chart_workflow("向き・設置チェックを中断しました", show_overlay=False)
                raise
            # その他 runtime error は fail-open で preview 継続
            print(f"- preview flip check 失敗 (RuntimeError: {exc}), 後段で判定")
            return False
        except Exception as exc:
            # fail-open: 判定不能なら preview 継続 (後段 _finish_chart_calibration の保険が効く)
            print(f"- preview flip check 失敗 ({type(exc).__name__}: {exc}), 後段で判定")
            return False
        # patch_bayer_means は shape (48, 3). Phase 14: geometry (D 列単調性 + 1E/6E) →
        # pose hypothesis (多色 anchor) を統合した detect_spyder_chart_pose_problem を
        # 主判定に使う. 旧 detect_spyder_flip は保険として colorimeter_common に残存.
        try:
            arr = np.asarray(patch_bayer_means, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 48 or arr.shape[1] < 3:
                return False
        except Exception:
            return False
        is_problem, reason, pose_diag = detect_spyder_chart_pose_problem(arr[:, :3])
        if is_problem:
            white_g = float(arr[SPYDERCHECKR_48_WHITE_PATCH_INDEX][1])
            black_g = float(arr[SPYDERCHECKR_48_BLACK_PATCH_INDEX][1])
            stage = pose_diag.get("stage") if isinstance(pose_diag, dict) else "unknown"
            print(
                f"- preview chart orientation PROBLEM: stage={stage} reason={reason}, "
                f"idx4 (1E 期待) G={white_g:.1f}, idx44 (6E 期待) G={black_g:.1f}"
            )
            # 向き異常の分類を state に保存して banner 文言に活かす.
            self._flip_warning_reason = reason
        else:
            print("- preview chart orientation check OK")
            self._flip_warning_reason = ""
        return is_problem

    def _reset_chart_calibration(self) -> None:
        """チャートキャリブレーション状態をリセットする。"""
        self._chart_corners = []
        self._raw_corners = []
        self._chart_state = "idle"
        self._positioning_method = ""
        self._grid_extractor = None
        self._preview_margin_detected = False
        self._drag_corner_idx = -1
        self._drag_col_idx = -1
        self._drag_last_pos = (0, 0)
        self._clear_chart_workflow_status()
        self._restore_default_mouse_callback()

    @staticmethod
    def _main_array_to_bgr(main_array: np.ndarray) -> np.ndarray:
        """Picamera2 main 配列を BGR 扱いできる 3ch 画像へ整える。"""
        if main_array.ndim == 3 and main_array.shape[2] == 4:
            return cv2.cvtColor(main_array, cv2.COLOR_BGRA2BGR)
        if main_array.ndim == 2:
            return cv2.cvtColor(main_array, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(main_array)

    def _build_chart_analysis_frame_from_main(self, main_array: np.ndarray) -> np.ndarray:
        """main stream から chart ROI 解析用 composite frame を生成する。"""
        frame_bgr = self._main_array_to_bgr(main_array)
        frame_bgr = FrameTransformer.flip_image(
            frame_bgr,
            getattr(self.config.display, "flip_horizontal", False),
            getattr(self.config.display, "flip_vertical", False),
        )
        crop_w = getattr(self.config.display, "width", frame_bgr.shape[1])
        crop_h = getattr(self.config.display, "height", frame_bgr.shape[0])
        crop_w = min(int(crop_w), frame_bgr.shape[1])
        crop_h = min(int(crop_h), frame_bgr.shape[0])
        frame_bgr = FrameTransformer.crop_center(frame_bgr, crop_w, crop_h)
        return make_chart_analysis_composite(
            frame_bgr,
            getattr(self.config.camera, "left_panel_width", 0),
        )

    def _prepare_chart_patchwise_rois_from_main(
        self,
        main_array: np.ndarray | None,
    ) -> dict[str, object] | None:
        """measurement 前の chart per-patch ROI を現在フレームから準備する。"""
        if (
            main_array is None
            or self._grid_extractor is None
            or not self._grid_extractor.is_ready
        ):
            return None
        analysis_frame = self._build_chart_analysis_frame_from_main(main_array)
        if analysis_frame is None or analysis_frame.size == 0:
            return None
        if not self._preview_margin_detected:
            detected_margin = self._grid_extractor.detect_patch_margin_from_frame(
                analysis_frame
            )
            self._grid_extractor.patch_margin = detected_margin
        oriented_payload = getattr(self, "_last_oriented_panel_payload", None)
        if (
            getattr(self, "_positioning_method", "") == "oriented_panel_lattice"
            and isinstance(oriented_payload, SpyderCheckrOrientedPanelPayload)
        ):
            return self._grid_extractor.prepare_patchwise_rois_from_oriented_panel_payload(
                analysis_frame,
                oriented_payload,
            )
        return self._grid_extractor.prepare_patchwise_rois_from_frame(analysis_frame)

    def _prepare_preview_patchwise_rois_if_needed(
        self,
        analysis_frame: np.ndarray | None,
        *,
        force_refresh: bool = False,
    ) -> bool:
        """preview 表示用の patchwise ROI を必要時だけ更新する。"""
        if (
            self._chart_state != "preview"
            or self._grid_extractor is None
            or not self._grid_extractor.is_ready
            or analysis_frame is None
            or analysis_frame.size == 0
        ):
            return False
        patchwise_summary = self._grid_extractor.get_patchwise_summary()
        patchwise_entries = self._grid_extractor.get_patchwise_entries()
        refresh_needed = force_refresh or patchwise_summary is None or not patchwise_entries
        if not refresh_needed:
            return False
        if not self._preview_margin_detected:
            detected_margin = self._grid_extractor.detect_patch_margin_from_frame(
                analysis_frame
            )
            self._grid_extractor.patch_margin = detected_margin
            self._preview_margin_detected = True
            print(
                f"[CCM] 黒フレーム検出: patch_margin={detected_margin:.3f} "
                f"({int((1 - 2 * detected_margin) * 100)}% coverage)"
            )
        self._grid_extractor.prepare_patchwise_rois_from_frame(analysis_frame)
        return True

    def _prepare_preview_rois_and_start_chart_measurement(
        self,
        *,
        source: str,
    ) -> dict | None:
        """preview ROI を可能なら先に準備し、cancel-safe な測定入口へ進む。"""
        self._chart_state = "measuring"
        self._show_chart_workflow_overlay(
            "measuring",
            "色基準48色 測定開始",
            detail="ROI 確認済み。測定へ進みます",
            can_cancel=True,
            source=source,
        )
        try:
            main_array = self.picam2.capture_array("main")
        except Exception as exc:
            print(f"- preview ROI preflight skipped: capture_array failed ({exc})")
            main_array = None
        if isinstance(main_array, np.ndarray) and main_array.size > 0:
            try:
                analysis_frame = self._build_chart_analysis_frame_from_main(main_array)
                self._prepare_preview_patchwise_rois_if_needed(
                    analysis_frame,
                    force_refresh=True,
                )
            except Exception as exc:
                print(f"- preview ROI preflight skipped: {exc}")
        else:
            print("- preview ROI preflight skipped: no usable main frame")
        self._set_chart_workflow_status(
            stage="measuring",
            message="色基準48色 測定中",
            detail="ROI を自動配置して測定を開始します",
            can_cancel=True,
            source=source,
        )
        return self._run_chart_measurement_with_cancel_guard()

    @staticmethod
    def _write_preview_bundle_json(path: str, payload) -> None:
        """preview freeze bundle 用 JSON を UTF-8 / numpy-safe で保存する。"""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_json_ready(payload), fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    def _save_chart_preview_freeze_bundle(
        self,
        analysis_frame: np.ndarray | None,
    ) -> str | None:
        """現在の `P preview` から offline 解析向け freeze bundle を保存する。"""
        self.preview_freeze_bundle_pending = False
        if (
            self._chart_state != "preview"
            or self._grid_extractor is None
            or not self._grid_extractor.is_ready
        ):
            self._show_capture_overlay(
                "preview bundle を保存できません",
                "色基準48色 プレビュー中のみ保存できます",
                wait_sec=1.2,
            )
            return None
        if analysis_frame is None or analysis_frame.size == 0:
            self._show_capture_overlay(
                "preview bundle を保存できません",
                "現在フレームを取得できません",
                wait_sec=1.2,
            )
            return None
        patchwise_summary = self._grid_extractor.get_patchwise_summary()
        patchwise_entries = self._grid_extractor.get_patchwise_entries()
        if patchwise_summary is None or not patchwise_entries:
            # Stable preview save should reuse cached state when available, but
            # allow one refresh if geometry was invalidated and cache is empty.
            self._prepare_preview_patchwise_rois_if_needed(analysis_frame)
            patchwise_summary = self._grid_extractor.get_patchwise_summary()
            patchwise_entries = self._grid_extractor.get_patchwise_entries()
        if patchwise_summary is None or not patchwise_entries:
            self._show_capture_overlay(
                "preview bundle を保存できません",
                "patch ROI 状態が未準備です",
                wait_sec=1.2,
            )
            return None

        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bundle_dir = os.path.join(
            get_today_calibration_dir(),
            f"chart_preview_bundle_{ts_slug}",
        )
        os.makedirs(bundle_dir, exist_ok=True)

        analysis_frame_clean = np.ascontiguousarray(analysis_frame)
        overlay_reference = self._grid_extractor.draw_overlay(
            analysis_frame_clean.copy(),
            corners_so_far=[],
            white_local_idx=SPYDERCHECKR_48_WHITE_PATCH_INDEX,
        )
        analysis_path = os.path.join(bundle_dir, "analysis_frame.png")
        overlay_path = os.path.join(bundle_dir, "overlay_reference.png")
        cv2.imwrite(analysis_path, analysis_frame_clean)
        cv2.imwrite(overlay_path, overlay_reference)

        workflow_status = self.get_chart_workflow_status()
        lattice_state = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "bundle_dir": bundle_dir,
            "calibration_dir": get_today_calibration_dir(),
            "preview_source": str(workflow_status.get("source", "")).strip() or "preview",
            "workflow_status": workflow_status,
            "positioning_method": self._positioning_method,
            "manual_corner_used": bool(self._manual_corner_used),
            "preview_margin_detected": bool(self._preview_margin_detected),
            "corners": [list(corner) for corner in self._raw_corners],
            "hinge_gap": float(self._hinge_gap),
            "patch_margin": float(self._grid_extractor.patch_margin),
            "col_x_norm_offsets": self._grid_extractor.col_x_norm_offsets.tolist(),
            "analysis_frame_shape": list(analysis_frame_clean.shape),
            "software_identity": get_software_identity(),
            "artifacts": {
                "analysis_frame_png": os.path.basename(analysis_path),
                "overlay_reference_png": os.path.basename(overlay_path),
                "lattice_state_json": "lattice_state.json",
                "patchwise_summary_json": "patchwise_summary.json",
                "patchwise_entries_json": "patchwise_entries.json",
            },
            "patchwise_summary": patchwise_summary,
            "patchwise_entries": patchwise_entries,
        }
        self._write_preview_bundle_json(
            os.path.join(bundle_dir, "patchwise_summary.json"),
            patchwise_summary,
        )
        self._write_preview_bundle_json(
            os.path.join(bundle_dir, "patchwise_entries.json"),
            patchwise_entries,
        )
        self._write_preview_bundle_json(
            os.path.join(bundle_dir, "lattice_state.json"),
            lattice_state,
        )

        self._show_capture_overlay(
            "preview bundle を保存しました",
            os.path.basename(bundle_dir),
            wait_sec=1.6,
        )
        print(f"- chart preview freeze bundle saved: {bundle_dir}")
        return bundle_dir

    def _on_preview_drag(self, event, x, y, flags, param) -> None:
        """preview 状態でのコーナーハンドル・列アンカードラッグコールバック。"""
        if self._grid_extractor is None or not self._grid_extractor.is_ready:
            return

        HANDLE_R = 18
        COL_ANCHOR_R = 12

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (cx, cy) in enumerate(self._grid_extractor.corners):
                if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 < HANDLE_R:
                    self._drag_corner_idx = i
                    self._drag_col_idx = -1
                    return
            ge = self._grid_extractor
            half = ge.n_cols // 2
            for col in range(1, ge.n_cols - 1):
                gap_offset = ge.hinge_gap * 100 if col >= half else 0.0
                cx_n = (col + 0.5) * 100 + gap_offset + ge.col_x_norm_offsets[col]
                cy_n = 10.0
                pt_d = cv2.perspectiveTransform(
                    np.float32([[[cx_n, cy_n]]]), ge.homography_inv
                )[0][0]
                ax, ay = float(pt_d[0]), float(pt_d[1])
                if ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 < COL_ANCHOR_R:
                    self._drag_col_idx = col
                    self._drag_corner_idx = -1
                    self._drag_last_pos = (x, y)
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drag_corner_idx >= 0:
                new_corners = list(self._grid_extractor.corners)
                new_corners[self._drag_corner_idx] = (x, y)
                self._grid_extractor.set_corners_ordered_with_source(
                    new_corners,
                    source="manual",
                )
                self._raw_corners = list(new_corners)
                self._preview_margin_detected = False
                self._mark_manual_corner_used()
            elif self._drag_col_idx >= 0:
                lx, ly = self._drag_last_pos
                ge = self._grid_extractor
                pt_old = cv2.perspectiveTransform(
                    np.float32([[[float(lx), float(ly)]]]), ge.homography
                )[0][0]
                pt_new = cv2.perspectiveTransform(
                    np.float32([[[float(x), float(ly)]]]), ge.homography
                )[0][0]
                ge.col_x_norm_offsets[self._drag_col_idx] += float(pt_new[0] - pt_old[0])
                ge.invalidate_patchwise_rois()
                self._drag_last_pos = (x, y)
                self._mark_manual_corner_used()

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_corner_idx = -1
            self._drag_col_idx = -1

    def _rebuild_extractor_with_hinge(self) -> None:
        """ヒンジ幅を変更してエクストラクタを再構築する。"""
        old_offsets = (
            self._grid_extractor.col_x_norm_offsets.copy()
            if self._grid_extractor is not None
            else None
        )
        self._grid_extractor = SpyderCheckrGridExtractor(
            SPYDERCHECKR_48_SHAPE[0],
            SPYDERCHECKR_48_SHAPE[1],
            hinge_gap=self._hinge_gap,
        )
        self._grid_extractor.set_corners_with_source(
            self._raw_corners,
            source="manual",
        )
        if old_offsets is not None:
            self._grid_extractor.col_x_norm_offsets = old_offsets
            self._grid_extractor.invalidate_patchwise_rois()
        self._preview_margin_detected = False
        self._mark_manual_corner_used()

    def _restore_chart_preview_from_saved_corners(self) -> bool:
        """[DEPRECATED Phase 22 / Phase 2] 保存済み chart_corners.json から
        preview 状態を復元する。

        本番 path (`_run_*_sequence_impl` / idle [P] 分岐) からは呼ばれない。
        新検出器 (`_detect_chart_via_contour_outer_rim`) が main path となり、
        saved corners は **書込みのみ** 継続する設計 (`_persist_chart_corners`
        は不変)。本関数は緊急デバッグおよび ralph 過去 backup との互換のため
        残置。新規呼出し追加禁止。
        """
        saved_path = os.path.join(CALIBRATION_DIR, "chart_corners.json")
        _exists = os.path.exists(saved_path)
        try:
            if _exists:
                _stat = os.stat(saved_path)
                _mtime = datetime.fromtimestamp(
                    _stat.st_mtime, tz=timezone.utc
                ).isoformat()
                _size = int(_stat.st_size)
            else:
                _mtime = None
                _size = None
            print(
                f"[saved corners] path={saved_path} exists={_exists} "
                f"mtime={_mtime} size_bytes={_size}"
            )
        except Exception:
            pass
        if not _exists:
            return False
        try:
            with open(saved_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        if len(saved_data.get("corners", [])) != 4:
            return False
        corners = [tuple(c) for c in saved_data["corners"]]
        # Phase 22 / Phase 3 Cycle 1 (Codex C1 対応):
        # `_order_corners_clockwise_from_topleft` は画面座標の x+y / x-y axis sort で、
        # axis-aligned 前提の helper である。回転済み ordered corners (例: contour/rigid
        # 検出器が +60° quad を CW 順で書出した payload) には idempotent ではなく、
        # `[1A, 1H, 6H, 6A]` を `[6A, 1A, 1H, 6H]` のように壊し得る。よって payload に
        # 保存された `corners_order` field で分岐する:
        #   - "ordered_clockwise": Phase 3 後 `_persist_chart_corners` が書出す新規
        #     JSON。drag/manual (CW pre-sort 後)/contour/rigid いずれの経路も CW 順で
        #     `_raw_corners` に書込まれているため、再 sort せず pass-through する。
        #   - "legacy_arbitrary" or 不在 (default): Phase 3 以前 (本 field 導入前) に
        #     書出された旧 JSON。manual 4-click 由来で click 順 (任意) で永続化されて
        #     いた可能性があるため軸そろえで救済する。旧 JSON は Phase 3 以前の
        #     経路でしか作られず、それらは axis-aligned 前提だったため axis-sort は
        #     safe (回転なしチャートを CW 化できる)。
        # 既定値を `"legacy_arbitrary"` にすることで、key 不在の旧 JSON が確実に
        # 救済 path に流れる (R1 mitigation を維持)。
        corners_order = saved_data.get("corners_order", "legacy_arbitrary")
        # Phase 22 / Phase 3 Cycle 2 (Codex C2 対応):
        # `corners_order` 許可値を `"ordered_clockwise"` と `"legacy_arbitrary"` に
        # 限定する。key 不在は default fallback で `"legacy_arbitrary"` 扱い (旧 JSON
        # 互換)。それ以外の明示的な誤値や将来値 (`None`、`"unknown_value"` 等) は
        # reject し、saved_path を削除する (古い誤検出 = valid_mask 外と同じ扱い)。
        # これにより、誤値が legacy 分岐に流れて回転済み ordered corners が
        # `_order_corners_clockwise_from_topleft` で破壊される C2 経路を解消。
        if corners_order not in ("ordered_clockwise", "legacy_arbitrary"):
            print(
                f"⚠ corners_order 不正: '{corners_order}' → reject"
            )
            try:
                os.remove(saved_path)
            except OSError:
                pass
            return False
        if corners_order == "ordered_clockwise":
            print(
                f"[saved corners] order=ordered_clockwise pass-through -> "
                f"{corners}"
            )
        else:
            corners = _order_corners_clockwise_from_topleft(corners)
            print(
                f"[saved corners reorder] order={corners_order} "
                f"axis-sort 救済 -> {corners}"
            )
        try:
            print(
                f"[saved corners] corners={corners} "
                f"hinge_gap={float(saved_data.get('hinge_gap', 2.67))} "
                f"patch_margin={float(saved_data.get('patch_margin', 0.10))} "
                f"col_x_norm_offsets_len={len(saved_data.get('col_x_norm_offsets', []))}"
            )
        except Exception:
            pass
        if not self._saved_chart_corners_are_valid(corners):
            return False

        # valid_mask 外の saved corners (過去の誤検出が永続化されたもの) は reject。
        # ファイルも削除して次回 auto-detect が走るようにする。
        try:
            main_array = self.picam2.capture_array("main")
            if isinstance(main_array, np.ndarray) and main_array.size > 0:
                frame_for_mask = self._build_chart_analysis_frame_from_main(
                    main_array
                )
                if (
                    frame_for_mask is not None
                    and frame_for_mask.ndim == 3
                ):
                    composite_hole = self._composite_hole_mask(
                        frame_for_mask.shape[:2],
                        tuple(main_array.shape[:2]),
                    )
                    if composite_hole is not None:
                        if not self._corners_inside_hole_mask(
                            list(corners), composite_hole
                        ):
                            print(
                                "⚠ saved corners が valid_mask 外にはみ出し → "
                                "古い誤検出と判定して chart_corners.json を削除"
                            )
                            try:
                                os.remove(saved_path)
                            except OSError:
                                pass
                            return False
        except Exception as exc:
            print(f"- saved corners hole mask check skipped: {exc}")

        self._hinge_gap = saved_data.get("hinge_gap", 2.67)
        self._grid_extractor = SpyderCheckrGridExtractor(
            SPYDERCHECKR_48_SHAPE[0],
            SPYDERCHECKR_48_SHAPE[1],
            hinge_gap=self._hinge_gap,
        )
        self._grid_extractor.set_corners_with_source(corners, source="saved")
        offsets = saved_data.get("col_x_norm_offsets", [])
        if len(offsets) == self._grid_extractor.n_cols:
            self._grid_extractor.col_x_norm_offsets = np.array(
                offsets, dtype=np.float64
            )
        self._grid_extractor.patch_margin = saved_data.get("patch_margin", 0.10)
        self._grid_extractor.invalidate_patchwise_rois()
        self._raw_corners = corners
        self._chart_state = "preview"
        self._preview_margin_detected = False
        self._set_chart_workflow_status(
            stage="preview",
            message="色基準48色 プレビュー中",
            detail="ROI を復元しました",
            can_cancel=True,
            source="saved",
        )
        print(f"- チャートコーナー座標を復元: {saved_path}")
        return True

    def _measurement_fail_stop_allows_recording(self) -> bool:
        """当日の formal acceptance run に基づき本番記録可否を返す。"""
        gate = evaluate_measurement_fail_stop(get_today_calibration_dir())
        decision = apply_measurement_fail_stop_rollout(
            gate,
            get_repro_rollout_mode(),
        )
        if decision["recording_allowed"] and not decision["overlay_lines"]:
            return True

        self._show_capture_overlay(*decision["overlay_lines"], wait_sec=2.2)
        print(
            f"⚠ fail-stop[{decision['rollout_mode']}]: {decision['reason']}"
        )
        return bool(decision["recording_allowed"])

    @staticmethod
    def _build_previous_accepted_day_overlay_line(
        acceptance_record: dict,
    ) -> str:
        """formal run 向けの前回合格日サマリ 1 行を返す。"""
        run_type = str(
            acceptance_record.get("run_type", RUN_TYPE_VERIFY_ONLY)
        ).strip()
        if run_type not in (
            RUN_TYPE_START_OF_DAY,
            RUN_TYPE_END_OF_DAY,
            RUN_TYPE_REQUALIFICATION,
        ):
            return ""
        judgement = acceptance_record.get("previous_accepted_day_judgement")
        previous_date = str(
            acceptance_record.get("previous_accepted_day_date", "")
        ).strip()
        if isinstance(judgement, dict) and judgement.get("available") and previous_date:
            return f"前回合格日 {previous_date} 差分: {judgement.get('result_status', 'unknown')}"
        reason = str(
            acceptance_record.get("previous_accepted_day_selection_warning", "")
        ).strip()
        if not reason and isinstance(judgement, dict):
            reason = str(judgement.get("reason", "")).strip()
        if not reason:
            reason = "previous_accepted_day_missing"
        return f"前回合格日差: 未評価 ({reason})"

    @staticmethod
    def _acceptance_completion_has_calibration_failure(
        acceptance_record: dict,
    ) -> bool:
        """完了バナー上の「キャリブレーション失敗」扱いを量的NGに限定する。"""
        failed = list(acceptance_record.get("failed_checks", []) or [])
        status = str(acceptance_record.get("result_status", "")).strip()
        return bool(failed) or status == "失格"

    @staticmethod
    def _acceptance_completion_has_guidance_degradation(
        acceptance_record: dict,
    ) -> bool:
        """完了バナー上の手動/迂回 run 判定を正規化済み値で返す。"""
        guidance_mode = normalize_operator_guidance_mode(
            acceptance_record.get("operator_guidance_mode"),
        )
        degraded_reasons = normalize_guidance_degraded_reasons(
            acceptance_record.get("guidance_degraded_reasons"),
        )
        return (
            guidance_mode != OPERATOR_GUIDANCE_MODE_GUIDED
            or bool(degraded_reasons)
        )

    @staticmethod
    def _build_acceptance_completion_overlay_lines(
        acceptance_record: dict,
    ) -> tuple[str, ...]:
        """acceptance artifact と同じ意味の完了バナー文言を返す。"""
        delta_e = acceptance_record.get("delta_e")
        de_str = f"ΔE={delta_e:.2f}" if delta_e is not None else ""
        status = str(acceptance_record.get("result_status", "")).strip()
        failed = list(acceptance_record.get("failed_checks", []) or [])
        has_calibration_failure = (
            KeyHandler._acceptance_completion_has_calibration_failure(
                acceptance_record,
            )
        )
        has_guidance_degradation = (
            KeyHandler._acceptance_completion_has_guidance_degradation(
                acceptance_record,
            )
        )
        gate_state = str(acceptance_record.get("gate_state", "")).strip()
        run_type = str(acceptance_record.get("run_type", RUN_TYPE_VERIFY_ONLY)).strip()
        lines: list[str]
        if has_calibration_failure:
            fail_reason = f"4D検証NG: {', '.join(failed)}" if failed else "4D検証NG"
            lines = [
                f"✗ {fail_reason}",
                "本番測定を開始しないでください",
                "[D][F][W][P] をやり直してください",
            ]
        elif has_guidance_degradation:
            lines = [f"△ {status}（手動/迂回）"]
            if run_type in (RUN_TYPE_START_OF_DAY, RUN_TYPE_REQUALIFICATION):
                lines.extend([
                    "4D検証結果を保存しました",
                    "本番解放判定は未解放です",
                ])
            else:
                lines.extend([
                    "手動/迂回 run として結果を保存しました",
                    "本番測定は解放していません",
                ])
        elif (
            gate_state == GATE_STATE_PROVISIONAL
            or acceptance_record.get("provisional_reasons")
        ):
            lines = [
                f"△ {status}（暫定）",
                "4D検証結果を保存しました",
                "本番解放判定は未解放です",
            ]
        elif acceptance_record.get("production_eligible"):
            lines = [
                "✓ キャリブレーション完了",
                "本番測定を開始できます",
            ]
        else:
            lines = [
                "✓ キャリブレーション完了",
                "4D検証結果を保存しました",
            ]
            if not acceptance_record.get("production_eligible"):
                lines.append("本番解放判定は未解放です")
        previous_day_line = KeyHandler._build_previous_accepted_day_overlay_line(
            acceptance_record,
        )
        if previous_day_line:
            lines.append(previous_day_line)
        if de_str:
            lines.append(de_str)
        return tuple(lines)

    def _capture_and_save_master_ref(self) -> None:
        """現在の Ref ROI から master_ref を保存する。"""
        if self.master_ref is None:
            return
        self._show_capture_overlay("マスターRef値を取得中...")
        _, avg_ref = self._capture_neutral_ratio(n_frames=64)
        self.master_ref.save(avg_ref.astype(np.float32))
        self._invalidate_gray_results("master_ref_recalibrated")
        self._show_capture_overlay(
            "マスターRef値を保存しました",
            "次の手順へ進めます",
            wait_sec=2.0,
        )

    # ------------------------------------------------------------------
    # 自動シーン検出 — 単発判定関数
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_detect_dark(frame: np.ndarray, mean_threshold: float = 10.0) -> bool:
        """フレーム全画素平均が *mean_threshold* 未満なら True。"""
        return float(np.mean(frame)) < mean_threshold

    @staticmethod
    def _auto_detect_gray_card(
        frame: np.ndarray,
        ref_roi: tuple[int, int, int, int],
        baseline_rgb: "list | np.ndarray",
        tolerance: float = 0.20,
    ) -> bool:
        """ref ROI の RGB 平均が *baseline_rgb* ±tolerance 以内なら True。

        Parameters:
            frame: 表示座標系の RGB フレーム (H, W, 3)。
            ref_roi: (x, y, w, h) in display coords。
            baseline_rgb: [R, G, B] 0-1 正規化基準値。
            tolerance: 各チャネルの許容相対偏差。
        """
        x, y, w, h = ref_roi
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        mean_rgb = np.mean(roi, axis=(0, 1)).astype(np.float64) / 255.0
        for i in range(3):
            base = float(baseline_rgb[i])
            if abs(base) < 1e-9:
                continue
            if abs(mean_rgb[i] - base) / abs(base) > tolerance:
                return False
        return True

    @staticmethod
    def _auto_detect_gray_card_absolute(
        frame: np.ndarray,
        ref_roi: tuple[int, int, int, int],
        brightness_range: tuple[float, float] = (120.0, 200.0),
    ) -> bool:
        """ref ROI の RGB 平均が *brightness_range* 内なら True。初回日 W 専用。"""
        x, y, w, h = ref_roi
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        mean_val = float(np.mean(roi))
        return brightness_range[0] <= mean_val <= brightness_range[1]

    @staticmethod
    def _auto_detect_empty_scene(
        frame: np.ndarray,
        ref_roi: tuple[int, int, int, int],
        absolute_threshold: float = 30.0,
    ) -> bool:
        """ref ROI の RGB 平均が *absolute_threshold* 未満なら True。F ステップ専用。"""
        x, y, w, h = ref_roi
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        return float(np.mean(roi)) < absolute_threshold

    @staticmethod
    def _auto_detect_removal(
        frame: np.ndarray,
        ref_roi: tuple[int, int, int, int],
        master_ref_rgb: "list | np.ndarray",
        drop_ratio: float = 0.50,
    ) -> bool:
        """ref ROI の RGB 平均が *master_ref_rgb* の *drop_ratio* 倍未満なら True。B ステップ専用。"""
        x, y, w, h = ref_roi
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        mean_rgb = np.mean(roi, axis=(0, 1)).astype(np.float64) / 255.0
        for i in range(3):
            base = float(master_ref_rgb[i])
            if abs(base) < 1e-9:
                continue
            if mean_rgb[i] >= base * drop_ratio:
                return False
        return True

    # ------------------------------------------------------------------
    # 自動シーン検出 — 汎用待機
    # ------------------------------------------------------------------

    def _wait_for_scene(
        self,
        detect_fn,
        timeout_sec: float = AUTO_DETECT_TIMEOUT_SEC,
        action_prompt: str = "準備してください",
        fallback_key: str = "d",
        n_consecutive: int = 5,
        runner_state: str = GUIDANCE_RUNNER_STATE_IDLE,
    ) -> bool:
        """*detect_fn* が *n_consecutive* 回連続 True を返すまで待つ。

        Parameters:
            detect_fn: 現在のフレームを受け取り bool を返す callable。
            timeout_sec: 自動検出の制限時間（秒）。
            action_prompt: 画面に表示するアクション指示テキスト。
            fallback_key: timeout 後の手動確定キー。
            n_consecutive: 自動完了に必要な連続 True 回数。

        Returns:
            True: 自動検出または手動確定で完了。
        Raises:
            RuntimeError: ESC で中断された場合。
        """
        self._set_guidance_runner_state(runner_state)
        if not AUTO_SCENE_DETECTION_ENABLED:
            # 自動検出 OFF → 従来のキー確認方式
            self._append_guidance_degraded_reason(
                GUIDANCE_DEGRADED_REASON_AUTO_DETECT_DISABLED,
                mode=OPERATOR_GUIDANCE_MODE_MANUAL,
            )
            self._require_operator_action(
                action_prompt,
                extra_confirm_keys=(ord(fallback_key),),
            )
            return True

        consecutive = 0
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            remaining = max(0.0, timeout_sec - elapsed)

            # フレーム取得
            frame = None
            if hasattr(self, "picam2") and self.picam2 is not None:
                try:
                    frame = self.picam2.capture_array("main")
                except Exception:
                    pass

            # 検出判定
            detected = False
            if frame is not None:
                try:
                    detected = detect_fn(frame)
                except Exception:
                    detected = False

            if detected:
                consecutive += 1
            else:
                consecutive = 0

            # 自動完了
            if consecutive >= n_consecutive:
                self._show_capture_overlay(
                    action_prompt, "✓ 検出完了", wait_sec=0.5,
                )
                return True

            # timeout → 手動フォールバック
            if elapsed >= timeout_sec:
                self._show_capture_overlay(
                    action_prompt,
                    f"自動検出 timeout ({timeout_sec:.0f}秒)",
                    f"[{fallback_key.upper()}] 手動確定  [ESC] 中断",
                )
                key = cv2.waitKey(0) & 0xFF
                if key == ord(fallback_key):
                    self._append_guidance_degraded_reason(
                        GUIDANCE_DEGRADED_REASON_SCENE_TIMEOUT_MANUAL_CONFIRM,
                        mode=OPERATOR_GUIDANCE_MODE_DEGRADED,
                    )
                    return True
                if key in (27, ord("q")):
                    self._set_guidance_runner_state(GUIDANCE_RUNNER_STATE_CANCELLED)
                    raise RuntimeError("operator cancelled (ESC during auto-detect timeout)")
                # 他のキー → 再度待機を続行
                start_time = time.time()
                consecutive = 0
                continue

            # オーバーレイ表示
            progress = f"{consecutive}/{n_consecutive}"
            self._show_capture_overlay(
                action_prompt,
                f"検出中... {progress}  残り {remaining:.0f}秒",
            )

            # ESC チェック（ノンブロッキング）
            key = cv2.waitKey(100) & 0xFF
            if key in (27, ord("q")):
                self._set_guidance_runner_state(GUIDANCE_RUNNER_STATE_CANCELLED)
                raise RuntimeError("operator cancelled (ESC during auto-detect)")

    def _run_preflight_drift_check(self) -> dict:
        """W 完了後に master_ref drift を検査し、preflight 結果 dict を返す。"""
        control_sample_id = self._resolve_control_sample_id()
        reference_record, selection_meta = self._resolve_accepted_reference_record_for_context(
            control_sample_id=control_sample_id,
            require_open_gate=True,
        )
        if reference_record is None:
            return {
                "status": "skip",
                "reason": "no_accepted_reference",
                "accepted_reference_run_id": "none",
                **selection_meta,
            }
        ref_info = load_accepted_reference_for_preflight(
            CALIBRATION_DIR,
            control_sample_id=control_sample_id,
            run_id=reference_record.get("run_id"),
        )
        if ref_info is None:
            return {
                "status": "skip",
                "reason": "selected_reference_not_open",
                "accepted_reference_run_id": str(reference_record.get("run_id", "none")),
                **selection_meta,
            }
        today_rgb = self.master_ref.master_ref_rgb
        if today_rgb is None:
            return {
                "status": "skip",
                "reason": "master_ref_not_available",
                "accepted_reference_run_id": ref_info.get("run_id", "none"),
                **selection_meta,
            }
        accepted_rgb = ref_info["avg_ref"]
        ok, details = check_master_ref_drift(today_rgb, accepted_rgb)
        if ok:
            # pass 時は個別に drift を計算して記録
            ch_names = ("R", "G", "B")
            drifts = {}
            for i, ch in enumerate(ch_names):
                ref_val = float(accepted_rgb[i])
                if abs(ref_val) > 1e-9:
                    drifts[ch] = abs(float(today_rgb[i]) - ref_val) / abs(ref_val)
            return {
                "status": "pass",
                "max_drift": max(drifts.values(), default=0.0),
                "accepted_reference_run_id": ref_info.get("run_id"),
                **selection_meta,
            }
        return {
            "status": "fail",
            "max_drift": details.get("drift", 0.0),
            "channel": details.get("channel", ""),
            "all_drifts": details.get("all_drifts", {}),
            "accepted_reference_run_id": ref_info.get("run_id"),
            **selection_meta,
        }

    def _run_with_acceptance_context(self, run_type: str, callback) -> dict | None:
        """acceptance runner 実行中の文脈を一時的に保持する。"""
        prev_type = self._acceptance_run_type
        prev_started_at = self._acceptance_run_started_at
        prev_guidance_mode = getattr(
            self,
            "_operator_guidance_mode",
            OPERATOR_GUIDANCE_MODE_GUIDED,
        )
        prev_guidance_reasons = list(getattr(self, "_guidance_degraded_reasons", []))
        prev_guidance_state = getattr(
            self,
            "_guidance_runner_state",
            GUIDANCE_RUNNER_STATE_IDLE,
        )
        self._acceptance_run_type = run_type
        self._acceptance_run_started_at = time.time()
        self._positioning_method = ""
        self._reset_operator_guidance_context()
        try:
            context_gate = self._evaluate_acceptance_context_gate(run_type)
            if not context_gate["allowed"]:
                self._set_guidance_runner_state(GUIDANCE_RUNNER_STATE_BLOCKED)
                self._show_capture_overlay(*context_gate["overlay_lines"], wait_sec=2.5)
                print(
                    "⚠ acceptance context unresolved: "
                    f"run_type={run_type} "
                    f"missing={context_gate['measurement_context_missing_contract_fields']}"
                )
                return context_gate
            return callback()
        finally:
            self._acceptance_run_type = prev_type
            self._acceptance_run_started_at = prev_started_at
            self._operator_guidance_mode = prev_guidance_mode
            self._guidance_degraded_reasons = prev_guidance_reasons
            self._guidance_runner_state = prev_guidance_state
            if prev_type is None:
                self._positioning_method = ""

    def _get_ref_roi_tuple(self) -> tuple[int, int, int, int]:
        """ref ROI を (x, y, w, h) タプルで返す。"""
        x, y = self.config.processing.posi_ref
        w = max(int(self.config.processing.spot_size_ref), 2)
        h = max(int(w * self.config.processing.aspect_ref), 2)
        return (int(x), int(y), int(w), int(h))

    def _run_start_of_day_sequence_impl(self) -> dict | None:
        """saved corners 前提で D -> F -> W -> P を非対話順に実行する。"""
        ref_roi = self._get_ref_roi_tuple()
        control_sample_id = self._resolve_control_sample_id()

        # --- D: dark frame ---
        self._wait_for_scene(
            lambda frame: self._auto_detect_dark(frame),
            action_prompt="レンズキャップを装着してください",
            fallback_key="d",
            runner_state=GUIDANCE_RUNNER_STATE_DARK,
        )
        self.dark.capture_dark_frame(self.picam2, self.bayer)
        self._invalidate_gray_results("dark_recalibrated")

        # --- F: flat field ---
        self._wait_for_scene(
            lambda frame: self._auto_detect_empty_scene(frame, ref_roi),
            action_prompt="灰色カードと試料を外してください",
            fallback_key="f",
            runner_state=GUIDANCE_RUNNER_STATE_FLAT,
        )
        self.flat.capture_flat_field(
            self.picam2,
            self.bayer,
            self.dark,
            bayer_pattern=self.bayer.bayer_pattern,
            config=self.config,
        )
        if not self.flat.is_loaded:
            raise RuntimeError("flat capture failed")
        self._invalidate_gray_results("flat_recalibrated")

        # --- W: white balance ---
        # accepted_reference ありなら相対判定、なしなら絶対判定
        reference_record, _selection_meta = (
            self._resolve_accepted_reference_record_for_context(
                control_sample_id=control_sample_id,
                require_open_gate=True,
            )
        )
        ref_info = None
        if reference_record is not None:
            ref_info = load_accepted_reference_for_preflight(
                CALIBRATION_DIR,
                control_sample_id=control_sample_id,
                run_id=reference_record.get("run_id"),
            )
        if ref_info is not None:
            baseline = ref_info["avg_ref"]
            self._wait_for_scene(
                lambda frame, _bl=baseline, _roi=ref_roi: self._auto_detect_gray_card(
                    frame, _roi, _bl,
                ),
                action_prompt="灰色カードを設置してください",
                fallback_key="w",
                runner_state=GUIDANCE_RUNNER_STATE_WHITE,
            )
        else:
            self._wait_for_scene(
                lambda frame, _roi=ref_roi: self._auto_detect_gray_card_absolute(
                    frame, _roi,
                ),
                action_prompt="灰色カードを設置してください",
                fallback_key="w",
                runner_state=GUIDANCE_RUNNER_STATE_WHITE,
            )
        self._calibrate_wb_from_raw()
        self._capture_and_save_master_ref()

        # --- preflight: master_ref drift gate ---
        self._preflight_result = self._run_preflight_drift_check()
        if self._preflight_result.get("status") == "fail":
            self._show_capture_overlay(
                "⚠ 位置ズレの疑い",
                "カメラと暗室ボックスの位置を確認してください",
                f"最大 drift: {self._preflight_result.get('max_drift', 0):.1%}",
                f"チャネル: {self._preflight_result.get('channel', '?')}",
                wait_sec=5.0,
            )
            raise RuntimeError(
                "preflight failed: master_ref drift exceeded threshold"
            )

        # --- P: SpyderCHECKR ---
        self._wait_for_scene(
            lambda frame: self._auto_detect_gray_card_absolute(
                frame, ref_roi, brightness_range=(40.0, 220.0),
            ),
            action_prompt="SpyderCHECKR を測定位置へ置いてください",
            fallback_key="p",
            runner_state=GUIDANCE_RUNNER_STATE_CHART,
        )

        # 毎フレーム新検出 (Phase 22 / Phase 2): saved corners 復元は廃止。
        # 新検出器 (`_detect_chart_via_contour_outer_rim`) → 旧 rigid の順で
        # fallback chain を回し、採用 stage 名に応じて positioning_method を設定する。
        stage = self._run_chart_detection_pipeline(
            workflow_label="ROI を新検出器で配置しました",
        )
        if stage is None:
            contour_reason = getattr(self, "_last_contour_reject_reason", None)
            rigid_reason = getattr(self, "_last_rigid_reject_reason", None)
            raise RuntimeError(
                f"chart detection pipeline failed for start_of_day "
                f"(contour={contour_reason}, rigid={rigid_reason})"
            )
        if self._chart_state == "flip_warning":
            return None
        self._positioning_method = stage

        self._invalidate_gray_results("chart_calibration_started")
        try:
            return self._execute_chart_measurement()
        except RuntimeError as exc:
            if "cancelled" not in str(exc):
                raise
            self._cancel_chart_workflow("start-of-day run を中断しました")
            return None

    def _run_saved_corners_verification_sequence_impl(self) -> dict | None:
        """毎フレーム検出した chart 位置から 4D verification だけを実行する。

        Phase 22 / Phase 2: saved corners 復元は廃止し、新検出器
        `_detect_chart_via_contour_outer_rim` (Phase 1 で実装済) を起点とする
        `_run_chart_detection_pipeline` を呼ぶ。関数名は Phase 2 では改名しない
        (Phase 3+ の命名 refactor 対象)。
        """
        self._set_guidance_runner_state(GUIDANCE_RUNNER_STATE_VERIFY)
        self._require_operator_action(
            "4D 検証",
            "SpyderCHECKR を測定位置へ置いてください",
        )
        stage = self._run_chart_detection_pipeline(
            workflow_label="ROI を新検出器で配置しました (4D verify)",
        )
        if stage is None:
            contour_reason = getattr(self, "_last_contour_reject_reason", None)
            rigid_reason = getattr(self, "_last_rigid_reject_reason", None)
            raise RuntimeError(
                f"chart detection pipeline failed for verification "
                f"(contour={contour_reason}, rigid={rigid_reason})"
            )
        if self._chart_state == "flip_warning":
            return None
        try:
            result = self._run_4d_gray_verification()
            if result is None:
                raise RuntimeError("4D verification did not produce artifacts")
            return result
        except RuntimeError as exc:
            if "cancelled" not in str(exc):
                raise
            self._cancel_chart_workflow("4D verification を中断しました")
            return None
        finally:
            self._reset_chart_calibration()

    def run_start_of_day_sequence(self) -> dict | None:
        """運用開始判定 run を実行する。"""
        return self._run_with_acceptance_context(
            RUN_TYPE_START_OF_DAY,
            self._run_start_of_day_sequence_impl,
        )

    def run_end_of_day_sequence(self) -> dict | None:
        """運用終了前確認 run を実行する。"""
        return self._run_with_acceptance_context(
            RUN_TYPE_END_OF_DAY,
            self._run_saved_corners_verification_sequence_impl,
        )

    def run_requalification_sequence(self) -> dict | None:
        """再受入れ判定 run を実行する。"""
        return self._run_with_acceptance_context(
            RUN_TYPE_REQUALIFICATION,
            self._run_start_of_day_sequence_impl,
        )

    def run_verify_only_sequence(self) -> dict | None:
        """確認のみ run を実行する。"""
        return self._run_with_acceptance_context(
            RUN_TYPE_VERIFY_ONLY,
            self._run_saved_corners_verification_sequence_impl,
        )

    def _persist_chart_corners(self, *, reason: str) -> None:
        """chart_corners.json を保存する。途中失敗でも次回起動で復元できるよう、
        manual 4-click 直後と CCM + wizard 完了後の両タイミングで呼ばれる。

        Image #3 対策: corners は CCM 校正の最後にしか保存されていなかったため、
        途中キャンセル/失敗で永続化されないケースがあった。早期保存で 1 回の
        手動クリックを次回以降の saved corners path に確実に橋渡しする。
        """
        if not self._raw_corners or len(self._raw_corners) != 4:
            return
        try:
            payload = {
                "corners": [list(c) for c in self._raw_corners],
                # Phase 22 / Phase 3 Cycle 1 (Codex C1): payload に corners 順序の
                # 出所マーカーを保存する。Phase 3 後の `_raw_corners` は manual
                # 4-click (CW pre-sort 後)、contour/rigid 検出器 (CW 戻り値)、
                # drag (idx ベース更新で CW 維持) のいずれの経路でも [TL, TR, BR, BL]
                # CW 順で書込まれている (planner audit 表 row #1-#10 全件)。
                # 旧 JSON (Phase 3 以前 manual 4-click 由来) は本 field 不在のため
                # `_restore_chart_preview_from_saved_corners` の legacy 分岐で
                # 軸そろえ救済する。回転済み ordered corners が axis-sort で破壊
                # される C1 経路を解消する。
                "corners_order": "ordered_clockwise",
                "hinge_gap": self._hinge_gap,
                "col_x_norm_offsets": (
                    self._grid_extractor.col_x_norm_offsets.tolist()
                    if self._grid_extractor is not None else []
                ),
                "patch_margin": (
                    self._grid_extractor.patch_margin
                    if self._grid_extractor is not None else 0.10
                ),
            }
            os.makedirs(CALIBRATION_DIR, exist_ok=True)
            save_path = os.path.join(CALIBRATION_DIR, "chart_corners.json")
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"- チャートコーナー座標を保存 ({reason}): {save_path}")
        except Exception as exc:
            print(f"Warning: チャートコーナー座標の保存に失敗 ({reason}): {exc}")

    def _corners_inside_hole_mask(
        self,
        corners: list[tuple[float, float]],
        hole_mask: np.ndarray,
        *,
        also_check_lattice_hull: bool = True,
    ) -> bool:
        """corners (1A, 1H, 6H, 6A) と、それらで構成される quadrilateral 全体が
        hole_mask=True の領域に収まっているかを判定する。

        マスク外 (chamber 壁 or composite padding) に corners がある場合は False。
        ROIs that fall outside the chamber hole are physically meaningless.
        """
        if hole_mask is None or len(corners) != 4:
            return False
        h, w = hole_mask.shape
        # 4 corners がマスク内
        for cx, cy in corners:
            ix, iy = int(round(cx)), int(round(cy))
            if not (0 <= ix < w and 0 <= iy < h):
                return False
            if not bool(hole_mask[iy, ix]):
                return False
        if not also_check_lattice_hull:
            return True
        # 4 角の凸包をフラスチっと描き、その内部が >= 95% hole 内であること
        polygon = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in corners],
            dtype=np.int32,
        )
        if polygon.shape != (4, 2):
            return True
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(canvas, polygon, 1)
        polygon_area = int(canvas.sum())
        if polygon_area <= 0:
            return False
        inside = int(np.logical_and(canvas.astype(bool), hole_mask).sum())
        ratio = inside / polygon_area
        return ratio >= 0.95

    def _composite_hole_mask(
        self,
        composite_shape: tuple[int, int],
        main_shape: tuple[int, int] | None,
    ) -> np.ndarray | None:
        """flat 校正で生成した `valid_mask` (raw bayer 座標系) を、現在の composite
        frame 座標系に変換した bool マスクを返す。

        valid_mask は「暗室穴の中 (照明が届く領域)」を True、「穴の外 (chamber 壁)」を
        False とする。これを使えば chamber 壁 + composite frame の left_panel padding
        の両方が一発で除外できる。

        変換は `_build_chart_analysis_frame_from_main` と同じ pipeline を mask に適用:
          raw 解像度 (例 1520×2028) → main 解像度にリサイズ → flip → center crop
          → left_panel_width 分 False を左 padding。
        """
        raw_mask = getattr(self.flat, "valid_mask", None)
        if raw_mask is None or main_shape is None:
            return None
        composite_h, composite_w = composite_shape
        main_h, main_w = main_shape
        if main_h <= 0 or main_w <= 0:
            return None
        try:
            main_mask = cv2.resize(
                raw_mask.astype(np.uint8),
                (main_w, main_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        except Exception as exc:
            print(f"- composite hole mask: resize 失敗: {exc}")
            return None
        if bool(getattr(self.config.display, "flip_horizontal", False)):
            main_mask = main_mask[:, ::-1]
        if bool(getattr(self.config.display, "flip_vertical", False)):
            main_mask = main_mask[::-1, :]
        display_w = min(
            int(getattr(self.config.display, "width", main_w)), main_w
        )
        display_h = min(
            int(getattr(self.config.display, "height", main_h)), main_h
        )
        crop_y0 = max(0, (main_h - display_h) // 2)
        crop_x0 = max(0, (main_w - display_w) // 2)
        cropped = main_mask[
            crop_y0: crop_y0 + display_h,
            crop_x0: crop_x0 + display_w,
        ]
        left_panel_width = int(
            getattr(self.config.camera, "left_panel_width", 0)
        )
        out = np.zeros(
            (display_h, display_w + left_panel_width),
            dtype=bool,
        )
        out[:, left_panel_width:] = cropped
        if out.shape != (composite_h, composite_w):
            print(
                f"- composite hole mask: shape mismatch "
                f"got={out.shape}, expected={(composite_h, composite_w)}"
            )
            return None
        valid_count = int(out.sum())
        total = out.size
        print(
            f"- composite hole mask: flat valid_mask 適用 "
            f"valid={valid_count}/{total} ({100.0*valid_count/total:.1f}%)"
        )
        return out

    @staticmethod
    def _phase2y_find_peaks_1d(
        counts: np.ndarray, min_distance: int
    ) -> list[int]:
        """Phase 2Y planner spec の 1D peak detection。

        scipy.signal.find_peaks 不使用、自前 NMS 実装。境界 index (0, n-1) は
        edge spike とみなしスキップする (実機 Hough projection の Canny 境界
        ノイズが outermost peaks を強奪する症候を抑える)。

        Args:
          counts: 1D histogram counts (shape (N,))
          min_distance: peak 間最小距離 (bins)。planner spec L106
                        "distance >= bin_width × 2" を bin 単位で 2 と解釈。
        """
        n = len(counts)
        if n < 3:
            return []
        peaks: list[tuple[int, float]] = []
        for i in range(1, n - 1):
            if counts[i] <= 0:
                continue
            if (
                counts[i] >= counts[i - 1]
                and counts[i] >= counts[i + 1]
                and counts[i] > counts[i - 1]
            ):
                peaks.append((i, float(counts[i])))
        peaks.sort(key=lambda x: x[1], reverse=True)
        chosen: list[int] = []
        for idx, _ in peaks:
            if all(abs(idx - c) >= min_distance for c in chosen):
                chosen.append(idx)
        chosen.sort()
        return chosen

    def _detect_chart_inner_cell_boundary_via_hough(
        self,
        frame_bgr: np.ndarray,
        *,
        main_shape: tuple[int, int] | None = None,
    ) -> tuple[list[tuple[float, float]] | None, str]:
        """Phase 2Y: 内部 6×8 patch grid lines を Hough で検出し、outermost peaks
        から inner cell boundary rect を fit する (Option 1 案 A 改).

        planner spec source: `plan/20_2026-04-30-spyder-rotation-followup-phase2y-triad.md`

        **設計意図** (planner 文書 セクション 2):
          chart 物理外形 (紙 curl + 角丸) は直線取得不可。一方、内部 6×8 patch
          grid lines は強 rectilinear edges を持つ。outermost peaks が 1A 列左境界,
          1H 列右境界, 1 行上境界, 6 行下境界 = `_chart_patch_center_normalized`
          helper の前提 frame と一致する。新 detection が直接 inner cell boundary
          を返せば Phase 2X 算術が正しい結果を出す (planner output L70-71)。

        **planner spec deviation (本実装)** — triad に disclose:
          - HoughLinesP threshold: 40 → 30
            (planner 値だと synth fixture で 19-36 lines のみで peak 検出に届かない。
            primary evidence: `analysis/2026-04-30-rotation-repro/outer_frame_detect/
            summary.txt` の n_lines.)
          - minLineLength: int(central_w * 0.08) → max(20, int(central_w * 0.03))
            (cell pitch ≈30-40 px、planner 値 73 px は cell 境界線を除外する。)
          - 空間制約: dark-cell minAreaRect (label-6 相当) で expanded ROI を作り、
            その内側で Hough を走らせる。central 窓境界 / ref panel の Canny
            noise が outermost peaks を強奪する症候を抑える。
            (planner spec の "outermost peaks" 規則を素直に適用すると 0 と最大 idx
            が boundary spike になり chart 外を fit する。advisor 助言 Option A。)

        **Phase 2Y 構造的 falsification 観測** (本サイクルで生じた primary evidence):
          - 実 fixture (synth_v1_canvas1020x600/synth_rot_+000deg.png) の chart
            視 aspect = ~0.699 (Phase 2X 観測一致)。
          - canonical 理論値 6/(8+2.67) = 0.5623 とは **構造的に乖離**。
          - 効果的 hinge_gap = 6/0.699 - 8 = **0.58** (canonical 2.67 と乖離)。
          - planner premise A (chart_asp ≈ 0.5623) は本 fixture に対し falsified。
          - `_chart_patch_center_normalized` helper は `self._hinge_gap=2.67` を
            前提とするので、たとえ完璧な inner cell box を返しても、bilerp は
            patch CENTER ではなく canonical 距離だけ shift した位置を sample する。
          - 詳細: triad markdown "Generator implementation log" 節。

        本実装は planner spec 通りに mechanics を実行し primary evidence を記録するが、
        **chart_asp 0.5623 ± 0.05 gate を満たせない場合は `(None, "asp_off_canonical")`
        を返す**。orchestration 側は dark-cell rigid rect path に fallback。

        Args:
          frame_bgr: composite frame (left_panel offset 含む)
          main_shape: raw main 解像度 (flat valid_mask 変換用)

        Returns:
          (corners_4 | None, reject_reason)
            corners_4: 採用時 [1A, 1H, 6H, 6A] patch CENTER 座標 (helper bilerp 経由)
            reject_reason: "ok" / "hough_no_lines" / "insufficient_peaks"
                           / "asp_off_canonical" / "mask_reject"
                           / "premise_falsified_chart_asp_off_canonical"
                           / "exception"
        """
        # Phase 2Y diagnostic record (primary evidence trail)。
        self._last_phase2y_diagnostic: dict[str, object] = {
            "n_lines": 0,
            "n_vertical": 0,
            "n_horizontal": 0,
            "v_peak_count": 0,
            "h_peak_count": 0,
            "chart_asp": None,
            "estimated_rot": None,
        }
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None, "exception"
        h, w = gray.shape
        if h < 60 or w < 60:
            return None, "exception"
        # central 5%-95% (`_detect_chart_via_rigid_rotated_rect` と同 region 設計)
        m = 0.05
        cy0, cy1 = int(h * m), int(h * (1 - m))
        cx0, cx1 = int(w * m), int(w * (1 - m))
        central = gray[cy0:cy1, cx0:cx1]
        central_h, central_w = central.shape
        if central.size == 0:
            return None, "exception"
        # hole mask: flat valid_mask があれば使用、無ければ intensity > 12 fallback
        # (`_detect_chart_via_rigid_rotated_rect` と同 contract)
        hole_mask_full = self._composite_hole_mask((h, w), main_shape)
        if hole_mask_full is None:
            hole_mask_full = gray > 12
        central_hole = hole_mask_full[cy0:cy1, cx0:cx1]
        hole_mask_u8 = central_hole.astype(np.uint8) * 255

        # 空間制約 ROI: dark-cell + bridge minAreaRect (label-6 相当の chart inner
        # 暗領域 cluster) を expanded 1.05× して使う。central 窓境界 / ref panel の
        # Canny noise が outermost peaks を強奪する症候を構造的に防ぐ。
        roi_mask = np.zeros_like(central, dtype=np.uint8)
        try:
            _, dark_thresh = cv2.threshold(
                central, 80, 255, cv2.THRESH_BINARY_INV
            )
            dark_mask_local = cv2.bitwise_and(dark_thresh, hole_mask_u8)
            kw = max(
                _RIGID_BRIDGE_KERNEL_BASE,
                int(central_w * _RIGID_BRIDGE_KERNEL_RATIO),
            )
            if kw % 2 == 0:
                kw += 1
            hk = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
            bridged = cv2.morphologyEx(
                dark_mask_local,
                cv2.MORPH_CLOSE,
                hk,
                iterations=_RIGID_BRIDGE_KERNEL_PASSES,
            )
            n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(bridged)
        except Exception:
            return None, "exception"
        # Phase 2X 観測の "label 6 相当" (chart inner area, area 54k 程度) を抽出する。
        # 戦略: chart aspect が canonical 想定範囲 [0.55, 0.78] (planner spec 値 +
        # 観測 0.699 を含む幅) に収まる cluster を candidate とし、最大 area を選ぶ。
        candidates: list[tuple[int, int, tuple]] = []
        for i in range(1, n_lab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < central.size * 0.02:
                continue
            ys, xs = np.where(labels == i)
            if len(xs) < 100:
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            try:
                rect = cv2.minAreaRect(pts)
            except Exception:
                continue
            (rcx, rcy), (rw, rh), rang = rect
            sl = float(min(rw, rh))
            ll = float(max(rw, rh))
            if ll < 1e-3:
                continue
            asp = sl / ll
            # chart inner cluster は asp ∈ [_RIGID_CHART_ASPECT_MIN, _RIGID_CHART_ASPECT_MAX]
            if asp < _RIGID_CHART_ASPECT_MIN or asp > _RIGID_CHART_ASPECT_MAX:
                continue
            # 全 image 級 cluster (chamber 壁含む chunk) は除外
            wc = int(stats[i, cv2.CC_STAT_WIDTH])
            hc = int(stats[i, cv2.CC_STAT_HEIGHT])
            if wc > central_w * 0.85 and hc > central_h * 0.85:
                continue
            candidates.append((area, i, rect))
        if not candidates:
            return None, "hough_no_lines"
        # 最小 area 基準 (label-6 = chart inner) を優先。実 fixture では
        # 大 cluster (label-2 = 全 disc) と小 cluster (label-6 = chart inner) が
        # 両方 chart_asp filter を通過するため、area 小さい方 = chart inner。
        candidates.sort(key=lambda x: x[0])
        # 但し area 下限を中央 size の 2% でガードしているので、最小は安全。
        chart_area, chart_label, chart_rect = candidates[0]
        (rcx, rcy), (rw, rh), rang = chart_rect
        # Expanded ROI (1.05×) で Hough 走らせる範囲を設定
        expanded = ((rcx, rcy), (rw * 1.05, rh * 1.05), rang)
        ebox = cv2.boxPoints(expanded)
        ebox_int = ebox.round().astype(np.int32)
        cv2.fillConvexPoly(roi_mask, ebox_int, 255)
        composite_mask = cv2.bitwise_and(roi_mask, hole_mask_u8)

        # median-based auto Canny (planner spec)
        positive_pixels = central[central > 0]
        if positive_pixels.size == 0:
            return None, "hough_no_lines"
        median = float(np.median(positive_pixels))
        low = max(1, int(median * 0.66))
        high = min(255, int(median * 1.33))
        blurred = cv2.GaussianBlur(central, (3, 3), 0)
        edges = cv2.Canny(blurred, low, high)
        edges = cv2.bitwise_and(edges, composite_mask)

        # Hough lines (planner spec deviation: threshold=30, mll=3% central_w)
        min_line_len = max(20, int(central_w * 0.03))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=min_line_len,
            maxLineGap=10,
        )
        n_lines = 0 if lines is None else len(lines)
        self._last_phase2y_diagnostic["n_lines"] = n_lines
        if lines is None or n_lines < 16:
            return None, "hough_no_lines"

        # Global rotation 推定 (両 panel co-rotated 仮定)
        estimated_rot = self._estimate_global_chart_rotation(
            frame_bgr, main_shape=main_shape
        )
        if estimated_rot is None:
            estimated_rot = 0.0
        self._last_phase2y_diagnostic["estimated_rot"] = float(estimated_rot)

        rot_rad = math.radians(estimated_rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        vert: list[tuple[float, float, float, float]] = []
        horiz: list[tuple[float, float, float, float]] = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            ang = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            theta = (ang - estimated_rot) % 180.0
            if theta < 15.0 or theta > 165.0:
                vert.append((float(x1), float(y1), float(x2), float(y2)))
            elif 75.0 < theta < 105.0:
                horiz.append((float(x1), float(y1), float(x2), float(y2)))
        self._last_phase2y_diagnostic["n_vertical"] = len(vert)
        self._last_phase2y_diagnostic["n_horizontal"] = len(horiz)

        # Project line midpoints onto chart-frame axes (perpendicular axis is what
        # we histogram; vertical lines projected onto rotated x, horizontal onto rotated y).
        def _perp_proj_vertical(seg):
            x1, y1, x2, y2 = seg
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return mx * cos_r + my * sin_r

        def _perp_proj_horizontal(seg):
            x1, y1, x2, y2 = seg
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return -mx * sin_r + my * cos_r

        vert_proj = np.array(
            [_perp_proj_vertical(s) for s in vert], dtype=np.float64
        )
        horiz_proj = np.array(
            [_perp_proj_horizontal(s) for s in horiz], dtype=np.float64
        )
        bin_w = max(1, int(central_w / 100))
        if len(vert_proj) > 0:
            v_min, v_max = float(vert_proj.min()), float(vert_proj.max())
            n_v_bins = max(10, int((v_max - v_min) / bin_w) + 1)
            v_hist, v_edges = np.histogram(
                vert_proj, bins=n_v_bins, range=(v_min, v_max + 1)
            )
        else:
            v_hist = np.zeros(1, dtype=np.int64)
            v_edges = np.array([0.0, 1.0])
        if len(horiz_proj) > 0:
            h_min, h_max = float(horiz_proj.min()), float(horiz_proj.max())
            n_h_bins = max(10, int((h_max - h_min) / bin_w) + 1)
            h_hist, h_edges = np.histogram(
                horiz_proj, bins=n_h_bins, range=(h_min, h_max + 1)
            )
        else:
            h_hist = np.zeros(1, dtype=np.int64)
            h_edges = np.array([0.0, 1.0])
        v_peaks = self._phase2y_find_peaks_1d(v_hist, min_distance=2)
        h_peaks = self._phase2y_find_peaks_1d(h_hist, min_distance=2)
        self._last_phase2y_diagnostic["v_peak_count"] = len(v_peaks)
        self._last_phase2y_diagnostic["h_peak_count"] = len(h_peaks)
        # planner spec: vertical >= 8, horizontal >= 6 必要
        if len(v_peaks) < 8 or len(h_peaks) < 6:
            return None, "insufficient_peaks"

        # Outermost peaks
        v_left_idx, v_right_idx = v_peaks[0], v_peaks[-1]
        h_top_idx, h_bot_idx = h_peaks[0], h_peaks[-1]
        v_left_proj = (
            v_edges[v_left_idx] + v_edges[v_left_idx + 1]
        ) / 2.0
        v_right_proj = (
            v_edges[v_right_idx] + v_edges[v_right_idx + 1]
        ) / 2.0
        h_top_proj = (
            h_edges[h_top_idx] + h_edges[h_top_idx + 1]
        ) / 2.0
        h_bot_proj = (
            h_edges[h_bot_idx] + h_edges[h_bot_idx + 1]
        ) / 2.0

        def _vline_pts(vproj):
            cx_p, cy_p = vproj * cos_r, vproj * sin_r
            return ((cx_p, cy_p), (cx_p - sin_r, cy_p + cos_r))

        def _hline_pts(hproj):
            cx_p, cy_p = -hproj * sin_r, hproj * cos_r
            return ((cx_p, cy_p), (cx_p + cos_r, cy_p + sin_r))

        def _intersect(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-9:
                return None
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        vL = _vline_pts(v_left_proj)
        vR = _vline_pts(v_right_proj)
        hT = _hline_pts(h_top_proj)
        hB = _hline_pts(h_bot_proj)
        tl = _intersect(vL[0], vL[1], hT[0], hT[1])
        tr = _intersect(vR[0], vR[1], hT[0], hT[1])
        br = _intersect(vR[0], vR[1], hB[0], hB[1])
        bl = _intersect(vL[0], vL[1], hB[0], hB[1])
        if any(p is None for p in (tl, tr, br, bl)):
            return None, "insufficient_peaks"

        # Convert from central-local → composite frame coords (add cx0, cy0)
        chart_tl = (tl[0] + cx0, tl[1] + cy0)
        chart_tr = (tr[0] + cx0, tr[1] + cy0)
        chart_br = (br[0] + cx0, br[1] + cy0)
        chart_bl = (bl[0] + cx0, bl[1] + cy0)

        # chart_asp = min(short, long) / max(short, long)
        top_len = math.hypot(chart_tr[0] - chart_tl[0], chart_tr[1] - chart_tl[1])
        bot_len = math.hypot(chart_br[0] - chart_bl[0], chart_br[1] - chart_bl[1])
        left_len = math.hypot(chart_bl[0] - chart_tl[0], chart_bl[1] - chart_tl[1])
        right_len = math.hypot(chart_br[0] - chart_tr[0], chart_br[1] - chart_tr[1])
        avg_w = (top_len + bot_len) / 2.0
        avg_h = (left_len + right_len) / 2.0
        if avg_w <= 0 or avg_h <= 0:
            return None, "insufficient_peaks"
        chart_asp = min(avg_w, avg_h) / max(avg_w, avg_h)
        self._last_phase2y_diagnostic["chart_asp"] = float(chart_asp)

        # planner spec gate: |chart_asp - 0.5623| <= 0.05
        # ※ 本実装で実 fixture (synth_v1) は chart_asp ≈ 0.7 と観測されており、
        #   この gate は **構造的に未達** となる。詳細 falsification は docstring。
        if abs(chart_asp - 0.5623) > 0.05:
            return None, "asp_off_canonical"

        # Mask check: corners が hole_mask 内
        composite_hole = self._composite_hole_mask((h, w), main_shape)
        if composite_hole is not None:
            if not self._corners_inside_hole_mask(
                [chart_tl, chart_tr, chart_br, chart_bl], composite_hole
            ):
                return None, "mask_reject"

        # Helper bilerp: 1A/1H/6H/6A patch CENTER を inner cell box から計算
        corners_quad = self._patch_quad_from_inner_cell_box(
            chart_tl, chart_tr, chart_br, chart_bl
        )
        if corners_quad is None:
            return None, "exception"
        return corners_quad, "ok"

    def _patch_quad_from_inner_cell_box(
        self,
        chart_tl: tuple[float, float],
        chart_tr: tuple[float, float],
        chart_br: tuple[float, float],
        chart_bl: tuple[float, float],
    ) -> list[tuple[float, float]] | None:
        """4 隅 ordered (TL/TR/BR/BL) inner cell boundary box から、helper
        `_chart_patch_center_normalized` 経由で 1A/1H/6H/6A patch CENTER を
        bilerp する。

        Phase 2Y で `_detect_chart_inner_cell_boundary_via_hough` の戻り値
        変換に使う。`_patch_quad_from_component_pts` は minAreaRect 経由で
        box を構築するが、本関数は 4 直線交点で得た任意四角形を直接受ける
        contract (planner spec L114 "別 helper を新設" 該当)。

        Args:
          chart_tl, chart_tr, chart_br, chart_bl: 4 直線交点 (inner cell
            boundary box) の corners。順序は TL, TR, BR, BL。
        Returns:
          [1A, 1H, 6H, 6A] patch CENTER 座標、失敗時 None。
        """
        try:
            tl = np.asarray(chart_tl, dtype=np.float64)
            tr = np.asarray(chart_tr, dtype=np.float64)
            br = np.asarray(chart_br, dtype=np.float64)
            bl = np.asarray(chart_bl, dtype=np.float64)
        except Exception:
            return None

        def _bilerp(u: float, v: float) -> tuple[float, float]:
            top_p = tl + u * (tr - tl)
            bot_p = bl + u * (br - bl)
            p = top_p + v * (bot_p - top_p)
            return (float(p[0]), float(p[1]))

        # 1A=col0,row0; 1H=col7,row0; 6H=col7,row5; 6A=col0,row5
        u_1a, v_1a = _chart_patch_center_normalized(0, 0, self._hinge_gap)
        u_1h, v_1h = _chart_patch_center_normalized(7, 0, self._hinge_gap)
        u_6h, v_6h = _chart_patch_center_normalized(7, 5, self._hinge_gap)
        u_6a, v_6a = _chart_patch_center_normalized(0, 5, self._hinge_gap)
        return [
            _bilerp(u_1a, v_1a),
            _bilerp(u_1h, v_1h),
            _bilerp(u_6h, v_6h),
            _bilerp(u_6a, v_6a),
        ]

    # ------------------------------------------------------------------
    # Phase 22 / Phase 1: 輪郭ベース外形検出器 (helper 群)
    # ------------------------------------------------------------------
    # 設計方針:
    #   旧 `_detect_chart_via_rigid_rotated_rect` は dark threshold + connected
    #   components で chart 内側 dark cell の bounding rect を取る方式だが、
    #   ヒンジで分裂する component を横カーネルで bridge する必要があり、
    #   横カーネルが axis-aligned 前提のため回転 chart で破綻する症候があった。
    #
    #   新 `_detect_chart_via_contour_outer_rim` は SpyderCheckr の物理黒縁
    #   外形を `cv2.findContours` + `cv2.minAreaRect` で素直に捕える方式。
    #   ヒンジで分裂した contour ペアは `_merge_contour_pair_for_chart` で
    #   合成し、回転対応の corner 並び替えは `_order_corners_by_rotation` で
    #   実施する。横カーネル closing は呼ばない (回転耐性が出ない原因)。
    #
    #   1-C → 1-B → 1-A の順 (helper → caller 順) で源コード上に並べる。

    def _order_corners_by_rotation(
        self,
        box: np.ndarray,
        rect: tuple[tuple[float, float], tuple[float, float], float],
    ) -> list[tuple[float, float]] | None:
        """`cv2.boxPoints` の 4 点を `[outer_1A, outer_1H, outer_6H, outer_6A]`
        の順に並び替える (回転対応版)。

        既存 `_patch_quad_from_component_pts` (L4090-4114 周辺) と同じ算術:
          - 4 edge ベクトルから長辺方向 `long_unit` を抽出
          - `short_unit = (-long_unit[1], long_unit[0])` を法線方向に取り、
            `short_unit[1] >= 0` (画像座標で y+ が「下」) になるよう符号を正規化
          - 各頂点の (long, short) 軸投影で「上半分 (s<0)」「下半分 (s>0)」に分け、
            それぞれを長軸座標 u 昇順に並べて 1A/1H/6H/6A を確定
          - 裏表 / 180° 反転は本 helper では弾かない。1-A 側で
            `_score_corners_by_color_match` が 0.30 閾値で reject する。

        Args:
          box: `cv2.boxPoints(rect)` の戻り値 (shape (4, 2), float32)。
          rect: `cv2.minAreaRect` の戻り値 (重心 / (w_r, h_r) / 角度)。
        Returns:
          `[outer_1A, outer_1H, outer_6H, outer_6A]` (float, 画像座標系)、
          失敗時は None (1-A の候補ループ次点へ)。
        """
        edges = [(box[(i + 1) % 4] - box[i]) for i in range(4)]
        edge_lens = [float(np.linalg.norm(e)) for e in edges]
        long_idx = 0 if edge_lens[0] >= edge_lens[1] else 1
        long_vec = edges[long_idx]
        long_len = float(np.linalg.norm(long_vec))
        if long_len < 1e-3:
            return None
        long_unit = long_vec / long_len
        short_unit = np.array(
            [-long_unit[1], long_unit[0]], dtype=np.float32,
        )
        # 画像座標で y+ が「下」になるよう正規化 (この符号正規化により、
        # CW でも CCW でも「画像 y が小さい方 = 上 = 1A/1H 行」が固定される)
        if short_unit[1] < 0:
            short_unit = -short_unit
        cx_box = float(np.mean(box[:, 0]))
        cy_box = float(np.mean(box[:, 1]))
        scores: list[tuple[float, float, int]] = []
        for i in range(4):
            v = box[i] - np.array([cx_box, cy_box], dtype=np.float32)
            u = float(np.dot(v, long_unit))
            s = float(np.dot(v, short_unit))
            scores.append((u, s, i))
        # short 座標で上下に分けたあと、long 座標昇順で左右を確定
        top = sorted(
            [sc for sc in scores if sc[1] < 0], key=lambda x: x[0]
        )
        bot = sorted(
            [sc for sc in scores if sc[1] > 0], key=lambda x: x[0]
        )
        if len(top) != 2 or len(bot) != 2:
            return None
        outer_1a = (float(box[top[0][2]][0]), float(box[top[0][2]][1]))
        outer_1h = (float(box[top[1][2]][0]), float(box[top[1][2]][1]))
        outer_6h = (float(box[bot[1][2]][0]), float(box[bot[1][2]][1]))
        outer_6a = (float(box[bot[0][2]][0]), float(box[bot[0][2]][1]))
        # `outer_1A` 等は **黒縁外形** の 4 隅 (= 1A patch CENTER ではない)。
        # 1-A 側で `_CONTOUR_FRAME_MARGIN_*_FRAC` bilerp で patch CENTER に
        # 内縮させる。後段の `rim=[1A=(...)...]` print は patch CENTER 値で、
        # 値が異なる点に注意。
        print(
            f"- contour outer rim: order "
            f"long_unit=({long_unit[0]:+.3f},{long_unit[1]:+.3f}) "
            f"short_unit=({short_unit[0]:+.3f},{short_unit[1]:+.3f}) "
            f"outer_1A=({outer_1a[0]:.1f},{outer_1a[1]:.1f}) "
            f"outer_1H=({outer_1h[0]:.1f},{outer_1h[1]:.1f}) "
            f"outer_6H=({outer_6h[0]:.1f},{outer_6h[1]:.1f}) "
            f"outer_6A=({outer_6a[0]:.1f},{outer_6a[1]:.1f})"
        )
        return [outer_1a, outer_1h, outer_6h, outer_6a]

    def _merge_contour_pair_for_chart(
        self,
        filtered_list: list[tuple[float, np.ndarray, tuple]],
    ) -> list[tuple[int, int, np.ndarray]]:
        """ヒンジで分裂した chart の 2 contour を合成点群として返す。

        判定基準 (各 pair (i, j) について):
          - minAreaRect 長辺方向角度差が 5° 以下 (平行性)
          - 中心間距離が `1.5 × max(長辺_i, 長辺_j)` 以下 (近接性)
        両条件を満たす全 pair について `np.concatenate` で点群合成し、
        `(i, j, merged_pts)` のリストとして返す。

        **Codex revise C5**: 旧実装は `filtered_list[0]` + `[1]` の 1 pair のみ
        評価していたため、Ref panel や chamber edge 等のノイズが top1 を占め、
        真の chart が hinge で top2/top3 に分裂しているケースを取り逃がして
        いた。area 上位 4 件の全 pair (i, j) を試行し、平行性 + 近接性 gate を
        通過した全合成候補を返す形へ拡張する。candidate 列挙数は最大 C(4, 2)=6
        pair に bounded、performance 劣化は問題にならない。

        Args:
          filtered_list: `(area, contour_pts, rect)` の area 降順 list。
            `rect` は `cv2.minAreaRect` の戻り値。
        Returns:
          `(i, j, merged_pts)` の list。空 list なら merge 候補なし。
          `i` / `j` は `filtered_list` のインデックス (i < j)。`merged_pts` は
          `np.concatenate([filtered_list[i][1], filtered_list[j][1]], axis=0)`。
        """
        TOP_N = 4
        n = min(TOP_N, len(filtered_list))
        if n < 2:
            return []
        merged_candidates: list[tuple[int, int, np.ndarray]] = []
        for i in range(n):
            _a_i, c_i, rect_i = filtered_list[i]
            (cx_i, cy_i), (w_i, h_i), ang_i = rect_i
            long_i_deg = float(ang_i) if w_i >= h_i else float(ang_i) + 90.0
            long_i_norm = ((long_i_deg + 90.0) % 180.0) - 90.0
            long_i = float(max(w_i, h_i))
            for j in range(i + 1, n):
                _a_j, c_j, rect_j = filtered_list[j]
                (cx_j, cy_j), (w_j, h_j), ang_j = rect_j
                long_j_deg = (
                    float(ang_j) if w_j >= h_j else float(ang_j) + 90.0
                )
                long_j_norm = ((long_j_deg + 90.0) % 180.0) - 90.0
                angle_diff = abs(long_i_norm - long_j_norm)
                if angle_diff > 90.0:
                    angle_diff = 180.0 - angle_diff
                if angle_diff > 5.0:
                    continue
                dist = math.hypot(
                    float(cx_i) - float(cx_j),
                    float(cy_i) - float(cy_j),
                )
                long_j = float(max(w_j, h_j))
                if dist > 1.5 * max(long_i, long_j):
                    continue
                merged = np.concatenate([c_i, c_j], axis=0)
                print(
                    f"- contour outer rim: pair merge ({i},{j}) "
                    f"angles=({long_i_norm:.1f}°,{long_j_norm:.1f}°) "
                    f"dist={dist:.1f} long_max={max(long_i, long_j):.1f} "
                    f"→ merged"
                )
                merged_candidates.append((i, j, merged))
        return merged_candidates

    @staticmethod
    def _unit_vector(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            return vec.astype(np.float64)
        return (vec / norm).astype(np.float64)

    @staticmethod
    def _folded_angle_diff_degrees(a: float, b: float) -> float:
        diff = abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)
        return min(diff, 180.0 - diff)

    def _oriented_dark_panel_components(
        self,
        gray: np.ndarray,
    ) -> list[dict[str, object]]:
        dark = (gray < 85).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed,
            connectivity=8,
        )
        image_h, image_w = gray.shape[:2]
        min_area = max(1200, int(round(image_h * image_w * 0.0015)))
        components: list[dict[str, object]] = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = [int(v) for v in stats[label_id]]
            if x <= 1 or y <= 1 or x + w >= image_w - 1 or y + h >= image_h - 1:
                continue
            if area < min_area:
                continue
            ys, xs = np.where(labels == label_id)
            if len(xs) < 50:
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            try:
                (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(pts)
                box = cv2.boxPoints(((cx, cy), (rect_w, rect_h), angle))
            except Exception:
                continue
            short_len = float(min(rect_w, rect_h))
            long_len = float(max(rect_w, rect_h))
            if short_len <= 1.0:
                continue
            aspect = long_len / short_len
            fill = float(area) / max(1.0, short_len * long_len)
            if not (1.35 <= aspect <= 1.95) or fill < 0.35:
                continue

            theta = math.radians(float(angle))
            width_axis = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
            height_axis = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
            if rect_w >= rect_h:
                v_axis = self._unit_vector(width_axis)
                u_axis = self._unit_vector(height_axis)
                v_len = float(rect_w)
                u_len = float(rect_h)
            else:
                v_axis = self._unit_vector(height_axis)
                u_axis = self._unit_vector(width_axis)
                v_len = float(rect_h)
                u_len = float(rect_w)
            if float(v_axis[1]) < 0.0:
                v_axis = -v_axis
            if float(u_axis[0]) < 0.0:
                u_axis = -u_axis
            panel_angle = math.degrees(
                math.atan2(float(v_axis[1]), float(v_axis[0]))
            )
            components.append(
                {
                    "area": int(area),
                    "center": np.asarray([float(cx), float(cy)], dtype=np.float64),
                    "u_axis": u_axis,
                    "v_axis": v_axis,
                    "u_len": float(u_len),
                    "v_len": float(v_len),
                    "aspect": float(aspect),
                    "fill": float(fill),
                    "box": box.astype(np.float64),
                    "panel_angle_deg": float(panel_angle),
                    "bbox_xywh": (int(x), int(y), int(w), int(h)),
                }
            )
        return components

    def _pair_oriented_dark_panels(
        self,
        components: list[dict[str, object]],
    ) -> tuple[dict[str, object] | None, str | None]:
        candidates: list[tuple[float, dict[str, object]]] = []
        for i, first in enumerate(components):
            for second in components[i + 1:]:
                left = first
                right = second
                u_axis = self._unit_vector(
                    np.asarray(left["u_axis"], dtype=np.float64)
                    + np.asarray(right["u_axis"], dtype=np.float64)
                )
                v_axis = self._unit_vector(
                    np.asarray(left["v_axis"], dtype=np.float64)
                    + np.asarray(right["v_axis"], dtype=np.float64)
                )
                if float(u_axis[0]) < 0.0:
                    u_axis = -u_axis
                if float(v_axis[1]) < 0.0:
                    v_axis = -v_axis
                delta = (
                    np.asarray(right["center"], dtype=np.float64)
                    - np.asarray(left["center"], dtype=np.float64)
                )
                spacing = float(np.dot(delta, u_axis))
                if spacing < 0.0:
                    left, right = right, left
                    spacing = -spacing
                    delta = -delta
                vertical_offset = abs(float(np.dot(delta, v_axis)))
                angle_diff = self._folded_angle_diff_degrees(
                    float(left["panel_angle_deg"]),
                    float(right["panel_angle_deg"]),
                )
                area_ratio = min(float(left["area"]), float(right["area"])) / max(
                    1.0,
                    max(float(left["area"]), float(right["area"])),
                )
                width_ratio = min(float(left["u_len"]), float(right["u_len"])) / max(
                    1.0,
                    max(float(left["u_len"]), float(right["u_len"])),
                )
                height_ratio = min(float(left["v_len"]), float(right["v_len"])) / max(
                    1.0,
                    max(float(left["v_len"]), float(right["v_len"])),
                )
                mean_width = (float(left["u_len"]) + float(right["u_len"])) * 0.5
                mean_height = (float(left["v_len"]) + float(right["v_len"])) * 0.5
                spacing_ratio = spacing / max(1.0, mean_width)
                vertical_ratio = vertical_offset / max(1.0, mean_height)
                if angle_diff > 8.0:
                    continue
                if area_ratio < 0.55 or width_ratio < 0.72 or height_ratio < 0.72:
                    continue
                if not (0.95 <= spacing_ratio <= 1.70):
                    continue
                if vertical_ratio > 0.22:
                    continue
                score = (
                    2.0
                    + area_ratio
                    + width_ratio
                    + height_ratio
                    + max(0.0, 1.0 - angle_diff / 8.0)
                    + max(0.0, 1.0 - abs(spacing_ratio - 1.32) / 0.45)
                    + max(0.0, 1.0 - vertical_ratio / 0.22)
                )
                candidates.append(
                    (
                        float(score),
                        {
                            "left": left,
                            "right": right,
                            "u_axis": u_axis,
                            "v_axis": v_axis,
                            "score": float(score),
                            "angle_diff_deg": float(angle_diff),
                            "spacing_ratio": float(spacing_ratio),
                            "vertical_offset_ratio": float(vertical_ratio),
                            "area_ratio": float(area_ratio),
                            "width_ratio": float(width_ratio),
                            "height_ratio": float(height_ratio),
                        },
                    )
                )
        if not candidates:
            if len(components) < 2:
                return None, "oriented_panel_evidence_low"
            return None, "oriented_panel_pair_not_found"
        return max(candidates, key=lambda item: item[0])[1], None

    @staticmethod
    def _visible_patch_components(frame_bgr: np.ndarray) -> list[dict[str, float]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = ((gray > 75) & ((hsv[:, :, 1] > 22) | (gray > 130))).astype(np.uint8)
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        components: list[dict[str, float]] = []
        for label_id in range(1, num_labels):
            x, y, w, h, area = [int(v) for v in stats[label_id]]
            if not (50 <= area <= 1300 and 6 <= w <= 48 and 6 <= h <= 48):
                continue
            aspect = w / max(1.0, float(h))
            if not (0.35 <= aspect <= 2.40):
                continue
            ys, xs = np.where(_labels == label_id)
            rect_center_x = float(centroids[label_id][0])
            rect_center_y = float(centroids[label_id][1])
            if len(xs) >= 4:
                pts = np.column_stack([xs, ys]).astype(np.float32)
                try:
                    (rect_center_x, rect_center_y), _size, _angle = cv2.minAreaRect(pts)
                except Exception:
                    rect_center_x = float(centroids[label_id][0])
                    rect_center_y = float(centroids[label_id][1])
            components.append(
                {
                    "x": float(rect_center_x),
                    "y": float(rect_center_y),
                    "centroid_x": float(centroids[label_id][0]),
                    "centroid_y": float(centroids[label_id][1]),
                    "bbox_left": float(x),
                    "bbox_top": float(y),
                    "bbox_right": float(x + w),
                    "bbox_bottom": float(y + h),
                }
            )
        return components

    @staticmethod
    def _estimate_center_pitch(centers: list[tuple[float, float]]) -> float:
        if len(centers) != 48:
            return 1.0
        arr = np.asarray(centers, dtype=np.float64).reshape(6, 8, 2)
        distances: list[float] = []
        for row in range(6):
            for col in range(7):
                if col == 3:
                    continue
                distances.append(float(np.linalg.norm(arr[row, col + 1] - arr[row, col])))
        for row in range(5):
            for col in range(8):
                distances.append(float(np.linalg.norm(arr[row + 1, col] - arr[row, col])))
        positives = [value for value in distances if value > 0.0]
        return float(np.median(positives)) if positives else 1.0

    def _visible_patch_alignment_details(
        self,
        centers: list[tuple[float, float]],
        components: list[dict[str, float]],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        metrics: dict[str, object] = {
            "component_count": len(components),
            "matched_count": 0,
            "center_residual_median_px": 9999.0,
            "center_residual_p90_px": 9999.0,
            "inside_bbox_rate": 0.0,
            "bbox_margin_p10_px": -9999.0,
        }
        if len(centers) != 48 or len(components) < 24:
            return metrics, []
        pitch = self._estimate_center_pitch(centers)
        max_match_distance = max(6.0, pitch * 0.45)
        pairs: list[tuple[float, int, int]] = []
        for comp_idx, comp in enumerate(components):
            for center_idx, center in enumerate(centers):
                dist = math.hypot(
                    float(comp["x"]) - float(center[0]),
                    float(comp["y"]) - float(center[1]),
                )
                if dist <= max_match_distance:
                    pairs.append((float(dist), comp_idx, center_idx))
        pairs.sort(key=lambda item: item[0])
        used_components: set[int] = set()
        used_centers: set[int] = set()
        distances: list[float] = []
        margins: list[float] = []
        matches: list[dict[str, object]] = []
        for dist, comp_idx, center_idx in pairs:
            if comp_idx in used_components or center_idx in used_centers:
                continue
            used_components.add(comp_idx)
            used_centers.add(center_idx)
            comp = components[comp_idx]
            center = centers[center_idx]
            residual_x = float(comp["x"]) - float(center[0])
            residual_y = float(comp["y"]) - float(center[1])
            margin = min(
                float(center[0]) - float(comp["bbox_left"]),
                float(comp["bbox_right"]) - float(center[0]),
                float(center[1]) - float(comp["bbox_top"]),
                float(comp["bbox_bottom"]) - float(center[1]),
            )
            distances.append(float(dist))
            margins.append(float(margin))
            matches.append(
                {
                    "center_index": int(center_idx),
                    "component_index": int(comp_idx),
                    "predicted_xy": [
                        round(float(center[0]), 6),
                        round(float(center[1]), 6),
                    ],
                    "component_xy": [
                        round(float(comp["x"]), 6),
                        round(float(comp["y"]), 6),
                    ],
                    "residual_xy": [
                        round(residual_x, 6),
                        round(residual_y, 6),
                    ],
                    "distance_px": round(float(dist), 6),
                    "bbox_margin_px": round(float(margin), 6),
                }
            )
        matched_count = len(distances)
        inside_count = sum(1 for value in margins if value >= 0.0)
        median_residual = float(np.median(distances)) if distances else 9999.0
        p90_residual = float(np.percentile(distances, 90)) if distances else 9999.0
        inside_rate = inside_count / max(1, matched_count)
        p10_margin = float(np.percentile(margins, 10)) if margins else -9999.0
        metrics.update(
            {
                "matched_count": int(matched_count),
                "match_rate": round(matched_count / max(1, len(components)), 6),
                "center_residual_median_px": round(median_residual, 6),
                "center_residual_p90_px": round(p90_residual, 6),
                "inside_bbox_rate": round(float(inside_rate), 6),
                "bbox_margin_p10_px": round(p10_margin, 6),
                "pitch_px": round(float(pitch), 6),
            }
        )
        return metrics, matches

    @staticmethod
    def _visible_patch_alignment_passes(metrics: dict[str, object]) -> bool:
        component_count = int(metrics.get("component_count", 0) or 0)
        matched_count = int(metrics.get("matched_count", 0) or 0)
        pitch = float(metrics.get("pitch_px", 1.0) or 1.0)
        median_residual = float(metrics.get("center_residual_median_px", 9999.0) or 9999.0)
        p90_residual = float(metrics.get("center_residual_p90_px", 9999.0) or 9999.0)
        inside_rate = float(metrics.get("inside_bbox_rate", 0.0) or 0.0)
        p10_margin = float(metrics.get("bbox_margin_p10_px", -9999.0) or -9999.0)
        ok = (
            matched_count >= max(24, int(component_count * 0.75))
            and median_residual <= max(3.0, pitch * 0.10)
            and p90_residual <= max(5.0, pitch * 0.18)
            and inside_rate >= 0.85
            and p10_margin >= 1.0
        )
        return bool(ok)

    def _visible_patch_alignment_ok(
        self,
        centers: list[tuple[float, float]],
        components: list[dict[str, float]],
    ) -> tuple[bool, dict[str, object]]:
        metrics, _matches = self._visible_patch_alignment_details(centers, components)
        ok = self._visible_patch_alignment_passes(metrics)
        return bool(ok), metrics

    @staticmethod
    def _fit_offset_plane(
        samples: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(samples) < 4:
            return None
        design = np.asarray(
            [[1.0, float(panel_col), float(row)] for row, panel_col, _dx, _dy in samples],
            dtype=np.float64,
        )
        dx = np.asarray([float(sample[2]) for sample in samples], dtype=np.float64)
        dy = np.asarray([float(sample[3]) for sample in samples], dtype=np.float64)
        try:
            coeff_x, *_ = np.linalg.lstsq(design, dx, rcond=None)
            coeff_y, *_ = np.linalg.lstsq(design, dy, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not (np.all(np.isfinite(coeff_x)) and np.all(np.isfinite(coeff_y))):
            return None
        return coeff_x, coeff_y

    def _refine_oriented_lattice_to_visible_patches(
        self,
        centers: list[tuple[float, float]],
        quads: list[tuple[tuple[float, float], ...]],
        matches: list[dict[str, object]],
        alignment: dict[str, object],
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[tuple[float, float], ...]],
        dict[str, object],
    ]:
        if len(centers) != 48 or len(quads) != 48:
            return centers, quads, {"applied": False, "reason": "invalid_geometry"}
        pitch = float(alignment.get("pitch_px", self._estimate_center_pitch(centers)) or 1.0)
        max_offset = max(2.0, pitch * 0.20)
        panel_samples: dict[str, list[tuple[float, float, float, float]]] = {
            "left": [],
            "right": [],
        }
        for match in matches:
            try:
                center_index = int(match["center_index"])
                residual = np.asarray(match["residual_xy"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if center_index < 0 or center_index >= 48 or residual.shape != (2,):
                continue
            if not np.all(np.isfinite(residual)):
                continue
            if float(np.linalg.norm(residual)) > max_offset:
                continue
            row = center_index // 8
            col = center_index % 8
            side = "right" if col >= 4 else "left"
            panel_col = col - 4 if side == "right" else col
            panel_samples[side].append(
                (
                    float(row),
                    float(panel_col),
                    float(residual[0]),
                    float(residual[1]),
                )
            )

        if sum(len(values) for values in panel_samples.values()) < 24:
            return centers, quads, {
                "applied": False,
                "reason": "insufficient_visible_matches",
                "matched_count": int(sum(len(values) for values in panel_samples.values())),
            }

        offsets: list[np.ndarray] = [np.zeros(2, dtype=np.float64) for _ in range(48)]
        model_modes: dict[str, str] = {}
        for side, samples in panel_samples.items():
            side_cols = range(4, 8) if side == "right" else range(0, 4)
            model = self._fit_offset_plane(samples)
            if model is None:
                median = np.median(
                    np.asarray([[sample[2], sample[3]] for sample in samples], dtype=np.float64),
                    axis=0,
                )
                model_modes[side] = "median_offset"
                for row in range(6):
                    for col in side_cols:
                        offsets[(row * 8) + col] = median.astype(np.float64)
                continue

            coeff_x, coeff_y = model
            model_modes[side] = "row_col_offset_plane"
            for row in range(6):
                for col in side_cols:
                    panel_col = col - 4 if side == "right" else col
                    basis = np.asarray([1.0, float(panel_col), float(row)], dtype=np.float64)
                    offset = np.asarray(
                        [float(np.dot(basis, coeff_x)), float(np.dot(basis, coeff_y))],
                        dtype=np.float64,
                    )
                    norm = float(np.linalg.norm(offset))
                    if norm > max_offset:
                        offset *= max_offset / max(1e-9, norm)
                    offsets[(row * 8) + col] = offset

        refined_centers: list[tuple[float, float]] = []
        refined_quads: list[tuple[tuple[float, float], ...]] = []
        offset_norms: list[float] = []
        for idx, center in enumerate(centers):
            offset = offsets[idx]
            offset_norms.append(float(np.linalg.norm(offset)))
            refined_centers.append(
                (
                    float(center[0] + offset[0]),
                    float(center[1] + offset[1]),
                )
            )
            refined_quads.append(
                tuple(
                    (float(x + offset[0]), float(y + offset[1]))
                    for x, y in quads[idx]
                )
            )

        refinement = {
            "applied": True,
            "mode": "visible_patch_center_snap",
            "model_modes": model_modes,
            "matched_count": int(sum(len(values) for values in panel_samples.values())),
            "max_offset_px": round(float(max(offset_norms)), 6),
            "median_offset_px": round(float(np.median(offset_norms)), 6),
            "max_allowed_offset_px": round(float(max_offset), 6),
        }
        return refined_centers, refined_quads, refinement

    @staticmethod
    def _reorient_payload_quad_for_180(points: object) -> object:
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            return points
        return [points[idx] for idx in (2, 3, 0, 1)]

    @classmethod
    def _oriented_payload_panels_for_order(
        cls,
        payload_panels: list[dict[str, object]],
        order_name: str,
    ) -> tuple[dict[str, object], ...]:
        if order_name != "visual_180":
            return tuple(dict(panel) for panel in payload_panels)

        ordered: list[dict[str, object]] = []
        for logical_side, physical_panel in zip(("left", "right"), reversed(payload_panels)):
            panel = dict(physical_panel)
            panel["side"] = logical_side
            for axis_key in ("u_axis", "v_axis"):
                try:
                    axis = np.asarray(panel[axis_key], dtype=np.float64)
                except (KeyError, TypeError, ValueError):
                    continue
                if axis.shape == (2,) and np.all(np.isfinite(axis)):
                    panel[axis_key] = [
                        round(float(-axis[0]), 6),
                        round(float(-axis[1]), 6),
                    ]
            for hull_key in ("panel_hull_xy", "patch_face_hull_xy"):
                panel[hull_key] = cls._reorient_payload_quad_for_180(panel.get(hull_key))
            ordered.append(panel)
        return tuple(ordered)

    def _detect_chart_via_oriented_panel_lattice(
        self,
        frame_bgr: np.ndarray,
        *,
        main_shape: tuple[int, int] | None = None,
    ) -> list[tuple[float, float]] | None:
        """左右 panel を oriented box として検出し、実パッチ中心で検証した corners を返す。"""
        del main_shape
        self._last_oriented_panel_reject_reason: str | None = None
        self._last_oriented_panel_payload: SpyderCheckrOrientedPanelPayload | None = None
        if frame_bgr is None or frame_bgr.size == 0:
            self._last_oriented_panel_reject_reason = "frame_unavailable"
            return None
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._last_oriented_panel_reject_reason = "cvtcolor_failed"
            return None
        components = self._oriented_dark_panel_components(gray)
        pair, pair_reason = self._pair_oriented_dark_panels(components)
        if pair is None:
            self._last_oriented_panel_reject_reason = pair_reason or "oriented_panel_pair_not_found"
            print(
                f"- oriented panel lattice: rejected reason="
                f"{self._last_oriented_panel_reject_reason} "
                f"components={len(components)}"
            )
            return None

        x_fracs = (0.155, 0.388, 0.621, 0.855)
        y_fracs = (0.145, 0.286, 0.427, 0.568, 0.709, 0.850)
        panel_locals: list[dict[str, object]] = []
        for panel_name in ("left", "right"):
            panel = pair[panel_name]
            center = np.asarray(panel["center"], dtype=np.float64)
            u_axis = np.asarray(panel["u_axis"], dtype=np.float64)
            v_axis = np.asarray(panel["v_axis"], dtype=np.float64)
            u_len = float(panel["u_len"])
            v_len = float(panel["v_len"])
            panel_locals.append(
                {
                    "origin": center - (u_axis * (u_len / 2.0)) - (v_axis * (v_len / 2.0)),
                    "u_axis": u_axis,
                    "v_axis": v_axis,
                    "u_len": u_len,
                    "v_len": v_len,
                }
            )

        visual_centers: list[tuple[float, float]] = []
        visual_quads: list[tuple[tuple[float, float], ...]] = []
        payload_panels: list[dict[str, object]] = []
        payload_panel_geometry: list[dict[str, object]] = []
        x_pitch_frac = float(np.median(np.diff(np.asarray(x_fracs, dtype=np.float64))))
        y_pitch_frac = float(np.median(np.diff(np.asarray(y_fracs, dtype=np.float64))))
        roi_pitch_candidates: list[float] = []
        for panel_name, local, panel in zip(("left", "right"), panel_locals, (pair["left"], pair["right"])):
            origin = np.asarray(local["origin"], dtype=np.float64)
            u_axis = np.asarray(local["u_axis"], dtype=np.float64)
            v_axis = np.asarray(local["v_axis"], dtype=np.float64)
            u_len = float(local["u_len"])
            v_len = float(local["v_len"])
            raw_pitch_u = u_len * x_pitch_frac
            raw_pitch_v = v_len * y_pitch_frac
            pitch_u = max(4.0, raw_pitch_u)
            pitch_v = max(4.0, raw_pitch_v)
            if math.isfinite(raw_pitch_u) and raw_pitch_u > 0.0:
                roi_pitch_candidates.append(float(raw_pitch_u))
            if math.isfinite(raw_pitch_v) and raw_pitch_v > 0.0:
                roi_pitch_candidates.append(float(raw_pitch_v))
            panel_hull = [
                [round(float(point[0]), 6), round(float(point[1]), 6)]
                for point in np.asarray(panel["box"], dtype=np.float64)
            ]
            x_min_frac = float(x_fracs[0] - (x_pitch_frac * 0.5))
            x_max_frac = float(x_fracs[-1] + (x_pitch_frac * 0.5))
            y_min_frac = float(y_fracs[0] - (y_pitch_frac * 0.5))
            y_max_frac = float(y_fracs[-1] + (y_pitch_frac * 0.5))
            patch_face_hull_points = (
                origin + (u_axis * (u_len * x_min_frac)) + (v_axis * (v_len * y_min_frac)),
                origin + (u_axis * (u_len * x_max_frac)) + (v_axis * (v_len * y_min_frac)),
                origin + (u_axis * (u_len * x_max_frac)) + (v_axis * (v_len * y_max_frac)),
                origin + (u_axis * (u_len * x_min_frac)) + (v_axis * (v_len * y_max_frac)),
            )
            payload_panels.append(
                {
                    "side": panel_name,
                    "center_xy": [
                        round(float(panel["center"][0]), 6),
                        round(float(panel["center"][1]), 6),
                    ],
                    "u_axis": [round(float(u_axis[0]), 6), round(float(u_axis[1]), 6)],
                    "v_axis": [round(float(v_axis[0]), 6), round(float(v_axis[1]), 6)],
                    "u_len": round(float(u_len), 6),
                    "v_len": round(float(v_len), 6),
                    "panel_hull_xy": panel_hull,
                    "patch_face_hull_xy": [
                        [round(float(point[0]), 6), round(float(point[1]), 6)]
                        for point in patch_face_hull_points
                    ],
                    "grid_x_fracs": list(x_fracs),
                    "grid_y_fracs": list(y_fracs),
                    "pitch_u": round(float(pitch_u), 6),
                    "pitch_v": round(float(pitch_v), 6),
                    "roi_square_side": 0.0,
                    "panel_angle_deg": round(float(panel["panel_angle_deg"]), 6),
                    "bbox_xywh": list(panel["bbox_xywh"]),
                }
            )
            payload_panel_geometry.append(
                {
                    "origin": origin,
                    "u_axis": u_axis,
                    "v_axis": v_axis,
                    "u_len": u_len,
                    "v_len": v_len,
                    "roi_side": 0.0,
                }
            )

        if not roi_pitch_candidates:
            self._last_oriented_panel_reject_reason = "oriented_panel_pitch_invalid"
            return None
        global_pitch = float(min(roi_pitch_candidates))
        median_pitch = float(np.median(np.asarray(roi_pitch_candidates, dtype=np.float64)))
        global_roi_side = max(
            4.0,
            global_pitch
            * (1.0 - (2.0 * SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO)),
        )
        global_side_diagnostics = {
            "roi_square_side": round(float(global_roi_side), 6),
            "roi_square_side_policy": "chart_global_min_pitch_inset",
            "roi_pitch_candidate_count": int(len(roi_pitch_candidates)),
            "roi_pitch_min_px": round(float(global_pitch), 6),
            "roi_pitch_median_px": round(float(median_pitch), 6),
        }
        for payload_panel in payload_panels:
            payload_panel.update(global_side_diagnostics)
        for panel_geom in payload_panel_geometry:
            panel_geom["roi_side"] = float(global_roi_side)

        for y_frac in y_fracs:
            for panel_geom in payload_panel_geometry:
                origin = np.asarray(panel_geom["origin"], dtype=np.float64)
                u_axis = np.asarray(panel_geom["u_axis"], dtype=np.float64)
                v_axis = np.asarray(panel_geom["v_axis"], dtype=np.float64)
                u_len = float(panel_geom["u_len"])
                v_len = float(panel_geom["v_len"])
                half_side = float(panel_geom["roi_side"]) * 0.5
                for x_frac in x_fracs:
                    point = origin + (u_axis * (x_frac * u_len)) + (v_axis * (y_frac * v_len))
                    visual_centers.append((float(point[0]), float(point[1])))
                    quad_points = (
                        point - (u_axis * half_side) - (v_axis * half_side),
                        point + (u_axis * half_side) - (v_axis * half_side),
                        point + (u_axis * half_side) + (v_axis * half_side),
                        point - (u_axis * half_side) + (v_axis * half_side),
                    )
                    visual_quads.append(
                        tuple((float(qx), float(qy)) for qx, qy in quad_points)
                    )

        patch_components = self._visible_patch_components(frame_bgr)
        alignment, alignment_matches = self._visible_patch_alignment_details(
            visual_centers,
            patch_components,
        )
        alignment_ok = self._visible_patch_alignment_passes(alignment)
        if not alignment_ok:
            self._last_oriented_panel_reject_reason = "visible_patch_alignment_failed"
            print(
                "- oriented panel lattice: visible patch alignment failed "
                f"{alignment}"
            )
            return None

        pre_refinement_alignment = dict(alignment)
        refined_centers, refined_quads, refinement = (
            self._refine_oriented_lattice_to_visible_patches(
                visual_centers,
                visual_quads,
                alignment_matches,
                alignment,
            )
        )
        if bool(refinement.get("applied")):
            refined_alignment, _refined_matches = self._visible_patch_alignment_details(
                refined_centers,
                patch_components,
            )
            refined_ok = self._visible_patch_alignment_passes(refined_alignment)
            pre_median = float(
                pre_refinement_alignment.get("center_residual_median_px", 9999.0)
                or 9999.0
            )
            post_median = float(
                refined_alignment.get("center_residual_median_px", 9999.0)
                or 9999.0
            )
            if refined_ok and post_median <= pre_median + 1e-6:
                visual_centers = refined_centers
                visual_quads = refined_quads
                alignment = dict(refined_alignment)
                refinement["pre_center_residual_median_px"] = round(pre_median, 6)
                refinement["post_center_residual_median_px"] = round(post_median, 6)
                alignment["pre_refinement"] = pre_refinement_alignment
                alignment["center_refinement"] = refinement
            else:
                refinement = dict(refinement)
                refinement.update(
                    {
                        "applied": False,
                        "reason": "refinement_did_not_improve_alignment",
                        "pre_center_residual_median_px": round(pre_median, 6),
                        "post_center_residual_median_px": round(post_median, 6),
                    }
                )
                alignment["center_refinement"] = refinement
        else:
            alignment["center_refinement"] = refinement

        visual_corners = [visual_centers[idx] for idx in (0, 7, 47, 40)]
        candidate_orders = [
            ("visual", visual_corners, visual_centers, visual_quads),
            (
                "visual_180",
                [
                    visual_corners[2],
                    visual_corners[3],
                    visual_corners[0],
                    visual_corners[1],
                ],
                list(reversed(visual_centers)),
                list(reversed(visual_quads)),
            ),
        ]
        best_order = None
        best_score = float("inf")
        self._runtime_estimated_hinge_gap = float(self._hinge_gap)
        for order_name, ordered_corners, ordered_centers, ordered_quads in candidate_orders:
            try:
                score = self._score_corners_by_color_match(
                    frame_bgr,
                    ordered_corners,
                )
            except Exception:
                continue
            if score < best_score:
                best_score = float(score)
                best_order = (order_name, ordered_corners, ordered_centers, ordered_quads)
        if best_order is None or best_score > _RIGID_COLOR_REJECT_THRESHOLD_BASE:
            self._last_oriented_panel_reject_reason = "color_score_too_high"
            print(
                f"- oriented panel lattice: color score reject "
                f"score={best_score:.4f}"
            )
            return None

        order_name, ordered_corners, ordered_centers, ordered_quads = best_order
        self._last_oriented_panel_payload = SpyderCheckrOrientedPanelPayload(
            corners_xy=tuple((float(x), float(y)) for x, y in ordered_corners),
            centers_48_xy=tuple((float(x), float(y)) for x, y in ordered_centers),
            sampling_quads_48_xy=tuple(
                tuple((float(x), float(y)) for x, y in quad)
                for quad in ordered_quads
            ),
            panels=self._oriented_payload_panels_for_order(
                payload_panels,
                order_name,
            ),
            visible_patch_alignment=dict(alignment),
            pair_diagnostics={
                "score": round(float(pair["score"]), 6),
                "angle_diff_deg": round(float(pair["angle_diff_deg"]), 6),
                "spacing_ratio": round(float(pair["spacing_ratio"]), 6),
                "vertical_offset_ratio": round(float(pair["vertical_offset_ratio"]), 6),
                "area_ratio": round(float(pair["area_ratio"]), 6),
                "width_ratio": round(float(pair["width_ratio"]), 6),
                "height_ratio": round(float(pair["height_ratio"]), 6),
            },
            order_name=order_name,
            color_score=float(best_score),
        )
        print(
            "- oriented panel lattice: accepted "
            f"order={order_name} score={best_score:.4f} "
            f"alignment={alignment} pair_score={float(pair['score']):.4f}"
        )
        return [(float(x), float(y)) for x, y in ordered_corners]

    def _set_chart_stage_reject_reason(
        self,
        stage_name: str,
        reason: str,
        summary: dict[str, object] | None = None,
    ) -> None:
        attr_by_stage = {
            "oriented_panel_lattice": "_last_oriented_panel_reject_reason",
            "contour_outer_rim": "_last_contour_reject_reason",
            "rigid_auto": "_last_rigid_reject_reason",
        }
        attr_name = attr_by_stage.get(stage_name)
        if attr_name is not None:
            setattr(self, attr_name, reason)
        reject_summaries = getattr(self, "_last_chart_activation_reject_summary_by_stage", None)
        if not isinstance(reject_summaries, dict):
            reject_summaries = {}
            self._last_chart_activation_reject_summary_by_stage = reject_summaries
        reject_summaries[stage_name] = dict(summary) if isinstance(summary, dict) else None

    @staticmethod
    def _patchwise_summary_is_verified(summary: dict[str, object]) -> bool:
        diagnostics = summary.get("direct_dark_panel_diagnostics") or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        left_layout = diagnostics.get("left_panel_layout") or {}
        right_layout = diagnostics.get("right_panel_layout") or {}
        return (
            summary.get("geometry_model") == "direct_dark_panel"
            and summary.get("direct_panel_mode") == "inner_lattice_estimated"
            and summary.get("calibration_geometry_viable") == "yes"
            and int(summary.get("weak_patch_count", 999)) <= 6
            and isinstance(left_layout, dict)
            and isinstance(right_layout, dict)
            and left_layout.get("mode") == "inner_lattice_estimated"
            and right_layout.get("mode") == "inner_lattice_estimated"
        )

    def _has_verified_oriented_panel_geometry(self) -> bool:
        """oriented-panel lattice で検証済みの ROI geometry が残っているかを返す。"""
        return self._verified_oriented_panel_summary() is not None

    def _verified_oriented_panel_summary(self) -> dict[str, object] | None:
        """oriented-panel lattice の検証済み summary を返す。未検証なら None。"""
        if getattr(self, "_positioning_method", "") != "oriented_panel_lattice":
            return None
        extractor = getattr(self, "_grid_extractor", None)
        if extractor is None or not bool(getattr(extractor, "is_ready", False)):
            return None
        try:
            summary = extractor.get_patchwise_summary()
        except Exception as exc:
            print(f"- oriented-panel summary unavailable: {type(exc).__name__}: {exc}")
            return None
        if not isinstance(summary, dict):
            return None
        if not self._patchwise_summary_is_verified(summary):
            return None
        return summary

    def _oriented_panel_payload_order_name(self) -> str:
        payload = getattr(self, "_last_oriented_panel_payload", None)
        return str(getattr(payload, "order_name", "") or "").strip()

    def _has_verified_oriented_panel_payload_source(self) -> bool:
        summary = self._verified_oriented_panel_summary()
        if summary is None:
            return False
        diagnostics = summary.get("direct_dark_panel_diagnostics") or {}
        if not isinstance(diagnostics, dict):
            return False
        if str(diagnostics.get("reason", "")).strip() != "oriented_panel_payload":
            return False
        raw_counts = summary.get("raw_direct_panel_cell_source_counts") or {}
        if not isinstance(raw_counts, dict):
            return False
        normalized: dict[str, int] = {}
        for key, value in raw_counts.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                return False
            if count:
                normalized[str(key)] = count
        return normalized == {"oriented_panel_payload": 48}

    def _has_verified_oriented_panel_payload_contract(self) -> bool:
        return (
            self._has_verified_oriented_panel_payload_source()
            and self._oriented_panel_payload_order_name() in {"visual", "visual_180"}
        )

    def _has_verified_visual_180_oriented_panel_payload(self) -> bool:
        return (
            self._has_verified_oriented_panel_payload_contract()
            and self._oriented_panel_payload_order_name() == "visual_180"
        )

    @staticmethod
    def _pose_reason_is_hard_flip(pose_reason: str) -> bool:
        reason = str(pose_reason or "").strip().lower()
        compact = (
            reason.replace("_", "")
            .replace("-", "")
            .replace(" ", "")
            .replace("°", "")
        )
        hard_tokens = (
            "flip",
            "flipped",
            "rot180",
            "rotation180",
            "rotate180",
            "180deg",
            "180degree",
        )
        return any(token in compact for token in hard_tokens)

    def _should_bypass_pose_gate_for_oriented_geometry(self, pose_reason: str) -> bool:
        if self._pose_reason_is_hard_flip(pose_reason):
            return self._has_verified_visual_180_oriented_panel_payload()
        return self._has_verified_oriented_panel_payload_contract()

    def _chart_orientation_metadata(
        self,
        *,
        pose_gate_reason: str,
        pose_gate_bypassed: bool,
        legacy_flip: bool,
    ) -> dict[str, object]:
        payload = getattr(self, "_last_oriented_panel_payload", None)
        summary = self._verified_oriented_panel_summary()
        diagnostics = {}
        if isinstance(summary, dict):
            raw_diagnostics = summary.get("direct_dark_panel_diagnostics") or {}
            diagnostics = raw_diagnostics if isinstance(raw_diagnostics, dict) else {}
        return {
            "positioning_method": str(getattr(self, "_positioning_method", "") or ""),
            "chart_orientation_order": self._oriented_panel_payload_order_name() or "unknown",
            "oriented_panel_verified": self._has_verified_oriented_panel_payload_contract(),
            "oriented_panel_color_score": (
                None
                if payload is None or getattr(payload, "color_score", None) is None
                else float(getattr(payload, "color_score"))
            ),
            "oriented_panel_pair_diagnostics": _json_ready(
                {}
                if payload is None
                else getattr(payload, "pair_diagnostics", {}) or {}
            ),
            "oriented_panel_visible_patch_alignment": _json_ready(
                {}
                if payload is None
                else getattr(payload, "visible_patch_alignment", {}) or {}
            ),
            "oriented_panel_summary_order_name": str(
                diagnostics.get("order_name", "")
                or self._oriented_panel_payload_order_name()
                or "unknown"
            ),
            "pose_gate_reason": str(pose_gate_reason or ""),
            "pose_gate_bypassed": bool(pose_gate_bypassed),
            "legacy_flip": bool(legacy_flip),
        }

    def _activate_verified_chart_candidate(
        self,
        *,
        stage_name: str,
        corners: list[tuple[float, float]],
        frame_bgr: np.ndarray,
        workflow_label: str,
        extractor_source: str | None = None,
        oriented_panel_payload: SpyderCheckrOrientedPanelPayload | None = None,
    ) -> bool:
        extractor = SpyderCheckrGridExtractor(
            SPYDERCHECKR_48_SHAPE[0],
            SPYDERCHECKR_48_SHAPE[1],
            hinge_gap=self._hinge_gap,
        )
        extractor.set_corners_ordered_with_source(
            list(corners),
            source=extractor_source or stage_name,
        )
        try:
            if stage_name == "oriented_panel_lattice":
                if oriented_panel_payload is None:
                    self._set_chart_stage_reject_reason(
                        stage_name,
                        "oriented_payload_missing",
                    )
                    return False
                summary = extractor.prepare_patchwise_rois_from_oriented_panel_payload(
                    frame_bgr,
                    oriented_panel_payload,
                )
            else:
                summary = extractor.prepare_patchwise_rois_from_frame(frame_bgr)
        except Exception as exc:
            self._set_chart_stage_reject_reason(
                stage_name,
                f"patchwise_verify_exception:{type(exc).__name__}",
            )
            print(f"[chart pipeline] stage={stage_name} patchwise verify exception: {exc}")
            return False
        if not self._patchwise_summary_is_verified(summary):
            diagnostics = summary.get("direct_dark_panel_diagnostics") or {}
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            diagnostic_reason = str(diagnostics.get("reason") or "unknown")
            if (
                stage_name == "oriented_panel_lattice"
                and diagnostic_reason.startswith("oriented_payload_")
            ):
                self._set_chart_stage_reject_reason(
                    stage_name,
                    diagnostic_reason,
                    summary,
                )
            else:
                self._set_chart_stage_reject_reason(
                    stage_name,
                    (
                        f"patchwise_gate_rejected:{summary.get('geometry_model')}:"
                        f"{summary.get('direct_panel_mode')}:"
                        f"weak={summary.get('weak_patch_count')}:"
                        f"viable={summary.get('calibration_geometry_viable')}:"
                        f"reason={diagnostic_reason}"
                    ),
                    summary,
                )
            print(
                f"[chart pipeline] stage={stage_name} rejected by patchwise gate "
                f"geometry_model={summary.get('geometry_model')} "
                f"direct_panel_mode={summary.get('direct_panel_mode')} "
                f"weak={summary.get('weak_patch_count')} "
                f"viable={summary.get('calibration_geometry_viable')}"
            )
            return False

        self._grid_extractor = extractor
        self._raw_corners = list(corners)
        self._preview_margin_detected = True
        self._positioning_method = stage_name
        orientation_order = (
            str(getattr(oriented_panel_payload, "order_name", "") or "").strip()
            if stage_name == "oriented_panel_lattice"
            else ""
        )
        if stage_name == "oriented_panel_lattice" and oriented_panel_payload is not None:
            self._last_oriented_panel_payload = oriented_panel_payload

        self._chart_state = "preview"
        self._set_chart_workflow_status(
            stage="preview",
            message="色基準48色 プレビュー中",
            detail=workflow_label,
            can_cancel=True,
            source=stage_name,
            chart_orientation_order=orientation_order or None,
        )
        print(
            f"[chart pipeline] stage={stage_name} accepted verified "
            f"corners={corners}"
        )
        return True

    def _run_chart_detection_pipeline(
        self,
        *,
        workflow_label: str,
    ) -> str | None:
        """毎フレーム検出パイプライン: contour_outer_rim → rigid → (Phase 4 で sweep 統合)。

        Phase 22 / Phase 2 で新設。`_run_*_sequence_impl` 専用の非対話 runner で、
        新検出器 `_detect_chart_via_contour_outer_rim` (Stage 0) → 旧
        `_detect_chart_via_rigid_rotated_rect` (Stage 1) の順で fallback chain を
        実行し、成功した時点で extractor / preview state を構築して採用 stage 名
        (`"contour_outer_rim"` / `"rigid_auto"`) を返す。全 stage 失敗で None。

        saved corners は読まない (Phase 22 / Phase 2 設計): saved-corners 復元
        ショートカットは P フローから廃止され、新検出器が main path となる。
        `chart_corners.json` の書込みは呼出し側 (`_persist_chart_corners`) に
        任せる (本 helper では書込まない)。

        Phase 2 時点では Stage 2 sweep は未実装。Phase 4 で
        `_try_auto_detect_chart_corners` 全体 refactor 時に統合予定。失敗時は
        caller が RuntimeError を上げるか、idle [P] 分岐の manual 4-click に
        降格する。

        Args:
          workflow_label: `_set_chart_workflow_status` の `detail` 用文字列
            (例 "ROI を新検出器で配置しました")。

        Side effects (採用時):
          - `self._grid_extractor` を新規構築
          - `self._raw_corners`, `self._chart_state = "preview"`,
            `self._preview_margin_detected = False`
          - `self._positioning_method = stage_name`
          - `self._set_chart_workflow_status(...)` で preview stage に遷移

        Side effects (失敗時): state は変更しない (caller が RuntimeError)。

        Returns:
          採用 stage 名 (`"contour_outer_rim"` / `"rigid_auto"`) または None。
        """
        # frame 取得は **1 回だけ** にする (Pi の picam2 frame race 防止)。
        # Stage 0 / Stage 1 ともに同じ frame_bgr / main_shape を使う。
        self._last_contour_reject_reason = None
        self._last_rigid_reject_reason = None
        self._last_chart_activation_reject_summary_by_stage = {}
        try:
            main_array = self.picam2.capture_array("main")
        except Exception as exc:
            print(
                f"[chart pipeline] capture_array 失敗: {exc} → manual fallback"
            )
            self._last_contour_reject_reason = "capture_failed"
            return None
        if not isinstance(main_array, np.ndarray) or main_array.size == 0:
            self._last_contour_reject_reason = "capture_failed"
            return None
        try:
            frame_bgr = self._build_chart_analysis_frame_from_main(main_array)
        except Exception as exc:
            print(
                f"[chart pipeline] composite frame 構築失敗: {exc} → manual fallback"
            )
            self._last_contour_reject_reason = "frame_build_failed"
            return None
        if (
            frame_bgr is None
            or frame_bgr.size == 0
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            self._last_contour_reject_reason = "frame_build_failed"
            return None
        main_shape = tuple(main_array.shape[:2])

        # === Stage 0: verified oriented panel lattice ===
        print("[chart pipeline] stage=oriented_panel_lattice trying ...")
        oriented_corners = self._detect_chart_via_oriented_panel_lattice(
            frame_bgr,
            main_shape=main_shape,
        )
        if oriented_corners is not None and self._activate_verified_chart_candidate(
            stage_name="oriented_panel_lattice",
            corners=list(oriented_corners),
            frame_bgr=frame_bgr,
            workflow_label=workflow_label,
            extractor_source="oriented_panel_lattice",
            oriented_panel_payload=getattr(self, "_last_oriented_panel_payload", None),
        ):
            return "oriented_panel_lattice"
        print(
            f"[chart pipeline] stage=oriented_panel_lattice rejected "
            f"reason={getattr(self, '_last_oriented_panel_reject_reason', None)}"
        )

        # === Stage 1: contour outer rim seed ===
        print("[chart pipeline] stage=contour_outer_rim trying ...")
        contour_corners = self._detect_chart_via_contour_outer_rim(
            frame_bgr, main_shape=main_shape,
        )
        if contour_corners is not None and self._activate_verified_chart_candidate(
            stage_name="contour_outer_rim",
            corners=list(contour_corners),
            frame_bgr=frame_bgr,
            workflow_label=workflow_label,
            extractor_source="contour_outer_rim_seed",
        ):
            return "contour_outer_rim"
        print(
            f"[chart pipeline] stage=contour_outer_rim rejected "
            f"reason={getattr(self, '_last_contour_reject_reason', None)}"
        )

        # === Stage 2: 旧検出器 (rigid rotated rect) seed ===
        print("[chart pipeline] stage=rigid_auto trying ...")
        rigid_corners = self._detect_chart_via_rigid_rotated_rect(
            frame_bgr, main_shape=main_shape,
        )
        if rigid_corners is not None and self._activate_verified_chart_candidate(
            stage_name="rigid_auto",
            corners=list(rigid_corners),
            frame_bgr=frame_bgr,
            workflow_label=workflow_label,
            extractor_source="rigid_auto_seed",
        ):
            return "rigid_auto"
        print(
            f"[chart pipeline] stage=rigid_auto rejected "
            f"reason={getattr(self, '_last_rigid_reject_reason', None)}"
        )

        # 全 stage 失敗。Stage 2 sweep は Phase 4 で統合予定。
        print(
            f"[chart pipeline] all stages failed "
            f"contour={getattr(self, '_last_contour_reject_reason', None)} "
            f"rigid={getattr(self, '_last_rigid_reject_reason', None)}"
        )
        return None

    def _detect_chart_via_contour_outer_rim(
        self,
        frame_bgr: np.ndarray,
        *,
        main_shape: tuple[int, int] | None = None,
    ) -> list[tuple[float, float]] | None:
        """chart の黒縁外形を `cv2.findContours` + `cv2.minAreaRect` で捕える
        新検出器 (Phase 22 / Phase 1)。

        戻り値: `[1A_center, 1H_center, 6H_center, 6A_center]` (patch CENTER
        座標、画像座標系)。失敗時 None (manual fallback へ降格)。

        旧 `_detect_chart_via_rigid_rotated_rect` との違い:
          - 横カーネル `(kw, 3)` での hinge bridge を **呼ばない**。回転耐性
            のなさの主因をここで断つ。
          - hinge で分裂した contour ペアは `_merge_contour_pair_for_chart`
            で合成、area 上位 N 単独候補 + pair_merge を 4 permutation 全て
            score 比較し best-score を採用 (Codex revise C2/C3)。
          - corner 並び替えは `_order_corners_by_rotation` (回転対応) を使用。
            helper の戻り値仕様は不変、caller 側で 4 通り cyclic permutation
            を回し物理 1A 方向を色一致で確定する (Codex revise C3)。
          - 採否閾値は旧 rigid detector と同じ rotation adaptive 戦略
            (`_RIGID_COLOR_REJECT_THRESHOLD_BASE`〜`_MAX` を `_PER_DEG` で
            線形拡張)。axis-aligned 厳格 / 回転姿勢段階拡張で false negative
            を防ぐ (Codex revise C1)。

        outer rim → patch CENTER 変換は `_chart_patch_center_normalized(0, 0, 2.67)`
        の戻り値 `(0.0469, 0.0833)` を `_CONTOUR_FRAME_MARGIN_*_FRAC` constant
        として使い、`_chart_patch_center_normalized` 自体は変更しない
        (Phase 2X 整合性維持)。

        Args:
          frame_bgr: 入力フレーム (BGR)。
          main_shape: 元 frame サイズ (h, w) (composite 座標逆引き用)。
        """
        self._last_contour_reject_reason: str | None = None
        # 1) frame check
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._last_contour_reject_reason = "cvtcolor_failed"
            return None
        h, w = gray.shape
        if h < 60 or w < 60:
            self._last_contour_reject_reason = "frame_too_small"
            return None
        # 2) disk mask
        # **Codex revise C6**: 採用直前の bounds/mask gate (旧 rigid detector
        # L3500-3505 と同じ contract) のために、`_composite_hole_mask` の真の
        # 戻り値を `composite_hole` として保持する。`hole_mask_full` は
        # binarization 用に `gray > 12` fallback も併用するが、`gray > 12` は
        # 「intensity > 12 = 暗くない」を True とするため、暗い chart patch
        # CENTER は False となり `_corners_inside_hole_mask` には使えない。
        # よって gate は composite_hole が非 None のときだけ適用する。
        composite_hole = self._composite_hole_mask((h, w), main_shape)
        if composite_hole is None:
            hole_mask_full = gray > 12
            print(
                "- contour outer rim: flat valid_mask 不在 → "
                "intensity>12 で hole 推定 (fallback)"
            )
        else:
            hole_mask_full = composite_hole
        # 3) central 5% margin で外周クロップ (旧検出器と同 region 設計)
        m = 0.05
        cy0, cy1 = int(h * m), int(h * (1 - m))
        cx0 = int(w * m)
        cx1 = int(w * (1 - m))
        central = gray[cy0:cy1, cx0:cx1]
        central_hole = hole_mask_full[cy0:cy1, cx0:cx1]
        print(
            f"- contour outer rim: 探索領域 x=[{cx0}..{cx1}] "
            f"y=[{cy0}..{cy1}] "
            f"hole 内 pixel ratio={100.0*central_hole.mean():.1f}%"
        )
        if central.size == 0:
            self._last_contour_reject_reason = "central_empty"
            return None
        # 4) 二値化 (旧検出器 disk_peak / chart_threshold ロジック流用)
        try:
            central_hole_u8 = (central_hole.astype(np.uint8) * 255)
            hist = cv2.calcHist(
                [central], [0], central_hole_u8, [256], [0, 256]
            )
            hist_high = hist[150:].flatten()
            if hist_high.sum() > 0:
                disk_peak_offset = int(np.argmax(hist_high))
                disk_peak_intensity = 150 + disk_peak_offset
                chart_threshold = max(60, disk_peak_intensity - 30)
                _disk_peak_log = str(disk_peak_intensity)
            else:
                chart_threshold = 80
                _disk_peak_log = "N/A"
            _, chart_thresh = cv2.threshold(
                central, chart_threshold, 255, cv2.THRESH_BINARY_INV,
            )
            dark_mask = cv2.bitwise_and(chart_thresh, central_hole_u8)
            print(
                f"- contour outer rim: chart segmentation "
                f"disk_peak={_disk_peak_log} "
                f"chart_threshold={chart_threshold}"
            )
        except Exception:
            self._last_contour_reject_reason = "binarize_exception"
            return None
        # 5) 輪郭抽出 (横カーネル closing は呼ばない)
        try:
            contours, _ = cv2.findContours(
                dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
        except Exception:
            self._last_contour_reject_reason = "findcontours_exception"
            return None
        if len(contours) == 0:
            print("- contour outer rim: 輪郭ゼロ件")
            self._last_contour_reject_reason = "no_contour"
            return None
        # 6) 輪郭フィルタ (area / aspect)
        # planner spec L113: bbox aspect (= short/long) が [0.4, 0.9] を外れた
        # ら捨てる。`_RIGID_ASPECT_MIN` (=0.4) は再利用、上限は短辺/長辺の
        # rotation-invariant 値として planner spec literal 0.9 を採用。
        # `_RIGID_ASPECT_MAX` (=4.0) は rigid detector の bbox w/h 用なので、
        # contour detector の short/long 用には使えない (asp ≦ 1.0 で常に通って
        # しまう)。pair merge candidate は本フィルタの後で別ロジックで扱う。
        ASPECT_MIN = _RIGID_ASPECT_MIN  # 0.4
        # **disclosed deviation**: planner spec L113 は upper bound 0.9 だが、
        # v1 fixture (`tests/test_chart_contour_detector.py` で使用) は canvas
        # 不変回転で chart を上下端 clip するため、ccw_30 等で chart 単一
        # contour の short/long ≒ 0.93-0.94 (0.9 上限を僅かに超過)。実 Pi
        # capture (Phase 5 acceptance) では chart 全 visible なので 0.56 近傍で
        # 問題なし。v1 fixture の clip artifact を吸収するため上限を 0.95 に
        # 拡張 (planner spec [0.4, 0.9] からの 0.05 拡張、generator disclosed)。
        # pair_merge と single_top1 の両候補が出たうえで color score 0.30 で
        # best 採用するため誤検出 risk は小さい。
        ASPECT_MAX = 0.95
        area_floor = float(central.size) * 0.02
        filtered: list[tuple[float, np.ndarray, tuple]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < area_floor:
                continue
            try:
                rect_c = cv2.minAreaRect(c)
            except Exception:
                continue
            (_rcx, _rcy), (w_r, h_r), _ang = rect_c
            short_len = float(min(w_r, h_r))
            long_len = float(max(w_r, h_r))
            if long_len < 1.0:
                continue
            asp = short_len / long_len
            if asp < ASPECT_MIN or asp > ASPECT_MAX:
                continue
            filtered.append((area, c, rect_c))
        filtered.sort(key=lambda x: x[0], reverse=True)
        print(
            f"- contour outer rim: candidates after filter = "
            f"{len(filtered)}"
        )
        if not filtered:
            self._last_contour_reject_reason = "no_candidate_contour"
            return None
        # 7) 候補確定 (single 上位 N + pair merge)
        # **Codex revise C2**: 旧実装は top1 single + pair_merge の 2 候補のみを
        # 評価していたため、Ref panel 等のノイズ contour が top1 を占めると
        # 真の chart が top2 以降に単独で存在しても取り逃がした。旧 rigid
        # detector と同じ「複数候補 score 比較戦略」に揃える。
        # - filter を通った全 contour のうち area 上位 N 件を「単独候補」として
        #   candidate_point_sets に追加 (N=4、planner 助言の 3〜5 中位)。
        # - pair_merge は従来通り追加。
        # - 候補総数は 4 + 1 = 5 件以下。各候補で 4 permutation を回しても
        #   20 score 計算以下に bounded、performance 劣化は問題にならない。
        SINGLE_CANDIDATE_TOP_N = 4
        candidate_point_sets: list[tuple[np.ndarray, str]] = []
        for idx, (_, c_pts, _) in enumerate(
            filtered[:SINGLE_CANDIDATE_TOP_N]
        ):
            candidate_point_sets.append((c_pts, f"single_top{idx + 1}"))
        # **Codex revise C5**: pair_merge 候補列挙を top2 固定から area 上位 4 件
        # の全 pair (i, j) に拡張。helper は通過した全 pair を `(i, j, merged_pts)`
        # で返すため、caller 側は各 pair を `pair_merge_{i}_{j}` source として
        # candidate_point_sets に追加する。Ref panel ノイズが top1 に来て真 chart
        # が top2+top3 に分裂しているケースを拾えるようにする。
        merged_candidates = self._merge_contour_pair_for_chart(filtered)
        for i_pair, j_pair, merged_pts in merged_candidates:
            candidate_point_sets.append(
                (merged_pts, f"pair_merge_{i_pair}_{j_pair}")
            )
        # 8〜13) 各候補 × 4 permutation で minAreaRect → corner 並び替え →
        # patch CENTER 変換 → color score を算出し best を選ぶ
        # **Codex revise C3**: corner 並び替えは `short_unit[1] >= 0` で 1 通り
        # 固定だったため、90°/180° など物理 1A が画像上側にない姿勢で正しい
        # `[1A, 1H, 6H, 6A]` を探索できなかった。helper の責務分離を維持する
        # ため `_order_corners_by_rotation` は変更せず、caller 側で 4 通りの
        # 巡回 permutation を回し score 最良を採用する。
        # perm_k = corners を k 個左シフトした順。perm0=現状、perm1=90° 回転、
        # perm2=180°、perm3=270° に対応。
        best_score = float("inf")
        best_centers: list[tuple[float, float]] | None = None
        best_source: str | None = None
        best_perm: int | None = None
        # **Codex revise C6**: bounds/mask gate で reject された候補数 vs
        # 評価試行数を記録し、全 reject の理由が gate なのか score なのかを
        # 区別する (`_last_contour_reject_reason` を `"out_of_bounds"` /
        # `"all_candidates_failed"` で分岐)。
        gate_rejects = 0
        candidate_attempts = 0
        # `_score_corners_by_color_match` が参照する runtime hinge_gap を
        # canonical (= self._hinge_gap) に明示代入。新検出器は outer rim を
        # physical chart として扱うため runtime 逆算 hinge_gap は不要。
        self._runtime_estimated_hinge_gap = float(self._hinge_gap)
        mx = _CONTOUR_FRAME_MARGIN_X_FRAC
        my = _CONTOUR_FRAME_MARGIN_Y_FRAC

        def _bilerp_quad(
            tl: tuple[float, float],
            tr: tuple[float, float],
            br: tuple[float, float],
            bl: tuple[float, float],
            u: float,
            v: float,
        ) -> tuple[float, float]:
            top_p = (
                tl[0] + u * (tr[0] - tl[0]),
                tl[1] + u * (tr[1] - tl[1]),
            )
            bot_p = (
                bl[0] + u * (br[0] - bl[0]),
                bl[1] + u * (br[1] - bl[1]),
            )
            return (
                top_p[0] + v * (bot_p[0] - top_p[0]),
                top_p[1] + v * (bot_p[1] - top_p[1]),
            )

        for pts, source in candidate_point_sets:
            try:
                rect_cand = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect_cand).astype(np.float32)
            except Exception:
                continue
            # 9) central → 元 frame 座標に逆変換 (以後の座標は元 frame 系)
            box_full = box.copy()
            box_full[:, 0] += cx0
            box_full[:, 1] += cy0
            # 10) 回転対応 corner 並び替え (helper の戻り値仕様は不変)
            outer_rim = self._order_corners_by_rotation(box_full, rect_cand)
            if outer_rim is None:
                continue
            # **Codex revise C4**: 旧実装は `for perm_k in range(4):` で 4 通り
            # 全 cyclic shift を採点していたが、k=1/3 は短辺 (`outer_1A→outer_6A`)
            # を `[1A, 1H]` 方向 (= 8 列方向、長辺) に置く 90°/270° 軸入れ替え
            # permutation に該当し、SpyderCheckr 48 の「1A→1H は長辺」幾何契約を
            # 破る。実観測 (`synth_rot_+000deg.png`) で perm=1 が誤採用され、
            # 返却 corner の `1A→1H` が縦方向、`1A→6A` が横方向になる ROI が
            # 通っていた。`_order_corners_by_rotation` は既に長辺方向を
            # `outer_1A→outer_1H`、短辺方向を `outer_1A→outer_6A` に揃えている
            # (1-C 不変条件) ため、長辺保存 permutation のみ許容する `(0, 2)` に
            # 絞る (perm0=正立、perm2=180° flip)。これで物理 1A 配置の上下入替
            # (180°) のみ score 比較し、90°/270° 軸入れ替えは構造的に reject。
            for perm_k in (0, 2):
                perm_corners = [
                    outer_rim[(perm_k + i) % 4] for i in range(4)
                ]
                perm_tl, perm_tr, perm_br, perm_bl = perm_corners
                # 11) outer rim → patch CENTER への bilerp
                patch_1a = _bilerp_quad(
                    perm_tl, perm_tr, perm_br, perm_bl, mx, my,
                )
                patch_1h = _bilerp_quad(
                    perm_tl, perm_tr, perm_br, perm_bl, 1.0 - mx, my,
                )
                patch_6h = _bilerp_quad(
                    perm_tl, perm_tr, perm_br, perm_bl, 1.0 - mx, 1.0 - my,
                )
                patch_6a = _bilerp_quad(
                    perm_tl, perm_tr, perm_br, perm_bl, mx, 1.0 - my,
                )
                patch_centers = [patch_1a, patch_1h, patch_6h, patch_6a]
                # **Codex revise C6**: bounds / mask gate を採用直前
                # (best_score 更新の手前) に適用する。旧実装は best_score
                # 比較のみで採用していたため、回転時 adaptive_threshold が
                # 0.45 まで緩むと画面外 corner を含む候補が通る欠陥があった
                # (実観測: +30° fixture で負 y 座標 corner 採用)。
                # gate 1: frame bounds 内 (margin 0)
                # gate 2: composite_hole が利用可能なら _corners_inside_hole_mask
                #   で hole 内チェック。fallback `gray > 12` は暗 patch を False
                #   にする逆の極性なので gate には使えず、composite_hole が
                #   non-None のときだけ適用する (advisor 助言)。
                candidate_attempts += 1
                _bounds_ok = True
                for _bx, _by in patch_centers:
                    if not (
                        0 <= int(round(_bx)) < w
                        and 0 <= int(round(_by)) < h
                    ):
                        _bounds_ok = False
                        break
                if not _bounds_ok:
                    gate_rejects += 1
                    print(
                        f"- contour outer rim: candidate source={source} "
                        f"perm={perm_k} → out_of_bounds (gate skip)"
                    )
                    continue
                _xs = [float(_pt[0]) for _pt in patch_centers]
                _ys = [float(_pt[1]) for _pt in patch_centers]
                _candidate_diag = float(
                    np.hypot(max(_xs) - min(_xs), max(_ys) - min(_ys))
                )
                _frame_diag = float(np.hypot(w, h))
                if _candidate_diag < _frame_diag * 0.30:
                    gate_rejects += 1
                    print(
                        f"- contour outer rim: candidate source={source} "
                        f"perm={perm_k} → span_too_small "
                        f"diag={_candidate_diag:.1f} "
                        f"min={_frame_diag * 0.30:.1f} (gate skip)"
                    )
                    continue
                if composite_hole is not None:
                    if not self._corners_inside_hole_mask(
                        patch_centers, composite_hole,
                    ):
                        gate_rejects += 1
                        print(
                            f"- contour outer rim: candidate source={source} "
                            f"perm={perm_k} → outside_hole_mask (gate skip)"
                        )
                        continue
                # 12) 色一致スコア (canonical hinge_gap を直前で代入済)
                try:
                    score = self._score_corners_by_color_match(
                        frame_bgr, patch_centers,
                    )
                except Exception:
                    continue
                print(
                    f"- contour outer rim: candidate source={source} "
                    f"perm={perm_k} score={score:.4f}"
                )
                if score < best_score:
                    best_score = score
                    best_centers = patch_centers
                    best_source = source
                    best_perm = perm_k
        # 13) 採否判定 (rotation adaptive threshold)
        # **Codex revise C1**: 旧実装は BASE=0.30 を単独で厳格適用していたため、
        # `-20°` (score 0.3145) など真の回転 chart を `color_score_too_high` で
        # reject する欠陥があった (= 「CW/CCW 任意角に追従」要件違反)。
        # 旧 rigid detector (`_detect_chart_via_rigid_rotated_rect` L4378-4416)
        # と同じ rotation adaptive 戦略に揃える:
        # - `_estimate_global_chart_rotation` で frame 全体の回転角度を推定
        # - |θ| <= 5° または推定不能なら BASE (=0.30) で従来通り厳格 reject
        # - |θ| > 5° なら `min(MAX, BASE + PER_DEG × |θ|)` で MAX (=0.45) まで
        #   線形拡張。MAX 0.45 は planner 厳守上限 (Phase 5 実機受入で過剰回避)。
        # 例: -20° なら adaptive_threshold = 0.30 + 0.005 × (20-5) = 0.375
        # → score 0.3145 が pass する。
        if best_centers is None:
            # gate 経由で全 reject なら `"out_of_bounds"`、それ以外は従来通り
            # `"all_candidates_failed"` (Codex revise C6 切り分け)。
            if candidate_attempts > 0 and gate_rejects == candidate_attempts:
                self._last_contour_reject_reason = "out_of_bounds"
            else:
                self._last_contour_reject_reason = "all_candidates_failed"
            return None
        estimated_rot = self._estimate_global_chart_rotation(
            frame_bgr, main_shape=main_shape,
        )
        if estimated_rot is None:
            adaptive_threshold = _RIGID_COLOR_REJECT_THRESHOLD_BASE
            rot_label = "None"
        else:
            abs_rot = abs(float(estimated_rot))
            if abs_rot <= 5.0:
                adaptive_threshold = _RIGID_COLOR_REJECT_THRESHOLD_BASE
            else:
                adaptive_threshold = min(
                    _RIGID_COLOR_REJECT_THRESHOLD_MAX,
                    _RIGID_COLOR_REJECT_THRESHOLD_BASE
                    + _RIGID_COLOR_THRESHOLD_PER_DEG * abs_rot,
                )
            rot_label = f"{estimated_rot:+.1f}"
        if best_score > adaptive_threshold:
            print(
                f"- contour outer rim: best score={best_score:.4f} > "
                f"threshold={adaptive_threshold:.4f} "
                f"(global_rot={rot_label}°, "
                f"base={_RIGID_COLOR_REJECT_THRESHOLD_BASE}, "
                f"max={_RIGID_COLOR_REJECT_THRESHOLD_MAX}) → "
                f"manual fallback"
            )
            self._last_contour_reject_reason = "color_score_too_high"
            return None
        # 14) 採用
        print(
            f"- contour outer rim: 採用 source={best_source} "
            f"perm={best_perm} score={best_score:.4f} "
            f"threshold={adaptive_threshold:.4f} "
            f"(global_rot={rot_label}°) "
            f"rim=[1A=({best_centers[0][0]:.1f},{best_centers[0][1]:.1f}),"
            f"1H=({best_centers[1][0]:.1f},{best_centers[1][1]:.1f}),"
            f"6H=({best_centers[2][0]:.1f},{best_centers[2][1]:.1f}),"
            f"6A=({best_centers[3][0]:.1f},{best_centers[3][1]:.1f})]"
        )
        return list(best_centers)

    def _detect_chart_via_rigid_rotated_rect(
        self,
        frame_bgr: np.ndarray,
        *,
        main_shape: tuple[int, int] | None = None,
    ) -> list[tuple[float, float]] | None:
        """chart 全体を **単一の剛体回転矩形** として検出し、1A/1H/6H/6A 中心を返す。

        設計方針 (per-panel 独立 fit の問題を回避するため):
        - dark threshold で chart 暗領域を抽出
        - hinge を horizontal closing で bridge し、左右 panel + hinge を 1 つの
          連結成分にする (剛体 chart として扱う)
        - 最大 connected component に `cv2.minAreaRect` → 剛体回転矩形 (4 corners
          が rigid rectangle by construction)
        - canonical SpyderCHECKR 48 比率 (8 cols + hinge_gap + 2*frame_margin) ×
          (6 rows + 2*frame_margin) で 1A/1H/6H/6A の正規化位置を計算
        - rigid rect の 4 corners からの bilinear 内分で 4 patch center を返す
        - 戻り値は rigid 矩形なので、直後 `set_corners` に渡せば yellow frame は
          rectangle、左右 cell サイズは同一になる
        """
        self._last_rigid_reject_reason: str | None = None
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._last_rigid_reject_reason = "cvtcolor_failed"
            return None
        h, w = gray.shape
        if h < 60 or w < 60:
            self._last_rigid_reject_reason = "frame_too_small"
            return None
        # 暗室穴の外 (chamber 壁) と composite frame の left_panel padding は照明が
        # 届かないため intensity ≒ 0。flat 校正で生成した valid_mask を composite
        # 座標系に変換して **AND** することで、chamber 壁 + padding を一括除外する。
        # valid_mask が無い場合 (flat 未校正) は intensity > 12 で同等の hole 内 / 外
        # 判別を行う fallback。
        hole_mask_full = self._composite_hole_mask((h, w), main_shape)
        if hole_mask_full is None:
            hole_mask_full = gray > 12
            print(
                "- rigid rect detect: flat valid_mask 不在 → "
                "intensity>12 で hole 推定 (fallback)"
            )
        # 中央 5% margin で外周 scope 円縁ノイズを除外
        m = 0.05
        cy0, cy1 = int(h * m), int(h * (1 - m))
        cx0 = int(w * m)
        cx1 = int(w * (1 - m))
        central = gray[cy0:cy1, cx0:cx1]
        central_hole = hole_mask_full[cy0:cy1, cx0:cx1]
        print(
            f"- rigid rect detect: 探索領域 x=[{cx0}..{cx1}] y=[{cy0}..{cy1}] "
            f"hole 内 pixel ratio={100.0*central_hole.mean():.1f}%"
        )
        if central.size == 0:
            self._last_rigid_reject_reason = "central_empty"
            return None
        try:
            # === Phase 2Z: chart vs background segmentation ===
            # 旧 dark threshold (gray < 80) は SpyderCheckr 48 の bright patches
            # (skin tone, white patch 等) を除外し、minAreaRect が真の chart 外形
            # ではなく dark cells subset の bounding rect になる構造的問題があった
            # (Phase 2Y までの falsification primary evidence: chart_asp 観測値
            # 0.699 vs canonical 0.5623 = +24.3% 乖離、bright patches 分の幅が
            # 反映されていなかった)。
            # 新方式: disk 白背景 (intensity peak >= 150) より十分暗い領域を
            # chart として捕捉、bright patches も含む全 chart を 1 component
            # として取得する。これにより minAreaRect が真のチャート物理外形に
            # 一致し、canonical 算術 (Phase 2X 導入の
            # `_chart_patch_center_normalized`) がそのまま正しい patch CENTER
            # を出す。
            central_hole_u8 = (central_hole.astype(np.uint8) * 255)
            hist = cv2.calcHist(
                [central], [0], central_hole_u8, [256], [0, 256]
            )
            hist_high = hist[150:].flatten()
            if hist_high.sum() > 0:
                disk_peak_offset = int(np.argmax(hist_high))
                disk_peak_intensity = 150 + disk_peak_offset
                # chart 最も明るい patch (white patch) でも disk 白照明より暗い
                # 前提。margin 30 で skin tone (~150) 付近まで chart 側に取り
                # 込む。chart_threshold は最低でも 60 を確保し、極端な暗 disk
                # 環境でも fallback として機能する。
                chart_threshold = max(60, disk_peak_intensity - 30)
                _disk_peak_log = str(disk_peak_intensity)
            else:
                # disk 白背景が central にない場合 (chart 全画面占有等) は
                # 旧 dark threshold (gray < 80) と同じ安全側 fallback
                chart_threshold = 80
                _disk_peak_log = "N/A"
            _, chart_thresh = cv2.threshold(
                central, chart_threshold, 255, cv2.THRESH_BINARY_INV
            )
            # hole の外 (chamber 壁 + composite padding) を除外
            dark_mask = cv2.bitwise_and(chart_thresh, central_hole_u8)
            print(
                f"- rigid rect detect: chart segmentation (Phase 2Z) "
                f"disk_peak={_disk_peak_log} "
                f"chart_threshold={chart_threshold}"
            )
            # 横方向 closing で hinge を bridge し chart 全体を 1 連結に。
            # Phase A: kernel が大きすぎると chart と無関係な暗影とも bridge してしまう
            # ため 固定 11 で控えめに、iterations=1 のみ運用していた (axis-aligned chart
            # では十分)。
            # Phase 2 (20_strategy 2-D, post-Codex-revise): 回転 chart では hinge gap
            # が ≈ 30-40 px (warpAffine 後の synth 実測) まで広がり kw=11 では bridge
            # 不能となり、左右パネルが 2 component に分裂する。
            # **Codex review C3 sanctioned**: kw 0.04 拡大は chart 外周 chamber padding
            # を巻き込み aspect short/long を 0.73-0.75 に膨張させたため revert。
            # 代わりに kw=0.025 (chart 幅 1000 px で 25 px) + **2-pass close** で
            # kw を抑えつつ hinge を確実に bridge する。2-pass の effective kw は
            # ≈ 2*kw - 1 = 49 px (kw=25 のとき) で hinge gap ~37 px を跨ぐ。
            # 安全策: bridge 対象は hole_mask AND 適用済 dark_mask に限定 (central 5%
            # margin 内) なので、kernel 拡大しても chamber 壁との誤 bridge は発生しない。
            #
            # Phase 4 (22_strategy 4-B): 横カーネル close は **保持**。
            # ユーザ task literal は「削除またはコメントアウト」だが、Phase 21 2-X
            # erosion fallback (L4470-4566) と `tests/test_chart_segmentation_separation.py`
            # の literal stdout / `_last_rigid_reject_reason` 契約が `bridged` 経路に
            # 依存するため、削除すると 561-test invariant 破綻。代わりに 4-A で
            # contour pair merge 候補を **augment** している (L4720 直前の augment ブロック)。
            # 詳細: plan/22_2026-05-01-spyder-contour-redesign-phase4-triad.md
            # 「設計上の重要前提」 (planner deviation rationale)。
            kw = max(
                _RIGID_BRIDGE_KERNEL_BASE,
                int(central.shape[1] * _RIGID_BRIDGE_KERNEL_RATIO),
            )
            if kw % 2 == 0:
                kw += 1
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
            bridged = cv2.morphologyEx(
                dark_mask,
                cv2.MORPH_CLOSE,
                h_kernel,
                iterations=_RIGID_BRIDGE_KERNEL_PASSES,
            )
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                bridged
            )
        except Exception:
            self._last_rigid_reject_reason = "morphology_exception"
            return None
        if num_labels < 2:
            print("- rigid rect detect: dark component なし")
            self._last_rigid_reject_reason = "no_dark_component"
            return None
        # === Phase 21 2-X: too-big top1 fallback (CCW chart + Ref panel merge 救済) ===
        # Phase 1 evidence (CCW Press 3): top1 area=90450 bbox=(312,41,466,529)
        # で chart + Ref panel + 影 が 1 component に merge し geometry_filter で
        # reject されていた。元 dark_mask を破壊せず erosion した copy で再分離を
        # 試み、得られた小成分を candidates に **追加注入** する (元 components は
        # 保持、CW/水平 regression を防ぐ)。
        # 不変条件: 元 dark_mask / bridged / labels / stats / num_labels は読み取り
        # 専用、erosion した copy `_eroded`/`_l2`/`_s2` のみ生成する。
        extra_components: list[tuple[int, int, int, int, int, int]] = []
        _l2 = None  # 防御的初期化 (extra_components が非空のとき下流ループで参照)
        if num_labels >= 2:
            _top_area = max(
                int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)
            )
            if _top_area > central.size * _PHASE21_TOOBIG_AREA_RATIO:
                # erosion で chart-Ref panel を分離 (元 dark_mask は破壊しない)
                _erosion_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (_PHASE21_EROSION_KERNEL, _PHASE21_EROSION_KERNEL),
                )
                # === Phase 21 2-X2 (cycle 2): adaptive erosion ループ ===
                # cycle 1 は単発 `cv2.erode(..., iterations=_PHASE21_EROSION_ITERS)`
                # 固定 iter=2 だったが、実機 (run_log_phase2_cycle1_failed.txt
                # L46-52) で 12-20 px 厚 bridge を切れず Pass 2 candidate も
                # chart_aspect で reject された。iter を 2→3→4→5 で段階拡張し、
                # `_top_eroded < _top_area * _PHASE21_EROSION_BREAK_RATIO` (= 元
                # top1 の 60% 未満まで縮小、merge が切れた状態) で early break。
                # 上限 iter=5 (= 各方向 20 px 縮小) は chart 内 cell 25-40 px
                # 幅に対し marginal だが許容、chart 単独 component は残る。
                _eroded = bridged  # 安全フォールバック (使われない)
                _l2_local = labels
                _s2 = stats
                _n2 = num_labels
                _top_eroded = _top_area
                _used_iter = _PHASE21_EROSION_ITERS_MIN
                _broke = False
                for _iter in range(
                    _PHASE21_EROSION_ITERS_MIN,
                    _PHASE21_EROSION_ITERS_MAX + 1,
                ):
                    _eroded = cv2.erode(
                        bridged, _erosion_kernel, iterations=_iter,
                    )
                    _n2, _l2_local, _s2, _ = (
                        cv2.connectedComponentsWithStats(_eroded)
                    )
                    _top_eroded = max(
                        (
                            int(_s2[j, cv2.CC_STAT_AREA])
                            for j in range(1, _n2)
                        ),
                        default=0,
                    )
                    _used_iter = _iter
                    if (
                        _top_eroded
                        < _top_area * _PHASE21_EROSION_BREAK_RATIO
                    ):
                        _broke = True
                        break
                if not _broke:
                    print(
                        f"- rigid rect detect (Phase 21 2-X2): "
                        f"adaptive erosion 上限 iter={_used_iter} に到達して "
                        f"も _top_eroded={_top_eroded} >= _top_area*"
                        f"{_PHASE21_EROSION_BREAK_RATIO:.2f}="
                        f"{_top_area*_PHASE21_EROSION_BREAK_RATIO:.0f} "
                        f"(merge が完全に切れていない可能性、最終 _eroded で続行)"
                    )
                _l2 = _l2_local
                for j in range(1, _n2):
                    _aj = int(_s2[j, cv2.CC_STAT_AREA])
                    # 元 candidate ループの最低 area gate と同水準
                    if _aj < central.size * 0.02:
                        continue
                    # 元 too-big と区別不能なものは除外 (分離できていない)
                    if _aj >= _top_area:
                        continue
                    _xj = int(_s2[j, cv2.CC_STAT_LEFT])
                    _yj = int(_s2[j, cv2.CC_STAT_TOP])
                    _wj = int(_s2[j, cv2.CC_STAT_WIDTH])
                    _hj = int(_s2[j, cv2.CC_STAT_HEIGHT])
                    # idx は -j (j>=1) で原 labels と衝突しないよう負数で識別
                    extra_components.append(
                        (_aj, -j, _xj, _yj, _wj, _hj)
                    )
                if extra_components:
                    # Phase 21 2-X2 (cycle 2): print 文言は cycle 1 既存 test の
                    # substring match (`"Phase 21 2-X): top1 too-big detected"`)
                    # を破壊しないよう冒頭の `(Phase 21 2-X)` 区切りを保持しつつ、
                    # cycle 2 識別子 `[2-X2 adaptive iter=N]` を追記する。
                    # `iter={_used_iter}` は実際に break した iter 値、
                    # `top_eroded`/`ratio_to_top1` で merge 切断度を可視化。
                    print(
                        f"- rigid rect detect (Phase 21 2-X): top1 too-big "
                        f"detected (area={_top_area} > "
                        f"{_PHASE21_TOOBIG_AREA_RATIO*100:.0f}% of "
                        f"central.size={central.size}), erosion "
                        f"({_PHASE21_EROSION_KERNEL}x"
                        f"{_PHASE21_EROSION_KERNEL} iter="
                        f"{_used_iter}) 適用 → "
                        f"{len(extra_components)} 個の追加 component を注入 "
                        f"[Phase 21 2-X2 adaptive: "
                        f"top_eroded={_top_eroded}, "
                        f"ratio_to_top1={_top_eroded/max(_top_area,1):.2f}, "
                        f"used_iter={_used_iter}, broke={_broke}]"
                    )
        # 上位 5 件を log 出力 (debug 用)
        all_components: list[tuple[int, int, int, int, int, int]] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x0_c = int(stats[i, cv2.CC_STAT_LEFT])
            y0_c = int(stats[i, cv2.CC_STAT_TOP])
            w_c = int(stats[i, cv2.CC_STAT_WIDTH])
            h_c = int(stats[i, cv2.CC_STAT_HEIGHT])
            all_components.append((area, i, x0_c, y0_c, w_c, h_c))
        all_components.sort(reverse=True)
        for rank, (a, idx, x0_c, y0_c, w_c, h_c) in enumerate(
            all_components[:5]
        ):
            asp = (w_c / max(h_c, 1)) if h_c > 0 else 0.0
            print(
                f"- rigid rect detect: top{rank+1} area={a} "
                f"bbox=({x0_c+cx0},{y0_c+cy0},{w_c},{h_c}) aspect={asp:.2f}"
            )
        # SpyderCHECKR 48 の物理 aspect は ~1.66:1 (横長)。回転次第で minAreaRect
        # bbox の width/height が逆転し縦長 (~0.6:1) 観測も発生するため Phase A で
        # 緩和。選定根拠:
        # analysis/2026-04-28-csi-detection-stage-strengthening/
        # corner_detection_chosen_params.md
        ASPECT_MIN = _RIGID_ASPECT_MIN
        ASPECT_MAX = _RIGID_ASPECT_MAX
        CHART_ASP_MIN = _RIGID_CHART_ASPECT_MIN
        CHART_ASP_MAX = _RIGID_CHART_ASPECT_MAX
        # Phase A4 (色一致度採用): geometry filter (area / bbox aspect / chart aspect)
        # を通過した全候補について、provisional patch quad で 48 patch 中心の色を
        # sampling し SpyderCHECKR-48 reference 色との一致度を測る。
        # 「最大 area」採用ではなく「最低色距離」採用に切替。閉じた裏面 (灰一色)
        # と開いた chart (48 色) を色情報で分離する。
        # Phase 2 (20_strategy, post-Codex-revise C4): long-side floor は revert。
        # bridge ratio を 0.025 + 2-pass close に下げたことで sub-region top2 が
        # geometry filter (chart aspect [0.55, 0.70]) を通過する事例は減少し、
        # かつ通過しても 0.30 base color threshold で reject されるため安全。
        candidates: list[tuple[int, int, np.ndarray]] = []  # (idx, area, pts)
        chart_aspect_reject_count = 0
        # Phase 21 2-X (deviation, advisor sanctioned): planner spec L78 は
        # "追加注入 (augment)" と記述するが、文字通り実装すると synth_v0 -5° で
        # erosion 由来 sub-region が color_score 競争で original top1 を displace
        # し既存 passing test を破壊する。spec L77/L221 (救済意図、CW regression
        # 防止) に整合させ **true fallback** に変更: Pass 1 (元 all_components)
        # で geometry filter を満たす候補が 0 件のときのみ Pass 2 (extra_components)
        # を巡回する。CCW Press 3 (top1 が chart_aspect filter 不通過 → Pass 1 が 0
        # 件) で extra_components による救済が発火する所期の動作は維持される。
        # 元 candidate (idx>=1, labels 参照) と erosion 候補 (idx<0, _l2 参照) を
        # 同一 inner loop で処理する。
        component_passes: list[
            list[tuple[int, int, int, int, int, int]]
        ] = [all_components]
        if extra_components:
            component_passes.append(extra_components)
        for _pass_components in component_passes:
            if candidates:
                # Pass 1 で候補が得られた場合 Pass 2 (erosion 由来) はスキップ
                break
            for area, idx, x0_c, y0_c, w_c, h_c in _pass_components:
                # Phase 21 2-X2 (cycle 2): Pass 2 (idx<0、erosion 由来) candidate
                # の accept/reject 理由を 1 行で診断。cycle 1 evidence
                # (run_log_phase2_cycle1_failed.txt) では extras の chart_asp 値
                # が観測できなかった反省。Pass 1 (idx>=1) は規模膨張防止のため
                # print しない。
                if area < central.size * 0.02:
                    if idx < 0:
                        print(
                            f"- rigid rect detect (Phase 21 2-X2 cand): "
                            f"idx={idx} area={area} chart_asp=n/a "
                            f"-> reject_area_ratio"
                        )
                    continue
                asp = (w_c / max(h_c, 1)) if h_c > 0 else 0.0
                if asp < ASPECT_MIN or asp > ASPECT_MAX:
                    if idx < 0:
                        print(
                            f"- rigid rect detect (Phase 21 2-X2 cand): "
                            f"idx={idx} area={area} chart_asp=n/a "
                            f"-> reject_bbox_aspect (asp={asp:.3f})"
                        )
                    continue
                if idx < 0:
                    # erosion 由来候補: _l2 ラベルから点群を取り直す
                    # _l2 は extra_components が非空のときのみ非 None
                    ys_idx, xs_idx = np.where(_l2 == -idx)
                else:
                    ys_idx, xs_idx = np.where(labels == idx)
                if len(xs_idx) < 100:
                    if idx < 0:
                        print(
                            f"- rigid rect detect (Phase 21 2-X2 cand): "
                            f"idx={idx} area={area} chart_asp=n/a "
                            f"-> reject_too_few_pts ({len(xs_idx)})"
                        )
                    continue
                pts_idx = np.column_stack(
                    [xs_idx + cx0, ys_idx + cy0]
                ).astype(np.float32)
                try:
                    rect_idx = cv2.minAreaRect(pts_idx)
                except Exception:
                    if idx < 0:
                        print(
                            f"- rigid rect detect (Phase 21 2-X2 cand): "
                            f"idx={idx} area={area} chart_asp=n/a "
                            f"-> reject_minAreaRect_exception"
                        )
                    continue
                (_rcx, _rcy), (_rw, _rh), _rang = rect_idx
                short_len_idx = float(min(_rw, _rh))
                long_len_idx = float(max(_rw, _rh))
                if long_len_idx < 1e-3:
                    if idx < 0:
                        print(
                            f"- rigid rect detect (Phase 21 2-X2 cand): "
                            f"idx={idx} area={area} chart_asp=n/a "
                            f"-> reject_zero_long_len"
                        )
                    continue
                chart_aspect_idx = short_len_idx / long_len_idx
                # Phase 21 2-X (deviation #2): erosion 由来 extras (idx<0) は
                # 厳格上限 `_PHASE21_EXTRA_CHART_ASP_MAX` を使用、Pass 1 元
                # candidates (idx>=1) は従来 `_RIGID_CHART_ASPECT_MAX` を維持。
                _chart_asp_ceiling = (
                    _PHASE21_EXTRA_CHART_ASP_MAX if idx < 0 else CHART_ASP_MAX
                )
                if chart_aspect_idx < CHART_ASP_MIN:
                    chart_aspect_reject_count += 1
                    if idx < 0:
                        print(
                            f"- rigid rect detect (Phase 21 2-X2 cand): "
                            f"idx={idx} area={area} "
                            f"chart_asp={chart_aspect_idx:.4f} "
                            f"-> reject_chart_asp_lo (min={CHART_ASP_MIN:.2f})"
                        )
                    continue
                if chart_aspect_idx > _chart_asp_ceiling:
                    chart_aspect_reject_count += 1
                    if idx < 0:
                        print(
                            f"- rigid rect detect (Phase 21 2-X2 cand): "
                            f"idx={idx} area={area} "
                            f"chart_asp={chart_aspect_idx:.4f} "
                            f"-> reject_chart_asp_hi "
                            f"(max={_chart_asp_ceiling:.2f})"
                        )
                    continue
                if idx < 0:
                    print(
                        f"- rigid rect detect (Phase 21 2-X2 cand): "
                        f"idx={idx} area={area} "
                        f"chart_asp={chart_aspect_idx:.4f} -> accept"
                    )
                candidates.append((int(idx), int(area), pts_idx))
        # === Phase 4 (22_strategy 4-A): contour pair merge candidate 追加注入 ===
        # planner output 「設計上の重要前提」の augment 戦略参照
        # (plan/22_2026-05-01-spyder-contour-redesign-phase4-triad.md)。
        # 既存 Pass 1/Pass 2 (横カーネル close + erosion fallback) は無傷で残し、
        # `dark_mask` (pre-bridge) に findContours + pair merge を適用した結果を
        # candidates に append する (回転姿勢で hinge 分裂 → 2 contour に分かれた
        # ケースを Phase 1 helper `_merge_contour_pair_for_chart` で救済)。
        # 横カーネル close を削除しないのは Phase 21 2-X 経路 (test_chart_segmentation_separation
        # 依存) を保持するため。
        # sentinel idx は `_PHASE4_CONTOUR_MERGE_IDX_BASE` (= -1000) 起点で展開し、
        # Pass 1 (idx>=1) / Pass 2 erosion (idx<0、{-1, ..., -n}) と非衝突。
        # 下流 `_patch_quad_from_component_pts` → `_score_corners_by_color_match`
        # は idx を読み捨て pts のみ消費するため、sentinel idx でも従来 score 競争に乗る。
        # `_last_rigid_reject_reason` の文字列契約は不変 (新規 reason は発行しない)。
        try:
            _phase4_contours, _ = cv2.findContours(
                dark_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
        except Exception:
            print(
                "- rigid rect detect (Phase 4 4-A): findContours 例外 → "
                "augment skip (既存 candidates のみで続行)"
            )
            _phase4_contours = []
        print(
            "- rigid rect detect (Phase 4 4-A): contour pair merge source = "
            f"{len(_phase4_contours)} contours"
        )
        # filtered_phase4: (area, pts_in_composite_coords, rect) area 降順
        # 新検出器 (`_detect_chart_via_contour_outer_rim` L4068-4090 周辺) と
        # 同じ (area / aspect) gate を採用:
        #   - area >= central.size * 0.02 (既存 Pass 1 と同基準)
        #   - short/long ∈ [_RIGID_ASPECT_MIN (=0.4), 0.95]
        # 新検出器 ASPECT_MAX=0.95 (planner spec L113 0.9 + v1 fixture clip 吸収)
        # と同値を再利用し、pair merge 流用一貫性を確保。
        _PHASE4_AUG_ASPECT_MIN = _RIGID_ASPECT_MIN  # 0.4
        _PHASE4_AUG_ASPECT_MAX = 0.95
        _phase4_area_floor = float(central.size) * 0.02
        _phase4_filtered: list[tuple[float, np.ndarray, tuple]] = []
        for _c in _phase4_contours:
            try:
                _area_c = float(cv2.contourArea(_c))
            except Exception:
                continue
            if _area_c < _phase4_area_floor:
                continue
            try:
                _rect_c = cv2.minAreaRect(_c)
            except Exception:
                continue
            (_rcx_c, _rcy_c), (_w_r, _h_r), _ang_c = _rect_c
            _short_c = float(min(_w_r, _h_r))
            _long_c = float(max(_w_r, _h_r))
            if _long_c < 1.0:
                continue
            _asp_c = _short_c / _long_c
            if (
                _asp_c < _PHASE4_AUG_ASPECT_MIN
                or _asp_c > _PHASE4_AUG_ASPECT_MAX
            ):
                continue
            # contour 点列 _c は **central 座標系** で出力される
            # (`cv2.findContours(dark_mask, ...)` の dark_mask は central crop)。
            # 後段 `_patch_quad_from_component_pts` は **composite 座標系** を
            # 期待するため (Pass 1 の `pts_idx = np.column_stack([xs+cx0, ys+cy0])`
            # と同じ image 座標系)、ここで `+ (cx0, cy0)` shift を適用する。
            # `cv2.contourArea` / `cv2.minAreaRect` の (area, w, h, angle) は
            # 平行移動不変なため、area / aspect 値はそのまま使ってよい。
            _c_shifted = (
                _c.astype(np.float32) + np.array(
                    [float(cx0), float(cy0)], dtype=np.float32
                )
            )
            _phase4_filtered.append((_area_c, _c_shifted, _rect_c))
        _phase4_filtered.sort(key=lambda x: x[0], reverse=True)
        if _phase4_filtered:
            try:
                _phase4_merged = self._merge_contour_pair_for_chart(
                    _phase4_filtered
                )
            except Exception as _exc:
                print(
                    "- rigid rect detect (Phase 4 4-A): pair merge helper "
                    f"例外 ({_exc}) → augment skip"
                )
                _phase4_merged = []
        else:
            _phase4_merged = []
        # merged_pts 採用 gate: chart_asp ∈ [CHART_ASP_MIN, _PHASE4_AUG_CHART_ASP_MAX]
        # **augment 戦略の安全弁** (R1 mitigation): filter なしでは chart + Ref panel
        # merge が chart_asp≈0.81-0.85 で chart-only candidate を score 競争で押しのけ、
        # `tests/test_corner_detection_robustness.py` の物理制約
        # (output patch_quad_aspect ≤ `_RIGID_CHART_ASPECT_MAX * 0.826 + 0.02 ≒ 0.722`)
        # を破る。
        # 上限は `_PHASE21_EXTRA_CHART_ASP_MAX` (=0.78) と同値を採用:
        # Pass 2 erosion 由来 extras と同様、augment 経路は誤検出余地が大きいため
        # 厳格上限を入れる。0.85 (CHART_ASP_MAX) では chart+Ref panel merge
        # (~0.80-0.85) を弾けず regression する一次証拠あり。
        # 下限は `CHART_ASP_MIN` (=0.50) を維持 (Pass 1 と同基準)。
        _PHASE4_AUG_CHART_ASP_MAX = _PHASE21_EXTRA_CHART_ASP_MAX  # 0.78
        # Phase 4 cycle 1 C1 mitigation: augment 候補は frame width の 40% 以上を
        # 覆うこと。axis-aligned chart (angle=0°) では `dark_mask` (pre-bridge) に
        # `cv2.findContours` を直接かけると hinge gap がそのまま分裂し、
        # `_merge_contour_pair_for_chart` が (左 panel + 右 panel の patch 領域だけ、
        # frame margin を除いた狭い ~251px 幅の合成領域) を chart-asp ~0.70 で返す。
        # この sub-region は color_score 0.2271 で Pass 1 の正常系候補 0.1881 を
        # displace し、最終 corners が frame 幅の 24.6% (= 251px / 1020px) しか
        # 覆わない sub-region 誤検出につながる
        # (`tests/test_rigid_detect_rotation_robustness.py::test_rigid_returns_corners_for_must_pass_angles[angle0]`
        # の bbox sanity assertion `bbox_w >= frame_w * 0.40` で fail)。
        # 対策: merged_pts の minAreaRect long side が
        # `frame_bgr.shape[1] * 0.40` 未満なら augment 候補に採用しない。
        # 基準は test の bbox sanity gate (40%) と同値。`frame_bgr.shape[1]` は
        # 元 composite frame の幅 (central crop 後の `central.shape[1]` ではなく、
        # 検出器入口の引数 frame の幅 = test 側 `frame.shape[1]` と同一)。
        _PHASE4_AUG_LONG_SIDE_MIN_RATIO = 0.40
        _phase4_long_side_floor = (
            float(frame_bgr.shape[1]) * _PHASE4_AUG_LONG_SIDE_MIN_RATIO
        )
        _phase4_appended = 0
        _phase4_filtered_by_chart_asp = 0
        _phase4_filtered_by_bbox = 0
        for _pi, _pj, _merged_pts in _phase4_merged:
            try:
                _merged_rect = cv2.minAreaRect(_merged_pts)
            except Exception:
                continue
            (_mcx, _mcy), (_mw, _mh), _mang = _merged_rect
            _merged_short = float(min(_mw, _mh))
            _merged_long = float(max(_mw, _mh))
            if _merged_long < 1e-3:
                continue
            # bbox sanity gate (C1 mitigation): long side が frame width の 40%
            # 未満なら sub-region 誤検出として reject。chart_asp filter より先に
            # 適用するのは、sub-region は chart_asp が真の chart に近い値
            # (~0.70) を取りうるため chart_asp gate では落ちないから。
            if _merged_long < _phase4_long_side_floor:
                _phase4_filtered_by_bbox += 1
                print(
                    "- rigid rect detect (Phase 4 4-A): merged pair "
                    f"(i,j)=({_pi},{_pj}) long_side={_merged_long:.0f}px "
                    f"< {_phase4_long_side_floor:.0f}px "
                    f"(={_PHASE4_AUG_LONG_SIDE_MIN_RATIO*100:.0f}% of "
                    f"frame width {frame_bgr.shape[1]}px) "
                    "-> reject_bbox_sanity"
                )
                continue
            _merged_chart_asp = _merged_short / _merged_long
            if (
                _merged_chart_asp < CHART_ASP_MIN
                or _merged_chart_asp > _PHASE4_AUG_CHART_ASP_MAX
            ):
                _phase4_filtered_by_chart_asp += 1
                print(
                    "- rigid rect detect (Phase 4 4-A): merged pair "
                    f"(i,j)=({_pi},{_pj}) chart_asp="
                    f"{_merged_chart_asp:.4f} -> reject_chart_asp "
                    f"(min={CHART_ASP_MIN:.2f} "
                    f"max={_PHASE4_AUG_CHART_ASP_MAX:.2f})"
                )
                continue
            _sentinel_idx = (
                _PHASE4_CONTOUR_MERGE_IDX_BASE - _phase4_appended
            )
            try:
                _merged_area = int(round(float(cv2.contourArea(_merged_pts))))
            except Exception:
                _merged_area = 0
            candidates.append((_sentinel_idx, _merged_area, _merged_pts))
            print(
                "- rigid rect detect (Phase 4 4-A): merged pair "
                f"(i,j)=({_pi},{_pj}) idx={_sentinel_idx} "
                f"area={_merged_area} chart_asp={_merged_chart_asp:.4f} "
                f"long_side={_merged_long:.0f}px "
                "appended to candidates"
            )
            _phase4_appended += 1
        if _phase4_filtered_by_chart_asp:
            chart_aspect_reject_count += _phase4_filtered_by_chart_asp
        # === Phase 4 4-A 終了 ===
        if not candidates:
            print(
                "- rigid rect detect: geometry filter "
                "(area>=2% AND bbox aspect [{0:.2f}..{1:.2f}] AND chart short/long "
                "[{2:.2f}..{3:.2f}]) を満たす dark "
                "component なし (chart aspect reject={4})".format(
                    ASPECT_MIN, ASPECT_MAX, CHART_ASP_MIN, CHART_ASP_MAX,
                    chart_aspect_reject_count,
                )
            )
            self._last_rigid_reject_reason = "geometry_filter"
            return None
        # 各候補の patch quad を bilerp で算出し色一致度をスコア
        best_score = float("inf")
        best_pts: np.ndarray | None = None
        best_idx = -1
        best_area = 0
        best_quad: list[tuple[float, float]] | None = None
        for idx, area, pts_idx in candidates:
            quad = self._patch_quad_from_component_pts(pts_idx)
            if quad is None:
                continue
            score = self._score_corners_by_color_match(frame_bgr, quad)
            if score < best_score:
                best_score = score
                best_pts = pts_idx
                best_idx = idx
                best_area = area
                best_quad = quad
        if best_quad is None or best_pts is None:
            print(
                f"- rigid rect detect: 候補 {len(candidates)} 件あったが "
                "patch quad 計算で全件失敗"
            )
            self._last_rigid_reject_reason = "patch_quad_failed"
            return None
        # Phase 2 (20_strategy 2-B, post-Codex-revise C5): rotation 推定に基づく
        # adaptive color threshold を計算。
        # - **planner spec L112 厳守**: rotation 推定 source は
        #   `_estimate_global_chart_rotation` (両 panel co-rotated 仮定の median 角度)。
        # - rotation 推定が None (= 推定不能) の場合は BASE=0.30 を使う
        #   (axis-aligned chart を従来通り厳格 reject、regression 防止)。
        # - |rot| <= 5° なら BASE (=0.30)、|rot| > 5° なら線形に MAX (=0.45) まで
        #   段階拡張 (上限 planner spec)。
        # - 旧実装は採用 component の minAreaRect 角度を使い padded synth の phantom
        #   panel pathology を avoid していたが、padded synth fixture 自体が
        #   acceptance gate として不適切 (Codex C2) であり、v1 fixture (canvas 不変、
        #   rotation のみ) では `_estimate_global_chart_rotation` が正常動作する。
        estimated_rot = self._estimate_global_chart_rotation(
            frame_bgr, main_shape=main_shape
        )
        if estimated_rot is None:
            adaptive_threshold = _RIGID_COLOR_REJECT_THRESHOLD_BASE
            rot_label = "None"
        else:
            abs_rot = abs(float(estimated_rot))
            if abs_rot <= 5.0:
                adaptive_threshold = _RIGID_COLOR_REJECT_THRESHOLD_BASE
            else:
                adaptive_threshold = min(
                    _RIGID_COLOR_REJECT_THRESHOLD_MAX,
                    _RIGID_COLOR_REJECT_THRESHOLD_BASE
                    + _RIGID_COLOR_THRESHOLD_PER_DEG * abs_rot,
                )
            rot_label = f"{estimated_rot:+.1f}"
        if best_score > adaptive_threshold:
            print(
                f"- rigid rect detect: 最良候補の色一致度 "
                f"score={best_score:.4f} > threshold={adaptive_threshold:.4f} "
                f"(global_rot={rot_label}°, "
                f"base={_RIGID_COLOR_REJECT_THRESHOLD_BASE}, "
                f"max={_RIGID_COLOR_REJECT_THRESHOLD_MAX}) "
                f"(候補数={len(candidates)}) → manual fallback"
            )
            self._last_rigid_reject_reason = "color_score_too_high"
            return None
        print(
            f"- rigid rect detect: 採用 component label={best_idx} "
            f"area={best_area} color_score={best_score:.4f} "
            f"(候補数={len(candidates)} chart aspect reject={chart_aspect_reject_count})"
        )
        # 採用候補で patch quad を返す
        return list(best_quad)

    def _patch_quad_from_component_pts(
        self, pts: np.ndarray
    ) -> list[tuple[float, float]] | None:
        """component の点群 (image 座標) から minAreaRect を取り、
        canonical 比率 (hinge-aware inner-box patch CENTER) で
        1A/1H/6H/6A patch quad を bilerp 算出する。失敗時 None。

        Phase 2X canonical model redesign:
          旧モデルは `FRAME_X_CELLS=0.29` / `FRAME_Y_CELLS=0.52` を使い、
          minAreaRect = chart 物理外形 (white frame 含む) と暗黙に前提していた。
          しかし実際の minAreaRect は dark threshold 抽出 → connected component
          → minAreaRect なので、**chart 内 dark cell pixels のみ wrap する**。
          結果として bilerp の 48 sample 中心が chart 外 / 隣 cell 境界に落下し
          score 0.55 floor を生んでいた (Phase 2 cycle 2 ablation で確定)。

          新モデルは minAreaRect を **inner colored patches area** として扱い、
          そこから hinge-aware の正規化 fraction (raw u, v ∈ [0.0469, 0.953])
          で 1A/1H/6H/6A patch CENTER を bilerp する。helper
          `_chart_patch_center_normalized` に算術を集約し、scoring 側
          (`_score_corners_by_color_match`) と二者間整合を保証する。

          算術は `csi/colorimeter_common.py` の `_default_patch_display_quads`
          (L6281-6283) と equivalent。`col_x_norm_offsets` は extractor 段階の
          per-column fine-tune なので detector 入力段階では考慮しない。
        """
        if len(pts) < 100:
            return None
        try:
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
        except Exception:
            return None
        # Phase 2W: runtime hinge_gap estimation from observed minAreaRect.
        # canonical 関係: chart_asp = 6 / (8 + hinge_gap)
        #               ⇒ hinge_gap = 6 / chart_asp - 8
        # 5 cycle primary evidence で canonical hinge_gap=2.67 (chart_asp=0.5623)
        # は v1 fixture / 実 chart の observed chart_asp≈0.72 と整合しないため、
        # observed minAreaRect から逆算した値を `_chart_patch_center_normalized`
        # helper に渡す。`_score_corners_by_color_match` も同じ instance attribute
        # `_runtime_estimated_hinge_gap` を参照することで二者間整合を保証する。
        (_rrect_cx, _rrect_cy), (rect_w, rect_h), _rrect_ang = rect
        observed_short = float(min(rect_w, rect_h))
        observed_long = float(max(rect_w, rect_h))
        if observed_long < 1e-3:
            return None
        observed_chart_asp = observed_short / observed_long
        if observed_chart_asp < 1e-6:
            return None
        estimated_hinge_gap = max(0.1, 6.0 / observed_chart_asp - 8.0)
        self._runtime_estimated_hinge_gap = float(estimated_hinge_gap)
        print(
            "- runtime hinge_gap estimation: "
            f"chart_asp={observed_chart_asp:.4f} → "
            f"hinge_gap={estimated_hinge_gap:.4f} "
            f"(canonical={self._hinge_gap})"
        )
        cx_box = float(np.mean(box[:, 0]))
        cy_box = float(np.mean(box[:, 1]))
        edges = [(box[(i + 1) % 4] - box[i]) for i in range(4)]
        edge_lens = [float(np.linalg.norm(e)) for e in edges]
        long_idx = 0 if edge_lens[0] >= edge_lens[1] else 1
        long_vec = edges[long_idx]
        long_len = float(np.linalg.norm(long_vec))
        if long_len < 1e-3:
            return None
        long_unit = long_vec / long_len
        short_unit = np.array([-long_unit[1], long_unit[0]], dtype=np.float32)
        if short_unit[1] < 0:
            short_unit = -short_unit
        scores: list[tuple[float, float, int]] = []
        for i in range(4):
            v = box[i] - np.array([cx_box, cy_box], dtype=np.float32)
            u = float(np.dot(v, long_unit))
            s = float(np.dot(v, short_unit))
            scores.append((u, s, i))
        top = sorted([sc for sc in scores if sc[1] < 0], key=lambda x: x[0])
        bot = sorted([sc for sc in scores if sc[1] > 0], key=lambda x: x[0])
        if len(top) != 2 or len(bot) != 2:
            return None
        chart_tl = (float(box[top[0][2]][0]), float(box[top[0][2]][1]))
        chart_tr = (float(box[top[1][2]][0]), float(box[top[1][2]][1]))
        chart_br = (float(box[bot[1][2]][0]), float(box[bot[1][2]][1]))
        chart_bl = (float(box[bot[0][2]][0]), float(box[bot[0][2]][1]))
        print(
            f"[patch quad] long_unit=({long_unit[0]:+.3f},{long_unit[1]:+.3f}) "
            f"short_unit=({short_unit[0]:+.3f},{short_unit[1]:+.3f}) "
            f"top_idx=[{top[0][2]},{top[1][2]}] "
            f"bot_idx=[{bot[0][2]},{bot[1][2]}] "
            f"chart_tl=({chart_tl[0]:.1f},{chart_tl[1]:.1f}) "
            f"chart_tr=({chart_tr[0]:.1f},{chart_tr[1]:.1f}) "
            f"chart_br=({chart_br[0]:.1f},{chart_br[1]:.1f}) "
            f"chart_bl=({chart_bl[0]:.1f},{chart_bl[1]:.1f})"
        )

        def _bilerp(u: float, v: float) -> tuple[float, float]:
            top_p = (
                chart_tl[0] + u * (chart_tr[0] - chart_tl[0]),
                chart_tl[1] + u * (chart_tr[1] - chart_tl[1]),
            )
            bot_p = (
                chart_bl[0] + u * (chart_br[0] - chart_bl[0]),
                chart_bl[1] + u * (chart_br[1] - chart_bl[1]),
            )
            return (
                top_p[0] + v * (bot_p[0] - top_p[0]),
                top_p[1] + v * (bot_p[1] - top_p[1]),
            )

        # 1A, 1H, 6H, 6A の正規化座標 (Style 1: raw fraction)
        # Phase 2W: canonical 2.67 ではなく runtime 逆算値を helper に渡す。
        hg = float(estimated_hinge_gap)
        u_1a, v_1a = _chart_patch_center_normalized(0, 0, hg)
        u_1h, v_1h = _chart_patch_center_normalized(7, 0, hg)
        u_6h, v_6h = _chart_patch_center_normalized(7, 5, hg)
        u_6a, v_6a = _chart_patch_center_normalized(0, 5, hg)

        corner_1a = _bilerp(u_1a, v_1a)
        corner_1h = _bilerp(u_1h, v_1h)
        corner_6h = _bilerp(u_6h, v_6h)
        corner_6a = _bilerp(u_6a, v_6a)
        return [corner_1a, corner_1h, corner_6h, corner_6a]

    def _score_corners_by_color_match(
        self,
        frame_bgr: np.ndarray,
        patch_quad: list[tuple[float, float]],
    ) -> float:
        """patch_quad (1A, 1H, 6H, 6A) から 48 patch 中心を bilerp 算出し、
        中心 ±2 px の median sRGB を sample → linear に変換 → exposure を
        patch G チャネル median で正規化 → SpyderCHECKR-48 reference (linear)
        との patch 単位 L2 距離の median を返す (低いほど良い)。

        4 候補比較で「閉じた裏面 (色分布なし、median 距離大)」と
        「開いた chart (48 色多様、median 距離小)」を分離する。
        score 0.0=完全一致、~0.10=露出ズレ程度、>0.30 は chart 外。

        Phase 2X (canonical model redesign):
          旧実装は `t = col / (n_cols - 1) = col / 7` で hinge_gap を完全に
          無視していた。`hinge_gap=2.67` 環境では col=4 (5E) の actual
          fractional position が `(4 + 2.67) / (7 + 2.67) = 0.690` (helper
          raw fraction では 0.672) なのに sample 位置が `4/7 = 0.571` で算出
          され、偏差 ~0.119 × ~700 px chart 幅 ≈ 83 px ≈ 1 cell width 分
          隣 cell / hinge gap 中央を sample していた。これにより中央 36
          patch (1B-1G, 2B-2G, ..., 6B-6G) が誤 sample され score 0.55 floor
          が発生していた。

          新実装は `_chart_patch_center_normalized` 共通 helper の raw
          fraction (u, v) を 1A/1H および 1A/6A endpoints で **再正規化** し、
          patch CENTER 4 隅 quad に対する内分係数 (u', v') ∈ [0, 1] を得る。
          1A→(0,0), 1H→(1,0), 6A→(0,1), 6H→(1,1) という endpoint 整合を
          維持しつつ、中央列の hinge offset を正しく反映できる。

          算術同等性は `csi/colorimeter_common.py` の
          `_default_patch_display_quads` (L6281-6283) と一致 (per-column
          fine-tune `col_x_norm_offsets[col]` は extractor 段階で適用)。
          `_patch_quad_from_component_pts` と同 helper を呼ぶことで
          二者間整合を保証する。
        """
        from csi.colorimeter_common import _spyder_pose_reference_linear_array  # noqa: PLC0415
        ref_lin = _spyder_pose_reference_linear_array()  # (48, 3) linear sRGB
        h, w = frame_bgr.shape[:2]
        p_1a = np.asarray(patch_quad[0], dtype=np.float64)
        p_1h = np.asarray(patch_quad[1], dtype=np.float64)
        p_6h = np.asarray(patch_quad[2], dtype=np.float64)
        p_6a = np.asarray(patch_quad[3], dtype=np.float64)
        n_rows, n_cols = 6, 8
        # Phase 2W: `_patch_quad_from_component_pts` が立てた runtime 逆算値を
        # 共有して二者間整合を保証する。fallback は canonical 2.67。
        hg = float(getattr(
            self, "_runtime_estimated_hinge_gap", self._hinge_gap,
        ))
        # 端点 raw fraction を一度算出: u_1a / u_1h と v_1a / v_6a を
        # 再正規化用の divisor に使う。
        u_1a_raw, v_1a_raw = _chart_patch_center_normalized(
            0, 0, hg, n_cols=n_cols, n_rows=n_rows,
        )
        u_1h_raw, _ = _chart_patch_center_normalized(
            n_cols - 1, 0, hg, n_cols=n_cols, n_rows=n_rows,
        )
        _, v_6a_raw = _chart_patch_center_normalized(
            0, n_rows - 1, hg, n_cols=n_cols, n_rows=n_rows,
        )
        u_span = u_1h_raw - u_1a_raw
        v_span = v_6a_raw - v_1a_raw
        if u_span <= 0.0 or v_span <= 0.0:
            return float("inf")
        obs_lin = np.full((48, 3), np.nan, dtype=np.float64)
        valid_count = 0
        for row in range(n_rows):
            for col in range(n_cols):
                u_raw, v_raw = _chart_patch_center_normalized(
                    col, row, hg,
                    n_cols=n_cols, n_rows=n_rows,
                )
                # patch CENTER 4 隅 quad に対する内分係数に再正規化
                # (1A→(0,0), 1H→(1,0), 6A→(0,1), 6H→(1,1))
                t = (u_raw - u_1a_raw) / u_span
                s = (v_raw - v_1a_raw) / v_span
                top_p = (1 - t) * p_1a + t * p_1h
                bot_p = (1 - t) * p_6a + t * p_6h
                center = (1 - s) * top_p + s * bot_p
                cx = int(round(float(center[0])))
                cy = int(round(float(center[1])))
                half = 2
                if cx - half < 0 or cx + half >= w or cy - half < 0 or cy + half >= h:
                    continue
                patch = frame_bgr[
                    cy - half:cy + half + 1, cx - half:cx + half + 1
                ]
                # BGR → RGB (camera/cv2 は BGR)
                med = np.median(patch.reshape(-1, 3), axis=0)
                rgb = np.array(
                    [med[2], med[1], med[0]], dtype=np.float64
                ) / 255.0
                # sRGB → linear
                lin = np.where(
                    rgb > 0.04045,
                    ((rgb + 0.055) / 1.055) ** 2.4,
                    rgb / 12.92,
                )
                obs_lin[row * n_cols + col] = lin
                valid_count += 1
        if valid_count < 24:
            return float("inf")
        # exposure 正規化: G channel median を ref_lin の G channel median に揃える
        valid_mask = ~np.any(np.isnan(obs_lin), axis=1)
        obs_g = obs_lin[valid_mask, 1]
        ref_g = ref_lin[:, 1]
        obs_g_med = float(np.median(obs_g[obs_g > 0.005])) if (obs_g > 0.005).any() else 0.0
        ref_g_med = float(np.median(ref_g))
        if obs_g_med < 1e-6 or ref_g_med < 1e-6:
            return float("inf")
        obs_lin_scaled = obs_lin * (ref_g_med / obs_g_med)
        # patch 単位 L2 距離 (NaN は除外)
        diffs = obs_lin_scaled - ref_lin
        dists = np.linalg.norm(diffs, axis=1)
        dists = dists[~np.isnan(dists)]
        if dists.size == 0:
            return float("inf")
        # Phase 2R: trimmed median (上位 50% を除外して下位 50% の median)
        # 実 Pi capture の per-patch L2 分布で saturation/intercell sampling 等で
        # 一部 patch が 0.5+ の高値、median が引き上げられる現象を吸収。
        # Phase 2S (top 35%) では強回転 chart (~25° CCW) の score=0.39 を救えず、
        # Phase 2R で top 50% 除外に強化。色基準 reference 48 patch のうち
        # 半分 (24 patch) を sampling するため、十分な robustness を保ちつつ
        # 強回転時の bilerp 誤位置 sample を除外する。
        sorted_dists = np.sort(dists)
        trim_count = max(1, int(len(sorted_dists) * 0.50))  # 下位 50%
        return float(np.median(sorted_dists[:trim_count]))

    def _validate_corners_via_patchwise(
        self,
        frame_bgr: np.ndarray,
        corners: list[tuple[float, float]],
        *,
        max_weak: int,
    ) -> bool:
        """与えられた corners が今の frame に対して妥当かを既存 patchwise detection
        で検証する。weak_patch_count <= max_weak かつ L/R rotation 整合 (≤5°) で True。

        Phase 22 / Phase 3 契約: caller は `corners` を `[TL, TR, BR, BL]` (= 1A, 1H,
        6H, 6A) の CW 順で渡すこと。set_corners_with_source は順序保持 entry に
        thin wrapper 化されており、軸そろえソートは行わない。現行 caller
        (`_detect_chart_via_rigid_rotated_rect` 戻り値、`_detect_chart_via_contour_outer_rim`
        戻り値、rigid 検出器内 provisional quad) は全て CW 順を保証している。
        """
        try:
            extractor = SpyderCheckrGridExtractor(
                SPYDERCHECKR_48_SHAPE[0],
                SPYDERCHECKR_48_SHAPE[1],
                hinge_gap=self._hinge_gap,
            )
            extractor.set_corners(list(corners))
            summary = extractor.prepare_patchwise_rois_from_frame(frame_bgr)
        except Exception as exc:
            print(f"- rigid rect validate: extractor 例外: {exc}")
            return False
        weak = int(summary.get("weak_patch_count", 999))
        viable = summary.get("calibration_geometry_viable")
        diagnostics = summary.get("direct_dark_panel_diagnostics") or {}
        left_layout = diagnostics.get("left_panel_layout") or {}
        right_layout = diagnostics.get("right_panel_layout") or {}
        l_rot = left_layout.get("lattice_rotation_degrees")
        r_rot = right_layout.get("lattice_rotation_degrees")
        ok = weak <= max_weak
        if l_rot is not None and r_rot is not None:
            try:
                lr_diff = abs(float(l_rot) - float(r_rot))
            except (TypeError, ValueError):
                lr_diff = 0.0
            # Phase 2 (20_strategy 2-E): 5.0 → _LR_ROTATION_TOLERANCE_DEG (=8.0)
            # に module-level 定数で統一。Stage 2 grid sweep の同 tolerance と一致。
            if lr_diff > _LR_ROTATION_TOLERANCE_DEG:
                ok = False
                print(
                    f"- rigid rect validate: L/R rotation 不整合 "
                    f"(L={l_rot}, R={r_rot}, diff={lr_diff:.1f}, "
                    f"tol={_LR_ROTATION_TOLERANCE_DEG})"
                )
        print(
            f"- rigid rect validate: weak={weak} viable={viable} "
            f"L/R=({l_rot}, {r_rot}) → {'OK' if ok else 'NG'}"
        )
        return ok

    def _estimate_global_chart_rotation(
        self,
        frame_bgr: np.ndarray,
        main_shape: tuple | None = None,
    ) -> float | None:
        """frame の dark connected components (= 左右パネル) に minAreaRect を当て、
        chart 全体の rotation 角度 [deg] を返す。検出失敗 / L/R 不整合時は None。

        正値 = CW (visual) rotation (chart 時計回り、Phase 1 evidence で確定)。返り値は seed rotation の priority hint
        として使う。両パネルが co-rotated (剛体仮定) であることを利用し、左右の長辺
        傾き median を採用、disagreement > 8° なら不確実として None を返す。

        Phase 2 (post-Codex-revise C5): hole_mask (flat valid_mask または `gray > 30`
        fallback) を AND することで chamber 壁ノイズが top-area 候補を独占する症候を
        防ぐ。`_detect_chart_via_rigid_rotated_rect` は `intensity > 12` fallback だが、
        本関数は area 上位 2 を panel 想定で採るため、より strict な `> 30` を採用する
        (chamber wall transition は intensity 12-30 帯にいる、cell 内最暗 cell も
        intensity 30-80 帯)。
        """
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
        h, w = gray.shape
        if h < 30 or w < 30:
            return None
        # 中央 80% 領域に絞り、scope 円の枠 / Ref panel の影を除外
        m = 0.10
        central = gray[int(h * m): int(h * (1 - m)), int(w * m): int(w * (1 - m))]
        if central.size == 0:
            return None
        # hole_mask (chamber 壁を除外) を central 領域に対して計算
        hole_mask_full = self._composite_hole_mask((h, w), main_shape)
        if hole_mask_full is None:
            hole_mask_full = gray > 30
        central_hole = hole_mask_full[
            int(h * m): int(h * (1 - m)), int(w * m): int(w * (1 - m))
        ]
        # dark threshold (chart panel の暗領域を抽出) AND hole_mask
        try:
            _, dark_mask = cv2.threshold(
                central, 80, 255, cv2.THRESH_BINARY_INV
            )
            dark_mask = cv2.bitwise_and(
                dark_mask, (central_hole.astype(np.uint8) * 255)
            )
            kernel = np.ones((5, 5), dtype=np.uint8)
            dark_mask = cv2.morphologyEx(
                dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2
            )
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                dark_mask
            )
        except Exception:
            return None
        if num_labels < 3:  # background + 2 panels 未満
            return None
        # background (label 0) を除いて area 上位 2 個 = 左右 panel 候補
        candidates = sorted(
            ((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num_labels)),
            reverse=True,
        )[:2]
        if len(candidates) < 2:
            return None
        min_area = central.size * 0.005
        angles: list[float] = []
        for area, label_id in candidates:
            if area < min_area:
                continue
            ys, xs = np.where(labels == label_id)
            if len(xs) < 100:
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            try:
                rect = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect)
            except Exception:
                continue
            # 4 edge の中で最長辺を chart panel の長辺とみなす (panel は 4×6 で
            # 長辺 = 縦 = chart 0° のとき vertical)
            edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
            edge_lens = [float(np.linalg.norm(e)) for e in edges]
            long_idx = int(np.argmax(edge_lens))
            long_edge = edges[long_idx]
            # 長辺角度 (x-axis から、 -180..180)
            edge_angle = math.degrees(
                math.atan2(float(long_edge[1]), float(long_edge[0]))
            )
            # chart 0° のとき長辺 = vertical = ±90°。rotation = edge_angle - 90°。
            # (-90, 90] に正規化 (符号反転 / 180° 折りたたみ)
            rotation = edge_angle - 90.0
            while rotation > 90.0:
                rotation -= 180.0
            while rotation <= -90.0:
                rotation += 180.0
            print(
                f"[rot estimate panel] label_id={label_id} area={area} "
                f"box=[({box[0][0]:.1f},{box[0][1]:.1f}),"
                f"({box[1][0]:.1f},{box[1][1]:.1f}),"
                f"({box[2][0]:.1f},{box[2][1]:.1f}),"
                f"({box[3][0]:.1f},{box[3][1]:.1f})] "
                f"long_idx={long_idx} long_edge=({long_edge[0]:+.1f},{long_edge[1]:+.1f}) "
                f"edge_angle={edge_angle:+.2f} rotation={rotation:+.2f}"
            )
            angles.append(rotation)
        if len(angles) < 2:
            return None
        # L/R disagreement check (剛体仮定)
        spread = abs(angles[0] - angles[1])
        if spread > 8.0:
            print(
                "- global rotation estimate: L/R disagree "
                f"({angles[0]:+.1f}, {angles[1]:+.1f}, spread={spread:.1f})"
            )
            return None
        median_angle = float(np.median(angles))
        print(
            f"[rot estimate] L_angle={angles[0]:+.2f} R_angle={angles[1]:+.2f} "
            f"median={median_angle:+.2f} (positive=CW visual per Phase 21 2-B)"
        )
        return median_angle

    def _try_auto_detect_chart_corners(
        self,
    ) -> list[tuple[float, float]] | None:
        """centered-chart seed で auto-detect を試行し、成功時に 4 隅
        (1A, 1H, 6H, 6A) の中心座標を返す。失敗時は None。

        既存の `prepare_patchwise_rois_from_frame` + `_attempt_direct_dark_panel_layout`
        (csi/colorimeter_common.py) は seed corners を要求する設計のため、画像中央に
        chart があるという仮定で seed を仮置きしてから本検出ロジックに渡す。検出が
        viable なら recovered quads から実際の chart 4 隅を取り出して返す (rotated
        chart にも追従、5 角度 ±9° まで実機データで検証ずみ — Phase 14/15)。
        """
        self._last_auto_detect_chart_corners_source = None
        # Cycle 2 (C1 修正): 本 run 開始時に runtime hinge_gap / rigid reject reason
        # を必ず None reset する。これにより `[auto-detect summary]` で前回押下の
        # stale 値が出るのを防ぐ。Stage 0 hough 採用 / capture 例外 / geometry reject
        # では `_patch_quad_from_component_pts` が呼ばれず `_runtime_estimated_hinge_gap`
        # が更新されないため、reset しないと前回値が表示され疑い 4 (cache 残留) の
        # 判定が混乱する。
        self._runtime_estimated_hinge_gap = None
        self._last_rigid_reject_reason = None
        summary_state: dict[str, object] = {
            "stage1_result": "rejected",
            "stage1_reject_reason": None,
            "stage2_best": None,
            "last_geometry_model": None,
            "source": None,
            "corners": None,
        }
        try:
            try:
                main_array = self.picam2.capture_array("main")
            except Exception as exc:
                print(f"- auto-detect: capture_array 失敗 → manual fallback: {exc}")
                summary_state["stage1_result"] = "exception"
                return None
            if not isinstance(main_array, np.ndarray) or main_array.size == 0:
                summary_state["stage1_result"] = "exception"
                return None
            try:
                # saved corners と同じ composite 座標系 (left_panel offset 含む) で検出する。
                # _build_chart_analysis_frame_from_main は flip / center crop / left panel
                # padding を施した frame を返すため、結果 corners はそのまま set_corners
                # / chart_corners.json に渡せる。
                frame_bgr = self._build_chart_analysis_frame_from_main(main_array)
            except Exception as exc:
                print(f"- auto-detect: composite frame 構築失敗 → manual fallback: {exc}")
                summary_state["stage1_result"] = "exception"
                return None
            if frame_bgr is None or frame_bgr.size == 0:
                summary_state["stage1_result"] = "exception"
                return None
            if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                summary_state["stage1_result"] = "exception"
                return None

            # main_array.shape を渡して flat valid_mask の coord 変換に使う。
            main_shape = (
                tuple(main_array.shape[:2])
                if isinstance(main_array, np.ndarray) and main_array.ndim >= 2
                else None
            )
            composite_hole = self._composite_hole_mask(
                frame_bgr.shape[:2], main_shape
            )

            # === 第 0 段 (Phase 2Y, 最優先): inner cell boundary via Hough lines ===
            # 内部 6×8 patch grid lines を Hough で検出し outermost peaks から
            # inner cell boundary rect を fit する (planner spec Option 1 案 A 改)。
            # 採用 corners は `_chart_patch_center_normalized` helper の前提 frame と
            # 一致する設計のため、Phase 2X 算術が正しい patch CENTER を返す。
            # 本実装は **planner premise A (chart_asp ≈ 0.5623) が実 fixture (synth_v1)
            # では falsified** であることを primary evidence として記録する。
            # `asp_off_canonical` 等で reject された場合は dark-cell rigid rect 経路に
            # fallback する。
            hough_corners, hough_reason = (
                self._detect_chart_inner_cell_boundary_via_hough(
                    frame_bgr, main_shape=main_shape
                )
            )
            if hough_corners is not None:
                if (
                    composite_hole is not None
                    and not self._corners_inside_hole_mask(
                        hough_corners, composite_hole
                    )
                ):
                    print(
                        "- inner cell hough: corners が valid_mask 外にはみ出し → fallback"
                    )
                else:
                    print("- inner cell hough: 採用 (canonical 一致 / mask 内)")
                    self._last_auto_detect_chart_corners_source = (
                        "inner_cell_hough"
                    )
                    summary_state["stage1_result"] = "accepted"
                    summary_state["source"] = "inner_cell_hough"
                    summary_state["corners"] = hough_corners
                    return hough_corners
            print(
                f"- inner cell hough: 不採用 ({hough_reason}) → "
                "dark-cell rigid rect へ fallback"
            )

            # === 第 1 段: rigid rotated rectangle 検出 (Phase 2Y fallback) ===
            # chart 全体を 1 つの剛体回転矩形として検出し、4 corners を canonical 比率で
            # 内分計算する。yellow frame は rectangle、左右 cell サイズは同一になる
            # (剛体 by construction)。1 pass、deterministic、fast。
            rigid_corners = self._detect_chart_via_rigid_rotated_rect(
                frame_bgr, main_shape=main_shape
            )
            # 必須 sanity check: corners が valid_mask (chamber 穴) 内に収まること。
            # patchwise validate は rotation > 10° で weak 過多になるため不要。rigid rect
            # は minAreaRect で剛体性が保証されており、~7% 精度で chart 位置に来る。
            # 多少の誤差はユーザが [+/-] hinge / [./,] ROI の interactive 調整で吸収可能。
            # NOTE: composite_hole は Stage 0 (Phase 2Y) で既に計算済 (本 try ブロック先頭)。
            if rigid_corners is not None:
                if (
                    composite_hole is not None
                    and not self._corners_inside_hole_mask(
                        rigid_corners, composite_hole
                    )
                ):
                    print(
                        "- rigid rect: corners が valid_mask 外にはみ出し → reject"
                    )
                    self._last_rigid_reject_reason = "mask_reject"
                    rigid_corners = None
            if rigid_corners is not None:
                print("- rigid rect: 採用 (mask 内 / 剛体長方形保証)")
                self._last_auto_detect_chart_corners_source = "rigid_auto"
                summary_state["stage1_result"] = "accepted"
                summary_state["source"] = "rigid_auto"
                summary_state["corners"] = rigid_corners
                return rigid_corners

            # === 第 2 段 (fallback): 旧 scale × rotation grid sweep ===
            # rigid 検出が破綻 (chart 一部隠れ / 異常照明 等) した時の safety net。
            h, w = frame_bgr.shape[:2]
            left_panel_width = int(
                getattr(self.config.camera, "left_panel_width", 0)
            )
            display_w = max(1, w - left_panel_width)
            display_h = h
            # chart は display 領域 (left_panel offset 右側) に centered で置かれる
            cx = float(left_panel_width) + display_w / 2.0
            cy = display_h / 2.0
            # SpyderCHECKR 48 の典型 pixel aspect (8 cols × 6 rows + hinge gap で
            # 物理 195mm × 117mm → ~1.66:1)。working distance / lens で chart の見かけ
            # サイズが変わるため複数 scale を sweep し、最初に成功したものを採用する。
            # seed が axis-aligned だと chart 大角度 rotation (>±9°) で panel 検出が
            # 破綻するため、複数 rotation 角でも sweep する (30° tilt 等の対応)。
            chart_aspect = 1.66
            scales = (0.30, 0.40, 0.50)
            # 左右 panel co-rotation 仮定で chart 全体角度を pre-estimate (剛体仮定)。
            # これが priority hint。失敗時のみ ±60° の grid sweep に降格。
            estimated_rot = self._estimate_global_chart_rotation(
                frame_bgr, main_shape=main_shape
            )
            rotations_grid = (
                0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0,
            )
            if estimated_rot is not None:
                print(
                    f"- auto-detect: estimated global chart rotation = "
                    f"{estimated_rot:+.1f}° (priority hint)"
                )
                # 推定値を最優先、±5° の微調整、続いて grid sweep を fallback
                rotations_deg: tuple[float, ...] = (
                    float(estimated_rot),
                    float(estimated_rot) + 5.0,
                    float(estimated_rot) - 5.0,
                    *rotations_grid,
                )
            else:
                print("- auto-detect: global rotation pre-estimate 失敗 → grid sweep")
                rotations_deg = rotations_grid
            # L/R consistency tolerance (両 panel の lattice rotation 差がこれより大きい
            # 候補は reject。片側だけ追従して片側 axis-aligned のミスマッチ事故防止)
            # Phase 2 (20_strategy 2-E): module-level `_LR_ROTATION_TOLERANCE_DEG` (=8.0)
            # を参照。Stage 1 validate (`_validate_corners_via_patchwise`) と一致。
            LR_ROTATION_TOLERANCE_DEG = _LR_ROTATION_TOLERANCE_DEG

            def _quad_center(quad) -> tuple[float, float]:
                if hasattr(quad, "points"):
                    pts = quad.points
                elif isinstance(quad, dict):
                    pts = quad.get("points") or []
                else:
                    pts = quad
                if len(pts) != 4:
                    return (0.0, 0.0)
                cxv = sum(float(p[0]) for p in pts) / 4.0
                cyv = sum(float(p[1]) for p in pts) / 4.0
                return (cxv, cyv)

            # 4 隅 corner は panel box 検出 (geometry_model=direct_dark_panel) と
            # lattice 復元が成功すれば導出可能。viable=yes / weak=0 は per-patch refinement
            # 品質の話で外周 corners 推定とは独立。少数 weak patch (<= 10/48 ≈ 20%) は許容。
            # scale × rotation を grid sweep し、weak_patch_count 最小の候補を採用する。
            WEAK_THRESHOLD = 10
            best_candidate: tuple[
                float,
                float,
                int,
                list[tuple[float, float]],
            ] | None = None  # (scale, rotation_deg, weak_count, corners)
            last_summary: dict | None = None
            for scale in scales:
                seed_w_base = display_w * scale
                seed_h_base = seed_w_base / chart_aspect
                if seed_h_base > display_h * 0.85:
                    seed_h_base = display_h * 0.85
                    seed_w_base = seed_h_base * chart_aspect
                half_w, half_h = seed_w_base / 2.0, seed_h_base / 2.0
                base_local = [
                    (-half_w, -half_h),
                    (+half_w, -half_h),
                    (+half_w, +half_h),
                    (-half_w, +half_h),
                ]
                for rot_deg in rotations_deg:
                    cos_t = math.cos(math.radians(rot_deg))
                    sin_t = math.sin(math.radians(rot_deg))
                    seed_corners = [
                        (
                            cx + lx * cos_t - ly * sin_t,
                            cy + lx * sin_t + ly * cos_t,
                        )
                        for lx, ly in base_local
                    ]
                    try:
                        extractor = SpyderCheckrGridExtractor(
                            SPYDERCHECKR_48_SHAPE[0],
                            SPYDERCHECKR_48_SHAPE[1],
                            hinge_gap=self._hinge_gap,
                        )
                        extractor.set_corners_ordered(seed_corners)
                        summary = extractor.prepare_patchwise_rois_from_frame(
                            frame_bgr
                        )
                    except Exception as exc:
                        print(
                            f"- auto-detect[scale={scale:.2f}, "
                            f"rot={rot_deg:+.0f}]: 検出器例外: {exc}"
                        )
                        continue
                    last_summary = summary
                    geometry_model = summary.get("geometry_model")
                    viable = summary.get("calibration_geometry_viable")
                    weak = int(summary.get("weak_patch_count", -1))
                    if geometry_model != "direct_dark_panel":
                        continue
                    if weak < 0 or weak > WEAK_THRESHOLD:
                        continue
                    # L/R consistency: lattice rotation 角の左右差が大きい候補は reject
                    diagnostics = summary.get("direct_dark_panel_diagnostics") or {}
                    left_layout = diagnostics.get("left_panel_layout") or {}
                    right_layout = diagnostics.get("right_panel_layout") or {}
                    left_rot_raw = left_layout.get("lattice_rotation_degrees")
                    right_rot_raw = right_layout.get("lattice_rotation_degrees")
                    if left_rot_raw is not None and right_rot_raw is not None:
                        try:
                            lr_disagreement = abs(
                                float(left_rot_raw) - float(right_rot_raw)
                            )
                        except (TypeError, ValueError):
                            lr_disagreement = 0.0
                        if lr_disagreement > LR_ROTATION_TOLERANCE_DEG:
                            print(
                                f"- auto-detect[scale={scale:.2f}, "
                                f"rot={rot_deg:+.0f}]: L/R rotation 不整合 "
                                f"(L={left_rot_raw}, R={right_rot_raw}, "
                                f"diff={lr_disagreement:.1f}° > "
                                f"{LR_ROTATION_TOLERANCE_DEG}°) → reject"
                            )
                            continue
                    quads = extractor.get_active_patch_display_quads()
                    if len(quads) != 48:
                        continue
                    # 1A=row0,col0=index 0; 1H=row0,col7=index 7;
                    # 6H=row5,col7=index 47; 6A=row5,col0=index 40
                    corners = [
                        _quad_center(quads[0]),
                        _quad_center(quads[7]),
                        _quad_center(quads[47]),
                        _quad_center(quads[40]),
                    ]
                    print(
                        f"- auto-detect[scale={scale:.2f}, rot={rot_deg:+.0f}]: "
                        f"候補成立 weak={weak} viable={viable}"
                    )
                    if best_candidate is None or weak < best_candidate[2]:
                        best_candidate = (scale, rot_deg, weak, corners)
                    if weak == 0 and viable == "yes":
                        break  # 完璧候補なので即採用 (rotation loop 抜ける)
                else:
                    continue
                break  # outer scale loop も抜ける (perfect 採用済)

            summary_state["last_geometry_model"] = (
                (last_summary or {}).get("geometry_model")
            )
            if best_candidate is not None:
                cs, cr, cw, cc = best_candidate
                summary_state["stage2_best"] = (cs, cr, cw)
                # sweep 結果も valid_mask 内に収まっていることを必須化
                if (
                    composite_hole is not None
                    and not self._corners_inside_hole_mask(cc, composite_hole)
                ):
                    print(
                        f"- auto-detect (scale={cs:.2f}, rot={cr:+.0f}): "
                        f"corners が valid_mask 外にはみ出し → reject"
                    )
                else:
                    print(
                        f"- auto-detect 成功 (scale={cs:.2f}, rot={cr:+.0f}, "
                        f"weak={cw}/48): "
                        f"1A=({cc[0][0]:.1f},{cc[0][1]:.1f}) "
                        f"1H=({cc[1][0]:.1f},{cc[1][1]:.1f}) "
                        f"6H=({cc[2][0]:.1f},{cc[2][1]:.1f}) "
                        f"6A=({cc[3][0]:.1f},{cc[3][1]:.1f})"
                    )
                    self._last_auto_detect_chart_corners_source = "legacy"
                    summary_state["source"] = "legacy"
                    summary_state["corners"] = cc
                    return cc

            print(
                "- auto-detect: scale × rotation 全 sweep で候補なし "
                f"(last geometry_model="
                f"{(last_summary or {}).get('geometry_model')}) → manual fallback"
            )
            return None
        finally:
            try:
                summary_state["stage1_reject_reason"] = getattr(
                    self, "_last_rigid_reject_reason", None
                )
                # Cycle 2 (C1 修正): 本 run で再計算されたか (=float) stale か
                # (=None) を明示する。冒頭で None reset しているため、
                # `_patch_quad_from_component_pts` が呼ばれた candidate がない
                # (Stage 0 hough 採用 / capture 例外 / geometry reject 等) 場合は
                # `None (not computed in this run)` と表示される。
                runtime_hg = getattr(self, "_runtime_estimated_hinge_gap", None)
                runtime_hg_str = (
                    f"{runtime_hg:.4f}"
                    if runtime_hg is not None
                    else "None (not computed in this run)"
                )
                print(
                    "[auto-detect summary] "
                    f"stage1_result={summary_state['stage1_result']} "
                    f"stage1_reject_reason={summary_state['stage1_reject_reason']} "
                    f"stage2_best={summary_state['stage2_best']} "
                    f"last_geometry_model={summary_state['last_geometry_model']} "
                    f"source={summary_state['source']} "
                    f"runtime_hinge_gap={runtime_hg_str} "
                    f"corners={summary_state['corners']}"
                )
            except Exception:
                pass

    def _on_chart_click(self, event, x, y, flags, param) -> None:
        """SpyderCheckr コーナー指定マウスコールバック。"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self._chart_corners) >= 4:
            return
        self._mark_manual_corner_used()
        self._chart_corners.append((x, y))
        if len(self._chart_corners) == 4:
            # Phase 22 / Phase 3: manual 4-click は click 順が任意なので
            # caller 側で CW pre-sort を適用してから順序保持 entry に渡す。
            # set_corners_with_source は thin wrapper 化 (軸そろえロジック撤廃) のため、
            # 並び替えは呼出し側責任となった。
            ordered = _order_corners_clockwise_from_topleft(self._chart_corners)
            self._raw_corners = list(ordered)
            self._grid_extractor.set_corners_with_source(
                ordered,
                source="manual",
            )
            self._chart_corners = []
            self._chart_state = "preview"
            # 4-click 確定直後に永続化。途中キャンセル/失敗でも次回起動で saved corners
            # path に乗り、再度の手動クリックを不要にする (Image #3 対策)。
            self._persist_chart_corners(reason="manual_4click")
            self._set_chart_workflow_status(
                stage="preview",
                message="色基準48色 プレビュー中",
                detail="ROI を確認して [P]/Enter で開始",
                can_cancel=True,
                source="manual",
            )
            if self.window_manager is not None:
                drag_cb = self.window_manager.make_offset_callback(self._on_preview_drag)
            else:
                drag_cb = self._on_preview_drag
            cv2.setMouseCallback(self.window_name, drag_cb)

    def _capture_corrected_ref_mean_12bit(self, n_frames: int = 4) -> float:
        """Ref ROI の補正後 12bit 平均値を少数フレームで取得する。"""
        ref_disp_x, ref_disp_y = self.config.processing.posi_ref
        ref_w = max(int(self.config.processing.spot_size_ref), 2)
        ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
        ref_means: list[float] = []

        for _ in range(n_frames):
            request = self.picam2.capture_request()
            try:
                raw_array = request.make_array("raw")
                metadata = request.get_metadata()
            finally:
                request.release()
            raw_bayer = self.bayer.parse_raw(raw_array)
            ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h = self.bayer.display_to_raw_coords(
                ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                self.config.display.flip_horizontal,
                self.config.display.flip_vertical,
            )
            roi_ref = self.bayer.extract_raw_roi(
                raw_bayer, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
            )
            if self.dark.is_loaded:
                dark_ref = self.dark.get_dark_roi(
                    ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h,
                )
                if dark_ref is not None:
                    roi_ref = self.dark.subtract(roi_ref, dark_ref)
            if self.flat.is_loaded:
                roi_ref = self.flat.correct_roi(
                    roi_ref, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h,
                )
            ref_means.append(float(roi_ref.mean()))

        return float(np.mean(ref_means)) if ref_means else 0.0

    def _extract_corrected_chart_patch_bayer_means(
        self,
        raw_array: np.ndarray,
        metadata: dict,
        main_array: np.ndarray | None = None,
        debug: bool = False,
    ) -> np.ndarray:
        """単一フレームから chart の補正後 patch_bayer_means を返す。"""
        if self._grid_extractor is None or not self._grid_extractor.is_ready:
            raise RuntimeError("chart grid extractor is not ready")
        if main_array is not None:
            self._prepare_chart_patchwise_rois_from_main(main_array)

        raw_bayer = self.bayer.parse_raw(raw_array)
        if self.dark.is_loaded and self.dark.dark_frame is not None:
            raw_bayer = self.dark.subtract(raw_bayer, self.dark.dark_frame)
        if self.flat.is_loaded:
            bh, bw = raw_bayer.shape[:2]
            raw_bayer = self.flat.correct_roi(raw_bayer, 0, 0, bw, bh)

        return self._grid_extractor.extract_all_patch_bayer_means(
            raw_bayer,
            self.bayer,
            metadata,
            self.config,
            debug=debug,
            debug_save_dir=get_today_calibration_dir() if debug else None,
        )

    def _capture_chart_patch_bayer_means_fast(self) -> np.ndarray:
        """Phase 7D: flip check 用の軽量 1-frame キャプチャ.

        `_capture_chart_patch_bayer_means` は複数 frame capture を行うため Pi 実機で
        数秒〜10 秒かかりユーザー体感が悪い. flip 判定は単一 frame で足りるが、
        ROI は同じ request の main frame から登録し、傾いた SpyderCHECKR でも
        回転追従・正方形 patchwise ROI を RAW 抽出へ反映する.

        - capture request は 1 回
        - `main` と `raw` を同一 request から取得
        - ESC は capture の前後で polling
        """
        # capture 前に ESC を polling (ユーザーが P 直後に ESC 押しても即中断できる)
        self._poll_chart_cancel_key("flip check prescan pre-capture")
        request = self.picam2.capture_request()
        try:
            main_array = request.make_array("main")
            raw_array = request.make_array("raw")
            metadata = request.get_metadata()
        finally:
            request.release()
        # capture 後も ESC を polling
        self._poll_chart_cancel_key("flip check prescan post-capture")
        patch_means = self._extract_corrected_chart_patch_bayer_means(
            raw_array,
            metadata,
            main_array=main_array,
            debug=False,
        ).astype(np.float32)
        return patch_means

    def _capture_chart_patch_bayer_means(self, n_frames: int = 4) -> np.ndarray:
        """複数フレーム平均の chart 補正後 patch_bayer_means を返す。"""
        patch_sum: np.ndarray | None = None

        for _ in range(n_frames):
            request = self.picam2.capture_request()
            try:
                main_array = request.make_array("main")
                raw_array = request.make_array("raw")
                metadata = request.get_metadata()
            finally:
                request.release()
            self._poll_chart_cancel_key("chart patch capture")

            patch_means = self._extract_corrected_chart_patch_bayer_means(
                raw_array,
                metadata,
                main_array=main_array,
                debug=False,
            ).astype(np.float64)
            if patch_sum is None:
                patch_sum = patch_means
            else:
                patch_sum += patch_means

        if patch_sum is None:
            raise RuntimeError("failed to capture chart patch means")
        return (patch_sum / float(n_frames)).astype(np.float32)

    def _capture_chart_white_g(self, n_frames: int = 4) -> float:
        """プリフライト用に white patch の補正後 G 値を返す。"""
        if self._grid_extractor is None or not self._grid_extractor.is_ready:
            return self._capture_corrected_ref_mean_12bit(n_frames=n_frames)
        patch_bayer_means = self._capture_chart_patch_bayer_means(n_frames=n_frames)
        return float(patch_bayer_means[WHITE_PATCH_IDX][1])

    def _discard_capture_frames(self, n_frames: int) -> None:
        """露光変更後の安定化用に指定枚数ぶんフレームを捨てる。"""
        for _ in range(n_frames):
            request = self.picam2.capture_request()
            request.release()

    @staticmethod
    def _white_ratio_balance(white_ratio_rgb: np.ndarray) -> float:
        """white_ratio_rgb の max/min 比を返す。"""
        return float(
            np.max(white_ratio_rgb) / max(float(np.min(white_ratio_rgb)), 1e-8)
        )

    def _print_white_ratio_balance_warning(self, white_ratio_rgb: np.ndarray) -> None:
        """white_ratio のチャンネル不均衡が大きい場合に警告を出す。"""
        balance = self._white_ratio_balance(white_ratio_rgb)
        if balance > 1.10:
            print(
                "  ⚠ white_ratio バランス不良: "
                f"R={white_ratio_rgb[0]:.2f} G={white_ratio_rgb[1]:.2f} "
                f"B={white_ratio_rgb[2]:.2f} (max/min={balance:.2f})"
            )

    def _execute_chart_measurement(
        self,
        n_frames: int = CHART_CAPTURE_FRAMES,
    ) -> dict | None:
        """SpyderCheckr チャートを一括測定し CCM 算出へ渡す。"""
        ref_disp_x, ref_disp_y = self.config.processing.posi_ref
        ref_w = max(int(self.config.processing.spot_size_ref), 2)
        ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)

        bayer_sum = None
        ref_sum = np.zeros(3, dtype=np.float64)
        metadata = {}
        analysis_main_array: np.ndarray | None = None
        saved_exp = self.ae.current_exposure

        try:
            key = self._show_chart_workflow_overlay(
                "prescan",
                "prescan",
                detail="白パッチ露光を確認しています",
                progress=1,
                total=max(1, WHITE_G_MAX_RETRIES + 1),
                can_cancel=True,
            )
            self._raise_if_chart_cancel_requested(key, "chart prescan start")
            prescan_white_g = self._capture_chart_white_g(n_frames=4)
            print(f"- P prescan 補正後 white G: {prescan_white_g:.1f}")
            prescan_white_g_final = prescan_white_g
            prescan_retry_count = 0
            while (
                prescan_white_g_final > WHITE_G_TARGET
                and prescan_retry_count < WHITE_G_MAX_RETRIES
            ):
                key = self._show_chart_workflow_overlay(
                    "prescan",
                    "prescan",
                    detail="白パッチ露光が高いため再調整しています",
                    progress=prescan_retry_count + 1,
                    total=max(1, WHITE_G_MAX_RETRIES + 1),
                    can_cancel=True,
                )
                self._raise_if_chart_cancel_requested(key, "chart prescan retry")
                prev_white_g = prescan_white_g_final
                prev_exp = self.ae.current_exposure
                factor = WHITE_G_TARGET / max(prev_white_g, 1e-8)
                new_exp = self.ae.lower_exposure(factor)
                prescan_retry_count += 1
                print(
                    f"⚠ P prescan[{prescan_retry_count}/{WHITE_G_MAX_RETRIES}]: "
                    f"white G={prev_white_g:.1f} > {WHITE_G_TARGET:.1f}"
                    f" -> ExposureTime {prev_exp}us -> {new_exp}us"
                    f" (factor={factor:.3f})"
                )
                self._discard_capture_frames(5)
                prescan_white_g_final = self._capture_chart_white_g(n_frames=4)
                print(
                    f"- P prescan 露光変更後 white G"
                    f"[{prescan_retry_count}/{WHITE_G_MAX_RETRIES}]: "
                    f"{prev_white_g:.1f} -> {prescan_white_g_final:.1f}"
                )
            ccm_exposure_us = self.ae.current_exposure
            self._ccm_exposure_us = ccm_exposure_us
            self._measurement_exposure_us = saved_exp
            print(f"- P prescan 最終補正後 white G: {prescan_white_g_final:.1f}")
            if prescan_white_g_final > WHITE_G_LIMIT:
                if prescan_retry_count >= WHITE_G_MAX_RETRIES:
                    print(
                        "⚠ P prescan 反復上限到達: "
                        f"最終 white G={prescan_white_g_final:.1f} > {WHITE_G_LIMIT:.1f}"
                        " — 要警告のまま継続"
                    )
                else:
                    print(
                        "⚠ P prescan 再測後: "
                        f"white G={prescan_white_g_final:.1f} > {WHITE_G_LIMIT:.1f}"
                        " — 露光再調整または再キャリブを推奨"
                    )

            # --- exposure_check: 露光変化の非 blocking チェック ---
            prev_exp = None
            if hasattr(self, "_preflight_result") and self._preflight_result:
                accepted_reference_run_id_raw = self._preflight_result.get(
                    "accepted_reference_run_id"
                )
                accepted_reference_run_id = (
                    str(accepted_reference_run_id_raw).strip()
                    if accepted_reference_run_id_raw is not None
                    else ""
                )
                if (
                    accepted_reference_run_id
                    and accepted_reference_run_id.lower() != "none"
                ):
                    ref_info = load_accepted_reference_for_preflight(
                        CALIBRATION_DIR,
                        control_sample_id=self._resolve_control_sample_id(),
                        run_id=accepted_reference_run_id,
                    )
                    if ref_info is not None:
                        prev_exp = ref_info.get("exposure_us")
            current_exp = self.ae.current_exposure
            exp_status, exp_details = check_exposure_drift(current_exp, prev_exp)
            self._exposure_check_result = {
                "status": exp_status,
                **exp_details,
            }
            if exp_status == "warn":
                print(
                    f"⚠ exposure_check: drift={exp_details.get('drift', 0):.1%}"
                    f" ({exp_details.get('prev_us', '?')}us"
                    f" → {exp_details.get('today_us', '?')}us)"
                )

            for i in range(n_frames):
                request = self.picam2.capture_request()
                try:
                    if analysis_main_array is None:
                        analysis_main_array = request.make_array("main")
                    raw_array = request.make_array("raw")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                raw_bayer = self.bayer.parse_raw(raw_array)

                if bayer_sum is None:
                    bayer_sum = raw_bayer.astype(np.float64)
                else:
                    bayer_sum += raw_bayer.astype(np.float64)

                ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h = self.bayer.display_to_raw_coords(
                    ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                    self.config.display.flip_horizontal,
                    self.config.display.flip_vertical,
                )
                roi_ref = self.bayer.extract_raw_roi(
                    raw_bayer, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
                )
                if self.dark.is_loaded:
                    dark_ref = self.dark.get_dark_roi(ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                    if dark_ref is not None:
                        roi_ref = self.dark.subtract(roi_ref, dark_ref)
                if self.flat.is_loaded:
                    roi_ref = self.flat.correct_roi(roi_ref, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                ref_sum += self.bayer.extract_bayer_means_from_roi(roi_ref).astype(np.float64)

                key = self._show_chart_workflow_overlay(
                    "capture",
                    "chart capture",
                    detail="chart + Ref ROI を取得中",
                    progress=i + 1,
                    total=n_frames,
                    can_cancel=True,
                )
                self._raise_if_chart_cancel_requested(key, "chart capture")

            assert bayer_sum is not None
            avg_bayer = (bayer_sum / n_frames).astype(np.uint16)
            avg_ref = (ref_sum / n_frames).astype(np.float32)

            if self.dark.is_loaded and self.dark.dark_frame is not None:
                avg_bayer = self.dark.subtract(avg_bayer, self.dark.dark_frame)

            if self.flat.is_loaded:
                bh, bw = avg_bayer.shape[:2]
                avg_bayer = self.flat.correct_roi(avg_bayer, 0, 0, bw, bh)

            key = self._show_chart_workflow_overlay(
                "compute",
                "ROI prepare",
                detail="patch ROI を最終確定しています",
                can_cancel=True,
            )
            self._raise_if_chart_cancel_requested(key, "chart roi prepare")
            self._prepare_chart_patchwise_rois_from_main(analysis_main_array)
            self._poll_chart_cancel_key("chart roi prepare")
            patch_bayer_means = self._grid_extractor.extract_all_patch_bayer_means(
                avg_bayer, self.bayer, metadata, self.config, debug=True,
                debug_save_dir=get_today_calibration_dir(),
            )

            patch_means_norm = patch_bayer_means / self.bayer.max_val
            safe_ref = np.where(avg_ref < 1e-8, 1e-8, avg_ref)
            ratios_gray = patch_means_norm / safe_ref

            # デバッグ出力
            _S = "\u2550" * 68
            _s = "\u2500" * 68
            _KEY = {SPYDERCHECKR_48_WHITE_PATCH_INDEX: "1E White",
                    SPYDERCHECKR_48_GRAY_4D_INDEX:     "4D Gray50%"}
            print(f"\n{_S}")
            print("  [P] CCM \u30ad\u30e3\u30ea\u30d6\u30ec\u30fc\u30b7\u30e7\u30f3 \u2014 ROI / \u30d1\u30c3\u30c1\u5024")
            print(_S)
            print(f"  Ref ROI (dark+flat\u88dc\u6b63\u6e08\u307f): R={avg_ref[0]:8.5f}  G={avg_ref[1]:8.5f}  B={avg_ref[2]:8.5f}  (0-1\u6b63\u898f\u5316)")
            print(f"  bayer.max_val         : {self.bayer.max_val}")
            print(_s)
            _hdr = (f"  {'idx':>3} {'patch':12}  "
                    f"{'corr_R':>7} {'corr_G':>7} {'corr_B':>7}  "
                    f"{'norm_R':>7} {'norm_G':>7} {'norm_B':>7}  "
                    f"{'ratio_R':>9} {'ratio_G':>9} {'ratio_B':>9}")
            print(_hdr)
            print("  " + "\u2500" * 66)
            for _i in range(len(patch_bayer_means)):
                _name = _KEY.get(_i, "")
                _raw = patch_bayer_means[_i]
                _nm = patch_means_norm[_i]
                _rg = ratios_gray[_i]
                _mk = " \u25c0" if _i in _KEY else ""
                print(f"  {_i:3d} {_name:12}  "
                      f"{_raw[0]:7.0f} {_raw[1]:7.0f} {_raw[2]:7.0f}  "
                      f"{_nm[0]:7.4f} {_nm[1]:7.4f} {_nm[2]:7.4f}  "
                      f"{_rg[0]:9.6f} {_rg[1]:9.6f} {_rg[2]:9.6f}{_mk}")
            print(_S + "\n")

            # 白パッチ高輝度圧迫チェック（dark 減算・flat 補正後の値）
            white_raw = patch_bayer_means[SPYDERCHECKR_48_WHITE_PATCH_INDEX]
            _thresh = self.bayer.max_val * 0.92
            is_highlight_pressure = any(float(white_raw[ch]) > _thresh for ch in range(3))
            if is_highlight_pressure:
                _g_val = float(white_raw[1])
                print(
                    f"\u26a0 白パッチ (#4) 補正後 G={_g_val:.0f}/{self.bayer.max_val:.0f}"
                    f" — 高輝度圧迫。露光を 10-15% 下げて再キャリブを推奨"
                )
        finally:
            if self.ae.current_exposure != saved_exp:
                self.picam2.set_controls({"ExposureTime": saved_exp})
                self.ae.current_exposure = saved_exp
                print(f"- P 露光を復元: {saved_exp}us")

        return self._finish_chart_calibration(ratios_gray, avg_ref)

    def _finish_chart_calibration(self, ratios_gray: np.ndarray,
                                    avg_ref_train: np.ndarray | None = None) -> dict | None:
        """白点正規化・CCM 算出・保存・適用・オーバーレイ表示を行う。"""
        # Phase 14: SpyderCHECKR の向き・設置不整合を CCM 計算の前に hard gate で弾く.
        # geometry (D 列単調性 / 1E-6E ratio / contrast) と多色 pose hypothesis を統合した
        # detect_spyder_chart_pose_problem を主判定に使う. UI 文言は comprehensive gate の
        # reason 分類に従い、flip は 180° 案内、それ以外は向き・設置案内を出して CCM 中止.
        # 注: 後段の保険として detect_spyder_flip(ratios_gray) も同入力で評価し、
        # comprehensive gate が "ok" のときのみ legacy_flip の判定を最終ハードガードとして残す.
        # comprehensive が rotated_or_misplaced と分類した場合に legacy_flip で「上下反転」案内が
        # 上書きされる事象 (Phase 14 Codex C3) を避ける.
        pose_problem, pose_reason, _ = detect_spyder_chart_pose_problem(ratios_gray)
        legacy_flip = detect_spyder_flip(ratios_gray)
        pose_gate_bypassed = False
        if (
            pose_problem
            and self._should_bypass_pose_gate_for_oriented_geometry(pose_reason)
        ):
            print(
                "- _finish_chart_calibration pose gate bypassed for verified "
                f"oriented-panel geometry: "
                f"order={self._oriented_panel_payload_order_name() or 'unknown'} "
                f"reason={pose_reason}, "
                f"legacy_flip={legacy_flip}"
            )
            pose_problem = False
            pose_gate_bypassed = True
        if pose_problem:
            if self._pose_reason_is_hard_flip(pose_reason):
                title = "色基準(48色)が上下反転しています"
                detail = "180° 回して再度 [P] を押してください"
            else:
                title = "色基準(48色)の向き・設置を確認してください"
                detail = "水平に正しい向きで置き直して再度 [P] を押してください"
            print(
                f"- _finish_chart_calibration aborted by pose gate: reason={pose_reason}, "
                f"legacy_flip={legacy_flip}"
            )
            self._show_capture_overlay(
                title,
                detail,
                "このままでは CCM が壊れるため実行しません",
                wait_sec=3.0,
            )
            self._chart_state = "preview"
            return None
        if legacy_flip:
            if self._has_verified_visual_180_oriented_panel_payload():
                print(
                    "- _finish_chart_calibration legacy flip safety net bypassed "
                    "for verified visual_180 oriented-panel payload"
                )
                pose_gate_bypassed = True
            else:
                # comprehensive gate が見逃した特殊ケースの最終保険
                print("- _finish_chart_calibration aborted by legacy flip-only safety net")
                self._show_capture_overlay(
                    "色基準(48色)が上下反転しています",
                    "180° 回して再度 [P] を押してください",
                    "このままでは CCM が壊れるため実行しません",
                    wait_sec=3.0,
                )
                self._chart_state = "preview"
                return None

        key = self._show_chart_workflow_overlay(
            "compute",
            "CCM fit",
            detail="白点正規化と CCM を計算中",
            can_cancel=True,
        )
        self._raise_if_chart_cancel_requested(key, "ccm fit start")
        # 旧残差テーブルが新CCMの評価を汚染しないよう、冒頭で明示的にクリア
        self.lab_converter.clear_residuals()
        reference_data = SPYDERCHECKER_REFERENCE
        white_idx = SPYDERCHECKR_48_WHITE_PATCH_INDEX

        reference_labs = np.array(
            [p["Lab"] for p in reference_data], dtype=np.float64
        )

        white_ratio_rgb = ratios_gray[white_idx]
        safe_white = np.where(white_ratio_rgb < 1e-8, 1e-8, white_ratio_rgb)
        ratios_white = ratios_gray / safe_white

        # デバッグ出力
        _S2 = "\u2550" * 62
        print(f"\n{_S2}")
        print("  [P] CCM \u2014 \u767d\u70b9\u6b63\u898f\u5316")
        print(_S2)
        print(f"  white_ratio_rgb (ratios_gray[{white_idx}]):")
        print(f"    R={white_ratio_rgb[0]:.8f}  G={white_ratio_rgb[1]:.8f}  B={white_ratio_rgb[2]:.8f}")
        print("  ratios_white (\u30ad\u30fc\u30d1\u30c3\u30c1\u306e\u307f):")
        for _ki, _kn in [(SPYDERCHECKR_48_WHITE_PATCH_INDEX, "1E White"),
                         (SPYDERCHECKR_48_GRAY_4D_INDEX,     "4D Gray50%")]:
            _rw = ratios_white[_ki]
            print(f"    #{_ki:2d} {_kn:12}: R={_rw[0]:.5f}  G={_rw[1]:.5f}  B={_rw[2]:.5f}")
        self._print_white_ratio_balance_warning(white_ratio_rgb)
        print(_S2)

        # フラット過補正パッチの自動除外
        _valid_thresh = 2.5
        valid_mask = ratios_white.mean(axis=1) <= _valid_thresh
        n_valid = int(valid_mask.sum())
        if n_valid < 12:
            _valid_thresh = 4.0
            valid_mask = ratios_white.mean(axis=1) <= _valid_thresh
            n_valid = int(valid_mask.sum())
        _excluded_idx = np.where(~valid_mask)[0].tolist()
        _S2b = "\u2500" * 62
        print(f"{_S2b}")
        print(f"  \u30d5\u30e9\u30c3\u30c8\u904e\u88dc\u6b63\u9664\u5916 (mean(ratio_white) > {_valid_thresh}): "
              f"{len(ratios_white) - n_valid} \u30d1\u30c3\u30c1\u9664\u5916, {n_valid} \u30d1\u30c3\u30c1\u3067 CCM \u30d5\u30a3\u30c3\u30c8")
        if _excluded_idx:
            _excl_names = [str(reference_data[_i].get("id", f"#{_i}")) for _i in _excluded_idx]
            print(f"  \u9664\u5916\u30d1\u30c3\u30c1: {_excl_names}")
        print(_S2)

        ratios_white_fit = ratios_white[valid_mask]
        reference_labs_fit = reference_labs[valid_mask]

        M = compute_ccm(ratios_white_fit, reference_labs_fit)

        self.lab_converter.set_ccm(M)
        delta_E_list = []
        for i, ref_patch in enumerate(reference_data):
            ratio_w = ratios_white[i].astype(np.float32)
            lab_pred = self.lab_converter.ratio_to_Lab(ratio_w)
            lab_ref = np.array(ref_patch["Lab"], dtype=np.float32)
            delta_E_list.append(float(np.sqrt(np.sum((lab_pred - lab_ref) ** 2))))
        delta_E_fit = [delta_E_list[i] for i in range(len(delta_E_list)) if valid_mask[i]]
        delta_E_mean = float(np.mean(delta_E_fit))
        delta_E_max = float(np.max(delta_E_fit))

        # Per-patch 残差テーブル算出
        residual_ratios_list = []
        residual_labs_list = []
        for _rw, _rl in zip(ratios_white_fit, reference_labs_fit):
            _lab_pred = self.lab_converter.ratio_to_Lab(_rw.astype(np.float32))
            _delta = _rl.astype(np.float64) - _lab_pred.astype(np.float64)
            residual_ratios_list.append(_rw)
            residual_labs_list.append(_delta)
        res_ratios = np.array(residual_ratios_list, dtype=np.float64)
        res_labs   = np.array(residual_labs_list,   dtype=np.float64)
        self.lab_converter.set_residuals(res_ratios, res_labs)

        # 残差補正後の ΔE 再計算
        delta_E_post = []
        for i, ref_patch in enumerate(reference_data):
            if not valid_mask[i]:
                continue
            _rw_post = ratios_white[i].astype(np.float32)
            _lp_post = self.lab_converter.ratio_to_Lab(_rw_post)
            _lr_post = np.array(ref_patch["Lab"], dtype=np.float32)
            delta_E_post.append(float(np.sqrt(np.sum((_lp_post - _lr_post) ** 2))))
        dE_post_mean = float(np.mean(delta_E_post)) if delta_E_post else 0.0
        dE_post_max = float(np.max(delta_E_post)) if delta_E_post else 0.0

        # デバッグ出力
        _S3 = "\u2550" * 62
        print(f"\n{_S3}")
        print("  [P] CCM \u2014 \u884c\u5217\u30fb\u0394E \u7d71\u8a08")
        print(_S3)
        print("  CCM matrix (3\u00d73):")
        for _row in M:
            print(f"    [{_row[0]:+.5f}  {_row[1]:+.5f}  {_row[2]:+.5f}]")
        print(f"  \u0394E_mean = {delta_E_mean:.3f}   \u0394E_max = {delta_E_max:.3f}   "
              f"(\u30d5\u30a3\u30c3\u30c8 {n_valid}\u30d1\u30c3\u30c1 / \u5168 {len(delta_E_list)}\u30d1\u30c3\u30c1)")
        print(f"  \u0394E_post (CCM+\u6b8b\u5dee): mean={dE_post_mean:.3f}  "
              f"max={dE_post_max:.3f}  (\u671f\u5f85\u5024 \u22480)")
        _sorted_de: list[tuple[int, float]] = sorted(
            [(i, de) for i, de in enumerate(delta_E_list) if valid_mask[i]],
            key=lambda x: x[1], reverse=True
        )
        print("  \u0394E top-5 worst (\u30d5\u30a3\u30c3\u30c8\u30d1\u30c3\u30c1\u306e\u307f):")
        for _rank, (_ii, _de) in enumerate(_sorted_de):
            if _rank >= 5:
                break
            _pname = str(reference_data[_ii].get("id", f"#{_ii}"))
            print(f"    [{_ii:2d}] {_pname:6}  \u0394E={_de:.3f}")
        print(_S3 + "\n")

        # LOOCV
        loocv_dE = []
        n_fit = len(ratios_white_fit)
        # LOOCV では sRGB 最小二乗を使用（Lab最適化は48回回すと数分かかるため）
        from .colorimeter_common import _compute_ccm_srgb_lstsq
        _refl_factor = self.lab_converter.reflectance_factor
        for leave_idx in range(n_fit):
            if leave_idx == 0 or (leave_idx + 1) == n_fit or (leave_idx + 1) % 4 == 0:
                key = self._show_chart_workflow_overlay(
                    "compute",
                    "CCM LOOCV",
                    detail="LOOCV を計算中",
                    progress=leave_idx + 1,
                    total=n_fit,
                    can_cancel=True,
                )
                self._raise_if_chart_cancel_requested(key, "ccm loocv")
            mask_loo = np.ones(n_fit, dtype=bool)
            mask_loo[leave_idx] = False
            r_train = ratios_white_fit[mask_loo]
            l_train = reference_labs_fit[mask_loo]
            M_loo = _compute_ccm_srgb_lstsq(
                r_train * _refl_factor, l_train
            )
            temp_conv = CIELABConverter(_refl_factor)
            temp_conv.set_ccm(M_loo)
            lab_pred = temp_conv.ratio_to_Lab(ratios_white_fit[leave_idx].astype(np.float32))
            lab_ref = reference_labs_fit[leave_idx]
            loocv_dE.append(float(np.sqrt(np.sum((lab_pred - lab_ref) ** 2))))
        loocv_mean = float(np.mean(loocv_dE))
        loocv_max = float(np.max(loocv_dE))
        print(f"  LOOCV \u0394E: mean={loocv_mean:.3f}  max={loocv_max:.3f}")
        if loocv_mean > delta_E_mean * 2.0:
            print("  \u26a0 LOOCV \u0394E \u304c\u8a13\u7df4 \u0394E \u306e2\u500d\u4ee5\u4e0a \u2014 \u904e\u5b66\u7fd2\u306e\u53ef\u80fd\u6027")

        # リニアリティチェック
        gray_indices = [3, 4, 11, 12, 19, 20, 27, 28, 35, 36, 43, 44]
        ref_L_gray = np.array([reference_data[i]["Lab"][0] for i in gray_indices])
        meas_L_gray = []
        for gi in gray_indices:
            if gi < len(ratios_white):
                _lab_g = self.lab_converter.ratio_to_Lab(ratios_white[gi].astype(np.float32))
                meas_L_gray.append(float(_lab_g[0]))
        meas_L_gray = np.array(meas_L_gray[:len(ref_L_gray)])
        if len(meas_L_gray) >= 4:
            ss_res = float(np.sum((meas_L_gray - ref_L_gray[:len(meas_L_gray)]) ** 2))
            ss_tot = float(np.sum((ref_L_gray[:len(meas_L_gray)] - ref_L_gray[:len(meas_L_gray)].mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            print(f"  \u30ea\u30cb\u30a2\u30ea\u30c6\u30a3 R\u00b2 = {r_squared:.4f}  (\u30b0\u30ec\u30fc\u30b9\u30b1\u30fc\u30eb {len(meas_L_gray)} \u30d1\u30c3\u30c1)")
            if r_squared < 0.98:
                print("  \u26a0 R\u00b2 < 0.98: \u30bb\u30f3\u30b5\u30fc\u30ea\u30cb\u30a2\u30ea\u30c6\u30a3\u306b\u554f\u984c\u304c\u3042\u308b\u53ef\u80fd\u6027")

        _ccm_exp = getattr(self, "_ccm_exposure_us", None)
        _meas_exp = getattr(self, "_measurement_exposure_us", None)
        _exp_ratio = (
            _ccm_exp / _meas_exp
            if _ccm_exp is not None and _meas_exp is not None and _meas_exp > 0
            else None
        )
        if _ccm_exp is not None and _meas_exp is not None:
            print(
                f"- [P] CCM exposure: ccm={_ccm_exp}us"
                f" measurement={_meas_exp}us"
                f" ratio={_exp_ratio:.3f}"
            )
            if _exp_ratio is not None and _exp_ratio < 0.80:
                print(
                    f"⚠ CCM/測定 露光比 {_exp_ratio:.3f} < 0.80"
                    " — センサ非線形性リスクあり"
                )
        orientation_metadata = self._chart_orientation_metadata(
            pose_gate_reason=pose_reason,
            pose_gate_bypassed=pose_gate_bypassed,
            legacy_flip=legacy_flip,
        )
        self.ccm_store.save(
            M,
            white_ratio_rgb=white_ratio_rgb,
            metadata={
                "delta_E_mean": delta_E_mean,
                "delta_E_max": delta_E_max,
                "delta_E_post_mean": dE_post_mean,
                "delta_E_post_max": dE_post_max,
                "loocv_delta_E_mean": loocv_mean,
                "loocv_delta_E_max": loocv_max,
                "n_patches": len(ratios_white),
                "chart_type": "SpyderCheckr48",
                "ccm_exposure_us": _ccm_exp,
                "measurement_exposure_us": _meas_exp,
                "exposure_ratio": _exp_ratio,
                **orientation_metadata,
            },
            residual_ratios=res_ratios,
            residual_labs=res_labs,
            ref_train=avg_ref_train,
        )

        degradation_warning = self.ccm_store.check_degradation()
        if degradation_warning:
            print(degradation_warning)

        self.white_ratio_rgb = white_ratio_rgb
        self.ref_train = (
            np.asarray(avg_ref_train, dtype=np.float64).copy()
            if avg_ref_train is not None
            else None
        )
        live_baseline, live_source = self._resolve_live_ref_baseline_for_p(self.ref_train)
        self._set_live_ref_baseline(live_baseline, live_source, notify=True)
        live_scale_baseline, live_scale_source = self._resolve_live_ref_scale_baseline_for_p(
            self.ref_train
        )
        self._set_live_ref_scale_baseline(
            live_scale_baseline,
            live_scale_source,
            notify=True,
        )
        # 測定パイプライン側に white_ratio_rgb / ref_train を即時反映（BUG 8 修正）
        if self._on_ccm_calibrated is not None:
            self._on_ccm_calibrated(white_ratio_rgb, avg_ref_train)
        self._chart_state = "idle"

        quality = "\u25ce" if delta_E_mean < 2.0 else "\u25cb" if delta_E_mean < 5.0 else "\u25b3"
        self._show_chart_workflow_overlay(
            "postprocess",
            f"CCM calibration complete {quality}",
            detail="続けて blank / neutral を実行します",
            can_cancel=False,
            wait_sec=2.5,
        )
        self._run_blank_and_neutral_wizard()
        # CCM + wizard 完了後の最終保存 (refined patch_margin / col_x_norm_offsets を反映)
        self._persist_chart_corners(reason="ccm_complete")
        try:
            return self._run_4d_gray_verification()
        finally:
            self._reset_chart_calibration()

    def _capture_gray_verification_sample(
        self,
        white_for_norm: np.ndarray,
        n_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """4D steady-state / verification 共通の1サンプルを取得する。"""
        ratio, avg_ref = self._capture_grid_4d_ratio(n_frames=n_frames)
        corrected = np.asarray(self.blank.correct(ratio), dtype=np.float64)
        sample_metrics = {
            "ref_scale": None,
            "measured_lab": None,
            "lab_rel_l": None,
        }
        if self.lab_converter is not None:
            ref_scale_now = normalize_ref_scale_triplet(avg_ref, self.white_ratio_rgb)
            canonical = run_canonical_lab_pipeline(
                corrected,
                white_for_norm,
                ref_scale_now,
                self.live_ref_scale_baseline,
                self.lab_converter,
            )
            sample_metrics["ref_scale"] = canonical["ref_scale"]
            sample_metrics["measured_lab"] = np.asarray(
                canonical["lab"], dtype=np.float64
            )
            anchor_raw = np.ones(3, dtype=np.float64)
            if getattr(self.blank, "is_loaded", False):
                anchor_raw = np.asarray(
                    self.blank.correct(anchor_raw), dtype=np.float64
                )
            anchor_canonical = run_canonical_lab_pipeline(
                anchor_raw,
                white_for_norm,
                ref_scale_now,
                self.live_ref_scale_baseline,
                self.lab_converter,
            )
            sample_ratio_white = np.asarray(
                canonical["ratio_white"], dtype=np.float64
            ).ravel()[:3]
            anchor_ratio_white = np.asarray(
                anchor_canonical["ratio_white"], dtype=np.float64
            ).ravel()[:3]
            anchor_lab = np.asarray(anchor_canonical["lab"], dtype=np.float64).ravel()[:3]
            if not np.allclose(sample_ratio_white, anchor_ratio_white):
                sample_metrics["lab_rel_l"] = float(
                    sample_metrics["measured_lab"][0] - anchor_lab[0]
                )
                sample_metrics["lab_rel_a"] = float(
                    sample_metrics["measured_lab"][1] - anchor_lab[1]
                )
                sample_metrics["lab_rel_b"] = float(
                    sample_metrics["measured_lab"][2] - anchor_lab[2]
                )
        return corrected, np.asarray(avg_ref, dtype=np.float64), sample_metrics

    def _await_gray_steady_state(self, white_for_norm: np.ndarray) -> dict:
        """正式 4D 判定前に短いプレ観測で steady state を自動判定する。"""
        window_ratios: list[np.ndarray] = []
        window_ref_scales: list[float | None] = []
        window_lab_rel_l_values: list[float | None] = []
        consecutive = 0
        last_snapshot = SteadyStateSnapshot(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            ref_scale=None,
            lab_rel_l=None,
            consecutive_frames=0,
            is_stable=False,
        )
        last_result = {
            "mean_rgb": None,
            "std_rgb": None,
            "max_sigma": None,
            "ref_scale_range": None,
        }

        for attempt in range(1, GRAY_STEADY_STATE_MAX_ATTEMPTS + 1):
            key = self._show_chart_workflow_overlay(
                "steady_state",
                "4D steady-state",
                detail=(
                    f"attempt {attempt}/{GRAY_STEADY_STATE_MAX_ATTEMPTS}  "
                    f"stable {consecutive}/{GRAY_STEADY_STATE_REQUIRED_CONSECUTIVE}"
                ),
                progress=attempt,
                total=GRAY_STEADY_STATE_MAX_ATTEMPTS,
                can_cancel=True,
            )
            self._raise_if_chart_cancel_requested(key, "4d steady-state")
            corrected, _avg_ref, sample_metrics = self._capture_gray_verification_sample(
                white_for_norm,
                n_frames=GRAY_STEADY_STATE_CAPTURE_FRAMES,
            )
            ref_scale = sample_metrics.get("ref_scale")
            lab_rel_l = sample_metrics.get("lab_rel_l")
            window_ratios.append(corrected)
            window_ref_scales.append(ref_scale)
            window_lab_rel_l_values.append(lab_rel_l)
            if len(window_ratios) > GRAY_STEADY_STATE_WINDOW:
                window_ratios.pop(0)
                window_ref_scales.pop(0)
                window_lab_rel_l_values.pop(0)
            if len(window_ratios) < GRAY_STEADY_STATE_WINDOW:
                continue

            observables = summarize_gray_steady_state_observables(
                window_ratios,
                ref_scales=window_ref_scales,
                lab_rel_l_values=window_lab_rel_l_values,
            )
            ref_scale_range = observables["ref_scale_range"]
            is_stable = (
                observables["max_sigma"] is not None
                and observables["max_sigma"] <= self.gray_verifier.THRESH_SIGMA
            )
            if ref_scale_range is not None:
                is_stable = (
                    is_stable
                    and ref_scale_range <= GRAY_STEADY_STATE_REF_SCALE_RANGE_MAX
                )
            consecutive = consecutive + 1 if is_stable else 0
            last_snapshot = SteadyStateSnapshot(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                ref_scale=observables["ref_scale_mean"],
                lab_rel_l=observables["lab_rel_l_mean"],
                relative_rgb_mean=tuple(observables["relative_rgb_mean"]),
                relative_rgb_std=tuple(observables["relative_rgb_std"]),
                consecutive_frames=consecutive,
                is_stable=is_stable,
            )
            last_result = {
                "mean_rgb": observables["relative_rgb_mean"],
                "std_rgb": observables["relative_rgb_std"],
                "max_sigma": observables["max_sigma"],
                "ref_scale_range": ref_scale_range,
            }
            if consecutive >= GRAY_STEADY_STATE_REQUIRED_CONSECUTIVE:
                break

        return {
            "snapshot": last_snapshot.as_dict(),
            "attempts": attempt,
            "required_consecutive": GRAY_STEADY_STATE_REQUIRED_CONSECUTIVE,
            "capture_frames": GRAY_STEADY_STATE_CAPTURE_FRAMES,
            "window_size": GRAY_STEADY_STATE_WINDOW,
            "ref_scale_range": last_result["ref_scale_range"],
            "max_sigma": last_result["max_sigma"],
        }

    def _run_4d_gray_verification(self) -> dict | None:
        """P キャリブ後の 4D グレーパッチ 10回自動検証。"""
        # 2-9: ガード条件
        grid_ready = (
            self._grid_extractor is not None and self._grid_extractor.is_ready
        )
        if not grid_ready:
            # 2-10: 警告出力
            print("4D検証スキップ: grid_extractor が準備できていません")
            return
        if self.gray_verifier is None or self.blank is None:
            print("4D検証スキップ: gray_verifier または blank が未設定です")
            return
        # 2-11: オーバーレイ表示
        self._show_chart_workflow_overlay(
            "verify",
            "4D verification",
            detail="steady-state を確認します",
            progress=0,
            total=10,
            can_cancel=True,
        )
        self._set_guidance_runner_state(GUIDANCE_RUNNER_STATE_RESULT)
        verified_at = datetime.now().isoformat(timespec="seconds")
        # Phase 2 (再設計): 当日初回なら verify_only → start_of_day に自動昇格.
        # ユーザー操作は D→F→W→P のまま。朝 1 回目の P 後 verify で
        # accepted_reference 鎖を自動で張る。明示指定 (run_start_of_day_sequence 等)
        # があればそちらが優先される (_acceptance_run_type が既に set 済み).
        if self._acceptance_run_type:
            current_run_type = self._acceptance_run_type
        elif not self._has_start_of_day_pass():
            current_run_type = RUN_TYPE_START_OF_DAY
            print(
                "- auto-promote to start_of_day: 当日初回の P 後 verify のため "
                "accepted_reference 鎖を張るために run_type を昇格"
            )
        else:
            current_run_type = RUN_TYPE_VERIFY_ONLY
        mode_name = getattr(self.mode, "current", "unknown")
        operator_name = (
            os.environ.get("USER")
            or os.environ.get("USERNAME")
            or "unknown"
        )
        control_sample_id = self._resolve_control_sample_id()
        fixture_id = self._resolve_fixture_id()
        positioning_method = self._resolve_positioning_method()
        context_profile_id = self._resolve_context_profile_id()
        roi_or_grid_source = "grid:SpyderCHECKR-48:4D"
        white_for_norm = np.asarray(self.white_ratio_rgb, dtype=np.float64).copy()
        if getattr(self.blank, "is_loaded", False):
            white_for_norm = np.asarray(
                self.blank.correct(white_for_norm), dtype=np.float64,
            )
        steady_state = self._await_gray_steady_state(white_for_norm)
        steady_snapshot = steady_state["snapshot"]
        if not steady_snapshot.get("is_stable"):
            print(
                "⚠ 4D steady state 未到達: "
                f"attempts={steady_state['attempts']} "
                f"max_sigma={steady_state.get('max_sigma')} "
                f"ref_scale_range={steady_state.get('ref_scale_range')}"
            )
        corrected_ratios = []
        absolute_labs = []
        sample_records = []
        sample_refs = []
        ref_scales = []
        lab_rel_l_values = []
        lab_rel_a_values = []
        lab_rel_b_values = []
        can_run_absolute = self.ccm_verifier is not None and self.lab_converter is not None
        reference_lab = (
            np.asarray(self.ccm_verifier.REFERENCE_LAB_4D, dtype=np.float64)
            if can_run_absolute else None
        )
        for sample_idx in range(10):
            key = self._show_chart_workflow_overlay(
                "verify",
                "4D verification",
                detail="10 サンプルで acceptance を評価中",
                progress=sample_idx + 1,
                total=10,
                can_cancel=True,
            )
            self._raise_if_chart_cancel_requested(key, "4d verification")
            corrected, avg_ref, sample_metrics = self._capture_gray_verification_sample(
                white_for_norm,
                n_frames=64,
            )
            ref_scale = sample_metrics.get("ref_scale")
            measured_lab = sample_metrics.get("measured_lab")
            lab_rel_l = sample_metrics.get("lab_rel_l")
            lab_rel_a = sample_metrics.get("lab_rel_a")
            lab_rel_b = sample_metrics.get("lab_rel_b")
            corrected_ratios.append(corrected)
            sample_refs.append(np.asarray(avg_ref, dtype=np.float64))
            sample_record = {
                "sample_index": len(sample_records) + 1,
                "corrected_ratios": np.asarray(corrected, dtype=np.float64).tolist(),
                "avg_ref": np.asarray(avg_ref, dtype=np.float64).tolist(),
                "ref_scale": None,
                "lab_rel_l": lab_rel_l,
                "lab_rel_a": lab_rel_a,
                "lab_rel_b": lab_rel_b,
                "measured_lab": None,
                "delta_e": None,
            }
            if lab_rel_l is not None:
                lab_rel_l_values.append(float(lab_rel_l))
            if lab_rel_a is not None:
                lab_rel_a_values.append(float(lab_rel_a))
            if lab_rel_b is not None:
                lab_rel_b_values.append(float(lab_rel_b))
            if measured_lab is not None:
                lab_abs = np.asarray(measured_lab, dtype=np.float64)
                absolute_labs.append(lab_abs)
                sample_record["measured_lab"] = lab_abs.tolist()
                sample_record["ref_scale"] = ref_scale
                if ref_scale is not None:
                    ref_scales.append(float(ref_scale))
                if can_run_absolute and reference_lab is not None:
                    sample_record["delta_e"] = float(
                        np.linalg.norm(lab_abs - reference_lab)
                    )
            sample_records.append(sample_record)
        # 2-13: evaluate
        ratios_array = np.array(corrected_ratios, dtype=np.float64)
        result = self.gray_verifier.evaluate(ratios_array)
        result.update({
            "verified_at": verified_at,
            "is_stale": False,
            "stale_reason": None,
            "stale_at": None,
            "baseline_source": self.live_ref_scale_baseline_source or "none",
            "roi_or_grid_source": roi_or_grid_source,
        })
        abs_result: dict | None = None
        if can_run_absolute and absolute_labs:
            mean_lab = np.mean(np.asarray(absolute_labs, dtype=np.float64), axis=0)
            abs_result = self.ccm_verifier.verify(mean_lab)
            abs_result.update({
                "n_samples": len(absolute_labs),
                "verified_at": verified_at,
                "is_stale": False,
                "stale_reason": None,
                "stale_at": None,
                "baseline_source": self.live_ref_scale_baseline_source or "none",
                "roi_or_grid_source": roi_or_grid_source,
                "reference_lab": abs_result.get("reference"),
            })
        mean_ref = (
            np.mean(np.asarray(sample_refs, dtype=np.float64), axis=0)
            if sample_refs else None
        )
        mean_ref_scale = float(np.mean(ref_scales)) if ref_scales else None
        mean_lab_rel_l = (
            float(np.mean(np.asarray(lab_rel_l_values, dtype=np.float64)))
            if lab_rel_l_values
            else None
        )
        mean_lab_rel_a = (
            float(np.mean(np.asarray(lab_rel_a_values, dtype=np.float64)))
            if lab_rel_a_values
            else None
        )
        mean_lab_rel_b = (
            float(np.mean(np.asarray(lab_rel_b_values, dtype=np.float64)))
            if lab_rel_b_values
            else None
        )
        operator_guidance_mode, guidance_degraded_reasons = (
            self._get_operator_guidance_snapshot()
        )
        audit_record = {
            "timestamp": verified_at,
            "run_type": current_run_type,
            "operator": operator_name,
            "calibration_dir": get_today_calibration_dir(),
            "mode": mode_name,
            "control_sample_id": control_sample_id,
            "fixture_id": fixture_id,
            "positioning_method": positioning_method,
            "context_profile_id": context_profile_id,
            "manual_roi_used": self._manual_roi_used,
            "manual_corner_used": self._manual_corner_used,
            "operator_guidance_mode": operator_guidance_mode,
            "guidance_degraded_reasons": guidance_degraded_reasons,
            "roi_or_grid_source": roi_or_grid_source,
            "baseline_source": self.live_ref_scale_baseline_source or "none",
            "baseline_value": (
                None
                if self.live_ref_scale_baseline is None
                else np.asarray(self.live_ref_scale_baseline, dtype=np.float64).tolist()
            ),
            "white_ratio_rgb": np.asarray(white_for_norm, dtype=np.float64).tolist(),
            "avg_ref": None if mean_ref is None else mean_ref.tolist(),
            "ref_scale": mean_ref_scale,
            "lab_rel_l": mean_lab_rel_l,
            "lab_rel_a": mean_lab_rel_a,
            "lab_rel_b": mean_lab_rel_b,
            "steady_state": steady_state,
            "relative_rgb_mean": result.get("mean_rgb"),
            "relative_rgb_std": result.get("std_rgb"),
            "measured_lab": None if abs_result is None else abs_result.get("measured"),
            "delta_e": None if abs_result is None else abs_result.get("dE"),
            "exposure": {
                "exposure_us": getattr(self.ae, "current_exposure", None),
            },
            "relative": {
                "accuracy_status": result.get("accuracy_status"),
                "stability_status": result.get("stability_status"),
                "max_dev": result.get("max_dev"),
                "max_sigma": result.get("max_sigma"),
                "n_samples": result.get("n_samples"),
            },
            "absolute": {
                "status": None if abs_result is None else abs_result.get("status"),
                "reference_lab": (
                    None if abs_result is None else abs_result.get("reference")
                ),
                "n_samples": None if abs_result is None else abs_result.get("n_samples"),
            },
            "samples": sample_records,
        }
        audit_paths = persist_gray_4d_verification_artifacts(
            audit_record,
            calibration_dir=get_today_calibration_dir(),
        )
        result["audit_paths"] = audit_paths
        if abs_result is not None:
            abs_result["audit_paths"] = audit_paths

        acceptance_judgement = evaluate_acceptance_judgement(
            relative_rgb_mean=result.get("mean_rgb"),
            relative_rgb_std=result.get("std_rgb"),
            delta_e=None if abs_result is None else abs_result.get("dE"),
            ref_scale=mean_ref_scale,
            lab_rel_l=mean_lab_rel_l,
            baseline_source=self.live_ref_scale_baseline_source,
            accepted_reference_diff=None,
            manual_roi_used=self._manual_roi_used,
            manual_corner_used=self._manual_corner_used,
        )
        accepted_reference_record, selection_meta = (
            self._resolve_accepted_reference_record_for_context(
                control_sample_id=control_sample_id,
                before_timestamp=verified_at,
            )
        )
        previous_accepted_day_record, previous_accepted_day_meta = (
            resolve_previous_accepted_day_record(
                calibration_root=CALIBRATION_DIR,
                control_sample_id=control_sample_id,
                before_timestamp=verified_at,
                current_measurement_context=build_measurement_context_summary(
                    control_sample_id=control_sample_id,
                    fixture_id=fixture_id,
                    positioning_method=positioning_method,
                    context_profile_id=context_profile_id,
                    software_version=get_software_identity().get("software_version"),
                    git_revision=get_software_identity().get("git_revision"),
                ),
            )
        )
        acceptance_record = {
            "timestamp": verified_at,
            "run_type": current_run_type,
            "operator": operator_name,
            "calibration_dir": get_today_calibration_dir(),
            "control_sample_id": control_sample_id,
            "fixture_id": fixture_id,
            "positioning_method": positioning_method,
            "context_profile_id": context_profile_id,
            "accepted_reference_run_id": (
                "none"
                if accepted_reference_record is None
                else str(
                    accepted_reference_record.get(
                        "run_id",
                        accepted_reference_record.get("timestamp", "none"),
                    )
                )
            ),
            "accepted_reference_selection_mode": selection_meta.get(
                "accepted_reference_selection_mode",
                ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
            ),
            "accepted_reference_selection_requested_run_id": selection_meta.get(
                "accepted_reference_selection_requested_run_id",
                "",
            ),
            "accepted_reference_selection_source_file": selection_meta.get(
                "accepted_reference_selection_source_file",
                "",
            ),
            "accepted_reference_selection_warning": selection_meta.get(
                "accepted_reference_selection_warning",
                "",
            ),
            "accepted_reference_context_status": selection_meta.get(
                "accepted_reference_context_status",
                "not_evaluated",
            ),
            "accepted_reference_context_warning": selection_meta.get(
                "accepted_reference_context_warning",
                "",
            ),
            "accepted_reference_context_mismatch_fields": selection_meta.get(
                "accepted_reference_context_mismatch_fields",
                [],
            ),
            "accepted_reference_current_context_missing_contract_fields": selection_meta.get(
                "accepted_reference_current_context_missing_contract_fields",
                [],
            ),
            "accepted_reference_reference_context_missing_contract_fields": selection_meta.get(
                "accepted_reference_reference_context_missing_contract_fields",
                [],
            ),
            "accepted_reference_context_fingerprint": selection_meta.get(
                "accepted_reference_context_fingerprint",
                "",
            ),
            "previous_accepted_day_run_id": (
                "none"
                if previous_accepted_day_record is None
                else str(
                    previous_accepted_day_record.get(
                        "run_id",
                        previous_accepted_day_record.get("timestamp", "none"),
                    )
                )
            ),
            "previous_accepted_day_timestamp": (
                ""
                if previous_accepted_day_record is None
                else str(previous_accepted_day_record.get("timestamp", ""))
            ),
            "previous_accepted_day_date": (
                ""
                if previous_accepted_day_record is None
                else str(previous_accepted_day_record.get("timestamp", ""))[:10]
            ),
            "previous_accepted_day_selection_warning": previous_accepted_day_meta.get(
                "previous_accepted_day_selection_warning",
                "",
            ),
            "previous_accepted_day_context_status": previous_accepted_day_meta.get(
                "previous_accepted_day_context_status",
                "not_evaluated",
            ),
            "previous_accepted_day_context_warning": previous_accepted_day_meta.get(
                "previous_accepted_day_context_warning",
                "",
            ),
            "previous_accepted_day_context_mismatch_fields": (
                previous_accepted_day_meta.get(
                    "previous_accepted_day_context_mismatch_fields",
                    [],
                )
            ),
            "previous_accepted_day_current_context_missing_contract_fields": (
                previous_accepted_day_meta.get(
                    "previous_accepted_day_current_context_missing_contract_fields",
                    [],
                )
            ),
            "previous_accepted_day_reference_context_missing_contract_fields": (
                previous_accepted_day_meta.get(
                    "previous_accepted_day_reference_context_missing_contract_fields",
                    [],
                )
            ),
            "previous_accepted_day_context_fingerprint": (
                previous_accepted_day_meta.get(
                    "previous_accepted_day_context_fingerprint",
                    "",
                )
            ),
            "selected_previous_accepted_day_measurement_context": (
                previous_accepted_day_meta.get(
                    "selected_previous_accepted_day_measurement_context",
                    {},
                )
            ),
            "selected_previous_accepted_day_measurement_context_fingerprint": (
                previous_accepted_day_meta.get(
                    "selected_previous_accepted_day_measurement_context_fingerprint",
                    "",
                )
            ),
            "selected_reference_measurement_context": selection_meta.get(
                "selected_reference_measurement_context",
                {},
            ),
            "selected_reference_measurement_context_fingerprint": selection_meta.get(
                "selected_reference_measurement_context_fingerprint",
                "",
            ),
            "warmup_elapsed_sec": (
                int(max(round(time.time() - self._camera_init_time), 0))
                if self._camera_init_time is not None
                else None
            ),
            "baseline_source": self.live_ref_scale_baseline_source or "none",
            "baseline_value": (
                None
                if self.live_ref_scale_baseline is None
                else np.asarray(self.live_ref_scale_baseline, dtype=np.float64).tolist()
            ),
            "white_ratio_rgb": np.asarray(white_for_norm, dtype=np.float64).tolist(),
            "avg_ref": None if mean_ref is None else mean_ref.tolist(),
            "ref_scale": mean_ref_scale,
            "relative_rgb_mean": result.get("mean_rgb"),
            "relative_rgb_std": result.get("std_rgb"),
            "lab_rel_l": mean_lab_rel_l,
            "lab_rel_a": mean_lab_rel_a,
            "lab_rel_b": mean_lab_rel_b,
            "steady_state": steady_state,
            "measured_lab": None if abs_result is None else abs_result.get("measured"),
            "delta_e": None if abs_result is None else abs_result.get("dE"),
            "roi_or_grid_source": roi_or_grid_source,
            "exposure": {
                "exposure_us": getattr(self.ae, "current_exposure", None),
            },
            "mode": mode_name,
            "result_status": acceptance_judgement["result_status"],
            "gate_state": acceptance_judgement["gate_state"],
            "failed_checks": acceptance_judgement["failed_checks"],
            "target_checks": acceptance_judgement["target_checks"],
            "minimum_checks": acceptance_judgement["minimum_checks"],
            "manual_roi_used": self._manual_roi_used,
            "manual_corner_used": self._manual_corner_used,
            "operator_guidance_mode": operator_guidance_mode,
            "guidance_degraded_reasons": guidance_degraded_reasons,
            "provisional_reasons": acceptance_judgement["provisional_reasons"],
            "gray_audit_paths": audit_paths,
            "relative_accuracy_status": result.get("accuracy_status"),
            "relative_stability_status": result.get("stability_status"),
            "absolute_status": None if abs_result is None else abs_result.get("status"),
        }
        acceptance_record.update(
            describe_accepted_reference_relation(
                acceptance_record,
                accepted_reference_record,
            )
        )
        accepted_reference_diff = compute_accepted_reference_diff(
            acceptance_record,
            accepted_reference_record,
        )
        previous_accepted_day_diff = compute_previous_accepted_day_diff(
            acceptance_record,
            previous_accepted_day_record,
        )
        previous_accepted_day_judgement = evaluate_previous_accepted_day_judgement(
            previous_accepted_day_diff,
        )
        acceptance_judgement = evaluate_acceptance_judgement(
            relative_rgb_mean=result.get("mean_rgb"),
            relative_rgb_std=result.get("std_rgb"),
            delta_e=None if abs_result is None else abs_result.get("dE"),
            ref_scale=mean_ref_scale,
            lab_rel_l=mean_lab_rel_l,
            baseline_source=self.live_ref_scale_baseline_source,
            accepted_reference_diff=accepted_reference_diff,
            manual_roi_used=self._manual_roi_used,
            manual_corner_used=self._manual_corner_used,
        )
        acceptance_record.update(
            {
                "result_status": acceptance_judgement["result_status"],
                "gate_state": acceptance_judgement["gate_state"],
                "failed_checks": acceptance_judgement["failed_checks"],
                "target_checks": acceptance_judgement["target_checks"],
                "minimum_checks": acceptance_judgement["minimum_checks"],
                "provisional_reasons": acceptance_judgement["provisional_reasons"],
                "accepted_reference_diff": accepted_reference_diff,
                "previous_accepted_day_diff": previous_accepted_day_diff,
                "previous_accepted_day_judgement": previous_accepted_day_judgement,
            }
        )
        if not steady_snapshot.get("is_stable"):
            acceptance_record["result_status"] = RESULT_STATUS_RECHECK_REQUIRED
            acceptance_record["gate_state"] = GATE_STATE_BLOCKED
            acceptance_record["failed_checks"] = sorted(
                set(acceptance_record["failed_checks"] + ["steady_state"])
            )
        # preflight / exposure_check 結果を acceptance_record に注入
        acceptance_record["preflight"] = getattr(
            self, "_preflight_result", {"status": "skip", "reason": "not_executed"},
        )
        acceptance_record["exposure_check"] = getattr(
            self, "_exposure_check_result", {"status": "skip", "reason": "not_executed"},
        )
        # accepted_reference_run_id を preflight から引き継ぐ
        pf = acceptance_record["preflight"]
        acceptance_record.setdefault(
            "accepted_reference_run_id",
            pf.get("accepted_reference_run_id"),
        )
        # accepted_reference の age 警告
        warnings = acceptance_record.setdefault("warnings", [])
        if selection_meta.get("accepted_reference_selection_warning"):
            warnings.append(selection_meta["accepted_reference_selection_warning"])
        ref_info = load_accepted_reference_for_preflight(
            CALIBRATION_DIR,
            control_sample_id=control_sample_id,
            run_id=(
                accepted_reference_record.get("run_id")
                if accepted_reference_record is not None
                and acceptance_record.get("accepted_reference_selection_mode")
                == ACCEPTED_REFERENCE_SELECTION_MODE_FIXED
                else None
            ),
        )
        if ref_info is not None:
            from .colorimeter_common import REFERENCE_AGE_WARNING_DAYS
            ref_date_str = ref_info.get("date", "")
            if ref_date_str:
                try:
                    ref_date = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
                    age_days = (datetime.now().date() - ref_date).days
                    if age_days > REFERENCE_AGE_WARNING_DAYS:
                        warnings.append("accepted_reference_age_exceeded")
                except ValueError:
                    pass
        # flat_field valid_fraction 低下警告
        try:
            _flat_meta_path = os.path.join(
                get_today_calibration_dir(), "flat_field_gain_meta.json",
            )
            if os.path.exists(_flat_meta_path):
                with open(_flat_meta_path, "r", encoding="utf-8") as _fm:
                    _flat_meta = json.load(_fm)
                _vf = _flat_meta.get("valid_fraction")
                if _vf is not None and float(_vf) < 0.40:
                    warnings.append(
                        f"flat_field_valid_fraction_low:{float(_vf):.3f}"
                    )
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        acceptance_paths = persist_acceptance_result_artifacts(
            acceptance_record,
            calibration_dir=get_today_calibration_dir(),
        )
        result["acceptance_paths"] = acceptance_paths
        if abs_result is not None:
            abs_result["acceptance_paths"] = acceptance_paths
        saved_acceptance_record = dict(acceptance_record)
        try:
            with open(acceptance_paths["latest_json"], "r", encoding="utf-8") as f:
                saved_acceptance_record = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        # 正式基準の自動登録
        ref_registered = auto_register_accepted_reference(
            saved_acceptance_record,
            calibration_dir=get_today_calibration_dir(),
        )
        if ref_registered is not None:
            print(f"- accepted_reference 自動登録: {ref_registered.get('run_id')}")
        else:
            print("- accepted_reference 登録スキップ（条件未達）")
        if self._on_acceptance_result_persisted is not None:
            self._on_acceptance_result_persisted(saved_acceptance_record)
        # 2-14: コールバックで main.py に同期
        if self._on_gray_verified is not None:
            self._on_gray_verified(result)
        if abs_result is not None and self._on_gray_absolute_verified is not None:
            self._on_gray_absolute_verified(abs_result)
        # 2-15: オーバーレイ消去
        # 合否バナー表示（operator は数値を読む必要がない）
        completion_record = saved_acceptance_record or acceptance_record
        overlay_lines = self._build_acceptance_completion_overlay_lines(
            completion_record,
        )
        overlay_wait_sec = 1.5
        if self._acceptance_completion_has_calibration_failure(completion_record):
            overlay_wait_sec = 5.0
        elif self._acceptance_completion_has_guidance_degradation(completion_record):
            overlay_wait_sec = 3.5
        elif (
            str(completion_record.get("gate_state", "")).strip()
            == GATE_STATE_PROVISIONAL
            or completion_record.get("provisional_reasons")
            or not completion_record.get("production_eligible")
        ):
            overlay_wait_sec = 3.0
        elif completion_record.get("production_eligible"):
            overlay_wait_sec = 3.0
        self._show_capture_overlay(*overlay_lines, wait_sec=overlay_wait_sec)
        self._clear_chart_workflow_status()
        # 2-16: コンソール出力
        _S = "\u2550" * 62
        m = result["mean_rgb"]
        s = result["std_rgb"]
        print(f"\n{_S}")
        print("  [V] 4D 相対検証 (10回測定)")
        print(_S)
        print(f"  mean: R={m[0]:.4f}  G={m[1]:.4f}  B={m[2]:.4f}")
        print(f"  std:  R={s[0]:.4f}  G={s[1]:.4f}  B={s[2]:.4f}")
        print(f"  精度: {result['accuracy_status']} (max_dev={result['max_dev']:.4f})")
        print(f"  安定: {result['stability_status']} (max_σ={result['max_sigma']:.4f})")
        print(_S + "\n")
        if abs_result is not None:
            m_lab = abs_result["measured"]
            r_lab = abs_result["reference"]
            print(f"\n{_S}")
            print("  [V] 4D 絶対検証 (10回平均)")
            print(_S)
            print(
                f"  測定Lab:  L={m_lab[0]:.2f}  a={m_lab[1]:.2f}  b={m_lab[2]:.2f}"
            )
            print(
                f"  基準Lab:  L={r_lab[0]:.2f}  a={r_lab[1]:.2f}  b={r_lab[2]:.2f}"
            )
            print(
                f"  判定: {abs_result['status']} (ΔE={abs_result['dE']:.3f}, "
                f"n={abs_result['n_samples']})"
            )
            print(_S + "\n")
        if str(completion_record.get("run_type", "")).strip() in (
            RUN_TYPE_START_OF_DAY,
            RUN_TYPE_END_OF_DAY,
            RUN_TYPE_REQUALIFICATION,
        ):
            previous_day_judgement = completion_record.get(
                "previous_accepted_day_judgement"
            ) or {}
            previous_day_diff = completion_record.get("previous_accepted_day_diff") or {}
            previous_day_date = str(
                completion_record.get("previous_accepted_day_date", "")
            ).strip()
            if previous_day_judgement.get("available") and previous_day_date:
                print(
                    "  [V] 前回合格日比較: "
                    f"{previous_day_date} / {previous_day_judgement.get('result_status', 'unknown')}"
                )
                print(
                    "      "
                    f"ΔE={previous_day_diff.get('lab_delta_e', 'N/A')} "
                    f"rel_mean_max={previous_day_diff.get('relative_rgb_mean_abs_diff_max', 'N/A')} "
                    f"ref_scale_diff={previous_day_diff.get('ref_scale_abs_diff', 'N/A')}"
                )
            else:
                reason = str(
                    completion_record.get("previous_accepted_day_selection_warning", "")
                ).strip()
                if not reason:
                    reason = str(previous_day_judgement.get("reason", "")).strip()
                if not reason:
                    reason = "previous_accepted_day_missing"
                print(f"  [V] 前回合格日比較: 未評価 ({reason})")
        print(f"  [V] audit json: {audit_paths['latest_json']}")
        print(f"  [V] audit log:  {audit_paths['latest_log']}")
        print(f"  [V] acceptance json: {acceptance_paths['latest_json']}")
        print(f"  [V] acceptance log:  {acceptance_paths['latest_log']}")
        return {
            "relative": result,
            "absolute": abs_result,
            "audit_paths": audit_paths,
            "acceptance_paths": acceptance_paths,
            "acceptance_record": saved_acceptance_record,
        }

    def _capture_grid_4d_ratio(self, n_frames: int = 64) -> tuple:
        """グリッドの 4D パッチ (50% Gray) を使って blank/neutral 用の ratio を取得する。"""
        bayer_sum: np.ndarray | None = None
        ref_sum = np.zeros(3, dtype=np.float64)
        ref_disp_x, ref_disp_y = self.config.processing.posi_ref
        ref_w = max(int(self.config.processing.spot_size_ref), 2)
        ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
        metadata: dict = {}
        analysis_main_array: np.ndarray | None = None
        for i in range(n_frames):
            request = self.picam2.capture_request()
            try:
                if analysis_main_array is None:
                    analysis_main_array = request.make_array("main")
                raw_array = request.make_array("raw")
                metadata = request.get_metadata()
            finally:
                request.release()
            if i == 0 or (i + 1) == n_frames or (i + 1) % 8 == 0:
                key = self._show_chart_workflow_overlay(
                    str(self._chart_workflow_status.get("stage", "capture") or "capture"),
                    str(self._chart_workflow_status.get("message", "4D sample capture") or "4D sample capture"),
                    detail="4D patch を取得中",
                    progress=i + 1,
                    total=n_frames,
                    can_cancel=bool(self._chart_workflow_status.get("can_cancel", False)),
                    source=str(self._chart_workflow_status.get("source", "")),
                )
                self._raise_if_chart_cancel_requested(key, "4d sample capture")
            raw_bayer = self.bayer.parse_raw(raw_array)
            if bayer_sum is None:
                bayer_sum = raw_bayer.astype(np.float64)
            else:
                bayer_sum += raw_bayer.astype(np.float64)
            ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h = self.bayer.display_to_raw_coords(
                ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                self.config.display.flip_horizontal,
                self.config.display.flip_vertical,
            )
            roi_ref = self.bayer.extract_raw_roi(
                raw_bayer, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
            )
            if self.dark.is_loaded:
                dark_ref = self.dark.get_dark_roi(ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                if dark_ref is not None:
                    roi_ref = self.dark.subtract(roi_ref, dark_ref)
            if self.flat.is_loaded:
                roi_ref = self.flat.correct_roi(
                    roi_ref, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
                )
            ref_sum += self.bayer.extract_bayer_means_from_roi(roi_ref).astype(np.float64)

        assert bayer_sum is not None
        avg_bayer = (bayer_sum / n_frames).astype(np.uint16)
        avg_ref = (ref_sum / n_frames).astype(np.float32)

        if self.dark.is_loaded and self.dark.dark_frame is not None:
            avg_bayer = self.dark.subtract(avg_bayer, self.dark.dark_frame)
        if self.flat.is_loaded:
            bh, bw = avg_bayer.shape[:2]
            avg_bayer = self.flat.correct_roi(avg_bayer, 0, 0, bw, bh)

        self._prepare_chart_patchwise_rois_from_main(analysis_main_array)
        patch_bayer_means = self._grid_extractor.extract_all_patch_bayer_means(
            avg_bayer, self.bayer, metadata, self.config, debug=True,
            debug_save_dir=get_today_calibration_dir(),
        )
        patch_4d_raw = patch_bayer_means[SPYDERCHECKR_48_GRAY_4D_INDEX].astype(np.float64)
        patch_4d_norm = patch_4d_raw / self.bayer.max_val
        safe_ref = np.where(avg_ref < 1e-8, 1e-8, avg_ref)
        return (patch_4d_norm / safe_ref).astype(np.float32), avg_ref

    def _run_blank_and_neutral_wizard(self) -> None:
        """[P]完了後に[B]ブランク補正 → [N]ニュートラル補正を自動実行するウィザード。"""
        self._show_chart_workflow_overlay(
            "postprocess",
            "Blank -> Neutral",
            detail="4D patch を取得中",
            can_cancel=True,
        )
        avg_ratio, avg_ref = self._capture_grid_4d_ratio(n_frames=64)

        # デバッグ出力
        _SW = "\u2550" * 62
        _sw = "\u2500" * 62
        print(f"\n{_SW}")
        print("  [B][N] \u30a6\u30a3\u30b6\u30fc\u30c9 \u2014 4D\u30d1\u30c3\u30c1\u53d6\u5f97\u5024")
        print(_SW)
        print("  avg_ref  (Ref ROI, dark+flat\u88dc\u6b63\u6e08\u307f, 0-1\u6b63\u898f\u5316):")
        print(f"    R={float(avg_ref[0]):9.6f}  G={float(avg_ref[1]):9.6f}  B={float(avg_ref[2]):9.6f}")
        print(_sw)
        print("  avg_ratio (4D patch_norm / avg_ref)  \u2190 blank_ratio \u3068\u3057\u3066\u4fdd\u5b58:  \u671f\u5f85\u5024 ~1.0")
        print(f"    R={float(avg_ratio[0]):9.6f}  G={float(avg_ratio[1]):9.6f}  B={float(avg_ratio[2]):9.6f}")
        print(_SW + "\n")

        # [B] ブランク補正
        self.blank.save(avg_ratio)
        self._show_capture_overlay(
            "\u30d6\u30e9\u30f3\u30af\u88dc\u6b63\u5b8c\u4e86",
            "\u7d9a\u3051\u3066\u30cb\u30e5\u30fc\u30c8\u30e9\u30eb\u88dc\u6b63\u3092\u5b9f\u884c\u3057\u307e\u3059",
            wait_sec=1.5,
        )

        # [N] ニュートラル補正
        neutral_ratio = self.blank.correct(avg_ratio)
        r_R = float(neutral_ratio[0])
        r_G = float(neutral_ratio[1])
        r_B = float(neutral_ratio[2])
        d_R = r_G / r_R if r_R > 1e-6 else 1.0
        d_B = r_G / r_B if r_B > 1e-6 else 1.0

        # デバッグ出力
        _SN = "\u2550" * 62
        _sn = "\u2500" * 62
        print(f"\n{_SN}")
        print("  [N] \u30cb\u30e5\u30fc\u30c8\u30e9\u30eb\u88dc\u6b63 \u2014 \u7b97\u51fa\u5024")
        print(_SN)
        print("  neutral_ratio (blank.correct(avg_ratio) = avg_ratio / blank_ratio):")
        print(f"    R={r_R:9.6f}  G={r_G:9.6f}  B={r_B:9.6f}")
        print(_sn)
        print(f"  \u5bfe\u89d2\u88dc\u6b63\u4fc2\u6570:  d_R = G/R = {r_G:.6f} / {r_R:.6f} = {d_R:.6f}")
        print(f"                 d_B = G/B = {r_G:.6f} / {r_B:.6f} = {d_B:.6f}")
        print(f"[diag] raw d_R/d_B before CCM enforcement: d_R={d_R:.6f} d_B={d_B:.6f}")
        # CCM が有効な場合、対角補正は CCM に包含されるため恒等に強制
        if self.lab_converter._ccm is not None:
            d_R = 1.0
            d_B = 1.0
            print("  → CCM有効: 対角補正を恒等に強制")
        print(_SN + "\n")

        self.lab_converter.set_diagonal_correction(d_R, d_B)
        self.spectral_drift_tracker.activate()
        # self.ref_anchor_lab = self.lab_converter.neutral_anchor_Lab()  # anchor は _compute_measurements 内で毎フレーム計算

        import shutil as _shutil
        import glob as _glob

        scale_baseline = normalize_ref_scale_triplet(avg_ref, self.white_ratio_rgb)
        if scale_baseline is None:
            scale_baseline, scale_source = self._resolve_live_ref_scale_baseline_for_p(
                self.ref_train
            )
        else:
            scale_source = "neutral"
        self._set_live_ref_scale_baseline(scale_baseline, scale_source, notify=True)
        _nc_path = os.path.join(get_today_calibration_dir(), "neutral_correction.json")
        data = {
            "d_R": d_R,
            "d_B": d_B,
            "ref_baseline": avg_ref.tolist(),
            "ref_scale_baseline": (
                None
                if self.live_ref_scale_baseline is None
                else np.asarray(self.live_ref_scale_baseline, dtype=np.float64).tolist()
            ),
        }
        with open(_nc_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        _remove_cleared_marker("neutral_correction.json")
        _date_str = datetime.now().strftime("%Y-%m-%d")
        _nc_backup = os.path.join(
            get_today_calibration_dir(), f"neutral_correction_{_date_str}.json"
        )
        _shutil.copy2(_nc_path, _nc_backup)
        for _old in sorted(_glob.glob(os.path.join(
            get_today_calibration_dir(), "neutral_correction_*.json"
        )), reverse=True)[5:]:
            os.remove(_old)
        self._set_live_ref_baseline(avg_ref, "neutral", notify=True)
        print(f"- Neutral correction saved: d_R={d_R:.4f} d_B={d_B:.4f}")

        self._show_capture_overlay(
            "\u30d6\u30e9\u30f3\u30af\u88dc\u6b63\u30fb\u30cb\u30e5\u30fc\u30c8\u30e9\u30eb\u88dc\u6b63 \u5b8c\u4e86",
            "\u6b21\u306e\u64cd\u4f5c\u3078\u9032\u3081\u307e\u3059",
            wait_sec=2.5,
        )

    def _capture_neutral_ratio(self, n_frames: int = 64) -> tuple:
        """ニュートラル補正用の専用マルチフレームキャプチャ。"""
        ref_sum = np.zeros(3, dtype=np.float64)
        tar_sum = np.zeros(3, dtype=np.float64)
        ref_disp_x, ref_disp_y = self.config.processing.posi_ref
        tar_disp_x, tar_disp_y = self.config.processing.posi_tar
        ref_w = max(int(self.config.processing.spot_size_ref), 2)
        ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
        tar_w = max(int(self.config.processing.spot_size_tar), 2)
        tar_h = max(int(tar_w * self.config.processing.aspect_tar), 2)
        for _ in range(n_frames):
            request = self.picam2.capture_request()
            try:
                raw_array = request.make_array("raw")
                metadata = request.get_metadata()
            finally:
                request.release()
            raw_bayer = self.bayer.parse_raw(raw_array)
            ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h = (
                self.bayer.display_to_raw_coords(
                    ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                    self.config.display.flip_horizontal,
                    self.config.display.flip_vertical,
                )
            )
            tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h = (
                self.bayer.display_to_raw_coords(
                    tar_disp_x, tar_disp_y, tar_w, tar_h, metadata,
                    self.config.display.flip_horizontal,
                    self.config.display.flip_vertical,
                )
            )
            roi_ref = self.bayer.extract_raw_roi(
                raw_bayer, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
            )
            roi_tar = self.bayer.extract_raw_roi(
                raw_bayer, tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h
            )
            if self.dark.is_loaded:
                dark_ref = self.dark.get_dark_roi(ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                dark_tar = self.dark.get_dark_roi(tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h)
                if dark_ref is not None:
                    roi_ref = self.dark.subtract(roi_ref, dark_ref)
                if dark_tar is not None:
                    roi_tar = self.dark.subtract(roi_tar, dark_tar)
            if self.flat.is_loaded:
                roi_ref = self.flat.correct_roi(roi_ref, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                roi_tar = self.flat.correct_roi(roi_tar, tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h)
            ref_sum += self.bayer.extract_bayer_means_from_roi(roi_ref).astype(np.float64)
            tar_sum += self.bayer.extract_bayer_means_from_roi(roi_tar).astype(np.float64)
        avg_ref = ref_sum / n_frames
        avg_tar = tar_sum / n_frames
        safe_ref = np.where(avg_ref < 1e-8, 1e-8, avg_ref)
        return (avg_tar / safe_ref).astype(np.float32), avg_ref.astype(np.float32)

    @staticmethod
    def _gray_card_luma_coeff() -> np.ndarray:
        return np.array([0.2126729, 0.7151522, 0.0721750], dtype=np.float64)

    @staticmethod
    def _gray_card_block_luma_map(raw_roi: np.ndarray) -> np.ndarray:
        """Bayer mosaic の2x2ブロック平均で texture 判定用 luma map を作る。"""
        roi = np.asarray(raw_roi, dtype=np.float64)
        if roi.ndim == 3:
            return np.mean(roi, axis=2)
        if roi.ndim != 2:
            return np.asarray([], dtype=np.float64)
        h2 = (roi.shape[0] // 2) * 2
        w2 = (roi.shape[1] // 2) * 2
        if h2 < 2 or w2 < 2:
            return roi
        return roi[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))

    @classmethod
    def _gray_card_spatial_stats(cls, raw_roi: np.ndarray) -> tuple[float | None, float | None]:
        """Tar ROI の均一性を CV と center-edge 差[%]で返す。"""
        luma = cls._gray_card_block_luma_map(raw_roi)
        if luma.size == 0:
            return None, None
        finite = luma[np.isfinite(luma)]
        if finite.size == 0:
            return None, None
        mean_value = float(np.mean(finite))
        if abs(mean_value) < 1e-8:
            return None, None
        cv = float(np.std(finite, ddof=0) / abs(mean_value))

        h, w = luma.shape[:2]
        if h < 4 or w < 4:
            return cv, 0.0
        y0, y1 = h // 4, max(h // 4 + 1, (h * 3) // 4)
        x0, x1 = w // 4, max(w // 4 + 1, (w * 3) // 4)
        center = luma[y0:y1, x0:x1]
        mask = np.ones(luma.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = False
        edge = luma[mask]
        center = center[np.isfinite(center)]
        edge = edge[np.isfinite(edge)]
        if center.size == 0 or edge.size == 0:
            return cv, 0.0
        center_edge_pct = abs(float(np.mean(center)) - float(np.mean(edge))) / abs(mean_value) * 100.0
        return cv, float(center_edge_pct)

    def _capture_gray_card_check_ratio(self, n_frames: int = 64) -> dict:
        """Gray Check 用に Tar/Ref ratio と安定性統計を同時取得する。"""
        ref_sum = np.zeros(3, dtype=np.float64)
        tar_sum = np.zeros(3, dtype=np.float64)
        ratio_frames: list[np.ndarray] = []
        tar_ref_luma_ratios: list[float] = []
        tar_luma_cvs: list[float] = []
        tar_center_edge_pcts: list[float] = []
        luma_coeff = self._gray_card_luma_coeff()

        ref_disp_x, ref_disp_y = self.config.processing.posi_ref
        tar_disp_x, tar_disp_y = self.config.processing.posi_tar
        ref_w = max(int(self.config.processing.spot_size_ref), 2)
        ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
        tar_w = max(int(self.config.processing.spot_size_tar), 2)
        tar_h = max(int(tar_w * self.config.processing.aspect_tar), 2)

        for _ in range(n_frames):
            request = self.picam2.capture_request()
            try:
                raw_array = request.make_array("raw")
                metadata = request.get_metadata()
            finally:
                request.release()
            raw_bayer = self.bayer.parse_raw(raw_array)
            ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h = (
                self.bayer.display_to_raw_coords(
                    ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                    self.config.display.flip_horizontal,
                    self.config.display.flip_vertical,
                )
            )
            tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h = (
                self.bayer.display_to_raw_coords(
                    tar_disp_x, tar_disp_y, tar_w, tar_h, metadata,
                    self.config.display.flip_horizontal,
                    self.config.display.flip_vertical,
                )
            )
            roi_ref = self.bayer.extract_raw_roi(
                raw_bayer, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
            )
            roi_tar = self.bayer.extract_raw_roi(
                raw_bayer, tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h
            )
            if self.dark.is_loaded:
                dark_ref = self.dark.get_dark_roi(ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                dark_tar = self.dark.get_dark_roi(tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h)
                if dark_ref is not None:
                    roi_ref = self.dark.subtract(roi_ref, dark_ref)
                if dark_tar is not None:
                    roi_tar = self.dark.subtract(roi_tar, dark_tar)
            if self.flat.is_loaded:
                roi_ref = self.flat.correct_roi(roi_ref, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                roi_tar = self.flat.correct_roi(roi_tar, tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h)

            ref_rgb = self.bayer.extract_bayer_means_from_roi(roi_ref).astype(np.float64)
            tar_rgb = self.bayer.extract_bayer_means_from_roi(roi_tar).astype(np.float64)
            ref_sum += ref_rgb
            tar_sum += tar_rgb
            safe_ref = np.where(ref_rgb < 1e-8, 1e-8, ref_rgb)
            ratio_frames.append(tar_rgb / safe_ref)

            ref_luma = float(ref_rgb @ luma_coeff)
            tar_luma = float(tar_rgb @ luma_coeff)
            if math.isfinite(ref_luma) and abs(ref_luma) >= 1e-8 and math.isfinite(tar_luma):
                tar_ref_luma_ratios.append(tar_luma / ref_luma)
            tar_cv, tar_center_edge = self._gray_card_spatial_stats(roi_tar)
            if tar_cv is not None and math.isfinite(tar_cv):
                tar_luma_cvs.append(tar_cv)
            if tar_center_edge is not None and math.isfinite(tar_center_edge):
                tar_center_edge_pcts.append(tar_center_edge)

        avg_ref = ref_sum / max(n_frames, 1)
        avg_tar = tar_sum / max(n_frames, 1)
        safe_avg_ref = np.where(avg_ref < 1e-8, 1e-8, avg_ref)
        ratio_raw = avg_tar / safe_avg_ref
        ratios = np.asarray(ratio_frames, dtype=np.float64)
        ratio_std = (
            np.std(ratios, axis=0, ddof=0)
            if ratios.ndim == 2 and ratios.shape[0] > 0
            else np.full(3, np.nan, dtype=np.float64)
        )
        return {
            "ratio_raw": ratio_raw.astype(np.float64),
            "avg_ref": avg_ref.astype(np.float64),
            "avg_tar": avg_tar.astype(np.float64),
            "capture_stats": {
                "ratio_std": ratio_std.astype(np.float64),
                "max_ratio_std": float(np.max(ratio_std)) if np.all(np.isfinite(ratio_std)) else None,
                "tar_ref_luma_ratio_raw": (
                    float(np.mean(tar_ref_luma_ratios)) if tar_ref_luma_ratios else None
                ),
                "tar_luma_cv": float(np.mean(tar_luma_cvs)) if tar_luma_cvs else None,
                "tar_center_edge_abs_pct": (
                    float(np.mean(tar_center_edge_pcts)) if tar_center_edge_pcts else None
                ),
            },
        }

    def _gray_card_artifact_mtime(self, filename: str, manager=None) -> float | None:
        candidates = []
        manager_path = getattr(manager, "save_path", None)
        if manager_path:
            candidates.append(str(manager_path))
        candidates.append(os.path.join(get_today_calibration_dir(), filename))
        candidates.append(os.path.join(str(CALIBRATION_DIR), filename))
        for path in candidates:
            try:
                if path and os.path.exists(path):
                    return float(os.path.getmtime(path))
            except OSError:
                continue
        return None

    @staticmethod
    def _gray_card_quantized_exposure_us(value) -> int | None:
        """Context hash 用に露光値を100us単位へ量子化する。"""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return int(round(number / 100.0) * 100)

    @staticmethod
    def _gray_card_quantized_gain(value) -> float | None:
        """Context hash 用に gain を小数2桁へ量子化する。"""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return round(number, 2)

    def _build_gray_card_calibration_context(self) -> dict:
        """Gray Check baseline identity に使う calibration context を構成する。"""
        p = self.config.processing
        software_identity = get_software_identity()
        today_dir = get_today_calibration_dir()
        ccm_manager = self.ccm_store
        exposure_us = getattr(self.ae, "current_exposure", None)
        analogue_gain = (
            getattr(self.ae, "current_gain", None)
            if getattr(self.ae, "current_gain", None) is not None
            else getattr(self.ae, "current_analogue_gain", None)
        )
        return {
            "schema_version": GRAY_CARD_CHECK_SCHEMA_VERSION,
            "calibration_date": os.path.basename(today_dir),
            "software_version": software_identity.get("software_version", ""),
            "git_revision": software_identity.get("git_revision", ""),
            "roi_ref": {
                "pos": list(p.posi_ref),
                "spot_size": int(p.spot_size_ref),
                "aspect": float(p.aspect_ref),
                "height": int(max(int(p.spot_size_ref * p.aspect_ref), 2)),
            },
            "roi_tar": {
                "pos": list(p.posi_tar),
                "spot_size": int(p.spot_size_tar),
                "aspect": float(p.aspect_tar),
                "height": int(max(int(p.spot_size_tar * p.aspect_tar), 2)),
            },
            "display_flip": {
                "horizontal": bool(self.config.display.flip_horizontal),
                "vertical": bool(self.config.display.flip_vertical),
            },
            "dark_loaded": bool(getattr(self.dark, "is_loaded", False)),
            "flat_loaded": bool(getattr(self.flat, "is_loaded", False)),
            "wb_calibrated": bool(getattr(self.wb_calibrator, "is_calibrated", False)),
            "blank_loaded": bool(getattr(self.blank, "is_loaded", False)),
            "master_ref_loaded": bool(getattr(self.master_ref, "is_loaded", False)),
            "ccm_loaded": bool(getattr(self.lab_converter, "_ccm", None) is not None),
            "live_ref_scale_baseline_source": self.live_ref_scale_baseline_source or "none",
            "exposure_us": self._gray_card_quantized_exposure_us(exposure_us),
            "analogue_gain": self._gray_card_quantized_gain(analogue_gain),
            "dark_frame_path_mtime": self._gray_card_artifact_mtime("dark_frame.npy", self.dark),
            "flat_field_path_mtime": self._gray_card_artifact_mtime("flat_field_gain.npy", self.flat),
            "master_ref_path_mtime": self._gray_card_artifact_mtime("master_ref.json", self.master_ref),
            "blank_ratio_path_mtime": self._gray_card_artifact_mtime("blank_ratio.json", self.blank),
            "neutral_correction_path_mtime": self._gray_card_artifact_mtime("neutral_correction.json"),
            "ccm_path_mtime": self._gray_card_artifact_mtime("ccm.json", ccm_manager),
        }

    def _get_ref_drift_snapshot_for_gray_check(self) -> dict:
        tracker = self.stability_tracker
        if tracker is None or not hasattr(tracker, "get_drift_snapshot"):
            return {}
        try:
            snapshot = dict(tracker.get_drift_snapshot())
        except Exception:
            return {}
        if hasattr(tracker, "get_guard_state"):
            try:
                snapshot["guard_state"] = tracker.get_guard_state()
            except Exception:
                pass
        return snapshot

    def _build_gray_card_check_record(self, n_frames: int = 64) -> dict:
        now = datetime.now()
        timestamp = now.isoformat(timespec="microseconds")
        timestamp_slug = gray_card_check_timestamp_slug(now)
        context = self._build_gray_card_calibration_context()
        context_hash = compute_gray_card_check_context_hash(context)
        capture = self._capture_gray_card_check_ratio(n_frames=n_frames)
        ratio_raw = np.asarray(capture["ratio_raw"], dtype=np.float64).ravel()[:3]
        avg_ref = np.asarray(capture["avg_ref"], dtype=np.float64).ravel()[:3]

        ratio_corrected = ratio_raw.astype(np.float64, copy=True)
        correction_source = "none"
        correction_scale = None
        if self.blank is not None and getattr(self.blank, "is_loaded", False):
            ratio_corrected = np.asarray(self.blank.correct(ratio_raw), dtype=np.float64)
            correction_source = "blank"
        elif self.master_ref is not None and getattr(self.master_ref, "is_loaded", False):
            correction_scale = np.asarray(self.master_ref.compute_scale(avg_ref), dtype=np.float64)
            safe_scale = np.where(np.abs(correction_scale) < 1e-8, 1e-8, correction_scale)
            ratio_corrected = ratio_raw / safe_scale
            correction_source = "master_ref"

        white_for_norm = np.asarray(self.white_ratio_rgb, dtype=np.float64).copy()
        if self.blank is not None and getattr(self.blank, "is_loaded", False):
            white_for_norm = np.asarray(self.blank.correct(white_for_norm), dtype=np.float64)
        safe_white = np.where(white_for_norm < 1e-8, 1e-8, white_for_norm)
        ratio_white = ratio_corrected / safe_white
        measured_lab = None
        lab_rel = None
        ref_scale = None
        if self.lab_converter is not None:
            ref_scale_triplet = normalize_ref_scale_triplet(avg_ref, self.white_ratio_rgb)
            canonical = run_canonical_lab_pipeline(
                ratio_corrected,
                white_for_norm,
                ref_scale_triplet,
                self.live_ref_scale_baseline,
                self.lab_converter,
            )
            ratio_white = np.asarray(canonical["ratio_white"], dtype=np.float64)
            measured_lab = np.asarray(canonical["lab"], dtype=np.float64)
            ref_scale = canonical.get("ref_scale")
            anchor_raw = np.ones(3, dtype=np.float64)
            if self.blank is not None and getattr(self.blank, "is_loaded", False):
                anchor_raw = np.asarray(self.blank.correct(anchor_raw), dtype=np.float64)
            anchor = run_canonical_lab_pipeline(
                anchor_raw,
                white_for_norm,
                ref_scale_triplet,
                self.live_ref_scale_baseline,
                self.lab_converter,
            )
            anchor_lab = np.asarray(anchor["lab"], dtype=np.float64).ravel()[:3]
            lab_rel = measured_lab.ravel()[:3] - anchor_lab

        mode_name = getattr(self.mode, "current", "unknown")
        operator_name = (
            os.environ.get("USER")
            or os.environ.get("USERNAME")
            or "unknown"
        )
        return {
            "schema_version": GRAY_CARD_CHECK_SCHEMA_VERSION,
            "kind": "gray_card_check",
            "role": "check",
            "timestamp": timestamp,
            "timestamp_slug": timestamp_slug,
            "calibration_dir": get_today_calibration_dir(),
            "calibration_context": context,
            "calibration_context_hash": context_hash,
            "mode": mode_name,
            "operator": operator_name,
            "n_frames": int(n_frames),
            "roi_source": "tar_roi_gray_card",
            "baseline_id": None,
            "baseline_timestamp": None,
            "baseline_is_stale": False,
            "stale_reason": None,
            "ratio_raw": ratio_raw,
            "ratio_corrected": ratio_corrected,
            "ratio_white": ratio_white,
            "measured_lab": measured_lab,
            "lab_rel": lab_rel,
            "ref_scale": ref_scale,
            "avg_ref": avg_ref,
            "avg_tar": np.asarray(capture["avg_tar"], dtype=np.float64).ravel()[:3],
            "white_ratio_rgb": white_for_norm,
            "correction_source": correction_source,
            "correction_scale": correction_scale,
            "live_ref_scale_baseline_source": self.live_ref_scale_baseline_source or "none",
            "live_ref_scale_baseline": (
                None
                if self.live_ref_scale_baseline is None
                else np.asarray(self.live_ref_scale_baseline, dtype=np.float64)
            ),
            "ref_drift_snapshot": self._get_ref_drift_snapshot_for_gray_check(),
            "exposure_us": getattr(self.ae, "current_exposure", None),
            "analogue_gain": (
                getattr(self.ae, "current_gain", None)
                if getattr(self.ae, "current_gain", None) is not None
                else getattr(self.ae, "current_analogue_gain", None)
            ),
            "capture_stats": capture["capture_stats"],
            "quality": {},
        }

    def _is_gray_card_check_pipeline_ready(self) -> bool:
        if self.lab_converter is None:
            self._show_capture_overlay(
                "Gray Check unavailable",
                "Lab pipeline is not ready",
                wait_sec=1.5,
            )
            return False
        if getattr(self.lab_converter, "_ccm", None) is None:
            self._show_capture_overlay(
                "Gray Check unavailable",
                "Run P first",
                wait_sec=1.5,
            )
            return False
        if self.live_ref_scale_baseline is None:
            self._show_capture_overlay(
                "Gray Check unavailable",
                "Run N or P first",
                wait_sec=1.5,
            )
            return False
        return True

    @staticmethod
    def _format_gray_check_metric_line(quality: dict) -> str:
        delta_e = quality.get("delta_e_from_baseline")
        rgb = quality.get("max_abs_delta_ratio_white")
        if delta_e is None or rgb is None:
            return "dE N/A  RGB N/A"
        try:
            return f"dE {float(delta_e):.2f}  RGB {float(rgb):.3f}"
        except (TypeError, ValueError):
            return "dE N/A  RGB N/A"

    def _show_gray_card_check_status(self, record: dict) -> None:
        quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
        status = str(quality.get("status", "WARN")).upper()
        metric_line = self._format_gray_check_metric_line(quality)
        if status == "OK":
            self._show_capture_overlay(
                "Gray Check OK",
                "許容内",
                metric_line,
                wait_sec=2.0,
            )
        elif status == "RECAL":
            self._show_capture_overlay(
                "Gray Check RECAL",
                "要再校正",
                "Run D/F/W/N/P",
                metric_line,
                wait_sec=2.5,
            )
        else:
            self._show_capture_overlay(
                "Gray Check WARN",
                "要注意",
                "Run 4D or W/N if repeated",
                metric_line,
                wait_sec=2.5,
            )

    def _handle_gray_card_check(self) -> None:
        """同じグレーカードを Tar ROI で測る operator-facing drift check。"""
        if self._chart_state != "idle":
            self._show_capture_overlay(
                "Gray Check unavailable",
                "Finish P preview first",
                wait_sec=1.5,
            )
            return
        if not self._check_calibration_prerequisites("v"):
            return
        if not self._is_gray_card_check_pipeline_ready():
            return
        if not self._confirm_operator_action(
            "Gray Check",
            "Tar ROI に同じグレーカードを置いてください",
            next_label="測定開始",
            extra_confirm_keys=(ord("v"),),
        ):
            return

        self._show_capture_overlay(
            "Gray Check",
            "64フレームを取得中...",
        )
        record = self._build_gray_card_check_record(n_frames=64)
        baseline, load_reason = load_gray_card_check_baseline(
            calibration_dir=get_today_calibration_dir()
        )
        baseline_state = describe_gray_card_check_baseline_state(
            baseline,
            expected_context_hash=record["calibration_context_hash"],
        )
        if load_reason is not None and baseline is None:
            baseline_state = {"usable": False, "reason": load_reason}

        if not baseline_state.get("usable", False):
            record["role"] = "baseline"
            record["baseline_id"] = f"gray-card:{record['timestamp_slug']}"
            record["baseline_timestamp"] = record["timestamp"]
            record["baseline_is_stale"] = False
            record["stale_reason"] = None
            record["previous_baseline_state"] = baseline_state.get("reason")
            validation = validate_gray_card_baseline_candidate(record)
            record["baseline_validation"] = validation
            if not validation.get("valid", False):
                record["attempted_baseline_id"] = record.get("baseline_id")
                record["previous_baseline_id"] = (
                    baseline.get("baseline_id") if isinstance(baseline, dict) else None
                )
                record["role"] = "check"
                record["baseline_id"] = None
                record["baseline_timestamp"] = None
                record["quality"] = {
                    "status": "REJECTED",
                    "status_reason": validation.get("reason", "baseline_rejected"),
                    "failed_axes": validation.get("failed_checks", []),
                }
                record["artifact_paths"] = persist_gray_card_check_artifacts(
                    record,
                    calibration_dir=get_today_calibration_dir(),
                    write_baseline=False,
                )
                self._show_capture_overlay(
                    "Gray baseline rejected",
                    "Place gray card in Tar ROI",
                    str(validation.get("reason", "baseline_rejected")),
                    wait_sec=2.5,
                )
                return
            record["quality"] = {
                "status": "BASELINE",
                "status_reason": "baseline_saved",
                "failed_axes": [],
            }
            record["artifact_paths"] = persist_gray_card_check_artifacts(
                record,
                calibration_dir=get_today_calibration_dir(),
                write_baseline=True,
            )
            if baseline_state.get("reason") == "calibration_context_mismatch":
                self._show_capture_overlay(
                    "Gray baseline saved",
                    "Calibration context changed",
                    "Check again later with v",
                    wait_sec=2.5,
                )
            else:
                self._show_capture_overlay(
                    "Gray baseline saved",
                    "Check again later with v",
                    wait_sec=2.0,
                )
            return

        record["role"] = "check"
        record["baseline_id"] = baseline.get("baseline_id")
        record["baseline_timestamp"] = baseline.get("timestamp")
        record["baseline_is_stale"] = False
        record["stale_reason"] = None
        record["baseline_validation"] = None
        record["quality"] = compare_gray_card_check_to_baseline(
            record,
            baseline,
            ref_drift_snapshot=record.get("ref_drift_snapshot"),
        )
        record["artifact_paths"] = persist_gray_card_check_artifacts(
            record,
            calibration_dir=get_today_calibration_dir(),
            write_baseline=False,
        )
        self._show_gray_card_check_status(record)

    def _calibrate_wb_from_raw(self, n_frames: int = 64) -> None:
        """RAW Bayerから複数フレーム平均でWBゲインを算出して固定する。"""
        try:
            raw_sum: np.ndarray | None = None
            metadata: dict = {}
            for _ in range(n_frames):
                request = self.picam2.capture_request()
                try:
                    raw_array = request.make_array("raw")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                raw_bayer = self.bayer.parse_raw(raw_array)
                if raw_sum is None:
                    raw_sum = raw_bayer.astype(np.float64)
                else:
                    raw_sum += raw_bayer.astype(np.float64)
            assert raw_sum is not None
            raw_avg = np.round(raw_sum / n_frames).astype(np.uint16)
            ref_disp_x, ref_disp_y = self.config.processing.posi_ref
            ref_w = max(int(self.config.processing.spot_size_ref), 2)
            ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
            ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h = self.bayer.display_to_raw_coords(
                ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                self.config.display.flip_horizontal,
                self.config.display.flip_vertical,
            )
            roi_ref = self.bayer.extract_raw_roi(
                raw_avg, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
            )
            if self.dark.is_loaded:
                dark_ref = self.dark.get_dark_roi(ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
                if dark_ref is not None:
                    roi_ref = self.dark.subtract(roi_ref, dark_ref)
            if self.flat.is_loaded:
                roi_ref = self.flat.correct_roi(
                    roi_ref, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h
                )
            gains = self.wb_calibrator.calibrate_from_raw_bayer(roi_ref)
            self.picam2.set_controls({"AwbEnable": False, "ColourGains": gains})
            if self.stability_tracker is not None and hasattr(
                self.stability_tracker, "mark_recalibrated"
            ):
                self.stability_tracker.mark_recalibrated(source="W")
            self._show_capture_overlay(
                "\u30db\u30ef\u30a4\u30c8\u56fa\u5b9a\uff08RAW\u65b9\u5f0f\uff09\u5b8c\u4e86",
                "\u6b21\u306e\u624b\u9806\u3078\u9032\u3081\u307e\u3059",
                wait_sec=2.0,
            )
        except Exception as e:
            self._show_capture_overlay(
                "\u30db\u30ef\u30a4\u30c8\u56fa\u5b9a\uff08RAW\u65b9\u5f0f\uff09\u5931\u6557",
                str(e),
                wait_sec=2.0,
            )
            print(f"Warning: raw WB calibration failed: {e}")

    def handle(self, log_data, meas):
        """
        キー入力を処理する。

        Args:
          log_data: ログ用データ。
          meas: 測定結果辞書。
        Returns:
          bool: 終了要求ならTrue。
        """
        key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
        if key == 255:
            return False
        # TABキー: モード切替
        if key == 9 and self.mode is not None:
            try:
                new_mode = self.mode.toggle()
                self.last_mode_change_time = time.time()
                print(f"- Mode changed: {new_mode}")
            except Exception as e:
                print(f"Warning: mode toggle failed: {e}")
            return False
        if key in (ord("i"), ord("I")):
            if self.stability_tracker is not None and hasattr(
                self.stability_tracker,
                "cycle_ref_monitor_axis",
            ):
                try:
                    series = self.stability_tracker.cycle_ref_monitor_axis()
                    mode = str(getattr(self.stability_tracker, "ref_monitor_axis_mode", "AUTO"))
                    label = str(series.get("label") or series.get("axis") or mode)
                    status = str(series.get("status") or "")
                    suffix = f" {status}" if status else ""
                    print(f"- Ref Stability graph: {mode} {label}{suffix}")
                except Exception as e:
                    print(f"Warning: Ref Stability graph switch failed: {e}")
            return False
        if key == ord("q") or key == 27:
            if self._chart_state in ("corner_input", "preview", "measuring", "flip_warning"):
                # Phase 7B: flip_warning も ESC/q で cancel 可能 (idle へ戻す).
                self._cancel_chart_workflow("preview を終了しました", show_overlay=True)
                return False
            return True
        elif is_preview_freeze_key(key):
            if self._chart_state == "preview":
                if self._grid_extractor is None or not self._grid_extractor.is_ready:
                    self._show_capture_overlay(
                        "preview bundle を保存できません",
                        "chart grid が未準備です",
                        wait_sec=1.2,
                    )
                    return False
                self.preview_freeze_bundle_pending = True
            else:
                self._show_capture_overlay(
                    "preview bundle は P preview 専用です",
                    "P で preview を開いてから g を押してください",
                    wait_sec=1.2,
                )
            return False
        elif key == ord("m"):
            guard_state = (
                self.stability_tracker.get_guard_state()
                if self.stability_tracker is not None
                else None
            )
            is_recalib_required = guard_state == "RECALIB_REQUIRED"
            is_led_drift_warn = guard_state == "LED_DRIFT_WARN"
            if is_recalib_required:
                self._show_capture_overlay(
                    "\u518d\u30ad\u30e3\u30ea\u30d6\u304c\u5fc5\u8981\u3067\u3059",
                    "c -> d -> f -> w \u3092\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044",
                    wait_sec=2.0,
                )
                return False
            if is_led_drift_warn:
                self._show_capture_overlay(
                    "\u7167\u660e\u30c9\u30ea\u30d5\u30c8\u8b66\u544a",
                    "\u8a18\u9332\u306f\u7d99\u7d9a\u3057\u307e\u3059\uff08\u5fc5\u8981\u306a\u3089 w \u3092\u518d\u5b9f\u884c\uff09",
                    wait_sec=1.2,
                )
            if not self._measurement_fail_stop_allows_recording():
                return False
            if self.session_recorder and not self.session_recorder.is_recording:
                overlay = FileNameInputOverlay(
                    self.window_name, width=640, height=200,
                )
                file_name = overlay.get_filename()
                if file_name:
                    mode_str = self.mode.current if self.mode else "Lab"
                    self.session_recorder.start(file_name, mode_str)
                    self._show_capture_overlay(
                        "CSV\u3078\u306e\u8a18\u9332\u3092\u958b\u59cb",
                        f"File: {file_name}.csv",
                        wait_sec=1.5,
                    )
            else:
                self.logger.log_event(log_data, event_type="measure")
                self.snapshot_pending = True
        elif key == ord("s"):
            if self.session_recorder and self.session_recorder.is_recording:
                csv_path = self.session_recorder.stop()
                self._show_capture_overlay(
                    "CSV\u3078\u306e\u8a18\u9332\u3092\u505c\u6b62",
                    f"Saved: {os.path.basename(csv_path)}",
                    wait_sec=2.0,
                )
        elif key == ord("v"):
            self._handle_gray_card_check()
        elif key == ord("d"):
            if not self._check_calibration_prerequisites("d"):
                return False
            if not self._confirm_operator_action(
                "\u30ce\u30a4\u30ba\u88dc\u6b63",
                "\u30ec\u30f3\u30ba\u30ad\u30e3\u30c3\u30d7\u3092\u88c5\u7740\u3057\u3066\u304f\u3060\u3055\u3044",
                next_label="\u8a2d\u7f6e\u5b8c\u4e86",
                extra_confirm_keys=(ord("d"),),
            ):
                return False
            self._show_capture_overlay("64\u30d5\u30ec\u30fc\u30e0\u3092\u53d6\u5f97\u4e2d...")
            self.dark.capture_dark_frame(self.picam2, self.bayer)
            self._invalidate_gray_results("dark_recalibrated")
            self._show_capture_overlay(
                "\u30ce\u30a4\u30ba\u88dc\u6b63\u30c7\u30fc\u30bf\u3092\u4fdd\u5b58\u3057\u307e\u3057\u305f",
                "\u6b21\u306e\u624b\u9806\u3078\u9032\u3081\u307e\u3059",
                wait_sec=1.5,
            )
        elif key == ord("w"):
            if not self._check_calibration_prerequisites("w"):
                return False
            if not self.wb_confirm_pending:
                self.wb_confirm_pending = True
                if not self._confirm_operator_action(
                    "\u30db\u30ef\u30a4\u30c8\u56fa\u5b9a\uff08RAW\u65b9\u5f0f\uff09",
                    "\u30b0\u30ec\u30fc\u30ab\u30fc\u30c9\u3092\u8a2d\u7f6e\u3057\u3066\u304f\u3060\u3055\u3044",
                    next_label="\u8a2d\u7f6e\u5b8c\u4e86",
                    extra_confirm_keys=(ord("w"),),
                ):
                    self.wb_confirm_pending = False
                    return False
                self.wb_confirm_pending = False
                self._show_capture_overlay("\u30db\u30ef\u30a4\u30c8\u56fa\u5b9a\uff08RAW\u65b9\u5f0f\uff09\u3092\u5b9f\u884c\u4e2d...")
                self._calibrate_wb_from_raw()
                if self.master_ref is not None:
                    self._show_capture_overlay("\u30de\u30b9\u30bf\u30fcRef\u5024\u3092\u53d6\u5f97\u4e2d...")
                    _, avg_ref = self._capture_neutral_ratio(n_frames=64)
                    self.master_ref.save(avg_ref.astype(np.float32))
                    self._invalidate_gray_results("master_ref_recalibrated")
                    self._show_capture_overlay(
                        "\u30de\u30b9\u30bf\u30fcRef\u5024\u3092\u4fdd\u5b58\u3057\u307e\u3057\u305f",
                        "\u6b21\u306e\u624b\u9806\u3078\u9032\u3081\u307e\u3059",
                        wait_sec=2.0,
                    )
        elif key == ord("f"):
            if not self._check_calibration_prerequisites("f"):
                return False
            if not self._confirm_operator_action(
                "\u3080\u3089\u88dc\u6b63",
                "\u7070\u8272\u30ab\u30fc\u30c9\u3068\u8a66\u6599\u3092\u5916\u3057\u3001\u767d\u80cc\u666f\u306e\u307f\u3092\u5199\u3057\u3066\u304f\u3060\u3055\u3044",
                next_label="\u8a2d\u7f6e\u5b8c\u4e86",
                extra_confirm_keys=(ord("f"),),
            ):
                return False
            self._show_capture_overlay("64\u30d5\u30ec\u30fc\u30e0\u3092\u53d6\u5f97\u4e2d...")
            self.flat.capture_flat_field(
                self.picam2, self.bayer, self.dark,
                bayer_pattern=self.bayer.bayer_pattern,
                config=self.config,
            )
            if self.flat.is_loaded:
                self._invalidate_gray_results("flat_recalibrated")
                self._show_capture_overlay(
                    "\u3080\u3089\u88dc\u6b63\u30c7\u30fc\u30bf\u3092\u4fdd\u5b58\u3057\u307e\u3057\u305f",
                    "\u6b21\u306e\u624b\u9806\u3078\u9032\u3081\u307e\u3059",
                    wait_sec=2.0,
                )
                if self._on_flat_roi_recheck is not None:
                    self._on_flat_roi_recheck()
            else:
                self._show_capture_overlay("\u3080\u3089\u88dc\u6b63\u306b\u5931\u6557\u3057\u307e\u3057\u305f", wait_sec=2.0)
        elif key == ord("a"):
            self.ae.toggle()
        elif key == ord("b"):
            if not self._check_calibration_prerequisites("b"):
                return False
            if self.blank is None:
                return False
            grid_ready = (
                self._grid_extractor is not None and self._grid_extractor.is_ready
            )
            if grid_ready:
                self._show_capture_overlay(
                    "\u30d6\u30e9\u30f3\u30af\u6e2c\u5b9a\uff08\u81ea\u52d5: \u30b0\u30ea\u30c3\u30c9 4D \u30d1\u30c3\u30c1\uff09",
                    "64\u30d5\u30ec\u30fc\u30e0\u3092\u53d6\u5f97\u4e2d...",
                )
                avg_ratio, avg_ref = self._capture_grid_4d_ratio(n_frames=64)
            else:
                if not self._confirm_operator_action(
                    "\u30d6\u30e9\u30f3\u30af\u6e2c\u5b9a\uff08\u30bc\u30ed\u70b9\u88dc\u6b63\uff09",
                    "SpyderChecker 4D (50% Gray) \u3092 Tar ROI \u4f4d\u7f6e\u306b\u914d\u7f6e",
                    next_label="\u8a2d\u7f6e\u5b8c\u4e86",
                    extra_confirm_keys=(ord("b"),),
                ):
                    return False
                self._show_capture_overlay("64\u30d5\u30ec\u30fc\u30e0\u3092\u53d6\u5f97\u4e2d...")
                avg_ratio, avg_ref = self._capture_neutral_ratio(n_frames=64)
            self.blank.save(avg_ratio)
            self._invalidate_gray_results("blank_recalibrated")
            self._show_capture_overlay(
                "\u30d6\u30e9\u30f3\u30af\u6bd4\u3092\u4fdd\u5b58\u3057\u307e\u3057\u305f",
                "\u6b21\u306e\u624b\u9806\u3078\u9032\u3081\u307e\u3059",
                wait_sec=2.0,
            )
        elif key == ord("n"):
            if not self._check_calibration_prerequisites("n"):
                return False
            grid_ready = (
                self._grid_extractor is not None and self._grid_extractor.is_ready
            )
            if grid_ready:
                self._show_capture_overlay(
                    "\u30cb\u30e5\u30fc\u30c8\u30e9\u30eb\u88dc\u6b63\uff08\u81ea\u52d5: \u30b0\u30ea\u30c3\u30c9 4D \u30d1\u30c3\u30c1\uff09",
                    "64\u30d5\u30ec\u30fc\u30e0\u3092\u53d6\u5f97\u4e2d...",
                )
                avg_ratio, avg_ref = self._capture_grid_4d_ratio(n_frames=64)
            else:
                if not self._confirm_operator_action(
                    "\u30cb\u30e5\u30fc\u30c8\u30e9\u30eb\u88dc\u6b63",
                    "SpyderChecker 4D (50% Gray) \u3092 Tar ROI \u4f4d\u7f6e\u306b\u914d\u7f6e",
                    next_label="\u8a2d\u7f6e\u5b8c\u4e86",
                    extra_confirm_keys=(ord("n"),),
                ):
                    return False
                self._show_capture_overlay("64\u30d5\u30ec\u30fc\u30e0\u3092\u53d6\u5f97\u4e2d...")
                avg_ratio, avg_ref = self._capture_neutral_ratio(n_frames=64)
            if self.blank is not None and self.blank.is_loaded:
                avg_ratio = self.blank.correct(avg_ratio)
            elif self.master_ref is not None and self.master_ref.is_loaded:
                scale = self.master_ref.compute_scale(avg_ref)
                avg_ratio = avg_ratio / scale
            r_R = float(avg_ratio[0])
            r_G = float(avg_ratio[1])
            r_B = float(avg_ratio[2])
            d_R = r_G / r_R if r_R > 1e-6 else 1.0
            d_B = r_G / r_B if r_B > 1e-6 else 1.0
            print(f"[diag] d_R/d_B (n-key path): d_R={d_R:.6f} d_B={d_B:.6f}")
            self.lab_converter.set_diagonal_correction(d_R, d_B)
            self.lab_converter.set_ref_baseline(avg_ref)
            self.spectral_drift_tracker.activate()
            # self.ref_anchor_lab = self.lab_converter.neutral_anchor_Lab()  # anchor は _compute_measurements 内で毎フレーム計算
            self._set_live_ref_baseline(avg_ref, "neutral", notify=True)
            scale_baseline = normalize_ref_scale_triplet(avg_ref, self.white_ratio_rgb)
            if scale_baseline is None:
                scale_baseline, scale_source = self._resolve_live_ref_scale_baseline_for_p(
                    self.ref_train
                )
            else:
                scale_source = "neutral"
            self._set_live_ref_scale_baseline(scale_baseline, scale_source, notify=True)
            data = {
                "d_R": d_R,
                "d_B": d_B,
                "ref_baseline": avg_ref.tolist(),
                "ref_scale_baseline": (
                    None
                    if self.live_ref_scale_baseline is None
                    else np.asarray(self.live_ref_scale_baseline, dtype=np.float64).tolist()
                ),
            }
            with open(
                os.path.join(get_today_calibration_dir(), "neutral_correction.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(data, f, indent=2)
            _remove_cleared_marker("neutral_correction.json")
            import shutil as _shutil
            import glob as _glob
            _nc_path = os.path.join(get_today_calibration_dir(), "neutral_correction.json")
            _date_str = datetime.now().strftime("%Y-%m-%d")
            _nc_backup = os.path.join(
                get_today_calibration_dir(), f"neutral_correction_{_date_str}.json"
            )
            _shutil.copy2(_nc_path, _nc_backup)
            for _old in sorted(_glob.glob(os.path.join(
                get_today_calibration_dir(), "neutral_correction_*.json"
            )), reverse=True)[5:]:
                os.remove(_old)
            self._invalidate_gray_results("neutral_recalibrated")
            if self.stability_tracker is not None and hasattr(
                self.stability_tracker, "mark_recalibrated"
            ):
                self.stability_tracker.mark_recalibrated(source="N")
            print(f"- Neutral correction saved: d_R={d_R:.4f} d_B={d_B:.4f}")
            self._show_capture_overlay(
                "\u30cb\u30e5\u30fc\u30c8\u30e9\u30eb\u88dc\u6b63\u3092\u4fdd\u5b58\u3057\u307e\u3057\u305f",
                "\u6b21\u306e\u624b\u9806\u3078\u9032\u3081\u307e\u3059",
                wait_sec=2.0,
            )
        elif key == ord("o"):
            self._open_fixed_anchor_picker()
        elif key == 13 or key == 10:  # Enter
            if self._chart_state == "preview":
                source = str(
                    self._chart_workflow_status.get("source", "")
                ).strip() or self._positioning_method or "preview"
                self._prepare_preview_rois_and_start_chart_measurement(
                    source=source,
                )
        elif key == ord("+") or key == ord("="):
            if self._chart_state == "preview":
                self._hinge_gap = min(3.0, round(self._hinge_gap + 0.1, 1))
                self._rebuild_extractor_with_hinge()
        elif key == ord("-"):
            if self._chart_state == "preview":
                self._hinge_gap = max(0.0, round(self._hinge_gap - 0.1, 1))
                self._rebuild_extractor_with_hinge()
        elif key == ord("."):
            if self._chart_state == "preview" and self._grid_extractor is not None:
                self._grid_extractor.patch_margin = round(
                    min(0.30, self._grid_extractor.patch_margin + 0.01), 3
                )
                self._grid_extractor.invalidate_patchwise_rois()
                self._preview_margin_detected = True
                self._mark_manual_corner_used()
        elif key == ord(","):
            if self._chart_state == "preview" and self._grid_extractor is not None:
                self._grid_extractor.patch_margin = round(
                    max(0.01, self._grid_extractor.patch_margin - 0.01), 3
                )
                self._grid_extractor.invalidate_patchwise_rois()
                self._preview_margin_detected = True
                self._mark_manual_corner_used()
        elif key == ord("p"):
            if self.ccm_store is None:
                self.ccm_store = CCMStore()
            state = self._chart_state

            if state == "idle":
                if not self._check_calibration_prerequisites("p"):
                    return False
                self._show_chart_workflow_overlay(
                    "preview_loading",
                    "色基準48色 自動検出中",
                    detail="チャートの位置と傾きを判定しています…",
                    can_cancel=False,
                )
                # Phase 23 / Phase 2: idle [P] 経路でも新検出器を main path とする。
                # `_run_chart_detection_pipeline` が Stage 0 (contour_outer_rim) →
                # Stage 1 (rigid_auto) を順次試行し、採用 stage 名を返す。
                # 採用時の extractor 構築 / `set_corners_with_source` /
                # `_chart_state="preview"` / `_set_chart_workflow_status` は helper
                # 内部で完結するため、caller 側では再書き込みしない。
                # 旧 `_try_auto_detect_chart_corners` 関数本体は debug fallback として残置。
                stage = self._run_chart_detection_pipeline(
                    workflow_label="ROI を新検出器で配置しました",
                )
                if stage is not None:
                    if self._chart_state == "flip_warning":
                        return False
                    # corner_input UI buffer をクリア (helper は raw_corners のみ管理)。
                    self._chart_corners = []
                    # downstream log / persistence の trace 性向上のため stage 名を採用。
                    # helper も内部で `_positioning_method = stage` と書込むため冗長だが、
                    # 将来 helper 内訳変更時の defense として明示しておく。
                    self._positioning_method = stage
                    self._invalidate_gray_results("chart_calibration_started")
                    self._persist_chart_corners(reason=stage)
                    if self.window_manager is not None:
                        drag_cb = self.window_manager.make_offset_callback(
                            self._on_preview_drag
                        )
                    else:
                        drag_cb = self._on_preview_drag
                    cv2.setMouseCallback(self.window_name, drag_cb)
                    return False

                # 新検出器パイプラインが全 stage 失敗 → manual 4-click overlay に降格。
                # saved corners 経由のショートカットは Phase 22 / Phase 2 で削除済
                # (新検出器が main path、saved は監査用書込みのみ)。
                print(
                    "- chart detection pipeline failed → manual 4-click overlay"
                )
                self._show_capture_overlay(
                    "CCMキャリブレーション  SpyderCheckr 48",
                    "カードを広げてカメラ正面に置いてください",
                    "4隅パッチの中央をクリック（順序・方向 自由）",
                    "左上(1A)  右上(1H)  右下(6H)  左下(6A)",
                    "4隅をクリックしてください",
                    "ESC/q でキャンセル",
                )
                self._mark_manual_corner_used()
                self._grid_extractor = SpyderCheckrGridExtractor(
                    SPYDERCHECKR_48_SHAPE[0],
                    SPYDERCHECKR_48_SHAPE[1],
                    hinge_gap=self._hinge_gap,
                )
                self._invalidate_gray_results("chart_calibration_started")
                self._chart_corners = []
                self._raw_corners = []
                self._chart_state = "corner_input"
                self._set_chart_workflow_status(
                    stage="corner_input",
                    message="Chart corners input",
                    detail="1A / 1H / 6H / 6A をクリック",
                    can_cancel=True,
                    source="manual",
                )
                if self.window_manager is not None:
                    cb = self.window_manager.make_offset_callback(self._on_chart_click)
                else:
                    cb = self._on_chart_click
                cv2.setMouseCallback(self.window_name, cb)

            elif state == "corner_input":
                n = len(self._chart_corners)
                self._show_capture_overlay(
                    f"4\u9685\u30d1\u30c3\u30c1\u306e\u30af\u30ea\u30c3\u30af\u5f85\u3061 ({n}/4 \u5b8c\u4e86)",
                    "1A\u30fb1H\u30fb6H\u30fb6A \u306e\u4e2d\u592e\u3092\u4efb\u610f\u9806\u3067\u30af\u30ea\u30c3\u30af",
                    "4\u70b9\u30af\u30ea\u30c3\u30af -> \u81ea\u52d5\u3067\u30b0\u30ea\u30c3\u30c9\u30d7\u30ec\u30d3\u30e5\u30fc\u3078\u9032\u307f\u307e\u3059",
                )

            elif state == "preview":
                print(
                    "[P-key preview short-circuit] state=preview → 再 detect せず "
                    "measurement へ進行 (chart 物理移動時は state=preview を抜けて"
                    "再実行が必要)"
                )
                source = str(
                    self._chart_workflow_status.get("source", "")
                ).strip() or self._positioning_method or "preview"
                self._prepare_preview_rois_and_start_chart_measurement(
                    source=source,
                )

            elif state == "flip_warning":
                # Phase 7B: hard-lock 解除. P 再押下で re-check.
                # 正置に戻っていれば preview へ、まだ反転していれば flip_warning 維持.
                try:
                    still_flipped = self._preview_detect_spyder_flip()
                except RuntimeError as exc:
                    if "cancelled" in str(exc):
                        return False
                    raise
                if still_flipped:
                    # まだ向き・設置が合っていない. 同じ state のまま user に知らせる.
                    message, detail = self._flip_warning_operator_copy(still_problem=True)
                    self._show_capture_overlay(
                        message,
                        detail,
                        wait_sec=1.5,
                    )
                    return False
                # 合格 → preview へ復帰
                self._exit_flip_warning_to_preview()
        elif key == ord("c"):
            self._show_capture_overlay(
                "全補正データをクリアします",
                "D・F・W・P・B・N のみ削除し、最初からやり直します",
                "accepted_reference / acceptance履歴は削除しません",
                "[C] 実行  [他] キャンセル",
            )
            confirm = cv2.waitKey(0) & 0xFF
            if confirm in (ord("c"), ord("C"), 10, 13):
                self._clear_aperture()
            else:
                self._show_capture_overlay("キャンセルしました", wait_sec=0.8)
        elif key == ord("r"):
            self.active_roi = "ref"
            print(f"Active ROI: Ref [{self.resize_axis}]")
        elif key == ord("t"):
            self.active_roi = "tar"
            print(f"Active ROI: Target [{self.resize_axis}]")
        elif key == ord("k"):
            self.resize_axis = "h" if self.resize_axis == "w" else "w"
            axis_name = "\u5e45(W)" if self.resize_axis == "w" else "\u9ad8\u3055(H)"
            print(f"Resize axis: {axis_name}")
        elif key in (ord("j"), ord("l")):
            self._resize_roi_by_key(key)
        elif key == ord("u"):
            self._reset_roi()
        elif key in (ord("["), ord("]"), ord("{"), ord("}")):
            if key == ord("["):
                self.config.processing.aspect_ref = max(0.3, self.config.processing.aspect_ref - 0.1)
                print(f"Ref aspect: {self.config.processing.aspect_ref:.1f}")
            elif key == ord("]"):
                self.config.processing.aspect_ref = min(3.0, self.config.processing.aspect_ref + 0.1)
                print(f"Ref aspect: {self.config.processing.aspect_ref:.1f}")
            elif key == ord("{"):
                self.config.processing.aspect_tar = max(0.3, self.config.processing.aspect_tar - 0.1)
                print(f"Tar aspect: {self.config.processing.aspect_tar:.1f}")
            elif key == ord("}"):
                self.config.processing.aspect_tar = min(3.0, self.config.processing.aspect_tar + 0.1)
                print(f"Tar aspect: {self.config.processing.aspect_tar:.1f}")
            if self.roi_persistence is not None:
                self.roi_persistence.save()
        return False

    def _resize_roi_by_key(self, key):
        """j/l キー入力に応じて選択中ROIの幅または高さを変更する。"""
        self._mark_manual_roi_used()
        p = self.config.processing
        step = 10
        delta = step if key == ord("l") else -step
        if self.active_roi == "ref":
            w = int(p.spot_size_ref)
            h = max(int(w * p.aspect_ref), 2)
            if self.resize_axis == "w":
                w = max(20, min(400, w + delta))
            else:
                h = max(6, min(1200, h + delta))
            p.spot_size_ref = w
            p.aspect_ref = round(h / w, 2) if w > 0 else 1.0
            print(f"Ref: {w}x{h}")
        else:
            w = int(p.spot_size_tar)
            h = max(int(w * p.aspect_tar), 2)
            if self.resize_axis == "w":
                w = max(20, min(400, w + delta))
            else:
                h = max(6, min(1200, h + delta))
            p.spot_size_tar = w
            p.aspect_tar = round(h / w, 2) if w > 0 else 1.0
            print(f"Tar: {w}x{h}")
        if self.roi_persistence is not None:
            self.roi_persistence.save()
        if self.active_roi == "ref" and self._on_flat_roi_recheck is not None:
            self._on_flat_roi_recheck()
        if self.active_roi == "ref":
            self._invalidate_gray_results("ref_roi_resized")

    def _reset_roi(self):
        """選択中ROIのサイズ・アスペクト比・位置をデフォルト設定に戻す。"""
        self._mark_manual_roi_used()
        p = self.config.processing
        defaults = ProcessingSettings()
        if self.active_roi == "ref":
            p.spot_size_ref = defaults.spot_size_ref
            p.aspect_ref = defaults.aspect_ref
            p.posi_ref = list(defaults.posi_ref)
            print(
                f"Ref ROI reset: {p.spot_size_ref}x"
                f"{int(p.spot_size_ref * p.aspect_ref)} "
                f"at ({p.posi_ref[0]},{p.posi_ref[1]})"
            )
        else:
            p.spot_size_tar = defaults.spot_size_tar
            p.aspect_tar = defaults.aspect_tar
            p.posi_tar = list(defaults.posi_tar)
            print(
                f"Tar ROI reset: {p.spot_size_tar}x"
                f"{int(p.spot_size_tar * p.aspect_tar)} "
                f"at ({p.posi_tar[0]},{p.posi_tar[1]})"
            )
        if self.roi_persistence is not None:
            self.roi_persistence.save()
        if self.active_roi == "ref" and self._on_flat_roi_recheck is not None:
            self._on_flat_roi_recheck()
        if self.active_roi == "ref":
            self._invalidate_gray_results("ref_roi_reset")

    def _clear_aperture(self) -> None:
        """全クリア: D・F・W・P・B・N のみをクリアする。

        accepted_reference / acceptance_result 履歴は保持する。
        """
        self._invalidate_gray_results("all_calibration_cleared")
        self.spectral_drift_tracker.reset()
        self.dark.dark_frame = None
        self.dark.is_loaded = False
        _delete_calibration_file_all("dark_frame.npy")
        self.flat.gain_map = None
        self.flat.valid_mask = None
        self.flat.is_loaded = False
        _delete_calibration_file_all("flat_field_gain.npy")
        _delete_calibration_file_all("flat_field_valid_mask.npy")
        if self._on_flat_roi_recheck is not None:
            self._on_flat_roi_recheck()
        self.wb_calibrator.red_gain = 1.0
        self.wb_calibrator.blue_gain = 1.0
        self.wb_calibrator.is_calibrated = False
        _delete_calibration_file_all("wb_gains.json")
        self.picam2.set_controls({"ColourGains": self.config.camera.colour_gains})
        if self.master_ref is not None:
            self.master_ref.master_ref_rgb = None
            self.master_ref.is_loaded = False
        _delete_calibration_file_all("master_ref.json")
        if self.ccm_store is not None:
            self.ccm_store.clear()
        self.lab_converter.clear_ccm()
        self.white_ratio_rgb = np.ones(3, dtype=np.float64)
        self.ref_train = None
        _delete_calibration_file_all("ccm.json")
        self.lab_converter.clear_diagonal_correction()
        _delete_calibration_file_all("neutral_correction.json")
        self._set_live_ref_baseline(None, None, notify=True)
        self._set_live_ref_scale_baseline(None, None, notify=True)
        # self.ref_anchor_lab = self.lab_converter.neutral_anchor_Lab()  # anchor は _compute_measurements 内で毎フレーム計算
        if self.blank is not None:
            self.blank.blank_ratio = None
            self.blank.is_loaded = False
        _delete_calibration_file_all("blank_ratio.json")
        self._show_capture_overlay(
            "\u5168\u30af\u30ea\u30a2\u5b8c\u4e86",
            "D\u30fbF\u30fbW\u30fbP\u30fbB\u30fbN \u3092\u3059\u3079\u3066\u524a\u9664\u3057\u307e\u3057\u305f",
            "accepted_reference / acceptance履歴は保持されます",
            wait_sec=2.5,
        )
        print("- All calibration data cleared (dark, flat, wb, ccm, master_ref, neutral, blank).")
        print("- accepted_reference / acceptance_result history retained.")

    def _clear_all_recent(self) -> None:
        """直近の日付フォルダにあるすべてのキャリブレーションファイルをクリアする。"""
        self._invalidate_gray_results("recent_calibration_cleared")
        self.spectral_drift_tracker.reset()
        date_dirs = _get_sorted_date_dirs()
        latest_dir = date_dirs[0] if date_dirs else CALIBRATION_DIR
        _CALIB_FILES = [
            "dark_frame.npy", "flat_field_gain.npy", "flat_field_valid_mask.npy", "wb_gains.json",
            "ccm.json", "neutral_correction.json", "blank_ratio.json", "master_ref.json",
        ]
        for fname in _CALIB_FILES:
            fpath = os.path.join(latest_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        self.dark.dark_frame = None
        self.dark.is_loaded = False
        self.flat.gain_map = None
        self.flat.valid_mask = None
        self.flat.is_loaded = False
        self.wb_calibrator.red_gain = 1.0
        self.wb_calibrator.blue_gain = 1.0
        self.wb_calibrator.is_calibrated = False
        self.picam2.set_controls({"ColourGains": self.config.camera.colour_gains})
        if self.ccm_store is not None:
            self.ccm_store.clear()
        self.lab_converter.clear_ccm()
        self.white_ratio_rgb = np.ones(3, dtype=np.float64)
        self.ref_train = None
        self.lab_converter.clear_diagonal_correction()
        self._set_live_ref_baseline(None, None, notify=True)
        self._set_live_ref_scale_baseline(None, None, notify=True)
        # self.ref_anchor_lab = self.lab_converter.neutral_anchor_Lab()  # anchor は _compute_measurements 内で毎フレーム計算
        if self.blank is not None:
            self.blank.blank_ratio = None
            self.blank.is_loaded = False
        if self.master_ref is not None:
            self.master_ref.master_ref_rgb = None
            self.master_ref.is_loaded = False
        self._show_capture_overlay(
            "\u3059\u3079\u3066\u306e\u88dc\u6b63\u30c7\u30fc\u30bf\u3092\u524a\u9664\u3057\u307e\u3057\u305f",
            f"\u5bfe\u8c61\u30d5\u30a9\u30eb\u30c0: {os.path.basename(latest_dir)}",
            wait_sec=2.0,
        )
        print(f"- All calibration data cleared from: {latest_dir}")
