"""
CSIメインモジュール。

CSIVideoProcessorクラスとmain()エントリポイントを提供する。
フレーム処理パイプライン（取得→ROI抽出→補正→Lab変換→描画→ログ）を統括する。
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .colorimeter_common import (
    _find_calibration_file,
    _remove_cleared_marker,  # noqa: F401 - kept for external import compatibility
    DisplaySettings,
    MeasurementMode,
    draw_mode_toast,
    SystemConfig,
    DoubleBeamProcessor,
    CIELABConverter,
    CIEDE2000Calculator,
    StabilityMonitor,
    DarkFlatStabilityTracker,
    RefSpatialAnalyzer,
    SpectralDriftTracker,
    FrameTransformer,
    LiveColorCorrector,
    WindowManager,
    DarkFrameManager,
    WhiteBalanceCalibrator,
    SessionRecorder,
    HistogramQualityChecker,
    ROIMouseHandler,
    ROIConfigPersistence,
    BlankRatioManager,
    MasterRefManager,
    CCMStore,
    CCMVerifier,
    ensure_calibration_dir,
    get_today_calibration_dir,  # noqa: F401 - kept for external import compatibility
    SPYDERCHECKR_48_WHITE_PATCH_INDEX,
    SPYDERCHECKR_48_GRAY_4D_INDEX,  # noqa: F401 - kept for external import compatibility
    compute_gain_percentile,
    make_chart_analysis_composite,
    normalize_acceptance_run_type,
    normalize_ref_scale_triplet,
    run_canonical_lab_pipeline,
    RelativeGrayVerifier,
    select_live_ref_baseline,
    select_live_ref_scale_baseline,
    split_neutral_correction_baselines,
    get_software_identity,
)

from csi.camera import (
    CSICameraSettings,  # noqa: F401 - kept for external import compatibility
    CSIConfig,
    CSICameraConnector,
    BayerROIExtractor,
    FlatFieldManager,
    AdaptiveExposureController,
)
from csi.overlay import CSIOverlayRenderer
from csi.logger import CSIMeasurementLogger
from csi.key_handler import KeyHandler

REF_SCALE_WARN_DELTA = 0.05
LAB_REL_L_WARN_MIN = 9.5
LAB_REL_L_WARN_MAX = 11.0
ACCEPTANCE_RUNNER_ENTRYPOINT = "csi.main.run_acceptance_runner"

# Phase 23 / Phase 1: Preview badge 用 source → 表示 tag mapping。
# `_set_chart_workflow_status(source=...)` で書き込まれる source 値を
# UI 上の短い tag に変換する。未登録 / 空文字列は `_preview_badge_source_label`
# 側で `"preview"` に fallback。
_PREVIEW_BADGE_LABEL_MAP: dict[str, str] = {
    "oriented_panel_lattice": "panel",
    "contour_outer_rim": "contour",
    "rigid_auto": "rigid",
    "auto": "auto",
    "legacy": "auto",
    "saved": "saved",
    "manual": "manual",
    "inner_cell_hough": "hough",
}


_CHART_ORIENTATION_BADGE_LINES: dict[str, tuple[str, ...]] = {
    "visual": ("ORIENT: UPRIGHT",),
    "visual_180": ("ORIENT: UPSIDE-DOWN", "REMAP APPLIED"),
}


def _preview_badge_source_label(source: str) -> str:
    """Preview badge 用 source 表示文字列を返す。

    workflow_status の source 値 (`contour_outer_rim` / `rigid_auto` /
    `auto` / `legacy` / `saved` / `manual` / `inner_cell_hough`) を
    badge 表示用の短い tag に変換する。未知 / 空文字列は `"preview"` に
    fallback (Phase 23 / Phase 1)。

    引数の前後 whitespace は `strip()` で除去するため、
    workflow_status 由来の余計な空白が混じっても安定して動作する。
    """
    return _PREVIEW_BADGE_LABEL_MAP.get(source.strip(), "preview")


def _chart_orientation_badge_lines(workflow_status: dict | None) -> tuple[str, ...]:
    """Main camera badge lines for SpyderCHECKR orientation state."""
    if not workflow_status:
        return ()
    orientation_order = str(
        workflow_status.get("chart_orientation_order", "")
    ).strip()
    if not orientation_order:
        return ()
    return _CHART_ORIENTATION_BADGE_LINES.get(
        orientation_order,
        (f"ORIENT: {orientation_order}",),
    )


def _finite_correction_value(value: float | None) -> float | None:
    """Correction Applied 系の有限値を返す。欠損/非数は None。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def is_correction_applied_axis_warn(value: float | None) -> bool:
    """Correction Applied の単一軸が 1.00±許容幅を外れているか。"""
    number = _finite_correction_value(value)
    return (
        number is not None
        and abs(number - 1.0) > (REF_SCALE_WARN_DELTA + 1e-9)
    )


def is_ref_scale_warn(ref_scale: float | None) -> bool:
    """ref_scale が暫定警告閾値を外れているかを返す。"""
    return is_correction_applied_axis_warn(ref_scale)


def build_correction_applied_warning(
    f_R: float | None,
    f_B: float | None,
    ref_scale: float | None,
) -> dict | None:
    """Correction Applied の R/B/Y 警告 payload を作る。"""
    values = {
        axis: number
        for axis, number in (
            ("R", _finite_correction_value(f_R)),
            ("B", _finite_correction_value(f_B)),
            ("Y", _finite_correction_value(ref_scale)),
        )
        if number is not None
    }
    axes = [
        axis
        for axis in ("R", "B", "Y")
        if axis in values and is_correction_applied_axis_warn(values[axis])
    ]
    if not axes:
        return None
    detail = " ".join(
        f"{axis}{values[axis]:.2f}" for axis in ("R", "B", "Y") if axis in values
    )
    return {
        "status": "WARN",
        "title": "補正ドリフト警告",
        "message": "色または明るさが基準からずれています",
        "action": "vでGray Check / 必要なら再校正",
        "axes": axes,
        "values": values,
        "detail": detail,
    }


def is_lab_rel_l_warn(lab_rel_l: float) -> bool:
    """Lab_rel L が暫定警告帯を外れているかを返す。"""
    return float(lab_rel_l) < LAB_REL_L_WARN_MIN or float(lab_rel_l) > LAB_REL_L_WARN_MAX


def get_acceptance_runner_entry(run_type: str | None) -> dict:
    """運用可否判定 runner の開始点と run_type を返す。"""
    normalized = normalize_acceptance_run_type(run_type)
    return {
        "entrypoint": ACCEPTANCE_RUNNER_ENTRYPOINT,
        "run_type": normalized,
    }


def run_acceptance_runner(proc: "CSIVideoProcessor", run_type: str | None) -> dict | None:
    """運用可否判定 runner の開始点。"""
    normalized = normalize_acceptance_run_type(run_type)
    dispatch = {
        "start_of_day": proc.key_handler.run_start_of_day_sequence,
        "end_of_day": proc.key_handler.run_end_of_day_sequence,
        "requalification": proc.key_handler.run_requalification_sequence,
        "verify_only": proc.key_handler.run_verify_only_sequence,
    }
    return dispatch[normalized]()


class CSIVideoProcessor:
    """
    CSIのフレーム処理パイプラインを統括するクラス。

    機能:
      - 設定読込、カメラ接続、計測系・表示系サブシステムの初期化を行う。
      - RAW取得→ROI抽出→ダーク/フラット補正→Lab変換→描画→ログのループを実行する。
      - 画面表示、動画/静止画保存、CSVログ出力、キー入力処理を統括する。
    入力:
      - CSIカメラRAWフレーム、`CSIConfig` 設定値、計測補助クラス群、永続化済みROI設定。
      - キー入力、ROI状態、ダーク/フラット補正状態、露出制御状態、品質判定結果。
    出力:
      - Lab/ΔE/警告を含む計測結果、描画済みフレーム、保存成果物、終了判定を生成する。
    """

    def __init__(self):
        """設定読み込みから各サブシステム初期化までを行う。"""
        self._camera_init_time = time.time()
        ensure_calibration_dir()
        self.config = CSIConfig()
        self.mode = MeasurementMode()
        self.last_mode_change_time = 0.0
        self._last_recalib_reminder_time = 0.0
        self.FPS = self.config.camera.frame_rate
        self.target_Lab = np.array(self.config.measurement.target_Lab, dtype=np.float32)
        self.software_identity = get_software_identity()

        self.roi_persistence = ROIConfigPersistence(self.config)
        self.roi_persistence.load()

        self._init_camera()
        self._check_flat_ref_roi_gain()
        self._init_measurement()
        self._init_logger()
        self.results_dir = self.logger.log_dir
        self.session_recorder = SessionRecorder(self.results_dir)
        self._init_display()
        self.live_color_corrector = LiveColorCorrector()
        self.key_handler = KeyHandler(
            self.FPS,
            self.logger,
            self.dark,
            self.flat,
            self.bayer,
            self.ae,
            self.picam2,
            self.config,
            window_name=self.window_manager.window_name,
            roi_persistence=self.roi_persistence,
            mode=self.mode,
            wb_calibrator=self.wb_calibrator,
            session_recorder=self.session_recorder,
            stability_tracker=self.df_tracker,
            lab_converter=self.lab_converter,
            stability=self.stability,
            blank=self.blank,
            master_ref=self.master_ref,
            window_manager=self.window_manager,
            spectral_drift_tracker=self.spectral_drift_tracker,
            gray_verifier=self.gray_verifier,
            ccm_verifier=self.ccm_verifier,
            camera_init_time=self.get_camera_init_time(),
        )
        # roi_mouse を key_handler に注入
        self.key_handler.roi_mouse = self.roi_mouse
        # CCMキャリブ完了時に測定パイプラインを即時同期するコールバック (BUG 8 修正)
        self.key_handler._on_ccm_calibrated = self._sync_ccm_to_pipeline
        self.key_handler._on_live_ref_baseline_changed = self._sync_live_ref_baseline
        self.key_handler._on_live_ref_scale_baseline_changed = (
            self._sync_live_ref_scale_baseline
        )
        self.key_handler._set_live_ref_baseline(
            self.live_ref_baseline,
            self.live_ref_baseline_source,
        )
        self.key_handler._set_live_ref_scale_baseline(
            self.live_ref_scale_baseline,
            self.live_ref_scale_baseline_source,
        )
        # 4D 相対検証完了時に結果を同期するコールバック
        self.key_handler._on_gray_verified = self._sync_gray_verify_result
        self.key_handler._on_gray_absolute_verified = (
            self._sync_gray_absolute_verify_result
        )
        self.key_handler._on_acceptance_result_persisted = (
            self._sync_acceptance_result
        )
        self.key_handler._on_gray_results_invalidated = self._invalidate_gray_results
        self.key_handler._on_flat_roi_recheck = self._check_flat_ref_roi_gain
        self.roi_mouse._on_ref_roi_moved = self._on_ref_roi_moved
        self.roi_mouse._on_roi_changed = self.key_handler._mark_manual_roi_used

    def _sync_ccm_to_pipeline(
        self, white_ratio_rgb: np.ndarray, ref_train: "np.ndarray | None",
    ) -> None:
        """CCMキャリブ完了後に white_ratio_rgb / ref_train を測定パイプラインに反映する。

        live ref_scale baseline は別 callback で同期し、この関数では上書きしない。
        """
        self.white_ratio_rgb = white_ratio_rgb.copy()
        self.ref_train = ref_train.copy() if ref_train is not None else None
        self._invalidate_gray_results("ccm_pipeline_synced")
        print("  [sync] white_ratio_rgb / ref_train を測定パイプラインに反映")

    def _sync_live_ref_baseline(
        self,
        baseline: np.ndarray | None,
        source: str | None,
    ) -> None:
        """raw neutral baseline state を同期する。"""
        if baseline is None or source is None:
            self.live_ref_baseline = None
            self.live_ref_baseline_source = None
            self.lab_converter.clear_ref_baseline()
        else:
            self.live_ref_baseline = np.asarray(baseline, dtype=np.float64).copy()
            self.live_ref_baseline_source = source
            self.lab_converter.set_ref_baseline(self.live_ref_baseline)
        self._invalidate_gray_results("live_ref_baseline_changed")
        print(
            "  [sync] live_ref_baseline を測定パイプラインに反映:"
            f" source={self.live_ref_baseline_source or 'none'}"
        )

    def _sync_live_ref_scale_baseline(
        self,
        baseline: np.ndarray | None,
        source: str | None,
    ) -> None:
        """live ref_scale 用 baseline state を同期する。"""
        if baseline is None or source is None:
            self.live_ref_scale_baseline = None
            self.live_ref_scale_baseline_source = None
        else:
            self.live_ref_scale_baseline = np.asarray(baseline, dtype=np.float64).copy()
            self.live_ref_scale_baseline_source = source
        self._invalidate_gray_results("live_ref_scale_baseline_changed")
        print(
            "  [sync] live_ref_scale_baseline を測定パイプラインに反映:"
            f" source={self.live_ref_scale_baseline_source or 'none'}"
        )

    def _sync_gray_verify_result(self, result: dict) -> None:
        """4D 相対検証完了後に結果を測定パイプラインに反映する。"""
        result["is_stale"] = False
        result["stale_reason"] = None
        result["stale_at"] = None
        result.setdefault("verified_at", datetime.now().isoformat(timespec="seconds"))
        self.last_gray_verify_result = result

    def _sync_gray_absolute_verify_result(self, result: dict) -> None:
        """4D 絶対検証完了後に結果を測定パイプラインに反映する。"""
        result["is_stale"] = False
        result["stale_reason"] = None
        result["stale_at"] = None
        result.setdefault("verified_at", datetime.now().isoformat(timespec="seconds"))
        self.last_gray_absolute_verify_result = result

    def _sync_acceptance_result(self, result: dict) -> None:
        """acceptance_result を current run の結果として同期する。"""
        result["is_stale"] = False
        result["stale_reason"] = None
        result["stale_at"] = None
        result["is_current"] = True
        result.setdefault(
            "verified_at",
            result.get("timestamp", datetime.now().isoformat(timespec="seconds")),
        )
        self.last_acceptance_result = result

    def _invalidate_gray_results(self, reason: str) -> None:
        """保持中の4D検証結果を stale として扱う。"""
        stale_at = datetime.now().isoformat(timespec="seconds")
        for attr_name in (
            "last_gray_verify_result",
            "last_gray_absolute_verify_result",
            "last_acceptance_result",
        ):
            result = getattr(self, attr_name, None)
            if result is None:
                continue
            result["is_stale"] = True
            result["stale_reason"] = reason
            result["stale_at"] = stale_at
            if attr_name == "last_acceptance_result":
                result["is_current"] = False

    def _on_ref_roi_moved(self) -> None:
        """Ref ROI 移動時の後処理。"""
        self._check_flat_ref_roi_gain()
        self._invalidate_gray_results("ref_roi_moved")

    def _get_ref_roi_raw_rect(self) -> tuple[int, int, int, int] | None:
        """現在の Ref ROI を RAW 座標へ変換して返す。"""
        try:
            ref_disp_x, ref_disp_y = self.config.processing.posi_ref
            ref_w = max(int(self.config.processing.spot_size_ref), 2)
            ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
            metadata = self.picam2.capture_metadata()
            return self.bayer.display_to_raw_coords(
                ref_disp_x,
                ref_disp_y,
                ref_w,
                ref_h,
                metadata,
                self.config.display.flip_horizontal,
                self.config.display.flip_vertical,
            )
        except Exception:
            return None

    def _compute_flat_ref_p95(self) -> float | None:
        """flat gain map 上の Ref ROI local p95 を返す。"""
        if not hasattr(self, "flat") or not self.flat.is_loaded or self.flat.gain_map is None:
            return None
        raw_rect = self._get_ref_roi_raw_rect()
        if raw_rect is None:
            return None
        return compute_gain_percentile(self.flat.gain_map, 95, roi=raw_rect)

    def _check_flat_ref_roi_gain(self) -> None:
        """Ref ROI 領域の flat gain 最大値を計算し、閾値超過時に警告する。"""
        if not hasattr(self, "flat") or not self.flat.is_loaded:
            return
        if not hasattr(self, "bayer") or self.flat.gain_map is None:
            return
        self.flat_ref_roi_mask_coverage = None
        try:
            # ディスプレイ座標を取得
            ref_disp_x, ref_disp_y = self.config.processing.posi_ref
            ref_w = max(int(self.config.processing.spot_size_ref), 2)
            ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
            # メタデータ取得（ScalerCrop が必要）
            metadata = self.picam2.capture_metadata()
            # RAW 座標に変換
            rx, ry, rw, rh = self.bayer.display_to_raw_coords(
                ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                self.config.display.flip_horizontal,
                self.config.display.flip_vertical,
            )
            # gain_map 領域を切り出し
            gain_map = self.flat.gain_map
            gy_end = min(ry + rh, gain_map.shape[0])
            gx_end = min(rx + rw, gain_map.shape[1])
            ry = max(ry, 0)
            rx = max(rx, 0)
            if gy_end <= ry or gx_end <= rx:
                return
            roi_gain = gain_map[ry:gy_end, rx:gx_end]
            self.flat_ref_roi_gain_max = float(roi_gain.max())
            if getattr(self.flat, "valid_mask", None) is not None:
                roi_mask = self.flat.valid_mask[ry:gy_end, rx:gx_end]
                if roi_mask.size > 0:
                    self.flat_ref_roi_mask_coverage = float(roi_mask.mean())
        except Exception:
            # 座標変換失敗時は None のまま、警告は出さない
            self.flat_ref_roi_gain_max = None
            self.flat_ref_roi_mask_coverage = None
            return
        if (
            self.flat_ref_roi_mask_coverage is not None
            and self.flat_ref_roi_mask_coverage < 0.99
        ):
            print(
                f"\u26a0 Flat field: Ref ROI が有効視野 mask の外周にかかっています "
                f"(coverage={self.flat_ref_roi_mask_coverage*100:.1f}%) — "
                f"Ref ROI を有効視野の内側へ収めてください"
            )
        # 閾値チェック
        if self.flat_ref_roi_gain_max is not None and self.flat_ref_roi_gain_max > 3.5:
            print(
                f"\u26a0 Flat field: Ref ROI 領域の gain max="
                f"{self.flat_ref_roi_gain_max:.3f} > 3.5 — "
                f"Ref ROI を有効視野の内側へ収めてください"
            )

    def _init_camera(self):
        """カメラ接続・Bayer処理・ダークフレーム・露出制御・品質チェッカーを初期化する。"""
        self.connector = CSICameraConnector(self.config)
        self.picam2 = self.connector.connect()
        cam_cfg = self.picam2.camera_configuration()
        self.main_stream_format = str(cam_cfg.get("main", {}).get("format", "")).upper()
        print(f"- Main stream format: {self.main_stream_format or 'UNKNOWN'}")
        sensor_size = self.picam2.camera_properties.get(
            "PixelArraySize", self.config.camera.raw_size
        )
        self.bayer = BayerROIExtractor(self.config, sensor_size=sensor_size)
        self.dark = DarkFrameManager()
        self.dark.load_if_exists()
        self.flat = FlatFieldManager()
        self.flat.load_if_exists()
        self.ae = AdaptiveExposureController(self.picam2, self.config)
        self.wb_calibrator = WhiteBalanceCalibrator()
        if self.wb_calibrator.load_if_exists():
            gains = self.wb_calibrator.get_gains()
            self.picam2.set_controls({"ColourGains": gains})
            print(f"- WB gains applied: R={gains[0]:.4f} B={gains[1]:.4f}")
        self.quality = HistogramQualityChecker(
            center_edge_diff_fraction=self.config.processing.center_edge_warn_threshold
        )
        self.ref_spatial_analyzer = RefSpatialAnalyzer(
            saturated_threshold=self.quality.SATURATED_THRESHOLD,
            saturated_fraction=self.quality.SATURATED_FRACTION,
            underexposed_threshold=self.quality.UNDEREXPOSED_THRESHOLD,
            underexposed_fraction=self.quality.UNDEREXPOSED_FRACTION,
        )

    def _init_measurement(self):
        """計測パイプラインを初期化する。"""
        self.double_beam = DoubleBeamProcessor()
        self.lab_converter = CIELABConverter(self.config.measurement.reflectance_factor)
        # self.ref_anchor_lab: anchor は _compute_measurements 内で毎フレーム計算するため不要
        self.de2000 = CIEDE2000Calculator()
        self.stability = StabilityMonitor(
            buffer_size=self.config.measurement.temporal_buffer_size,
            cv_threshold=self.config.measurement.stability_cv_threshold,
            sigma_threshold=self.config.measurement.stability_sigma_threshold,
        )
        self.df_tracker = DarkFlatStabilityTracker()
        self.spectral_drift_tracker = SpectralDriftTracker()
        self.blank = BlankRatioManager()
        self.blank.load_if_exists()
        self.master_ref = MasterRefManager()
        self.master_ref.load_if_exists()
        neutral_ref_baseline = None
        neutral_ref_scale_baseline = None
        neutral_correction_record = None

        # ニュートラル補正の復元
        neutral_path = _find_calibration_file("neutral_correction.json")
        if neutral_path is not None:
            try:
                with open(neutral_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                neutral_correction_record = data
                self.lab_converter.set_diagonal_correction(data["d_R"], data["d_B"])
                print(
                    f"- Neutral correction loaded:"
                    f" d_R={data['d_R']:.4f} d_B={data['d_B']:.4f}"
                )
                # self.ref_anchor_lab: anchor は _compute_measurements 内で毎フレーム計算
                self.spectral_drift_tracker.activate()
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: neutral correction file broken: {e}")

        # CCM の復元
        self.ccm_store = CCMStore()
        ccm_matrix = self.ccm_store.load()
        self.white_ratio_rgb = np.ones(3, dtype=np.float64)
        self.ref_train = None
        self.live_ref_baseline = None
        self.live_ref_baseline_source = None
        self.live_ref_scale_baseline = None
        self.live_ref_scale_baseline_source = None
        if ccm_matrix is not None:
            self.lab_converter.set_ccm(ccm_matrix)
            self.white_ratio_rgb = self.ccm_store.white_ratio_rgb
            self.ref_train = self.ccm_store.ref_train
            if (self.ccm_store.residual_ratios is not None
                    and self.ccm_store.residual_labs is not None):
                self.lab_converter.set_residuals(
                    self.ccm_store.residual_ratios,
                    self.ccm_store.residual_labs,
                )
                print(f"  Residual table loaded: {len(self.ccm_store.residual_ratios)} patches")
        if neutral_correction_record is not None:
            neutral_ref_baseline, neutral_ref_scale_baseline = (
                split_neutral_correction_baselines(
                    neutral_correction_record,
                    self.white_ratio_rgb,
                )
            )
        live_baseline, live_source = select_live_ref_baseline(
            neutral_ref_baseline,
            self.master_ref.master_ref_rgb if self.master_ref.is_loaded else None,
            self.ref_train,
        )
        self._sync_live_ref_baseline(
            live_baseline,
            None if live_source == "none" else live_source,
        )
        live_scale_baseline, live_scale_source = select_live_ref_scale_baseline(
            neutral_ref_scale_baseline,
            self.master_ref.master_ref_rgb if self.master_ref.is_loaded else None,
            self.ref_train,
            self.white_ratio_rgb,
        )
        self._sync_live_ref_scale_baseline(
            live_scale_baseline,
            None if live_scale_source == "none" else live_scale_source,
        )
        self.ccm_verifier = CCMVerifier()
        self.gray_verifier = RelativeGrayVerifier()
        self.last_gray_verify_result: dict | None = None
        self.last_gray_absolute_verify_result: dict | None = None
        self.last_acceptance_result: dict | None = None
        self.ref_scale_warn = False
        self.lab_rel_l_warn = False
        self.flat_ref_roi_gain_max: float | None = None
        self.flat_ref_roi_mask_coverage: float | None = None
        self._check_calibration_consistency()
        self._print_calib_summary()

    def _check_calibration_consistency(self) -> None:
        """CCM と Blank のタイムスタンプ整合性をチェックする。"""
        ccm_created_str = None
        if hasattr(self, "ccm_store") and self.ccm_store is not None:
            ccm_created_str = self.ccm_store.metadata.get("created")

        blank_created_str = None
        if hasattr(self, "blank") and self.blank is not None and self.blank.is_loaded:
            blank_path = _find_calibration_file("blank_ratio.json")
            if blank_path is not None:
                try:
                    with open(blank_path, "r", encoding="utf-8") as f:
                        blank_data = json.load(f)
                    blank_created_str = blank_data.get("created")
                except (json.JSONDecodeError, OSError):
                    pass

        if ccm_created_str is None or blank_created_str is None:
            return

        try:
            ccm_dt = datetime.fromisoformat(ccm_created_str)
            blank_dt = datetime.fromisoformat(blank_created_str)
            diff_hours = abs((ccm_dt - blank_dt).total_seconds()) / 3600.0
            if diff_hours > 24.0:
                print(
                    f"\u26a0 CCM \u3068 Blank \u306e\u4f5c\u6210\u65e5\u6642\u304c {diff_hours:.1f} \u6642\u9593\u96e2\u308c\u3066\u3044\u307e\u3059\u3002"
                    " \u540c\u4e00\u30bb\u30c3\u30b7\u30e7\u30f3\u3067\u518d\u30ad\u30e3\u30ea\u30d6\u30ec\u30fc\u30b7\u30e7\u30f3\u3092\u63a8\u5968\u3057\u307e\u3059\u3002"
                )
        except (ValueError, TypeError):
            pass

    def _print_calib_summary(self) -> None:  # noqa: C901
        """起動時に読み込まれたキャリブレーション値をターミナルに出力する。"""
        _S = "\u2550" * 62
        print(f"\n{_S}")
        print("  \u30ad\u30e3\u30ea\u30d6\u30ec\u30fc\u30b7\u30e7\u30f3\u72b6\u614b (\u8d77\u52d5\u6642\u8aad\u307f\u8fbc\u307f)")
        print(_S)
        _dark = getattr(self, "dark", None)
        if _dark is not None and _dark.is_loaded and _dark.dark_frame is not None:
            _mean = float(_dark.dark_frame.mean())
            print(f"  [D] Dark         : \u25cf loaded   mean={_mean:.1f} counts")
        else:
            print("  [D] Dark         : \u25cb not loaded")
        _flat = getattr(self, "flat", None)
        if _flat is not None and _flat.is_loaded and _flat.gain_map is not None:
            _gmin = float(_flat.gain_map.min())
            _gmax = float(_flat.gain_map.max())
            print(f"  [F] Flat         : \u25cf loaded   gain min={_gmin:.3f}  max={_gmax:.3f}")
        else:
            print("  [F] Flat         : \u25cb not loaded")
        _wb = getattr(self, "wb_calibrator", None)
        if _wb is not None and _wb.is_calibrated:
            _gains = _wb.get_gains()
            print(f"  [W] WhiteBalance : \u25cf loaded   R_gain={_gains[0]:.4f}  B_gain={_gains[1]:.4f}")
        else:
            print("  [W] WhiteBalance : \u25cb not loaded")
        _blank = getattr(self, "blank", None)
        if _blank is not None and _blank.is_loaded and _blank.blank_ratio is not None:
            _b = _blank.blank_ratio
            print(f"  [B] Blank        : \u25cf loaded   R={float(_b[0]):.6f}  G={float(_b[1]):.6f}  B={float(_b[2]):.6f}")
        else:
            print("  [B] Blank        : \u25cb not loaded")
        _lc = getattr(self, "lab_converter", None)
        if _lc is not None:
            _dc = _lc._diag_correction
            _rb = _lc._ref_baseline_rgb
            if _lc._diag_correction_calibrated:
                print(f"  [N] Neutral      : \u25cf loaded   d_R={_dc[0]:.4f}  d_B={_dc[2]:.4f}")
                if _rb is not None:
                    print(f"       ref_baseline: R={_rb[0]:.1f}  G={_rb[1]:.1f}  B={_rb[2]:.1f}")
            else:
                print("  [N] Neutral      : \u25cb not loaded")
            _ccm = _lc.get_ccm()
        else:
            print("  [N] Neutral      : \u25cb not loaded")
            _ccm = None
        _wr = getattr(self, "white_ratio_rgb", None)
        _store = getattr(self, "ccm_store", None)
        _meta = _store.metadata if _store is not None else {}
        if _ccm is not None and _wr is not None:
            _de_mean = _meta.get("delta_E_mean", float("nan"))
            _de_max = _meta.get("delta_E_max", float("nan"))
            print(f"  [P] CCM          : \u25cf loaded   \u0394E_mean={_de_mean:.2f}  \u0394E_max={_de_max:.2f}")
            print(f"       white_ratio : R={_wr[0]:.6f}  G={_wr[1]:.6f}  B={_wr[2]:.6f}")
            print("       CCM matrix  :")
            for _row in _ccm:
                print(f"         [{_row[0]:+.4f}  {_row[1]:+.4f}  {_row[2]:+.4f}]")
        else:
            print("  [P] CCM          : \u25cb not loaded  (white_ratio=[1,1,1])")
        print(f"{_S}\n")

    def _init_logger(self):
        """CSVロガーを初期化する。"""
        system_config = SystemConfig()
        log_dir = (
            self.config.measurement.csv_log_dir or system_config.get_result_directory()
        )
        self.logger = CSIMeasurementLogger(
            log_dir=log_dir,
            interval_frames=self.config.measurement.csv_log_interval,
        )
        self._system_config = system_config

    def _init_display(self):
        """オーバーレイ・ウィンドウ・動画録画を初期化する。"""
        self.overlay = CSIOverlayRenderer(self.config)
        self.frame_transformer = FrameTransformer(self.config)
        self.window_manager = WindowManager("CSI Precision Colorimeter", self.config)
        self.roi_mouse = ROIMouseHandler(self.config, persistence=self.roi_persistence)

        self.video_out = None
        if self.config.display.save_movie:
            composite_w = (
                self.config.camera.left_panel_width
                + self.config.camera.display_size[0]
                + self.config.camera.panel_width
            )
            composite_h = self.config.camera.display_size[1]
            video_dir = self._system_config.get_result_directory()
            os.makedirs(video_dir, exist_ok=True)
            ts = datetime.now().strftime("%y%m%d_%H%M%S")
            video_path = os.path.join(video_dir, f"{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_out = cv2.VideoWriter(
                video_path, fourcc, self.FPS, (composite_w, composite_h)
            )

        self.window_manager.setup()
        cv2.setMouseCallback(
            self.window_manager.window_name,
            self.window_manager.make_offset_callback(self.roi_mouse.callback),
        )
        # Phase 15: 旧 process が残した cv2 frame を即座に上書きするため、起動直後に
        # 黒画像を 1 回描画する。これによりデプロイ後の再起動 → main loop 1 周到達まで
        # 旧 overlay 文言 (例: 旧版の「上下反転チェック中」) が画面に残る現象を防ぐ。
        self._flush_initial_window()

    def _flush_initial_window(self) -> None:
        """起動直後の cv2 ウィンドウを黒一色で flush する (Phase 15 stale-frame 対策)."""
        composite_w = self.config.camera.left_panel_width + self.config.display.width
        composite_h = self.config.display.height
        blank = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
        self.window_manager.show(blank)

    def _capture_and_preprocess(self) -> tuple:
        """フレーム取得とRAW整形、表示フレームの前処理を行う。"""
        request = self.picam2.capture_request()
        try:
            main_array = request.make_array("main")
            raw_array = request.make_array("raw")
            metadata = request.get_metadata()
        finally:
            request.release()
        raw_bayer = self.bayer.parse_raw(raw_array)
        frame_bgr = self._main_array_to_bgr(main_array)
        frame_bgr = self.frame_transformer.transform(frame_bgr)
        return frame_bgr, raw_bayer, metadata

    def _main_array_to_bgr(self, main_array: np.ndarray) -> np.ndarray:
        """Picamera2 mainストリーム配列をBGRへ正規化する。"""
        fmt = str(getattr(self, "main_stream_format", "")).upper()
        if main_array.ndim == 3 and main_array.shape[2] == 4:
            if (
                fmt.startswith("XRGB")
                or fmt.startswith("ARGB")
                or fmt.startswith("RGBA")
            ):
                return cv2.cvtColor(main_array, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(main_array, cv2.COLOR_BGRA2BGR)
        if main_array.ndim == 3 and main_array.shape[2] == 3:
            if fmt.startswith("BGR"):
                return cv2.cvtColor(main_array, cv2.COLOR_RGB2BGR)
            return main_array
        return main_array

    def _extract_tar_rgb_channel_planes(self, tar_roi_raw: np.ndarray) -> np.ndarray:
        """Tar ROIからBayerのR/G/B平面配列を生成する。"""
        if tar_roi_raw is None or tar_roi_raw.size == 0:
            return np.empty((0, 0, 3), dtype=np.float32)
        r = tar_roi_raw[1::2, 1::2].astype(np.float32)
        g1 = tar_roi_raw[0::2, 1::2].astype(np.float32)
        g2 = tar_roi_raw[1::2, 0::2].astype(np.float32)
        b = tar_roi_raw[0::2, 0::2].astype(np.float32)
        h = min(r.shape[0], g1.shape[0], g2.shape[0], b.shape[0])
        w = min(r.shape[1], g1.shape[1], g2.shape[1], b.shape[1])
        if h <= 0 or w <= 0:
            return np.empty((0, 0, 3), dtype=np.float32)
        g = (g1[:h, :w] + g2[:h, :w]) * 0.5
        return np.stack([r[:h, :w], g, b[:h, :w]], axis=-1)

    def _compute_luma_from_rgb_planes(self, rgb_planes: np.ndarray) -> np.ndarray:
        """RGB平面配列から輝度Y配列を算出する。"""
        if rgb_planes is None or rgb_planes.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        coeff = np.array([0.2126729, 0.7151522, 0.0721750], dtype=np.float32)
        return np.tensordot(rgb_planes, coeff, axes=([-1], [0])).astype(np.float32)

    def _extract_rois(self, raw_bayer: np.ndarray, metadata: dict) -> dict:
        """参照/ターゲットROIをRAW空間で抽出しRGB平均を算出する。"""
        ref_disp_x, ref_disp_y = self.config.processing.posi_ref
        tar_disp_x, tar_disp_y = self.config.processing.posi_tar
        ref_w = max(int(self.config.processing.spot_size_ref), 2)
        ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
        tar_w = max(int(self.config.processing.spot_size_tar), 2)
        tar_h = max(int(tar_w * self.config.processing.aspect_tar), 2)
        ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h = self.bayer.display_to_raw_coords(
            ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
            self.config.display.flip_horizontal, self.config.display.flip_vertical,
        )
        tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h = self.bayer.display_to_raw_coords(
            tar_disp_x, tar_disp_y, tar_w, tar_h, metadata,
            self.config.display.flip_horizontal, self.config.display.flip_vertical,
        )
        roi_ref_raw = self.bayer.extract_raw_roi(raw_bayer, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
        raw_ref_12bit_pre = self.bayer.get_raw_12bit_means_from_roi(roi_ref_raw)
        roi_tar_raw = self.bayer.extract_raw_roi(raw_bayer, tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h)
        if self.dark.is_loaded:
            dark_ref = self.dark.get_dark_roi(ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
            dark_tar = self.dark.get_dark_roi(tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h)
            if dark_ref is not None:
                roi_ref_raw = self.dark.subtract(roi_ref_raw, dark_ref)
            if dark_tar is not None:
                roi_tar_raw = self.dark.subtract(roi_tar_raw, dark_tar)
        if self.flat.is_loaded:
            roi_ref_raw = self.flat.correct_roi(roi_ref_raw, ref_raw_x, ref_raw_y, ref_raw_w, ref_raw_h)
            roi_tar_raw = self.flat.correct_roi(roi_tar_raw, tar_raw_x, tar_raw_y, tar_raw_w, tar_raw_h)
        ref_rgb = self.bayer.extract_bayer_means_from_roi(roi_ref_raw)
        tar_rgb = self.bayer.extract_bayer_means_from_roi(roi_tar_raw)
        tar_rgb_planes = self._extract_tar_rgb_channel_planes(roi_tar_raw)
        tar_luma = self._compute_luma_from_rgb_planes(tar_rgb_planes)
        raw_ref_12bit = self.bayer.get_raw_12bit_means_from_roi(roi_ref_raw)
        raw_tar_12bit = self.bayer.get_raw_12bit_means_from_roi(roi_tar_raw)
        return {
            "ref_rgb": ref_rgb, "tar_rgb": tar_rgb,
            "roi_ref_raw": roi_ref_raw, "roi_tar_raw": roi_tar_raw,
            "tar_luma": tar_luma,
            "raw_ref_12bit_pre": raw_ref_12bit_pre,
            "raw_ref_12bit": raw_ref_12bit, "raw_tar_12bit": raw_tar_12bit,
            "ref_disp_w": ref_w, "ref_disp_h": ref_h,
            "tar_disp_w": tar_w, "tar_disp_h": tar_h,
        }

    def _compute_measurements(self, roi: dict) -> dict:
        """ROIからLabや安定性指標、警告を計算する。"""
        warnings_ref = self.quality.check(roi["roi_ref_raw"])
        warnings_tar = self.quality.check(roi["roi_tar_raw"])
        ref_raw_mean_12bit = float(roi["roi_ref_raw"].mean())
        tar_raw_mean_12bit = float(roi["roi_tar_raw"].mean())
        ref_roi_for_clip = np.asarray(roi["roi_ref_raw"], dtype=np.float64)
        finite_ref_roi = ref_roi_for_clip[np.isfinite(ref_roi_for_clip)]
        if finite_ref_roi.size:
            ref_clip_high_pct = float(np.mean(finite_ref_roi >= 4090.0) * 100.0)
            ref_clip_low_pct = float(np.mean(finite_ref_roi <= 5.0) * 100.0)
        else:
            ref_clip_high_pct = 0.0
            ref_clip_low_pct = 0.0
        ref_raw_in_band = abs(ref_raw_mean_12bit - float(self.ae.target_mid)) <= 250.0
        ref_center_edge_diff_pct = (
            self.quality.center_edge_diff_fraction(roi["roi_ref_raw"]) * 100.0
        )
        ref_spatial_analyzer = getattr(self, "ref_spatial_analyzer", None)
        ref_sat_threshold = float(getattr(self.quality, "SATURATED_THRESHOLD", 4090.0))
        ref_sat_fraction = float(getattr(self.quality, "SATURATED_FRACTION", 0.01))
        ref_under_threshold = float(getattr(self.quality, "UNDEREXPOSED_THRESHOLD", 10.0))
        ref_under_fraction = float(getattr(self.quality, "UNDEREXPOSED_FRACTION", 0.05))
        if ref_spatial_analyzer is None:
            ref_spatial_analyzer = RefSpatialAnalyzer(
                saturated_threshold=ref_sat_threshold,
                saturated_fraction=ref_sat_fraction,
                underexposed_threshold=ref_under_threshold,
                underexposed_fraction=ref_under_fraction,
            )
            self.ref_spatial_analyzer = ref_spatial_analyzer
        ref_spatial_abs = ref_spatial_analyzer.analyze_ref_roi(
            roi["roi_ref_raw"],
            source_kind="bayer",
            underexposed_threshold_luma=ref_under_threshold,
            saturated_threshold_luma=ref_sat_threshold,
            correction_stage="dark_flat_corrected",
        )
        ratio = self.double_beam.compute_ratio(roi["ref_rgb"], roi["tar_rgb"])
        if self.blank.is_loaded:
            ratio = self.blank.correct(ratio)
        elif self.master_ref.is_loaded:
            ref_for_scale = self.lab_converter.get_ref_ema()
            if ref_for_scale is None:
                ref_for_scale = roi["ref_rgb"]
            scale = self.master_ref.compute_scale(ref_for_scale)
            ratio = ratio / scale
        self.stability.push(ratio)
        median_ratio = self.stability.get_median_ratio()
        self.lab_converter.update_dynamic_correction(roi["ref_rgb"])
        if self.spectral_drift_tracker.is_active:
            dc = self.lab_converter._dynamic_correction
            self.spectral_drift_tracker.push(float(dc[0]), float(dc[2]))
            if self.spectral_drift_tracker._n % 100 == 1:
                print(f"  [dc] f_R={dc[0]:.6f}  f_B={dc[2]:.6f}")
        correction_f_R = None
        correction_f_B = None
        if getattr(self.spectral_drift_tracker, "is_active", False):
            try:
                correction_f_R, correction_f_B = self.spectral_drift_tracker.get_current()
            except Exception:
                correction_f_R, correction_f_B = None, None
        white_for_norm = self.white_ratio_rgb.copy()
        if self.blank.is_loaded:
            white_for_norm = self.blank.correct(white_for_norm)
        # Y係数ベースの輝度スカラー補正（EMA平滑化、チャネル別ベクトルを廃止、分光歪み除去）
        # dynamic_correction が既に色比ドリフトを補正するため、ここでは輝度のみ補正
        _ref_scale_val: float | None = None
        _ref_ema = self.lab_converter.get_ref_ema()
        _ref_now = (
            _ref_ema
            if _ref_ema is not None
            else np.asarray(roi["ref_rgb"], dtype=np.float64)[:3]
        )
        _ref_scale_now = normalize_ref_scale_triplet(_ref_now, self.white_ratio_rgb)
        # 起動直後（diag未初期化）は旧CCMの残差テーブルでkNN警告が出るため抑制
        _suppress_knn = not hasattr(self, '_diag_frame_count')
        canonical_live = run_canonical_lab_pipeline(
            median_ratio,
            white_for_norm,
            _ref_scale_now,
            self.live_ref_scale_baseline,
            self.lab_converter,
            suppress_domain_warning=_suppress_knn,
        )
        ratio_white = canonical_live["ratio_white"]
        _ref_scale_val = canonical_live["ref_scale"]
        # グレーカード（tar=ref → ratio=[1,1,1]）が現パイプラインを通ったときの ratio_white
        # 測定パスと同一の blank 補正を適用して blank_ratio の未相殺を防ぐ
        _anchor_raw = np.ones(3, dtype=np.float32)
        if self.blank.is_loaded:
            _anchor_raw = self.blank.correct(_anchor_raw)
        canonical_anchor = run_canonical_lab_pipeline(
            _anchor_raw,
            white_for_norm,
            _ref_scale_now,
            self.live_ref_scale_baseline,
            self.lab_converter,
            suppress_domain_warning=_suppress_knn,
        )
        _anchor_lab_abs = canonical_anchor["lab"]
        _ref_anchor = _anchor_lab_abs.astype(np.float32).copy()
        Lab_abs = canonical_live["lab"].astype(np.float32)
        Lab_rel = Lab_abs - _ref_anchor
        ref_scale_warn = is_ref_scale_warn(_ref_scale_val)
        lab_rel_l_warn = is_lab_rel_l_warn(float(Lab_rel[0]))
        correction_applied_warning = build_correction_applied_warning(
            correction_f_R,
            correction_f_B,
            _ref_scale_val,
        )
        self.ref_scale_warn = ref_scale_warn
        self.lab_rel_l_warn = lab_rel_l_warn
        # 100フレームに1回のパイプライン全変数診断
        if not hasattr(self, '_diag_frame_count'):
            self._diag_frame_count = 0
        self._diag_frame_count += 1
        if self._diag_frame_count % 100 == 1:
            _ref_rgb_raw = np.asarray(roi["ref_rgb"], dtype=np.float64)[:3]
            _ref_ema_diag = self.lab_converter.get_ref_ema()
            _dc_diag = self.lab_converter._dynamic_correction
            _rs_str = f"{_ref_scale_val:.4f}" if _ref_scale_val is not None else "N/A"
            _ema_str = (f"[{_ref_ema_diag[0]:.4f},{_ref_ema_diag[1]:.4f},{_ref_ema_diag[2]:.4f}]"
                        if _ref_ema_diag is not None else "N/A")
            print(
                f"  [diag] ref_rgb=[{_ref_rgb_raw[0]:.4f},{_ref_rgb_raw[1]:.4f},{_ref_rgb_raw[2]:.4f}]"
                f"  ref_ema={_ema_str}"
                f"  dc=[{_dc_diag[0]:.4f},{_dc_diag[1]:.4f},{_dc_diag[2]:.4f}]"
                f"  median_ratio=[{median_ratio[0]:.4f},{median_ratio[1]:.4f},{median_ratio[2]:.4f}]"
                f"  ratio_white=[{ratio_white[0]:.4f},{ratio_white[1]:.4f},{ratio_white[2]:.4f}]"
                f"  ref_scale={_rs_str}"
                f"  baseline_source={self.live_ref_scale_baseline_source or 'none'}"
                f"  anchor=[{_ref_anchor[0]:.2f},{_ref_anchor[1]:.2f},{_ref_anchor[2]:.2f}]"
                f"  Lab_rel=[{Lab_rel[0]:.2f},{Lab_rel[1]:.2f},{Lab_rel[2]:.2f}]"
            )
        self.stability.push_lab(Lab_abs)
        cv_L = self.stability.get_cv_L()
        sigma_a = self.stability.get_sigma_a()
        sigma_b = self.stability.get_sigma_b()
        state = self.stability.get_stability_state()
        if self.stability.lab_count >= 2:
            valid_lab = self.stability.lab_buffer[: self.stability.lab_count]
            sigma_L_raw = float(valid_lab[:, 0].std(ddof=1))
            sigma_a_raw = float(valid_lab[:, 1].std(ddof=1))
            sigma_b_raw = float(valid_lab[:, 2].std(ddof=1))
        else:
            sigma_L_raw = float("inf")
            sigma_a_raw = float("inf")
            sigma_b_raw = float("inf")
        U_L = 2.0 * sigma_L_raw
        U_a = 2.0 * sigma_a_raw
        U_b = 2.0 * sigma_b_raw
        deltaE = self.de2000.compute(Lab_abs, self.target_Lab)
        deltaE_ref = self.de2000.compute(Lab_abs, _ref_anchor)
        flat_global_p90 = None
        if self.flat.is_loaded and self.flat.gain_map is not None:
            flat_global_p90 = compute_gain_percentile(
                self.flat.gain_map,
                90,
                mask=getattr(self.flat, "valid_mask", None),
            )
        flat_ref_p95 = self._compute_flat_ref_p95()
        ref_rgb = np.asarray(roi["ref_rgb"], dtype=np.float32).reshape(-1)
        if ref_rgb.size == 3 and float(ref_rgb[1]) > 1e-8:
            ref_chromaticity = np.array(
                [
                    float(ref_rgb[0]) / float(ref_rgb[1]),
                    1.0,
                    float(ref_rgb[2]) / float(ref_rgb[1]),
                ],
                dtype=np.float32,
            )
        else:
            ref_chromaticity = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.df_tracker.push(
            ref_raw_mean_12bit,
            ref_center_edge_diff_pct,
            chromaticity=ref_chromaticity,
            clip_high_pct=ref_clip_high_pct,
            clip_low_pct=ref_clip_low_pct,
            spatial_abs=ref_spatial_abs,
        )
        if hasattr(self.df_tracker, "get_drift_snapshot"):
            ref_drift_snapshot = self.df_tracker.get_drift_snapshot()
        else:
            ref_drift_snapshot = {
                "drift_state": "OK",
                "drift_reason_axes": "",
                "baseline_label": "Base UNVER",
                "baseline_source": "none",
                "baseline_kind": "none",
                "raw_mean": ref_raw_mean_12bit,
                "raw_baseline": None,
                "intensity_drift_pct": None,
                "chroma_drift": None,
                "uniformity_drift_pct": None,
                "intensity_jitter_pct": None,
                "clip_high_pct": ref_clip_high_pct,
                "clip_low_pct": ref_clip_low_pct,
            }
        ratio_validated = DoubleBeamProcessor.validate_ratio(median_ratio)
        return {
            "Lab": Lab_abs, "Lab_rel": Lab_rel, "Lab_ref_anchor": _ref_anchor.copy(),
            "U_L": U_L, "U_a": U_a, "U_b": U_b,
            "deltaE": deltaE, "deltaE_ref": deltaE_ref,
            "cv_L": cv_L, "sigma_L": sigma_L_raw, "sigma_a": sigma_a, "sigma_b": sigma_b,
            "state": state, "warnings_ref": warnings_ref, "warnings_tar": warnings_tar,
            "ref_raw_mean_12bit": ref_raw_mean_12bit, "tar_raw_mean_12bit": tar_raw_mean_12bit,
            "ref_raw_in_band": ref_raw_in_band, "ref_center_edge_diff_pct": ref_center_edge_diff_pct,
            "ref_spatial_abs": ref_spatial_abs,
            "ratio": median_ratio.astype(np.float32), "ratio_validated": ratio_validated,
            "mode": self.mode.current, "ref_scale": _ref_scale_val,
            "ref_scale_warn": ref_scale_warn,
            "lab_rel_l_warn": lab_rel_l_warn,
            "correction_applied_warn": correction_applied_warning is not None,
            "correction_applied_warning": correction_applied_warning,
            "correction_applied_warning_axes": (
                [] if correction_applied_warning is None else correction_applied_warning["axes"]
            ),
            "live_ref_baseline_source": self.live_ref_baseline_source,
            "live_ref_scale_baseline_source": self.live_ref_scale_baseline_source,
            "ref_drift": ref_drift_snapshot,
            "ref_clip_high_pct": ref_clip_high_pct,
            "ref_clip_low_pct": ref_clip_low_pct,
            "flat_global_p90": flat_global_p90, "flat_ref_p95": flat_ref_p95,
        }

    def _draw_flip_warning_banner(self, composite: np.ndarray) -> np.ndarray:
        """Phase 7C/7E: flip_warning state の日本語警告バナーを PIL で描画する.

        - 赤帯 + 点滅赤枠 (cv2 で高速描画)
        - 体言止めタイトル (大サイズ, 中央揃え)
        - 操作手順の箇条書き (中央揃え)
        - grid overlay (視覚フィードバック)
        """
        from PIL import Image, ImageDraw
        from .colorimeter_common import _find_japanese_font

        h_img, w_img = composite.shape[:2]
        banner_h = 200  # 箇条書き分広げる

        # 赤帯 (BGR) を cv2 で敷く
        cv2.rectangle(composite, (0, 0), (w_img, banner_h), (0, 0, 255), cv2.FILLED)

        # 点滅する赤い縁取り枠 (毎秒 2 回)
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(composite, (0, 0), (w_img - 1, h_img - 1), (0, 0, 255), 8)

        # Phase 7E: 専用の大きな日本語フォントを毎回作る (既存の 14/12 px では小さすぎる)
        title_font = _find_japanese_font(46)  # 体言止めタイトル用 (大)
        body_font = _find_japanese_font(28)   # 箇条書き用 (中)
        reason = getattr(self.key_handler, "_flip_warning_reason", "") or ""

        if title_font is not None and body_font is not None:
            pil_img = Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # --- タイトル (体言止め、中央揃え、黒アウトライン + 白文字) ---
            # Phase 7F/8: 反転だけ 180° 案内、それ以外は向き・設置案内にする
            if reason.startswith("flip"):
                title_text = "‼ 色基準(48色) 上下反転 ‼"
            elif reason.startswith("rotated_or_misplaced"):
                title_text = "‼ 色基準(48色) 向き異常 / 斜め配置 ‼"
            else:
                title_text = "‼ 色基準(48色) 向き・設置エラー ‼"
            title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
            title_w = title_bbox[2] - title_bbox[0]
            title_x = (w_img - title_w) // 2
            title_y = 10
            for dx, dy in [(-3,-3),(-3,0),(-3,3),(0,-3),(0,3),(3,-3),(3,0),(3,3)]:
                draw.text((title_x + dx, title_y + dy), title_text,
                          font=title_font, fill=(0, 0, 0))
            draw.text((title_x, title_y), title_text,
                      font=title_font, fill=(255, 255, 255))

            # --- 箇条書き (中央揃えブロック) ---
            if reason.startswith("flip"):
                bullets = [
                    "① 色基準(48色) を 180° 回して置き直す",
                    "② 置き直したら [P] で再チェック",
                    "③ やめるなら [ESC] または [q]",
                ]
            else:
                bullets = [
                    "① 色基準(48色) を水平に正しい向きで置き直す",
                    "② 置き直したら [P] で再チェック",
                    "③ やめるなら [ESC] または [q]",
                ]
            # 最長行の幅を基準にブロック左端を決め、中央寄せ
            max_line_w = max(
                draw.textbbox((0, 0), line, font=body_font)[2]
                - draw.textbbox((0, 0), line, font=body_font)[0]
                for line in bullets
            )
            block_x = (w_img - max_line_w) // 2
            line_y = title_y + (title_bbox[3] - title_bbox[1]) + 16
            line_gap = 8
            for line in bullets:
                bbox = draw.textbbox((0, 0), line, font=body_font)
                line_h = bbox[3] - bbox[1]
                for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
                    draw.text((block_x + dx, line_y + dy), line,
                              font=body_font, fill=(0, 0, 0))
                draw.text((block_x, line_y), line,
                          font=body_font, fill=(255, 255, 255))
                line_y += line_h + line_gap

            composite = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # PIL/日本語フォントが使えない環境は cv2 fallback (ASCII).
            if reason.startswith("flip"):
                fallback_text = "!!  CHART UPSIDE-DOWN / ROTATE 180 DEG  !!"
            else:
                fallback_text = "!!  CHART ORIENTATION / PLACEMENT  !!"
            cv2.putText(
                composite, fallback_text,
                (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA,
            )
            cv2.putText(
                composite, fallback_text,
                (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA,
            )

        # grid overlay も描画して user に反転状態の視覚フィードバックを与える
        kh = self.key_handler
        if kh._grid_extractor is not None:
            composite = kh._grid_extractor.draw_overlay(
                composite, [],
                white_local_idx=SPYDERCHECKR_48_WHITE_PATCH_INDEX,
            )
        return composite

    def _draw_chart_orientation_badge(
        self,
        composite: np.ndarray,
        lines: tuple[str, ...],
    ) -> np.ndarray:
        """Draw the verified chart orientation where the operator is looking."""
        if not lines:
            return composite

        h_img, w_img = composite.shape[:2]
        left_panel_width = int(getattr(self.config.camera, "left_panel_width", 0) or 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.68
        thickness = 2
        line_gap = 8
        pad_x = 12
        pad_y = 9
        sizes = [
            cv2.getTextSize(line, font, scale, thickness)[0]
            for line in lines
        ]
        text_w = max((size[0] for size in sizes), default=0)
        text_h = max((size[1] for size in sizes), default=0)
        badge_w = text_w + pad_x * 2
        badge_h = (
            (text_h * len(lines))
            + (line_gap * max(0, len(lines) - 1))
            + pad_y * 2
        )

        x1 = max(left_panel_width + 12, 8)
        if x1 + badge_w >= w_img:
            x1 = max(8, w_img - badge_w - 8)
        y1 = 10
        x2 = min(w_img - 1, x1 + badge_w)
        y2 = min(h_img - 1, y1 + badge_h)

        upside_down = any("UPSIDE-DOWN" in line for line in lines)
        border_color = (0, 210, 255) if upside_down else (0, 220, 80)
        text_color = (0, 255, 255) if upside_down else (80, 255, 120)

        cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(composite, (x1, y1), (x2, y2), border_color, 2)

        baseline_y = y1 + pad_y + text_h
        for idx, line in enumerate(lines):
            y = baseline_y + idx * (text_h + line_gap)
            cv2.putText(
                composite,
                line,
                (x1 + pad_x, y),
                font,
                scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                composite,
                line,
                (x1 + pad_x, y),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
        return composite

    def _render(self, display_frame_bgr: np.ndarray, roi: dict, meas: dict, correction_info: Optional[dict] = None) -> np.ndarray:
        """ROI矩形とサイドパネルを描画した合成フレームを生成する。"""
        if correction_info is None:
            correction_info = {}
        self.ae.tick(roi["raw_ref_12bit_pre"].mean())
        _ccm_loaded = self.lab_converter.get_ccm() is not None
        _gray_status = (
            self.gray_verifier.status_text()
            if self.last_gray_verify_result is not None else None
        )
        _gray_absolute_status = (
            self.ccm_verifier.status_text()
            if self.last_gray_absolute_verify_result is not None else None
        )
        left_panel = self.overlay.build_left_panel(
            self.dark.is_loaded, self.flat.is_loaded,
            wb_calibrated=self.wb_calibrator.is_calibrated,
            neutral_calibrated=self.lab_converter._diag_correction_calibrated,
            blank_loaded=self.blank.is_loaded,
            master_ref_loaded=self.master_ref.is_loaded,
            ccm_loaded=_ccm_loaded,
            flat_global_p90=meas.get("flat_global_p90"),
            flat_ref_p95=meas.get("flat_ref_p95"),
            flat_ref_roi_gain_max=self.flat_ref_roi_gain_max,
            flat_has_valid_mask=getattr(self.flat, "valid_mask", None) is not None,
            gray_verify_status=_gray_status,
            gray_absolute_status=_gray_absolute_status,
            software_version=self.software_identity.get("software_version", "unknown"),
        )
        composite = self.overlay.create_composite_frame(display_frame_bgr.copy(), left_panel=left_panel)
        ref_w = int(roi.get("ref_disp_w", self.config.processing.spot_size_ref))
        ref_h = int(roi.get("ref_disp_h", self.config.processing.spot_size_ref * self.config.processing.aspect_ref))
        tar_w = int(roi.get("tar_disp_w", self.config.processing.spot_size_tar))
        tar_h = int(roi.get("tar_disp_h", self.config.processing.spot_size_tar * self.config.processing.aspect_tar))
        self.overlay.draw_roi_rectangles(
            composite, self.config.processing.posi_ref, ref_w, ref_h,
            self.config.processing.posi_tar, tar_w, tar_h,
            ref_raw_mean_12bit=meas["ref_raw_mean_12bit"],
            ref_raw_in_band=meas["ref_raw_in_band"],
            ref_center_edge_diff_pct=meas["ref_center_edge_diff_pct"],
            center_edge_warn_pct=self.config.processing.center_edge_warn_threshold * 100.0,
            tar_raw_mean_12bit=meas["tar_raw_mean_12bit"],
        )
        self.overlay.draw_side_panel(
            composite, Lab_abs=meas["Lab"], Lab_rel=meas["Lab_rel"],
            U_L=meas["U_L"], U_a=meas["U_a"], U_b=meas["U_b"],
            dE00_target=meas["deltaE"], dE00_ref=meas["deltaE_ref"],
            state=meas["state"], dark_loaded=self.dark.is_loaded,
            exposure_us=self.ae.current_exposure, gain=self.config.camera.analogue_gain,
            ae_on=self.ae.enabled, cv_L=meas["cv_L"],
            sigma_L=meas["sigma_L"], sigma_a=meas["sigma_a"], sigma_b=meas["sigma_b"],
            warnings_ref=meas["warnings_ref"], warnings_tar=meas["warnings_tar"],
            tar_luma=roi["tar_luma"],
            ref_raw_mean_12bit=meas["ref_raw_mean_12bit"],
            ref_raw_in_band=meas["ref_raw_in_band"],
            ref_center_edge_diff_pct=meas["ref_center_edge_diff_pct"],
            ref_L_raw=correction_info.get("ref_L_raw"),
            ref_L_corr=correction_info.get("ref_L_corr"),
            bright_scale=correction_info.get("bright_scale", 1.0),
            ref_roi_raw=roi["roi_ref_raw"], flat_loaded=self.flat.is_loaded,
            stability_tracker=self.df_tracker, ref_rgb=roi["ref_rgb"],
            center_edge_warn_pct=self.config.processing.center_edge_warn_threshold * 100.0,
            mode=self.mode, ratio=meas.get("ratio_validated"),
            drift_tracker=self.spectral_drift_tracker,
            ref_scale=meas.get("ref_scale"),
            ref_scale_warn=meas.get("ref_scale_warn", False),
            gray_absolute_result=self.last_gray_absolute_verify_result,
            acceptance_result=self.last_acceptance_result,
            lab_rel_l_warn=meas.get("lab_rel_l_warn", False),
            flat_global_p90=meas.get("flat_global_p90"),
            flat_ref_p95=meas.get("flat_ref_p95"),
            workflow_status=self.key_handler.get_chart_workflow_status(),
        )
        chart_state = str(getattr(self.key_handler, "_chart_state", ""))
        if chart_state != "flip_warning":
            composite = self.overlay.draw_correction_applied_warning_banner(
                composite,
                meas.get("correction_applied_warning"),
            )
        return composite

    def _save_snapshot(self, composite: np.ndarray, log_data: dict, meas: dict) -> None:
        """計測スナップショットを保存する。"""
        os.makedirs(self.results_dir, exist_ok=True)
        ts = datetime.now().strftime("%y%m%d_%H%M%S")

        png_path = os.path.join(self.results_dir, f"{ts}_measure.png")
        cv2.imwrite(png_path, composite)

        csv_path = os.path.join(self.results_dir, f"{ts}_measure.csv")
        header = [
            "timestamp", "L_star", "a_star", "b_star", "dE00", "U_L", "U_a", "U_b",
            "ref_raw_mean", "ref_pos_x", "ref_pos_y", "ref_spot_size", "ref_aspect",
            "tar_pos_x", "tar_pos_y", "tar_spot_size", "tar_aspect",
            "dark_loaded", "flat_loaded", "exposure_us", "analogue_gain",
            "quality_warnings", "mode", "R_lin", "G_lin", "B_lin",
        ]

        def _fmt_u(v):
            return f"{v:.4f}" if v < float("inf") else "inf"

        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            f"{meas['Lab'][0]:.4f}", f"{meas['Lab'][1]:.4f}", f"{meas['Lab'][2]:.4f}",
            f"{meas['deltaE']:.4f}",
            _fmt_u(meas["U_L"]), _fmt_u(meas["U_a"]), _fmt_u(meas["U_b"]),
            f"{meas['ref_raw_mean_12bit']:.1f}",
            str(self.config.processing.posi_ref[0]), str(self.config.processing.posi_ref[1]),
            str(self.config.processing.spot_size_ref), f"{self.config.processing.aspect_ref:.2f}",
            str(self.config.processing.posi_tar[0]), str(self.config.processing.posi_tar[1]),
            str(self.config.processing.spot_size_tar), f"{self.config.processing.aspect_tar:.2f}",
            str(self.dark.is_loaded), str(self.flat.is_loaded),
            str(self.ae.current_exposure), f"{self.config.camera.analogue_gain:.1f}",
            log_data.get("quality_warnings", ""),
            meas.get("mode", "Lab"),
            f"{meas.get('ratio_validated', np.zeros(3))[0]:.6f}",
            f"{meas.get('ratio_validated', np.zeros(3))[1]:.6f}",
            f"{meas.get('ratio_validated', np.zeros(3))[2]:.6f}",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)

        print(f"  Snapshot: {png_path}")
        print(f"  Snapshot CSV: {csv_path}")

    def _output(self, composite: np.ndarray, roi: dict, meas: dict) -> dict:
        """画面表示・動画保存・ログ出力を行う。"""
        self.window_manager.show(composite)
        if self.video_out is not None:
            self.video_out.write(composite)
        key_handler = getattr(self, "key_handler", None)
        if key_handler is not None and hasattr(key_handler, "get_chart_trace_state"):
            chart_trace = key_handler.get_chart_trace_state()
        else:
            chart_trace = {
                "chart_state": "unknown",
                "chart_warning_reason": "",
            }
        lab_rel = meas.get("Lab_rel")
        lab_rel_l = None
        if lab_rel is not None:
            lab_rel_l = float(lab_rel[0])
        ref_uneven_item = None
        try:
            ref_uneven_item = next(
                item
                for item in self.df_tracker.get_ref_monitor_axis_statuses()
                if item.get("axis") == "U"
            )
        except Exception:
            ref_uneven_item = None
        ref_spatial_abs = meas.get("ref_spatial_abs")
        log_data = {
            "L_star": float(meas["Lab"][0]), "a_star": float(meas["Lab"][1]),
            "b_star": float(meas["Lab"][2]),
            "raw_R": float(roi["raw_tar_12bit"][0]), "raw_G": float(roi["raw_tar_12bit"][1]),
            "raw_B": float(roi["raw_tar_12bit"][2]),
            "CV_L": float(meas["cv_L"]),
            "sigma_a": float(meas["sigma_a"]), "sigma_b": float(meas["sigma_b"]),
            "deltaE00": float(meas["deltaE"]),
            "U_L": float(meas["U_L"]) if meas["U_L"] < float("inf") else 0.0,
            "U_a": float(meas["U_a"]) if meas["U_a"] < float("inf") else 0.0,
            "U_b": float(meas["U_b"]) if meas["U_b"] < float("inf") else 0.0,
            "exposure_us": self.ae.current_exposure,
            "analogue_gain": self.config.camera.analogue_gain,
            "quality_warnings": ";".join(meas["warnings_ref"] + meas["warnings_tar"]),
            "guard_state": self.df_tracker.get_guard_state(),
            "dev_I_pct": float(self.df_tracker.get_dev_I_pct() or 0.0),
            "dev_uniformity_pct": float(self.df_tracker.get_dev_uniformity_pct() or 0.0),
            "dev_chromaticity": float(self.df_tracker.get_dev_chromaticity() or 0.0),
            "guard_reason": self.df_tracker.get_guard_reason(),
            "drift_guard_state": getattr(self.df_tracker, "get_drift_state", lambda: "OK")(),
            "drift_reason_axes": getattr(self.df_tracker, "get_drift_reason_axes", lambda: "")(),
            "ref_raw_mean": float(meas.get("ref_raw_mean_12bit", 0.0)),
            "ref_raw_baseline": float(
                getattr(self.df_tracker, "ref_raw_mean", None) or 0.0
            ),
            "ref_drift_i_pct": float(self.df_tracker.get_dev_I_pct() or 0.0),
            "ref_drift_c": float(self.df_tracker.get_dev_chromaticity() or 0.0),
            "ref_drift_u_pct": float(self.df_tracker.get_dev_uniformity_pct() or 0.0),
            "legacy_u_rel_pct": float(self.df_tracker.get_dev_uniformity_pct() or 0.0),
            "legacy_center_edge_pct": float(meas.get("ref_center_edge_diff_pct", 0.0)),
            "ref_uneven_value_pct": (
                None
                if ref_uneven_item is None or ref_uneven_item.get("value_kind") != "absolute"
                else ref_uneven_item.get("value")
            ),
            "ref_uneven_drift_delta_pp": (
                None if ref_uneven_item is None else ref_uneven_item.get("drift_delta_pp")
            ),
            "ref_uneven_risk": (
                None if ref_uneven_item is None else ref_uneven_item.get("graph_risk")
            ),
            "ref_uneven_abs_dominant": (
                "" if ref_uneven_item is None else (ref_uneven_item.get("abs_dominant") or "")
            ),
            "ref_uneven_drift_dominant": (
                "" if ref_uneven_item is None else (ref_uneven_item.get("drift_dominant") or "")
            ),
            "ref_uneven_reason": (
                "" if ref_uneven_item is None else (ref_uneven_item.get("reason") or "")
            ),
            "ref_uneven_value_kind": (
                "" if ref_uneven_item is None else (ref_uneven_item.get("value_kind") or "")
            ),
            "ref_uneven_direction_reversal": bool(
                ref_uneven_item is not None and ref_uneven_item.get("direction_reversal")
            ),
            "ref_center_edge_pct": (
                None if ref_spatial_abs is None else ref_spatial_abs.center_edge_pct
            ),
            "ref_left_right_pct": (
                None if ref_spatial_abs is None else ref_spatial_abs.left_right_pct
            ),
            "ref_top_bottom_pct": (
                None if ref_spatial_abs is None else ref_spatial_abs.top_bottom_pct
            ),
            "ref_diag_pct": (
                None if ref_spatial_abs is None else ref_spatial_abs.diag_pct
            ),
            "ref_tile_p95_p05_pct": (
                None if ref_spatial_abs is None else ref_spatial_abs.tile_p95_p05_pct
            ),
            "ref_tile_max_dev_pct": (
                None if ref_spatial_abs is None else ref_spatial_abs.tile_max_dev_pct
            ),
            "ref_tile_cv_pct": (
                None if ref_spatial_abs is None else ref_spatial_abs.cv_pct
            ),
            "ref_drift_j_pct": float(
                getattr(self.df_tracker, "get_intensity_jitter_pct", lambda: 0.0)() or 0.0
            ),
            "ref_clip_high_pct": float(
                getattr(self.df_tracker, "get_clip_high_pct", lambda: 0.0)() or 0.0
            ),
            "ref_clip_low_pct": float(
                getattr(self.df_tracker, "get_clip_low_pct", lambda: 0.0)() or 0.0
            ),
            "ref_baseline_label": getattr(
                self.df_tracker, "get_baseline_label", lambda: "Base UNVER"
            )(),
            "ref_baseline_kind": getattr(
                self.df_tracker, "get_baseline_kind", lambda: "none"
            )(),
            "mode": meas.get("mode", "Lab"),
            "ratio_R": float(meas.get("ratio_validated", np.zeros(3))[0]),
            "ratio_G": float(meas.get("ratio_validated", np.zeros(3))[1]),
            "ratio_B": float(meas.get("ratio_validated", np.zeros(3))[2]),
            "Lab_rel_L": lab_rel_l,
            "lab_rel_l_warn": bool(meas.get("lab_rel_l_warn", False)),
            "ref_scale": meas.get("ref_scale"),
            "ref_scale_warn": bool(meas.get("ref_scale_warn", False)),
            "baseline_source": meas.get("live_ref_scale_baseline_source") or "none",
            "chart_state": chart_trace.get("chart_state", "unknown"),
            "chart_warning_reason": chart_trace.get("chart_warning_reason", "") or "",
        }
        if self.config.measurement.enable_interval_csv_log:
            self.logger.tick(log_data)
        return log_data

    def _handle_key(self, log_data: dict, meas: dict) -> bool:
        """キー入力に応じた操作を実行する。"""
        return self.key_handler.handle(log_data, meas)

    def get_camera_init_time(self) -> float:
        """カメラ初期化時刻 (time.time()) を返す。"""
        return self._camera_init_time

    def process_frame(self) -> bool:
        """1フレーム分の取得・処理・描画・ログ・入力処理を行う。"""
        frame_bgr, raw_bayer, metadata = self._capture_and_preprocess()
        roi = self._extract_rois(raw_bayer, metadata)
        meas = self._compute_measurements(roi)
        analysis_frame_for_bundle = None
        display_frame_bgr = frame_bgr
        correction_info = {"ref_L_raw": None, "ref_L_corr": None, "bright_scale": 1.0}
        apply_live_display_correction = False
        display_mode = self.config.display.display_color_mode
        is_legacy_mode = display_mode == DisplaySettings.DISPLAY_COLOR_MODE_LEGACY
        is_live_correction_enabled = self.config.display.enable_live_color_correction
        if display_mode == DisplaySettings.DISPLAY_COLOR_MODE_NATURAL:
            apply_live_display_correction = False
        ref_w = max(int(roi.get("ref_disp_w", self.config.processing.spot_size_ref)), 2)
        ref_h = max(int(roi.get("ref_disp_h", self.config.processing.spot_size_ref * self.config.processing.aspect_ref)), 2)
        roi_ref_raw_disp = self.live_color_corrector.get_ROI(
            frame_bgr, self.config.processing.posi_ref[0], self.config.processing.posi_ref[1], ref_w, ref_h,
        )
        lab_ref_raw_disp = self.live_color_corrector.get_Lab_mean(roi_ref_raw_disp)
        if lab_ref_raw_disp is not None:
            correction_info["ref_L_raw"] = float(lab_ref_raw_disp[0])
            correction_info["ref_L_corr"] = float(lab_ref_raw_disp[0])
        if is_legacy_mode and is_live_correction_enabled and lab_ref_raw_disp is not None:
            apply_live_display_correction = True
        if apply_live_display_correction:
            try:
                display_frame_bgr, bright_scale = (
                    self.live_color_corrector.bgr_color_correct_Lab(frame_bgr, lab_ref_raw_disp)
                )
                correction_info["bright_scale"] = float(bright_scale)
                roi_ref_corr_disp = self.live_color_corrector.get_ROI(
                    display_frame_bgr, self.config.processing.posi_ref[0],
                    self.config.processing.posi_ref[1], ref_w, ref_h,
                )
                lab_ref_corr_disp = self.live_color_corrector.get_Lab_mean(roi_ref_corr_disp)
                if lab_ref_corr_disp is not None:
                    correction_info["ref_L_corr"] = float(lab_ref_corr_disp[0])
            except Exception as e:
                display_frame_bgr = frame_bgr
                print(f"Warning: live color correction skipped: {e}")
        else:
            display_frame_bgr = frame_bgr
        composite = self._render(display_frame_bgr, roi, meas, correction_info)
        # モード切替トースト
        if hasattr(self, "last_mode_change_time") and self.last_mode_change_time > 0:
            elapsed = time.time() - self.last_mode_change_time
            draw_mode_toast(composite, f"Mode: {self.mode.current}", elapsed)
        if hasattr(self.key_handler, "last_mode_change_time"):
            kh_time = self.key_handler.last_mode_change_time
            if kh_time > self.last_mode_change_time:
                self.last_mode_change_time = kh_time
        # SpyderCheckr グリッドオーバーレイ
        kh = self.key_handler

        # Phase 7B / 7C: flip_warning state のとき、毎フレーム派手な警告バナーを描画.
        # user が画面を見ていなくても再 [P] 合格まで絶対に進めない hard-lock 状態.
        # Phase 7C: 英語 → 日本語に差替え. cv2 は CJK 非対応のため PIL で描画.
        if kh._chart_state == "flip_warning":
            composite = self._draw_flip_warning_banner(composite)

        chart_overlay_states = ("corner_input", "preview", "measuring")
        if kh._chart_state in chart_overlay_states and kh._grid_extractor is not None:
            analysis_composite = None
            if kh._chart_state in ("corner_input", "preview"):
                analysis_composite = make_chart_analysis_composite(
                    display_frame_bgr,
                    self.config.camera.left_panel_width,
                )
            if kh._chart_state == "preview":
                analysis_frame_for_bundle = analysis_composite.copy()
                kh._prepare_preview_patchwise_rois_if_needed(analysis_composite)

            corners_so_far = kh._chart_corners if kh._chart_state == "corner_input" else []
            composite = kh._grid_extractor.draw_overlay(
                composite, corners_so_far,
                white_local_idx=SPYDERCHECKR_48_WHITE_PATCH_INDEX,
            )
            h_img = composite.shape[0]
            workflow_status = None
            if kh._chart_state in ("preview", "measuring"):
                workflow_status = kh.get_chart_workflow_status() or {}
                composite = self._draw_chart_orientation_badge(
                    composite,
                    _chart_orientation_badge_lines(workflow_status),
                )
            if kh._chart_state == "preview":
                source = str(workflow_status.get("source", "")).strip()
                orientation_order = str(
                    workflow_status.get("chart_orientation_order", "")
                ).strip()
                source_label = _preview_badge_source_label(source)
                badge_line = f"Preview: {source_label}"
                if orientation_order == "visual_180":
                    badge_line = f"{badge_line}  UPSIDE-DOWN / remap"
                elif orientation_order == "visual":
                    badge_line = f"{badge_line}  UPRIGHT"
                elif orientation_order:
                    badge_line = f"{badge_line}  orient={orientation_order}"
                hint_line = "P/Enter start  G save  ESC/q cancel  +/- hinge  ,/. ROI"
                bx = self.config.camera.left_panel_width + 10
                by = h_img - 28
                badge_size, _ = cv2.getTextSize(
                    badge_line,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    1,
                )
                cv2.rectangle(
                    composite,
                    (bx - 8, by - 18),
                    (bx + badge_size[0] + 8, by + 6),
                    (0, 0, 0),
                    cv2.FILLED,
                )
                cv2.putText(
                    composite,
                    badge_line,
                    (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    composite,
                    hint_line,
                    (bx, h_img - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (190, 205, 215),
                    1,
                    cv2.LINE_AA,
                )
            elif kh._chart_state == "corner_input":
                n = len(kh._chart_corners)
                banner = f"Click 4 corners ({n}/4): 1A  1H  6H  6A  [ESC] Cancel"
                bx = self.config.camera.left_panel_width + 10
                cv2.putText(composite, banner, (bx, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        # キャリブレーション有効期限リマインダー（4時間超で5分ごとに表示）
        if hasattr(self, "key_handler") and self.key_handler.ccm_store is not None:
            _ccm_meta = self.key_handler.ccm_store.metadata
            _created_str = _ccm_meta.get("created") if _ccm_meta else None
            if _created_str:
                try:
                    from datetime import datetime as _dt
                    _ccm_dt = _dt.fromisoformat(_created_str)
                    _elapsed_h = (_dt.now() - _ccm_dt).total_seconds() / 3600.0
                    _now = time.time()
                    if _elapsed_h > 4.0 and (_now - self._last_recalib_reminder_time) > 300:
                        _px = self.config.camera.left_panel_width + 10
                        cv2.putText(
                            composite,
                            f"RECALIB REMINDER ({_elapsed_h:.1f}h)",
                            (_px, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 200, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        self._last_recalib_reminder_time = _now
                except (ValueError, TypeError):
                    pass

        log_data = self._output(composite, roi, meas)

        if self.session_recorder and self.session_recorder.is_recording:
            self.session_recorder.push(meas)

        quit_requested = self._handle_key(log_data, meas)

        # ref_anchor_lab 同期は不要（anchor は _compute_measurements 内で毎フレーム計算）
        # if self.key_handler.ref_anchor_lab is not None:
        #     self.ref_anchor_lab = self.key_handler.ref_anchor_lab
        #     self.key_handler.ref_anchor_lab = None

        if self.key_handler.preview_freeze_bundle_pending:
            self.key_handler._save_chart_preview_freeze_bundle(
                analysis_frame_for_bundle
            )

        if self.key_handler.snapshot_pending:
            self._save_snapshot(composite, log_data, meas)
            self.key_handler.snapshot_pending = False

        return quit_requested

    def close(self) -> None:
        """使用中リソースを解放する。"""
        self.roi_persistence.save()
        self.logger.close()
        self.picam2.stop()
        self.picam2.close()
        self.window_manager.destroy()


def main():
    """CSIカメラ処理を起動し、終了処理まで制御する。"""
    try:
        proc = CSIVideoProcessor()
    except Exception as e:
        print(f"Error during CSIVideoProcessor initialization: {e}")
        sys.exit(1)

    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    try:
        while True:
            try:
                if proc.process_frame():
                    break
                consecutive_errors = 0
            except KeyboardInterrupt:
                break
            except Exception as e:
                consecutive_errors += 1
                print(
                    f"\u26a0 \u30d5\u30ec\u30fc\u30e0\u51e6\u7406\u30a8\u30e9\u30fc "
                    f"({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}"
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("\u9023\u7d9a\u30a8\u30e9\u30fc\u4e0a\u9650\u306b\u9054\u3057\u307e\u3057\u305f\u3002\u30ab\u30e1\u30e9\u518d\u63a5\u7d9a\u3092\u8a66\u307f\u307e\u3059...")
                    try:
                        proc.picam2.stop()
                        proc.picam2.close()
                        time.sleep(1)
                        proc._init_camera()
                        consecutive_errors = 0
                        print("\u30ab\u30e1\u30e9\u518d\u63a5\u7d9a\u306b\u6210\u529f\u3057\u307e\u3057\u305f\u3002")
                    except Exception as reconnect_err:
                        print(f"\u30ab\u30e1\u30e9\u518d\u63a5\u7d9a\u306b\u5931\u6557: {reconnect_err}")
                        break
    finally:
        proc.close()
        if proc.video_out is not None:
            proc.video_out.release()
            print("- Saved the video file.")
