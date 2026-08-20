"""
CSIカメラ接続・設定・ROI抽出・補正・露出制御モジュール。

CSIカメラ（Picamera2）の初期化、RAW Bayer処理、
フラットフィールド補正、適応的露出制御を提供する。
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2

from .colorimeter_common import (
    DisplaySettings,
    ProcessingSettings,
    MeasurementSettings,
    PanelLayoutSettings,
    DarkFrameManager,  # noqa: F401 - kept for external import compatibility
    compute_gain_percentile,
    _find_calibration_file,
    _remove_cleared_marker,
    get_today_calibration_dir,
)
from .flat_radial_profile import _compute_bayer_luminance
from .flat_2d_smoothed import _build_smoothed_gain_map

# ---------------------------------------------------------------------------
# Qt フォントディレクトリの早期設定
# ---------------------------------------------------------------------------
_QT_FONTDIR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
]
_qt_fontdir = os.environ.get("QT_QPA_FONTDIR", "").strip()
if (not _qt_fontdir) or (not os.path.isdir(_qt_fontdir)):
    for _cand in _QT_FONTDIR_CANDIDATES:
        if os.path.isdir(_cand):
            os.environ["QT_QPA_FONTDIR"] = _cand
            break


# ===========================================================================
# CSICameraSettings
# ===========================================================================


@dataclass
class CSICameraSettings:
    """
    CSIカメラの低レベル設定値を保持するデータクラス。

    機能:
      - 露光・ゲイン・RAWストリーム構成など、Picamera2初期化に必要な固定値を集約する。
    入力:
      - ユーザー指定または既定の露光値、ゲイン、カラーゲイン、RAWフォーマット設定。
    出力:
      - カメラ接続・設定適用処理で参照される設定値オブジェクトを提供する。

    Attributes:
      exposure_time_us (int): 初期露光時間[us]。
      analogue_gain (float): 初期アナログゲイン。
      colour_gains (tuple): AWB無効時の固定カラーゲイン。
      raw_format (str): RAWストリームのピクセルフォーマット。
      raw_size (tuple): RAWストリーム解像度 (W, H)。
      display_size (tuple): 表示フレーム解像度 (W, H)。
      panel_width (int): 右サイドパネル幅[px]。
      left_panel_width (int): 左サイドパネル幅[px]。
      frame_rate (float): 目標フレームレート[fps]。
      buffer_count (int): Picamera2の内部バッファ数。
    """

    exposure_time_us: int = 20000
    analogue_gain: float = 1.0
    colour_gains: tuple = (1.5, 1.2)
    raw_format: str = "SBGGR12"
    raw_size: tuple = (2028, 1520)
    display_size: tuple = (800, 600)
    panel_width: int = 280
    left_panel_width: int = 220
    frame_rate: float = 30.0
    buffer_count: int = 6


# ===========================================================================
# CSIConfig
# ===========================================================================


class CSIConfig:
    """
    CSI向けの設定集約クラス。

    機能:
      - CSIカメラ設定・表示設定・処理設定・計測設定を統合して保持する。
      - 各サブシステムへの依存注入元として機能する。
    入力:
      - 各設定クラスの既定値または呼び出し元で上書きされた設定値。
    出力:
      - パイプライン各コンポーネントへ共有される統合設定オブジェクトを提供する。

    Attributes:
      camera (CSICameraSettings): カメラI/Oに関する設定。
      display (DisplaySettings): 表示・UI描画に関する設定。
      processing (ProcessingSettings): ROIや補正など処理パラメータ。
      measurement (MeasurementSettings): 測色・安定性判定の設定。
      panel_layout (PanelLayoutSettings): パネル要素のレイアウト設定。
    """

    def __init__(self):
        """設定サブクラスを初期化して保持する。"""
        self.camera = CSICameraSettings()
        self.display = DisplaySettings()
        self._apply_display_fallbacks()
        self.processing = ProcessingSettings()
        self.measurement = MeasurementSettings()
        self.panel_layout = PanelLayoutSettings()

    def _apply_display_fallbacks(self):
        """旧設定互換: display_color_mode 未定義時に既定値を補完する。"""
        if not hasattr(self.display, "display_color_mode"):
            self.display.display_color_mode = DisplaySettings.DISPLAY_COLOR_MODE_NATURAL
            return
        mode = str(getattr(self.display, "display_color_mode", "")).strip()
        if mode == "":
            self.display.display_color_mode = DisplaySettings.DISPLAY_COLOR_MODE_NATURAL


# ===========================================================================
# CSICameraConnector
# ===========================================================================


class CSICameraConnector:
    """
    Picamera2を用いてCSIカメラを初期化・接続するクラス。

    機能:
      - CSIConfigを入力としてRAW/表示ストリーム構成を作成する。
      - 固定露光・固定ホワイトバランス条件でカメラ接続を確立する。
    入力:
      - `CSIConfig` のカメラ設定値（露光、ゲイン、RAWフォーマット、解像度）。
    出力:
      - 起動済み `Picamera2` インスタンスとカメラ情報の標準出力を返す。
    """

    def __init__(self, config: CSIConfig):
        """
        接続に必要な設定を保持する。

        Args:
          config: カメラ接続時に使用するCSI設定。
        """
        self.config = config

    @staticmethod
    def _build_controls_natural(cam: CSICameraSettings) -> dict:
        """natural表示モード向けの初期controlsを返す。"""
        return {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": cam.exposure_time_us,
            "AnalogueGain": cam.analogue_gain,
            "ColourGains": cam.colour_gains,
        }

    @staticmethod
    def _build_controls_legacy(cam: CSICameraSettings) -> dict:
        """legacy表示モード向けの初期controlsを返す。"""
        return {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": cam.exposure_time_us,
            "AnalogueGain": cam.analogue_gain,
            "ColourGains": cam.colour_gains,
        }

    def connect(self) -> Picamera2:
        """
        Picamera2を構成してカメラを起動する。

        Returns:
          Picamera2: 接続済みカメラインスタンス。
        """
        # LED照明環境向け: ISPのCCMを単位行列に置換し、
        # ColourGainsのみで色補正する（daylight用CCMの緑かぶりを防止）。
        tuning = Picamera2.load_tuning_file("imx477.json")
        ccm_algo = Picamera2.find_tuning_algo(tuning, "rpi.ccm")
        if ccm_algo and "ccms" in ccm_algo:
            for entry in ccm_algo["ccms"]:
                entry["ccm"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        picam2 = Picamera2(tuning=tuning)
        cam = self.config.camera
        display_mode = self.config.display.display_color_mode
        # Picamera2 の "RGB888" は実際に BGR メモリ順（OpenCV互換）を返す。
        # "BGR888" は逆に RGB 順を返すため、使ってはならない。
        main_cfg = {"size": cam.display_size, "format": "RGB888"}
        video_config = picam2.create_video_configuration(
            main=main_cfg,
            raw={"format": cam.raw_format, "size": cam.raw_size},
            buffer_count=cam.buffer_count,
            controls={"FrameRate": cam.frame_rate},
        )
        picam2.configure(video_config)
        if display_mode == DisplaySettings.DISPLAY_COLOR_MODE_LEGACY:
            controls = self._build_controls_legacy(cam)
        else:
            controls = self._build_controls_natural(cam)
        picam2.set_controls(controls)
        picam2.start()
        self.print_camera_info(picam2)
        return picam2

    @staticmethod
    def print_camera_info(picam2):
        """
        カメラモデルや露出/ゲイン範囲を標準出力に表示する。

        Args:
          picam2: 接続済みのPicamera2インスタンス。
        """
        props = picam2.camera_properties
        print("\n### CSI Camera Info:")
        print(f"    Model: {props.get('Model', 'unknown')}")
        print(f"    PixelArraySize: {props.get('PixelArraySize', 'unknown')}")
        controls = picam2.camera_controls
        exp_limits = controls.get("ExposureTime", (0, 0, 0))
        gain_limits = controls.get("AnalogueGain", (0, 0, 0))
        print(f"    ExposureTime range: {exp_limits[0]} - {exp_limits[1]}")
        print(f"    AnalogueGain range: {gain_limits[0]} - {gain_limits[1]}")

    @staticmethod
    def get_exposure_limits(picam2) -> tuple:
        """
        露光時間の最小/最大値を取得する。

        Args:
          picam2: Picamera2インスタンス。
        Returns:
          tuple: (min_exposure, max_exposure)
        """
        limits = picam2.camera_controls.get("ExposureTime", (100, 1000000, 20000))
        return limits[0], limits[1]


# ===========================================================================
# BayerROIExtractor
# ===========================================================================


class BayerROIExtractor:
    """
    RAW Bayer配列からROIを抽出し、RGB平均値を算出するクラス。

    機能:
      - 12bit RAW Bayer入力を整形し、表示座標からRAW座標へ変換する。
      - 変換後のROIを抽出し、正規化済み/未正規化のRGB統計量を算出する。
    入力:
      - `CSIConfig`（RAWサイズ、表示サイズ）、センサーサイズ、RAW配列、メタデータ。
    出力:
      - RAW座標系のROI配列、正規化/未正規化のRGB平均値 `[R, G, B]` を返す。
    """

    def __init__(self, config: CSIConfig, sensor_size: tuple = None, bayer_pattern=None):
        """
        変換に必要な座標系パラメータを初期化する。

        Args:
          config: RAWサイズと表示サイズを含むCSI設定。
          sensor_size: センサー全体サイズ (W, H)。NoneならRAWサイズを使用。
          bayer_pattern: Bayerパターン。Noneなら raw_format から自動判定。
        """
        from .colorimeter_common import BayerPattern
        self.raw_w, self.raw_h = config.camera.raw_size
        self.disp_w, self.disp_h = config.camera.display_size
        self.max_val = 4095.0
        if bayer_pattern is not None:
            self.bayer_pattern = bayer_pattern
        else:
            try:
                self.bayer_pattern = BayerPattern.from_raw_format(config.camera.raw_format)
            except ValueError:
                self.bayer_pattern = BayerPattern.BGGR
        # フルセンサーサイズ (ScalerCropの座標系)
        if sensor_size is not None:
            self.sensor_w, self.sensor_h = sensor_size
        else:
            self.sensor_w, self.sensor_h = self.raw_w, self.raw_h

    def parse_raw(self, raw_array: np.ndarray) -> np.ndarray:
        """
        RAW配列をuint16の2次元Bayer配列に整形する。

        Args:
          raw_array: Picamera2から取得したRAW配列。
        Returns:
          np.ndarray: 形状(H, W)のuint16 RAW配列。
        """
        raw_uint16 = raw_array.view(np.uint16)
        if raw_uint16.ndim == 1:
            stride_words = len(raw_uint16) // self.raw_h
            raw_uint16 = raw_uint16.reshape(self.raw_h, stride_words)
        if raw_uint16.shape[1] > self.raw_w:
            raw_uint16 = raw_uint16[:, : self.raw_w]
        return raw_uint16

    def display_to_raw_coords(
        self,
        disp_x,
        disp_y,
        disp_w,
        disp_h,
        metadata,
        flip_horizontal=False,
        flip_vertical=False,
    ) -> tuple:
        """
        表示座標系のROIをRAW座標系に変換する。

        Args:
          disp_x, disp_y: 表示座標の左上。
          disp_w, disp_h: 表示座標のROI幅・高さ。
          metadata: Picamera2のメタデータ。
          flip_horizontal: 表示で水平反転されているか。
          flip_vertical: 表示で垂直反転されているか。
        Returns:
          tuple: (raw_x, raw_y, raw_w, raw_h)
        """
        disp_w = max(int(disp_w), 2)
        disp_h = max(int(disp_h), 2)
        if flip_horizontal:
            disp_x = self.disp_w - int(disp_x) - disp_w
        if flip_vertical:
            disp_y = self.disp_h - int(disp_y) - disp_h
        disp_x = max(0, min(int(disp_x), self.disp_w - disp_w))
        disp_y = max(0, min(int(disp_y), self.disp_h - disp_h))

        scaler_crop = metadata.get("ScalerCrop", (0, 0, self.sensor_w, self.sensor_h))
        crop_x, crop_y, crop_w, crop_h = scaler_crop
        # ScalerCropはフルセンサー座標系 → RAWストリーム座標系へ変換
        s2r_x = self.raw_w / self.sensor_w
        s2r_y = self.raw_h / self.sensor_h
        scale_x = crop_w / self.disp_w
        scale_y = crop_h / self.disp_h
        raw_x = int((crop_x + disp_x * scale_x) * s2r_x) & ~1
        raw_y = int((crop_y + disp_y * scale_y) * s2r_y) & ~1
        raw_w = int(disp_w * scale_x * s2r_x) & ~1
        raw_h = int(disp_h * scale_y * s2r_y) & ~1
        raw_w = max(raw_w, 2)
        raw_h = max(raw_h, 2)
        raw_x = max(0, min(raw_x, self.raw_w - raw_w))
        raw_y = max(0, min(raw_y, self.raw_h - raw_h))
        return raw_x, raw_y, raw_w, raw_h

    def extract_raw_roi(self, raw_bayer, x, y, w, h) -> np.ndarray:
        """
        RAW Bayer配列からROIを切り出す。

        Args:
          raw_bayer: 2次元RAW Bayer配列。
          x, y: ROI左上座標。
          w, h: ROI幅・高さ。
        Returns:
          np.ndarray: ROI配列。
        """
        y_end = min(y + h, raw_bayer.shape[0])
        x_end = min(x + w, raw_bayer.shape[1])
        return raw_bayer[y:y_end, x:x_end]

    def _extract_bayer_channel_means(self, roi_uint16: np.ndarray) -> np.ndarray:
        """
        Bayerパターンに応じてR/G/B平均を求める。

        Args:
          roi_uint16: RAW ROI配列。
        Returns:
          np.ndarray: [R, G, B]の平均値(float32)。
        """
        # Bayerパターンからチャンネルスライスを取得
        # 空配列ガード: ROI高さまたは幅が1の場合に奇数インデックス行/列サブアレイが
        # 空になり .mean() が NaN を返すため、事前にサイズチェックする。
        sl = self.bayer_pattern.channel_slices()
        B_arr = roi_uint16[sl["B"]]
        G1_arr = roi_uint16[sl["G1"]]
        G2_arr = roi_uint16[sl["G2"]]
        R_arr = roi_uint16[sl["R"]]
        B = float(B_arr.mean()) if B_arr.size > 0 else 0.0
        G1 = float(G1_arr.mean()) if G1_arr.size > 0 else 0.0
        G2 = float(G2_arr.mean()) if G2_arr.size > 0 else 0.0
        R = float(R_arr.mean()) if R_arr.size > 0 else 0.0
        G = (G1 + G2) / 2.0
        return np.array([R, G, B], dtype=np.float32)

    def extract_bayer_means_from_roi(self, roi_uint16: np.ndarray) -> np.ndarray:
        """
        12bit RAW ROIから正規化済みRGB平均を取得する。

        Args:
          roi_uint16: RAW ROI配列。
        Returns:
          np.ndarray: [R, G, B] (0..1) 正規化平均。
        """
        return self._extract_bayer_channel_means(roi_uint16) / self.max_val

    def get_raw_12bit_means_from_roi(self, roi_uint16: np.ndarray) -> np.ndarray:
        """
        12bit RAW ROIの未正規化RGB平均を取得する。

        Args:
          roi_uint16: RAW ROI配列。
        Returns:
          np.ndarray: [R, G, B] 12bit平均値。
        """
        return self._extract_bayer_channel_means(roi_uint16)


# ===========================================================================
# FlatFieldManager
# ===========================================================================


class FlatFieldManager:
    """
    フラットフィールド補正の取得・保存・適用を行うクラス。

    機能:
      - 均一白色板の撮影結果から **per-pixel luminance 逆数ゲイン** を生成・保存する。
      - RAW ROI にゲインマップを乗算し、pixel-level で照明ムラを補正する。
      - 全 Bayer チャネル (B, G1, G2, R) に **同じ gain 値** を適用するため、flat 面の色に
        関係なく Ref ROI の R:G:B 比率が維持される (CCM 行列は不影響)。

    アルゴリズム (gain map 生成、Phase 7 方式):
      1. 半解像度 luminance を計算する ((B + G1 + G2 + R) / 4)。
      2. `global_mean_lum = mean(lum_img[valid_mask_half])` を取得する
         (valid 領域内 luminance の平均)。
      3. `gain_half(x,y) = global_mean_lum / max(lum_img(x,y), 1e-6)` を計算する
         (per-pixel 逆数、平滑化なし)。
      4. `valid_mask` 外は `gain = 1.0` に固定する。
      5. `config` 指定時、Ref ROI 位置での gain 平均で全体を除算し
         Ref 位置での gain を 1.0 に正規化する (Ref anchor 正規化:
         cross-substrate 再現性補強)。
      6. `np.repeat` で Bayer 原寸 shape へ upsample し、全 ch に同じ値を代入する。

    特徴:
      - **pixel-level flatness**: 同一光学系で flat 撮影 → 同じ光学系で target
        撮影した場合、vignetting だけでなく高周波不均一も補正される。
      - radial 前提を仮定しない (非対称 vignetting に対応)。
      - 色非依存 (luminance-only gain を全 Bayer ch に同一適用)。
      - scipy 非依存 (Phase 7 で平滑化処理を廃止したため cv2.GaussianBlur も
        gain 生成では不要。参考実装のみ `_smooth_luminance_2d_masked` 関数として残置)。

    運用上の留意:
      - flat 画像の single frame per-pixel ノイズが gain に直接入るため、
        `capture_flat_field` は複数フレーム平均 (実機 64 枚) で SNR を確保する。
      - flat 取得時の dust / scratch は gain に焼き付くため、target 撮影時も
        同じ光学系位置で運用する前提。

    入力:
      - RAW フレーム列 (キャプチャ時)、RAW ROI 配列、ゲインマップ保存パス。
    出力:
      - 保存済みゲインマップ (`.npy`)、`flat_field_gain_meta.json` にメタデータ
        (`gain_map_version="per_pixel_lum_v1"`, `gain_method="per_pixel_luminance_v1"`,
        `optical_center_*`, `ref_anchor_gain` 等)、フラット補正後の `uint16` ROI 配列。
    """

    def __init__(self, save_path: str = ""):
        """
        保存先パスと内部状態を初期化する。

        Args:
          save_path: ゲインマップ (`.npy`) の保存先パス。空文字列ならデフォルト。
        """
        self.save_path = save_path or os.path.join(
            get_today_calibration_dir(), "flat_field_gain.npy"
        )
        self.gain_map = None
        self.valid_mask = None
        self.is_loaded = False

    def _valid_mask_path(self, gain_path: Optional[str] = None) -> str:
        """flat valid mask の保存先パスを返す。"""
        base_path = gain_path or self.save_path
        return os.path.join(os.path.dirname(base_path), "flat_field_valid_mask.npy")

    @staticmethod
    def _build_valid_mask(illuminated: np.ndarray) -> np.ndarray | None:
        """照明領域から最大連結領域の valid mask を生成する。"""
        mask_u8 = np.asarray(illuminated, dtype=np.uint8)
        if mask_u8.size == 0 or not np.any(mask_u8):
            return None
        try:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        except cv2.error:
            return mask_u8.astype(bool)
        if n_labels <= 1:
            return mask_u8.astype(bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return mask_u8.astype(bool)
        largest_label = int(np.argmax(areas)) + 1
        valid_mask = labels == largest_label
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        eroded_mask = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=1) > 0
        if np.any(eroded_mask):
            return eroded_mask
        if np.any(valid_mask):
            return valid_mask
        return mask_u8.astype(bool)

    def load_if_exists(self) -> bool:
        """
        既存のゲインマップを読み込む。最新の日付フォルダを優先して検索する。

        Returns:
          bool: 読み込み成功時True。
        """
        path = _find_calibration_file("flat_field_gain.npy")
        if path is None:
            return False
        try:
            loaded = np.load(path)
            self.gain_map = loaded.astype(np.float32)
            valid_mask_path = self._valid_mask_path(path)
            self.valid_mask = None
            if os.path.exists(valid_mask_path):
                try:
                    loaded_mask = np.load(valid_mask_path)
                    if loaded_mask.shape == loaded.shape:
                        self.valid_mask = loaded_mask.astype(bool)
                except Exception:
                    self.valid_mask = None

            # 後方互換性チェック: gain_map_version が per-pixel luminance 版か確認
            meta_path = path.replace(".npy", "_meta.json")
            gain_map_version = None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        _meta = json.load(f)
                    gain_map_version = _meta.get("gain_map_version")
                except Exception:
                    gain_map_version = None
            expected_version = "per_pixel_lum_v1"
            if gain_map_version != expected_version:
                print(
                    f"⚠ 古い flat_field_gain.npy (version={gain_map_version}) "
                    f"は per-pixel luminance 化されていません。再取得してください"
                )
                # 自動無効化: 古い版 (radial_v1, v2.0.0-radial-profile, None 等) は使用不可
                self.is_loaded = False
                self.gain_map = None
                self.valid_mask = None
                return False

            self.is_loaded = True
            _mask_str = "loaded" if self.valid_mask is not None else "legacy"
            print(
                f"- Flat field loaded: {path} shape={loaded.shape}"
                f" gain range: {self.gain_map.min():.3f}-{self.gain_map.max():.3f}"
                f" valid_mask={_mask_str} version={gain_map_version}"
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to load flat field: {e}")
            return False

    def capture_flat_field(
        self, picam2, bayer_extractor, dark_manager, n_frames: int = 64,
        bayer_pattern=None, config=None,
    ):
        """
        均一白色板を複数枚平均してゲインマップを生成・保存する。

        Args:
          picam2: Picamera2インスタンス。
          bayer_extractor: BayerROIExtractor。
          dark_manager: DarkFrameManager。
          n_frames: 取得フレーム数。
          bayer_pattern: Bayer パターン (None なら BGGR)。
          config: AppConfig 相当 (`processing.posi_ref` / `processing.spot_size_ref`
            / `processing.aspect_ref` / `display.flip_horizontal` /
            `display.flip_vertical` を参照)。**必須**: `None` を渡すと
            Ref anchor 正規化が実行されず、保存される gain map が
            `gain_map_version="per_pixel_lum_v1"` の意味を
            満たさなくなるため、`ValueError` を送出する (Codex C1 対策)。

        Raises:
          ValueError: `config is None` の場合。Phase 6 以降は Ref-anchored
            calibration が前提となり、config 無しでの呼出は許可しない。
          RuntimeError: Ref rect が `valid_mask_half` と 50 % 未満しか
            重ならない場合 (Ref ROI が黒縁/無効域に落ちている)。この状態で
            gain map を保存すると anchor 成功扱いの silently unanchored map
            になるため、明示的に calibration 失敗とする (Codex C2 対策)。
        """
        if config is None:
            raise ValueError(
                "config is required for Ref-anchored flat calibration "
                "(per_pixel_lum_v1). Pass AppConfig with "
                "processing.posi_ref / spot_size_ref / aspect_ref and "
                "display.flip_horizontal / flip_vertical."
            )
        print(f"Capturing {n_frames} flat field frames...")
        accumulator = None
        for i in range(n_frames):
            raw_array = picam2.capture_array("raw")
            raw_bayer = bayer_extractor.parse_raw(raw_array)
            if accumulator is None:
                accumulator = raw_bayer.astype(np.float64)
            else:
                accumulator += raw_bayer.astype(np.float64)
        assert accumulator is not None  # ループは必ず1回以上実行される
        flat_avg = accumulator / n_frames

        if dark_manager.is_loaded and dark_manager.dark_frame is not None:
            flat_clean = flat_avg - dark_manager.dark_frame.astype(np.float64)
            flat_clean = np.maximum(flat_clean, 1.0)
        else:
            flat_clean = np.maximum(flat_avg, 1.0)

        # per-pixel luminance 逆数による pixel-level gain
        # (全 Bayer ch 共通、色非依存、平滑化なし)
        from .colorimeter_common import BayerPattern
        if bayer_pattern is None:
            bayer_pattern = BayerPattern.BGGR

        # 厳しいマスク (valid_mask 用): 照明強度 10 % 以上
        # C3 対策: 計算中は self.valid_mask を書き換えず、ローカル変数のみで扱う。
        # 全計算成功後に関数末尾で self.* にコミットする (transactional update)。
        illuminated = flat_clean > (flat_clean.max() * 0.10)
        valid_mask_local = self._build_valid_mask(illuminated)

        # 緩いマスク (2D smoothed 計算用): 照明強度 2 % 以上
        illuminated_loose = flat_clean > (flat_clean.max() * 0.02)

        # 半解像度 luminance を計算
        lum_img = _compute_bayer_luminance(flat_clean, bayer_pattern)

        # 半解像度の valid_mask (緩い方を 2×2 ブロックで OR 縮約)
        h_full, w_full = flat_clean.shape
        h_half, w_half = h_full // 2, w_full // 2
        loose_half = (
            illuminated_loose[0:2 * h_half:2, 0:2 * w_half:2]
            | illuminated_loose[1:2 * h_half:2, 0:2 * w_half:2]
            | illuminated_loose[0:2 * h_half:2, 1:2 * w_half:2]
            | illuminated_loose[1:2 * h_half:2, 1:2 * w_half:2]
        )

        # 情報目的: luminance 重み付き重心 (meta の optical_center 用)
        ys_h, xs_h = np.mgrid[0:lum_img.shape[0], 0:lum_img.shape[1]]
        masked_lum = np.where(loose_half, lum_img, 0.0)
        total_lum = float(masked_lum.sum())
        if total_lum > 0.0:
            cx_lum = float((masked_lum * xs_h).sum() / total_lum)
            cy_lum = float((masked_lum * ys_h).sum() / total_lum)
        else:
            cx_lum = lum_img.shape[1] / 2.0
            cy_lum = lum_img.shape[0] / 2.0
        cx_bayer = cx_lum * 2.0 + 0.5
        cy_bayer = cy_lum * 2.0 + 0.5

        # Ref ROI 半解像度 rect を config から計算 (Ref anchor 正規化用)。
        # ScalerCrop / display_to_raw_coords 失敗時は RuntimeError を送出
        # (Phase 6 は Ref-anchored calibration が前提のため、silently skip しない)。
        try:
            ref_disp_x, ref_disp_y = config.processing.posi_ref
            ref_w = max(int(config.processing.spot_size_ref), 2)
            ref_h = max(int(ref_w * config.processing.aspect_ref), 2)
            metadata = picam2.capture_metadata()
            rx, ry, rw, rh = bayer_extractor.display_to_raw_coords(
                ref_disp_x, ref_disp_y, ref_w, ref_h, metadata,
                config.display.flip_horizontal,
                config.display.flip_vertical,
            )
            # RAW 座標 → 半解像度 (//2)。1 px 以上を保証して端の丸め誤差を吸収。
            ref_rect_half: tuple[int, int, int, int] = (
                int(rx) // 2,
                int(ry) // 2,
                max(1, int(rw) // 2),
                max(1, int(rh) // 2),
            )
        except Exception as e:
            raise RuntimeError(
                f"Ref rect 計算失敗 ({type(e).__name__}: {e}) — "
                "Ref anchor 正規化不可。config.processing.posi_ref / "
                "spot_size_ref / aspect_ref と ScalerCrop metadata を確認"
            ) from e

        # per-pixel luminance gain map を構築 (Bayer 原寸、全 ch 共通)
        # C3 対策: 戻り値はローカル変数に格納し、RuntimeError 前に self.* を
        # 触らない。全ての計算と検査を通過した後、関数末尾で一括コミットする。
        gain_map_local, ref_anchor_gain = _build_smoothed_gain_map(
            flat_clean.shape, lum_img, loose_half,
            ref_rect_half=ref_rect_half,
        )
        # Ref rect を渡したにも関わらず正規化が skip された場合
        # (valid_mask との重なり不足 / near-zero mean) は calibration 失敗扱い
        # (C2 対策: silently unanchored map を「成功」として保存させない)。
        # ここで RuntimeError を raise しても、self.gain_map / self.valid_mask /
        # self.is_loaded は未変更のため呼出元の既存 state がそのまま保たれる (C3)。
        if ref_anchor_gain is None:
            raise RuntimeError(
                "Ref anchor 正規化に失敗しました "
                f"(ref_rect_half={ref_rect_half})。"
                "Ref ROI が有効領域 (valid_mask) 外または被覆率 50 % 未満の "
                "可能性があります。Ref 位置を再設定するか、均一照明を確認してください。"
            )

        # C1 対策: clip [0.25, 4.0] と valid_mask 外 1.0 固定は
        # `_build_smoothed_gain_map` 内 (Ref anchor 正規化の前) で適用済み。
        # 本関数側で再適用すると anchor 後の Ref 平均が 1.0 からズレるため、
        # ここでは一切触らない (Codex Cycle 1 C1)。
        gain_map_local = gain_map_local.astype(np.float32)

        # 既存コードとの互換のため effective_mask 変数も残す (ローカル参照)
        effective_mask = valid_mask_local if valid_mask_local is not None else illuminated

        # --- ここまで全て成功。最終 commit (self.* への一括代入) ---
        # (C3 対策: 失敗時は self.* が一切変更されず、既存 state が維持される)
        self.valid_mask = valid_mask_local
        self.gain_map = gain_map_local
        self.is_loaded = True

        np.save(self.save_path, self.gain_map)
        valid_mask_path = self._valid_mask_path()
        if self.valid_mask is not None:
            np.save(valid_mask_path, self.valid_mask.astype(bool))
            _remove_cleared_marker("flat_field_valid_mask.npy")
        elif os.path.exists(valid_mask_path):
            os.remove(valid_mask_path)
        _remove_cleared_marker("flat_field_gain.npy")
        n_illum = int(effective_mask.sum())
        n_total = int(flat_clean.size)
        gain_p90 = compute_gain_percentile(self.gain_map, 90, mask=self.valid_mask)
        valid_fraction = float(n_illum / n_total) if n_total > 0 else 0.0
        print(f"- Flat field gain map saved: {self.save_path}")
        print(
            f"  valid: {n_illum}/{n_total} px ({valid_fraction*100:.1f}%)"
            f"  gain range: {self.gain_map.min():.3f} - {self.gain_map.max():.3f}"
            f"  valid p90={gain_p90:.2f}"
        )
        if valid_fraction < 0.40:
            print(
                f"⚠ flat_field valid_fraction={valid_fraction:.3f}"
                " < 0.40 — 光学系点検推奨"
            )
        # メタデータ保存（露出時間・ゲイン・作成日時）
        try:
            meta = picam2.capture_metadata()
            flat_meta = {
                "exposure_us": meta.get("ExposureTime", 0),
                "analogue_gain": meta.get("AnalogueGain", 0.0),
                "gain_p90": gain_p90,
                "valid_fraction": valid_fraction,
                "created": datetime.now().isoformat(),
                "gain_map_version": "per_pixel_lum_v1",
                "gain_method": "per_pixel_luminance_v1",
                "optical_center_lum": [float(cx_lum), float(cy_lum)],
                "optical_center_bayer": [float(cx_bayer), float(cy_bayer)],
                # C1/C2 対策以降、ここに到達する時点で ref_anchor_gain は
                # 必ず float。None の場合は既に RuntimeError で reject 済み。
                "ref_anchor_gain": float(ref_anchor_gain),
            }
            meta_path = self.save_path.replace(".npy", "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(flat_meta, f, indent=2)
        except Exception as e:
            print(f"Warning: フラットフィールドメタデータの保存に失敗: {e}")

    def check_exposure_consistency(self, current_exposure_us: int) -> None:
        """保存時の露出時間と現在の露出を比較し、50%以上乖離していれば警告する。

        Args:
          current_exposure_us: 現在の露出時間[us]。
        """
        meta_path = self.save_path.replace(".npy", "_meta.json")
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            saved_exp = meta.get("exposure_us", 0)
            if saved_exp > 0 and current_exposure_us > 0:
                ratio = abs(saved_exp - current_exposure_us) / saved_exp
                if ratio > 0.5:
                    print(
                        f"⚠ フラットフィールド取得時の露出({saved_exp}us)と"
                        f"現在の露出({current_exposure_us}us)が"
                        f"{ratio*100:.0f}%乖離しています。再取得を推奨します。"
                    )
        except (json.JSONDecodeError, OSError):
            pass

    def correct_roi(
        self, raw_roi: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """
        ROIにフラットフィールドゲイン補正を適用する。

        Args:
          raw_roi: ダーク減算済みRAW ROI配列。
          x, y: RAW座標系でのROI左上。
          w, h: RAW座標系でのROI幅・高さ。
        Returns:
          np.ndarray: 補正後のuint16 ROI配列。
        """
        if self.gain_map is None:
            return raw_roi
        y_end = min(y + h, self.gain_map.shape[0])
        x_end = min(x + w, self.gain_map.shape[1])
        gain_roi = self.gain_map[y:y_end, x:x_end]
        if self.valid_mask is not None:
            mask_roi = self.valid_mask[y:y_end, x:x_end]
            gain_roi = np.where(mask_roi, gain_roi, 1.0)
        rh, rw = raw_roi.shape[:2]
        gh, gw = gain_roi.shape[:2]
        h_min, w_min = min(rh, gh), min(rw, gw)
        corrected = (
            raw_roi[:h_min, :w_min].astype(np.float32) * gain_roi[:h_min, :w_min]
        )
        return np.clip(corrected, 0, 4095).astype(np.uint16)


# ===========================================================================
# AdaptiveExposureController
# ===========================================================================


class AdaptiveExposureController:
    """
    参照ROIの明るさに応じて露光を調整する制御クラス。

    機能:
      - 参照ROIの12bit平均値をもとに露光時間を目標輝度へ段階的に追従させる。
    入力:
      - `Picamera2` インスタンス、`CSIConfig` の露光初期値、参照ROIの12bit平均値。
    出力:
      - ON/OFF状態、更新後露光時間、Picamera2への露光設定反映結果を返す。
    """

    def __init__(self, picam2, config: CSIConfig):
        """
        露光制御パラメータと制御状態を初期化する。

        Args:
          picam2: 露光時間を書き込むPicamera2インスタンス。
          config: 初期露光値を参照するCSI設定。
        """
        self.picam2 = picam2
        self.enabled = False
        self.current_exposure = config.camera.exposure_time_us
        self.target_mid = 750
        self.gain = 0.3
        self.update_interval = 30
        self.frame_count = 0
        self.min_exp, self.max_exp = CSICameraConnector.get_exposure_limits(picam2)

    def set_target(self, target_12bit: int) -> None:
        """AE の目標輝度を変更する（12bit スケール、200-3000 にクランプ）。"""
        self.target_mid = max(200, min(3000, int(target_12bit)))

    def lower_exposure(self, factor: float) -> int:
        """現在露光に係数を掛けて即時に露光時間を更新する。"""
        new_exp = int(self.current_exposure * factor)
        new_exp = max(self.min_exp, min(new_exp, self.max_exp))
        self.picam2.set_controls({"ExposureTime": new_exp})
        self.current_exposure = new_exp
        return new_exp

    def toggle(self) -> bool:
        """
        露光自動調整のON/OFFを切り替える。

        Returns:
          bool: 切替後の有効状態。
        """
        self.enabled = not self.enabled
        print(f"- Adaptive Exposure: {'ON' if self.enabled else 'OFF'}")
        return self.enabled

    def tick(self, ref_raw_mean_12bit: float) -> Optional[int]:
        """
        フレーム周期で露光時間を更新する。

        Args:
          ref_raw_mean_12bit: 参照ROIの12bit平均値。
        Returns:
          Optional[int]: 更新した露光時間。更新なしの場合はNone。
        """
        self.frame_count += 1
        if not self.enabled:
            return None
        if self.frame_count % self.update_interval != 0:
            return None
        error = self.target_mid - ref_raw_mean_12bit
        adjustment = self.gain * error / self.target_mid
        new_exp = int(self.current_exposure * (1.0 + adjustment))
        new_exp = max(self.min_exp, min(new_exp, self.max_exp))
        if new_exp != self.current_exposure:
            self.picam2.set_controls({"ExposureTime": new_exp})
            self.current_exposure = new_exp
        return new_exp
