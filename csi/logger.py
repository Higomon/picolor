"""
CSI計測ログモジュール。

CSI固有のヘッダ（露光/ゲイン列付き）でCSVログを出力する。
"""

import os
from datetime import datetime

from .colorimeter_common import MeasurementLogger


class CSIMeasurementLogger(MeasurementLogger):
    """
    CSI計測結果のCSVログを出力するロガー。

    機能:
      - `MeasurementLogger` を継承し、CSI固有のヘッダ（露光/ゲイン列付き）で記録する。
      - 定期ログとイベントログをCSVへ書き出し、計測トレースを保存する。
    入力:
      - 出力ディレクトリ、記録間隔、計測データ辞書（Lab/RAW/露光/品質警告）。
    出力:
      - ヘッダ付きCSVファイルと1行単位の計測ログを出力する。
    """

    _HEADER = [
        "timestamp",
        "L_star",
        "a_star",
        "b_star",
        "raw_R",
        "raw_G",
        "raw_B",
        "CV_L",
        "sigma_a",
        "sigma_b",
        "deltaE00",
        "event_type",
        "U_L",
        "U_a",
        "U_b",
        "exposure_us",
        "analogue_gain",
        "quality_warnings",
        "guard_state",
        "dev_I_pct",
        "dev_uniformity_pct",
        "dev_chromaticity",
        "guard_reason",
        "drift_guard_state",
        "drift_reason_axes",
        "ref_raw_mean",
        "ref_raw_baseline",
        "ref_drift_i_pct",
        "ref_drift_c",
        "ref_drift_u_pct",
        "legacy_u_rel_pct",
        "legacy_center_edge_pct",
        "ref_uneven_value_pct",
        "ref_uneven_drift_delta_pp",
        "ref_uneven_risk",
        "ref_uneven_abs_dominant",
        "ref_uneven_drift_dominant",
        "ref_uneven_reason",
        "ref_uneven_value_kind",
        "ref_uneven_direction_reversal",
        "ref_center_edge_pct",
        "ref_left_right_pct",
        "ref_top_bottom_pct",
        "ref_diag_pct",
        "ref_tile_p95_p05_pct",
        "ref_tile_max_dev_pct",
        "ref_tile_cv_pct",
        "ref_drift_j_pct",
        "ref_clip_high_pct",
        "ref_clip_low_pct",
        "ref_baseline_label",
        "ref_baseline_kind",
        "mode",
        "R_lin",
        "G_lin",
        "B_lin",
        "Lab_rel_L",
        "lab_rel_l_warn",
        "ref_scale",
        "ref_scale_warn",
        "baseline_source",
        "chart_state",
        "chart_warning_reason",
    ]

    @staticmethod
    def _format_optional_float(value, digits: int = 4) -> str:
        """未取得の数値は空欄、取得済みの数値は固定小数で出力する。"""
        if value is None:
            return ""
        return f"{float(value):.{digits}f}"

    @staticmethod
    def _format_bool(value) -> str:
        """CSV上で安定した lowercase boolean にする。"""
        return "true" if bool(value) else "false"

    def _ensure_file_open(self) -> None:
        """初回呼び出し時にCSVファイルを開き、メタデータコメントとヘッダーを書き込む。"""
        if self.csv_file is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        csv_path = os.path.join(self.log_dir, f"{ts}_measurement.csv")
        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        # メタデータコメント行
        meta_lines = [
            f"# created: {datetime.now().isoformat()}",
            "# illuminant: D65 (sRGB assumption)",
            "# reflectance_factor: 0.18",
        ]
        for line in meta_lines:
            self.csv_file.write(line + "\n")
        import csv
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self._HEADER)
        print(f"- CSV log started: {csv_path}")

    def _write_row(self, data: dict, event_type: str) -> None:
        """
        計測結果をCSVに1行書き込む。

        Args:
          data: 計測結果の辞書。
          event_type: イベント種別。
        """
        self._ensure_file_open()
        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            f"{data.get('L_star', 0.0):.4f}",
            f"{data.get('a_star', 0.0):.4f}",
            f"{data.get('b_star', 0.0):.4f}",
            f"{data.get('raw_R', 0.0):.1f}",
            f"{data.get('raw_G', 0.0):.1f}",
            f"{data.get('raw_B', 0.0):.1f}",
            f"{data.get('CV_L', 0.0):.6f}",
            f"{data.get('sigma_a', 0.0):.4f}",
            f"{data.get('sigma_b', 0.0):.4f}",
            f"{data.get('deltaE00', 0.0):.4f}",
            event_type,
            f"{data.get('U_L', 0.0):.4f}",
            f"{data.get('U_a', 0.0):.4f}",
            f"{data.get('U_b', 0.0):.4f}",
            f"{data.get('exposure_us', 0)}",
            f"{data.get('analogue_gain', 0.0):.1f}",
            data.get("quality_warnings", ""),
            data.get("guard_state", "NORMAL"),
            f"{data.get('dev_I_pct', 0.0):.4f}",
            f"{data.get('dev_uniformity_pct', 0.0):.4f}",
            f"{data.get('dev_chromaticity', 0.0):.6f}",
            data.get("guard_reason", ""),
            data.get("drift_guard_state", "OK"),
            data.get("drift_reason_axes", ""),
            f"{data.get('ref_raw_mean', 0.0):.1f}",
            f"{data.get('ref_raw_baseline', 0.0):.1f}",
            f"{data.get('ref_drift_i_pct', 0.0):.4f}",
            f"{data.get('ref_drift_c', 0.0):.6f}",
            f"{data.get('ref_drift_u_pct', 0.0):.4f}",
            f"{data.get('legacy_u_rel_pct', 0.0):.4f}",
            f"{data.get('legacy_center_edge_pct', 0.0):.4f}",
            self._format_optional_float(data.get("ref_uneven_value_pct")),
            self._format_optional_float(data.get("ref_uneven_drift_delta_pp")),
            self._format_optional_float(data.get("ref_uneven_risk")),
            data.get("ref_uneven_abs_dominant", ""),
            data.get("ref_uneven_drift_dominant", ""),
            data.get("ref_uneven_reason", ""),
            data.get("ref_uneven_value_kind", ""),
            self._format_bool(data.get("ref_uneven_direction_reversal", False)),
            self._format_optional_float(data.get("ref_center_edge_pct")),
            self._format_optional_float(data.get("ref_left_right_pct")),
            self._format_optional_float(data.get("ref_top_bottom_pct")),
            self._format_optional_float(data.get("ref_diag_pct")),
            self._format_optional_float(data.get("ref_tile_p95_p05_pct")),
            self._format_optional_float(data.get("ref_tile_max_dev_pct")),
            self._format_optional_float(data.get("ref_tile_cv_pct")),
            f"{data.get('ref_drift_j_pct', 0.0):.4f}",
            f"{data.get('ref_clip_high_pct', 0.0):.4f}",
            f"{data.get('ref_clip_low_pct', 0.0):.4f}",
            data.get("ref_baseline_label", "Base UNVER"),
            data.get("ref_baseline_kind", "none"),
            data.get("mode", "Lab"),
            f"{data.get('ratio_R', 0.0):.6f}",
            f"{data.get('ratio_G', 0.0):.6f}",
            f"{data.get('ratio_B', 0.0):.6f}",
            self._format_optional_float(data.get("Lab_rel_L")),
            self._format_bool(data.get("lab_rel_l_warn", False)),
            self._format_optional_float(data.get("ref_scale")),
            self._format_bool(data.get("ref_scale_warn", False)),
            data.get("baseline_source", "none"),
            data.get("chart_state", "unknown"),
            data.get("chart_warning_reason", ""),
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()
