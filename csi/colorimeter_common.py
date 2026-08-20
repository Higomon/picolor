"""
Colorimeter Common Module
=========================

CSI版・USB版で共有するクラス・関数を集約した共通モジュール。

機能:
  - CSI版とUSB版で共通利用する設定・処理・UI・保存ロジックを一元管理する。
  - 色変換、色差計算、安定性評価、品質判定などの測色処理を提供する。
  - フレーム変換、ROI操作、ログ保存など実運用に必要な補助機能を提供する。

入力:
  - 設定値オブジェクト群（表示設定、処理設定、計測設定、レイアウト設定）。
  - 画像配列、ROI座標、時系列バッファ、測色対象の数値データ。
  - 実行環境情報（OS種別、ユーザー名、保存パス、ログ出力先）。

出力:
  - Lab値、dE2000、安定性状態、ヒストグラム品質判定などの計測結果。
  - 補正後フレーム、描画済みUIフレーム、ROI更新結果などの処理結果。
  - CSVログ、ROI設定JSON、補正データファイルなどの保存成果物。
"""

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, ClassVar, Optional

class BayerPattern(Enum):
    """Bayerパターン定義。

    値は (R行オフセット, R列オフセット, G1行, G1列, G2行, G2列, B行, B列) のタプル。
    各オフセットは 0 または 1 で、2×2 サブサンプリングの開始位置を表す。
    """

    RGGB = (0, 0, 0, 1, 1, 0, 1, 1)
    BGGR = (1, 1, 0, 1, 1, 0, 0, 0)
    GRBG = (0, 1, 0, 0, 1, 1, 1, 0)
    GBRG = (1, 0, 0, 0, 1, 1, 0, 1)

    @classmethod
    def from_raw_format(cls, fmt: str) -> "BayerPattern":
        """Picamera2 の raw_format 文字列（例: 'SBGGR12'）からパターンを判定する。

        Args:
          fmt: RAW フォーマット文字列。
        Returns:
          BayerPattern: 判定されたパターン。
        Raises:
          ValueError: 不明なフォーマットの場合。
        """
        fmt_upper = fmt.upper()
        for pattern in cls:
            if pattern.name in fmt_upper:
                return pattern
        raise ValueError(f"未知の Bayer パターン: {fmt}")

    def channel_slices(self) -> dict:
        """R, G1, G2, B の numpy スライスを返す。

        Returns:
          dict: {"R": (row_slice, col_slice), "G1": ..., "G2": ..., "B": ...}
        """
        r_row, r_col, g1_row, g1_col, g2_row, g2_col, b_row, b_col = self.value
        return {
            "R": (slice(r_row, None, 2), slice(r_col, None, 2)),
            "G1": (slice(g1_row, None, 2), slice(g1_col, None, 2)),
            "G2": (slice(g2_row, None, 2), slice(g2_col, None, 2)),
            "B": (slice(b_row, None, 2), slice(b_col, None, 2)),
        }


_QT_FONTDIR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
]


def _ensure_qt_fontdir_env() -> None:
    """
    OpenCV(Qt)の初期化前に有効なフォントディレクトリを環境変数へ設定する。

    既存の QT_QPA_FONTDIR が無効パスなら上書きする。
    """
    current = os.environ.get("QT_QPA_FONTDIR", "").strip()
    if current and os.path.isdir(current):
        return
    for path in _QT_FONTDIR_CANDIDATES:
        if os.path.isdir(path):
            os.environ["QT_QPA_FONTDIR"] = path
            return


_ensure_qt_fontdir_env()

import cv2  # noqa: E402 - QT_QPA_FONTDIR must be set before importing cv2
import numpy as np  # noqa: E402 - keep import beside cv2 after Qt fontdir setup

try:
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401 - ImageFont used; Image/ImageDraw kept for compatibility

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ===========================================================================
# ref_scale: Y係数ベースの輝度スカラー
# ===========================================================================

_Y_COEFF = np.array([0.2126729, 0.7151522, 0.0721750], dtype=np.float64)


RUN_TYPE_START_OF_DAY = "start_of_day"
RUN_TYPE_END_OF_DAY = "end_of_day"
RUN_TYPE_REQUALIFICATION = "requalification"
RUN_TYPE_VERIFY_ONLY = "verify_only"
VALID_RUN_TYPES = (
    RUN_TYPE_START_OF_DAY,
    RUN_TYPE_END_OF_DAY,
    RUN_TYPE_REQUALIFICATION,
    RUN_TYPE_VERIFY_ONLY,
)
FORMAL_ACCEPTANCE_RUN_TYPES = (
    RUN_TYPE_START_OF_DAY,
    RUN_TYPE_END_OF_DAY,
    RUN_TYPE_REQUALIFICATION,
)
RUN_TYPE_DISPLAY_NAMES = {
    RUN_TYPE_START_OF_DAY: "運用開始判定 run",
    RUN_TYPE_END_OF_DAY: "運用終了前確認 run",
    RUN_TYPE_REQUALIFICATION: "再受入れ判定 run",
    RUN_TYPE_VERIFY_ONLY: "確認のみ run",
}
OPERATOR_GUI_ACCEPTANCE_HINT = (
    "画面の手順どおりに [D][F][W][P] を実行してください"
)
OPERATOR_GUI_RECHECK_HINT = "画面の手順どおりに [D][F][W][P] をやり直してください"
OPERATOR_GUI_CONTRACT_HINT = "設定を確認し、[D][F][W][P] をやり直してください"
DISPLAY_NAME_TO_RUN_TYPE = {
    display_name: run_type for run_type, display_name in RUN_TYPE_DISPLAY_NAMES.items()
}
RESULT_STATUS_ACCEPTED = "合格"
RESULT_STATUS_MINIMUM = "最低合格"
RESULT_STATUS_REJECTED = "失格"
RESULT_STATUS_RECHECK_REQUIRED = "要再判定"
VALID_RESULT_STATUSES = (
    RESULT_STATUS_ACCEPTED,
    RESULT_STATUS_MINIMUM,
    RESULT_STATUS_REJECTED,
    RESULT_STATUS_RECHECK_REQUIRED,
)
GATE_STATE_OPEN = "open"
GATE_STATE_BLOCKED = "blocked"
GATE_STATE_PROVISIONAL = "provisional"
VALID_GATE_STATES = (
    GATE_STATE_OPEN,
    GATE_STATE_BLOCKED,
    GATE_STATE_PROVISIONAL,
)
STEADY_STATE_REQUIRED_FIELDS = (
    "timestamp",
    "ref_scale",
    "lab_rel_l",
    "relative_rgb_mean",
    "relative_rgb_std",
    "consecutive_frames",
    "is_stable",
)
ACCEPTANCE_RESULT_JSON_REQUIRED_FIELDS = (
    "timestamp",
    "operator",
    "calibration_dir",
    "run_type",
    "accepted_reference_run_id",
    "warmup_elapsed_sec",
    "baseline_source",
    "baseline_value",
    "white_ratio_rgb",
    "avg_ref",
    "ref_scale",
    "relative_rgb_mean",
    "relative_rgb_std",
    "measured_lab",
    "delta_e",
    "roi_or_grid_source",
    "exposure",
    "mode",
    "result_status",
    "gate_state",
)
ACCEPTANCE_RESULT_JSON_ALTERNATIVE_FIELD_GROUPS = (
    ("software_version", "git_revision"),
    ("fixture_id", "positioning_method"),
)
ACCEPTANCE_RESULT_JSON_REQUIRED_ID_FIELDS = (
    "control_sample_id",
)
ACCEPTANCE_RESULT_LOG_REQUIRED_ITEMS = (
    "timestamp",
    "run_type",
    "operator",
    "calibration_dir",
    "accepted_reference_run_id",
    "warmup_elapsed_sec",
    "baseline_source",
    "baseline_value",
    "white_ratio_rgb",
    "avg_ref",
    "ref_scale",
    "relative_rgb_mean",
    "relative_rgb_std",
    "measured_lab",
    "delta_e",
    "roi_or_grid_source",
    "exposure",
    "mode",
    "result_status",
    "gate_state",
)
ACCEPTANCE_RESULT_LOG_ALTERNATIVE_ITEM_GROUPS = (
    ("software_version", "git_revision"),
    ("fixture_id", "positioning_method"),
)
ACCEPTANCE_RESULT_LOG_REQUIRED_ID_ITEMS = (
    "control_sample_id",
)
ACCEPTANCE_RESULT_OPERATIONAL_REQUIRED_FIELDS = (
    "control_sample_id",
    "context_profile_id",
    "measurement_context",
    "measurement_context_fingerprint",
)
ACCEPTANCE_RESULT_OPERATIONAL_ALTERNATIVE_FIELD_GROUPS = (
    ("fixture_id", "positioning_method"),
)
ACCEPTANCE_RESULT_OPERATIONAL_SELECTED_REFERENCE_FIELDS = (
    "accepted_reference_context_status",
    "selected_reference_measurement_context",
)
SOFTWARE_IDENTITY_FIELDS = (
    "software_version",
    "git_revision",
)
MEASUREMENT_CONTEXT_FIELDS = (
    "control_sample_id",
    "fixture_id",
    "positioning_method",
    "context_profile_id",
    "software_version",
    "git_revision",
)
MEASUREMENT_CONTEXT_REQUIRED_FIELDS = (
    "control_sample_id",
    "context_profile_id",
)
MEASUREMENT_CONTEXT_ALTERNATIVE_FIELD_GROUPS = (
    ("fixture_id", "positioning_method"),
)
DEFAULT_CONTROL_SAMPLE_ID = "SpyderCHECKR-48:4D"
VALID_BASELINE_SOURCES = (
    "neutral",
    "master_ref",
    "ref_train",
)
MEASUREMENT_GATE_RUN_TYPES = (
    RUN_TYPE_START_OF_DAY,
    RUN_TYPE_REQUALIFICATION,
)
REFERENCE_RELATION_NONE = "none"
REFERENCE_RELATION_SAME_DAY_PREVIOUS_RUN = "same_day_previous_run"
REFERENCE_RELATION_PREVIOUS_DAY_OR_OLDER = "previous_day_or_older"
ACCEPTED_REFERENCE_SELECTION_MODE_AUTO = "auto_latest"
ACCEPTED_REFERENCE_SELECTION_MODE_FIXED = "manual_fixed"
FIXED_ANCHOR_SELECTION_FILENAME = "fixed_anchor_selection.json"
ACCEPTANCE_REL_MEAN_TARGET_DELTA = 0.005
ACCEPTANCE_REL_MEAN_MIN_DELTA = 0.010
ACCEPTANCE_REL_STD_TARGET_MAX = 0.003
ACCEPTANCE_REL_STD_MIN_MAX = 0.005
ACCEPTANCE_ABSOLUTE_TARGET_MAX_DE = 1.0
ACCEPTANCE_ABSOLUTE_MIN_MAX_DE = 3.0
ACCEPTANCE_REF_SCALE_TARGET_DELTA = 0.010
ACCEPTANCE_REF_SCALE_MIN_DELTA = 0.050
ACCEPTANCE_LAB_REL_L_TARGET_MIN = 10.0
ACCEPTANCE_LAB_REL_L_TARGET_MAX = 11.0
ACCEPTANCE_LAB_REL_L_MIN_MIN = 9.5
ACCEPTANCE_LAB_REL_L_MIN_MAX = 11.0
ACCEPTANCE_REFERENCE_DIFF_TARGET_MAX_DE = 1.0
ACCEPTANCE_REFERENCE_DIFF_MIN_MAX_DE = 3.0
ACCEPTANCE_REFERENCE_DIFF_REL_MEAN_TARGET_DELTA = 0.005
ACCEPTANCE_REFERENCE_DIFF_REL_MEAN_MIN_DELTA = 0.010
ACCEPTANCE_REFERENCE_DIFF_SCALE_TARGET_DELTA = 0.010
ACCEPTANCE_REFERENCE_DIFF_SCALE_MIN_DELTA = 0.050
PREFLIGHT_REF_DRIFT_MAX = 0.05
EXPOSURE_DRIFT_WARN_MAX = 0.10
REFERENCE_AGE_WARNING_DAYS = 7
ACCEPTANCE_RUNNER_EXIT_CODE_MINIMUM = 21
ACCEPTANCE_RUNNER_EXIT_CODE_REJECTED = 22
ACCEPTANCE_RUNNER_EXIT_CODE_RECHECK_REQUIRED = 23
ACCEPTANCE_RUNNER_EXIT_CODE_MISSING_RESULT = 24
ACCEPTANCE_RUNNER_EXIT_CODE_RUNTIME_ERROR = 25
OPERATOR_GUIDANCE_MODE_GUIDED = "guided_runner"
OPERATOR_GUIDANCE_MODE_DEGRADED = "degraded_runner"
OPERATOR_GUIDANCE_MODE_MANUAL = "manual_interactive"
VALID_OPERATOR_GUIDANCE_MODES = (
    OPERATOR_GUIDANCE_MODE_GUIDED,
    OPERATOR_GUIDANCE_MODE_DEGRADED,
    OPERATOR_GUIDANCE_MODE_MANUAL,
)
REPRO_ROLLOUT_MODE_SHADOW = "shadow"
REPRO_ROLLOUT_MODE_ADVISORY = "advisory"
REPRO_ROLLOUT_MODE_ENFORCED = "enforced"
VALID_REPRO_ROLLOUT_MODES = (
    REPRO_ROLLOUT_MODE_SHADOW,
    REPRO_ROLLOUT_MODE_ADVISORY,
    REPRO_ROLLOUT_MODE_ENFORCED,
)
GUIDANCE_DEGRADED_REASON_AUTO_DETECT_DISABLED = "auto_detect_disabled"
GUIDANCE_DEGRADED_REASON_SCENE_TIMEOUT_MANUAL_CONFIRM = (
    "scene_timeout_manual_confirm"
)
GUIDANCE_DEGRADED_REASON_MANUAL_ROI_ADJUSTMENT = "manual_roi_adjustment"
GUIDANCE_DEGRADED_REASON_MANUAL_CORNER_ADJUSTMENT = "manual_corner_adjustment"
GUIDANCE_DEGRADED_REASON_SAVED_CORNERS_UNVERIFIED = "saved_corners_unverified"
VALID_GUIDANCE_DEGRADED_REASONS = (
    GUIDANCE_DEGRADED_REASON_AUTO_DETECT_DISABLED,
    GUIDANCE_DEGRADED_REASON_SCENE_TIMEOUT_MANUAL_CONFIRM,
    GUIDANCE_DEGRADED_REASON_MANUAL_ROI_ADJUSTMENT,
    GUIDANCE_DEGRADED_REASON_MANUAL_CORNER_ADJUSTMENT,
    GUIDANCE_DEGRADED_REASON_SAVED_CORNERS_UNVERIFIED,
)


def normalize_acceptance_run_type(
    run_type: str | None,
    *,
    default: str = RUN_TYPE_VERIFY_ONLY,
) -> str:
    """run_type を既定値付きで正規化し、許可値以外を拒否する。"""
    normalized_default = (default or "").strip()
    if normalized_default not in VALID_RUN_TYPES:
        raise ValueError(f"invalid default run_type: {default!r}")

    candidate = (run_type or "").strip() or normalized_default
    if candidate not in VALID_RUN_TYPES:
        raise ValueError(
            f"invalid run_type: {candidate!r}; expected one of {VALID_RUN_TYPES}"
        )
    return candidate


def get_run_type_display_name(run_type: str | None) -> str:
    """内部 run_type から UI 表示用の日本語名を返す。"""
    normalized = normalize_acceptance_run_type(run_type)
    return RUN_TYPE_DISPLAY_NAMES[normalized]


def get_run_type_from_display_name(display_name: str | None) -> str:
    """UI 表示用の日本語名から内部 run_type を返す。"""
    candidate = (display_name or "").strip()
    if candidate not in DISPLAY_NAME_TO_RUN_TYPE:
        raise ValueError(
            "invalid run_type display_name: "
            f"{candidate!r}; expected one of {tuple(DISPLAY_NAME_TO_RUN_TYPE)}"
        )
    return DISPLAY_NAME_TO_RUN_TYPE[candidate]


def normalize_result_status(result_status: str | None) -> str:
    """result_status を正規化し、許可値以外を拒否する。"""
    candidate = (result_status or "").strip()
    if candidate not in VALID_RESULT_STATUSES:
        raise ValueError(
            "invalid result_status: "
            f"{candidate!r}; expected one of {VALID_RESULT_STATUSES}"
        )
    return candidate


def normalize_gate_state(gate_state: str | None) -> str:
    """gate_state を正規化し、許可値以外を拒否する。"""
    candidate = (gate_state or "").strip()
    if candidate not in VALID_GATE_STATES:
        raise ValueError(
            f"invalid gate_state: {candidate!r}; expected one of {VALID_GATE_STATES}"
        )
    return candidate


def normalize_operator_guidance_mode(
    guidance_mode: str | None,
    *,
    default: str = OPERATOR_GUIDANCE_MODE_GUIDED,
) -> str:
    """operator guidance mode を固定値へ正規化する。"""
    normalized_default = (default or "").strip()
    if normalized_default not in VALID_OPERATOR_GUIDANCE_MODES:
        raise ValueError(
            "invalid default operator_guidance_mode: "
            f"{default!r}; expected one of {VALID_OPERATOR_GUIDANCE_MODES}"
        )
    candidate = (guidance_mode or "").strip() or normalized_default
    if candidate not in VALID_OPERATOR_GUIDANCE_MODES:
        raise ValueError(
            "invalid operator_guidance_mode: "
            f"{candidate!r}; expected one of {VALID_OPERATOR_GUIDANCE_MODES}"
        )
    return candidate


def normalize_guidance_degraded_reasons(reasons) -> list[str]:
    """guidance degraded reasons を固定値リストへ正規化する。"""
    if reasons is None or reasons == "":
        return []
    if isinstance(reasons, str):
        candidates = [reasons]
    else:
        candidates = list(reasons)
    normalized: list[str] = []
    for item in candidates:
        candidate = str(item or "").strip()
        if not candidate:
            continue
        if candidate not in VALID_GUIDANCE_DEGRADED_REASONS:
            raise ValueError(
                "invalid guidance_degraded_reason: "
                f"{candidate!r}; expected one of {VALID_GUIDANCE_DEGRADED_REASONS}"
            )
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def normalize_repro_rollout_mode(
    rollout_mode: str | None,
    *,
    default: str = REPRO_ROLLOUT_MODE_ENFORCED,
) -> str:
    """repro rollout mode を固定値へ正規化する。"""
    normalized_default = (default or "").strip().lower()
    if normalized_default not in VALID_REPRO_ROLLOUT_MODES:
        raise ValueError(
            "invalid default repro_rollout_mode: "
            f"{default!r}; expected one of {VALID_REPRO_ROLLOUT_MODES}"
        )
    candidate = (rollout_mode or "").strip().lower() or normalized_default
    if candidate not in VALID_REPRO_ROLLOUT_MODES:
        raise ValueError(
            "invalid repro_rollout_mode: "
            f"{candidate!r}; expected one of {VALID_REPRO_ROLLOUT_MODES}"
        )
    return candidate


def get_repro_rollout_mode() -> str:
    """現在の rollout mode を環境変数から返す。既定は enforced。"""
    return normalize_repro_rollout_mode(
        os.environ.get("PICOLOR_REPRO_POLICY_MODE"),
        default=REPRO_ROLLOUT_MODE_ENFORCED,
    )


def apply_measurement_fail_stop_rollout(
    gate_result: dict | None,
    rollout_mode: str | None = None,
) -> dict:
    """fail-stop evidence を rollout mode に応じた recording decision へ変換する。"""
    normalized_mode = normalize_repro_rollout_mode(rollout_mode)
    gate = dict(gate_result) if isinstance(gate_result, dict) else {}
    reason = str(gate.get("reason", "")).strip()
    operator_detail = str(gate.get("operator_detail", "")).strip()
    operator_hint = str(gate.get("operator_hint", "")).strip()
    evidence_allowed = bool(gate.get("allowed"))

    if evidence_allowed:
        return {
            "rollout_mode": normalized_mode,
            "recording_allowed": True,
            "blocking_effective": False,
            "overlay_lines": [],
            "reason": reason,
            "operator_detail": operator_detail,
            "operator_hint": operator_hint,
        }

    if normalized_mode == REPRO_ROLLOUT_MODE_SHADOW:
        overlay_lines = ["shadow: fail-stop 条件を記録します"]
        if operator_detail:
            overlay_lines.append(operator_detail)
        overlay_lines.append("本番測定は継続します")
        return {
            "rollout_mode": normalized_mode,
            "recording_allowed": True,
            "blocking_effective": False,
            "overlay_lines": overlay_lines,
            "reason": reason,
            "operator_detail": operator_detail,
            "operator_hint": "",
        }

    if normalized_mode == REPRO_ROLLOUT_MODE_ADVISORY:
        overlay_lines = ["advisory: fail-stop 条件です"]
        if operator_detail:
            overlay_lines.append(operator_detail)
        if operator_hint:
            overlay_lines.append(operator_hint)
        overlay_lines.append("本番測定は継続します")
        return {
            "rollout_mode": normalized_mode,
            "recording_allowed": True,
            "blocking_effective": False,
            "overlay_lines": overlay_lines,
            "reason": reason,
            "operator_detail": operator_detail,
            "operator_hint": operator_hint,
        }

    overlay_lines = ["fail-stop: 本番測定を開始できません"]
    if operator_detail:
        overlay_lines.append(operator_detail)
    if operator_hint:
        overlay_lines.append(operator_hint)
    return {
        "rollout_mode": normalized_mode,
        "recording_allowed": False,
        "blocking_effective": True,
        "overlay_lines": overlay_lines,
        "reason": reason,
        "operator_detail": operator_detail,
        "operator_hint": operator_hint,
    }


@dataclass(frozen=True)
class SteadyStateSnapshot:
    """steady state 判定で使う最小観測スナップショット。"""

    timestamp: str
    ref_scale: float | None
    lab_rel_l: float | None
    relative_rgb_mean: tuple[float, float, float] | None = None
    relative_rgb_std: tuple[float, float, float] | None = None
    consecutive_frames: int = 0
    is_stable: bool = False

    def as_dict(self) -> dict:
        """strategy.md で固定した項目順の dict へ変換する。"""
        return {
            "timestamp": self.timestamp,
            "ref_scale": self.ref_scale,
            "lab_rel_l": self.lab_rel_l,
            "relative_rgb_mean": self.relative_rgb_mean,
            "relative_rgb_std": self.relative_rgb_std,
            "consecutive_frames": self.consecutive_frames,
            "is_stable": self.is_stable,
        }


@dataclass(frozen=True)
class AcceptedReferenceSelectionPolicy:
    """直前の正式合格 run を選ぶための固定ポリシー。"""

    accepted_run_types: tuple[str, ...] = (
        *FORMAL_ACCEPTANCE_RUN_TYPES,
    )
    accepted_result_statuses: tuple[str, str] = (
        RESULT_STATUS_ACCEPTED,
        RESULT_STATUS_MINIMUM,
    )
    allowed_gate_states: tuple[str, ...] = (
        GATE_STATE_OPEN,
        GATE_STATE_PROVISIONAL,
    )
    require_same_control_sample: bool = True
    prefer_latest_timestamp: bool = True


DEFAULT_ACCEPTED_REFERENCE_SELECTION_POLICY = AcceptedReferenceSelectionPolicy()


@dataclass(frozen=True)
class MeasurementProductionGatePolicy:
    """本番測定の unlock に使う厳格ポリシー。"""

    allowed_run_types: tuple[str, ...] = (
        RUN_TYPE_START_OF_DAY,
        RUN_TYPE_REQUALIFICATION,
    )
    accepted_result_statuses: tuple[str, str] = (
        RESULT_STATUS_ACCEPTED,
        RESULT_STATUS_MINIMUM,
    )
    required_gate_state: str = GATE_STATE_OPEN


DEFAULT_MEASUREMENT_PRODUCTION_GATE_POLICY = MeasurementProductionGatePolicy()


def get_acceptance_result_json_schema_contract() -> dict:
    """acceptance_result.json の最小 schema 契約を返す。"""
    return {
        "required_fields": ACCEPTANCE_RESULT_JSON_REQUIRED_FIELDS,
        "required_id_fields": ACCEPTANCE_RESULT_JSON_REQUIRED_ID_FIELDS,
        "alternative_field_groups": ACCEPTANCE_RESULT_JSON_ALTERNATIVE_FIELD_GROUPS,
    }


def get_acceptance_result_log_contract() -> dict:
    """acceptance_result.log の最小人間可読項目契約を返す。"""
    return {
        "required_items": ACCEPTANCE_RESULT_LOG_REQUIRED_ITEMS,
        "required_id_items": ACCEPTANCE_RESULT_LOG_REQUIRED_ID_ITEMS,
        "alternative_item_groups": ACCEPTANCE_RESULT_LOG_ALTERNATIVE_ITEM_GROUPS,
    }


def get_acceptance_result_operational_contract() -> dict:
    """production-evidence 用の operational contract を返す。"""
    return {
        "required_fields": ACCEPTANCE_RESULT_OPERATIONAL_REQUIRED_FIELDS,
        "alternative_field_groups": ACCEPTANCE_RESULT_OPERATIONAL_ALTERNATIVE_FIELD_GROUPS,
        "selected_reference_fields": ACCEPTANCE_RESULT_OPERATIONAL_SELECTED_REFERENCE_FIELDS,
        "known_identity_required": True,
    }


def has_known_software_identity(record: dict | None) -> bool:
    """software_version / git_revision のいずれかが既知なら True を返す。"""
    if not isinstance(record, dict):
        return False
    for field_name in SOFTWARE_IDENTITY_FIELDS:
        value = str(record.get(field_name, "")).strip()
        if value and value.lower() != "unknown":
            return True
    return False


def _normalize_measurement_context_value(value) -> str:
    """measurement context 用の文字列値を正規化する。"""
    text = str(value or "").strip()
    if text.lower() == "unknown":
        return ""
    return text


def build_measurement_context_summary(
    *,
    control_sample_id: str | None = None,
    fixture_id: str | None = None,
    positioning_method: str | None = None,
    context_profile_id: str | None = None,
    software_version: str | None = None,
    git_revision: str | None = None,
) -> dict:
    """比較用 measurement context summary を正規化して返す。"""
    summary = {
        "control_sample_id": _normalize_measurement_context_value(
            control_sample_id or DEFAULT_CONTROL_SAMPLE_ID
        ),
        "fixture_id": _normalize_measurement_context_value(fixture_id),
        "positioning_method": _normalize_measurement_context_value(positioning_method),
        "context_profile_id": _normalize_measurement_context_value(context_profile_id),
        "software_version": _normalize_measurement_context_value(software_version),
        "git_revision": _normalize_measurement_context_value(git_revision),
    }
    if not summary["control_sample_id"]:
        summary["control_sample_id"] = DEFAULT_CONTROL_SAMPLE_ID
    return summary


def extract_measurement_context_summary(record: dict | None) -> dict:
    """record から measurement context summary を backfill して返す。"""
    if not isinstance(record, dict):
        return build_measurement_context_summary()
    persisted = record.get("measurement_context")
    context_payload = persisted if isinstance(persisted, dict) else {}
    return build_measurement_context_summary(
        control_sample_id=context_payload.get(
            "control_sample_id",
            record.get("control_sample_id", DEFAULT_CONTROL_SAMPLE_ID),
        ),
        fixture_id=context_payload.get("fixture_id", record.get("fixture_id", "")),
        positioning_method=context_payload.get(
            "positioning_method",
            record.get("positioning_method", ""),
        ),
        context_profile_id=context_payload.get(
            "context_profile_id",
            record.get("context_profile_id", ""),
        ),
        software_version=context_payload.get(
            "software_version",
            record.get("software_version", ""),
        ),
        git_revision=context_payload.get("git_revision", record.get("git_revision", "")),
    )


def compute_measurement_context_fingerprint(summary: dict | None) -> str:
    """resolved measurement context summary から fingerprint を返す。"""
    resolution = describe_measurement_context_resolution(summary)
    normalized = resolution["summary"]
    if not resolution["is_resolved"]:
        return ""
    canonical = json.dumps(
        {field_name: normalized.get(field_name, "") for field_name in MEASUREMENT_CONTEXT_FIELDS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def attach_measurement_context_metadata(record: dict | None) -> dict:
    """record に measurement context summary / fingerprint を付与して返す。"""
    record_copy = dict(record) if isinstance(record, dict) else {}
    summary = extract_measurement_context_summary(record_copy)
    fingerprint = compute_measurement_context_fingerprint(summary)
    record_copy["context_profile_id"] = summary["context_profile_id"]
    record_copy["measurement_context"] = summary
    record_copy["measurement_context_fingerprint"] = fingerprint
    return record_copy


def describe_measurement_context_resolution(summary: dict | None) -> dict:
    """measurement context の解決状態と不足契約項目を返す。"""
    normalized = extract_measurement_context_summary(summary)
    missing_fields = [
        field_name
        for field_name in MEASUREMENT_CONTEXT_REQUIRED_FIELDS
        if not normalized.get(field_name)
    ]
    missing_alternative_field_groups = [
        "/".join(field_group)
        for field_group in MEASUREMENT_CONTEXT_ALTERNATIVE_FIELD_GROUPS
        if not any(normalized.get(field_name) for field_name in field_group)
    ]
    missing_contract_fields = sorted(
        set(missing_fields + missing_alternative_field_groups)
    )
    return {
        "summary": normalized,
        "is_resolved": not missing_contract_fields,
        "missing_fields": missing_fields,
        "missing_alternative_field_groups": missing_alternative_field_groups,
        "missing_contract_fields": missing_contract_fields,
    }


def compare_measurement_context(
    current_context: dict | None,
    reference_context: dict | None,
) -> dict:
    """current / reference の measurement context を比較して返す。"""
    if current_context is None:
        return {
            "status": "not_evaluated",
            "warning": "",
            "match": None,
            "mismatch_fields": [],
            "current_summary": {},
            "reference_summary": {},
            "current_fingerprint": "",
            "reference_fingerprint": "",
            "current_missing_contract_fields": [],
            "reference_missing_contract_fields": [],
        }

    current_resolution = describe_measurement_context_resolution(current_context)
    reference_resolution = describe_measurement_context_resolution(reference_context)
    current_summary = current_resolution["summary"]
    reference_summary = reference_resolution["summary"]
    current_fingerprint = (
        compute_measurement_context_fingerprint(current_summary)
        if current_resolution["is_resolved"]
        else ""
    )
    reference_fingerprint = (
        compute_measurement_context_fingerprint(reference_summary)
        if reference_resolution["is_resolved"]
        else ""
    )
    if not current_fingerprint:
        return {
            "status": "current_context_unresolved",
            "warning": "current_context_unresolved",
            "match": False,
            "mismatch_fields": [],
            "current_summary": current_summary,
            "reference_summary": reference_summary,
            "current_fingerprint": current_fingerprint,
            "reference_fingerprint": reference_fingerprint,
            "current_missing_contract_fields": current_resolution[
                "missing_contract_fields"
            ],
            "reference_missing_contract_fields": [],
        }
    if not reference_fingerprint:
        return {
            "status": "reference_context_unresolved",
            "warning": "reference_context_unresolved",
            "match": False,
            "mismatch_fields": [],
            "current_summary": current_summary,
            "reference_summary": reference_summary,
            "current_fingerprint": current_fingerprint,
            "reference_fingerprint": reference_fingerprint,
            "current_missing_contract_fields": [],
            "reference_missing_contract_fields": reference_resolution[
                "missing_contract_fields"
            ],
        }
    if current_fingerprint == reference_fingerprint:
        return {
            "status": "match",
            "warning": "",
            "match": True,
            "mismatch_fields": [],
            "current_summary": current_summary,
            "reference_summary": reference_summary,
            "current_fingerprint": current_fingerprint,
            "reference_fingerprint": reference_fingerprint,
            "current_missing_contract_fields": [],
            "reference_missing_contract_fields": [],
        }
    mismatch_fields = [
        field_name
        for field_name in MEASUREMENT_CONTEXT_FIELDS
        if current_summary.get(field_name, "") != reference_summary.get(field_name, "")
    ]
    return {
        "status": "mismatch",
        "warning": "context_mismatch",
        "match": False,
        "mismatch_fields": mismatch_fields,
        "current_summary": current_summary,
        "reference_summary": reference_summary,
        "current_fingerprint": current_fingerprint,
        "reference_fingerprint": reference_fingerprint,
        "current_missing_contract_fields": [],
        "reference_missing_contract_fields": [],
    }


def get_accepted_reference_selection_policy() -> AcceptedReferenceSelectionPolicy:
    """直前の正式合格 run を選ぶ固定ポリシーを返す。"""
    return DEFAULT_ACCEPTED_REFERENCE_SELECTION_POLICY


def get_measurement_production_gate_policy() -> MeasurementProductionGatePolicy:
    """本番測定 unlock に使う厳格ポリシーを返す。"""
    return DEFAULT_MEASUREMENT_PRODUCTION_GATE_POLICY


def _fixed_anchor_selection_basename(control_sample_id: str | None = None) -> str:
    """control_sample_id ごとの fixed anchor selection filename を返す。"""
    candidate = str(control_sample_id or "").strip()
    if not candidate:
        return FIXED_ANCHOR_SELECTION_FILENAME
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", candidate).strip("._")
    if not slug:
        slug = "default"
    return f"fixed_anchor_selection_{slug}.json"


def get_fixed_anchor_selection_path(
    calibration_root: str | None = None,
    *,
    control_sample_id: str | None = None,
) -> str:
    """固定 anchor 選択状態の保存パスを返す。"""
    root = calibration_root or CALIBRATION_DIR
    return os.path.join(root, _fixed_anchor_selection_basename(control_sample_id))


def _with_acceptance_source_metadata(
    record: dict,
    *,
    source_file: str,
    calibration_root: str,
    record_dt,
) -> dict:
    """acceptance_result record に source file metadata を付与して返す。"""
    record_copy = attach_measurement_context_metadata(record)
    record_copy.setdefault("run_id", build_acceptance_run_id(record_copy))
    record_copy["source_file"] = source_file
    try:
        record_copy["source_file_rel"] = os.path.relpath(source_file, calibration_root)
    except ValueError:
        record_copy["source_file_rel"] = source_file
    record_copy["_timestamp_sort_key"] = (
        record_dt.isoformat()
        if record_dt is not None
        else str(record_copy.get("timestamp", ""))
    )
    return record_copy


def _default_accepted_reference_context_meta() -> dict:
    """accepted reference context 診断の既定メタデータを返す。"""
    return {
        "accepted_reference_context_status": "not_evaluated",
        "accepted_reference_context_warning": "",
        "accepted_reference_context_mismatch_fields": [],
        "accepted_reference_current_context_missing_contract_fields": [],
        "accepted_reference_reference_context_missing_contract_fields": [],
        "accepted_reference_context_fingerprint": "",
        "selected_reference_measurement_context": {},
        "selected_reference_measurement_context_fingerprint": "",
    }


def _accepted_reference_context_meta_from_comparison(comparison: dict) -> dict:
    """context comparison 結果を accepted reference 用 meta へ整形する。"""
    selected_reference_summary = extract_measurement_context_summary(
        comparison.get("reference_summary")
    )
    selected_reference_fingerprint = str(
        comparison.get("reference_fingerprint", "")
    ).strip() or (
        compute_measurement_context_fingerprint(selected_reference_summary)
        if selected_reference_summary
        else ""
    )
    return {
        "accepted_reference_context_status": comparison.get("status", "not_evaluated"),
        "accepted_reference_context_warning": comparison.get("warning", ""),
        "accepted_reference_context_mismatch_fields": list(
            comparison.get("mismatch_fields", [])
        ),
        "accepted_reference_current_context_missing_contract_fields": list(
            comparison.get("current_missing_contract_fields", [])
        ),
        "accepted_reference_reference_context_missing_contract_fields": list(
            comparison.get("reference_missing_contract_fields", [])
        ),
        "accepted_reference_context_fingerprint": comparison.get(
            "reference_fingerprint",
            "",
        ),
        "selected_reference_measurement_context": selected_reference_summary,
        "selected_reference_measurement_context_fingerprint": (
            selected_reference_fingerprint
        ),
    }


def _default_previous_accepted_day_context_meta() -> dict:
    """previous accepted-day context 診断の既定メタデータを返す。"""
    return {
        "previous_accepted_day_context_status": "not_evaluated",
        "previous_accepted_day_context_warning": "",
        "previous_accepted_day_context_mismatch_fields": [],
        "previous_accepted_day_current_context_missing_contract_fields": [],
        "previous_accepted_day_reference_context_missing_contract_fields": [],
        "previous_accepted_day_context_fingerprint": "",
        "selected_previous_accepted_day_measurement_context": {},
        "selected_previous_accepted_day_measurement_context_fingerprint": "",
    }


def _previous_accepted_day_context_meta_from_comparison(comparison: dict) -> dict:
    """context comparison 結果を previous accepted-day 用 meta へ整形する。"""
    selected_reference_summary = extract_measurement_context_summary(
        comparison.get("reference_summary")
    )
    selected_reference_fingerprint = str(
        comparison.get("reference_fingerprint", "")
    ).strip() or (
        compute_measurement_context_fingerprint(selected_reference_summary)
        if selected_reference_summary
        else ""
    )
    return {
        "previous_accepted_day_context_status": comparison.get(
            "status", "not_evaluated"
        ),
        "previous_accepted_day_context_warning": comparison.get("warning", ""),
        "previous_accepted_day_context_mismatch_fields": list(
            comparison.get("mismatch_fields", [])
        ),
        "previous_accepted_day_current_context_missing_contract_fields": list(
            comparison.get("current_missing_contract_fields", [])
        ),
        "previous_accepted_day_reference_context_missing_contract_fields": list(
            comparison.get("reference_missing_contract_fields", [])
        ),
        "previous_accepted_day_context_fingerprint": comparison.get(
            "reference_fingerprint",
            "",
        ),
        "selected_previous_accepted_day_measurement_context": selected_reference_summary,
        "selected_previous_accepted_day_measurement_context_fingerprint": (
            selected_reference_fingerprint
        ),
    }


def _collect_accepted_reference_candidates(
    *,
    calibration_root: str,
    control_sample_id: str,
    before_timestamp: str | None = None,
    allowed_gate_states: tuple[str, ...] | None = None,
    require_open_gate: bool = False,
    require_avg_ref: bool = False,
) -> list[dict]:
    """accepted reference 候補 record を source path 付きで列挙する。"""
    if not os.path.isdir(calibration_root):
        return []

    import glob

    policy = get_accepted_reference_selection_policy()
    before_dt = _parse_acceptance_timestamp(before_timestamp)
    gate_filter = (
        (GATE_STATE_OPEN,)
        if require_open_gate
        else (
            allowed_gate_states
            if allowed_gate_states is not None
            else policy.allowed_gate_states
        )
    )
    pattern = os.path.join(calibration_root, "**", "acceptance_result*.json")
    candidates = []
    for path in sorted(set(glob.glob(pattern, recursive=True)), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        try:
            run_type = normalize_acceptance_run_type(record.get("run_type"))
            result_status = normalize_result_status(record.get("result_status"))
            gate_state = normalize_gate_state(record.get("gate_state"))
        except ValueError:
            continue
        if run_type not in policy.accepted_run_types:
            continue
        if result_status not in policy.accepted_result_statuses:
            continue
        if gate_state not in gate_filter:
            continue
        if (
            policy.require_same_control_sample
            and str(record.get("control_sample_id", "")).strip() != control_sample_id
        ):
            continue
        if require_avg_ref and _acceptance_triplet_or_none(record.get("avg_ref")) is None:
            continue
        record_dt = _parse_acceptance_timestamp(record.get("timestamp"))
        if before_dt is not None and record_dt is not None and record_dt >= before_dt:
            continue
        candidates.append(
            _with_acceptance_source_metadata(
                record,
                source_file=path,
                calibration_root=calibration_root,
                record_dt=record_dt,
            )
        )
    candidates.sort(
        key=lambda record: record.get("_timestamp_sort_key", ""),
        reverse=policy.prefer_latest_timestamp,
    )
    return candidates


def list_fixed_anchor_candidates(
    *,
    calibration_root: str | None = None,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
) -> list[dict]:
    """GUI 固定 anchor picker 向けに open accepted reference 候補を返す。"""
    root = calibration_root or CALIBRATION_DIR
    candidates = _collect_accepted_reference_candidates(
        calibration_root=root,
        control_sample_id=control_sample_id,
        require_open_gate=True,
        require_avg_ref=True,
    )
    cleaned = []
    for record in candidates:
        record_copy = dict(record)
        record_copy.pop("_timestamp_sort_key", None)
        cleaned.append(record_copy)
    return cleaned


def load_fixed_anchor_selection(
    *,
    calibration_root: str | None = None,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
) -> dict:
    """固定 anchor 選択状態を返す。未設定/不正時は auto_latest を返す。"""
    normalized_control_sample_id = str(control_sample_id or "").strip()
    default = {
        "accepted_reference_selection_mode": ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
        "accepted_reference_selection_requested_run_id": "",
        "accepted_reference_selection_source_file": "",
        "control_sample_id": normalized_control_sample_id,
        "selected_measurement_context": {},
        "selected_measurement_context_fingerprint": "",
    }
    candidate_paths = [
        get_fixed_anchor_selection_path(
            calibration_root,
            control_sample_id=normalized_control_sample_id,
        ),
    ]
    legacy_path = get_fixed_anchor_selection_path(calibration_root)
    if legacy_path not in candidate_paths:
        candidate_paths.append(legacy_path)
    payload = None
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = None
        payload_control_sample_id = ""
        if isinstance(payload, dict):
            payload_control_sample_id = str(
                payload.get("control_sample_id", "")
            ).strip()
        if (
            isinstance(payload, dict)
            and payload_control_sample_id == normalized_control_sample_id
        ):
            break
        payload = None
    if not isinstance(payload, dict):
        return dict(default)
    mode = str(
        payload.get(
            "accepted_reference_selection_mode",
            ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
        )
    ).strip()
    if mode not in (
        ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
        ACCEPTED_REFERENCE_SELECTION_MODE_FIXED,
    ):
        return dict(default)
    return {
        "accepted_reference_selection_mode": mode,
        "accepted_reference_selection_requested_run_id": str(
            payload.get("accepted_reference_selection_requested_run_id", "")
        ).strip(),
        "accepted_reference_selection_source_file": str(
            payload.get("accepted_reference_selection_source_file", "")
        ).strip(),
        "control_sample_id": normalized_control_sample_id,
        "selected_measurement_context": extract_measurement_context_summary(
            payload.get("selected_measurement_context", {})
        ) if payload.get("selected_measurement_context") else {},
        "selected_measurement_context_fingerprint": str(
            payload.get("selected_measurement_context_fingerprint", "")
        ).strip(),
    }


def save_fixed_anchor_selection(
    *,
    run_id: str,
    source_file: str,
    calibration_root: str | None = None,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
    selected_measurement_context: dict | None = None,
    selected_measurement_context_fingerprint: str | None = None,
) -> dict:
    """固定 anchor 選択状態を保存して返す。"""
    root = calibration_root or CALIBRATION_DIR
    os.makedirs(root, exist_ok=True)
    context_summary = (
        extract_measurement_context_summary(selected_measurement_context)
        if selected_measurement_context
        else {}
    )
    payload = {
        "accepted_reference_selection_mode": ACCEPTED_REFERENCE_SELECTION_MODE_FIXED,
        "accepted_reference_selection_requested_run_id": str(run_id).strip(),
        "accepted_reference_selection_source_file": str(source_file).strip(),
        "control_sample_id": control_sample_id,
        "selected_measurement_context": context_summary,
        "selected_measurement_context_fingerprint": (
            str(selected_measurement_context_fingerprint or "").strip()
            or (
                compute_measurement_context_fingerprint(context_summary)
                if context_summary
                else ""
            )
        ),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(
        get_fixed_anchor_selection_path(root, control_sample_id=control_sample_id),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def clear_fixed_anchor_selection(
    calibration_root: str | None = None,
    *,
    control_sample_id: str | None = None,
) -> None:
    """固定 anchor 選択状態を解除する。"""
    normalized_control_sample_id = str(control_sample_id or "").strip()
    path = get_fixed_anchor_selection_path(
        calibration_root,
        control_sample_id=normalized_control_sample_id,
    )
    legacy_path = get_fixed_anchor_selection_path(calibration_root)
    candidate_paths = [path]
    if legacy_path not in candidate_paths:
        candidate_paths.append(legacy_path)

    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            continue
        if candidate_path == legacy_path and normalized_control_sample_id:
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                payload = None
            payload_control_sample_id = ""
            if isinstance(payload, dict):
                payload_control_sample_id = str(
                    payload.get("control_sample_id", "")
                ).strip()
            if payload_control_sample_id != normalized_control_sample_id:
                continue
        os.remove(candidate_path)


def resolve_accepted_reference_record(
    *,
    calibration_root: str | None = None,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
    before_timestamp: str | None = None,
    accepted_reference_selection_mode: str = ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
    accepted_reference_selection_requested_run_id: str | None = None,
    accepted_reference_selection_source_file: str | None = None,
    require_open_gate: bool = False,
    current_measurement_context: dict | None = None,
) -> tuple["dict | None", dict]:
    """manual fixed anchor を優先しつつ accepted reference record を解決する。"""
    root = calibration_root or CALIBRATION_DIR
    requested_run_id = str(
        accepted_reference_selection_requested_run_id or ""
    ).strip()
    source_file = str(accepted_reference_selection_source_file or "").strip()
    selection_mode = str(
        accepted_reference_selection_mode or ACCEPTED_REFERENCE_SELECTION_MODE_AUTO
    ).strip()
    if selection_mode not in (
        ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
        ACCEPTED_REFERENCE_SELECTION_MODE_FIXED,
    ):
        selection_mode = ACCEPTED_REFERENCE_SELECTION_MODE_AUTO
    meta = {
        "accepted_reference_selection_mode": selection_mode,
        "accepted_reference_selection_requested_run_id": requested_run_id,
        "accepted_reference_selection_source_file": source_file,
        **_default_accepted_reference_context_meta(),
    }
    current_context_summary = extract_measurement_context_summary(
        current_measurement_context
    )
    if selection_mode == ACCEPTED_REFERENCE_SELECTION_MODE_FIXED and requested_run_id:
        record = find_acceptance_record_by_run_id(
            calibration_root=root,
            run_id=requested_run_id,
            control_sample_id=control_sample_id,
        )
        if record is not None:
            record_dt = _parse_acceptance_timestamp(record.get("timestamp"))
            before_dt = _parse_acceptance_timestamp(before_timestamp)
            try:
                gate_state = normalize_gate_state(record.get("gate_state"))
            except ValueError:
                gate_state = ""
            if (
                before_timestamp
                and (before_dt is None or record_dt is None)
            ):
                meta["accepted_reference_selection_warning"] = (
                    "fixed_anchor_timestamp_invalid"
                )
            elif (
                before_dt is not None
                and record_dt is not None
                and record_dt >= before_dt
            ):
                meta["accepted_reference_selection_warning"] = (
                    "fixed_anchor_not_before_current_run"
                )
            elif require_open_gate and gate_state != GATE_STATE_OPEN:
                meta["accepted_reference_selection_warning"] = (
                    "fixed_anchor_not_open_for_preflight"
                )
            else:
                comparison = compare_measurement_context(current_context_summary, record)
                if comparison.get("status") == "not_evaluated":
                    comparison = {
                        **comparison,
                        "status": "current_context_unresolved",
                        "warning": "current_context_unresolved",
                        "match": False,
                    }
                meta.update(_accepted_reference_context_meta_from_comparison(comparison))
                if comparison.get("status") in (
                    "not_evaluated",
                    "current_context_unresolved",
                ):
                    meta["accepted_reference_selection_warning"] = (
                        "fixed_anchor_current_context_unresolved"
                    )
                    return None, meta
                if comparison.get("status") == "reference_context_unresolved":
                    meta["accepted_reference_selection_warning"] = (
                        "fixed_anchor_reference_context_unresolved"
                    )
                    return None, meta
                if comparison.get("status") == "mismatch":
                    meta["accepted_reference_selection_warning"] = (
                        "fixed_anchor_context_mismatch"
                    )
                    return None, meta
                if not meta["accepted_reference_selection_source_file"]:
                    meta["accepted_reference_selection_source_file"] = str(
                        record.get("source_file_rel", record.get("source_file", ""))
                    )
                return record, meta
        else:
            meta["accepted_reference_selection_warning"] = "fixed_anchor_missing"
        return None, meta

    candidates = _collect_accepted_reference_candidates(
        calibration_root=root,
        control_sample_id=control_sample_id,
        before_timestamp=before_timestamp,
        require_open_gate=require_open_gate,
        require_avg_ref=require_open_gate,
    )
    first_context_issue = None
    for candidate in candidates:
        comparison = compare_measurement_context(current_context_summary, candidate)
        if comparison.get("status") == "not_evaluated":
            comparison = {
                **comparison,
                "status": "current_context_unresolved",
                "warning": "current_context_unresolved",
                "match": False,
            }
        if comparison.get("status") == "match":
            record = dict(candidate)
            meta.update(_accepted_reference_context_meta_from_comparison(comparison))
            if not meta["accepted_reference_selection_source_file"]:
                meta["accepted_reference_selection_source_file"] = str(
                    record.get("source_file_rel", record.get("source_file", ""))
                )
            return record, meta
        if first_context_issue is None:
            first_context_issue = comparison

    if first_context_issue is not None:
        meta.update(_accepted_reference_context_meta_from_comparison(first_context_issue))
        warning = first_context_issue.get("warning")
        if warning == "current_context_unresolved":
            meta["accepted_reference_selection_warning"] = (
                "auto_reference_current_context_unresolved"
            )
        elif warning == "reference_context_unresolved":
            meta["accepted_reference_selection_warning"] = (
                "auto_reference_reference_context_unresolved"
            )
        elif warning == "context_mismatch":
            meta["accepted_reference_selection_warning"] = (
                "auto_reference_context_mismatch"
            )
    return None, meta


def resolve_previous_accepted_day_record(
    *,
    calibration_root: str | None = None,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
    before_timestamp: str | None = None,
    current_measurement_context: dict | None = None,
) -> tuple["dict | None", dict]:
    """同一 context における前回合格日の formal run を解決する。"""
    root = calibration_root or CALIBRATION_DIR
    meta = {
        "previous_accepted_day_selection_warning": "",
        **_default_previous_accepted_day_context_meta(),
    }
    current_context_summary = extract_measurement_context_summary(
        current_measurement_context
    )
    current_resolution = describe_measurement_context_resolution(
        current_context_summary
    )
    if not current_resolution["is_resolved"]:
        meta.update(
            {
                "previous_accepted_day_selection_warning": (
                    "previous_accepted_day_current_context_unresolved"
                ),
                "previous_accepted_day_context_status": "current_context_unresolved",
                "previous_accepted_day_context_warning": "current_context_unresolved",
                "previous_accepted_day_current_context_missing_contract_fields": list(
                    current_resolution["missing_contract_fields"]
                ),
            }
        )
        return None, meta

    current_date = _acceptance_date_slug(before_timestamp)
    if not current_date:
        meta["previous_accepted_day_selection_warning"] = (
            "previous_accepted_day_timestamp_invalid"
        )
        return None, meta

    candidates = _collect_accepted_reference_candidates(
        calibration_root=root,
        control_sample_id=control_sample_id,
        before_timestamp=before_timestamp,
    )
    first_context_issue = None
    for candidate in candidates:
        candidate_date = _acceptance_date_slug(candidate.get("timestamp"))
        if not candidate_date or candidate_date == current_date:
            continue
        comparison = compare_measurement_context(current_context_summary, candidate)
        if comparison.get("status") == "match":
            record = dict(candidate)
            record.pop("_timestamp_sort_key", None)
            meta.update(
                _previous_accepted_day_context_meta_from_comparison(comparison)
            )
            return record, meta
        if first_context_issue is None:
            first_context_issue = comparison

    if first_context_issue is not None:
        meta.update(
            _previous_accepted_day_context_meta_from_comparison(first_context_issue)
        )
        warning_map = {
            "current_context_unresolved": (
                "previous_accepted_day_current_context_unresolved"
            ),
            "reference_context_unresolved": (
                "previous_accepted_day_reference_context_unresolved"
            ),
            "context_mismatch": "previous_accepted_day_context_mismatch",
        }
        meta["previous_accepted_day_selection_warning"] = warning_map.get(
            first_context_issue.get("warning"),
            "",
        )
        return None, meta

    meta["previous_accepted_day_selection_warning"] = "previous_accepted_day_missing"
    return None, meta


def _acceptance_triplet_or_none(value) -> "np.ndarray | None":
    """受入れ判定用に3要素ベクトルを正規化して返す。"""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size < 3:
        return None
    triplet = arr[:3].copy()
    if not np.all(np.isfinite(triplet)):
        return None
    return triplet


def _acceptance_scalar_or_none(value) -> "float | None":
    """受入れ判定用に有限スカラーを正規化して返す。"""
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def summarize_gray_steady_state_observables(
    relative_rgb_samples,
    ref_scales=None,
    lab_rel_l_values=None,
) -> dict:
    """steady state 判定に使う観測量を 1 箇所で集約して返す。"""
    samples = np.asarray(relative_rgb_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] < 3:
        return {
            "relative_rgb_mean": None,
            "relative_rgb_std": None,
            "max_sigma": None,
            "ref_scale_mean": None,
            "ref_scale_range": None,
            "lab_rel_l_mean": None,
        }

    mean_rgb = samples.mean(axis=0)[:3]
    std_rgb = (
        samples.std(axis=0, ddof=1)[:3]
        if samples.shape[0] > 1
        else np.zeros(3, dtype=np.float64)
    )
    ref_scale_values = [] if ref_scales is None else ref_scales
    lab_rel_l_source = [] if lab_rel_l_values is None else lab_rel_l_values
    valid_scales = [
        float(value) for value in ref_scale_values
        if _acceptance_scalar_or_none(value) is not None
    ]
    valid_lab_rel_l = [
        float(value) for value in lab_rel_l_source
        if _acceptance_scalar_or_none(value) is not None
    ]
    return {
        "relative_rgb_mean": mean_rgb.tolist(),
        "relative_rgb_std": std_rgb.tolist(),
        "max_sigma": float(np.max(std_rgb)),
        "ref_scale_mean": (
            float(np.mean(np.asarray(valid_scales, dtype=np.float64)))
            if valid_scales
            else None
        ),
        "ref_scale_range": (
            float(max(valid_scales) - min(valid_scales))
            if len(valid_scales) >= 2
            else None
        ),
        "lab_rel_l_mean": (
            float(np.mean(np.asarray(valid_lab_rel_l, dtype=np.float64)))
            if valid_lab_rel_l
            else None
        ),
    }


def compute_accepted_reference_diff(
    current_record: dict | None,
    reference_record: dict | None,
) -> dict:
    """直前の正式合格 run との差分を比較可能な形へ正規化する。"""
    if not isinstance(current_record, dict):
        return {
            "available": False,
            "reason": "current_record_missing",
        }
    if not isinstance(reference_record, dict):
        return {
            "available": False,
            "reason": "accepted_reference_missing",
        }

    current_lab = _acceptance_triplet_or_none(current_record.get("measured_lab"))
    reference_lab = _acceptance_triplet_or_none(reference_record.get("measured_lab"))
    current_mean = _acceptance_triplet_or_none(current_record.get("relative_rgb_mean"))
    reference_mean = _acceptance_triplet_or_none(reference_record.get("relative_rgb_mean"))
    current_scale = _acceptance_scalar_or_none(current_record.get("ref_scale"))
    reference_scale = _acceptance_scalar_or_none(reference_record.get("ref_scale"))

    lab_delta_e = (
        float(np.linalg.norm(current_lab - reference_lab))
        if current_lab is not None and reference_lab is not None
        else None
    )
    relative_rgb_mean_abs_diff = (
        np.abs(current_mean - reference_mean)
        if current_mean is not None and reference_mean is not None
        else None
    )
    relative_rgb_mean_abs_diff_max = (
        float(np.max(relative_rgb_mean_abs_diff))
        if relative_rgb_mean_abs_diff is not None
        else None
    )
    ref_scale_abs_diff = (
        abs(current_scale - reference_scale)
        if current_scale is not None and reference_scale is not None
        else None
    )
    available = (
        lab_delta_e is not None
        and relative_rgb_mean_abs_diff_max is not None
        and ref_scale_abs_diff is not None
    )
    if not available:
        return {
            "available": False,
            "reason": "accepted_reference_metrics_missing",
            "reference_run_id": str(
                reference_record.get("run_id", build_acceptance_run_id(reference_record))
            ),
            "reference_timestamp": reference_record.get("timestamp"),
        }

    target_checks = {
        "accepted_reference_delta_e": (
            lab_delta_e <= ACCEPTANCE_REFERENCE_DIFF_TARGET_MAX_DE
        ),
        "accepted_reference_relative_rgb_mean": (
            relative_rgb_mean_abs_diff_max
            <= ACCEPTANCE_REFERENCE_DIFF_REL_MEAN_TARGET_DELTA
        ),
        "accepted_reference_ref_scale": (
            ref_scale_abs_diff <= ACCEPTANCE_REFERENCE_DIFF_SCALE_TARGET_DELTA
        ),
    }
    minimum_checks = {
        "accepted_reference_delta_e": (
            lab_delta_e < ACCEPTANCE_REFERENCE_DIFF_MIN_MAX_DE
        ),
        "accepted_reference_relative_rgb_mean": (
            relative_rgb_mean_abs_diff_max
            <= ACCEPTANCE_REFERENCE_DIFF_REL_MEAN_MIN_DELTA
        ),
        "accepted_reference_ref_scale": (
            ref_scale_abs_diff <= ACCEPTANCE_REFERENCE_DIFF_SCALE_MIN_DELTA
        ),
    }
    return {
        "available": True,
        "reason": "ok",
        "reference_run_id": str(
            reference_record.get("run_id", build_acceptance_run_id(reference_record))
        ),
        "reference_timestamp": reference_record.get("timestamp"),
        "lab_delta_e": lab_delta_e,
        "relative_rgb_mean_abs_diff": relative_rgb_mean_abs_diff.tolist(),
        "relative_rgb_mean_abs_diff_max": relative_rgb_mean_abs_diff_max,
        "ref_scale_abs_diff": ref_scale_abs_diff,
        "target_checks": target_checks,
        "minimum_checks": minimum_checks,
    }


def compute_previous_accepted_day_diff(
    current_record: dict | None,
    previous_record: dict | None,
) -> dict:
    """前回合格日の formal run との差分を返す。"""
    diff = compute_accepted_reference_diff(current_record, previous_record)
    if not diff.get("available"):
        reason_map = {
            "accepted_reference_missing": "previous_accepted_day_missing",
            "accepted_reference_metrics_missing": (
                "previous_accepted_day_metrics_missing"
            ),
        }
        diff["reason"] = reason_map.get(diff.get("reason"), diff.get("reason"))
    return diff


def evaluate_previous_accepted_day_judgement(
    previous_accepted_day_diff: dict | None,
) -> dict:
    """前回合格日との差分だけに基づく day-over-day judgement を返す。"""
    if not (
        isinstance(previous_accepted_day_diff, dict)
        and previous_accepted_day_diff.get("available")
    ):
        return {
            "available": False,
            "reason": (
                previous_accepted_day_diff.get("reason", "previous_accepted_day_missing")
                if isinstance(previous_accepted_day_diff, dict)
                else "previous_accepted_day_missing"
            ),
            "result_status": RESULT_STATUS_RECHECK_REQUIRED,
            "failed_checks": [],
            "target_checks": {},
            "minimum_checks": {},
        }

    raw_target_checks = dict(previous_accepted_day_diff.get("target_checks", {}))
    raw_minimum_checks = dict(previous_accepted_day_diff.get("minimum_checks", {}))
    target_checks = {
        "delta_e": bool(raw_target_checks.get("accepted_reference_delta_e")),
        "relative_rgb_mean": bool(
            raw_target_checks.get("accepted_reference_relative_rgb_mean")
        ),
        "ref_scale": bool(raw_target_checks.get("accepted_reference_ref_scale")),
    }
    minimum_checks = {
        "delta_e": bool(raw_minimum_checks.get("accepted_reference_delta_e")),
        "relative_rgb_mean": bool(
            raw_minimum_checks.get("accepted_reference_relative_rgb_mean")
        ),
        "ref_scale": bool(raw_minimum_checks.get("accepted_reference_ref_scale")),
    }
    if all(target_checks.values()):
        result_status = RESULT_STATUS_ACCEPTED
        failed_checks = []
    elif all(minimum_checks.values()):
        result_status = RESULT_STATUS_MINIMUM
        failed_checks = [name for name, ok in target_checks.items() if not ok]
    else:
        result_status = RESULT_STATUS_REJECTED
        failed_checks = [name for name, ok in minimum_checks.items() if not ok]
    return {
        "available": True,
        "reason": "ok",
        "result_status": result_status,
        "failed_checks": failed_checks,
        "target_checks": target_checks,
        "minimum_checks": minimum_checks,
    }


def evaluate_acceptance_judgement(
    *,
    relative_rgb_mean,
    relative_rgb_std,
    delta_e,
    ref_scale,
    lab_rel_l,
    baseline_source: str | None,
    accepted_reference_diff: dict | None = None,
    manual_roi_used: bool = False,
    manual_corner_used: bool = False,
) -> dict:
    """受入れ判定の target/minimum/blocked を固定規則で評価する。"""
    contract_errors = []
    baseline = str(baseline_source or "").strip()
    if baseline not in VALID_BASELINE_SOURCES:
        contract_errors.append("baseline_source")

    mean_rgb = _acceptance_triplet_or_none(relative_rgb_mean)
    std_rgb = _acceptance_triplet_or_none(relative_rgb_std)
    if mean_rgb is None:
        contract_errors.append("relative_rgb_mean")
    if std_rgb is None:
        contract_errors.append("relative_rgb_std")

    try:
        d_e = float(delta_e)
        if not math.isfinite(d_e):
            raise ValueError
    except (TypeError, ValueError):
        d_e = None
        contract_errors.append("delta_e")

    try:
        scale = float(ref_scale)
        if not math.isfinite(scale):
            raise ValueError
    except (TypeError, ValueError):
        scale = None
        contract_errors.append("ref_scale")

    if contract_errors:
        return {
            "result_status": RESULT_STATUS_RECHECK_REQUIRED,
            "gate_state": GATE_STATE_BLOCKED,
            "target_checks": {},
            "minimum_checks": {},
            "failed_checks": sorted(set(contract_errors)),
            "relative_max_dev": None,
            "relative_max_sigma": None,
            "accepted_reference_available": False,
            "provisional_reasons": [],
        }

    max_dev = float(np.max(np.abs(mean_rgb - 1.0)))
    max_sigma = float(np.max(std_rgb))
    target_checks = {
        "relative_rgb_mean": max_dev <= ACCEPTANCE_REL_MEAN_TARGET_DELTA,
        "relative_rgb_std": max_sigma <= ACCEPTANCE_REL_STD_TARGET_MAX,
        "delta_e": d_e <= ACCEPTANCE_ABSOLUTE_TARGET_MAX_DE,
        "ref_scale": abs(scale - 1.0) <= ACCEPTANCE_REF_SCALE_TARGET_DELTA,
    }
    minimum_checks = {
        "relative_rgb_mean": max_dev <= ACCEPTANCE_REL_MEAN_MIN_DELTA,
        "relative_rgb_std": max_sigma <= ACCEPTANCE_REL_STD_MIN_MAX,
        "delta_e": d_e < ACCEPTANCE_ABSOLUTE_MIN_MAX_DE,
        "ref_scale": abs(scale - 1.0) <= ACCEPTANCE_REF_SCALE_MIN_DELTA,
    }
    try:
        rel_l = float(lab_rel_l)
        if not math.isfinite(rel_l):
            raise ValueError
    except (TypeError, ValueError):
        rel_l = None
    if rel_l is not None:
        target_checks["lab_rel_l"] = (
            ACCEPTANCE_LAB_REL_L_TARGET_MIN
            <= rel_l
            <= ACCEPTANCE_LAB_REL_L_TARGET_MAX
        )
        minimum_checks["lab_rel_l"] = (
            ACCEPTANCE_LAB_REL_L_MIN_MIN
            <= rel_l
            <= ACCEPTANCE_LAB_REL_L_MIN_MAX
        )
    reference_diff_available = bool(
        isinstance(accepted_reference_diff, dict)
        and accepted_reference_diff.get("available")
    )
    if reference_diff_available:
        target_checks.update(accepted_reference_diff.get("target_checks", {}))
        minimum_checks.update(accepted_reference_diff.get("minimum_checks", {}))

    if all(target_checks.values()):
        result_status = RESULT_STATUS_ACCEPTED
        gate_state = GATE_STATE_OPEN
        failed_checks = []
    elif all(minimum_checks.values()):
        result_status = RESULT_STATUS_MINIMUM
        gate_state = GATE_STATE_PROVISIONAL
        failed_checks = [name for name, ok in target_checks.items() if not ok]
    else:
        result_status = RESULT_STATUS_REJECTED
        gate_state = GATE_STATE_BLOCKED
        failed_checks = [name for name, ok in minimum_checks.items() if not ok]

    provisional_reasons = []
    if manual_roi_used:
        provisional_reasons.append("manual_roi")
    if manual_corner_used:
        provisional_reasons.append("manual_corner")
    if provisional_reasons and gate_state != GATE_STATE_BLOCKED:
        gate_state = GATE_STATE_PROVISIONAL

    return {
        "result_status": result_status,
        "gate_state": gate_state,
        "target_checks": target_checks,
        "minimum_checks": minimum_checks,
        "failed_checks": failed_checks,
        "relative_max_dev": max_dev,
        "relative_max_sigma": max_sigma,
        "accepted_reference_available": reference_diff_available,
        "provisional_reasons": provisional_reasons,
    }


def build_acceptance_run_id(record: dict) -> str:
    """受入れ判定 run を一意に識別する run_id を返す。"""
    run_type = normalize_acceptance_run_type(record.get("run_type"))
    ts = str(record.get("timestamp", datetime.now().isoformat(timespec="seconds")))
    ts_slug = re.sub(r"[^0-9A-Za-z]+", "", ts) or datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{run_type}:{ts_slug}"


def _parse_acceptance_timestamp(value) -> "datetime | None":
    """ISO時刻を datetime へ変換する。失敗時は None。"""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _acceptance_date_slug(value) -> str:
    """acceptance timestamp/date から YYYY-MM-DD を抽出する。"""
    parsed = _parse_acceptance_timestamp(value)
    if parsed is not None:
        return parsed.date().isoformat()
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text[:10]):
        return text[:10]
    return ""


def describe_accepted_reference_relation(
    current_record: dict | None,
    reference_record: dict | None,
) -> dict:
    """accepted_reference との関係を診断用フィールドへ正規化する。"""
    diagnostics = {
        "reference_is_same_day": False,
        "reference_is_same_session": False,
        "reference_relation": REFERENCE_RELATION_NONE,
    }
    if not isinstance(current_record, dict) or not isinstance(reference_record, dict):
        return diagnostics

    current_date = _acceptance_date_slug(current_record.get("timestamp"))
    reference_date = _acceptance_date_slug(reference_record.get("timestamp"))
    same_day = bool(current_date and reference_date and current_date == reference_date)

    current_dt = _parse_acceptance_timestamp(current_record.get("timestamp"))
    reference_dt = _parse_acceptance_timestamp(reference_record.get("timestamp"))
    try:
        current_run_type = normalize_acceptance_run_type(current_record.get("run_type"))
    except ValueError:
        current_run_type = None
    try:
        reference_run_type = normalize_acceptance_run_type(reference_record.get("run_type"))
    except ValueError:
        reference_run_type = None

    same_session = bool(
        same_day
        and current_dt is not None
        and reference_dt is not None
        and reference_dt < current_dt
        and current_run_type == RUN_TYPE_VERIFY_ONLY
        and reference_run_type == RUN_TYPE_VERIFY_ONLY
    )
    diagnostics["reference_is_same_day"] = same_day
    diagnostics["reference_is_same_session"] = same_session
    diagnostics["reference_relation"] = (
        REFERENCE_RELATION_SAME_DAY_PREVIOUS_RUN
        if same_day
        else REFERENCE_RELATION_PREVIOUS_DAY_OR_OLDER
    )
    return diagnostics


def find_latest_accepted_reference_record(
    *,
    calibration_root: str | None = None,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
    before_timestamp: str | None = None,
) -> "dict | None":
    """ポリシーに合う直前の正式合格 run を返す。見つからなければ None。"""
    root = calibration_root or CALIBRATION_DIR
    candidates = _collect_accepted_reference_candidates(
        calibration_root=root,
        control_sample_id=control_sample_id,
        before_timestamp=before_timestamp,
    )
    if not candidates:
        return None
    best = dict(candidates[0])
    best.pop("_timestamp_sort_key", None)
    return best


def find_acceptance_record_by_run_id(
    *,
    calibration_root: str | None = None,
    run_id: str | None = None,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
) -> "dict | None":
    """run_id に一致する acceptance_result record を返す。"""
    root = calibration_root or CALIBRATION_DIR
    target_run_id = str(run_id or "").strip()
    if not os.path.isdir(root) or not target_run_id:
        return None

    candidates = _collect_accepted_reference_candidates(
        calibration_root=root,
        control_sample_id=control_sample_id,
    )
    for record in candidates:
        if str(record.get("run_id", "")).strip() != target_run_id:
            continue
        record_copy = dict(record)
        record_copy.pop("_timestamp_sort_key", None)
        return record_copy
    return None


def load_accepted_reference_for_preflight(
    calibration_base_dir: str | None = None,
    *,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
    run_id: str | None = None,
    current_measurement_context: dict | None = None,
) -> "dict | None":
    """preflight 用の正式基準を返す。``gate_state == "open"`` のみ対象。

    Returns ``None`` if no qualifying record exists.
    ``run_id`` 未指定の auto lookup は context-bound resolver を通すため、
    current context が未解決なら ``None`` を返す。
    戻り値は ``{"avg_ref": ..., "exposure_us": ..., "run_id": ..., "date": ...}``。
    ``exposure_us`` は旧フォーマットで欠損していれば ``None``。
    """
    if str(run_id or "").strip():
        if current_measurement_context is not None:
            record, _selection_meta = resolve_accepted_reference_record(
                calibration_root=calibration_base_dir,
                control_sample_id=control_sample_id,
                accepted_reference_selection_mode=ACCEPTED_REFERENCE_SELECTION_MODE_FIXED,
                accepted_reference_selection_requested_run_id=run_id,
                require_open_gate=True,
                current_measurement_context=current_measurement_context,
            )
        else:
            record = find_acceptance_record_by_run_id(
                calibration_root=calibration_base_dir,
                run_id=run_id,
                control_sample_id=control_sample_id,
            )
    else:
        record, _selection_meta = resolve_accepted_reference_record(
            calibration_root=calibration_base_dir,
            control_sample_id=control_sample_id,
            accepted_reference_selection_mode=ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
            require_open_gate=True,
            current_measurement_context=current_measurement_context,
        )
    if record is None:
        return None
    # preflight は gate_state == "open" のみ（provisional / blocked を除外）
    if normalize_gate_state(record.get("gate_state")) != GATE_STATE_OPEN:
        return None
    avg_ref = _acceptance_triplet_or_none(record.get("avg_ref"))
    if avg_ref is None:
        return None
    return {
        "avg_ref": avg_ref.tolist() if hasattr(avg_ref, "tolist") else list(avg_ref),
        "exposure_us": record.get("exposure_us"),  # None if missing
        "run_id": record.get("run_id"),
        "date": record.get("timestamp", "")[:10],
        "source_file": record.get("source_file", ""),
        "source_file_rel": record.get("source_file_rel", ""),
        "context_profile_id": record.get("context_profile_id", ""),
        "measurement_context": extract_measurement_context_summary(record),
        "measurement_context_fingerprint": compute_measurement_context_fingerprint(record),
    }


def check_master_ref_drift(
    today_ref: "list | np.ndarray",
    accepted_ref: "list | np.ndarray",
    threshold: float = PREFLIGHT_REF_DRIFT_MAX,
) -> "tuple[bool, dict]":
    """今日の master_ref と正式基準の相対変化を検査する。

    Returns:
        ``(True, {})`` if all channels within *threshold*.
        ``(False, {"channel": str, "drift": float, "all_drifts": dict})`` otherwise.
    """
    channel_names = ("R", "G", "B")
    drifts: dict[str, float] = {}
    for i, ch in enumerate(channel_names):
        ref_val = float(accepted_ref[i])
        if abs(ref_val) < 1e-9:
            drifts[ch] = float("inf")
        else:
            drifts[ch] = abs(float(today_ref[i]) - ref_val) / abs(ref_val)
    worst_ch = max(drifts, key=drifts.get)  # type: ignore[arg-type]
    worst_drift = drifts[worst_ch]
    if worst_drift <= threshold:
        return (True, {})
    return (False, {"channel": worst_ch, "drift": worst_drift, "all_drifts": drifts})


def check_exposure_drift(
    today_exposure_us: float,
    prev_exposure_us: "float | None",
    threshold: float = EXPOSURE_DRIFT_WARN_MAX,
) -> "tuple[str, dict]":
    """P prescan 後の露光変化を検査する（non-blocking）。

    Returns:
        ``("skip", {})`` if *prev_exposure_us* is ``None`` (no baseline).
        ``("pass", {"drift": float})`` if within *threshold*.
        ``("warn", {"drift": float, "today_us": float, "prev_us": float})`` otherwise.
    """
    if prev_exposure_us is None:
        return ("skip", {})
    if abs(prev_exposure_us) < 1e-9:
        return ("warn", {"drift": float("inf"),
                         "today_us": today_exposure_us,
                         "prev_us": prev_exposure_us})
    drift = abs(today_exposure_us - prev_exposure_us) / abs(prev_exposure_us)
    if drift <= threshold:
        return ("pass", {"drift": drift})
    return ("warn", {"drift": drift,
                     "today_us": today_exposure_us,
                     "prev_us": prev_exposure_us})


def find_latest_measurement_gate_record(
    calibration_dir: str | None = None,
    *,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
) -> "dict | None":
    """本番測定可否に使う当日 formal gate record を返す。"""
    records = _load_daily_acceptance_records(
        calibration_dir,
        control_sample_id=control_sample_id,
    )
    policy = get_measurement_production_gate_policy()
    for record in records:
        if str(record.get("run_type")) in policy.allowed_run_types:
            record_copy = dict(record)
            record_copy.pop("_timestamp_sort_key", None)
            return record_copy
    return None


def _load_daily_acceptance_records(
    calibration_dir: str | None = None,
    *,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
) -> list[dict]:
    """当日の acceptance_result record を新しい順で返す。"""
    target_dir = calibration_dir or get_today_calibration_dir()
    if not os.path.isdir(target_dir):
        return []

    import glob

    candidates: list[dict] = []
    for path in sorted(set(glob.glob(os.path.join(target_dir, "acceptance_result*.json")))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        try:
            run_type = normalize_acceptance_run_type(record.get("run_type"))
            result_status = normalize_result_status(record.get("result_status"))
            gate_state = normalize_gate_state(record.get("gate_state"))
        except ValueError:
            continue
        if str(record.get("control_sample_id", "")).strip() != control_sample_id:
            continue
        record_dt = _parse_acceptance_timestamp(record.get("timestamp"))
        record_copy = dict(record)
        record_copy["run_type"] = run_type
        record_copy["result_status"] = result_status
        record_copy["gate_state"] = gate_state
        record_copy.setdefault("run_id", build_acceptance_run_id(record_copy))
        record_copy["_timestamp_sort_key"] = (
            record_dt.isoformat() if record_dt is not None else str(record.get("timestamp", ""))
        )
        record_copy["source_file"] = path
        candidates.append(record_copy)

    candidates.sort(key=lambda record: record.get("_timestamp_sort_key", ""), reverse=True)
    return candidates


def _summarize_measurement_gate_record(record: dict | None, *, label: str) -> str:
    """fail-stop の operator 向けに record 要約を返す。"""
    if not isinstance(record, dict):
        return f"{label}=none"
    return (
        f"{label}={get_run_type_display_name(record.get('run_type'))} "
        f"{record.get('result_status')} / {record.get('gate_state')}"
    )


def _build_measurement_fail_stop_result(
    *,
    allowed: bool,
    reason: str,
    record: dict | None,
    operator_detail: str,
    operator_hint: str = "",
    detail: str = "",
) -> dict:
    """evaluate_measurement_fail_stop の戻り値を一箇所で整形する。"""
    record_copy = dict(record) if isinstance(record, dict) else None
    if record_copy is not None:
        record_copy.pop("_timestamp_sort_key", None)
    return {
        "allowed": allowed,
        "reason": reason,
        "record": record_copy,
        "operator_detail": operator_detail,
        "operator_hint": operator_hint,
        "detail": detail,
    }


def evaluate_measurement_fail_stop(
    calibration_dir: str | None = None,
    *,
    control_sample_id: str = DEFAULT_CONTROL_SAMPLE_ID,
) -> dict:
    """当日の formal acceptance run に基づき本番測定可否を返す。"""
    records = _load_daily_acceptance_records(
        calibration_dir,
        control_sample_id=control_sample_id,
    )
    policy = get_measurement_production_gate_policy()
    latest_record = records[0] if records else None
    latest_formal = next(
        (
            record
            for record in records
            if str(record.get("run_type")) in policy.allowed_run_types
        ),
        None,
    )
    if latest_formal is None:
        if latest_record is not None and str(latest_record.get("run_type")) == RUN_TYPE_VERIFY_ONLY:
            return _build_measurement_fail_stop_result(
                allowed=False,
                reason="verify_only_non_production",
                record=latest_record,
                operator_detail="確認のみ run では本番測定を解放できません",
                operator_hint=OPERATOR_GUI_ACCEPTANCE_HINT,
                detail=_summarize_measurement_gate_record(latest_record, label="latest"),
            )
        return _build_measurement_fail_stop_result(
            allowed=False,
            reason="no_formal_acceptance_run",
            record=latest_record,
            operator_detail="当日の正式判定 run がまだありません",
            operator_hint=OPERATOR_GUI_ACCEPTANCE_HINT,
            detail=_summarize_measurement_gate_record(latest_record, label="latest"),
        )

    result_status = str(latest_formal.get("result_status"))
    if result_status not in policy.accepted_result_statuses:
        return _build_measurement_fail_stop_result(
            allowed=False,
            reason="formal_result_not_accepted",
            record=latest_formal,
            operator_detail="最新の正式判定 run が合格条件を満たしていません",
            operator_hint=OPERATOR_GUI_RECHECK_HINT,
            detail=_summarize_measurement_gate_record(latest_formal, label="latest_formal"),
        )

    gate_state = str(latest_formal.get("gate_state"))
    if gate_state != policy.required_gate_state:
        return _build_measurement_fail_stop_result(
            allowed=False,
            reason="formal_gate_not_open",
            record=latest_formal,
            operator_detail="最新の正式判定 run が open ではありません",
            operator_hint=OPERATOR_GUI_RECHECK_HINT,
            detail=_summarize_measurement_gate_record(latest_formal, label="latest_formal"),
        )

    return _build_measurement_fail_stop_result(
        allowed=True,
        reason="accepted",
        record=latest_formal,
        operator_detail="",
        detail=_summarize_measurement_gate_record(latest_formal, label="latest_formal"),
    )


def get_acceptance_runner_exit_code(result_status: str | None) -> int:
    """result_status を runner 用終了コードへ変換する。"""
    status = normalize_result_status(result_status)
    mapping = {
        RESULT_STATUS_ACCEPTED: 0,
        RESULT_STATUS_MINIMUM: ACCEPTANCE_RUNNER_EXIT_CODE_MINIMUM,
        RESULT_STATUS_REJECTED: ACCEPTANCE_RUNNER_EXIT_CODE_REJECTED,
        RESULT_STATUS_RECHECK_REQUIRED: ACCEPTANCE_RUNNER_EXIT_CODE_RECHECK_REQUIRED,
    }
    return mapping[status]


def evaluate_acceptance_runner_exit_code(record: dict | None) -> int:
    """acceptance_result record から runner 終了コードを返す。"""
    if not isinstance(record, dict):
        return ACCEPTANCE_RUNNER_EXIT_CODE_MISSING_RESULT
    try:
        return get_acceptance_runner_exit_code(record.get("result_status"))
    except ValueError:
        return ACCEPTANCE_RUNNER_EXIT_CODE_MISSING_RESULT


def compute_ref_scale(
    ref_now: "np.ndarray",
    ref_train: "np.ndarray",
) -> float:
    """Ref RGB の輝度比をスカラーで返す。

    Y係数ベースの輝度近似（入力はセンサー RGB）。
    チャネル別ベクトルではなく単一スカラーを返すことで、
    分光歪みの注入を防ぎ、輝度補正のみを行う。

    Args:
        ref_now: (3,) 現フレーム（または EMA）の Ref RGB。
        ref_train: (3,) CCM学習時の Ref RGB。
    Returns:
        Y_now / Y_train（float スカラー）。
    """
    y_now = float(np.asarray(ref_now, dtype=np.float64).ravel()[:3] @ _Y_COEFF)
    y_train = float(np.asarray(ref_train, dtype=np.float64).ravel()[:3] @ _Y_COEFF)
    return y_now / max(y_train, 1e-8)


def _normalize_ref_triplet(candidate: "np.ndarray | None") -> "np.ndarray | None":
    """3要素 RGB 候補を安全に正規化する。"""
    if candidate is None:
        return None
    rgb = np.asarray(candidate, dtype=np.float64).ravel()
    if rgb.size < 3:
        return None
    rgb = rgb[:3].copy()
    if not np.all(np.isfinite(rgb)):
        return None
    if float(rgb[1]) < 1e-8:
        return None
    return rgb


def normalize_ref_scale_triplet(
    ref_rgb: "np.ndarray | None",
    white_ratio_rgb: "np.ndarray | None",
) -> "np.ndarray | None":
    """ref_scale 用に Ref RGB を白点比で正規化した triplet を返す。"""
    ref = _normalize_ref_triplet(ref_rgb)
    white = _normalize_ref_triplet(white_ratio_rgb)
    if ref is None or white is None:
        return None
    safe_white = np.where(white < 1e-8, 1e-8, white)
    normalized = ref / safe_white
    if not np.all(np.isfinite(normalized)):
        return None
    return normalized.astype(np.float64)


def split_neutral_correction_baselines(
    record: dict,
    white_ratio_rgb: "np.ndarray | None" = None,
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """neutral_correction.json の raw / ref_scale baseline を分離する。"""
    if not isinstance(record, dict):
        return None, None
    raw_baseline = _normalize_ref_triplet(record.get("ref_baseline"))
    scale_baseline = _normalize_ref_triplet(record.get("ref_scale_baseline"))
    if scale_baseline is None:
        scale_baseline = normalize_ref_scale_triplet(raw_baseline, white_ratio_rgb)
    return raw_baseline, scale_baseline


def select_live_ref_scale_baseline(
    ref_scale_baseline: "np.ndarray | None",
    master_ref_rgb: "np.ndarray | None",
    ref_train: "np.ndarray | None",
    white_ratio_rgb: "np.ndarray | None",
) -> "tuple[np.ndarray | None, str]":
    """live ref_scale 用 baseline と取得元を返す。

    優先順:
      1. `neutral_correction.ref_scale_baseline`
      2. `master_ref.json`
      3. `ccm.json.ref_train`

    `ref_baseline` の raw neutral capture はここでは使わない。
    """
    neutral = _normalize_ref_triplet(ref_scale_baseline)
    if neutral is not None:
        return neutral, "neutral"

    for source, candidate in (
        ("master_ref", master_ref_rgb),
        ("ref_train", ref_train),
    ):
        rgb = normalize_ref_scale_triplet(candidate, white_ratio_rgb)
        if rgb is not None:
            return rgb, source
    return None, "none"


def select_live_ref_baseline(
    neutral_baseline: "np.ndarray | None",
    master_ref_rgb: "np.ndarray | None",
    ref_train: "np.ndarray | None",
) -> "tuple[np.ndarray | None, str]":
    """live ref_scale 用 baseline と取得元を返す。

    優先順:
      1. `neutral_correction.ref_baseline`
      2. `master_ref.json`
      3. `ccm.json.ref_train`

    Args:
        neutral_baseline: live 測定ドメインで取得された neutral baseline。
        master_ref_rgb: W で保存された master ref。
        ref_train: P 学習時の ref_train。最後の安全側フォールバック。
    Returns:
        `(baseline, source)`。
        `source` は `neutral` / `master_ref` / `ref_train` / `none`。
    """

    for source, candidate in (
        ("neutral", neutral_baseline),
        ("master_ref", master_ref_rgb),
        ("ref_train", ref_train),
    ):
        rgb = _normalize_ref_triplet(candidate)
        if rgb is not None:
            return rgb, source
    return None, "none"


def run_canonical_lab_pipeline(
    corrected_ratio_rgb: "np.ndarray",
    white_ratio_rgb: "np.ndarray",
    ref_rgb_now: "np.ndarray | None",
    live_ref_baseline: "np.ndarray | None",
    lab_converter,
    *,
    suppress_domain_warning: bool = False,
) -> dict:
    """blank補正後の ratio を canonical pipeline で Lab へ変換する。

    処理順:
      1. white_ratio_rgb で白正規化
      2. live_ref_baseline があれば ref_scale を適用
      3. ratio_to_Lab() で Lab へ変換

    Args:
        corrected_ratio_rgb: blank補正済み ratio。
        white_ratio_rgb: white patch の ratio。blank補正済みを渡す。
        ref_rgb_now: 現在の Ref RGB。`ref_scale` 用。
        live_ref_baseline: live ref_scale 基準。無い場合は ref_scale を省略。
        lab_converter: `ratio_to_Lab()` を持つ変換器。
    Returns:
        `{"ratio_white": np.ndarray, "ref_scale": float|None, "lab": np.ndarray}`。
    """
    corrected_ratio = np.asarray(corrected_ratio_rgb, dtype=np.float64).ravel()[:3]
    white_ratio = np.asarray(white_ratio_rgb, dtype=np.float64).ravel()[:3]
    safe_white = np.where(white_ratio < 1e-8, 1e-8, white_ratio)
    ratio_white = corrected_ratio / safe_white

    ref_scale: float | None = None
    if (
        live_ref_baseline is not None
        and ref_rgb_now is not None
        and getattr(lab_converter, "_ccm", None) is not None
    ):
        safe_baseline = np.where(
            np.asarray(live_ref_baseline, dtype=np.float64).ravel()[:3] < 1e-8,
            1e-8,
            np.asarray(live_ref_baseline, dtype=np.float64).ravel()[:3],
        )
        ref_scale = compute_ref_scale(ref_rgb_now, safe_baseline)
        ratio_white = ratio_white * ref_scale

    lab = np.asarray(
        lab_converter.ratio_to_Lab(
            ratio_white.astype(np.float32),
            suppress_domain_warning=suppress_domain_warning,
        ),
        dtype=np.float64,
    ).ravel()[:3]
    return {
        "ratio_white": ratio_white.astype(np.float64),
        "ref_scale": ref_scale,
        "lab": lab,
    }


def _run_git_identity_command(args: list[str]) -> str:
    """software identity 用 git コマンドを実行して結果を返す。"""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        value = subprocess.check_output(
            args,
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"
    return value or "unknown"


def _get_software_version() -> str:
    """現在の software_version を返す。取得できない場合は `unknown`。"""
    env_value = str(os.environ.get("PICOLOR_SOFTWARE_VERSION", "")).strip()
    if env_value:
        return env_value
    file_value = _read_identity_file(".software_version")
    if file_value:
        return file_value
    return _run_git_identity_command(["git", "describe", "--tags", "--dirty", "--always"])


def _read_identity_file(filename: str) -> str:
    """デプロイ時に書き出された identity ファイルを読む。見つからなければ空文字。"""
    from pathlib import Path
    # csi/ の親ディレクトリ（プロジェクトルート）に identity ファイルがある
    for base in [Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent]:
        candidate = base / filename
        if candidate.is_file():
            try:
                value = candidate.read_text(encoding="utf-8").strip()
                if value and value != "unknown":
                    return value
            except OSError:
                pass
    return ""


def _get_git_revision() -> str:
    """現在の git revision を返す。取得できない場合は `unknown`。"""
    env_value = str(os.environ.get("PICOLOR_GIT_REVISION", "")).strip()
    if env_value:
        return env_value
    file_value = _read_identity_file(".git_revision")
    if file_value:
        return file_value
    return _run_git_identity_command(["git", "rev-parse", "--short", "HEAD"])


def get_software_identity() -> dict:
    """保存用の software_version / git_revision をまとめて返す。"""
    return {
        "software_version": _get_software_version(),
        "git_revision": _get_git_revision(),
    }


def _json_ready(value):
    """JSON保存できる形へ numpy / tuple を再帰的に変換する。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _format_triplet(value) -> str:
    """RGB/Lab 3要素配列を人間可読文字列へ整形する。"""
    if value is None:
        return "N/A"
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size < 3:
        return "N/A"
    return f"[{arr[0]:.6f}, {arr[1]:.6f}, {arr[2]:.6f}]"


def _has_meaningful_value(value) -> bool:
    """受入れ契約の必須値が空でないかを判定する。"""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _find_acceptance_schema_contract_errors(record: dict) -> list[str]:
    """acceptance_result の最小 schema / structural contract 違反を返す。"""
    errors = []
    for field_name in ACCEPTANCE_RESULT_JSON_REQUIRED_FIELDS:
        if not _has_meaningful_value(record.get(field_name)):
            errors.append(field_name)
    for field_name in ACCEPTANCE_RESULT_JSON_REQUIRED_ID_FIELDS:
        if not _has_meaningful_value(record.get(field_name)):
            errors.append(field_name)
    for field_group in ACCEPTANCE_RESULT_JSON_ALTERNATIVE_FIELD_GROUPS:
        if not any(_has_meaningful_value(record.get(field_name)) for field_name in field_group):
            errors.append("/".join(field_group))
    return sorted(set(errors))


def _is_production_relevant_acceptance_record(record: dict) -> bool:
    """operational evidence contract の対象かを返す。"""
    try:
        run_type = normalize_acceptance_run_type(record.get("run_type"))
    except ValueError:
        run_type = RUN_TYPE_VERIFY_ONLY
    return run_type in MEASUREMENT_GATE_RUN_TYPES


def _accepted_reference_evidence_required(record: dict) -> bool:
    """selected reference の説明証跡が必要な record かを返す。"""
    accepted_reference_run_id = str(
        record.get("accepted_reference_run_id", "")
    ).strip().lower()
    if accepted_reference_run_id and accepted_reference_run_id != "none":
        return True
    status = str(record.get("accepted_reference_context_status", "")).strip()
    return bool(status and status != "not_evaluated")


def _find_acceptance_operational_contract_errors(record: dict) -> list[str]:
    """production-relevant artifact 用 operational contract 違反を返す。"""
    errors = []
    if not has_known_software_identity(record):
        errors.append("software_identity")
    for field_name in ACCEPTANCE_RESULT_OPERATIONAL_REQUIRED_FIELDS:
        if not _has_meaningful_value(record.get(field_name)):
            errors.append(field_name)
    for field_group in ACCEPTANCE_RESULT_OPERATIONAL_ALTERNATIVE_FIELD_GROUPS:
        if not any(_has_meaningful_value(record.get(field_name)) for field_name in field_group):
            errors.append("/".join(field_group))
    if _accepted_reference_evidence_required(record):
        for field_name in ACCEPTANCE_RESULT_OPERATIONAL_SELECTED_REFERENCE_FIELDS:
            if not _has_meaningful_value(record.get(field_name)):
                errors.append(field_name)
        accepted_reference_run_id = str(
            record.get("accepted_reference_run_id", "")
        ).strip().lower()
        if accepted_reference_run_id and accepted_reference_run_id != "none":
            if not _has_meaningful_value(record.get("accepted_reference_context_fingerprint")):
                errors.append("accepted_reference_context_fingerprint")
            if not _has_meaningful_value(
                record.get("selected_reference_measurement_context_fingerprint")
            ):
                errors.append("selected_reference_measurement_context_fingerprint")
    return sorted(set(errors))


def _apply_operator_guidance_quarantine(record: dict) -> dict:
    """guided runner 逸脱時は既存 provisional gate へ寄せる。"""
    record["operator_guidance_mode"] = normalize_operator_guidance_mode(
        record.get("operator_guidance_mode"),
    )
    record["guidance_degraded_reasons"] = normalize_guidance_degraded_reasons(
        record.get("guidance_degraded_reasons"),
    )
    if not _is_production_relevant_acceptance_record(record):
        return record
    if (
        record["operator_guidance_mode"] == OPERATOR_GUIDANCE_MODE_GUIDED
        and not record["guidance_degraded_reasons"]
    ):
        return record

    try:
        gate_state = normalize_gate_state(record.get("gate_state"))
    except ValueError:
        gate_state = GATE_STATE_BLOCKED
    if gate_state == GATE_STATE_BLOCKED:
        return record

    provisional_reasons = [
        str(reason).strip()
        for reason in list(record.get("provisional_reasons") or [])
        if str(reason).strip()
    ]
    if "operator_guidance_not_strict" not in provisional_reasons:
        provisional_reasons.append("operator_guidance_not_strict")
    for reason in record["guidance_degraded_reasons"]:
        tagged_reason = f"guidance:{reason}"
        if tagged_reason not in provisional_reasons:
            provisional_reasons.append(tagged_reason)
    record["provisional_reasons"] = provisional_reasons
    record["gate_state"] = GATE_STATE_PROVISIONAL
    return record


def _build_acceptance_production_decision(
    record: dict,
    *,
    contract_errors: list[str] | None = None,
) -> dict:
    """artifact 単体から production eligibility snapshot を組み立てる。"""
    errors = list(contract_errors or [])
    policy = get_measurement_production_gate_policy()
    try:
        run_type = normalize_acceptance_run_type(record.get("run_type"))
    except ValueError:
        run_type = RUN_TYPE_VERIFY_ONLY
    try:
        result_status = normalize_result_status(record.get("result_status"))
    except ValueError:
        result_status = str(record.get("result_status", ""))
    try:
        gate_state = normalize_gate_state(record.get("gate_state"))
    except ValueError:
        gate_state = str(record.get("gate_state", ""))
    display_name = get_run_type_display_name(run_type)
    detail_prefix = _summarize_measurement_gate_record(record, label="artifact")

    if run_type not in policy.allowed_run_types:
        return {
            "production_eligible": False,
            "production_decision_reason": (
                "verify_only_non_production"
                if run_type == RUN_TYPE_VERIFY_ONLY
                else "non_production_run_type"
            ),
            "production_decision_detail": detail_prefix,
            "production_operator_detail": (
                "確認のみ run では本番測定を解放できません"
                if run_type == RUN_TYPE_VERIFY_ONLY
                else f"{display_name} では本番測定を解放できません"
            ),
            "production_operator_hint": OPERATOR_GUI_ACCEPTANCE_HINT,
        }
    if result_status not in policy.accepted_result_statuses:
        return {
            "production_eligible": False,
            "production_decision_reason": "formal_result_not_accepted",
            "production_decision_detail": detail_prefix,
            "production_operator_detail": "最新の正式判定 run が合格条件を満たしていません",
            "production_operator_hint": OPERATOR_GUI_RECHECK_HINT,
        }
    if gate_state != policy.required_gate_state:
        return {
            "production_eligible": False,
            "production_decision_reason": "formal_gate_not_open",
            "production_decision_detail": detail_prefix,
            "production_operator_detail": "最新の正式判定 run が open ではありません",
            "production_operator_hint": OPERATOR_GUI_RECHECK_HINT,
        }
    if errors:
        return {
            "production_eligible": False,
            "production_decision_reason": "operational_contract_incomplete",
            "production_decision_detail": (
                f"{detail_prefix} contract_errors={sorted(set(errors))}"
            ),
            "production_operator_detail": "最新の正式判定 run の運用証跡が不足しています",
            "production_operator_hint": OPERATOR_GUI_CONTRACT_HINT,
        }
    return {
        "production_eligible": True,
        "production_decision_reason": "accepted",
        "production_decision_detail": detail_prefix,
        "production_operator_detail": "最新の正式判定 run が本番測定を解放しています",
        "production_operator_hint": "",
    }


def _build_acceptance_contract_snapshot(
    record: dict,
    *,
    contract_errors: list[str] | None = None,
    schema_contract_errors: list[str] | None = None,
    operational_contract_errors: list[str] | None = None,
) -> dict:
    """artifact に保存する contract / production snapshot を返す。"""
    selected_reference_summary = (
        extract_measurement_context_summary(record.get("selected_reference_measurement_context"))
        if _has_meaningful_value(record.get("selected_reference_measurement_context"))
        else {}
    )
    selected_reference_fingerprint = str(
        record.get("selected_reference_measurement_context_fingerprint", "")
    ).strip() or (
        compute_measurement_context_fingerprint(selected_reference_summary)
        if selected_reference_summary
        else ""
    )
    schema_errors = list(
        schema_contract_errors
        if schema_contract_errors is not None
        else _find_acceptance_schema_contract_errors(record)
    )
    operational_errors = list(
        operational_contract_errors
        if operational_contract_errors is not None
        else (
            _find_acceptance_operational_contract_errors(record)
            if _is_production_relevant_acceptance_record(record)
            else []
        )
    )
    all_errors = sorted(set(contract_errors or (schema_errors + operational_errors)))
    production = _build_acceptance_production_decision(
        record,
        contract_errors=all_errors,
    )
    return {
        "selected_reference_measurement_context": selected_reference_summary,
        "selected_reference_measurement_context_fingerprint": (
            selected_reference_fingerprint
        ),
        "schema_contract_errors": sorted(set(schema_errors)),
        "operational_contract_errors": sorted(set(operational_errors)),
        "contract_errors": all_errors,
        "contract_status": "failed" if all_errors else "ok",
        "production_unlock_reason": (
            production["production_decision_reason"]
            if production.get("production_eligible")
            else ""
        ),
        **production,
    }


def summarize_acceptance_artifact_operational_state(
    record: dict | None,
    *,
    selected_reference_record: dict | None = None,
) -> dict:
    """offline/report 用に artifact の contract / production snapshot を返す。"""
    record_copy = attach_measurement_context_metadata(record)
    for field_name, default_value in _default_accepted_reference_context_meta().items():
        record_copy.setdefault(field_name, default_value)
    if (
        not _has_meaningful_value(record_copy.get("selected_reference_measurement_context"))
        and isinstance(selected_reference_record, dict)
    ):
        selected_summary = extract_measurement_context_summary(selected_reference_record)
        record_copy["selected_reference_measurement_context"] = selected_summary
        record_copy["selected_reference_measurement_context_fingerprint"] = (
            compute_measurement_context_fingerprint(selected_summary)
        )
        if not _has_meaningful_value(record_copy.get("accepted_reference_context_fingerprint")):
            record_copy["accepted_reference_context_fingerprint"] = str(
                record_copy.get(
                    "selected_reference_measurement_context_fingerprint",
                    "",
                )
            ).strip()
    return _build_acceptance_contract_snapshot(record_copy)


def _find_acceptance_contract_errors(record: dict) -> list[str]:
    """acceptance_result 契約違反のフィールド名一覧を返す。"""
    schema_errors = _find_acceptance_schema_contract_errors(record)
    operational_errors = (
        _find_acceptance_operational_contract_errors(record)
        if _is_production_relevant_acceptance_record(record)
        else []
    )
    return sorted(set(schema_errors + operational_errors))


def build_acceptance_result_log_text(record: dict) -> str:
    """acceptance_result.log の人間可読本文を返す。"""
    lines = [
        "Acceptance result",
        f"timestamp: {record.get('timestamp', 'unknown')}",
        f"run_id: {record.get('run_id', 'unknown')}",
        f"run_type: {record.get('run_type', RUN_TYPE_VERIFY_ONLY)}",
        f"operator: {record.get('operator', 'unknown')}",
        f"git_revision: {record.get('git_revision', 'unknown')}",
        f"software_version: {record.get('software_version', 'unknown')}",
        f"calibration_dir: {record.get('calibration_dir', 'unknown')}",
        f"control_sample_id: {record.get('control_sample_id', 'unknown')}",
        f"fixture_id: {record.get('fixture_id', 'N/A')}",
        f"positioning_method: {record.get('positioning_method', 'N/A')}",
        f"context_profile_id: {record.get('context_profile_id', '')}",
        f"measurement_context: {record.get('measurement_context', {})}",
        "measurement_context_fingerprint: "
        f"{record.get('measurement_context_fingerprint', '')}",
        f"accepted_reference_run_id: {record.get('accepted_reference_run_id', 'none')}",
        "accepted_reference_selection_mode: "
        f"{record.get('accepted_reference_selection_mode', ACCEPTED_REFERENCE_SELECTION_MODE_AUTO)}",
        "accepted_reference_selection_requested_run_id: "
        f"{record.get('accepted_reference_selection_requested_run_id', '')}",
        "accepted_reference_selection_source_file: "
        f"{record.get('accepted_reference_selection_source_file', '')}",
        "accepted_reference_context_status: "
        f"{record.get('accepted_reference_context_status', 'not_evaluated')}",
        "accepted_reference_context_warning: "
        f"{record.get('accepted_reference_context_warning', '')}",
        "accepted_reference_context_mismatch_fields: "
        f"{record.get('accepted_reference_context_mismatch_fields', [])}",
        "accepted_reference_current_context_missing_contract_fields: "
        f"{record.get('accepted_reference_current_context_missing_contract_fields', [])}",
        "accepted_reference_reference_context_missing_contract_fields: "
        f"{record.get('accepted_reference_reference_context_missing_contract_fields', [])}",
        "accepted_reference_context_fingerprint: "
        f"{record.get('accepted_reference_context_fingerprint', '')}",
        "selected_reference_measurement_context: "
        f"{record.get('selected_reference_measurement_context', {})}",
        "selected_reference_measurement_context_fingerprint: "
        f"{record.get('selected_reference_measurement_context_fingerprint', '')}",
        "previous_accepted_day_run_id: "
        f"{record.get('previous_accepted_day_run_id', 'none')}",
        "previous_accepted_day_timestamp: "
        f"{record.get('previous_accepted_day_timestamp', '')}",
        "previous_accepted_day_date: "
        f"{record.get('previous_accepted_day_date', '')}",
        "previous_accepted_day_selection_warning: "
        f"{record.get('previous_accepted_day_selection_warning', '')}",
        "previous_accepted_day_context_status: "
        f"{record.get('previous_accepted_day_context_status', 'not_evaluated')}",
        "previous_accepted_day_context_warning: "
        f"{record.get('previous_accepted_day_context_warning', '')}",
        "previous_accepted_day_context_mismatch_fields: "
        f"{record.get('previous_accepted_day_context_mismatch_fields', [])}",
        "previous_accepted_day_current_context_missing_contract_fields: "
        f"{record.get('previous_accepted_day_current_context_missing_contract_fields', [])}",
        "previous_accepted_day_reference_context_missing_contract_fields: "
        f"{record.get('previous_accepted_day_reference_context_missing_contract_fields', [])}",
        "previous_accepted_day_context_fingerprint: "
        f"{record.get('previous_accepted_day_context_fingerprint', '')}",
        "selected_previous_accepted_day_measurement_context: "
        f"{record.get('selected_previous_accepted_day_measurement_context', {})}",
        "selected_previous_accepted_day_measurement_context_fingerprint: "
        f"{record.get('selected_previous_accepted_day_measurement_context_fingerprint', '')}",
        f"reference_is_same_day: {record.get('reference_is_same_day', False)}",
        f"reference_is_same_session: {record.get('reference_is_same_session', False)}",
        f"reference_relation: {record.get('reference_relation', REFERENCE_RELATION_NONE)}",
        f"warmup_elapsed_sec: {record.get('warmup_elapsed_sec', 'N/A')}",
        f"baseline_source: {record.get('baseline_source', 'none')}",
        f"baseline_value: {_format_triplet(record.get('baseline_value'))}",
        f"white_ratio_rgb: {_format_triplet(record.get('white_ratio_rgb'))}",
        f"avg_ref: {_format_triplet(record.get('avg_ref'))}",
        f"ref_scale: {record.get('ref_scale', 'N/A')}",
        f"relative_rgb_mean: {_format_triplet(record.get('relative_rgb_mean'))}",
        f"relative_rgb_std: {_format_triplet(record.get('relative_rgb_std'))}",
        f"lab_rel_l: {record.get('lab_rel_l', 'N/A')}",
        f"measured_lab: {_format_triplet(record.get('measured_lab'))}",
        f"delta_e: {record.get('delta_e', 'N/A')}",
        f"roi_or_grid_source: {record.get('roi_or_grid_source', 'unknown')}",
        f"exposure: {record.get('exposure', {})}",
        f"mode: {record.get('mode', 'unknown')}",
        f"result_status: {record.get('result_status', 'unknown')}",
        f"gate_state: {record.get('gate_state', 'unknown')}",
        f"repro_rollout_mode: {record.get('repro_rollout_mode', 'enforced')}",
        f"manual_roi_used: {record.get('manual_roi_used', False)}",
        f"manual_corner_used: {record.get('manual_corner_used', False)}",
        "operator_guidance_mode: "
        f"{record.get('operator_guidance_mode', OPERATOR_GUIDANCE_MODE_GUIDED)}",
        "guidance_degraded_reasons: "
        f"{record.get('guidance_degraded_reasons', [])}",
        f"provisional_reasons: {record.get('provisional_reasons', [])}",
        f"failed_checks: {record.get('failed_checks', [])}",
        f"production_eligible: {record.get('production_eligible', False)}",
        f"production_decision_reason: {record.get('production_decision_reason', '')}",
        f"production_unlock_reason: {record.get('production_unlock_reason', '')}",
        f"production_decision_detail: {record.get('production_decision_detail', '')}",
        f"production_operator_detail: {record.get('production_operator_detail', '')}",
        f"production_operator_hint: {record.get('production_operator_hint', '')}",
        f"contract_status: {record.get('contract_status', 'unknown')}",
        f"schema_contract_errors: {record.get('schema_contract_errors', [])}",
        f"operational_contract_errors: {record.get('operational_contract_errors', [])}",
        f"contract_errors: {record.get('contract_errors', [])}",
        f"accepted_reference_diff: {record.get('accepted_reference_diff', {})}",
        "previous_accepted_day_diff: "
        f"{record.get('previous_accepted_day_diff', {})}",
        "previous_accepted_day_judgement: "
        f"{record.get('previous_accepted_day_judgement', {})}",
        f"steady_state: {record.get('steady_state', {})}",
        f"gray_audit_paths: {record.get('gray_audit_paths', {})}",
    ]
    return "\n".join(lines) + "\n"


# NOTE: detect_spydercheckr_corners 等の contour ベース自動検出は
# 黒フレーム外縁検出による系統的ズレが解消できず撤去。
# 代わりに saved corners + パッチ輝度検証方式を採用。

SAVED_CORNERS_VALIDATION_BRIGHTNESS_DIFF = 80  # 1E-6E 輝度差の下限（暫定値）


def validate_saved_corners(
    gray_frame: "np.ndarray",
    corners: "list[list[int]]",
    hinge_gap: float = 2.67,
    brightness_diff_threshold: int = SAVED_CORNERS_VALIDATION_BRIGHTNESS_DIFF,
) -> bool:
    """saved corners の位置精度をパッチ輝度差で検証する。

    saved corners からホモグラフィを計算し、1E Card White (idx=4) と
    6E Card Black (idx=44) の ROI 平均輝度を比較する。
    輝度差が *brightness_diff_threshold* 以上なら位置が正しいと判定。

    閾値 80 の根拠: 1E (L*=96) と 6E (L*=17) の 8-bit グレー変換差は
    約 180。半分以下なら位置ズレと見なせる保守的下限。

    Returns:
        True: 位置精度 OK。False: 位置ズレの疑い。
    """
    import cv2 as _cv2

    if len(corners) != 4:
        return False

    n_cols, n_rows = 8, 6
    half = n_cols // 2
    chart_w = n_cols * 100 + hinge_gap * 100
    chart_h = n_rows * 100

    src_pts = np.float32([
        [0, 0], [chart_w, 0], [chart_w, chart_h], [0, chart_h],
    ])
    dst_pts = np.float32(corners)
    try:
        M = _cv2.getPerspectiveTransform(src_pts, dst_pts)
    except _cv2.error:
        return False

    def _patch_mean(row: int, col: int) -> float:
        """指定パッチの ROI 平均輝度を返す。"""
        gap_offset = hinge_gap * 100 if col >= half else 0.0
        cx_n = (col + 0.5) * 100 + gap_offset
        cy_n = (row + 0.5) * 100
        ph = 100 * 0.25  # patch_margin 相当
        pts_n = np.float32([[[cx_n - ph, cy_n - ph], [cx_n + ph, cy_n + ph]]])
        pts_d = _cv2.perspectiveTransform(pts_n, M)
        tl = pts_d[0][0].astype(int)
        br = pts_d[0][1].astype(int)
        h_f, w_f = gray_frame.shape[:2]
        x0, y0 = max(0, int(tl[0])), max(0, int(tl[1]))
        x1, y1 = min(w_f, int(br[0])), min(h_f, int(br[1]))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(np.mean(gray_frame[y0:y1, x0:x1]))

    # 1E Card White: row=0, col=4 (idx=4)
    white_mean = _patch_mean(0, 4)
    # 6E Card Black: row=5, col=4 (idx=44)
    black_mean = _patch_mean(5, 4)

    diff = white_mean - black_mean
    return diff >= brightness_diff_threshold


def auto_register_accepted_reference(
    acceptance_result: dict,
    calibration_dir: str | None = None,
) -> "dict | None":
    """合格 run を正式基準として自動登録する。

    formal run だけを accepted reference chain の対象にする。
    ``gate_state == "open"`` かつ ``result_status`` が ``合格`` or ``最低合格``
    のときのみ ``accepted_reference.json`` に保存する。
    ``provisional`` run は正式基準に昇格させない（別ファイルに保存）。
    `verify_only` は診断用途のため、accepted reference を更新しない。

    Returns:
        保存した dict。登録しなかった場合は ``None``。
    """
    cal_dir = calibration_dir or get_today_calibration_dir()
    os.makedirs(cal_dir, exist_ok=True)

    acceptance_result = attach_measurement_context_metadata(acceptance_result)
    run_type = normalize_acceptance_run_type(acceptance_result.get("run_type"))
    result_status = str(acceptance_result.get("result_status", ""))
    gate_state = str(acceptance_result.get("gate_state", ""))
    ref_data = {
        "run_type": run_type,
        "avg_ref": acceptance_result.get("avg_ref"),
        "exposure_us": acceptance_result.get("exposure", {}).get("exposure_us")
                       if isinstance(acceptance_result.get("exposure"), dict)
                       else acceptance_result.get("exposure_us"),
        "run_id": acceptance_result.get("run_id"),
        "date": str(acceptance_result.get("timestamp", ""))[:10],
        "result_status": result_status,
        "gate_state": gate_state,
        "measured_lab": acceptance_result.get("measured_lab"),
        "delta_e": acceptance_result.get("delta_e"),
        "ref_scale": acceptance_result.get("ref_scale"),
        "fixture_id": acceptance_result.get("fixture_id", ""),
        "positioning_method": acceptance_result.get("positioning_method", ""),
        "context_profile_id": acceptance_result.get("context_profile_id", ""),
        "measurement_context": acceptance_result.get("measurement_context", {}),
        "measurement_context_fingerprint": acceptance_result.get(
            "measurement_context_fingerprint",
            "",
        ),
        "selected_reference_measurement_context": acceptance_result.get(
            "selected_reference_measurement_context",
            {},
        ),
        "selected_reference_measurement_context_fingerprint": acceptance_result.get(
            "selected_reference_measurement_context_fingerprint",
            "",
        ),
        "accepted_reference_context_status": acceptance_result.get(
            "accepted_reference_context_status",
            "not_evaluated",
        ),
        "accepted_reference_context_warning": acceptance_result.get(
            "accepted_reference_context_warning",
            "",
        ),
        "accepted_reference_context_fingerprint": acceptance_result.get(
            "accepted_reference_context_fingerprint",
            "",
        ),
        "contract_status": acceptance_result.get("contract_status", "unknown"),
        "schema_contract_errors": acceptance_result.get("schema_contract_errors", []),
        "operational_contract_errors": acceptance_result.get(
            "operational_contract_errors",
            [],
        ),
        "contract_errors": acceptance_result.get("contract_errors", []),
        "production_eligible": acceptance_result.get("production_eligible", False),
        "production_decision_reason": acceptance_result.get(
            "production_decision_reason",
            "",
        ),
        "production_unlock_reason": acceptance_result.get(
            "production_unlock_reason",
            "",
        ),
        "production_decision_detail": acceptance_result.get(
            "production_decision_detail",
            "",
        ),
        "production_operator_detail": acceptance_result.get(
            "production_operator_detail",
            "",
        ),
        "production_operator_hint": acceptance_result.get(
            "production_operator_hint",
            "",
        ),
        "operator_guidance_mode": acceptance_result.get(
            "operator_guidance_mode",
            OPERATOR_GUIDANCE_MODE_GUIDED,
        ),
        "guidance_degraded_reasons": acceptance_result.get(
            "guidance_degraded_reasons",
            [],
        ),
        "software_version": acceptance_result.get("software_version", "unknown"),
        "git_revision": acceptance_result.get("git_revision", "unknown"),
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }

    policy = get_accepted_reference_selection_policy()
    if run_type not in policy.accepted_run_types:
        return None

    is_accepted = result_status in (RESULT_STATUS_ACCEPTED, RESULT_STATUS_MINIMUM)

    # provisional → 別ファイルに保存（正式基準には昇格させない）
    if gate_state == GATE_STATE_PROVISIONAL and is_accepted:
        prov_path = os.path.join(cal_dir, "provisional_reference.json")
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)
        return None

    # open + 合格/最低合格 のみ正式基準に登録
    if gate_state != GATE_STATE_OPEN or not is_accepted:
        return None

    ref_path = os.path.join(cal_dir, "accepted_reference.json")
    # バックアップ: 既存ファイルをタイムスタンプ付きでリネーム
    if os.path.exists(ref_path):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = os.path.join(cal_dir, f"accepted_reference_{ts}.json")
        os.rename(ref_path, backup_path)
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref_data, f, ensure_ascii=False, indent=2)
    return ref_data


def persist_acceptance_result_artifacts(
    record: dict,
    calibration_dir: str | None = None,
) -> dict:
    """acceptance_result.json/.log を履歴+latest の両方へ保存する。"""
    ensure_calibration_dir()
    save_dir = calibration_dir or get_today_calibration_dir()
    os.makedirs(save_dir, exist_ok=True)

    record_copy = _json_ready(dict(record))
    record_copy.setdefault(
        "timestamp",
        datetime.now().isoformat(timespec="seconds"),
    )
    record_copy["run_type"] = normalize_acceptance_run_type(record_copy.get("run_type"))
    software_identity = get_software_identity()
    record_copy.setdefault("software_version", software_identity["software_version"])
    record_copy.setdefault("git_revision", software_identity["git_revision"])
    record_copy.setdefault("control_sample_id", DEFAULT_CONTROL_SAMPLE_ID)
    record_copy.setdefault("fixture_id", "")
    record_copy.setdefault("positioning_method", "")
    record_copy.setdefault("context_profile_id", "")
    record_copy.setdefault("manual_roi_used", False)
    record_copy.setdefault("manual_corner_used", False)
    record_copy.setdefault("provisional_reasons", [])
    record_copy.setdefault("accepted_reference_run_id", "")
    record_copy.setdefault(
        "accepted_reference_selection_mode",
        ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
    )
    record_copy.setdefault("accepted_reference_selection_requested_run_id", "")
    record_copy.setdefault("accepted_reference_selection_source_file", "")
    record_copy.setdefault("accepted_reference_selection_warning", "")
    record_copy.setdefault("previous_accepted_day_run_id", "")
    record_copy.setdefault("previous_accepted_day_timestamp", "")
    record_copy.setdefault("previous_accepted_day_date", "")
    record_copy.setdefault("previous_accepted_day_selection_warning", "")
    record_copy.setdefault("reference_is_same_day", False)
    record_copy.setdefault("reference_is_same_session", False)
    record_copy.setdefault("reference_relation", REFERENCE_RELATION_NONE)
    for field_name, default_value in _default_accepted_reference_context_meta().items():
        record_copy.setdefault(field_name, default_value)
    for field_name, default_value in _default_previous_accepted_day_context_meta().items():
        record_copy.setdefault(field_name, default_value)
    record_copy.setdefault("previous_accepted_day_diff", {})
    record_copy.setdefault("previous_accepted_day_judgement", {})
    record_copy.setdefault("contract_status", "unknown")
    record_copy.setdefault("schema_contract_errors", [])
    record_copy.setdefault("operational_contract_errors", [])
    record_copy.setdefault("production_eligible", False)
    record_copy.setdefault("production_decision_reason", "")
    record_copy.setdefault("production_unlock_reason", "")
    record_copy.setdefault("production_decision_detail", "")
    record_copy.setdefault("production_operator_detail", "")
    record_copy.setdefault("production_operator_hint", "")
    record_copy.setdefault("repro_rollout_mode", get_repro_rollout_mode())
    record_copy.setdefault(
        "operator_guidance_mode",
        OPERATOR_GUIDANCE_MODE_GUIDED,
    )
    record_copy.setdefault("guidance_degraded_reasons", [])
    record_copy.setdefault("run_id", build_acceptance_run_id(record_copy))
    record_copy.setdefault("verified_at", record_copy.get("timestamp"))
    record_copy.setdefault("baseline_source", record_copy.get("baseline_source", "unknown"))
    record_copy = attach_measurement_context_metadata(record_copy)
    record_copy = _apply_operator_guidance_quarantine(record_copy)
    search_root = (
        os.path.dirname(save_dir)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", os.path.basename(save_dir))
        else save_dir
    )
    control_sample_id = str(
        record_copy.get("control_sample_id", DEFAULT_CONTROL_SAMPLE_ID)
    )
    ref_record = None
    if not str(record_copy.get("accepted_reference_run_id", "")).strip():
        ref_record, selection_meta = resolve_accepted_reference_record(
            calibration_root=search_root or CALIBRATION_DIR,
            control_sample_id=control_sample_id,
            before_timestamp=str(record_copy.get("timestamp", "")),
            accepted_reference_selection_mode=record_copy.get(
                "accepted_reference_selection_mode",
                ACCEPTED_REFERENCE_SELECTION_MODE_AUTO,
            ),
            accepted_reference_selection_requested_run_id=record_copy.get(
                "accepted_reference_selection_requested_run_id",
            ),
            accepted_reference_selection_source_file=record_copy.get(
                "accepted_reference_selection_source_file",
            ),
            current_measurement_context=record_copy.get("measurement_context"),
        )
        record_copy.update(selection_meta)
        record_copy["accepted_reference_run_id"] = (
            "none"
            if ref_record is None
            else str(ref_record.get("run_id", build_acceptance_run_id(ref_record)))
        )
    elif (
        str(record_copy.get("accepted_reference_run_id", "")).strip().lower() != "none"
        and record_copy.get("reference_relation") == REFERENCE_RELATION_NONE
        and not record_copy.get("reference_is_same_day")
        and not record_copy.get("reference_is_same_session")
    ):
        ref_record = find_acceptance_record_by_run_id(
            calibration_root=search_root or CALIBRATION_DIR,
            run_id=record_copy.get("accepted_reference_run_id"),
            control_sample_id=control_sample_id,
        )
    if ref_record is not None:
        comparison = compare_measurement_context(
            record_copy.get("measurement_context"),
            ref_record,
        )
        record_copy.update(_accepted_reference_context_meta_from_comparison(comparison))
        if comparison.get("status") in (
            "current_context_unresolved",
            "reference_context_unresolved",
            "mismatch",
        ):
            if not record_copy.get("accepted_reference_selection_warning"):
                warning_map = {
                    "current_context_unresolved": (
                        "accepted_reference_current_context_unresolved"
                    ),
                    "reference_context_unresolved": (
                        "accepted_reference_reference_context_unresolved"
                    ),
                    "mismatch": "accepted_reference_context_mismatch",
                }
                record_copy["accepted_reference_selection_warning"] = warning_map.get(
                    comparison.get("status"),
                    "",
                )
            record_copy["accepted_reference_run_id"] = "none"
            ref_record = None
    if ref_record is not None:
        record_copy.update(describe_accepted_reference_relation(record_copy, ref_record))

    previous_accepted_day_record = None
    previous_accepted_day_run_id = str(
        record_copy.get("previous_accepted_day_run_id", "")
    ).strip()
    current_date = _acceptance_date_slug(record_copy.get("timestamp"))
    if (
        previous_accepted_day_run_id
        and previous_accepted_day_run_id.lower() != "none"
    ):
        candidate_record = find_acceptance_record_by_run_id(
            calibration_root=search_root or CALIBRATION_DIR,
            run_id=previous_accepted_day_run_id,
            control_sample_id=control_sample_id,
        )
        if candidate_record is None:
            record_copy["previous_accepted_day_selection_warning"] = (
                "previous_accepted_day_missing"
            )
            record_copy["previous_accepted_day_run_id"] = "none"
        elif _acceptance_date_slug(candidate_record.get("timestamp")) == current_date:
            record_copy["previous_accepted_day_selection_warning"] = (
                "previous_accepted_day_missing"
            )
            record_copy["previous_accepted_day_run_id"] = "none"
        else:
            comparison = compare_measurement_context(
                record_copy.get("measurement_context"),
                candidate_record,
            )
            record_copy.update(
                _previous_accepted_day_context_meta_from_comparison(comparison)
            )
            if comparison.get("status") == "match":
                previous_accepted_day_record = candidate_record
            else:
                warning_map = {
                    "current_context_unresolved": (
                        "previous_accepted_day_current_context_unresolved"
                    ),
                    "reference_context_unresolved": (
                        "previous_accepted_day_reference_context_unresolved"
                    ),
                    "mismatch": "previous_accepted_day_context_mismatch",
                }
                record_copy["previous_accepted_day_selection_warning"] = warning_map.get(
                    comparison.get("status"),
                    "",
                )
                record_copy["previous_accepted_day_run_id"] = "none"
    else:
        previous_accepted_day_record, previous_accepted_day_meta = (
            resolve_previous_accepted_day_record(
                calibration_root=search_root or CALIBRATION_DIR,
                control_sample_id=control_sample_id,
                before_timestamp=str(record_copy.get("timestamp", "")),
                current_measurement_context=record_copy.get("measurement_context"),
            )
        )
        record_copy.update(previous_accepted_day_meta)
        record_copy["previous_accepted_day_run_id"] = (
            "none"
            if previous_accepted_day_record is None
            else str(
                previous_accepted_day_record.get(
                    "run_id",
                    build_acceptance_run_id(previous_accepted_day_record),
                )
            )
        )
    if previous_accepted_day_record is not None:
        record_copy["previous_accepted_day_timestamp"] = str(
            previous_accepted_day_record.get("timestamp", "")
        )
        record_copy["previous_accepted_day_date"] = _acceptance_date_slug(
            previous_accepted_day_record.get("timestamp")
        )
    else:
        record_copy["previous_accepted_day_timestamp"] = ""
        record_copy["previous_accepted_day_date"] = ""
    record_copy["previous_accepted_day_diff"] = compute_previous_accepted_day_diff(
        record_copy,
        previous_accepted_day_record,
    )
    record_copy["previous_accepted_day_judgement"] = (
        evaluate_previous_accepted_day_judgement(
            record_copy.get("previous_accepted_day_diff")
        )
    )

    schema_contract_errors = _find_acceptance_schema_contract_errors(record_copy)
    operational_contract_errors = (
        _find_acceptance_operational_contract_errors(record_copy)
        if _is_production_relevant_acceptance_record(record_copy)
        else []
    )
    contract_errors = sorted(set(schema_contract_errors + operational_contract_errors))
    record_copy.update(
        _build_acceptance_contract_snapshot(
            record_copy,
            contract_errors=contract_errors,
            schema_contract_errors=schema_contract_errors,
            operational_contract_errors=operational_contract_errors,
        )
    )
    if contract_errors:
        record_copy["result_status"] = RESULT_STATUS_RECHECK_REQUIRED
        record_copy["gate_state"] = GATE_STATE_BLOCKED
    record_copy["result_status"] = normalize_result_status(record_copy.get("result_status"))
    record_copy["gate_state"] = normalize_gate_state(record_copy.get("gate_state"))

    ts = str(record_copy.get("timestamp"))
    ts_slug = re.sub(r"[^0-9A-Za-z]+", "", ts) or datetime.now().strftime("%Y%m%d%H%M%S")
    history_json = os.path.join(save_dir, f"acceptance_result_{ts_slug}.json")
    latest_json = os.path.join(save_dir, "acceptance_result.json")
    history_log = os.path.join(save_dir, f"acceptance_result_{ts_slug}.log")
    latest_log = os.path.join(save_dir, "acceptance_result.log")

    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(record_copy, f, indent=2, ensure_ascii=False)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(record_copy, f, indent=2, ensure_ascii=False)

    human_text = build_acceptance_result_log_text(record_copy)
    with open(history_log, "w", encoding="utf-8") as f:
        f.write(human_text)
    with open(latest_log, "w", encoding="utf-8") as f:
        f.write(human_text)

    return {
        "history_json": history_json,
        "latest_json": latest_json,
        "history_log": history_log,
        "latest_log": latest_log,
    }


def build_gray_4d_verification_log_text(record: dict) -> str:
    """4D検証の人間可読ログ本文を返す。"""
    rel = record.get("relative") or {}
    abs_result = record.get("absolute") or {}
    lines = [
        "4D verification audit",
        "judgement_target: grid 4D patch (SpyderCHECKR-48 4D, not Tar ROI)",
        f"timestamp: {record.get('timestamp', 'unknown')}",
        f"run_type: {record.get('run_type', RUN_TYPE_VERIFY_ONLY)}",
        f"operator: {record.get('operator', 'unknown')}",
        f"git_revision: {record.get('git_revision', 'unknown')}",
        f"calibration_dir: {record.get('calibration_dir', 'unknown')}",
        f"mode: {record.get('mode', 'unknown')}",
        f"roi_or_grid_source: {record.get('roi_or_grid_source', 'unknown')}",
        f"baseline_source: {record.get('baseline_source', 'none')}",
        f"baseline_value: {_format_triplet(record.get('baseline_value'))}",
        f"white_ratio_rgb: {_format_triplet(record.get('white_ratio_rgb'))}",
        f"avg_ref: {_format_triplet(record.get('avg_ref'))}",
        f"ref_scale: {record.get('ref_scale', 'N/A')}",
        f"steady_state: {record.get('steady_state', {})}",
        (
            "relative: "
            f"accuracy={rel.get('accuracy_status', 'N/A')} "
            f"stability={rel.get('stability_status', 'N/A')} "
            f"mean={_format_triplet(record.get('relative_rgb_mean'))} "
            f"std={_format_triplet(record.get('relative_rgb_std'))}"
        ),
        (
            "absolute: "
            f"status={abs_result.get('status', 'N/A')} "
            f"measured_lab={_format_triplet(record.get('measured_lab'))} "
            f"reference_lab={_format_triplet(abs_result.get('reference_lab'))} "
            f"delta_e={record.get('delta_e', 'N/A')}"
        ),
        f"exposure: {record.get('exposure', {})}",
        "samples:",
    ]
    for sample in record.get("samples", []):
        ref_scale = sample.get("ref_scale")
        ref_scale_str = "N/A" if ref_scale is None else f"{float(ref_scale):.6f}"
        lines.append(
            f"  grid4D#{int(sample.get('sample_index', 0)):02d} "
            f"corrected={_format_triplet(sample.get('corrected_ratios'))} "
            f"avg_ref={_format_triplet(sample.get('avg_ref'))} "
            f"ref_scale={ref_scale_str} "
            f"measured_lab={_format_triplet(sample.get('measured_lab'))} "
            f"delta_e={sample.get('delta_e', 'N/A')}"
        )
    return "\n".join(lines) + "\n"


def persist_gray_4d_verification_artifacts(
    record: dict,
    calibration_dir: str | None = None,
) -> dict:
    """4D検証の JSON と人間可読ログを履歴+latest の両方へ保存する。"""
    ensure_calibration_dir()
    save_dir = calibration_dir or get_today_calibration_dir()
    os.makedirs(save_dir, exist_ok=True)

    record_copy = _json_ready(dict(record))
    record_copy["run_type"] = normalize_acceptance_run_type(record_copy.get("run_type"))
    software_identity = get_software_identity()
    record_copy.setdefault("software_version", software_identity["software_version"])
    record_copy.setdefault("git_revision", software_identity["git_revision"])
    ts = str(record_copy.get("timestamp", datetime.now().isoformat(timespec="seconds")))
    ts_slug = re.sub(r"[^0-9A-Za-z]+", "", ts) or datetime.now().strftime("%Y%m%d%H%M%S")

    history_json = os.path.join(save_dir, f"gray_4d_verification_{ts_slug}.json")
    latest_json = os.path.join(save_dir, "gray_4d_verification_latest.json")
    history_log = os.path.join(save_dir, f"gray_4d_verification_{ts_slug}.log")
    latest_log = os.path.join(save_dir, "gray_4d_verification_latest.log")

    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(record_copy, f, indent=2, ensure_ascii=False)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(record_copy, f, indent=2, ensure_ascii=False)

    human_text = build_gray_4d_verification_log_text(record_copy)
    with open(history_log, "w", encoding="utf-8") as f:
        f.write(human_text)
    with open(latest_log, "w", encoding="utf-8") as f:
        f.write(human_text)

    return {
        "history_json": history_json,
        "latest_json": latest_json,
        "history_log": history_log,
        "latest_log": latest_log,
    }


GRAY_CARD_CHECK_SCHEMA_VERSION = 1
GRAY_CARD_CHECK_KIND = "gray_card_check"
GRAY_CARD_CHECK_REQUIRED_BASELINE_FIELDS = (
    "schema_version",
    "kind",
    "role",
    "timestamp",
    "baseline_id",
    "calibration_context_hash",
    "ratio_corrected",
    "ratio_white",
    "measured_lab",
    "lab_rel",
    "ref_scale",
    "avg_ref",
    "baseline_validation",
)
GRAY_CARD_CHECK_WARN_THRESHOLDS = {
    "dE": 1.0,
    "RGB": 0.010,
    "ref_scale": 0.050,
    "L": 1.0,
}
GRAY_CARD_CHECK_RECAL_THRESHOLDS = {
    "dE": 3.0,
    "RGB": 0.030,
    "ref_scale": 0.100,
    "L": 3.0,
}


def gray_card_check_thresholds() -> dict:
    """Gray Check quality record 用 thresholds dict を返す。"""
    return {
        "warn": dict(GRAY_CARD_CHECK_WARN_THRESHOLDS),
        "recal": dict(GRAY_CARD_CHECK_RECAL_THRESHOLDS),
    }


def _gray_card_check_array3(value) -> "np.ndarray | None":
    """Gray Check 用の3要素有限配列へ正規化する。"""
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size < 3:
        return None
    arr = arr[:3].astype(np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _gray_card_check_has_nonfinite(value) -> bool:
    """値が存在していて NaN/Inf を含む場合だけ True を返す。"""
    if value is None:
        return False
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return False
    if arr.size == 0:
        return False
    return bool(np.any(~np.isfinite(arr)))


def _gray_card_check_scalar(value) -> "float | None":
    """Gray Check 用の有限 scalar へ正規化する。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _gray_card_check_nested_get(record: dict, key: str):
    """top-level 優先、なければ capture_stats / quality から値を拾う。"""
    if not isinstance(record, dict):
        return None
    if key in record:
        return record.get(key)
    for section in ("capture_stats", "quality"):
        payload = record.get(section)
        if isinstance(payload, dict) and key in payload:
            return payload.get(key)
    return None


def _gray_card_check_channel_spread(value) -> "float | None":
    arr = _gray_card_check_array3(value)
    if arr is None:
        return None
    mean = float(np.mean(arr))
    if abs(mean) < 1e-8:
        return None
    return float(np.max(np.abs(arr / mean - 1.0)))


def compute_gray_card_check_context_hash(context: dict | None) -> str:
    """Gray Check baseline identity 用の安定 hash を返す。"""
    normalized = _json_ready(context if isinstance(context, dict) else {})
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def gray_card_check_timestamp_slug(now: "datetime | None" = None) -> str:
    """履歴 artifact 用の local timestamp slug を返す。"""
    dt = datetime.now() if now is None else now
    return dt.strftime("%Y%m%dT%H%M%S%f")


def _gray_card_check_fsync_dir(directory: str) -> None:
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text_file(path: str, text: str) -> None:
    """同一ディレクトリ tmp + fsync + os.replace で text を保存する。"""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    basename = os.path.basename(path)
    tmp_path = os.path.join(
        directory,
        f".{basename}.tmp.{os.getpid()}.{time.time_ns()}",
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _gray_card_check_fsync_dir(directory)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def atomic_write_json_file(path: str, record: dict) -> None:
    """同一ディレクトリ tmp + fsync + os.replace で JSON を保存する。"""
    text = json.dumps(_json_ready(record), indent=2, ensure_ascii=False) + "\n"
    atomic_write_text_file(path, text)


def _gray_card_check_next_stem(
    save_dir: str,
    base_stem: str,
    suffixes: tuple[str, ...] = (".json", ".log"),
) -> str:
    for idx in range(1000):
        stem = base_stem if idx == 0 else f"{base_stem}_{idx:02d}"
        if all(not os.path.exists(os.path.join(save_dir, stem + suffix)) for suffix in suffixes):
            return stem
    return f"{base_stem}_{time.time_ns()}"


def build_gray_card_check_log_text(record: dict) -> str:
    """Gray Check の人間可読ログを返す。"""
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    validation = (
        record.get("baseline_validation")
        if isinstance(record.get("baseline_validation"), dict)
        else {}
    )
    lines = [
        "Gray Card Check",
        f"timestamp: {record.get('timestamp', '')}",
        f"role: {record.get('role', '')}",
        f"baseline_id: {record.get('baseline_id', '')}",
        f"calibration_context_hash: {record.get('calibration_context_hash', '')}",
        f"status: {quality.get('status', 'N/A')}",
        f"status_reason: {quality.get('status_reason', quality.get('reason', 'N/A'))}",
        f"failed_axes: {quality.get('failed_axes', [])}",
        f"delta_e_from_baseline: {quality.get('delta_e_from_baseline', 'N/A')}",
        f"max_abs_delta_ratio_white: {quality.get('max_abs_delta_ratio_white', 'N/A')}",
        f"max_abs_delta_ratio_corrected: {quality.get('max_abs_delta_ratio_corrected', 'N/A')}",
        f"ref_scale_abs_diff: {quality.get('ref_scale_abs_diff', 'N/A')}",
        f"lab_rel_l_abs_diff: {quality.get('lab_rel_l_abs_diff', 'N/A')}",
        f"ratio_raw: {_format_triplet(record.get('ratio_raw'))}",
        f"ratio_corrected: {_format_triplet(record.get('ratio_corrected'))}",
        f"ratio_white: {_format_triplet(record.get('ratio_white'))}",
        f"measured_lab: {_format_triplet(record.get('measured_lab'))}",
        f"lab_rel: {_format_triplet(record.get('lab_rel'))}",
        f"ref_scale: {record.get('ref_scale', 'N/A')}",
        f"avg_ref: {_format_triplet(record.get('avg_ref'))}",
        f"baseline_validation_valid: {validation.get('valid', 'N/A')}",
        f"baseline_validation_reason: {validation.get('reason', 'N/A')}",
        f"capture_stats: {record.get('capture_stats', {})}",
        f"ref_drift_snapshot: {record.get('ref_drift_snapshot', {})}",
    ]
    return "\n".join(lines) + "\n"


def persist_gray_card_check_artifacts(
    record: dict,
    calibration_dir: str | None = None,
    *,
    write_baseline: bool = False,
    timestamp_slug: str | None = None,
) -> dict:
    """Gray Check artifact を履歴+latestへ保存し、必要なら baseline pointer も更新する。"""
    ensure_calibration_dir()
    save_dir = calibration_dir or get_today_calibration_dir()
    os.makedirs(save_dir, exist_ok=True)

    record_copy = _json_ready(dict(record))
    record_copy.setdefault("schema_version", GRAY_CARD_CHECK_SCHEMA_VERSION)
    record_copy.setdefault("kind", GRAY_CARD_CHECK_KIND)
    software_identity = get_software_identity()
    record_copy.setdefault("software_version", software_identity["software_version"])
    record_copy.setdefault("git_revision", software_identity["git_revision"])

    slug = str(timestamp_slug or record_copy.get("timestamp_slug") or "").strip()
    if not slug:
        slug = gray_card_check_timestamp_slug()
    slug = re.sub(r"[^0-9A-Za-zT]+", "", slug) or gray_card_check_timestamp_slug()

    history_stem = _gray_card_check_next_stem(save_dir, f"gray_card_check_{slug}")
    history_json = os.path.join(save_dir, f"{history_stem}.json")
    history_log = os.path.join(save_dir, f"{history_stem}.log")
    latest_json = os.path.join(save_dir, "gray_card_check_latest.json")
    latest_log = os.path.join(save_dir, "gray_card_check_latest.log")
    human_text = build_gray_card_check_log_text(record_copy)

    atomic_write_json_file(history_json, record_copy)
    atomic_write_text_file(history_log, human_text)
    atomic_write_json_file(latest_json, record_copy)
    atomic_write_text_file(latest_log, human_text)

    paths = {
        "history_json": history_json,
        "history_log": history_log,
        "latest_json": latest_json,
        "latest_log": latest_log,
    }

    if write_baseline:
        baseline_stem = _gray_card_check_next_stem(
            save_dir,
            f"gray_card_check_baseline_{slug}",
        )
        baseline_history_json = os.path.join(save_dir, f"{baseline_stem}.json")
        baseline_history_log = os.path.join(save_dir, f"{baseline_stem}.log")
        baseline_json = os.path.join(save_dir, "gray_card_check_baseline.json")
        atomic_write_json_file(baseline_history_json, record_copy)
        atomic_write_text_file(baseline_history_log, human_text)
        atomic_write_json_file(baseline_json, record_copy)
        paths.update(
            {
                "baseline_history_json": baseline_history_json,
                "baseline_history_log": baseline_history_log,
                "baseline_json": baseline_json,
            }
        )
    return paths


def load_gray_card_check_baseline(
    calibration_dir: str | None = None,
) -> "tuple[dict | None, str | None]":
    """最新 Gray Check baseline pointer を読み込む。"""
    save_dir = calibration_dir or get_today_calibration_dir()
    path = os.path.join(save_dir, "gray_card_check_baseline.json")
    if not os.path.exists(path):
        return None, "missing"
    try:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
    except json.JSONDecodeError:
        return None, "corrupt_json"
    except OSError:
        return None, "read_error"
    if not isinstance(record, dict):
        return None, "malformed_json"
    return record, None


def validate_gray_card_baseline_candidate(record: dict) -> dict:
    """Tar ROI の gray-card baseline 候補が保存可能かを判定する。"""
    failed: list[str] = []
    metrics: dict[str, float | None] = {}

    ratio_corrected = _gray_card_check_array3(record.get("ratio_corrected"))
    ratio_white = _gray_card_check_array3(record.get("ratio_white"))
    measured_lab = _gray_card_check_array3(record.get("measured_lab"))
    lab_rel = _gray_card_check_array3(record.get("lab_rel"))
    avg_ref = _gray_card_check_array3(record.get("avg_ref"))
    ref_scale = _gray_card_check_scalar(record.get("ref_scale"))

    for name, value in (
        ("ratio_corrected", ratio_corrected),
        ("ratio_white", ratio_white),
        ("measured_lab", measured_lab),
        ("lab_rel", lab_rel),
        ("avg_ref", avg_ref),
    ):
        if value is None:
            failed.append(f"{name}_missing_or_nonfinite")
    if ref_scale is None:
        failed.append("ref_scale_missing_or_nonfinite")

    for name, arr in (("ratio_corrected", ratio_corrected), ("ratio_white", ratio_white)):
        if arr is None:
            continue
        if not np.all(arr > 0.0):
            failed.append(f"{name}_not_positive")
        if bool(np.any(arr < 0.30)) or bool(np.any(arr > 3.00)):
            failed.append(f"{name}_outside_range")
        spread = _gray_card_check_channel_spread(arr)
        metrics[f"max_abs_channel_spread_{name}"] = spread
        if spread is None or spread > 0.20:
            failed.append(f"{name}_channel_spread")

    tar_ref_luma = _gray_card_check_scalar(
        _gray_card_check_nested_get(record, "tar_ref_luma_ratio_raw")
    )
    tar_luma_cv = _gray_card_check_scalar(
        _gray_card_check_nested_get(record, "tar_luma_cv")
    )
    tar_center_edge = _gray_card_check_scalar(
        _gray_card_check_nested_get(record, "tar_center_edge_abs_pct")
    )
    max_ratio_std = _gray_card_check_scalar(
        _gray_card_check_nested_get(record, "max_ratio_std")
    )
    metrics.update(
        {
            "tar_ref_luma_ratio_raw": tar_ref_luma,
            "tar_luma_cv": tar_luma_cv,
            "tar_center_edge_abs_pct": tar_center_edge,
            "max_ratio_std": max_ratio_std,
        }
    )

    reason = ""
    if tar_ref_luma is None:
        failed.append("tar_ref_luma_ratio_raw_missing")
    else:
        if tar_ref_luma > 1.60:
            failed.append("suspected_4d_or_wrong_gray")
            reason = "suspected_4d_or_wrong_gray"
        elif tar_ref_luma < 0.70 or tar_ref_luma > 1.30:
            failed.append("tar_ref_luma_ratio_raw_outside_range")

    if tar_luma_cv is None:
        failed.append("tar_luma_cv_missing")
    else:
        if tar_luma_cv > 0.050:
            failed.append("target_not_uniform_gray")
            reason = reason or "target_not_uniform_gray"
        elif tar_luma_cv > 0.035:
            failed.append("tar_luma_cv")

    if tar_center_edge is None:
        failed.append("tar_center_edge_abs_pct_missing")
    else:
        if tar_center_edge > 5.0:
            failed.append("target_not_uniform_gray")
            reason = reason or "target_not_uniform_gray"
        elif tar_center_edge > 3.0:
            failed.append("tar_center_edge_abs_pct")

    if max_ratio_std is None:
        failed.append("max_ratio_std_missing")
    elif max_ratio_std > 0.010:
        failed.append("max_ratio_std")

    drift_snapshot = record.get("ref_drift_snapshot")
    if isinstance(drift_snapshot, dict):
        drift_state = str(drift_snapshot.get("drift_state", "")).upper()
        guard_state = str(drift_snapshot.get("guard_state", "")).upper()
        if drift_state == "RECAL" or guard_state == "RECALIB_REQUIRED":
            failed.append("ref_drift_recal")
        clip_high = _gray_card_check_scalar(drift_snapshot.get("clip_high_pct"))
        clip_low = _gray_card_check_scalar(drift_snapshot.get("clip_low_pct"))
        if clip_high is not None and clip_high >= 0.10:
            failed.append("ref_clip_high_warn")
        if clip_low is not None and clip_low >= 0.10:
            failed.append("ref_clip_low_warn")

    if not reason and failed:
        reason = failed[0]
    return {
        "valid": not failed,
        "failed_checks": failed,
        "reason": reason or "ok",
        "metrics": metrics,
    }


def describe_gray_card_check_baseline_state(
    record: dict | None,
    expected_context_hash: str | None = None,
) -> dict:
    """読み込んだ baseline が比較に使えるかを返す。"""
    if not isinstance(record, dict):
        return {"usable": False, "reason": "missing"}
    if record.get("schema_version") != GRAY_CARD_CHECK_SCHEMA_VERSION:
        return {"usable": False, "reason": "schema_mismatch"}
    if record.get("kind") != GRAY_CARD_CHECK_KIND:
        return {"usable": False, "reason": "kind_mismatch"}
    if record.get("role") != "baseline":
        return {"usable": False, "reason": "role_mismatch"}
    missing = [
        field_name
        for field_name in GRAY_CARD_CHECK_REQUIRED_BASELINE_FIELDS
        if field_name not in record
    ]
    if missing:
        return {"usable": False, "reason": "missing_field", "missing_fields": missing}
    if expected_context_hash and record.get("calibration_context_hash") != expected_context_hash:
        return {"usable": False, "reason": "calibration_context_mismatch"}
    for field_name in ("ratio_corrected", "ratio_white", "measured_lab", "lab_rel", "avg_ref"):
        if _gray_card_check_array3(record.get(field_name)) is None:
            return {"usable": False, "reason": f"{field_name}_nonfinite"}
    if _gray_card_check_scalar(record.get("ref_scale")) is None:
        return {"usable": False, "reason": "ref_scale_nonfinite"}
    validation = record.get("baseline_validation")
    if not isinstance(validation, dict) or validation.get("valid") is not True:
        return {"usable": False, "reason": "baseline_validation_failed"}
    return {"usable": True, "reason": "ok"}


def compare_gray_card_check_to_baseline(
    current: dict,
    baseline: dict,
    *,
    ref_drift_snapshot: dict | None = None,
) -> dict:
    """現在の Gray Check 結果を baseline と比較して OK/WARN/RECAL を返す。"""
    required_arrays = ("ratio_white", "ratio_corrected", "measured_lab", "lab_rel")
    missing: list[str] = []
    current_arrays = {}
    baseline_arrays = {}
    for field_name in required_arrays:
        if (
            _gray_card_check_has_nonfinite(current.get(field_name))
            or _gray_card_check_has_nonfinite(baseline.get(field_name))
        ):
            return {
                "status": "RECAL",
                "status_reason": "nonfinite_metric",
                "failed_axes": ["nonfinite_metric"],
                "thresholds": gray_card_check_thresholds(),
            }
        current_arrays[field_name] = _gray_card_check_array3(current.get(field_name))
        baseline_arrays[field_name] = _gray_card_check_array3(baseline.get(field_name))
        if current_arrays[field_name] is None or baseline_arrays[field_name] is None:
            missing.append(field_name)
    if (
        _gray_card_check_has_nonfinite(current.get("ref_scale"))
        or _gray_card_check_has_nonfinite(baseline.get("ref_scale"))
    ):
        return {
            "status": "RECAL",
            "status_reason": "nonfinite_metric",
            "failed_axes": ["nonfinite_metric"],
            "thresholds": gray_card_check_thresholds(),
        }
    current_ref_scale = _gray_card_check_scalar(current.get("ref_scale"))
    baseline_ref_scale = _gray_card_check_scalar(baseline.get("ref_scale"))
    if current_ref_scale is None or baseline_ref_scale is None:
        missing.append("ref_scale")
    if missing:
        return {
            "status": "WARN",
            "status_reason": "missing_metric",
            "failed_axes": ["missing_metric"],
            "missing_metrics": sorted(set(missing)),
            "delta_e_from_baseline": None,
            "max_abs_delta_ratio_white": None,
            "max_abs_delta_ratio_corrected": None,
            "ref_scale_abs_diff": None,
            "lab_rel_l_abs_diff": None,
            "thresholds": gray_card_check_thresholds(),
        }

    if not all(
        np.all(np.isfinite(arr))
        for arr in list(current_arrays.values()) + list(baseline_arrays.values())
    ):
        return {
            "status": "RECAL",
            "status_reason": "nonfinite_metric",
            "failed_axes": ["nonfinite_metric"],
            "thresholds": gray_card_check_thresholds(),
        }

    delta_e = float(
        np.linalg.norm(current_arrays["measured_lab"] - baseline_arrays["measured_lab"])
    )
    max_abs_delta_ratio_white = float(
        np.max(np.abs(current_arrays["ratio_white"] - baseline_arrays["ratio_white"]))
    )
    max_abs_delta_ratio_corrected = float(
        np.max(
            np.abs(
                current_arrays["ratio_corrected"]
                - baseline_arrays["ratio_corrected"]
            )
        )
    )
    ref_scale_abs_diff = float(abs(current_ref_scale - baseline_ref_scale))
    lab_rel_l_abs_diff = float(
        abs(current_arrays["lab_rel"][0] - baseline_arrays["lab_rel"][0])
    )
    if not all(
        math.isfinite(value)
        for value in (
            delta_e,
            max_abs_delta_ratio_white,
            max_abs_delta_ratio_corrected,
            ref_scale_abs_diff,
            lab_rel_l_abs_diff,
        )
    ):
        return {
            "status": "RECAL",
            "status_reason": "nonfinite_metric",
            "failed_axes": ["nonfinite_metric"],
            "thresholds": gray_card_check_thresholds(),
        }

    axis_values = {
        "dE": delta_e,
        "RGB": max(max_abs_delta_ratio_white, max_abs_delta_ratio_corrected),
        "ref_scale": ref_scale_abs_diff,
        "L": lab_rel_l_abs_diff,
    }
    status = "OK"
    failed_axes: list[str] = []
    recal_axes: list[str] = []
    warn_axes: list[str] = []
    for axis in ("dE", "RGB", "ref_scale", "L"):
        value = axis_values[axis]
        if value >= GRAY_CARD_CHECK_RECAL_THRESHOLDS[axis]:
            recal_axes.append(axis)
            failed_axes.append(axis)
        elif value >= GRAY_CARD_CHECK_WARN_THRESHOLDS[axis]:
            warn_axes.append(axis)
            failed_axes.append(axis)
    if recal_axes:
        status = "RECAL"
        status_reason = recal_axes[0]
    elif warn_axes:
        status = "WARN"
        status_reason = warn_axes[0]
    else:
        status_reason = "ok"

    snapshot = ref_drift_snapshot
    if snapshot is None and isinstance(current.get("ref_drift_snapshot"), dict):
        snapshot = current.get("ref_drift_snapshot")
    if isinstance(snapshot, dict):
        drift_state = str(snapshot.get("drift_state", "")).upper()
        guard_state = str(snapshot.get("guard_state", "")).upper()
        if drift_state in {"WARN", "RECAL"} or guard_state in {
            "LED_DRIFT_WARN",
            "RECALIB_REQUIRED",
        }:
            if "ref_drift" not in failed_axes:
                failed_axes.append("ref_drift")
            if status == "OK":
                status = "WARN"
                status_reason = "ref_drift"

    return {
        "status": status,
        "status_reason": status_reason,
        "failed_axes": failed_axes,
        "delta_e_from_baseline": delta_e,
        "max_abs_delta_ratio_white": max_abs_delta_ratio_white,
        "max_abs_delta_ratio_corrected": max_abs_delta_ratio_corrected,
        "ref_scale_abs_diff": ref_scale_abs_diff,
        "lab_rel_l_abs_diff": lab_rel_l_abs_diff,
        "thresholds": gray_card_check_thresholds(),
    }


def compute_gain_percentile(
    gain_map: "np.ndarray",
    percentile: float,
    roi: "tuple[int, int, int, int] | None" = None,
    mask: "np.ndarray | None" = None,
) -> float:
    """flat gain map の percentile 値を返す。

    Args:
        gain_map: flat gain map。
        percentile: 求める percentile 値。
        roi: `(x, y, w, h)` の ROI。`None` なら全体を対象にする。
            空 ROI や画面外 ROI の場合は安全な既定値 `1.0` を返す。
        mask: `gain_map` と同サイズの boolean mask。指定時は mask 内画素のみを対象にする。
    Returns:
        指定領域における gain percentile 値。
        `gain_map` が空、ROI が空、または対象画素が空なら `1.0`。
    """
    gain_map_f = np.asarray(gain_map, dtype=np.float64)
    if gain_map_f.size == 0:
        return 1.0

    mask_f = None
    if mask is not None:
        mask_f = np.asarray(mask, dtype=bool)
        if mask_f.shape != gain_map_f.shape:
            return 1.0

    if roi is None:
        roi_gain = gain_map_f
        roi_mask = mask_f
    else:
        x, y, w, h = roi
        x0 = max(int(x), 0)
        y0 = max(int(y), 0)
        x1 = min(int(x + w), gain_map_f.shape[1])
        y1 = min(int(y + h), gain_map_f.shape[0])
        if x1 <= x0 or y1 <= y0:
            return 1.0
        roi_gain = gain_map_f[y0:y1, x0:x1]
        roi_mask = None if mask_f is None else mask_f[y0:y1, x0:x1]

    if roi_gain.size == 0:
        return 1.0
    if roi_mask is not None:
        roi_gain = roi_gain[roi_mask]
    if roi_gain.size == 0:
        return 1.0
    return float(np.percentile(roi_gain, percentile))


# ===========================================================================
# キャリブレーションファイル管理
# ===========================================================================

CALIBRATION_DIR = "calibration"
"""キャリブレーションファイルの保存先ディレクトリ名。"""


def get_today_calibration_dir() -> str:
    """今日の日付でキャリブレーションサブディレクトリのパスを返す (calibration/YYYY-MM-DD)。"""
    date_override = os.environ.get("PICOLOR_CALIBRATION_DATE", "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_override):
        date_part = date_override
    else:
        date_part = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(CALIBRATION_DIR, date_part)


def _get_sorted_date_dirs() -> list:
    """calibration/ 配下の YYYY-MM-DD 形式ディレクトリを新しい順で返す。"""
    if not os.path.isdir(CALIBRATION_DIR):
        return []
    dirs = []
    for name in os.listdir(CALIBRATION_DIR):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", name):
            path = os.path.join(CALIBRATION_DIR, name)
            if os.path.isdir(path):
                dirs.append(path)
    dirs.sort(reverse=True)
    return dirs


def _cleared_marker_path(filename: str) -> str:
    """クリア済みマーカーファイルのパスを返す（CALIBRATION_DIR 直下）。"""
    return os.path.join(CALIBRATION_DIR, filename + ".cleared")


def _find_calibration_file(filename: str) -> Optional[str]:
    """最新日付ディレクトリ → 旧 calibration/ ルートの順でファイルを検索する。

    CALIBRATION_DIR 直下に filename.cleared マーカーが存在する場合は
    クリア済みとみなして None を返す（旧日付ファイルの復活を防ぐ）。
    見つからなければ None を返す。
    """
    if os.path.exists(_cleared_marker_path(filename)):
        return None
    for date_dir in _get_sorted_date_dirs():
        path = os.path.join(date_dir, filename)
        if os.path.exists(path):
            return path
    root_path = os.path.join(CALIBRATION_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    return None


def _delete_calibration_file_all(filename: str) -> None:
    """最新日付ディレクトリ + calibration/ ルートからファイルを削除し、
    クリア済みマーカーを置く。

    旧日付ディレクトリのファイルは削除しない（履歴保持）。
    accepted_reference / acceptance_result 系の証跡は対象外。
    マーカーにより _find_calibration_file が旧日付ファイルを返さなくなる。
    """
    date_dirs = _get_sorted_date_dirs()
    if date_dirs:
        path = os.path.join(date_dirs[0], filename)
        if os.path.exists(path):
            os.remove(path)
    root_path = os.path.join(CALIBRATION_DIR, filename)
    if os.path.exists(root_path):
        os.remove(root_path)
    # クリア済みマーカーを作成（旧日付ファイルの復活防止）
    marker = _cleared_marker_path(filename)
    with open(marker, "w") as _f:
        pass


def _remove_cleared_marker(filename: str) -> None:
    """クリア済みマーカーを削除する。キャリブレーションを保存した後に呼ぶ。"""
    marker = _cleared_marker_path(filename)
    if os.path.exists(marker):
        os.remove(marker)


def ensure_calibration_dir() -> None:
    """キャリブレーションディレクトリを作成し、旧ファイルを自動マイグレーションする。"""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    os.makedirs(get_today_calibration_dir(), exist_ok=True)
    # 旧ファイル（ルート直下）が残っていれば calibration/ へ移動する
    _LEGACY_FILES = [
        "dark_frame.npy",
        "flat_field_gain.npy",
        "wb_gains.json",
        "neutral_correction.json",
        "roi_config.json",
    ]
    import shutil

    for fname in _LEGACY_FILES:
        src = fname
        dst = os.path.join(CALIBRATION_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"- マイグレーション: {src} → {dst}")


# ===========================================================================
# ユーティリティ関数
# ===========================================================================


def _ascii_to_fullwidth(text: str) -> str:
    """ASCII印字可能文字(0x21-0x7E)を全角に変換する。CJKフォントの文字化け防止用。"""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x21 <= cp <= 0x7E:
            out.append(chr(cp + 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def _find_japanese_font(size: int = 14):
    """日本語フォントを検索して PIL ImageFont を返す。見つからなければ None。

    RPi OS Trixie / Debian Trixie・Bookworm / macOS の主要パスを網羅する。
    静的候補で見つからない場合は fc-list で動的検索する。
    """
    if not _PIL_AVAILABLE:
        return None
    candidates = [
        # Debian Trixie / Bookworm — fonts-noto-cjk (opentype)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        # Debian truetype 配置（旧パッケージ互換）
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        # fonts-takao
        "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
        # fonts-vlgothic
        "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
        "/usr/share/fonts/truetype/vlgothic/VL-PGothic-Regular.ttf",
        # RPi OS 標準同梱 — fonts-motoya-l-cedar / fonts-motoya-l-maruberi
        "/usr/share/fonts/truetype/motoya-l-cedar/MTLc3m.ttf",
        "/usr/share/fonts/truetype/motoya-l-maruberi/MTLmr3m.ttf",
        # fonts-ipafont / fonts-ipaexfont
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Supplemental/Osaka.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # fc-list フォールバック: CJK 対応フォントを動的検索
    try:
        import subprocess
        result = subprocess.run(
            ["fc-list", ":lang=ja", "file"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            fpath = line.split(":")[0].strip()
            if fpath and os.path.isfile(fpath):
                try:
                    return ImageFont.truetype(fpath, size)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _configure_qt_fontdir() -> None:
    """Qt HighGUI向けにフォントディレクトリを設定する。"""
    _ensure_qt_fontdir_env()


# ===========================================================================
# 設定データクラス
# ===========================================================================


@dataclass
class DisplaySettings:
    """
    表示・描画に関する設定値を保持する。

    機能:
      - 画面解像度、反転表示、表示色モード、ライブ補正表示などの表示挙動を定義する。
    入力:
      - 呼び出し元が与える表示設定値（幅、高さ、反転有無、保存フラグ）。
    出力:
      - 描画処理と表示処理で参照される表示設定オブジェクトを提供する。
    """

    DISPLAY_COLOR_MODE_NATURAL: ClassVar[str] = "natural"
    DISPLAY_COLOR_MODE_LEGACY: ClassVar[str] = "legacy"
    DISPLAY_COLOR_MODE_ALLOWED: ClassVar[tuple[str, str]] = (
        DISPLAY_COLOR_MODE_NATURAL,
        DISPLAY_COLOR_MODE_LEGACY,
    )

    width: int = 800
    height: int = 600
    flip_horizontal: bool = True
    flip_vertical: bool = True
    save_movie: bool = False
    # natural: ISP AWB中心の自然表示, legacy: 固定ColourGains+従来補正
    display_color_mode: str = "natural"
    # natural運用では既定でFalse（表示補正は原則OFF）
    enable_live_color_correction: bool = False
    show_target_delta_e: bool = False

    def __post_init__(self):
        """display_color_mode を許容値へ正規化する。"""
        mode = str(self.display_color_mode).strip().lower()
        if mode not in self.DISPLAY_COLOR_MODE_ALLOWED:
            mode = self.DISPLAY_COLOR_MODE_LEGACY
        self.display_color_mode = mode


@dataclass
class ProcessingSettings:
    """
    ROI位置・サイズなど処理系の設定値を保持する。

    機能:
      - Ref/Target ROIの位置、サイズ、アスペクト比、品質閾値を管理する。
    入力:
      - 呼び出し元が指定するROI関連パラメータと品質判定閾値。
    出力:
      - ROI抽出、品質チェック、UI描画が参照する処理設定オブジェクトを提供する。
    """

    spot_size_ref: int = 100
    spot_size_tar: int = 60
    aspect_ref: float = 1.0
    aspect_tar: float = 1.0
    posi_ref: list = field(default_factory=lambda: [300, 300])
    posi_tar: list = field(default_factory=lambda: [600, 300])
    center_edge_warn_threshold: float = 0.03  # Non-uniform警告閾値(fraction)


@dataclass
class MeasurementSettings:
    """
    計測・安定性評価・ログ出力に関する設定値を保持する。

    機能:
      - ガンマ、安定判定閾値、反射率係数、ログ設定など計測条件を定義する。
    入力:
      - 呼び出し元が与える計測条件（バッファ長、閾値、出力先、目標Lab）。
    出力:
      - 測色計算、安定性評価、ログ記録で参照される計測設定を提供する。
    """

    gamma: float = 2.2
    temporal_buffer_size: int = 60
    stability_cv_threshold: float = 0.001
    stability_sigma_threshold: float = 0.05
    reflectance_factor: float = 0.18
    csv_log_interval: int = 30
    enable_interval_csv_log: bool = False
    csv_log_dir: str = ""
    target_Lab: tuple = (50.0, 0.0, 0.0)


@dataclass
class PanelLayoutSettings:
    """
    右サイドパネルのレイアウト調整用パラメータ。

    機能:
      - フォントサイズ、余白、行間、グラフサイズなどUIレイアウトを定義する。
    入力:
      - 呼び出し元が調整する描画レイアウト値（フォント、座標、間隔、寸法）。
    出力:
      - パネル描画ロジックが使用するレイアウト設定オブジェクトを提供する。
    """

    # --- フォントスケール (cv2 fontScale) ---
    font_section_title: float = 0.8  # セクション見出し（Measurement, Ref Status ...）
    font_body: float = 1.1  # 本文（L,a,b / RAW / Non-uniform / Meas.OK）
    lab_outline_thickness: int = 10  # L,a,b 白縁の太さ（0で縁なし）
    font_warning: float = 0.35  # 警告テキスト（Attention セクション内）
    font_graph_label: float = 0.5  # グラフ内ラベル・数値

    # --- 配置 (px) ---
    margin_top: int = 20  # 上マージン（Measurement タイトル位置）
    margin_bot: int = 10  # 下マージン
    hdr_gap_text: int = 40  # タイトル → テキスト間隔（Measurement, Attention）
    hdr_gap_graph: int = 18  # タイトル → グラフ間隔（Ref Exposure, Illum. Stability）
    row_spacing: int = 40  # セクション内の行間（L→a→b, RAW→Non-uniform 等）
    warning_row_spacing: int = 16  # 警告テキストの行間

    # --- Ref ROI 直下の注釈 ---
    font_roi_annotation: float = 0.7  # ROI直下 RAW / NU のフォントスケール
    roi_annotation_spacing: int = 20  # ROI直下 RAW → NU の行間 (px)

    # --- グラフ (px) ---
    hist_height: int = 72  # ヒストグラム高さ（Ref Exposure）
    ts_height: int = 128  # 時系列グラフ＋状態ストリップ高さ（Ref Stability）
    drift_height: int = 72  # 補正グラフ＋値ストリップ高さ（Correction Applied）
    graph_width: int = 250  # グラフ横幅

    mode_bar_top: int = 5  # モード帯の画面上辺からの絶対位置(px)
    mode_bar_height: int = 26  # モード表示帯の高さ
    mode_bar_gap: int = 14  # モード帯→Measurement間の余白


# ===========================================================================
# モード管理
# ===========================================================================


class MeasurementMode:
    """
    Lab / LinearRGB デュアルモードの状態管理クラス。

    機能:
      - 測定モード（Lab / LinearRGB）の保持と切替を提供する。
      - 初期モードは Lab。TABキー等で切替可能。
    入力:
      - 外部からの toggle() 呼び出し。
    出力:
      - 現在のモード文字列、各モード判定 bool を返す。
    """

    MODE_LAB = "Lab"
    MODE_LINEAR = "LinearRGB"

    def __init__(self):
        """初期モードを Lab に設定する。"""
        self._mode = self.MODE_LAB

    @property
    def current(self) -> str:
        """現在のモード文字列を返す。"""
        return self._mode

    def toggle(self) -> str:
        """
        Lab ↔ LinearRGB を切り替え、新モード文字列を返す。

        Returns:
          str: 切替後のモード文字列。
        """
        if self._mode == self.MODE_LAB:
            self._mode = self.MODE_LINEAR
        else:
            self._mode = self.MODE_LAB
        return self._mode

    def is_lab(self) -> bool:
        """現在モードが Lab かどうかを返す。"""
        return self._mode == self.MODE_LAB

    def is_linear(self) -> bool:
        """現在モードが LinearRGB かどうかを返す。"""
        return self._mode == self.MODE_LINEAR


# ===========================================================================
# システム系
# ===========================================================================


class OSDetector:
    """
    OS種別判定のユーティリティ。

    機能:
      - 実行環境がUnix系かどうかを判定し、分岐条件を提供する。
    入力:
      - `os.name` など実行中Python環境が保持するOS情報。
    出力:
      - OS判定結果（Unix系ならTrue、非Unix系ならFalse）。
    """

    @staticmethod
    def is_unix_like():
        """
        OSがUnix系かどうかを判定する。

        Returns:
          bool: Unix系ならTrue。
        """
        return os.name == "posix"


class UserInfo:
    """
    現在のユーザー名取得を担当する。

    機能:
      - OSごとの手段でログインユーザー名を取得する。
    入力:
      - システムコマンド出力またはOS APIから得られるユーザー情報。
    出力:
      - 現在ユーザー名（文字列）。
    """

    def __init__(self):
        """初期化（状態は保持しない）。"""
        pass

    def _execute_shell_command(self, cmd, remove_line_feed=True):
        """
        シェルコマンドを実行して標準出力を取得する。

        Args:
          cmd: コマンド文字列。
          remove_line_feed: 改行の除去有無。
        Returns:
          list: 出力行のリスト。
        """
        output = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, shell=True
        ).stdout.readlines()
        if remove_line_feed:
            output = [str(x).rstrip("\n") for x in output]
        return output

    def _get_unix_user(self):
        """
        Unix系で現在のユーザー名を取得する。

        Returns:
          str: ユーザー名。
        """
        cmd = "who"
        output = self._execute_shell_command(cmd)
        unique_users = list(
            set([re.sub(r"\s+", " ", s).split(" ")[0][2:] for s in output])
        )
        return unique_users[0]

    def _get_windows_user(self):
        """
        Windowsで現在のユーザー名を取得する。

        Returns:
          str: ユーザー名。
        """
        return os.getlogin()

    def get_current_user(self):
        """
        OSに応じて現在のユーザー名を取得する。

        Returns:
          str: ユーザー名。
        """
        if OSDetector.is_unix_like():
            return self._get_unix_user()
        return self._get_windows_user()


class PathResolver:
    """
    保存先パスを解決するユーティリティ。

    機能:
      - ユーザー名とOS種別から標準的な保存先ディレクトリを生成する。
    入力:
      - `UserInfo` が返すユーザー名と `OSDetector` によるOS判定結果。
    出力:
      - 動画・結果保存で利用するディレクトリパス文字列。
    """

    def __init__(self, user_info: UserInfo):
        """UserInfoを保持する。"""
        self.user_info = user_info

    def get_video_directory(self):
        """
        動画保存先ディレクトリを生成する。

        Returns:
          str: 保存先パス。
        """
        user = self.user_info.get_current_user()
        if OSDetector.is_unix_like():  # Linux/Macの場合
            root = os.path.join("/home", user, "picolor", "results")
        else:  # Windowsの場合
            root = os.path.join("C:\\", "picolor", "results")
        return root


class SystemConfig:
    """
    OS判定・ユーザー取得・パス解決の窓口クラス。

    機能:
      - OS判定、ユーザー取得、保存先解決のヘルパーを統合して提供する。
    入力:
      - システム情報、ユーザー情報、外部から渡されるシェルコマンド文字列。
    出力:
      - OS判定結果、ユーザー名、保存先パス、シェル実行結果を返す。
    """

    def __init__(self):
        """内部ヘルパーを初期化する。"""
        self.os_detector = OSDetector()
        self.user_info = UserInfo()
        self.path_resolver = PathResolver(self.user_info)

    def execute_shell_command(self, cmd, remove_line_feed=True):
        """
        シェルコマンドを実行する。

        Args:
          cmd: コマンド文字列。
          remove_line_feed: 改行の除去有無。
        Returns:
          list: 出力行のリスト。
        """
        return self.user_info._execute_shell_command(
            cmd, remove_line_feed=remove_line_feed
        )

    def is_unix_like(self):
        """OSがUnix系かどうかを返す。"""
        return self.os_detector.is_unix_like()

    def get_current_user(self):
        """現在のユーザー名を返す。"""
        return self.user_info.get_current_user()

    def get_result_directory(self):
        """動画保存先ディレクトリを返す。"""
        return self.path_resolver.get_video_directory()


# ===========================================================================
# トースト描画
# ===========================================================================


def draw_mode_toast(frame, message: str, elapsed_sec: float) -> None:
    """
    モード切替時のトースト通知を画面下部に描画する。

    Args:
      frame: 描画対象のBGRフレーム (numpy配列)。
      message: 表示メッセージ文字列。
      elapsed_sec: モード切替からの経過秒数。
    """
    if elapsed_sec >= 1.5:
        return
    h, w = frame.shape[:2]
    bar_h = 40
    bar_y = h - bar_h - 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)
    alpha = max(0.0, min(1.0, 1.0 - (elapsed_sec / 1.5)))
    cv2.addWeighted(overlay, alpha * 0.8, frame, 1.0 - alpha * 0.8, 0, frame)
    if alpha > 0.05:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
        tx = (w - tw) // 2
        ty = bar_y + (bar_h + th) // 2
        color = (255, 255, 255)
        cv2.putText(
            frame, message, (tx, ty), font, scale, color, thickness, cv2.LINE_AA
        )


# ===========================================================================
# 色彩科学
# ===========================================================================


class DoubleBeamProcessor:
    """
    ダブルビーム法で照明強度変動をキャンセルする比率計算クラス。

    機能:
      - RefとTargetのリニアRGB平均からチャンネル別比率を算出する。
      - 露出確認用にROIの8bitチャンネル平均を算出する。
    入力:
      - リニア空間のRef/Target平均RGB配列、またはROIのBGR画像配列。
    出力:
      - 相対反射率比（ratio配列）および8bitチャンネル平均値を返す。
    """

    @staticmethod
    def compute_ratio(
        ref_linear_mean: np.ndarray, tar_linear_mean: np.ndarray
    ) -> np.ndarray:
        """
        チャンネル別の比率を算出。ratio = target / reference

        Args:
          ref_linear_mean: (3,) [R, G, B] のRefリニア平均値。
          tar_linear_mean: (3,) [R, G, B] のTargetリニア平均値。
        Returns:
          np.ndarray: (3,) [R, G, B] の相対反射率比。
        """
        safe_ref = np.where(ref_linear_mean < 1e-8, 1e-8, ref_linear_mean)
        return tar_linear_mean / safe_ref

    @staticmethod
    def compute_raw_8bit_means(roi_bgr_u8: np.ndarray) -> np.ndarray:
        """
        ROIのチャンネル別8bit平均値を算出（露出確認ダッシュボード用）。

        Args:
          roi_bgr_u8: (H, W, 3) のBGR画像配列。
        Returns:
          np.ndarray: (3,) [B_mean, G_mean, R_mean]。
        """
        return roi_bgr_u8.mean(axis=(0, 1)).astype(np.float32)

    @staticmethod
    def validate_ratio(ratio: "np.ndarray", max_val: float = 10.0) -> "np.ndarray":
        """
        ratio配列の異常値をガードする。

        NaN/Infを0.0に置換し、0〜max_val範囲にクリップする。
        物理的に反射率比>10は異常とみなす。

        Args:
          ratio: (3,) [R, G, B] の比率配列。
          max_val: クリップ上限値。
        Returns:
          np.ndarray: ガード済みratio配列。
        """
        import numpy as _np

        safe = _np.where(_np.isfinite(ratio), ratio, 0.0)
        return _np.clip(safe, 0.0, max_val).astype(_np.float32)


class CIELABConverter:
    """
    リニアRGB比率 → XYZ → CIELAB (float32実数スケール) 変換。
    OpenCVのcvtColor(BGR2Lab)は使用しない（二重リニア化の回避）。

    機能:
      - リニアRGB比率を反射率係数でスケーリングし、XYZ経由でLabへ変換する。
    入力:
      - 反射率係数と、比率化済みRGB配列 `[R, G, B]`。
    出力:
      - CIELAB値 `[L*, a*, b*]` を `np.ndarray(float32)` で返す。

    ⚠ 制限事項:
      - sRGB 色空間（IEC 61966-2-1）の分光感度を仮定している。
        実際の IMX477 センサーの分光感度とは異なるため、CCM で補正が必要。
      - D65 標準光源（CIE）の白色点を使用している。
        LED 照明の実スペクトルとは異なるが、ダブルビーム比で相殺される部分が大きい。
      - これらの仮定は CCM + 残差補正で実質的に吸収されるが、
        CCM 訓練データと大きく異なる色域では精度が低下する可能性がある。

    検証: ratio=(1,1,1), reflectance=0.18 の場合
      Y = 0.2126729*0.18 + 0.7151522*0.18 + 0.0721750*0.18 = 0.18
      L* = 116 * 0.18^(1/3) - 16 = 116 * 0.5646 - 16 ≈ 49.5
      a* ≈ 0, b* ≈ 0  (無彩色)
    """

    # sRGB → XYZ (D65) 変換行列 — IEC 61966-2-1
    _M_SRGB_TO_XYZ = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )

    # D65 白色点
    _XN = 0.95047
    _YN = 1.00000
    _ZN = 1.08883

    # CIELAB f関数の定数
    _DELTA = 6.0 / 29.0
    _DELTA_CU = (6.0 / 29.0) ** 3
    _INV_3_DELTA_SQ = 1.0 / (3.0 * (6.0 / 29.0) ** 2)
    _OFFSET = 4.0 / 29.0

    def __init__(self, reflectance_factor: float = 0.18):
        """
        反射率係数を設定する。

        Args:
          reflectance_factor: 灰色カード相当の反射率。
        """
        self.reflectance_factor = reflectance_factor
        self._diag_correction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self._diag_correction_calibrated = False
        # Phase 0: 動的Ref補正（LED経時ドリフト対策）
        self._ref_baseline_rgb: np.ndarray | None = (
            None  # キャリブレーション時の基準Ref RGB
        )
        self._dynamic_correction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        # Ref RGB の指数移動平均（EMA）: 単フレームノイズの平滑化
        self._ref_ema: np.ndarray | None = None
        self._ref_ema_alpha: float = 0.02  # 実効窓 ~50フレーム
        # 3×3 Color Correction Matrix（CCM）: 対角補正の上位互換
        self._ccm: np.ndarray | None = None
        # 3×10 多項式 CCM: 3×3 CCM の上位互換（非線形センサー応答を補正）
        self._poly_ccm: np.ndarray | None = None
        # パッチ別 ΔLab 残差補正テーブル
        self._residual_ratios: np.ndarray | None = None
        self._residual_labs: np.ndarray | None = None

    def _f(self, t: float) -> float:
        """CIELAB f関数: 立方根 or 線形近似"""
        if t > self._DELTA_CU:
            return t ** (1.0 / 3.0)
        else:
            return self._INV_3_DELTA_SQ * t + self._OFFSET

    def set_diagonal_correction(self, d_R: float, d_B: float) -> None:
        """対角補正行列 D=diag(d_R, 1.0, d_B) を設定する。

        センサー固有の分光感度ずれを補償するため、
        reflectance の R,B チャンネルを G 基準にスケーリングする。

        Args:
          d_R: R チャンネルの補正係数。
          d_B: B チャンネルの補正係数。
        """
        self._diag_correction = np.array([d_R, 1.0, d_B], dtype=np.float64)
        self._diag_correction_calibrated = True

    def clear_diagonal_correction(self) -> None:
        """対角補正を無効化して初期状態に戻す。"""
        self._diag_correction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self._diag_correction_calibrated = False

    def set_ref_baseline(self, ref_rgb: np.ndarray) -> None:
        """動的補正（Phase 0）の基準 Ref RGB を記録する。

        ニュートラルキャリブレーション完了時に呼び出す。
        以降の update_dynamic_correction がこの基準からのずれを補正する。

        Args:
          ref_rgb: (3,) [R, G, B] のキャリブレーション時 Ref 平均値。
        """
        rgb = np.asarray(ref_rgb, dtype=np.float64).ravel()
        if rgb.size < 3 or rgb[1] < 1e-8:
            return
        self._ref_baseline_rgb = rgb[:3].copy()
        self._dynamic_correction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self._ref_ema = None  # EMA を新ベースラインから再開

    def update_dynamic_correction(self, ref_rgb_now: np.ndarray) -> None:
        """現フレームの Ref RGB から動的補正係数を更新する（EMA平滑化版）。

        LED スペクトルの経時ドリフトにより Ref の R/G・B/G 比が変化した場合、
        その変動分を補正係数として反映する。
        単フレームノイズを EMA（指数移動平均）で平滑化してから算出する。

        補正係数の計算:
          f_R = (R_ref0/G_ref0) / (R_ema/G_ema)
          f_B = (B_ref0/G_ref0) / (B_ema/G_ema)

        Args:
          ref_rgb_now: (3,) [R, G, B] の現フレーム Ref 平均値。
        """
        if self._ref_baseline_rgb is None:
            return
        ref_now = np.asarray(ref_rgb_now, dtype=np.float64).ravel()[:3]
        if ref_now[1] < 1e-8 or ref_now[0] < 1e-8 or ref_now[2] < 1e-8:
            return
        # EMA 更新
        if self._ref_ema is None:
            self._ref_ema = ref_now.copy()
        else:
            a = self._ref_ema_alpha
            self._ref_ema = a * ref_now + (1.0 - a) * self._ref_ema
        r0, g0, b0 = self._ref_baseline_rgb
        r_e, g_e, b_e = self._ref_ema
        f_R = (r0 / g0) / (r_e / g_e)
        f_B = (b0 / g0) / (b_e / g_e)
        # 急激な変化は外れ値として除外（±50% 以上の変化は無視）
        f_R = float(np.clip(f_R, 0.5, 2.0))
        f_B = float(np.clip(f_B, 0.5, 2.0))
        self._dynamic_correction = np.array([f_R, 1.0, f_B], dtype=np.float64)

    def get_ref_ema(self) -> "np.ndarray | None":
        """EMA 平滑化された Ref RGB を返す。未初期化時は None。"""
        return self._ref_ema

    def clear_ref_baseline(self) -> None:
        """動的補正の基準をクリアし、補正係数を 1.0 に戻す。"""
        self._ref_baseline_rgb = None
        self._dynamic_correction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self._ref_ema = None

    def neutral_anchor_Lab(self) -> np.ndarray:
        """基準白色面の Lab アンカーを返す（a*=b*=0 を強制）。

        対角補正行列が有効な場合、ratio_to_Lab([1,1,1]) は
        sRGB→XYZ 行列のセンサー不一致に起因する偽の a*/b* を生じる。
        基準面は物理的に無彩色であるため a*=b*=0 を強制し、
        L* のみ対角補正による輝度効果を反映させる。

        Returns:
          np.ndarray: (3,) [L*, 0.0, 0.0] の Lab 配列 (float32)。
        """
        _lab = self.ratio_to_Lab(np.ones(3, dtype=np.float32))
        return np.array([float(_lab[0]), 0.0, 0.0], dtype=np.float32)

    def set_ccm(self, M: np.ndarray) -> None:
        """3×3 CCM を設定する。対角補正より優先して使用される。

        残差テーブルは旧 CCM に紐づくため、CCM 変更時に自動クリアする。

        Args:
          M: (3, 3) の色補正行列。
        """
        self._ccm = np.asarray(M, dtype=np.float64).reshape(3, 3)
        self._residual_ratios = None
        self._residual_labs = None

    def clear_ccm(self) -> None:
        """CCM をクリアし、対角補正にフォールバックする。残差テーブルも同時にクリア。"""
        self._ccm = None
        self._residual_ratios = None
        self._residual_labs = None

    def set_poly_ccm(self, M_poly: np.ndarray) -> None:
        """3×10 多項式 CCM を設定する。set_ccm より優先して使用される。

        Args:
          M_poly: (3, 10) の多項式色補正行列。
        """
        self._poly_ccm = np.asarray(M_poly, dtype=np.float64).reshape(3, -1)

    def clear_poly_ccm(self) -> None:
        """多項式 CCM をクリアし、通常 3×3 CCM にフォールバックする。"""
        self._poly_ccm = None

    def set_residuals(
        self, ref_ratios: np.ndarray, delta_labs: np.ndarray
    ) -> None:
        """パッチ別 ΔLab 残差テーブルを設定する。

        Args:
          ref_ratios: (N, 3) 各パッチの ratio_white（白点正規化済み）。
          delta_labs: (N, 3) 各パッチの ΔLab = Lab_ref - Lab_poly_pred。
        """
        self._residual_ratios = np.asarray(ref_ratios, dtype=np.float64)
        self._residual_labs = np.asarray(delta_labs, dtype=np.float64)

    def clear_residuals(self) -> None:
        """残差補正テーブルをクリアする。"""
        self._residual_ratios = None
        self._residual_labs = None

    def get_ccm(self) -> "np.ndarray | None":
        """現在の CCM を返す。未設定時は None。"""
        return self._ccm

    def ratio_to_Lab(
        self,
        ratio_rgb: np.ndarray,
        *,
        suppress_domain_warning: bool = False,
    ) -> np.ndarray:
        """
        リニアRGB比率をCIELAB実数値に変換。

        Args:
          ratio_rgb: (3,) [R, G, B] の比率配列（灰色カード=1.0）。
        Returns:
          np.ndarray: (3,) [L*, a*, b*] のLab配列。
        """
        reflectance = ratio_rgb.astype(np.float64) * self.reflectance_factor
        # 動的補正（LEDスペクトルドリフト補償）: 常時適用。
        # dc は reflectance 空間で作用し、残差 kNN は ratio 空間で検索するため
        # 両者は独立して適用可能。前回計画の排他制御を撤回。
        reflectance = reflectance * self._dynamic_correction
        # CCM 適用: poly_ccm 最優先 → 3×3 CCM → 対角補正にフォールバック
        if self._poly_ccm is not None:
            reflectance = self._poly_ccm @ poly_expand(reflectance)
        elif self._ccm is not None:
            reflectance = self._ccm @ reflectance
        else:
            # Phase 1（静的 k_R/k_B）のみ適用
            reflectance = reflectance * self._diag_correction
        XYZ = self._M_SRGB_TO_XYZ @ reflectance

        fx = self._f(XYZ[0] / self._XN)
        fy = self._f(XYZ[1] / self._YN)
        fz = self._f(XYZ[2] / self._ZN)

        L_star = 116.0 * fy - 16.0
        a_star = 500.0 * (fx - fy)
        b_star = 200.0 * (fy - fz)

        # Per-patch 残差補正: kNN (k=3) 距離加重補間で ΔLab を適用
        if self._residual_ratios is not None and self._residual_labs is not None:
            diffs = self._residual_ratios - ratio_rgb.astype(np.float64)
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            k = min(3, len(dists))
            nearest_idx = np.argpartition(dists, k)[:k]
            nearest_dists = dists[nearest_idx]
            # 学習データから大きく離れた入力: 残差補正の信頼性が低い
            if nearest_dists.min() > 0.5 and not suppress_domain_warning:
                import warnings
                warnings.warn(
                    f"残差kNN: 最近傍距離={nearest_dists.min():.3f} > 0.5 — "
                    f"ratio_white が学習ドメイン外の可能性",
                    stacklevel=2,
                )
            if nearest_dists.min() < 1e-8:
                delta = self._residual_labs[nearest_idx[nearest_dists.argmin()]]
            else:
                weights = 1.0 / nearest_dists
                delta = np.average(self._residual_labs[nearest_idx], axis=0, weights=weights)
            L_star += float(delta[0])
            a_star += float(delta[1])
            b_star += float(delta[2])

        return np.array([L_star, a_star, b_star], dtype=np.float32)


# ---------------------------------------------------------------------------
# G-3: Lab → linear RGB 逆変換ユーティリティ
# ---------------------------------------------------------------------------

# XYZ → linear sRGB 変換行列（_M_SRGB_TO_XYZ の逆行列）
_M_XYZ_TO_SRGB = np.linalg.inv(
    np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
)

_XN = 0.95047
_YN = 1.00000
_ZN = 1.08883
_DELTA = 6.0 / 29.0
_DELTA_CU = (6.0 / 29.0) ** 3
_3_DELTA_SQ = 3.0 * (6.0 / 29.0) ** 2
_OFFSET = 4.0 / 29.0


def _f_inv(t: float) -> float:
    """CIELAB f() の逆関数。"""
    if t > _DELTA:
        return t**3.0
    else:
        return _3_DELTA_SQ * (t - _OFFSET)


def Lab_to_XYZ(Lab: np.ndarray) -> np.ndarray:
    """CIELAB → XYZ (D65) 変換。

    Args:
      Lab: (3,) or (N, 3) の Lab 配列。
    Returns:
      np.ndarray: 同形状の XYZ 配列。
    """
    Lab = np.asarray(Lab, dtype=np.float64)
    if Lab.ndim == 1:
        L, a, b = Lab
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0
        return np.array(
            [_f_inv(fx) * _XN, _f_inv(fy) * _YN, _f_inv(fz) * _ZN],
            dtype=np.float64,
        )
    # バッチ処理 (N, 3)
    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    fi = np.vectorize(_f_inv)
    return np.column_stack([fi(fx) * _XN, fi(fy) * _YN, fi(fz) * _ZN])


def XYZ_to_linear_rgb(XYZ: np.ndarray) -> np.ndarray:
    """XYZ → linear sRGB 変換。

    Args:
      XYZ: (3,) or (N, 3) の XYZ 配列。
    Returns:
      np.ndarray: 同形状の linear sRGB 配列。
    """
    XYZ = np.asarray(XYZ, dtype=np.float64)
    if XYZ.ndim == 1:
        return _M_XYZ_TO_SRGB @ XYZ
    return XYZ @ _M_XYZ_TO_SRGB.T


def Lab_to_linear_rgb(Lab: np.ndarray) -> np.ndarray:
    """CIELAB → linear sRGB 変換。Lab_to_XYZ + XYZ_to_linear_rgb の合成。

    Args:
      Lab: (3,) or (N, 3) の Lab 配列。
    Returns:
      np.ndarray: 同形状の linear sRGB 配列。
    """
    return XYZ_to_linear_rgb(Lab_to_XYZ(Lab))


# ---------------------------------------------------------------------------
# G-4: CCM 算出ロジック
# ---------------------------------------------------------------------------


def poly_expand(rgb: np.ndarray) -> np.ndarray:
    """RGB (3,) または (N, 3) を 10次元多項式特徴量に展開する。

    出力: [R, G, B, R², G², B², RG, RB, GB, 1]  shape: (10,) または (N, 10)
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    scalar = rgb.ndim == 1
    if scalar:
        rgb = rgb[np.newaxis, :]
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    ones = np.ones(len(R), dtype=np.float64)
    expanded = np.stack([R, G, B, R*R, G*G, B*B, R*G, R*B, G*B, ones], axis=1)
    return expanded[0] if scalar else expanded


# --- CCM フィッティング用ヘルパー関数 ---

_CCM_M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_CCM_XN, _CCM_YN, _CCM_ZN = 0.95047, 1.00000, 1.08883
_CCM_DELTA_CU = (6.0 / 29.0) ** 3
_CCM_3_DELTA_SQ = 3.0 * (6.0 / 29.0) ** 2
_CCM_OFFSET = 4.0 / 29.0


def _ccm_f(t: float) -> float:
    """CIELAB f関数（CCM フィッティング用）。"""
    return t ** (1.0 / 3.0) if t > _CCM_DELTA_CU else t / _CCM_3_DELTA_SQ + _CCM_OFFSET


def _reflectance_to_lab(refl: np.ndarray) -> np.ndarray:
    """補正済み reflectance (3,) → Lab (3,) を返す。"""
    xyz = _CCM_M_SRGB_TO_XYZ @ refl
    fx = _ccm_f(xyz[0] / _CCM_XN)
    fy = _ccm_f(xyz[1] / _CCM_YN)
    fz = _ccm_f(xyz[2] / _CCM_ZN)
    return np.array([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)])


def _compute_ccm_srgb_lstsq(measured: np.ndarray, ref_labs: np.ndarray) -> np.ndarray:
    """sRGB 線形空間での最小二乗 CCM（Lab最適化の初期値用）。"""
    target = Lab_to_linear_rgb(ref_labs)
    M_T = np.linalg.lstsq(measured, target, rcond=None)[0]
    return M_T.T


def _lab_cost(M_flat: np.ndarray, measured: np.ndarray, ref_labs: np.ndarray) -> float:
    """Lab 空間 ΔE² の合計を返すコスト関数。"""
    M = M_flat.reshape(3, 3)
    total = 0.0
    for i in range(len(measured)):
        lab_pred = _reflectance_to_lab(M @ measured[i])
        diff = lab_pred - ref_labs[i]
        total += diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
    return total


def compute_ccm(
    measured_ratios: np.ndarray,
    known_labs: np.ndarray,
    reflectance_factor: float = 0.18,
    verbose: bool = True,
) -> np.ndarray:
    """測定 ratio と既知 Lab 値から 3×3 CCM を Lab 空間直接最適化で算出する。

    sRGB 最小二乗解を初期値として、scipy.optimize.minimize (L-BFGS-B) で
    Lab 空間 ΔE² を直接最小化する。最適化失敗時は sRGB 最小二乗解にフォールバック。

    Args:
      measured_ratios: (N, 3) 測定 R/G/B 比率配列。
      known_labs: (N, 3) 既知 Lab 値配列。
      reflectance_factor: 反射率係数（灰色カード基準）。
      verbose: True の場合に結果を標準出力に表示する。
    Returns:
      np.ndarray: (3, 3) CCM。
    """
    measured = np.asarray(measured_ratios, dtype=np.float64) * reflectance_factor
    ref_labs = np.asarray(known_labs, dtype=np.float64)

    # sRGB 最小二乗解を初期値として算出
    M0 = _compute_ccm_srgb_lstsq(measured, ref_labs)

    try:
        from scipy.optimize import minimize as _minimize
    except ModuleNotFoundError:
        cond = np.linalg.cond(M0)
        if verbose:
            print(f"CCM (sRGB最小二乗 — scipyなし fallback): 条件数={cond:.1f}")
        return M0

    # Lab 空間直接最適化
    result = _minimize(
        _lab_cost, M0.flatten(),
        args=(measured, ref_labs),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    if result.success:
        M = result.x.reshape(3, 3)
        method_label = "Lab空間最適化"
    else:
        M = M0
        method_label = "sRGB最小二乗 — フォールバック"

    cond = np.linalg.cond(M)
    if verbose:
        print(f"CCM ({method_label}): 条件数={cond:.1f}")
    return M


_POLY_CCM_COND_THRESH = 100.0  # poly CCM 条件数の閾値（超過時は 3×3 CCM のみ使用）


def compute_poly_ccm(
    measured_ratios: np.ndarray,
    known_labs: np.ndarray,
    reflectance_factor: float = 0.18,
    target_cond: float = 50.0,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """測定 ratio と既知 Lab 値から 3×10 多項式 CCM をリッジ回帰で算出する。

    設計行列のコンディション数が target_cond 以下になるよう
    最小リッジペナルティ α を自動算出し、安定したフィットを行う。

    Args:
      measured_ratios: (N, 3) 測定 R/G/B 比率配列（白点正規化済み）。
      known_labs: (N, 3) 既知 Lab 値配列。
      reflectance_factor: 反射率係数（灰色カード基準）。
      target_cond: 目標コンディション数（デフォルト 50）。
      verbose: True の場合にリッジ回帰の結果を標準出力に表示する。
    Returns:
      tuple[np.ndarray, float]: (3×10 多項式 CCM, 実効コンディション数)。
    """
    expanded = poly_expand(
        np.asarray(measured_ratios, dtype=np.float64) * reflectance_factor
    )  # (N, 10)
    target = Lab_to_linear_rgb(np.asarray(known_labs, dtype=np.float64))  # (N, 3)
    # 設計行列の特異値 → 最小リッジペナルティ α を算出
    sv = np.linalg.svd(expanded, compute_uv=False)
    sigma_max_sq = sv[0] ** 2
    sigma_min_sq = sv[-1] ** 2
    tc_sq = target_cond ** 2
    alpha_min = max(0.0, (sigma_max_sq - tc_sq * sigma_min_sq) / (tc_sq - 1.0))
    alpha = alpha_min * 1.1  # 10% マージン
    # 正規方程式: (X'X + αI) M' = X'y
    XtX = expanded.T @ expanded          # (10, 10)
    Xty = expanded.T @ target            # (10, 3)
    M_poly_T = np.linalg.solve(XtX + alpha * np.eye(10), Xty)  # (10, 3)
    M_poly = M_poly_T.T                  # (3, 10)
    eff_cond = float(np.sqrt((sigma_max_sq + alpha) / (sigma_min_sq + alpha)))
    if verbose:
        print(f"  Polynomial CCM (ridge): α={alpha:.6f}  "
              f"cond={eff_cond:.1f}  (raw={sv[0]/sv[-1]:.1f})")
    return M_poly, eff_cond


# ---------------------------------------------------------------------------
# G-5: CCMStore クラス
# ---------------------------------------------------------------------------


class CCMStore:
    """CCM の保存・読み込み・バックアップを管理するクラス。"""

    def __init__(self, save_dir: str = ""):
        """
        Args:
          save_dir: 保存先ディレクトリ。空文字列の場合は CALIBRATION_DIR を使用。
        """
        if not save_dir:
            save_dir = CALIBRATION_DIR
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, "ccm.json")
        self.ccm: np.ndarray | None = None
        self.poly_ccm: np.ndarray | None = None
        self.residual_ratios: np.ndarray | None = None
        self.residual_labs: np.ndarray | None = None
        # CCM 学習時の Ref ROI 平均値。live ref_scale baseline とは分離して保持する。
        self.ref_train: np.ndarray | None = None
        self.is_loaded: bool = False
        self.metadata: dict = {}
        # 白点パッチ（1E）の双光束比。save/load で永続化される。
        # 未ロード時は ones(3)（白点正規化なし ＝ 旧動作と互換）。
        self.white_ratio_rgb: np.ndarray = np.ones(3, dtype=np.float64)

    def save(
        self,
        M: np.ndarray,
        white_ratio_rgb: np.ndarray | None = None,
        metadata: dict | None = None,
        poly_ccm: np.ndarray | None = None,
        residual_ratios: np.ndarray | None = None,
        residual_labs: np.ndarray | None = None,
        ref_train: np.ndarray | None = None,
    ) -> None:
        """CCM と white_ratio_rgb を JSON 形式で保存し、バックアップを生成する。

        Args:
          M: (3, 3) CCM 行列。
          white_ratio_rgb: 白点パッチ（1E）の双光束比 shape=(3,)。None なら保存しない。
          metadata: オプションのメタデータ（ΔE 統計など）。
          poly_ccm: (3, 10) 多項式 CCM。None なら保存しない。
          residual_ratios: (N, 3) パッチ別 ratio_white。None なら保存しない。
          residual_labs: (N, 3) パッチ別 ΔLab。None なら保存しない。
          ref_train: (3,) CCM学習時の Ref ROI 平均値。学習条件の記録用。
        """
        os.makedirs(self.save_dir, exist_ok=True)
        data = {
            "ccm": M.tolist(),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if white_ratio_rgb is not None:
            data["white_ratio_rgb"] = np.asarray(white_ratio_rgb, dtype=np.float64).tolist()
            self.white_ratio_rgb = np.asarray(white_ratio_rgb, dtype=np.float64)
        if poly_ccm is not None:
            data["poly_ccm"] = np.asarray(poly_ccm, dtype=np.float64).tolist()
            self.poly_ccm = np.asarray(poly_ccm, dtype=np.float64)
        if residual_ratios is not None:
            data["residual_ratios"] = np.asarray(residual_ratios, dtype=np.float64).tolist()
            self.residual_ratios = np.asarray(residual_ratios, dtype=np.float64)
        if residual_labs is not None:
            data["residual_labs"] = np.asarray(residual_labs, dtype=np.float64).tolist()
            self.residual_labs = np.asarray(residual_labs, dtype=np.float64)
        if ref_train is not None:
            data["ref_train"] = np.asarray(ref_train, dtype=np.float64).tolist()
            self.ref_train = np.asarray(ref_train, dtype=np.float64)
        if metadata:
            data.update(metadata)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.ccm = np.asarray(M, dtype=np.float64)
        self.is_loaded = True
        self.metadata = data
        self.backup()
        _remove_cleared_marker("ccm.json")
        print(f"- CCM を保存: {self.save_path}")

    def load(self) -> "np.ndarray | None":
        """保存済み CCM を読み込む。見つからなければ None を返す。"""
        if not os.path.exists(self.save_path):
            return None
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            M = np.array(data["ccm"], dtype=np.float64).reshape(3, 3)
            self.ccm = M
            self.is_loaded = True
            self.metadata = data
            # white_ratio_rgb を読み込む。旧フォーマット（フィールドなし）は ones でフォールバック。
            if "white_ratio_rgb" in data:
                self.white_ratio_rgb = np.array(data["white_ratio_rgb"], dtype=np.float64)
            else:
                self.white_ratio_rgb = np.ones(3, dtype=np.float64)
                print("⚠ CCM ファイルに white_ratio_rgb がありません（旧フォーマット）。ones でフォールバックします。")
            # poly_ccm を読み込む。条件数が閾値超過なら無効化する。
            if "poly_ccm" in data:
                _M_poly = np.array(data["poly_ccm"], dtype=np.float64)
                try:
                    _sv = np.linalg.svd(_M_poly.reshape(3, 10), compute_uv=False)
                    _pcond = float(_sv[0] / _sv[-1]) if _sv[-1] > 1e-12 else float("inf")
                except Exception:
                    _pcond = float("inf")
                if _pcond <= _POLY_CCM_COND_THRESH:
                    self.poly_ccm = _M_poly
                else:
                    self.poly_ccm = None
                    print(f"⚠ 保存済み poly_ccm の条件数 {_pcond:.1f} > "
                          f"{_POLY_CCM_COND_THRESH}: 3×3 CCM のみ使用")
            else:
                self.poly_ccm = None
            # 残差テーブルを読み込む。旧フォーマット互換: フィールドがなければ None。
            self.residual_ratios = (
                np.array(data["residual_ratios"], dtype=np.float64)
                if "residual_ratios" in data else None
            )
            self.residual_labs = (
                np.array(data["residual_labs"], dtype=np.float64)
                if "residual_labs" in data else None
            )
            # CCM 学習時の Ref ROI 平均値。旧フォーマット互換: フィールドがなければ None。
            # live ref_scale baseline は startup / runtime で別途選択する。
            self.ref_train = (
                np.array(data["ref_train"], dtype=np.float64)
                if "ref_train" in data else None
            )
            created = data.get("created", "不明")
            print(f"- CCM をロード: {self.save_path} (作成: {created})")
            return M
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠ CCM ロード失敗: {e}")
            return None

    def backup(self) -> None:
        """タイムスタンプ付きバックアップを生成する。"""
        import shutil

        if not os.path.exists(self.save_path):
            return
        date_str = datetime.now().strftime("%Y-%m-%d")
        backup_path = os.path.join(self.save_dir, f"ccm_{date_str}.json")
        shutil.copy2(self.save_path, backup_path)
        self._cleanup_old_backups()

    def _cleanup_old_backups(self, max_keep: int = 5) -> None:
        """古いバックアップを削除し、最大 max_keep 世代を保持する。"""
        import glob

        pattern = os.path.join(self.save_dir, "ccm_*.json")
        backups = sorted(glob.glob(pattern), reverse=True)
        for old in backups[max_keep:]:
            os.remove(old)

    def clear(self) -> None:
        """CCM ファイルを削除し、状態をリセットする。"""
        self.ccm = None
        self.poly_ccm = None
        self.residual_ratios = None
        self.residual_labs = None
        self.ref_train = None
        self.white_ratio_rgb = np.ones(3, dtype=np.float64)
        self.is_loaded = False
        self.metadata = {}
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            print(f"- CCM を削除: {self.save_path}")

    def check_degradation(self) -> "str | None":
        """CCM ΔE_mean の履歴から基準物の劣化を検出する。

        calibration/ 以下の ccm_*.json バックアップファイルを時系列順に読み込み、
        最新の ΔE_mean が初回比50%以上増加していれば警告文字列を返す。

        Returns:
          str | None: 警告メッセージ。問題なければ None。
        """
        import glob as _glob
        pattern = os.path.join(self.save_dir, "ccm_*.json")
        backups = sorted(_glob.glob(pattern))
        dE_history = []
        for bp in backups:
            try:
                with open(bp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "delta_E_mean" in data:
                    dE_history.append(float(data["delta_E_mean"]))
            except Exception:
                continue
        if len(dE_history) < 2:
            return None
        first_dE = dE_history[0]
        latest_dE = dE_history[-1]
        if first_dE > 0 and latest_dE > first_dE * 1.5:
            return (
                f"⚠ CCM ΔE_mean が初回 {first_dE:.2f} → 最新 {latest_dE:.2f} に増加 "
                f"(+{(latest_dE / first_dE - 1) * 100:.0f}%): "
                f"基準物（グレーカード/SpyderCheckr）の劣化の可能性"
            )
        return None


# ---------------------------------------------------------------------------
# H-1: SpyderCheckr 48色 参照データ（Datacolor 公式値）
# ---------------------------------------------------------------------------
# パッチ ID: 行番号(1-6) + 列(A-H)
# 上段(SpyderCheckr 表面): 行1〜6, 列A〜D (24パッチ)
# 下段(SpyderCheckr 裏面): 行1〜6, 列E〜H (24パッチ、ColorChecker互換)

# SpyderCheckr 48 公式基準値（Datacolor 公式データシート準拠）
# 並び順: row-major（行優先）— 行0: 1A,1B,1C,1D,1E,1F,1G,1H → 行5: 6A,...,6H
# インデックス計算: index = row * 8 + col  (row=0..5, col=0..7)
# 白点パッチ 1E は row=0, col=4 → index=4
SPYDERCHECKER_REFERENCE = [
    # 行0 (row 1): 1A〜1H
    {"name": "1A", "Lab": [61.35, 34.81, 18.38], "sRGB": [210, 121, 117]},  # Low Sat. Red
    {"name": "1B", "Lab": [82.68,  5.03,  3.02], "sRGB": [218, 203, 201]},  # 10% Red Tint
    {"name": "1C", "Lab": [85.42,  9.41, 14.49], "sRGB": [237, 206, 186]},  # Lightest Skin
    {"name": "1D", "Lab": [92.72,  1.89,  2.76], "sRGB": [241, 233, 229]},  # 5% Gray
    {"name": "1E", "Lab": [96.04,  2.16,  2.60], "sRGB": [249, 242, 238]},  # Card White ★白点
    {"name": "1F", "Lab": [47.12, -32.50, -28.75], "sRGB": [  0, 127, 159]},  # Primary Cyan
    {"name": "1G", "Lab": [60.94, 38.21, 61.31], "sRGB": [222, 118,  32]},  # Primary Orange
    {"name": "1H", "Lab": [70.19, -31.90,  1.98], "sRGB": [ 98, 187, 166]},  # Aqua
    # 行1 (row 2): 2A〜2H
    {"name": "2A", "Lab": [75.50,  5.84, 50.42], "sRGB": [216, 179,  90]},  # Low Sat. Yellow
    {"name": "2B", "Lab": [82.25, -2.42,  3.78], "sRGB": [203, 205, 196]},  # 10% Green Tint
    {"name": "2C", "Lab": [74.28,  9.05, 27.21], "sRGB": [211, 175, 133]},  # Lighter Skin
    {"name": "2D", "Lab": [86.85,  1.59,  2.27], "sRGB": [229, 222, 220]},  # 10% Gray
    {"name": "2E", "Lab": [80.44,  1.17,  2.05], "sRGB": [202, 198, 195]},  # 20% Gray
    {"name": "2F", "Lab": [50.49, 53.45, -13.55], "sRGB": [192,  75, 145]},  # Primary Magenta
    {"name": "2G", "Lab": [37.80,  7.30, -43.04], "sRGB": [ 99,  86,  96]},  # Blueprint
    {"name": "2H", "Lab": [54.38,  8.84, -25.71], "sRGB": [126, 125, 174]},  # Lavender
    # 行2 (row 3): 3A〜3H
    {"name": "3A", "Lab": [66.82, -25.10, 23.47], "sRGB": [127, 175, 120]},  # Low Sat. Green
    {"name": "3B", "Lab": [82.29,  2.20, -2.04], "sRGB": [206, 203, 208]},  # 10% Blue Tint
    {"name": "3C", "Lab": [64.57, 12.39, 37.24], "sRGB": [193, 149,  91]},  # Moderate Skin
    {"name": "3D", "Lab": [73.42,  0.99,  1.89], "sRGB": [182, 178, 176]},  # 30% Gray
    {"name": "3E", "Lab": [65.52,  0.69,  1.86], "sRGB": [161, 157, 154]},  # 40% Gray
    {"name": "3F", "Lab": [83.61,  3.36, 87.02], "sRGB": [245, 205,   0]},  # Primary Yellow
    {"name": "3G", "Lab": [49.81, 48.50, 15.76], "sRGB": [195,  79,  95]},  # Pink
    {"name": "3H", "Lab": [42.03, -15.80, 22.93], "sRGB": [ 82, 106,  60]},  # Evergreen
    # 行3 (row 4): 4A〜4H
    {"name": "4A", "Lab": [60.53, -22.60, -20.40], "sRGB": [ 66, 157, 179]},  # Low Sat. Cyan
    {"name": "4B", "Lab": [24.89,  4.43,  0.78], "sRGB": [ 66,  57,  58]},  # 90% Red Tone
    {"name": "4C", "Lab": [44.49, 17.23, 26.24], "sRGB": [139,  93,  61]},  # Medium Skin
    {"name": "4D", "Lab": [57.15,  0.57,  1.19], "sRGB": [139, 136, 135]},  # 50% Gray ★アンカー
    {"name": "4E", "Lab": [49.62,  0.58,  1.56], "sRGB": [122, 118, 116]},  # 60% Gray
    {"name": "4F", "Lab": [41.05, 60.75, 31.17], "sRGB": [186,  26,  51]},  # Primary Red
    {"name": "4G", "Lab": [28.88, 19.36, -24.48], "sRGB": [ 83,  58, 106]},  # Violet
    {"name": "4H", "Lab": [48.82, -5.11, -23.08], "sRGB": [ 87, 120, 155]},  # Steel Blue
    # 行4 (row 5): 5A〜5H
    {"name": "5A", "Lab": [59.66, -2.03, -28.46], "sRGB": [116, 147, 194]},  # Low Sat. Blue
    {"name": "5B", "Lab": [25.16, -3.88,  2.13], "sRGB": [ 54,  61,  56]},  # 90% Green Tone
    {"name": "5C", "Lab": [25.29,  7.95,  8.87], "sRGB": [ 74,  55,  46]},  # Deep Skin
    {"name": "5D", "Lab": [41.57,  0.24,  1.45], "sRGB": [100,  99,  97]},  # 70% Gray
    {"name": "5E", "Lab": [33.55,  0.35,  1.40], "sRGB": [ 80,  80,  78]},  # 80% Gray
    {"name": "5F", "Lab": [54.14, -40.80, 34.75], "sRGB": [ 57, 146,  64]},  # Primary Green
    {"name": "5G", "Lab": [72.45, -23.60, 60.47], "sRGB": [157, 188,  54]},  # Apple Green
    {"name": "5H", "Lab": [65.10, 18.14, 18.68], "sRGB": [197, 145, 125]},  # Classic Light Skin
    # 行5 (row 6): 6A〜6H
    {"name": "6A", "Lab": [59.15, 30.83, -5.72], "sRGB": [190, 121, 154]},  # Low Sat. Magenta
    {"name": "6B", "Lab": [26.13,  2.61, -5.03], "sRGB": [ 63,  60,  69]},  # 90% Blue Tone
    {"name": "6C", "Lab": [22.67,  2.11, -1.10], "sRGB": [ 57,  54,  56]},  # 95% Gray
    {"name": "6D", "Lab": [25.65,  1.24,  0.05], "sRGB": [ 63,  61,  62]},  # 90% Gray
    {"name": "6E", "Lab": [16.91,  1.43, -0.81], "sRGB": [ 43,  41,  43]},  # Card Black
    {"name": "6F", "Lab": [24.75, 13.78, -49.48], "sRGB": [ 25,  55, 135]},  # Primary Blue
    {"name": "6G", "Lab": [71.65, 23.74, 72.28], "sRGB": [238, 158,  25]},  # Sunflower
    {"name": "6H", "Lab": [36.13, 14.15, 15.78], "sRGB": [112,  76,  60]},  # Classic Dark Skin
]

# パッチ名リスト（row-major 順）
SPYDERCHECKER_ORDER = [p["name"] for p in SPYDERCHECKER_REFERENCE]


# ---------------------------------------------------------------------------
# SpyderCheckr 24 基準値（Datacolor 公式データシート準拠）
# SpyderCheckr 48 の行0〜3、列A〜F（6列）のサブグリッド
# 並び順: row-major（行優先）— 行0: 1A,1B,1C,1D,1E,1F → 行3: 4A,...,4F
# インデックス計算: index = row * 6 + col  (row=0..3, col=0..5)
# 白点パッチ 1E は row=0, col=4 → index=4
# ---------------------------------------------------------------------------
SPYDERCHECKER_24_REFERENCE = [
    # 行0 (row 1): 1A〜1F
    {"name": "1A", "Lab": [61.35, 34.81, 18.38], "sRGB": [210, 121, 117]},  # Low Sat. Red
    {"name": "1B", "Lab": [82.68,  5.03,  3.02], "sRGB": [218, 203, 201]},  # 10% Red Tint
    {"name": "1C", "Lab": [85.42,  9.41, 14.49], "sRGB": [237, 206, 186]},  # Lightest Skin
    {"name": "1D", "Lab": [92.72,  1.89,  2.76], "sRGB": [241, 233, 229]},  # 5% Gray
    {"name": "1E", "Lab": [96.04,  2.16,  2.60], "sRGB": [249, 242, 238]},  # Card White ★白点
    {"name": "1F", "Lab": [47.12, -32.50, -28.75], "sRGB": [  0, 127, 159]},  # Primary Cyan
    # 行1 (row 2): 2A〜2F
    {"name": "2A", "Lab": [75.50,  5.84, 50.42], "sRGB": [216, 179,  90]},  # Low Sat. Yellow
    {"name": "2B", "Lab": [82.25, -2.42,  3.78], "sRGB": [203, 205, 196]},  # 10% Green Tint
    {"name": "2C", "Lab": [74.28,  9.05, 27.21], "sRGB": [211, 175, 133]},  # Lighter Skin
    {"name": "2D", "Lab": [86.85,  1.59,  2.27], "sRGB": [229, 222, 220]},  # 10% Gray
    {"name": "2E", "Lab": [80.44,  1.17,  2.05], "sRGB": [202, 198, 195]},  # 20% Gray
    {"name": "2F", "Lab": [50.49, 53.45, -13.55], "sRGB": [192,  75, 145]},  # Primary Magenta
    # 行2 (row 3): 3A〜3F
    {"name": "3A", "Lab": [66.82, -25.10, 23.47], "sRGB": [127, 175, 120]},  # Low Sat. Green
    {"name": "3B", "Lab": [82.29,  2.20, -2.04], "sRGB": [206, 203, 208]},  # 10% Blue Tint
    {"name": "3C", "Lab": [64.57, 12.39, 37.24], "sRGB": [193, 149,  91]},  # Moderate Skin
    {"name": "3D", "Lab": [73.42,  0.99,  1.89], "sRGB": [182, 178, 176]},  # 30% Gray
    {"name": "3E", "Lab": [65.52,  0.69,  1.86], "sRGB": [161, 157, 154]},  # 40% Gray
    {"name": "3F", "Lab": [83.61,  3.36, 87.02], "sRGB": [245, 205,   0]},  # Primary Yellow
    # 行3 (row 4): 4A〜4F
    {"name": "4A", "Lab": [60.53, -22.60, -20.40], "sRGB": [ 66, 157, 179]},  # Low Sat. Cyan
    {"name": "4B", "Lab": [24.89,  4.43,  0.78], "sRGB": [ 66,  57,  58]},  # 90% Red Tone
    {"name": "4C", "Lab": [44.49, 17.23, 26.24], "sRGB": [139,  93,  61]},  # Medium Skin
    {"name": "4D", "Lab": [57.15,  0.57,  1.19], "sRGB": [139, 136, 135]},  # 50% Gray ★アンカー
    {"name": "4E", "Lab": [49.62,  0.58,  1.56], "sRGB": [122, 118, 116]},  # 60% Gray
    {"name": "4F", "Lab": [41.05, 60.75, 31.17], "sRGB": [186,  26,  51]},  # Primary Red
]

# パッチ名リスト（SpyderCheckr 24 row-major 順）
SPYDERCHECKER_24_ORDER = [p["name"] for p in SPYDERCHECKER_24_REFERENCE]

# ---------------------------------------------------------------------------
# グリッド形状定数
# ---------------------------------------------------------------------------
SPYDERCHECKR_48_SHAPE = (6, 8)  # (n_rows, n_cols)
SPYDERCHECKR_24_SHAPE = (4, 6)
SPYDERCHECKR_PATCHWISE_SCORE_THRESHOLD = 0.60
SPYDERCHECKR_PATCHWISE_MIN_AREA_RATIO = 0.20
SPYDERCHECKR_PATCHWISE_CENTER_TOLERANCE_RATIO = 0.05
SPYDERCHECKR_CALIBRATION_GEOMETRY_MIN_AREA_RATIO = 0.35
SPYDERCHECKR_CALIBRATION_GEOMETRY_MAX_RESCUE_FRACTION = 0.50
SPYDERCHECKR_CALIBRATION_GEOMETRY_MAX_WEAK_FRACTION = 0.25
SPYDERCHECKR_DIRECT_PANEL_PATCH_MARGIN = 0.16
SPYDERCHECKR_DIRECT_PANEL_BRIGHT_THRESHOLD = 185
SPYDERCHECKR_DIRECT_PANEL_DARK_THRESHOLDS = (88, 96, 104, 112, 120, 128, 136)
SPYDERCHECKR_DIRECT_PANEL_MIN_HINGE_GAP = 1.40
SPYDERCHECKR_DIRECT_PANEL_MAX_HINGE_GAP = 4.00
SPYDERCHECKR_DIRECT_PANEL_SIGNAL_MARGIN_RATIO = 0.06
SPYDERCHECKR_DIRECT_PANEL_PATCH_SIGNAL_FLOOR = 90
SPYDERCHECKR_DIRECT_PANEL_MIN_PATCH_COMPONENTS = 10
SPYDERCHECKR_DIRECT_PANEL_FACE_SIZE_CAP_RATIO = 0.88
SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO = 0.16
SPYDERCHECKR_RAW_CANONICAL_SAMPLE_GRID_SIDE = 8
# Phase 14 (2026-04-27): direct-panel lattice の rotation 探索範囲は ±15° に固定する。
# 30° / 45° の geometry recovery は production テストで保証されておらず、
# 範囲を広げる場合は 15..45° の successful label-identity tests を追加してから
# 別フェーズで段階的に拡張する。
SPYDERCHECKR_DIRECT_PANEL_MAX_LATTICE_ROTATION_DEGREES = 15.0
SPYDERCHECKR_DIRECT_PANEL_LATTICE_ASSIGN_TOLERANCE = 0.58
SPYDERCHECKR_DIRECT_PANEL_LATTICE_MAX_RMSE_RATIO = 0.18


def _evaluate_patchwise_calibration_geometry(
    *,
    status: str,
    patch_count: int,
    rescue_patch_count: int,
    fallback_patch_count: int,
    weak_patch_count: int,
    median_area_ratio_to_cell_pitch_area: float,
) -> dict[str, object]:
    """ROI geometry が calibration 用として成立しているかを additive に要約する。"""
    patch_count = max(int(patch_count), 0)
    rescue_patch_fraction = (
        float(rescue_patch_count) / float(patch_count)
        if patch_count > 0
        else 0.0
    )
    fallback_patch_fraction = (
        float(fallback_patch_count) / float(patch_count)
        if patch_count > 0
        else 0.0
    )
    weak_patch_fraction = (
        float(weak_patch_count) / float(patch_count)
        if patch_count > 0
        else 0.0
    )
    failure_reasons: list[str] = []
    if status != "ok":
        failure_reasons.append("frame_or_geometry_unavailable")
    if (
        float(median_area_ratio_to_cell_pitch_area)
        < SPYDERCHECKR_CALIBRATION_GEOMETRY_MIN_AREA_RATIO - 1e-9
    ):
        failure_reasons.append("roi_area_below_floor")
    if rescue_patch_fraction > SPYDERCHECKR_CALIBRATION_GEOMETRY_MAX_RESCUE_FRACTION + 1e-9:
        failure_reasons.append("rescue_dominated")
    if weak_patch_fraction > SPYDERCHECKR_CALIBRATION_GEOMETRY_MAX_WEAK_FRACTION + 1e-9:
        failure_reasons.append("weak_patch_fraction_above_limit")
    return {
        "patch_count": int(patch_count),
        "rescue_patch_fraction": round(float(rescue_patch_fraction), 6),
        "fallback_patch_fraction": round(float(fallback_patch_fraction), 6),
        "weak_patch_fraction": round(float(weak_patch_fraction), 6),
        "rescue_dominant": bool(
            rescue_patch_fraction
            > SPYDERCHECKR_CALIBRATION_GEOMETRY_MAX_RESCUE_FRACTION + 1e-9
        ),
        "calibration_geometry_viable": "yes" if not failure_reasons else "no",
        "calibration_geometry_reason": (
            "ok" if not failure_reasons else ",".join(failure_reasons)
        ),
        "calibration_geometry_failure_reasons": list(failure_reasons),
        "calibration_geometry_min_area_ratio": round(
            SPYDERCHECKR_CALIBRATION_GEOMETRY_MIN_AREA_RATIO,
            6,
        ),
        "calibration_geometry_max_rescue_fraction": round(
            SPYDERCHECKR_CALIBRATION_GEOMETRY_MAX_RESCUE_FRACTION,
            6,
        ),
        "calibration_geometry_max_weak_fraction": round(
            SPYDERCHECKR_CALIBRATION_GEOMETRY_MAX_WEAK_FRACTION,
            6,
        ),
    }
SPYDERCHECKR_PATCHWISE_STANDARD_SIZE_FLOOR_RATIO = 0.70
SPYDERCHECKR_PATCHWISE_RESCUE_SIZE_FLOOR_RATIO = 0.50

# ---------------------------------------------------------------------------
# 白点パッチのインデックス定数
# 白点パッチ: 1E (Card White, L*=96.04) の row-major インデックス
# 48パッチ (6×8): row=0, col=4 → index = 0*8 + 4 = 4
SPYDERCHECKR_48_WHITE_PATCH_INDEX = 4
# 24パッチ (4×6): row=0, col=4 → index = 0*6 + 4 = 4
SPYDERCHECKR_24_WHITE_PATCH_INDEX = 4
# 50% グレー（B・N 補正アンカー）: 4D パッチのインデックス
# 48パッチ (6×8): row=3, col=3 → index = 3*8 + 3 = 27
SPYDERCHECKR_48_GRAY_4D_INDEX = 27
# ---------------------------------------------------------------------------
# Phase 4A: 上下反転検出
# 6E (Card Black, L*=16.91) の row-major インデックス
# 48パッチ (6×8): row=5, col=4 → index = 5*8 + 4 = 44
SPYDERCHECKR_48_BLACK_PATCH_INDEX = 44
# 正置判定の最小比率: ratios_gray[1E][G] / ratios_gray[6E][G] が
# これ以下なら上下反転と判定. 正置なら 1E は 6E の 3〜10 倍の G 値を持つ.
# 2.0 は safety margin を取った閾値.
SPYDER_FLIP_MIN_RATIO = 2.0

# Phase 7F: 回転 / 斜め配置 / 未設置 などを検出するための D 列グラデーション単調性テスト.
# D 列 = col 3 (row-major): 1D(3), 2D(11), 3D(19), 4D(27), 5D(35), 6D(43)
# 正置なら L 値は monotonic decreasing: 88 → 72 → 57 → 42 → 25 → 25 (40% → 90% gray).
# 45° 回転や未設置の場合、これらのインデックスは chart 外の位置や別パッチを拾って
# 単調性が崩れる → 異常と判定.
SPYDERCHECKR_48_D_COLUMN_INDICES = (3, 11, 19, 27, 35, 43)
# Phase 14: F 列 = 6 種の高彩度 primary (Cyan/Magenta/Yellow/Red/Green/Blue) の row-major
# インデックス. row r × 8 + 5 (col F = 5).
SPYDERCHECKR_48_F_COLUMN_INDICES = (5, 13, 21, 29, 37, 45)
# Phase 14 (Codex C4 対応): A 列 = 6 種の Low-Saturation 色 (Red/Yellow/Green/Cyan/Blue/Magenta).
# F 列だけだと左パネル (col 0-3) に chromatic anchor が皆無で、外側列の label 破損を検出
# できないため、A 列 (col 0) も補助 chromatic anchor として追加する.
SPYDERCHECKR_48_A_COLUMN_INDICES = (0, 8, 16, 24, 32, 40)
# 5 pair のうち 4 pair 以上で decreasing (または increasing for flip 判定) なら clear pattern.
SPYDER_D_COLUMN_MIN_MONOTONIC_PAIRS = 4
# 別名 (後方互換用); 意味は同じ
SPYDERCHECKR_48_D_COLUMN_THRESHOLD = SPYDER_D_COLUMN_MIN_MONOTONIC_PAIRS
# flat / low-contrast input is no-chart/misplacement evidence, not true 180° flip.
SPYDER_CHART_MIN_G_REL_CONTRAST = 0.15

# Phase 14: pose hypothesis confidence gate の閾値群. strategy で固定された数値を
# 一箇所に集約し、テスト/監視で書き換え可能にする (定数自体は monkeypatch で上書き不可な
# モジュール属性なので、変更時はソース修正 + 関連テスト更新が必要).
SPYDER_POSE_TOTAL_VALID_ANCHORS_FLOOR = 12
SPYDER_POSE_PER_PANEL_VALID_ANCHORS_FLOOR = 4
SPYDER_POSE_NEUTRAL_ANCHORS_FLOOR = 4
SPYDER_POSE_CHROMATIC_ANCHORS_FLOOR = 4
SPYDER_POSE_HYPOTHESIS_MIN_VALID_ANCHORS = 6
SPYDER_POSE_UPRIGHT_MEDIAN_RESIDUAL_MAX = 0.12
SPYDER_POSE_UPRIGHT_TAIL_RESIDUAL_MAX = 0.25
SPYDER_POSE_ABSOLUTE_GAP_MIN = 0.15
SPYDER_POSE_RELATIVE_GAP_MIN = 1.35


def _d_column_monotonic_decreasing_pairs(arr) -> int:
    """D 列 6 要素の隣接 pair のうち、G 値が decreasing な組の数を返す (0-5).

    arr: (48, 3) 相当の array-like.
    return: 0-5 の int. 5 は完全単調減少.
    """
    try:
        g_values = [float(arr[idx][1]) for idx in SPYDERCHECKR_48_D_COLUMN_INDICES]
    except (IndexError, TypeError, ValueError):
        return 0
    pairs_decreasing = 0
    for i in range(len(g_values) - 1):
        # decreasing = 次の要素がより小さい. 等しいもの (5D=6D 等) も許容する.
        if g_values[i] >= g_values[i + 1]:
            pairs_decreasing += 1
    return pairs_decreasing


def detect_spyder_flip(ratios_gray) -> bool:
    """SpyderCHECKR 48 の上下反転を ratios_gray から検出する (Phase 4A, 保険用).

    Phase 7F 以降は `detect_spyder_chart_misplaced` を主に使う。flip は部分集合.

    Returns:
        True: 1E/6E の G 比逆転 (180° 反転)
        False: 正置または判定不能
    """
    try:
        arr = [list(row) for row in ratios_gray]
    except TypeError:
        return False
    if len(arr) < max(
        SPYDERCHECKR_48_WHITE_PATCH_INDEX, SPYDERCHECKR_48_BLACK_PATCH_INDEX
    ) + 1:
        return False
    try:
        white_g = float(arr[SPYDERCHECKR_48_WHITE_PATCH_INDEX][1])
        black_g = float(arr[SPYDERCHECKR_48_BLACK_PATCH_INDEX][1])
    except (IndexError, TypeError, ValueError):
        return False
    denom = max(black_g, 1e-9)
    ratio = white_g / denom
    return ratio < SPYDER_FLIP_MIN_RATIO


def detect_spyder_chart_misplaced(ratios_gray) -> tuple[bool, str]:
    """SpyderCHECKR 48 の **向き異常全般** を検出する (Phase 7F 改訂).

    判定ロジック (優先順):
    1. **D 列単調性を最初に見る**. D 列は 1D(L=88)→6D(L=25) の grayscale.
       - 正置: decreasing pair >= 4/5
       - 180° flip: D 列が逆転、increasing pair >= 4/5
       - 45° 回転 / 未設置: ランダムまたは低コントラスト
    2. flat / low-contrast は chart 未設置・大きなズレとして generic placement 扱い
    3. D 列が reverse monotonic (inc pair >= 4) かつ 1E/6E 比が < 1.5 → 明確な 180° flip
    4. D 列が decreasing 十分 (dec pair >= 4) かつ 1E/6E 比が >= SPYDER_FLIP_MIN_RATIO → 正置
    5. それ以外 → "rotated_or_misplaced" (45° 回転、チルト、未設置)

    これにより 45° 回転がラッキーで flip_ratio < 2.0 に該当しても、D 列の
    monotonic 方向を見て rotated_or_misplaced を優先判定できる.

    Returns:
        (is_problem, reason): 正置なら (False, "ok"), 異常なら (True, reason)
    """
    try:
        arr = [list(row) for row in ratios_gray]
    except TypeError:
        return (False, "input_not_iterable")
    if len(arr) < 48:
        return (False, "input_too_short")

    try:
        white_g = float(arr[SPYDERCHECKR_48_WHITE_PATCH_INDEX][1])
        black_g = float(arr[SPYDERCHECKR_48_BLACK_PATCH_INDEX][1])
        chart_g_values = [float(row[1]) for row in arr[:48]]
    except (IndexError, TypeError, ValueError):
        return (False, "input_invalid")
    flip_ratio = white_g / max(black_g, 1e-9)

    # D 列の減少 / 増加 pair をカウント
    try:
        d_values = [float(arr[idx][1]) for idx in SPYDERCHECKR_48_D_COLUMN_INDICES]
    except (IndexError, TypeError, ValueError):
        return (False, "input_invalid_d_column")
    if not all(
        math.isfinite(value)
        for value in [white_g, black_g, *chart_g_values, *d_values]
    ):
        return (False, "input_invalid")
    chart_g_contrast = (max(chart_g_values) - min(chart_g_values)) / max(
        abs(float(np.median(chart_g_values))),
        1e-9,
    )
    d_g_contrast = (max(d_values) - min(d_values)) / max(
        abs(float(np.median(d_values))),
        1e-9,
    )
    dec_pairs = sum(
        1 for i in range(len(d_values) - 1) if d_values[i] >= d_values[i + 1]
    )
    inc_pairs = sum(
        1 for i in range(len(d_values) - 1) if d_values[i] <= d_values[i + 1]
    )

    # flat / low-contrast scenes can look both decreasing and increasing because
    # equal adjacent values satisfy both monotonic tests. Treat them as placement
    # failures so only real chart contrast can produce a true flip reason.
    if max(chart_g_contrast, d_g_contrast) < SPYDER_CHART_MIN_G_REL_CONTRAST:
        return (
            True,
            f"rotated_or_misplaced_low_contrast "
            f"(G contrast={chart_g_contrast:.3f}, D-col contrast={d_g_contrast:.3f}, "
            f"1E/6E ratio={flip_ratio:.2f})",
        )

    # 1) 正置判定: D 列 decreasing かつ 1E/6E 比も十分
    if (
        dec_pairs >= SPYDERCHECKR_48_D_COLUMN_THRESHOLD
        and flip_ratio >= SPYDER_FLIP_MIN_RATIO
    ):
        return (False, "ok")

    # 2) 180° flip: D 列が reverse monotonic かつ 1E/6E 比も逆転近く
    if (
        inc_pairs >= SPYDERCHECKR_48_D_COLUMN_THRESHOLD
        and flip_ratio < 1.5
    ):
        return (
            True,
            f"flip (D-col reversed inc_pairs={inc_pairs}/5, ratio={flip_ratio:.2f})",
        )

    # 3) それ以外 = 45° 回転 / チルト / 未設置 / ズレ
    return (
        True,
        f"rotated_or_misplaced (D-col dec={dec_pairs}/5 inc={inc_pairs}/5, "
        f"1E/6E ratio={flip_ratio:.2f})",
    )


# ---------------------------------------------------------------------------
# Phase 14: 多色アンカー pose hypothesis 信頼度ゲート
# ---------------------------------------------------------------------------
# 目的: detect_spyder_chart_misplaced の D 列 / 1E,6E geometry チェックを通過した
# 入力に対し、複数の色アンカー (neutral D 列 + 1E/6E + 高彩度 F 列) を使って
# upright が他候補 (rot180 / mirror_h/v / panel_swap / row/col one-cell shift) に
# 対して明確な勝者であることを確認する. CCM fit に進む直前の最後の安全網.

_SPYDER_POSE_NEUTRAL_ANCHOR_INDICES: tuple[int, ...] = tuple(
    sorted(
        set(SPYDERCHECKR_48_D_COLUMN_INDICES)
        | {
            SPYDERCHECKR_48_WHITE_PATCH_INDEX,
            SPYDERCHECKR_48_BLACK_PATCH_INDEX,
        }
    )
)
_SPYDER_POSE_CHROMATIC_ANCHOR_INDICES: tuple[int, ...] = tuple(
    sorted(
        set(SPYDERCHECKR_48_F_COLUMN_INDICES) | set(SPYDERCHECKR_48_A_COLUMN_INDICES)
    )
)
_SPYDER_POSE_ANCHOR_INDICES: tuple[int, ...] = tuple(
    sorted(
        set(_SPYDER_POSE_NEUTRAL_ANCHOR_INDICES)
        | set(_SPYDER_POSE_CHROMATIC_ANCHOR_INDICES)
    )
)


def _spyder_pose_panel_index(patch_index: int) -> int:
    """6×8 grid の col index から左/右パネル (0/1) を返す."""
    return 0 if (patch_index % 8) < 4 else 1


def _spyder_pose_hypothesis_index_map(name: str) -> "dict[int, int] | None":
    """名前付き hypothesis ごとの permutation を返す.

    map[expected_idx] = physical_idx (= reference index of the patch that physically
    occupies the position we sample as "expected_idx"). Out-of-range の対応は
    map に含めず、テスト側で valid anchor 数で判定する. rot90 / rot270 は 6×8 と
    8×6 の整合性が取れず invalid。
    """
    n_rows = 6
    n_cols = 8
    n = n_rows * n_cols

    def rc(i: int) -> tuple[int, int]:
        return divmod(i, n_cols)

    def to_idx(r: int, c: int) -> int:
        return r * n_cols + c

    if name == "upright":
        return {i: i for i in range(n)}
    if name == "rot180":
        return {i: (n - 1 - i) for i in range(n)}
    if name == "mirror_horizontal":
        return {i: to_idx(rc(i)[0], n_cols - 1 - rc(i)[1]) for i in range(n)}
    if name == "mirror_vertical":
        return {i: to_idx(n_rows - 1 - rc(i)[0], rc(i)[1]) for i in range(n)}
    if name == "panel_swap":
        result: dict[int, int] = {}
        half = n_cols // 2
        for i in range(n):
            r, c = rc(i)
            new_c = c + half if c < half else c - half
            result[i] = to_idx(r, new_c)
        return result
    if name == "shift_row_-1":
        return {i: to_idx(rc(i)[0] + 1, rc(i)[1]) for i in range(n) if rc(i)[0] < n_rows - 1}
    if name == "shift_row_+1":
        return {i: to_idx(rc(i)[0] - 1, rc(i)[1]) for i in range(n) if rc(i)[0] > 0}
    if name == "shift_col_-1":
        return {i: to_idx(rc(i)[0], rc(i)[1] + 1) for i in range(n) if rc(i)[1] < n_cols - 1}
    if name == "shift_col_+1":
        return {i: to_idx(rc(i)[0], rc(i)[1] - 1) for i in range(n) if rc(i)[1] > 0}
    if name in ("rot90", "rot270"):
        return None
    return None


SPYDER_POSE_HYPOTHESIS_NAMES: tuple[str, ...] = (
    "upright",
    "rot180",
    "mirror_horizontal",
    "mirror_vertical",
    "panel_swap",
    "shift_row_-1",
    "shift_row_+1",
    "shift_col_-1",
    "shift_col_+1",
)
SPYDER_POSE_INVALID_HYPOTHESIS_NAMES: tuple[str, ...] = ("rot90", "rot270")


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """sRGB 0-1 値を IEC 61966-2-1 の linear sRGB に変換する.

    Pose gate は camera からの linear bayer 比率を受け取る前提なので、
    SPYDERCHECKER_REFERENCE の gamma sRGB 値を linear に decoded して比較する.
    """
    arr = np.asarray(values, dtype=np.float64)
    return np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)


def _spyder_pose_reference_linear_array() -> np.ndarray:
    """SPYDERCHECKER_REFERENCE の sRGB を 0-1 正規化 → linear sRGB shape (48, 3)."""
    rows: list[list[float]] = []
    for patch in SPYDERCHECKER_REFERENCE[:48]:
        srgb = patch["sRGB"]
        rows.append([float(srgb[0]) / 255.0, float(srgb[1]) / 255.0, float(srgb[2]) / 255.0])
    srgb_arr = np.asarray(rows, dtype=np.float64)
    return _srgb_to_linear(srgb_arr)


def _spyder_pose_white_balance_array(
    arr: np.ndarray,
    anchor_indices: "list[int]",
) -> np.ndarray:
    """anchor 集合内の各チャンネル中央値で割って per-channel WB を等しくする.

    Pose gate に渡される入力は camera の linear bayer 比率で、ホワイトバランスが
    取れていない場合 (channel gain の差) に upright を false-block しうる.
    anchor 集合の中央値で各チャンネルを割ることで、どのような白色調整を受けた入力
    でも仮説間 score 比較が可能になる.
    """
    if not anchor_indices:
        return arr
    sub = arr[anchor_indices]
    medians = np.median(sub, axis=0)
    medians = np.where(medians > 1e-6, medians, 1.0)
    return arr / medians


def _spyder_pose_normalized_features(rgb: np.ndarray, median_g: float) -> np.ndarray:
    """RGB から exposure 不変の (L', a', b') 特徴ベクトルを返す.

    L' = G / median_G_over_anchors
    a' = (R - G) / median_G_over_anchors
    b' = (B - G) / median_G_over_anchors

    Phase 14 (Evaluator E3 対応): 入力は事前に `_spyder_pose_white_balance_array()` で per-channel
    中央値が概ね 1.0 になっている前提. WB 経路を将来変更する際は、median_G で R-G / B-G を割る
    本実装が崩れないかを確認すること. WB 等化を outsourcing せず R/B も別 median で割る方が
    docstring 不要だがオーバーヘッド増。
    """
    g = float(rgb[1])
    denom = max(float(median_g), 1e-6)
    return np.asarray(
        [
            g / denom,
            (float(rgb[0]) - g) / denom,
            (float(rgb[2]) - g) / denom,
        ],
        dtype=np.float64,
    )


def _spyder_pose_hypothesis_score(
    observed: np.ndarray,
    reference: np.ndarray,
    anchor_indices: "list[int]",
    index_map: "dict[int, int]",
) -> "dict[str, object]":
    """1 仮説に対し、有効 anchor を score して median+p90 残差を返す."""
    valid: list[int] = []
    obs_g_values: list[float] = []
    pred_g_values: list[float] = []
    for idx in anchor_indices:
        target = index_map.get(idx)
        if target is None:
            continue
        obs_rgb = observed[idx]
        if not np.all(np.isfinite(obs_rgb)):
            continue
        valid.append(idx)
        obs_g_values.append(float(obs_rgb[1]))
        pred_g_values.append(float(reference[target][1]))
    median_obs_g = float(np.median(obs_g_values)) if obs_g_values else 0.0
    median_pred_g = float(np.median(pred_g_values)) if pred_g_values else 0.0
    residuals: list[float] = []
    per_panel: dict[int, int] = {0: 0, 1: 0}
    neutral_count = 0
    chromatic_count = 0
    for idx in valid:
        target = index_map[idx]
        obs_feat = _spyder_pose_normalized_features(observed[idx], median_obs_g)
        pred_feat = _spyder_pose_normalized_features(reference[target], median_pred_g)
        diff = obs_feat - pred_feat
        residual = float(np.sqrt(float(np.mean(diff * diff))))
        residuals.append(residual)
        per_panel[_spyder_pose_panel_index(idx)] += 1
        if idx in _SPYDER_POSE_NEUTRAL_ANCHOR_INDICES:
            neutral_count += 1
        if idx in _SPYDER_POSE_CHROMATIC_ANCHOR_INDICES:
            chromatic_count += 1
    if residuals:
        median_residual = float(np.median(residuals))
        p90_residual = float(np.percentile(residuals, 90.0))
    else:
        median_residual = float("inf")
        p90_residual = float("inf")
    score = median_residual + 0.3 * p90_residual
    return {
        "valid_anchors": valid,
        "per_panel": per_panel,
        "neutral_count": neutral_count,
        "chromatic_count": chromatic_count,
        "residuals": residuals,
        "score": score,
        "median": median_residual,
        "p90": p90_residual,
    }


def evaluate_spydercheckr_pose_confidence(ratios_gray) -> dict:
    """多色アンカーで pose 仮説を比較し、upright が一意な最良であるかを評価する.

    入力前提: ratios_gray は現在の expected-label 順 shape (48, 3). geometry 段が
    ROI を確定している前提で、この関数は「サンプルされた色が期待 label と整合するか」
    だけを判定する.

    Returns:
        dict with keys:
            decision: "upright" / "blocked"
            blocking_reason: str (decision="blocked" の時のみ意味のある値)
            best_hypothesis: str
            scores: dict[name, dict] (各 hypothesis の詳細)
            valid_anchor_total: int
            invalid_hypotheses: list[str] (rot90/270 等)
            evidence: dict (median, p90, gap など)
    """
    diagnostic_invalid: list[str] = list(SPYDER_POSE_INVALID_HYPOTHESIS_NAMES)
    base = {
        "decision": "blocked",
        "blocking_reason": "pose_input_invalid",
        "best_hypothesis": "",
        "scores": {},
        "valid_anchor_total": 0,
        "invalid_hypotheses": diagnostic_invalid,
        "evidence": {},
    }
    try:
        observed = np.asarray(ratios_gray, dtype=np.float64)
    except (TypeError, ValueError):
        return base
    if observed.ndim != 2 or observed.shape[0] < 48 or observed.shape[1] < 3:
        return base
    observed = observed[:48, :3]
    if not np.all(np.isfinite(observed)):
        return base

    reference = _spyder_pose_reference_linear_array()
    anchor_list = list(_SPYDER_POSE_ANCHOR_INDICES)
    # C2 fix: anchor 集合の per-channel 中央値で観測を WB 等化することで、
    # camera の channel gain や白点正規化の有無に影響されず仮説比較できる.
    observed = _spyder_pose_white_balance_array(observed, anchor_list)
    reference = _spyder_pose_white_balance_array(reference, anchor_list)
    upright_map = _spyder_pose_hypothesis_index_map("upright") or {}
    upright_eval = _spyder_pose_hypothesis_score(observed, reference, anchor_list, upright_map)
    valid_total = len(upright_eval["valid_anchors"])

    scores: dict[str, dict] = {}
    for name in SPYDER_POSE_HYPOTHESIS_NAMES:
        index_map = _spyder_pose_hypothesis_index_map(name)
        if index_map is None:
            continue
        scores[name] = _spyder_pose_hypothesis_score(
            observed, reference, anchor_list, index_map
        )

    try:
        d_g_values = [float(observed[i][1]) for i in SPYDERCHECKR_48_D_COLUMN_INDICES]
        d_median = abs(float(np.median(d_g_values)))
        d_contrast = (max(d_g_values) - min(d_g_values)) / max(d_median, 1e-6)
    except Exception:
        d_contrast = 0.0

    evidence_common = {
        "valid_anchor_total": valid_total,
        "valid_anchor_per_panel": dict(upright_eval["per_panel"]),
        "neutral_anchor_count": upright_eval["neutral_count"],
        "chromatic_anchor_count": upright_eval["chromatic_count"],
        "d_column_g_contrast": d_contrast,
        "scores_summary": {
            name: {
                "score": result["score"],
                "median": result["median"],
                "p90": result["p90"],
                "valid": len(result["valid_anchors"]),
            }
            for name, result in scores.items()
        },
        "invalid_hypotheses": diagnostic_invalid,
    }

    eligible = {
        name: result
        for name, result in scores.items()
        if len(result["valid_anchors"]) >= SPYDER_POSE_HYPOTHESIS_MIN_VALID_ANCHORS
    }
    if "upright" not in eligible:
        return {
            **base,
            "blocking_reason": "pose_confidence_insufficient",
            "scores": scores,
            "valid_anchor_total": valid_total,
            "evidence": {
                **evidence_common,
                "reason_detail": "upright hypothesis lacks sufficient valid anchors",
            },
        }

    upright_panels = upright_eval["per_panel"]
    failed_anchor_floor = (
        valid_total < SPYDER_POSE_TOTAL_VALID_ANCHORS_FLOOR
        or min(upright_panels.values()) < SPYDER_POSE_PER_PANEL_VALID_ANCHORS_FLOOR
        or upright_eval["neutral_count"] < SPYDER_POSE_NEUTRAL_ANCHORS_FLOOR
        or upright_eval["chromatic_count"] < SPYDER_POSE_CHROMATIC_ANCHORS_FLOOR
        or d_contrast < SPYDER_CHART_MIN_G_REL_CONTRAST
    )

    sorted_eligible = sorted(eligible.items(), key=lambda kv: kv[1]["score"])
    best_name, best_result = sorted_eligible[0]
    second_name = sorted_eligible[1][0] if len(sorted_eligible) >= 2 else ""
    second_score = sorted_eligible[1][1]["score"] if len(sorted_eligible) >= 2 else float("inf")

    if best_result["score"] > 1e-9:
        relative_gap = second_score / best_result["score"]
    else:
        relative_gap = float("inf")
    absolute_gap = (
        second_score - best_result["score"]
        if math.isfinite(second_score)
        else float("inf")
    )
    evidence = {
        **evidence_common,
        "best_hypothesis": best_name,
        "best_score": best_result["score"],
        "second_hypothesis": second_name,
        "second_score": second_score,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "upright_median": eligible["upright"]["median"],
        "upright_p90": eligible["upright"]["p90"],
    }

    if failed_anchor_floor:
        return {
            "decision": "blocked",
            "blocking_reason": "pose_confidence_insufficient",
            "best_hypothesis": best_name,
            "scores": scores,
            "valid_anchor_total": valid_total,
            "invalid_hypotheses": diagnostic_invalid,
            "evidence": evidence,
        }

    if best_name != "upright":
        return {
            "decision": "blocked",
            "blocking_reason": f"pose_not_upright_best:{best_name}",
            "best_hypothesis": best_name,
            "scores": scores,
            "valid_anchor_total": valid_total,
            "invalid_hypotheses": diagnostic_invalid,
            "evidence": evidence,
        }

    upright_eligible = eligible["upright"]
    upright_max = (
        max(upright_eligible["residuals"]) if upright_eligible["residuals"] else 0.0
    )
    if (
        upright_eligible["median"] > SPYDER_POSE_UPRIGHT_MEDIAN_RESIDUAL_MAX
        or max(upright_eligible["p90"], upright_max) > SPYDER_POSE_UPRIGHT_TAIL_RESIDUAL_MAX
    ):
        return {
            "decision": "blocked",
            "blocking_reason": "pose_confidence_insufficient",
            "best_hypothesis": best_name,
            "scores": scores,
            "valid_anchor_total": valid_total,
            "invalid_hypotheses": diagnostic_invalid,
            "evidence": evidence,
        }

    if absolute_gap < SPYDER_POSE_ABSOLUTE_GAP_MIN or relative_gap < SPYDER_POSE_RELATIVE_GAP_MIN:
        return {
            "decision": "blocked",
            "blocking_reason": "pose_ambiguous",
            "best_hypothesis": best_name,
            "scores": scores,
            "valid_anchor_total": valid_total,
            "invalid_hypotheses": diagnostic_invalid,
            "evidence": evidence,
        }

    return {
        "decision": "upright",
        "blocking_reason": "",
        "best_hypothesis": "upright",
        "scores": scores,
        "valid_anchor_total": valid_total,
        "invalid_hypotheses": diagnostic_invalid,
        "evidence": evidence,
    }


def detect_spyder_chart_pose_problem(ratios_gray) -> tuple[bool, str, dict]:
    """geometry + pose hypothesis を統合した CCM 直前ガード.

    順序:
        1. detect_spyder_chart_misplaced で D 列単調性 / 1E-6E ratio / contrast を見る。
           失敗 → (True, geom_reason, {"stage": "geometry", "geometry_reason": geom_reason})
        2. evaluate_spydercheckr_pose_confidence で多色 anchor 比較。
           upright 受理 → (False, "ok", {"stage": "pose", "pose": pose_dict})
           それ以外 → (True, pose["blocking_reason"], {"stage": "pose", "pose": pose_dict})
    """
    geom_problem, geom_reason = detect_spyder_chart_misplaced(ratios_gray)
    if geom_problem:
        return (True, geom_reason, {"stage": "geometry", "geometry_reason": geom_reason})
    pose = evaluate_spydercheckr_pose_confidence(ratios_gray)
    if pose.get("decision") == "upright":
        return (False, "ok", {"stage": "pose", "pose": pose})
    reason = str(pose.get("blocking_reason") or "pose_confidence_insufficient")
    return (True, reason, {"stage": "pose", "pose": pose})


# ---------------------------------------------------------------------------


def make_chart_analysis_composite(
    frame_bgr: np.ndarray,
    left_panel_width: int,
) -> np.ndarray:
    """左パネル余白を付与した chart 解析用 composite を返す。"""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    if left_panel_width <= 0:
        return frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    composite = np.zeros((h, w + left_panel_width, 3), dtype=frame_bgr.dtype)
    composite[:, left_panel_width : left_panel_width + w] = frame_bgr
    return composite


@dataclass(frozen=True)
class _PatchwiseDisplayBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left + 1)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top + 1)

    @property
    def cx(self) -> float:
        return (self.left + self.right) * 0.5

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) * 0.5

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class _PatchwiseDisplayQuad:
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]

    @property
    def cx(self) -> float:
        return float(sum(point[0] for point in self.points) * 0.25)

    @property
    def cy(self) -> float:
        return float(sum(point[1] for point in self.points) * 0.25)

    @property
    def width(self) -> float:
        top = math.hypot(
            self.points[1][0] - self.points[0][0],
            self.points[1][1] - self.points[0][1],
        )
        bottom = math.hypot(
            self.points[2][0] - self.points[3][0],
            self.points[2][1] - self.points[3][1],
        )
        return float((top + bottom) * 0.5)

    @property
    def height(self) -> float:
        left = math.hypot(
            self.points[3][0] - self.points[0][0],
            self.points[3][1] - self.points[0][1],
        )
        right = math.hypot(
            self.points[2][0] - self.points[1][0],
            self.points[2][1] - self.points[1][1],
        )
        return float((left + right) * 0.5)

    @property
    def area(self) -> float:
        total = 0.0
        for point, next_point in zip(self.points, self.points[1:] + self.points[:1]):
            total += (point[0] * next_point[1]) - (next_point[0] * point[1])
        return abs(float(total)) * 0.5

    @property
    def rotation_degrees(self) -> float:
        dx = self.points[1][0] - self.points[0][0]
        dy = self.points[1][1] - self.points[0][1]
        return float(math.degrees(math.atan2(dy, dx)))

    @property
    def bounding_box(self) -> _PatchwiseDisplayBox:
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return _PatchwiseDisplayBox(
            left=int(math.floor(min(xs))),
            top=int(math.floor(min(ys))),
            right=int(math.ceil(max(xs))),
            bottom=int(math.ceil(max(ys))),
        )

    def translated(self, dx: float, dy: float) -> "_PatchwiseDisplayQuad":
        return _PatchwiseDisplayQuad(
            tuple((float(x + dx), float(y + dy)) for x, y in self.points)  # type: ignore[arg-type]
        )

    def shrunken(self, margin_ratio: float) -> "_PatchwiseDisplayQuad":
        margin = max(0.0, min(0.49, float(margin_ratio)))
        cx = self.cx
        cy = self.cy
        scale = 1.0 - (2.0 * margin)
        return _PatchwiseDisplayQuad(
            tuple(
                (
                    float(cx + ((x - cx) * scale)),
                    float(cy + ((y - cy) * scale)),
                )
                for x, y in self.points
            )  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        bbox = self.bounding_box
        return {
            "points": [
                [round(float(x), 3), round(float(y), 3)]
                for x, y in self.points
            ],
            "center": [round(float(self.cx), 3), round(float(self.cy), 3)],
            "width": round(float(self.width), 3),
            "height": round(float(self.height), 3),
            "area": round(float(self.area), 3),
            "rotation_degrees": round(float(self.rotation_degrees), 6),
            "bounding_box": bbox.as_dict(),
        }


@dataclass(frozen=True)
class SpyderCheckrOrientedPanelPayload:
    corners_xy: tuple[tuple[float, float], ...]
    centers_48_xy: tuple[tuple[float, float], ...]
    sampling_quads_48_xy: tuple[tuple[tuple[float, float], ...], ...]
    panels: tuple[dict[str, object], ...]
    visible_patch_alignment: dict[str, object]
    pair_diagnostics: dict[str, object]
    order_name: str
    color_score: float


# ---------------------------------------------------------------------------
# B: SpyderCheckrGridExtractor クラス
# ---------------------------------------------------------------------------


def _order_corners_clockwise_from_topleft(
    pts: list,
) -> list[tuple[float, float]]:
    """[Phase 22 / Phase 3] 任意順 4 点を [TL, TR, BR, BL] (時計回り) に並び替える。

    旧 SpyderCheckrGridExtractor.set_corners_with_source 内軸そろえロジックと
    bit-exact 同一: x+y 最小 = TL, x+y 最大 = BR, x-y 最大 = TR, x-y 最小 = BL。
    回転耐性は無い (axis-aligned 前提)。回転入力には新検出器内の
    `_order_corners_by_rotation` (Phase 1 1-C) を使う。

    本 helper は manual 4-click 経路のように click 順が任意な caller が、
    Phase 3 redesign 後の「順序保持入力」契約に橋渡しするためのみに使う。
    module-level 関数として class 外に置くことで unit test 容易性を確保し、
    既存 class API を肥大化させない設計判断 (planner output 設計上の重要前提)。

    Args:
      pts: 4 点 [(x, y), ...] 順不問。

    Returns:
      [TL, TR, BR, BL] (タプル) の長さ 4 list。

    Raises:
      ValueError: pts の長さが 4 でない場合。
    """
    pts_arr = [tuple(p) for p in pts]
    if len(pts_arr) != 4:
        raise ValueError(
            f"_order_corners_clockwise_from_topleft expects 4 points, "
            f"got {len(pts_arr)}"
        )
    tl = min(pts_arr, key=lambda p: p[0] + p[1])   # x+y 最小 = 左上
    br = max(pts_arr, key=lambda p: p[0] + p[1])   # x+y 最大 = 右下
    tr = max(pts_arr, key=lambda p: p[0] - p[1])   # x-y 最大 = 右上
    bl = min(pts_arr, key=lambda p: p[0] - p[1])   # x-y 最小 = 左下
    return [tl, tr, br, bl]


class SpyderCheckrGridExtractor:
    """SpyderCheckrの4コーナーから射影変換してパッチROIを抽出するクラス。

    機能:
      - 4コーナーのクリック座標から透視変換行列を算出する。
      - 全パッチ中心座標を表示空間に逆変換して返す。
      - 全パッチのBayer平均値を一括抽出する。
      - オーバーレイ（ROI矩形 + ラベル）を描画する。
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        patch_margin: float = 0.10,
        col_label_offset: int = 0,
        hinge_gap: float = 0.0,
    ):
        """フィールドを初期化する。

        Args:
          n_rows: グリッドの行数（SpyderCheckr 48 → 6）。
          n_cols: グリッドの列数（SpyderCheckr 48 → 8）。
          patch_margin: パッチ内縁マージン率（端部ノイズ回避, デフォルト 0.05 = 90%カバレッジ）。
          col_label_offset: ラベルの列オフセット（右パネルは4を指定して E〜H 表示）。
          hinge_gap: 左右パネル間のヒンジ余白（列幅単位, 例: 0.8 = 列幅の80%分）。
                     col >= n_cols//2 の正規化 x 座標にこの幅を加算する。
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.patch_margin = patch_margin  # パッチ内縁マージン率（端部ノイズ回避）
        self.col_label_offset = col_label_offset  # 右パネル用列ラベルオフセット
        self.hinge_gap = hinge_gap        # ヒンジ余白（列幅単位）
        self.corners: list = []           # 表示空間の4コーナー（時計回り）
        self.homography = None            # display → normalized 変換行列
        self.homography_inv = None        # normalized → display 逆変換行列
        self._rigid_corners_source: str | None = None
        # 列ごとの x 位置補正（正規化空間単位）: ドラッグで内側列を個別微調整
        self.col_x_norm_offsets = np.zeros(n_cols, dtype=np.float64)
        self._patchwise_boxes: list[_PatchwiseDisplayBox] | None = None
        self._patchwise_quads: list[_PatchwiseDisplayQuad] | None = None
        self._patchwise_entries: list[dict[str, object]] = []
        self._patchwise_summary: dict[str, object] | None = None
        self._last_raw_quad_sampling_summary: dict[str, object] | None = None

    def invalidate_patchwise_rois(self) -> None:
        """現在の per-patch ROI state を破棄する。"""
        self._patchwise_boxes = None
        self._patchwise_quads = None
        self._patchwise_entries = []
        self._patchwise_summary = None
        self._last_raw_quad_sampling_summary = None

    def _update_homography_from_ordered_corners(self) -> None:
        """現在の ordered corners から homography を更新する。"""
        dst_W = (self.n_cols + self.hinge_gap) * 100
        dst_H = self.n_rows * 100
        hw = 50.0   # パッチ半幅（正規化空間で常に 50 = セル幅 100 の半分）
        hh = 50.0   # パッチ半高（正規化空間で常に 50 = セル高 100 の半分）
        dst_pts = np.float32([
            [hw,          hh],           # TL = パッチ1A中心
            [dst_W - hw,  hh],           # TR = パッチ1H中心
            [dst_W - hw,  dst_H - hh],   # BR = パッチ6H中心
            [hw,          dst_H - hh],   # BL = パッチ6A中心
        ])
        src_pts = np.float32(self.corners)
        self.homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.homography_inv = np.linalg.inv(self.homography)
        self.invalidate_patchwise_rois()

    def set_corners(self, pts: list) -> None:
        self.set_corners_with_source(pts, source="legacy")

    def set_corners_with_source(self, pts: list, source: str) -> None:
        """順序保持の 4 corner で homography を更新する (rigid 検出器 / 保存復元用 entry)。

        Phase 22 / Phase 3 redesign: 旧 axis-aligned ソート (`tl = min(... key=x+y)` 等)
        は廃止。入力 4 点は呼出し側で `[TL, TR, BR, BL]` 順に並び替えて渡す契約に
        統一する。manual 4-click のように click 順が任意な caller は
        `_order_corners_clockwise_from_topleft` を呼出し側で適用すること。

        内部実装: `set_corners_ordered_with_source(pts, source=source)` への薄い
        wrapper。print のみ Phase 3 後も保持し source tracking を可視化する。
        print の意味は変わる (旧: ソート結果、新: pass-through で pts[0..3] そのまま)
        が形式は維持。

        Args:
          pts: `[TL, TR, BR, BL]` 順 (= 1A, 1H, 6H, 6A) の 4 点 [(x, y), ...]。
              表示座標系（flip 後）で渡すこと。flip による方向補正は不要。
              flip_h + flip_v 環境では表示 TL = 物理 1A なので、
              TL→1A の通常割り当てがそのまま正しい。
          source: corners の出所識別子 ("rigid_auto", "saved", "manual",
              "inner_cell_hough", "contour_outer_rim", "legacy" など)。
        """
        pts_arr = [tuple(p) for p in pts]
        if len(pts_arr) != 4:
            raise ValueError(
                f"set_corners_with_source expects 4 points, got {len(pts_arr)}"
            )
        tl, tr, br, bl = pts_arr
        # Phase 3 redesign: in= は呼出し側 CW 順契約、tl/tr/br/bl は pts[0..3] そのまま
        # (旧実装はソート結果を表示していた。形式は維持して本番デバッグ価値を残す)。
        print(
            f"[set_corners] source={source} in={pts_arr} -> "
            f"tl=({tl[0]:.1f},{tl[1]:.1f}) tr=({tr[0]:.1f},{tr[1]:.1f}) "
            f"br=({br[0]:.1f},{br[1]:.1f}) bl=({bl[0]:.1f},{bl[1]:.1f})"
        )
        self.set_corners_ordered_with_source(pts_arr, source=source)

    @property
    def is_ready(self) -> bool:
        """射影変換行列が設定済みかを返す。"""
        return self.homography is not None

    def has_rigid_quad(self) -> bool:
        """G1 source tracking: rigid-grid path に進める corners かを返す。

        Phase 2Y: `inner_cell_hough` も rigid corners 等価扱いとする
        (`_detect_chart_inner_cell_boundary_via_hough` 採用 corners は inner cell
        boundary 4 直線交点なので rigid quad として patchwise lock を適用してよい)。

        `contour_outer_rim` は実画像の patch 中心を検証する前の seed として扱う。
        水平設置でも外周輪郭だけでは ROI が patch 中央から drift するため、
        rigid-grid lock には入れず direct panel / inner lattice 検証へ回す。
        """
        return self._rigid_corners_source in {
            "rigid_auto",
            "saved",
            "inner_cell_hough",
        }

    def mark_corners_source(self, source: str) -> None:
        """Manual adjustment paths that do not reset corners can retag source."""
        self._rigid_corners_source = source

    def set_corners_ordered(self, ordered_pts: list) -> None:
        self.set_corners_ordered_with_source(ordered_pts, source="legacy")

    def set_corners_ordered_with_source(
        self,
        ordered_pts: list,
        source: str,
    ) -> None:
        """整列済み4コーナー [TL, TR, BR, BL] でホモグラフィを更新する（ドラッグ用）。

        set_corners と異なりソート処理をスキップする。
        コーナーハンドルをドラッグして位置微調整する際に呼び出す。

        Args:
          ordered_pts: [TL, TR, BR, BL] 順の表示座標リスト [(x, y), ...]。
        """
        self.corners = [tuple(p) for p in ordered_pts]
        self._rigid_corners_source = source
        self._update_homography_from_ordered_corners()

    def set_corners_ordered_with_rotation(
        self,
        ordered_pts: list,
        *,
        source: str = "contour_outer_rim",
    ) -> None:
        """順序保持の 4 corner で homography を更新する (回転対応 detector 専用 entry)。

        Phase 22 / Phase 1: 新 contour-based detector
        `_detect_chart_via_contour_outer_rim` の戻り値 (回転対応で並び替え済の
        patch CENTER 4 点) を直接 `_update_homography_from_ordered_corners`
        に渡す薄い wrapper。

        既存 `set_corners_ordered_with_source` と機能等価だが、`source` の
        default を新検出器名 `"contour_outer_rim"` に固定し、Phase 2 以降で
        saved-corners 経路と区別できるようにする。

        Args:
          ordered_pts: `[1A, 1H, 6H, 6A]` 順 (= TL, TR, BR, BL) の表示座標リスト。
            新検出器の戻り値順序と完全一致する。
          source: corners の出所識別子 (default `"contour_outer_rim"`)。
        """
        self.set_corners_ordered_with_source(ordered_pts, source=source)

    def get_patch_centers_display(self) -> list:
        """全パッチ中心の表示座標リストを返す（row-major 順）。

        Returns:
          list: [(x, y), ...] 長さ n_rows*n_cols、表示座標系。
        """
        half = self.n_cols // 2   # ヒンジ境界列インデックス（8列なら 4）
        pts_norm = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                # col >= half（右パネル）はヒンジ余白分だけ x を右シフト
                gap_offset = self.hinge_gap * 100 if col >= half else 0.0
                cx = (col + 0.5) * 100 + gap_offset + self.col_x_norm_offsets[col]
                cy = (row + 0.5) * 100
                pts_norm.append([cx, cy])
        pts_arr = np.array(pts_norm, dtype=np.float32).reshape(-1, 1, 2)
        pts_disp = cv2.perspectiveTransform(pts_arr, self.homography_inv)
        return [(float(p[0][0]), float(p[0][1])) for p in pts_disp]

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _outer_band_mask(height: int, width: int, inset: int) -> np.ndarray:
        mask = np.ones((height, width), dtype=np.uint8)
        if inset * 2 < height and inset * 2 < width:
            mask[inset : height - inset, inset : width - inset] = 0
        return mask.astype(bool)

    def _patch_label(self, row: int, col: int) -> str:
        return f"{row + 1}{chr(ord('A') + col + self.col_label_offset)}"

    @staticmethod
    def _positive_median(values: list[float], fallback: float) -> float:
        positives = [float(value) for value in values if float(value) > 0.0]
        if positives:
            return float(np.median(positives))
        return float(fallback)

    @staticmethod
    def _centered_seed_box(
        center_x: float,
        center_y: float,
        width: int,
        height: int,
    ) -> _PatchwiseDisplayBox:
        width_i = max(4, int(round(width)))
        height_i = max(4, int(round(height)))
        left = int(round(center_x - (width_i - 1) * 0.5))
        top = int(round(center_y - (height_i - 1) * 0.5))
        return _PatchwiseDisplayBox(
            left=left,
            top=top,
            right=left + width_i - 1,
            bottom=top + height_i - 1,
        )

    @staticmethod
    def _box_to_quad(box: _PatchwiseDisplayBox) -> _PatchwiseDisplayQuad:
        return _PatchwiseDisplayQuad(
            (
                (float(box.left), float(box.top)),
                (float(box.right), float(box.top)),
                (float(box.right), float(box.bottom)),
                (float(box.left), float(box.bottom)),
            )
        )

    @staticmethod
    def _centered_oriented_quad(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> _PatchwiseDisplayQuad:
        half_w = max(2.0, float(width) * 0.5)
        half_h = max(2.0, float(height) * 0.5)
        ux = np.asarray(x_axis, dtype=np.float64)
        uy = np.asarray(y_axis, dtype=np.float64)
        ux_norm = float(np.linalg.norm(ux))
        uy_norm = float(np.linalg.norm(uy))
        if ux_norm <= 1e-9:
            ux = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            ux = ux / ux_norm
        if uy_norm <= 1e-9:
            uy = np.asarray([0.0, 1.0], dtype=np.float64)
        else:
            uy = uy / uy_norm
        center = np.asarray([float(center_x), float(center_y)], dtype=np.float64)
        top_left = center - (half_w * ux) - (half_h * uy)
        top_right = center + (half_w * ux) - (half_h * uy)
        bottom_right = center + (half_w * ux) + (half_h * uy)
        bottom_left = center - (half_w * ux) + (half_h * uy)
        return _PatchwiseDisplayQuad(
            (
                (float(top_left[0]), float(top_left[1])),
                (float(top_right[0]), float(top_right[1])),
                (float(bottom_right[0]), float(bottom_right[1])),
                (float(bottom_left[0]), float(bottom_left[1])),
            )
        )

    @staticmethod
    def _axes_from_display_quad(
        quad: _PatchwiseDisplayQuad,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        points = np.asarray(quad.points, dtype=np.float64)
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            return None
        x_axis = points[1] - points[0]
        y_axis = points[3] - points[0]
        x_norm = float(np.linalg.norm(x_axis))
        y_norm = float(np.linalg.norm(y_axis))
        if x_norm <= 1e-9 or y_norm <= 1e-9:
            return None
        return x_axis / x_norm, y_axis / y_norm

    @staticmethod
    def _rasterized_quad_mask_area(points: object) -> int | None:
        try:
            pts = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            return None
        left = int(math.floor(float(np.min(pts[:, 0]))))
        top = int(math.floor(float(np.min(pts[:, 1]))))
        right = int(math.ceil(float(np.max(pts[:, 0]))))
        bottom = int(math.ceil(float(np.max(pts[:, 1]))))
        width = max(1, right - left + 1)
        height = max(1, bottom - top + 1)
        local = np.rint(pts - np.asarray([left, top], dtype=np.float64)).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local, 255)
        return int(np.count_nonzero(mask))

    @staticmethod
    def _orthonormal_axes_from_rotation_degrees(
        rotation_degrees: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = math.radians(float(rotation_degrees))
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return (
            np.asarray([cos_theta, sin_theta], dtype=np.float64),
            np.asarray([-sin_theta, cos_theta], dtype=np.float64),
        )

    @staticmethod
    def _direct_panel_square_face_side(
        component_widths: list[float],
        component_heights: list[float],
        legacy_box_width: float,
        legacy_box_height: float,
        col_step: float,
        row_step: float,
    ) -> float:
        if len(component_widths) != len(component_heights):
            raise ValueError("component width/height lists must be paired")
        component_sides = [
            min(float(width), float(height))
            for width, height in zip(component_widths, component_heights)
            if float(width) > 0.0 and float(height) > 0.0
        ]
        fallback_side = max(1.0, min(float(legacy_box_width), float(legacy_box_height)))
        component_side = (
            float(np.median(component_sides))
            if component_sides
            else fallback_side
        )
        pitch_side = max(1.0, min(float(col_step), float(row_step)))
        capped_side = min(
            max(component_side, pitch_side * 0.80),
            pitch_side * SPYDERCHECKR_DIRECT_PANEL_FACE_SIZE_CAP_RATIO,
        )
        if pitch_side * SPYDERCHECKR_DIRECT_PANEL_FACE_SIZE_CAP_RATIO < 12.0:
            return max(4.0, capped_side)
        return max(12.0, capped_side)

    def _default_patch_display_boxes(self) -> list[_PatchwiseDisplayBox]:
        if not self.is_ready:
            return []
        centers = self.get_patch_centers_display()
        dx_values = []
        dy_values = []
        for row in range(self.n_rows):
            for col in range(self.n_cols - 1):
                idx = row * self.n_cols + col
                dx_values.append(abs(centers[idx + 1][0] - centers[idx][0]))
        for row in range(self.n_rows - 1):
            for col in range(self.n_cols):
                idx = row * self.n_cols + col
                dy_values.append(abs(centers[idx + self.n_cols][1] - centers[idx][1]))
        dx = float(np.median(dx_values)) if dx_values else 0.0
        dy = float(np.median(dy_values)) if dy_values else 0.0
        positive = [value for value in (dx, dy) if value > 0.0]
        cell_disp = min(positive) if positive else 0.0
        patch_size_disp = max(4, int(round(cell_disp * (1.0 - 2.0 * self.patch_margin))))
        half = patch_size_disp // 2
        out: list[_PatchwiseDisplayBox] = []
        for cx, cy in centers:
            left = int(round(cx)) - half
            top = int(round(cy)) - half
            out.append(
                _PatchwiseDisplayBox(
                    left=left,
                    top=top,
                    right=left + patch_size_disp - 1,
                    bottom=top + patch_size_disp - 1,
                )
            )
        return out

    def get_active_patch_display_boxes(self) -> list[_PatchwiseDisplayBox]:
        if self._patchwise_boxes is not None and len(self._patchwise_boxes) == self.n_rows * self.n_cols:
            return list(self._patchwise_boxes)
        return self._default_patch_display_boxes()

    def _default_patch_display_quads(self) -> list[_PatchwiseDisplayQuad]:
        """Phase 15: saved corners 由来の homography から rotated quad を生成する fallback.

        `_default_patch_display_boxes()` は各 patch を **軸合わせ** 矩形で返すため、
        chart が物理的に傾いていると黄色 hull (saved corners 由来 → tilted) と内側パッチ
        (axis-aligned) が視覚的に不一致になる。本関数は normalized 座標で 4 corner を組み立て、
        `homography_inv` で display 空間に projective 変換することで、saved corners と同じ傾きを
        継承した quad を返す。`patch_margin` も反映するため CCM extraction の ROI とほぼ同形。
        """
        if not self.is_ready:
            return []
        half_norm = 50.0 * max(0.05, 1.0 - 2.0 * float(self.patch_margin))
        half = self.n_cols // 2
        pts_norm: list[list[float]] = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                gap_offset = self.hinge_gap * 100 if col >= half else 0.0
                cx = (col + 0.5) * 100 + gap_offset + self.col_x_norm_offsets[col]
                cy = (row + 0.5) * 100
                # 4 corner: top-left, top-right, bottom-right, bottom-left の順
                pts_norm.append([cx - half_norm, cy - half_norm])
                pts_norm.append([cx + half_norm, cy - half_norm])
                pts_norm.append([cx + half_norm, cy + half_norm])
                pts_norm.append([cx - half_norm, cy + half_norm])
        pts_arr = np.array(pts_norm, dtype=np.float32).reshape(-1, 1, 2)
        pts_disp = cv2.perspectiveTransform(pts_arr, self.homography_inv)
        out: list[_PatchwiseDisplayQuad] = []
        for idx in range(self.n_rows * self.n_cols):
            base = idx * 4
            corners = tuple(
                (float(pts_disp[base + k][0][0]), float(pts_disp[base + k][0][1]))
                for k in range(4)
            )
            out.append(_PatchwiseDisplayQuad(corners))
        return out

    def _rigid_patch_axes_from_corners(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self.corners) != 4:
            return None
        pts = np.asarray(self.corners, dtype=np.float64)
        x_vec = ((pts[1] - pts[0]) + (pts[2] - pts[3])) * 0.5
        y_hint = ((pts[3] - pts[0]) + (pts[2] - pts[1])) * 0.5
        x_norm = float(np.linalg.norm(x_vec))
        if x_norm <= 1e-9:
            return None
        x_axis = x_vec / x_norm
        y_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=np.float64)
        if float(np.dot(y_axis, y_hint)) < 0.0:
            y_axis = -y_axis
        return x_axis, y_axis

    def _rigid_patch_square_side_from_centers(
        self,
        centers: list[tuple[float, float]],
    ) -> float:
        if len(centers) != self.n_rows * self.n_cols:
            return 4.0
        centers_arr = np.asarray(centers, dtype=np.float64).reshape(
            self.n_rows,
            self.n_cols,
            2,
        )
        distances: list[float] = []
        hinge_left_col = (self.n_cols // 2) - 1
        for row in range(self.n_rows):
            for col in range(self.n_cols - 1):
                if col == hinge_left_col:
                    continue
                distances.append(
                    float(
                        np.linalg.norm(
                            centers_arr[row, col + 1] - centers_arr[row, col]
                        )
                    )
                )
        for row in range(self.n_rows - 1):
            for col in range(self.n_cols):
                distances.append(
                    float(
                        np.linalg.norm(
                            centers_arr[row + 1, col] - centers_arr[row, col]
                        )
                    )
                )
        pitch = self._positive_median(distances, fallback=4.0)
        coverage = max(0.05, 1.0 - 2.0 * float(self.patch_margin))
        return max(4.0, float(pitch) * coverage)

    def _rigid_patch_quads_from_homography(self) -> list[_PatchwiseDisplayQuad]:
        """Generate all 48 patch quads from one rigid square-face lock.

        Centers are projected once from the grid homography, but every patch
        footprint uses the same chart rotation and the same square side length.
        This keeps all 48 ROIs rotating with the SpyderCHECKR while preventing
        per-patch drift and homography-induced non-square ROI shapes.
        """
        if not self.is_ready or self.homography_inv is None:
            return []
        centers = self.get_patch_centers_display()
        if len(centers) != self.n_rows * self.n_cols:
            return []
        axes = self._rigid_patch_axes_from_corners()
        if axes is None:
            return []
        x_axis, y_axis = axes
        side = self._rigid_patch_square_side_from_centers(centers)
        return [
            self._centered_oriented_quad(cx, cy, side, side, x_axis, y_axis)
            for cx, cy in centers
        ]

    def _rigid_patch_boxes_from_quads(
        self,
        quads: list[_PatchwiseDisplayQuad],
    ) -> list[_PatchwiseDisplayBox]:
        """Return axis-aligned boxes as a legacy compatibility layer.

        Color sampling and overlay code still have box-based paths, so rigid
        quads expose their bounding boxes for that legacy code without changing
        the rigid quad geometry.
        """
        return [quad.bounding_box for quad in quads]

    def get_active_patch_display_quads(self) -> list[_PatchwiseDisplayQuad]:
        """active patch quads を返す。優先順位:

        1. `_patchwise_quads` (rotated, refined) — 最優先
        2. `_patchwise_boxes` (axis-aligned, refined position) — quad に変換して返す
        3. `_default_patch_display_quads()` (homography 経由 rotated, Phase 15 fallback)
        """
        if self._patchwise_quads is not None and len(self._patchwise_quads) == self.n_rows * self.n_cols:
            return list(self._patchwise_quads)
        if self._patchwise_boxes is not None and len(self._patchwise_boxes) == self.n_rows * self.n_cols:
            return [self._box_to_quad(box) for box in self._patchwise_boxes]
        return self._default_patch_display_quads()

    def _active_lattice_hull_for_col_range_from_quads(
        self,
        quads: list[_PatchwiseDisplayQuad],
        first_col: int,
        last_col: int,
    ) -> np.ndarray | None:
        corner_specs = (
            (first_col, 0),
            (last_col, 1),
            ((self.n_rows - 1) * self.n_cols + last_col, 2),
            ((self.n_rows - 1) * self.n_cols + first_col, 3),
        )
        points: list[list[int]] = []
        for quad_index, point_index in corner_specs:
            if quad_index < 0 or quad_index >= len(quads):
                return None
            quad = quads[quad_index]
            if len(quad.points) != 4:
                return None
            x, y = quad.points[point_index]
            if not (np.isfinite(x) and np.isfinite(y)):
                return None
            points.append([int(round(float(x))), int(round(float(y)))])
        return np.asarray(points, dtype=np.int32)

    def _active_lattice_panel_hulls_from_quads(self) -> list[np.ndarray]:
        """active patch quads の panel 外周を表示座標で返す。

        self.corners は 4 隅パッチ中心のハンドル位置であり、chart 外枠ではない。
        preview の dim mask / 黄色 frame は、現在採用中の patchwise ROI geometry
        から作る。SpyderCHECKR 48 は左右 2 panel の物理構造なので、ヒンジを
        またぐ single hull ではなく panel ごとの hull を返す。
        """
        expected_count = int(self.n_rows * self.n_cols)
        if expected_count <= 0 or self.n_rows <= 0 or self.n_cols <= 0:
            return []
        quads = self.get_active_patch_display_quads()
        if len(quads) != expected_count:
            return []

        if (self.n_rows, self.n_cols) == SPYDERCHECKR_48_SHAPE:
            left_hull = self._active_lattice_hull_for_col_range_from_quads(quads, 0, 3)
            right_hull = self._active_lattice_hull_for_col_range_from_quads(quads, 4, 7)
            if left_hull is None or right_hull is None:
                return []
            return [left_hull, right_hull]

        hull = self._active_lattice_hull_for_col_range_from_quads(
            quads,
            0,
            self.n_cols - 1,
        )
        return [hull] if hull is not None else []

    @staticmethod
    def _box_dict_to_panel_hull(box: object) -> np.ndarray | None:
        if not isinstance(box, dict):
            return None
        try:
            left = float(box["left"])
            top = float(box["top"])
            right = float(box["right"])
            bottom = float(box["bottom"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(np.isfinite(value) for value in (left, top, right, bottom)):
            return None
        return np.asarray(
            [
                [int(round(left)), int(round(top))],
                [int(round(right)), int(round(top))],
                [int(round(right)), int(round(bottom))],
                [int(round(left)), int(round(bottom))],
            ],
            dtype=np.int32,
        )

    def _direct_panel_face_hull_from_layout(
        self,
        layout: object,
    ) -> np.ndarray | None:
        """direct-panel 診断から黒フレーム内側の patch-face 外周を復元する。"""
        if not isinstance(layout, dict):
            return None
        panel_box = layout.get("panel_box")
        if not isinstance(panel_box, dict):
            return None
        try:
            panel_left = float(panel_box["left"])
            panel_top = float(panel_box["top"])
        except (KeyError, TypeError, ValueError):
            return None

        origin_raw = layout.get("lattice_origin_local")
        col_raw = layout.get("lattice_col_vector_local")
        row_raw = layout.get("lattice_row_vector_local")
        try:
            origin = np.asarray(origin_raw, dtype=np.float64)
            col_vector = np.asarray(col_raw, dtype=np.float64)
            row_vector = np.asarray(row_raw, dtype=np.float64)
            face_width = float(
                layout.get("estimated_face_width", layout.get("estimated_face_side"))
            )
            face_height = float(
                layout.get("estimated_face_height", layout.get("estimated_face_side"))
            )
        except (TypeError, ValueError):
            origin = np.asarray([], dtype=np.float64)
            col_vector = np.asarray([], dtype=np.float64)
            row_vector = np.asarray([], dtype=np.float64)
            face_width = 0.0
            face_height = 0.0

        if (
            origin.shape == (2,)
            and col_vector.shape == (2,)
            and row_vector.shape == (2,)
            and face_width > 0.0
            and face_height > 0.0
            and all(
                np.isfinite(value)
                for value in (*origin, *col_vector, *row_vector)
            )
        ):
            col_norm = float(np.linalg.norm(col_vector))
            row_norm = float(np.linalg.norm(row_vector))
            if col_norm > 1e-6 and row_norm > 1e-6:
                x_axis = col_vector / col_norm
                y_axis = row_vector / row_norm
                panel_cols = max(1, self.n_cols // 2)
                last_center = origin + ((panel_cols - 1) * col_vector)
                bottom_left_center = origin + ((self.n_rows - 1) * row_vector)
                bottom_right_center = last_center + (
                    (self.n_rows - 1) * row_vector
                )
                local_points = (
                    origin
                    - (0.5 * face_width * x_axis)
                    - (0.5 * face_height * y_axis),
                    last_center
                    + (0.5 * face_width * x_axis)
                    - (0.5 * face_height * y_axis),
                    bottom_right_center
                    + (0.5 * face_width * x_axis)
                    + (0.5 * face_height * y_axis),
                    bottom_left_center
                    - (0.5 * face_width * x_axis)
                    + (0.5 * face_height * y_axis),
                )
                points = [
                    [
                        int(round(float(point[0] + panel_left))),
                        int(round(float(point[1] + panel_top))),
                    ]
                    for point in local_points
                ]
                return np.asarray(points, dtype=np.int32)

        inner_box = layout.get("inner_signal_box_local")
        if isinstance(inner_box, dict):
            try:
                return np.asarray(
                    [
                        [
                            int(round(panel_left + float(inner_box["left"]))),
                            int(round(panel_top + float(inner_box["top"]))),
                        ],
                        [
                            int(round(panel_left + float(inner_box["right"]))),
                            int(round(panel_top + float(inner_box["top"]))),
                        ],
                        [
                            int(round(panel_left + float(inner_box["right"]))),
                            int(round(panel_top + float(inner_box["bottom"]))),
                        ],
                        [
                            int(round(panel_left + float(inner_box["left"]))),
                            int(round(panel_top + float(inner_box["bottom"]))),
                        ],
                    ],
                    dtype=np.int32,
                )
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def _active_panel_frame_hulls_for_overlay(self) -> list[np.ndarray]:
        """黄色 frame 用の panel 境界を返す。

        ROI quads は測定用に内側へ shrink するため、preview の黄色 frame には
        direct-panel が検出した patch-face 外周を優先する。
        """
        summary = (
            self._patchwise_summary
            if isinstance(self._patchwise_summary, dict)
            else None
        )
        diagnostics = (
            summary.get("direct_dark_panel_diagnostics")
            if isinstance(summary, dict)
            else None
        )
        if (
            isinstance(diagnostics, dict)
            and diagnostics.get("status") == "ok"
            and (self.n_rows, self.n_cols) == SPYDERCHECKR_48_SHAPE
        ):
            hulls = [
                self._direct_panel_face_hull_from_layout(
                    diagnostics.get("left_panel_layout")
                ),
                self._direct_panel_face_hull_from_layout(
                    diagnostics.get("right_panel_layout")
                ),
            ]
            if all(hull is not None for hull in hulls):
                return [hull for hull in hulls if hull is not None]

            panel_pair = diagnostics.get("panel_pair")
            if isinstance(panel_pair, dict):
                fallback_hulls = [
                    self._box_dict_to_panel_hull(panel_pair.get("left_box")),
                    self._box_dict_to_panel_hull(panel_pair.get("right_box")),
                ]
                if all(hull is not None for hull in fallback_hulls):
                    return [hull for hull in fallback_hulls if hull is not None]

        return self._active_lattice_panel_hulls_from_quads()

    def _active_lattice_hull_from_quads(self) -> np.ndarray | None:
        """single-panel active lattice hull を返す。

        Split-panel SpyderCHECKR 48 は plural helper を使う。ここでは legacy
        single-hull callers がヒンジをまたぐ hull を再生成しないよう None を返す。
        """
        hulls = self._active_lattice_panel_hulls_from_quads()
        if len(hulls) != 1:
            return None
        return hulls[0]

    def get_patchwise_summary(self) -> dict[str, object] | None:
        if self._patchwise_summary is None:
            return None
        return dict(self._patchwise_summary)

    def get_patchwise_entries(self) -> list[dict[str, object]]:
        return [dict(entry) for entry in self._patchwise_entries]

    def get_last_raw_quad_sampling_summary(self) -> dict[str, object] | None:
        if self._last_raw_quad_sampling_summary is None:
            return None
        return dict(self._last_raw_quad_sampling_summary)

    def _build_patchwise_seed_info(
        self,
        default_boxes: list[_PatchwiseDisplayBox],
    ) -> tuple[dict[str, dict[str, object]], list[str]]:
        seeds_by_label: dict[str, dict[str, object]] = {}
        ordered_labels: list[str] = []
        half = self.n_cols // 2
        for idx, seed_box in enumerate(default_boxes):
            row = idx // self.n_cols
            col = idx % self.n_cols
            label = self._patch_label(row, col)
            seeds_by_label[label] = {
                "row_index": row,
                "col_index": col,
                "panel_col_index": col if col < half else col - half,
                "is_right_panel": col >= half,
                "legacy_seed_box": seed_box,
                "seed_box": seed_box,
            }
            ordered_labels.append(label)

        label_by_pos = {
            (int(info["row_index"]), int(info["col_index"])): label
            for label, info in seeds_by_label.items()
        }
        panel_col_steps: dict[bool, list[float]] = {False: [], True: []}
        panel_row_steps: dict[bool, list[float]] = {False: [], True: []}
        global_col_steps: list[float] = []
        global_row_steps: list[float] = []

        for label, info in seeds_by_label.items():
            row_index = int(info["row_index"])
            col_index = int(info["col_index"])
            panel_col_index = int(info["panel_col_index"])
            is_right_panel = bool(info["is_right_panel"])
            seed_box = info["seed_box"]
            if panel_col_index < max(0, half - 1):
                next_label = label_by_pos.get((row_index, col_index + 1))
                if next_label is not None:
                    next_info = seeds_by_label[next_label]
                    if bool(next_info["is_right_panel"]) == is_right_panel:
                        step = abs(next_info["seed_box"].cx - seed_box.cx)
                        panel_col_steps[is_right_panel].append(float(step))
                        global_col_steps.append(float(step))
            next_label = label_by_pos.get((row_index + 1, col_index))
            if next_label is not None:
                next_info = seeds_by_label[next_label]
                step = abs(next_info["seed_box"].cy - seed_box.cy)
                panel_row_steps[is_right_panel].append(float(step))
                global_row_steps.append(float(step))

        global_col_pitch = self._positive_median(global_col_steps, 0.0)
        global_row_pitch = self._positive_median(global_row_steps, 0.0)

        for info in seeds_by_label.values():
            legacy_seed_box = info["legacy_seed_box"]
            is_right_panel = bool(info["is_right_panel"])
            info["predicted_cx"] = float(legacy_seed_box.cx)
            info["predicted_cy"] = float(legacy_seed_box.cy)
            info["col_pitch"] = self._positive_median(
                panel_col_steps[is_right_panel],
                global_col_pitch if global_col_pitch > 0.0 else float(legacy_seed_box.width),
            )
            info["row_pitch"] = self._positive_median(
                panel_row_steps[is_right_panel],
                global_row_pitch if global_row_pitch > 0.0 else float(legacy_seed_box.height),
            )
            standard_width = max(
                legacy_seed_box.width,
                int(round(float(info["col_pitch"]) * SPYDERCHECKR_PATCHWISE_STANDARD_SIZE_FLOOR_RATIO)),
            )
            standard_height = max(
                legacy_seed_box.height,
                int(round(float(info["row_pitch"]) * SPYDERCHECKR_PATCHWISE_STANDARD_SIZE_FLOOR_RATIO)),
            )
            rescue_min_width = max(
                legacy_seed_box.width,
                int(round(float(info["col_pitch"]) * SPYDERCHECKR_PATCHWISE_RESCUE_SIZE_FLOOR_RATIO)),
            )
            rescue_min_height = max(
                legacy_seed_box.height,
                int(round(float(info["row_pitch"]) * SPYDERCHECKR_PATCHWISE_RESCUE_SIZE_FLOOR_RATIO)),
            )
            info["seed_box"] = self._centered_seed_box(
                float(info["predicted_cx"]),
                float(info["predicted_cy"]),
                standard_width,
                standard_height,
            )
            info["rescue_min_width"] = int(rescue_min_width)
            info["rescue_min_height"] = int(rescue_min_height)
        return seeds_by_label, ordered_labels

    @staticmethod
    def _panel_hull_box_from_points(points: object) -> _PatchwiseDisplayBox | None:
        try:
            arr = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if arr.shape != (4, 2) or not np.all(np.isfinite(arr)):
            return None
        left = int(round(float(np.min(arr[:, 0]))))
        right = int(round(float(np.max(arr[:, 0]))))
        top = int(round(float(np.min(arr[:, 1]))))
        bottom = int(round(float(np.max(arr[:, 1]))))
        if right < left or bottom < top:
            return None
        return _PatchwiseDisplayBox(left=left, top=top, right=right, bottom=bottom)

    @staticmethod
    def _structured_oriented_payload_failure_summary(
        *,
        patch_count: int,
        reason: str,
    ) -> dict[str, object]:
        geometry_viability = _evaluate_patchwise_calibration_geometry(
            status="fallback",
            patch_count=patch_count,
            rescue_patch_count=0,
            fallback_patch_count=patch_count,
            weak_patch_count=0,
            median_area_ratio_to_cell_pitch_area=0.0,
        )
        summary = {
            "status": "fallback",
            "reason": reason,
            "refined_patch_count": 0,
            "rescue_patch_count": 0,
            "fallback_patch_count": patch_count,
            "weak_patch_count": 0,
            "direct_patch_count": 0,
            "geometry_model": "direct_dark_panel",
            "center_tolerance_ratio": SPYDERCHECKR_PATCHWISE_CENTER_TOLERANCE_RATIO,
            "max_center_deviation_ratio": 0.0,
            "mean_center_deviation_ratio": 0.0,
            "median_width_ratio_to_col_pitch": 0.0,
            "median_height_ratio_to_row_pitch": 0.0,
            "median_area_ratio_to_cell_pitch_area": 0.0,
            "failing_center_guard_patch_count": 0,
            "direct_dark_panel_diagnostics": {
                "status": "failed",
                "reason": reason,
            },
        }
        summary.update(geometry_viability)
        return summary

    @staticmethod
    def _panel_pitch_medians(
        centers: np.ndarray,
    ) -> tuple[float, float]:
        col_steps = [
            float(np.linalg.norm(centers[row, col + 1] - centers[row, col]))
            for row in range(centers.shape[0])
            for col in range(centers.shape[1] - 1)
        ]
        row_steps = [
            float(np.linalg.norm(centers[row + 1, col] - centers[row, col]))
            for row in range(centers.shape[0] - 1)
            for col in range(centers.shape[1])
        ]
        return (
            SpyderCheckrGridExtractor._positive_median(col_steps, 0.0),
            SpyderCheckrGridExtractor._positive_median(row_steps, 0.0),
        )

    @staticmethod
    def _coerce_oriented_payload_quad(
        quad_points: object,
    ) -> _PatchwiseDisplayQuad | None:
        try:
            arr = np.asarray(quad_points, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if arr.shape != (4, 2) or not np.all(np.isfinite(arr)):
            return None
        return _PatchwiseDisplayQuad(
            tuple((float(point[0]), float(point[1])) for point in arr)
        )

    @staticmethod
    def _component_boxes(mask: np.ndarray) -> list[tuple[_PatchwiseDisplayBox, int, float]]:
        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        boxes: list[tuple[_PatchwiseDisplayBox, int, float]] = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if w <= 0 or h <= 0 or area <= 0:
                continue
            box = _PatchwiseDisplayBox(x, y, x + w - 1, y + h - 1)
            occupancy = area / max(1.0, float(w * h))
            boxes.append((box, area, float(occupancy)))
        return boxes

    @staticmethod
    def _expand_display_box(
        box: _PatchwiseDisplayBox,
        dx: int,
        dy: int,
        image_w: int,
        image_h: int,
    ) -> _PatchwiseDisplayBox:
        return _PatchwiseDisplayBox(
            left=max(0, box.left - dx),
            top=max(0, box.top - dy),
            right=min(image_w - 1, box.right + dx),
            bottom=min(image_h - 1, box.bottom + dy),
        )

    def _panel_search_region_from_seed_boxes(
        self,
        default_boxes: list[_PatchwiseDisplayBox],
        *,
        is_right_panel: bool,
        image_w: int,
        image_h: int,
    ) -> dict[str, object]:
        half = self.n_cols // 2
        start_col = half if is_right_panel else 0
        stop_col = self.n_cols if is_right_panel else half
        panel_boxes = [
            default_boxes[row * self.n_cols + col]
            for row in range(self.n_rows)
            for col in range(start_col, stop_col)
        ]
        left = min(box.left for box in panel_boxes)
        top = min(box.top for box in panel_boxes)
        right = max(box.right for box in panel_boxes)
        bottom = max(box.bottom for box in panel_boxes)
        span_w = max(1, right - left + 1)
        span_h = max(1, bottom - top + 1)
        center_xs = [float(box.cx) for box in panel_boxes]
        center_ys = [float(box.cy) for box in panel_boxes]
        widths = [float(box.width) for box in panel_boxes]
        heights = [float(box.height) for box in panel_boxes]
        x_pad = max(10, int(round(np.median(widths) * 1.6)))
        y_pad = max(10, int(round(np.median(heights) * 1.8)))
        search_box = self._expand_display_box(
            _PatchwiseDisplayBox(left, top, right, bottom),
            dx=x_pad,
            dy=y_pad,
            image_w=image_w,
            image_h=image_h,
        )
        return {
            "search_box": search_box,
            "predicted_centers": list(zip(center_xs, center_ys)),
            "expected_width": int(span_w),
            "expected_height": int(span_h),
        }

    def _detect_dark_panel_box(
        self,
        gray: np.ndarray,
        panel_region: dict[str, object],
    ) -> tuple[_PatchwiseDisplayBox | None, dict[str, object]]:
        search_box = panel_region["search_box"]
        predicted_centers = list(panel_region["predicted_centers"])
        expected_width = max(1.0, float(panel_region["expected_width"]))
        expected_height = max(1.0, float(panel_region["expected_height"]))
        roi = gray[search_box.top : search_box.bottom + 1, search_box.left : search_box.right + 1]
        if roi.size == 0:
            return None, {
                "status": "failed",
                "reason": "empty_search_roi",
                "search_box": search_box.as_dict(),
            }

        kernel_w = max(5, int(round(expected_width / max(self.n_cols // 2, 1) * 0.45)))
        kernel_h = max(5, int(round(expected_height / max(self.n_rows, 1) * 0.45)))
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
        kernel = np.ones((kernel_h, kernel_w), dtype=np.uint8)

        best_box: _PatchwiseDisplayBox | None = None
        best_score = -1.0
        best_diag: dict[str, object] | None = None
        for threshold in SPYDERCHECKR_DIRECT_PANEL_DARK_THRESHOLDS:
            dark_mask = (roi <= threshold).astype(np.uint8)
            dark_mask = cv2.morphologyEx(
                dark_mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )
            dark_mask = cv2.dilate(dark_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            for local_box, area, occupancy in self._component_boxes(dark_mask):
                candidate = _PatchwiseDisplayBox(
                    left=local_box.left + search_box.left,
                    top=local_box.top + search_box.top,
                    right=local_box.right + search_box.left,
                    bottom=local_box.bottom + search_box.top,
                )
                if candidate.width < expected_width * 0.45 or candidate.height < expected_height * 0.55:
                    continue
                aspect = candidate.height / max(1.0, float(candidate.width))
                if not (1.0 <= aspect <= 2.4):
                    continue
                coverage = sum(
                    candidate.left <= cx <= candidate.right and candidate.top <= cy <= candidate.bottom
                    for cx, cy in predicted_centers
                ) / max(len(predicted_centers), 1)
                if coverage < 0.55:
                    continue
                width_score = self._clamp(
                    1.0 - abs(candidate.width - expected_width) / max(1.0, expected_width * 0.40),
                    0.0,
                    1.0,
                )
                height_score = self._clamp(
                    1.0 - abs(candidate.height - expected_height) / max(1.0, expected_height * 0.35),
                    0.0,
                    1.0,
                )
                occupancy_score = self._clamp((occupancy - 0.35) / 0.45, 0.0, 1.0)
                darkness = float(np.mean(roi[local_box.top : local_box.bottom + 1, local_box.left : local_box.right + 1]))
                darkness_score = self._clamp((145.0 - darkness) / 95.0, 0.0, 1.0)
                score = (
                    (0.55 * coverage)
                    + (0.15 * width_score)
                    + (0.12 * height_score)
                    + (0.10 * occupancy_score)
                    + (0.08 * darkness_score)
                )
                if score > best_score:
                    best_score = float(score)
                    best_box = candidate
                    best_diag = {
                        "status": "ok",
                        "dark_threshold": int(threshold),
                        "score": round(float(score), 6),
                        "coverage": round(float(coverage), 6),
                        "occupancy": round(float(occupancy), 6),
                        "mean_gray": round(float(darkness), 6),
                        "search_box": search_box.as_dict(),
                        "panel_box": candidate.as_dict(),
                    }

        if best_box is None or best_diag is None:
            return None, {
                "status": "failed",
                "reason": "no_panel_component",
                "search_box": search_box.as_dict(),
            }
        return best_box, best_diag

    @staticmethod
    def _make_local_inner_box(
        width: int,
        height: int,
        margin_ratio: float,
    ) -> _PatchwiseDisplayBox:
        dx = max(4, int(round(width * margin_ratio)))
        dy = max(4, int(round(height * margin_ratio)))
        return _PatchwiseDisplayBox(
            left=dx,
            top=dy,
            right=max(dx, width - 1 - dx),
            bottom=max(dy, height - 1 - dy),
        )

    @staticmethod
    def _centered_local_box(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        bounds_width: int,
        bounds_height: int,
    ) -> _PatchwiseDisplayBox:
        width_i = max(4, int(round(width)))
        height_i = max(4, int(round(height)))
        left = int(round(center_x - (width_i - 1) * 0.5))
        top = int(round(center_y - (height_i - 1) * 0.5))
        right = left + width_i - 1
        bottom = top + height_i - 1
        return _PatchwiseDisplayBox(
            left=max(0, left),
            top=max(0, top),
            right=min(bounds_width - 1, right),
            bottom=min(bounds_height - 1, bottom),
        )

    @staticmethod
    def _inset_display_box(
        box: _PatchwiseDisplayBox,
        dx: int,
        dy: int,
    ) -> _PatchwiseDisplayBox:
        max_dx = max(0, (box.width - 3) // 2)
        max_dy = max(0, (box.height - 3) // 2)
        dx = max(0, min(dx, max_dx))
        dy = max(0, min(dy, max_dy))
        return _PatchwiseDisplayBox(
            left=box.left + dx,
            top=box.top + dy,
            right=box.right - dx,
            bottom=box.bottom - dy,
        )

    @staticmethod
    def _translate_local_box(
        panel_box: _PatchwiseDisplayBox,
        local_box: _PatchwiseDisplayBox,
    ) -> _PatchwiseDisplayBox:
        return _PatchwiseDisplayBox(
            left=panel_box.left + local_box.left,
            top=panel_box.top + local_box.top,
            right=panel_box.left + local_box.right,
            bottom=panel_box.top + local_box.bottom,
        )

    @staticmethod
    def _cluster_axis_positions(values: list[float], cluster_count: int) -> list[float]:
        if len(values) < cluster_count:
            raise RuntimeError(
                f"Need at least {cluster_count} samples, got {len(values)}."
            )

        ordered_values = sorted(float(value) for value in values)
        lo = ordered_values[0]
        hi = ordered_values[-1]
        if hi <= lo:
            return [lo for _ in range(cluster_count)]

        centers = [
            lo + (hi - lo) * idx / max(1, cluster_count - 1)
            for idx in range(cluster_count)
        ]
        for _ in range(32):
            groups: list[list[float]] = [[] for _ in range(cluster_count)]
            for value in ordered_values:
                best_idx = min(
                    range(cluster_count),
                    key=lambda idx: (abs(value - centers[idx]), idx),
                )
                groups[best_idx].append(value)

            next_centers: list[float] = []
            for idx, group in enumerate(groups):
                if group:
                    next_centers.append(float(np.mean(group)))
                elif idx == 0:
                    next_centers.append(lo)
                elif idx == cluster_count - 1:
                    next_centers.append(hi)
                else:
                    next_centers.append((centers[idx - 1] + centers[idx + 1]) * 0.5)

            next_centers.sort()
            if max(abs(old - new) for old, new in zip(centers, next_centers)) < 1e-4:
                centers = next_centers
                break
            centers = next_centers

        return centers

    @staticmethod
    def _fit_regular_centers(raw_centers: list[float]) -> tuple[list[float], float, float]:
        count = len(raw_centers)
        if count == 0:
            raise RuntimeError("Cannot regularize an empty center list.")
        if count == 1:
            return list(raw_centers), 0.0, 0.0

        mean_idx = (count - 1) * 0.5
        mean_center = float(np.mean(raw_centers))
        denom = sum((idx - mean_idx) ** 2 for idx in range(count))
        slope = sum(
            (idx - mean_idx) * (center - mean_center)
            for idx, center in enumerate(raw_centers)
        ) / max(1e-9, denom)
        intercept = mean_center - slope * mean_idx
        fitted = [intercept + slope * idx for idx in range(count)]
        rmse = math.sqrt(
            sum((raw - fit) ** 2 for raw, fit in zip(raw_centers, fitted)) / count
        )
        return fitted, slope, rmse

    @staticmethod
    def _nearest_index(values: list[float], target: float) -> int:
        return min(
            range(len(values)),
            key=lambda idx: (abs(values[idx] - target), idx),
        )

    @staticmethod
    def _normalize_degrees(angle_degrees: float) -> float:
        angle = float(angle_degrees)
        while angle <= -180.0:
            angle += 360.0
        while angle > 180.0:
            angle -= 360.0
        return angle

    @staticmethod
    def _round_point(values: np.ndarray) -> list[float]:
        return [round(float(value), 6) for value in values.tolist()]

    @staticmethod
    def _component_centroid(component: dict[str, object]) -> np.ndarray:
        return np.asarray(
            [
                float(component["centroid_x"]),
                float(component["centroid_y"]),
            ],
            dtype=np.float64,
        )

    def _fit_affine_lattice_from_components(
        self,
        assigned_components: dict[tuple[int, int], dict[str, object]],
    ) -> dict[str, object]:
        if len(assigned_components) < 3:
            raise RuntimeError("Need at least 3 assigned cells for affine lattice fit.")

        design = []
        points = []
        ordered_keys = sorted(assigned_components)
        for row_index, panel_col_index in ordered_keys:
            design.append([1.0, float(panel_col_index), float(row_index)])
            points.append(self._component_centroid(assigned_components[(row_index, panel_col_index)]))

        design_arr = np.asarray(design, dtype=np.float64)
        points_arr = np.asarray(points, dtype=np.float64)
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
            design_arr,
            points_arr,
            rcond=None,
        )
        if rank < 3:
            raise RuntimeError("Affine lattice design matrix is rank deficient.")

        origin = coefficients[0]
        col_vector = coefficients[1]
        row_vector = coefficients[2]
        fitted = design_arr @ coefficients
        residuals_px = np.linalg.norm(points_arr - fitted, axis=1)
        rmse = math.sqrt(float(np.mean(residuals_px * residuals_px))) if len(residuals_px) else 0.0
        max_residual = float(np.max(residuals_px)) if len(residuals_px) else 0.0
        return {
            "origin": origin,
            "col_vector": col_vector,
            "row_vector": row_vector,
            "rmse": float(rmse),
            "max_residual": float(max_residual),
            "residuals_by_cell": {
                key: float(residual)
                for key, residual in zip(ordered_keys, residuals_px)
            },
        }

    def _assign_components_to_affine_lattice(
        self,
        patch_components: list[dict[str, object]],
        origin: np.ndarray,
        col_vector: np.ndarray,
        row_vector: np.ndarray,
        *,
        row_count: int,
        col_count: int,
    ) -> dict[tuple[int, int], dict[str, object]]:
        lattice_matrix = np.column_stack([col_vector, row_vector])
        det = float(np.linalg.det(lattice_matrix))
        col_step = float(np.linalg.norm(col_vector))
        row_step = float(np.linalg.norm(row_vector))
        min_pitch = min(col_step, row_step)
        if abs(det) <= 1e-6 or min_pitch <= 1e-6:
            return {}

        inverse = np.linalg.inv(lattice_matrix)
        assigned: dict[tuple[int, int], dict[str, object]] = {}
        for component in patch_components:
            point = self._component_centroid(component)
            col_float, row_float = inverse @ (point - origin)
            col_index = int(round(float(col_float)))
            row_index = int(round(float(row_float)))
            if not (0 <= col_index < col_count and 0 <= row_index < row_count):
                continue

            col_axis_residual = abs(float(col_float) - col_index)
            row_axis_residual = abs(float(row_float) - row_index)
            if col_axis_residual > 0.48 or row_axis_residual > 0.48:
                continue

            fitted_center = origin + (col_index * col_vector) + (row_index * row_vector)
            residual_px = float(np.linalg.norm(point - fitted_center))
            if residual_px > min_pitch * 0.62:
                continue

            score = (
                (residual_px / max(1.0, min_pitch))
                + col_axis_residual
                + row_axis_residual
            )
            key = (row_index, col_index)
            current = assigned.get(key)
            if current is None or score < float(current["score"]):
                assigned[key] = {
                    "score": float(score),
                    "component": component,
                    "residual_px": residual_px,
                    "lattice_col": float(col_float),
                    "lattice_row": float(row_float),
                }
        return assigned

    def _refine_panel_component_lattice(
        self,
        patch_components: list[dict[str, object]],
        initial_components_by_cell: dict[tuple[int, int], dict[str, object]],
        *,
        row_count: int,
        col_count: int,
    ) -> dict[str, object]:
        assigned_components = dict(initial_components_by_cell)
        previous_signature: tuple[tuple[int, int, int], ...] | None = None
        fit: dict[str, object] | None = None
        assigned_details: dict[tuple[int, int], dict[str, object]] = {}

        for _ in range(6):
            fit = self._fit_affine_lattice_from_components(assigned_components)
            assigned_details = self._assign_components_to_affine_lattice(
                patch_components,
                fit["origin"],
                fit["col_vector"],
                fit["row_vector"],
                row_count=row_count,
                col_count=col_count,
            )
            if len(assigned_details) < 3:
                break

            signature = tuple(
                sorted(
                    (
                        row_index,
                        panel_col_index,
                        id(detail["component"]),
                    )
                    for (row_index, panel_col_index), detail in assigned_details.items()
                )
            )
            assigned_components = {
                key: detail["component"]
                for key, detail in assigned_details.items()
            }
            if signature == previous_signature:
                break
            previous_signature = signature

        if fit is None:
            raise RuntimeError("Affine lattice refinement did not produce a fit.")
        if assigned_details:
            fit = self._fit_affine_lattice_from_components(
                {
                    key: detail["component"]
                    for key, detail in assigned_details.items()
                }
            )
            assigned_details = self._assign_components_to_affine_lattice(
                patch_components,
                fit["origin"],
                fit["col_vector"],
                fit["row_vector"],
                row_count=row_count,
                col_count=col_count,
            )
            if assigned_details:
                fit = self._fit_affine_lattice_from_components(
                    {
                        key: detail["component"]
                        for key, detail in assigned_details.items()
                    }
                )
                residuals_by_cell = fit["residuals_by_cell"]
                for key, detail in assigned_details.items():
                    detail["residual_px"] = float(residuals_by_cell.get(key, detail["residual_px"]))

        return {
            "fit": fit,
            "assigned_components": assigned_details,
        }

    def _fit_panel_component_lattice(
        self,
        patch_components: list[dict[str, object]],
        *,
        row_count: int,
        col_count: int,
    ) -> dict[str, object]:
        points = np.asarray(
            [
                [float(component["centroid_x"]), float(component["centroid_y"])]
                for component in patch_components
            ],
            dtype=np.float64,
        )
        if len(points) < max(3, SPYDERCHECKR_DIRECT_PANEL_MIN_PATCH_COMPONENTS):
            return {
                "ok": False,
                "reason": "insufficient_panel_components",
            }

        pivot = np.median(points, axis=0)
        best: dict[str, object] | None = None
        best_key: tuple[float, float, float, float] | None = None
        max_rotation = float(SPYDERCHECKR_DIRECT_PANEL_MAX_LATTICE_ROTATION_DEGREES)
        angle_values = np.linspace(-max_rotation, max_rotation, 61)
        for derotation_degrees in angle_values:
            theta = math.radians(float(derotation_degrees))
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            rotation = np.asarray(
                [
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta],
                ],
                dtype=np.float64,
            )
            derotated = (points - pivot) @ rotation.T + pivot
            try:
                raw_col_centers = self._cluster_axis_positions(
                    [float(point[0]) for point in derotated],
                    col_count,
                )
                raw_row_centers = self._cluster_axis_positions(
                    [float(point[1]) for point in derotated],
                    row_count,
                )
                col_centers, col_step, col_rmse = self._fit_regular_centers(raw_col_centers)
                row_centers, row_step, row_rmse = self._fit_regular_centers(raw_row_centers)
            except RuntimeError:
                continue
            if col_step <= 0.0 or row_step <= 0.0:
                continue

            seed_assignments: dict[tuple[int, int], tuple[float, dict[str, object]]] = {}
            for component, point in zip(patch_components, derotated):
                col_index = self._nearest_index(col_centers, float(point[0]))
                row_index = self._nearest_index(row_centers, float(point[1]))
                dx = abs(float(point[0]) - col_centers[col_index])
                dy = abs(float(point[1]) - row_centers[row_index])
                if dx > col_step * 0.48 or dy > row_step * 0.48:
                    continue
                score = (dx / max(1.0, abs(col_step))) + (dy / max(1.0, abs(row_step)))
                key = (row_index, col_index)
                current = seed_assignments.get(key)
                if current is None or score < current[0]:
                    seed_assignments[key] = (float(score), component)

            if len(seed_assignments) < 3:
                continue
            try:
                refined = self._refine_panel_component_lattice(
                    patch_components,
                    {
                        key: component
                        for key, (_score, component) in seed_assignments.items()
                    },
                    row_count=row_count,
                    col_count=col_count,
                )
            except RuntimeError:
                continue

            assigned_components = refined["assigned_components"]
            fit = refined["fit"]
            assigned_count = len(assigned_components)
            if assigned_count <= 0:
                continue
            distinct_rows = len({key[0] for key in assigned_components})
            distinct_cols = len({key[1] for key in assigned_components})
            candidate_key = (
                -float(assigned_count),
                -float(distinct_rows + distinct_cols),
                float(fit["rmse"]),
                float(fit["max_residual"]),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best = {
                    "derotation_degrees": float(derotation_degrees),
                    "raw_col_centers": raw_col_centers,
                    "raw_row_centers": raw_row_centers,
                    "col_centers": col_centers,
                    "row_centers": row_centers,
                    "col_rmse": float(col_rmse),
                    "row_rmse": float(row_rmse),
                    "fit": fit,
                    "assigned_components": assigned_components,
                }

        if best is None:
            return {
                "ok": False,
                "reason": "component_lattice_fit_failed",
            }

        fit = best["fit"]
        origin = np.asarray(fit["origin"], dtype=np.float64)
        col_vector = np.asarray(fit["col_vector"], dtype=np.float64)
        row_vector = np.asarray(fit["row_vector"], dtype=np.float64)
        assigned_components = best["assigned_components"]
        col_step = float(np.linalg.norm(col_vector))
        row_step = float(np.linalg.norm(row_vector))
        min_pitch = min(col_step, row_step)
        det = float(np.linalg.det(np.column_stack([col_vector, row_vector])))
        orthogonality = float(
            abs(np.dot(col_vector, row_vector)) / max(1e-9, col_step * row_step)
        )
        col_rotation = math.degrees(math.atan2(float(col_vector[1]), float(col_vector[0])))
        row_rotation = math.degrees(math.atan2(float(row_vector[1]), float(row_vector[0]))) - 90.0
        row_rotation = self._normalize_degrees(row_rotation)
        lattice_rotation = self._normalize_degrees((col_rotation + row_rotation) * 0.5)
        assigned_rows = sorted({key[0] for key in assigned_components})
        assigned_cols = sorted({key[1] for key in assigned_components})

        failure_reason: str | None = None
        if len(assigned_components) < SPYDERCHECKR_DIRECT_PANEL_MIN_PATCH_COMPONENTS:
            failure_reason = "component_lattice_coverage_too_sparse"
        elif len(assigned_cols) < 3 or len(assigned_rows) < 4:
            failure_reason = "component_lattice_row_column_coverage_too_sparse"
        elif min_pitch <= 4.0:
            failure_reason = "component_lattice_pitch_invalid"
        elif col_vector[0] <= 0.0 or row_vector[1] <= 0.0 or det <= max(1.0, col_step * row_step * 0.35):
            failure_reason = "component_lattice_orientation_invalid"
        elif orthogonality > 0.35:
            failure_reason = "component_lattice_axes_not_orthogonal"
        elif abs(lattice_rotation) > SPYDERCHECKR_DIRECT_PANEL_MAX_LATTICE_ROTATION_DEGREES:
            failure_reason = "component_lattice_rotation_out_of_range"
        elif float(fit["rmse"]) > min_pitch * 0.35:
            failure_reason = "component_lattice_rmse_too_large"
        elif float(fit["max_residual"]) > min_pitch * 0.70:
            failure_reason = "component_lattice_max_residual_too_large"

        diagnostics = {
            "lattice_rotation_degrees": round(float(lattice_rotation), 6),
            "lattice_fit_rmse_px": round(float(fit["rmse"]), 6),
            "lattice_fit_max_residual_px": round(float(fit["max_residual"]), 6),
            "lattice_origin_local": self._round_point(origin),
            "lattice_col_vector_local": self._round_point(col_vector),
            "lattice_row_vector_local": self._round_point(row_vector),
            "lattice_determinant": round(float(det), 6),
            "lattice_orthogonality_abs_cos": round(float(orthogonality), 6),
            "lattice_seed_derotation_degrees": round(float(best["derotation_degrees"]), 6),
            "assigned_lattice_cells_count": int(len(assigned_components)),
            "assigned_row_coverage_count": int(len(assigned_rows)),
            "assigned_column_coverage_count": int(len(assigned_cols)),
            "assigned_rows": [int(value) for value in assigned_rows],
            "assigned_panel_columns": [int(value) for value in assigned_cols],
            "component_coverage_fraction": round(
                float(len(assigned_components)) / max(1, row_count * col_count),
                6,
            ),
        }
        if failure_reason is not None:
            diagnostics["fit_rejection_reason"] = failure_reason
            return {
                "ok": False,
                "reason": failure_reason,
                "diagnostics": diagnostics,
            }

        return {
            "ok": True,
            "origin": origin,
            "col_vector": col_vector,
            "row_vector": row_vector,
            "col_step": col_step,
            "row_step": row_step,
            "rotation_degrees": float(lattice_rotation),
            "fit_rmse": float(fit["rmse"]),
            "fit_max_residual": float(fit["max_residual"]),
            "assigned_components": assigned_components,
            "raw_col_centers": best["raw_col_centers"],
            "raw_row_centers": best["raw_row_centers"],
            "col_centers": best["col_centers"],
            "row_centers": best["row_centers"],
            "col_rmse": float(best["col_rmse"]),
            "row_rmse": float(best["row_rmse"]),
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _component_stats(
        mask: np.ndarray,
    ) -> list[dict[str, object]]:
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        components: list[dict[str, object]] = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if w <= 0 or h <= 0 or area <= 0:
                continue
            box = _PatchwiseDisplayBox(x, y, x + w - 1, y + h - 1)
            occupancy = area / max(1.0, float(w * h))
            components.append(
                {
                    "box": box,
                    "area": int(area),
                    "occupancy": float(occupancy),
                    "centroid_x": float(centroids[label][0]),
                    "centroid_y": float(centroids[label][1]),
                    "aspect": float(box.height / max(1.0, float(box.width))),
                }
            )
        return components

    @staticmethod
    def _filter_patch_components(
        components: list[dict[str, object]],
        panel_box: _PatchwiseDisplayBox,
    ) -> list[dict[str, object]]:
        panel_area = panel_box.width * panel_box.height
        min_area = max(50, int(panel_area * 0.006))
        max_area = max(min_area + 1, int(panel_area * 0.06))
        min_width = max(16, panel_box.width // 11)
        min_height = max(16, panel_box.height // 14)

        filtered: list[dict[str, object]] = []
        for component in components:
            box = component["box"]
            if int(component["area"]) < min_area or int(component["area"]) > max_area:
                continue
            if box.width < min_width or box.height < min_height:
                continue
            if box.width > panel_box.width // 2 or box.height > panel_box.height // 4:
                continue
            if not (0.68 <= float(component["aspect"]) <= 1.45):
                continue
            if float(component["occupancy"]) < 0.45:
                continue
            filtered.append(component)

        filtered.sort(
            key=lambda item: (float(item["centroid_y"]), float(item["centroid_x"]))
        )
        return filtered

    def _estimate_panel_inner_lattice(
        self,
        frame_bgr: np.ndarray,
        seeds_by_label: dict[str, dict[str, object]],
        panel_labels: list[str],
        panel_box: _PatchwiseDisplayBox,
        *,
        panel_side: str,
    ) -> dict[str, object]:
        if not panel_labels:
            return {
                "ok": False,
                "diagnostics": {
                    "status": "failed",
                    "mode": "outer_box_fallback",
                    "reason": "empty_panel_labels",
                    "panel_side": panel_side,
                    "panel_box": panel_box.as_dict(),
                },
            }

        panel_crop = frame_bgr[
            panel_box.top : panel_box.bottom + 1,
            panel_box.left : panel_box.right + 1,
        ]
        if panel_crop.size == 0:
            return {
                "ok": False,
                "diagnostics": {
                    "status": "failed",
                    "mode": "outer_box_fallback",
                    "reason": "empty_panel_crop",
                    "panel_side": panel_side,
                    "panel_box": panel_box.as_dict(),
                },
            }

        hi = np.max(panel_crop, axis=2).astype(np.int16)
        lo = np.min(panel_crop, axis=2).astype(np.int16)
        signal = np.clip(hi + (hi - lo), 0, 255).astype(np.uint8)
        inner_box = self._make_local_inner_box(
            panel_box.width,
            panel_box.height,
            SPYDERCHECKR_DIRECT_PANEL_SIGNAL_MARGIN_RATIO,
        )
        inner_signal = signal[
            inner_box.top : inner_box.bottom + 1,
            inner_box.left : inner_box.right + 1,
        ]
        if inner_signal.size == 0:
            return {
                "ok": False,
                "diagnostics": {
                    "status": "failed",
                    "mode": "outer_box_fallback",
                    "reason": "empty_inner_signal_region",
                    "panel_side": panel_side,
                    "panel_box": panel_box.as_dict(),
                },
            }

        q35 = float(np.quantile(inner_signal, 0.35))
        q50 = float(np.quantile(inner_signal, 0.50))
        q65 = float(np.quantile(inner_signal, 0.65))
        threshold = min(
            160,
            max(
                SPYDERCHECKR_DIRECT_PANEL_PATCH_SIGNAL_FLOOR,
                int(round(((q35 + q50) * 0.5) + 10.0)),
            ),
        )
        signal_mask = (signal >= threshold).astype(np.uint8)
        raw_components = self._component_stats(signal_mask)
        patch_components = self._filter_patch_components(raw_components, panel_box)
        if len(patch_components) < SPYDERCHECKR_DIRECT_PANEL_MIN_PATCH_COMPONENTS:
            return {
                "ok": False,
                "diagnostics": {
                    "status": "failed",
                    "mode": "outer_box_fallback",
                    "reason": "insufficient_panel_components",
                    "panel_side": panel_side,
                    "panel_box": panel_box.as_dict(),
                    "signal_threshold": int(threshold),
                    "detected_patch_components_count": int(len(patch_components)),
                    "raw_connected_components_count": int(len(raw_components)),
                },
            }

        lattice_result = self._fit_panel_component_lattice(
            patch_components,
            row_count=self.n_rows,
            col_count=self.n_cols // 2,
        )
        if not bool(lattice_result.get("ok")):
            diagnostics = {
                "status": "failed",
                "mode": "outer_box_fallback",
                "reason": str(lattice_result.get("reason", "component_lattice_fit_failed")),
                "panel_side": panel_side,
                "panel_box": panel_box.as_dict(),
                "signal_threshold": int(threshold),
                "detected_patch_components_count": int(len(patch_components)),
                "raw_connected_components_count": int(len(raw_components)),
            }
            diagnostics.update(dict(lattice_result.get("diagnostics") or {}))
            return {
                "ok": False,
                "diagnostics": diagnostics,
            }

        origin = np.asarray(lattice_result["origin"], dtype=np.float64)
        col_vector = np.asarray(lattice_result["col_vector"], dtype=np.float64)
        row_vector = np.asarray(lattice_result["row_vector"], dtype=np.float64)
        col_step = float(lattice_result["col_step"])
        row_step = float(lattice_result["row_step"])
        roi_x_axis, roi_y_axis = self._orthonormal_axes_from_rotation_degrees(
            float(lattice_result["rotation_degrees"])
        )
        raw_col_centers = list(lattice_result["raw_col_centers"])
        raw_row_centers = list(lattice_result["raw_row_centers"])
        col_centers = list(lattice_result["col_centers"])
        row_centers = list(lattice_result["row_centers"])
        col_rmse = float(lattice_result["col_rmse"])
        row_rmse = float(lattice_result["row_rmse"])
        assigned_components: dict[tuple[int, int], dict[str, object]] = dict(
            lattice_result["assigned_components"]
        )
        accepted_components = [
            detail["component"]
            for detail in assigned_components.values()
        ]
        component_source = accepted_components if accepted_components else patch_components
        component_widths = [
            float(component["box"].width)
            for component in component_source
        ]
        component_heights = [
            float(component["box"].height)
            for component in component_source
        ]
        legacy_box_width = float(
            np.median(
                [
                    float(seeds_by_label[label]["legacy_seed_box"].width)
                    for label in panel_labels
                ]
            )
        )
        legacy_box_height = float(
            np.median(
                [
                    float(seeds_by_label[label]["legacy_seed_box"].height)
                    for label in panel_labels
                ]
            )
        )
        component_box_width = float(np.median(component_widths)) if component_widths else legacy_box_width
        component_box_height = float(np.median(component_heights)) if component_heights else legacy_box_height
        estimated_face_side = self._direct_panel_square_face_side(
            component_widths,
            component_heights,
            legacy_box_width,
            legacy_box_height,
            col_step,
            row_step,
        )
        estimated_face_width = estimated_face_side
        estimated_face_height = estimated_face_side
        roi_side = max(
            4.0,
            float(estimated_face_side)
            * (1.0 - (2.0 * SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO)),
        )

        updated_seeds: dict[str, dict[str, object]] = {}
        boxes_by_label: dict[str, _PatchwiseDisplayBox] = {}
        quads_by_label: dict[str, _PatchwiseDisplayQuad] = {}
        component_cells_count = 0
        lattice_cells_count = 0
        roi_widths: list[float] = []
        roi_heights: list[float] = []
        for label in panel_labels:
            seed_info = dict(seeds_by_label[label])
            panel_col_index = int(seed_info["panel_col_index"])
            row_index = int(seed_info["row_index"])
            anchor_local = origin + (panel_col_index * col_vector) + (row_index * row_vector)
            anchor_cx_local = float(anchor_local[0])
            anchor_cy_local = float(anchor_local[1])
            assignment = assigned_components.get((row_index, panel_col_index))
            component = assignment["component"] if assignment is not None else None
            if component is not None:
                face_box_local = component["box"]
                roi_center_local = np.asarray(
                    [float(component["centroid_x"]), float(component["centroid_y"])],
                    dtype=np.float64,
                )
                if (
                    face_box_local.width + 2 < estimated_face_width
                    or face_box_local.height + 2 < estimated_face_height
                ):
                    face_box_local = self._centered_local_box(
                        float(component["centroid_x"]),
                        float(component["centroid_y"]),
                        max(float(face_box_local.width), estimated_face_width),
                        max(float(face_box_local.height), estimated_face_height),
                        panel_box.width,
                        panel_box.height,
                    )
                component_cells_count += 1
                direct_panel_cell_source = "component"
            else:
                roi_center_local = anchor_local
                face_box_local = self._centered_local_box(
                    anchor_cx_local,
                    anchor_cy_local,
                    estimated_face_width,
                    estimated_face_height,
                    panel_box.width,
                    panel_box.height,
                )
                lattice_cells_count += 1
                direct_panel_cell_source = "lattice"
            face_box_full = self._translate_local_box(panel_box, face_box_local)
            roi_quad_local = self._centered_oriented_quad(
                float(roi_center_local[0]),
                float(roi_center_local[1]),
                roi_side,
                roi_side,
                roi_x_axis,
                roi_y_axis,
            )
            roi_quad_full = roi_quad_local.translated(float(panel_box.left), float(panel_box.top))
            roi_box_full = roi_quad_full.bounding_box
            roi_widths.append(float(roi_quad_full.width))
            roi_heights.append(float(roi_quad_full.height))
            seed_info["predicted_cx"] = float(roi_quad_full.cx)
            seed_info["predicted_cy"] = float(roi_quad_full.cy)
            seed_info["col_pitch"] = float(col_step)
            seed_info["row_pitch"] = float(row_step)
            seed_info["seed_box"] = roi_box_full
            seed_info["seed_quad"] = roi_quad_full
            seed_info["roi_quad"] = roi_quad_full
            seed_info["rescue_min_width"] = int(roi_box_full.width)
            seed_info["rescue_min_height"] = int(roi_box_full.height)
            seed_info["direct_panel_mode"] = "inner_lattice_estimated"
            seed_info["direct_panel_cell_source"] = direct_panel_cell_source
            seed_info["face_box"] = face_box_full
            seed_info["cluster_anchor_cx"] = float(panel_box.left + anchor_cx_local)
            seed_info["cluster_anchor_cy"] = float(panel_box.top + anchor_cy_local)
            updated_seeds[label] = seed_info
            boxes_by_label[label] = roi_box_full
            quads_by_label[label] = roi_quad_full

        return {
            "ok": True,
            "boxes_by_label": boxes_by_label,
            "quads_by_label": quads_by_label,
            "seeds_by_label": updated_seeds,
            "diagnostics": {
                "status": "ok",
                "mode": "inner_lattice_estimated",
                "reason": "panel_scope_affine_lattice_fit",
                "panel_side": panel_side,
                "panel_box": panel_box.as_dict(),
                "signal_threshold": int(threshold),
                "signal_quantiles": {
                    "q35": round(float(q35), 6),
                    "q50": round(float(q50), 6),
                    "q65": round(float(q65), 6),
                },
                "inner_signal_box_local": inner_box.as_dict(),
                "detected_patch_components_count": int(len(patch_components)),
                "raw_connected_components_count": int(len(raw_components)),
                "accepted_component_cells_count": int(len(accepted_components)),
                "cells_from_components_count": int(component_cells_count),
                "cells_from_lattice_only_count": int(lattice_cells_count),
                "raw_column_centers_local": [round(float(value), 6) for value in raw_col_centers],
                "raw_row_centers_local": [round(float(value), 6) for value in raw_row_centers],
                "fitted_column_centers_local": [round(float(value), 6) for value in col_centers],
                "fitted_row_centers_local": [round(float(value), 6) for value in row_centers],
                "column_step": round(float(col_step), 6),
                "row_step": round(float(row_step), 6),
                "column_fit_rmse": round(float(col_rmse), 6),
                "row_fit_rmse": round(float(row_rmse), 6),
                **dict(lattice_result.get("diagnostics") or {}),
                "component_box_width": round(float(component_box_width), 6),
                "component_box_height": round(float(component_box_height), 6),
                "estimated_face_side": round(float(estimated_face_side), 6),
                "estimated_face_width": round(float(estimated_face_width), 6),
                "estimated_face_height": round(float(estimated_face_height), 6),
                "legacy_box_width": round(float(legacy_box_width), 6),
                "legacy_box_height": round(float(legacy_box_height), 6),
                "median_roi_width": round(float(np.median(roi_widths)), 6) if roi_widths else 0.0,
                "median_roi_height": round(float(np.median(roi_heights)), 6) if roi_heights else 0.0,
                "roi_square_side": round(float(roi_side), 6),
                "roi_inset_ratio": round(float(SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO), 6),
            },
        }

    def _attempt_direct_dark_panel_layout(
        self,
        frame_bgr: np.ndarray,
        default_boxes: list[_PatchwiseDisplayBox],
    ) -> dict[str, object]:
        if self.n_rows != SPYDERCHECKR_48_SHAPE[0] or self.n_cols != SPYDERCHECKR_48_SHAPE[1]:
            return {
                "ok": False,
                "diagnostics": {
                    "status": "skipped",
                    "reason": "unsupported_grid_shape",
                },
            }
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "ok": False,
                "diagnostics": {
                    "status": "failed",
                    "reason": "frame_unavailable",
                },
            }
        if not default_boxes or len(default_boxes) != self.n_rows * self.n_cols:
            return {
                "ok": False,
                "diagnostics": {
                    "status": "failed",
                    "reason": "seed_geometry_unavailable",
                },
            }

        image_h, image_w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        left_region = self._panel_search_region_from_seed_boxes(
            default_boxes,
            is_right_panel=False,
            image_w=image_w,
            image_h=image_h,
        )
        right_region = self._panel_search_region_from_seed_boxes(
            default_boxes,
            is_right_panel=True,
            image_w=image_w,
            image_h=image_h,
        )
        left_panel_box, left_diag = self._detect_dark_panel_box(gray, left_region)
        right_panel_box, right_diag = self._detect_dark_panel_box(gray, right_region)
        diagnostics: dict[str, object] = {
            "status": "failed",
            "reason": "panel_detection_failed",
            "bright_threshold": SPYDERCHECKR_DIRECT_PANEL_BRIGHT_THRESHOLD,
            "left_panel_detection": left_diag,
            "right_panel_detection": right_diag,
        }
        if left_panel_box is None or right_panel_box is None:
            return {"ok": False, "diagnostics": diagnostics}

        left_cell_w = left_panel_box.width / max(self.n_cols // 2, 1)
        right_cell_w = right_panel_box.width / max(self.n_cols // 2, 1)
        left_cell_h = left_panel_box.height / max(self.n_rows, 1)
        right_cell_h = right_panel_box.height / max(self.n_rows, 1)
        mean_cell_w = (left_cell_w + right_cell_w) * 0.5
        mean_cell_h = (left_cell_h + right_cell_h) * 0.5
        if min(left_cell_w, right_cell_w, left_cell_h, right_cell_h) <= 4.0:
            diagnostics["reason"] = "cell_geometry_too_small"
            return {"ok": False, "diagnostics": diagnostics}

        left_d = left_panel_box.left + (3.5 * left_cell_w)
        right_e = right_panel_box.left + (0.5 * right_cell_w)
        hinge_gap = self._clamp(
            (right_e - left_d) / max(1.0, mean_cell_w) - 1.0,
            SPYDERCHECKR_DIRECT_PANEL_MIN_HINGE_GAP,
            SPYDERCHECKR_DIRECT_PANEL_MAX_HINGE_GAP,
        )
        direct_model = SpyderCheckrGridExtractor(
            self.n_rows,
            self.n_cols,
            patch_margin=SPYDERCHECKR_DIRECT_PANEL_PATCH_MARGIN,
            col_label_offset=self.col_label_offset,
            hinge_gap=hinge_gap,
        )
        direct_model.set_corners_ordered(
            [
                (
                    left_panel_box.left + (0.5 * left_cell_w),
                    left_panel_box.top + (0.5 * left_cell_h),
                ),
                (
                    right_panel_box.left + (3.5 * right_cell_w),
                    right_panel_box.top + (0.5 * right_cell_h),
                ),
                (
                    right_panel_box.left + (3.5 * right_cell_w),
                    right_panel_box.top + (5.5 * right_cell_h),
                ),
                (
                    left_panel_box.left + (0.5 * left_cell_w),
                    left_panel_box.top + (5.5 * left_cell_h),
                ),
            ]
        )
        direct_boxes = direct_model._default_patch_display_boxes()
        if len(direct_boxes) != self.n_rows * self.n_cols:
            diagnostics["reason"] = "direct_model_box_generation_failed"
            return {"ok": False, "diagnostics": diagnostics}
        seeds_by_label, ordered_labels = direct_model._build_patchwise_seed_info(direct_boxes)
        boxes_by_label = {
            label: seeds_by_label[label]["seed_box"] for label in ordered_labels
        }
        quads_by_label = {
            label: self._box_to_quad(seeds_by_label[label]["seed_box"])
            for label in ordered_labels
        }
        for info in seeds_by_label.values():
            info["direct_panel_mode"] = "outer_box_fallback"

        panel_layout_diagnostics: dict[str, object] = {}
        panel_success = True
        for is_right_panel, panel_box, panel_side in (
            (False, left_panel_box, "left"),
            (True, right_panel_box, "right"),
        ):
            panel_labels = [
                label
                for label in ordered_labels
                if bool(seeds_by_label[label]["is_right_panel"]) == is_right_panel
            ]
            lattice_result = self._estimate_panel_inner_lattice(
                frame_bgr,
                seeds_by_label,
                panel_labels,
                panel_box,
                panel_side=panel_side,
            )
            panel_layout_diagnostics[f"{panel_side}_panel_layout"] = lattice_result["diagnostics"]
            if bool(lattice_result.get("ok")):
                for label, seed_info in lattice_result["seeds_by_label"].items():
                    seeds_by_label[label] = seed_info
                    boxes_by_label[label] = lattice_result["boxes_by_label"][label]
                    quads_by_label[label] = lattice_result["quads_by_label"][label]
            else:
                panel_success = False
                for label in panel_labels:
                    seed_info = dict(seeds_by_label[label])
                    seed_info["seed_box"] = seed_info["legacy_seed_box"]
                    seed_info["seed_quad"] = self._box_to_quad(seed_info["legacy_seed_box"])
                    seed_info["roi_quad"] = seed_info["seed_quad"]
                    seed_info["rescue_min_width"] = int(seed_info["legacy_seed_box"].width)
                    seed_info["rescue_min_height"] = int(seed_info["legacy_seed_box"].height)
                    seed_info["direct_panel_mode"] = "outer_box_fallback"
                    seeds_by_label[label] = seed_info
                    boxes_by_label[label] = seed_info["seed_box"]
                    quads_by_label[label] = seed_info["seed_quad"]

        diagnostics = {
            "status": "ok",
            "reason": "panel_pair_detected",
            "direct_panel_mode": "inner_lattice_estimated" if panel_success else "outer_box_fallback",
            "bright_threshold": SPYDERCHECKR_DIRECT_PANEL_BRIGHT_THRESHOLD,
            "left_panel_detection": left_diag,
            "right_panel_detection": right_diag,
            "panel_pair": {
                "left_box": left_panel_box.as_dict(),
                "right_box": right_panel_box.as_dict(),
                "hinge_gap": round(float(hinge_gap), 6),
                "mean_cell_width": round(float(mean_cell_w), 6),
                "mean_cell_height": round(float(mean_cell_h), 6),
                "patch_margin": round(float(SPYDERCHECKR_DIRECT_PANEL_PATCH_MARGIN), 6),
            },
        }
        diagnostics.update(panel_layout_diagnostics)
        return {
            "ok": True,
            "boxes_by_label": boxes_by_label,
            "quads_by_label": quads_by_label,
            "seeds_by_label": seeds_by_label,
            "ordered_labels": ordered_labels,
            "diagnostics": diagnostics,
        }

    def _center_guard_metrics(
        self,
        candidate: _PatchwiseDisplayBox,
        seed_info: dict[str, object],
    ) -> dict[str, float | bool]:
        predicted_cx = float(seed_info["predicted_cx"])
        predicted_cy = float(seed_info["predicted_cy"])
        col_pitch = max(1.0, float(seed_info["col_pitch"]))
        row_pitch = max(1.0, float(seed_info["row_pitch"]))
        dx_px = abs(candidate.cx - predicted_cx)
        dy_px = abs(candidate.cy - predicted_cy)
        dx_ratio = dx_px / col_pitch
        dy_ratio = dy_px / row_pitch
        center_deviation_ratio = max(dx_ratio, dy_ratio)
        return {
            "predicted_cx": float(predicted_cx),
            "predicted_cy": float(predicted_cy),
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "dx_ratio": float(dx_ratio),
            "dy_ratio": float(dy_ratio),
            "center_deviation_ratio": float(center_deviation_ratio),
            "passes_center_guard": bool(
                center_deviation_ratio
                <= SPYDERCHECKR_PATCHWISE_CENTER_TOLERANCE_RATIO + 1e-9
            ),
        }

    def _normalize_patch_box(
        self,
        candidate: _PatchwiseDisplayBox,
        seed_info: dict[str, object],
        image_w: int,
        image_h: int,
        min_box_size: int,
        max_box_size: int,
        max_center_shift: float,
    ) -> "_PatchwiseDisplayBox | None":
        seed_box = seed_info["seed_box"]
        left = max(0, min(image_w - 1, int(candidate.left)))
        right = max(0, min(image_w - 1, int(candidate.right)))
        top = max(0, min(image_h - 1, int(candidate.top)))
        bottom = max(0, min(image_h - 1, int(candidate.bottom)))
        if right < left or bottom < top:
            return None
        box = _PatchwiseDisplayBox(left, top, right, bottom)
        if box.width < min_box_size or box.height < min_box_size:
            return None
        if box.width > max_box_size or box.height > max_box_size:
            return None
        if math.hypot(box.cx - seed_box.cx, box.cy - seed_box.cy) > max_center_shift:
            return None
        geometry = self._center_guard_metrics(box, seed_info)
        if not bool(geometry["passes_center_guard"]):
            return None
        return box

    def _generate_patch_candidates(
        self,
        current_box: _PatchwiseDisplayBox,
        seed_info: dict[str, object],
        image_w: int,
        image_h: int,
        step: int,
        min_box_size: int,
        max_box_size: int,
        max_center_shift: float,
    ) -> list[_PatchwiseDisplayBox]:
        raw_candidates = {
            current_box,
            _PatchwiseDisplayBox(current_box.left - step, current_box.top, current_box.right - step, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left + step, current_box.top, current_box.right + step, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left, current_box.top - step, current_box.right, current_box.bottom - step),
            _PatchwiseDisplayBox(current_box.left, current_box.top + step, current_box.right, current_box.bottom + step),
            _PatchwiseDisplayBox(current_box.left - step, current_box.top - step, current_box.right - step, current_box.bottom - step),
            _PatchwiseDisplayBox(current_box.left + step, current_box.top - step, current_box.right + step, current_box.bottom - step),
            _PatchwiseDisplayBox(current_box.left - step, current_box.top + step, current_box.right - step, current_box.bottom + step),
            _PatchwiseDisplayBox(current_box.left + step, current_box.top + step, current_box.right + step, current_box.bottom + step),
            _PatchwiseDisplayBox(current_box.left - step, current_box.top, current_box.right + step, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left + step, current_box.top, current_box.right - step, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left, current_box.top - step, current_box.right, current_box.bottom + step),
            _PatchwiseDisplayBox(current_box.left, current_box.top + step, current_box.right, current_box.bottom - step),
            _PatchwiseDisplayBox(current_box.left - step, current_box.top, current_box.right, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left + step, current_box.top, current_box.right, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left, current_box.top, current_box.right - step, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left, current_box.top, current_box.right + step, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left, current_box.top - step, current_box.right, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left, current_box.top + step, current_box.right, current_box.bottom),
            _PatchwiseDisplayBox(current_box.left, current_box.top, current_box.right, current_box.bottom - step),
            _PatchwiseDisplayBox(current_box.left, current_box.top, current_box.right, current_box.bottom + step),
        }
        out: list[_PatchwiseDisplayBox] = []
        for candidate in raw_candidates:
            normalized = self._normalize_patch_box(
                candidate,
                seed_info,
                image_w,
                image_h,
                min_box_size,
                max_box_size,
                max_center_shift,
            )
            if normalized is not None:
                out.append(normalized)
        return out

    def _evaluate_patch_quad(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        edge_map: np.ndarray,
        quad: _PatchwiseDisplayQuad,
        seed_box: _PatchwiseDisplayBox,
    ) -> dict[str, float]:
        del seed_box
        evaluation_quad = quad.shrunken(0.10)
        width = max(4, int(round(float(evaluation_quad.width))))
        height = max(4, int(round(float(evaluation_quad.height))))
        source_points = np.asarray(evaluation_quad.points, dtype=np.float32)
        target_points = np.asarray(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source_points, target_points)
        warped_frame = cv2.warpPerspective(
            frame_bgr,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        warped_gray = cv2.warpPerspective(
            gray,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        warped_edge = cv2.warpPerspective(
            edge_map,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        local_box = _PatchwiseDisplayBox(0, 0, width - 1, height - 1)
        return self._evaluate_patch_box(
            warped_frame,
            warped_gray,
            warped_edge,
            local_box,
            local_box,
        )

    def _evaluate_patch_box(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        edge_map: np.ndarray,
        box: _PatchwiseDisplayBox,
        seed_box: _PatchwiseDisplayBox,
    ) -> dict[str, float]:
        roi = frame_bgr[box.top : box.bottom + 1, box.left : box.right + 1]
        roi_gray = gray[box.top : box.bottom + 1, box.left : box.right + 1]
        roi_edge = edge_map[box.top : box.bottom + 1, box.left : box.right + 1]
        if roi.size == 0 or roi_gray.size == 0:
            return {
                "local_quality": 0.0,
                "leakage_ratio": 1.0,
                "contamination_ratio": 1.0,
                "edge_fraction": 1.0,
                "border_drop": 1.0,
                "border_edge_fraction": 1.0,
                "rgb_std_norm": 1.0,
                "area_ratio": 0.0,
                "usable_flag": 0.0,
            }

        rgb_float = roi.astype(np.float32) / 255.0
        channel_std = np.std(rgb_float, axis=(0, 1))
        rgb_std_norm = math.sqrt(float(np.mean(channel_std * channel_std)))
        edge_fraction = float(np.mean(roi_edge > 0))
        inset = max(2, min(box.width, box.height) // 5)
        border_mask = self._outer_band_mask(box.height, box.width, inset)
        if border_mask.any():
            border_gray = roi_gray[border_mask]
            border_edge = roi_edge[border_mask]
            inner_gray = roi_gray[~border_mask] if (~border_mask).any() else roi_gray.reshape(-1)
            core_mean = float(np.mean(inner_gray)) / 255.0
            border_mean = float(np.mean(border_gray)) / 255.0
            border_drop = self._clamp((core_mean - border_mean - 0.02) / 0.35, 0.0, 1.0)
            border_edge_fraction = float(np.mean(border_edge > 0))
            leakage = self._clamp(
                (0.65 * border_drop)
                + (0.35 * self._clamp(border_edge_fraction / 0.20, 0.0, 1.0)),
                0.0,
                1.0,
            )
        else:
            border_drop = 0.0
            border_edge_fraction = 0.0
            leakage = 0.0

        area_ratio = box.area / max(1.0, float(seed_box.area))
        area_score = self._clamp(1.0 - abs(area_ratio - 1.0) / 0.55, 0.0, 1.0)
        variability_score = 1.0 - self._clamp(rgb_std_norm / 0.18, 0.0, 1.0)
        edge_score = 1.0 - self._clamp(edge_fraction / 0.14, 0.0, 1.0)
        leakage_score = 1.0 - leakage
        local_quality = (
            (0.36 * leakage_score)
            + (0.30 * edge_score)
            + (0.22 * variability_score)
            + (0.12 * area_score)
        )
        usable_flag = 1.0 if (leakage <= 0.20 and edge_fraction <= 0.10 and rgb_std_norm <= 0.09) else 0.0
        return {
            "local_quality": float(local_quality),
            "leakage_ratio": float(leakage),
            "contamination_ratio": float(max(edge_fraction, leakage)),
            "edge_fraction": float(edge_fraction),
            "border_drop": float(border_drop),
            "border_edge_fraction": float(border_edge_fraction),
            "rgb_std_norm": float(rgb_std_norm),
            "area_ratio": float(area_ratio),
            "usable_flag": float(usable_flag),
        }

    @staticmethod
    def _intersection_ratio(a: _PatchwiseDisplayBox, b: _PatchwiseDisplayBox) -> float:
        left = max(a.left, b.left)
        top = max(a.top, b.top)
        right = min(a.right, b.right)
        bottom = min(a.bottom, b.bottom)
        if right < left or bottom < top:
            return 0.0
        intersection = float((right - left + 1) * (bottom - top + 1))
        return intersection / max(1.0, float(min(a.area, b.area)))

    def _regularization_penalty(
        self,
        label: str,
        candidate: _PatchwiseDisplayBox,
        boxes_by_label: dict[str, _PatchwiseDisplayBox],
        seeds_by_label: dict[str, dict[str, object]],
    ) -> dict[str, float]:
        seed = seeds_by_label[label]
        row_centers = [
            box.cy
            for other_label, box in boxes_by_label.items()
            if other_label != label and int(seeds_by_label[other_label]["row_index"]) == int(seed["row_index"])
        ]
        col_centers = [
            box.cx
            for other_label, box in boxes_by_label.items()
            if other_label != label
            and bool(seeds_by_label[other_label]["is_right_panel"]) == bool(seed["is_right_panel"])
            and int(seeds_by_label[other_label]["panel_col_index"]) == int(seed["panel_col_index"])
        ]
        panel_widths = [
            float(box.width)
            for other_label, box in boxes_by_label.items()
            if other_label != label and bool(seeds_by_label[other_label]["is_right_panel"]) == bool(seed["is_right_panel"])
        ]
        panel_heights = [
            float(box.height)
            for other_label, box in boxes_by_label.items()
            if other_label != label and bool(seeds_by_label[other_label]["is_right_panel"]) == bool(seed["is_right_panel"])
        ]

        row_target = float(np.median(row_centers)) if row_centers else candidate.cy
        col_target = float(np.median(col_centers)) if col_centers else candidate.cx
        width_target = float(np.median(panel_widths)) if panel_widths else float(candidate.width)
        height_target = float(np.median(panel_heights)) if panel_heights else float(candidate.height)
        row_penalty = abs(candidate.cy - row_target) / max(1.0, height_target)
        col_penalty = abs(candidate.cx - col_target) / max(1.0, width_target)
        size_penalty = (
            abs(candidate.width - width_target) / max(1.0, width_target)
            + abs(candidate.height - height_target) / max(1.0, height_target)
        ) * 0.5
        seed_box = seed["seed_box"]
        seed_drift_penalty = (
            math.hypot(candidate.cx - seed_box.cx, candidate.cy - seed_box.cy) / 10.0
            + abs((candidate.area / max(1.0, seed_box.area)) - 1.0) / 0.45
        ) * 0.5
        overlap_penalty = 0.0
        for other_label, other_box in boxes_by_label.items():
            if other_label == label:
                continue
            overlap_penalty = max(overlap_penalty, self._intersection_ratio(candidate, other_box))
        return {
            "row_penalty": float(row_penalty),
            "col_penalty": float(col_penalty),
            "size_penalty": float(size_penalty),
            "seed_drift_penalty": float(seed_drift_penalty),
            "overlap_penalty": float(overlap_penalty),
        }

    def _candidate_score(
        self,
        label: str,
        candidate: _PatchwiseDisplayBox,
        boxes_by_label: dict[str, _PatchwiseDisplayBox],
        seeds_by_label: dict[str, dict[str, object]],
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        edge_map: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        local = self._evaluate_patch_box(
            frame_bgr,
            gray,
            edge_map,
            candidate,
            seeds_by_label[label]["seed_box"],
        )
        penalties = self._regularization_penalty(
            label,
            candidate,
            boxes_by_label,
            seeds_by_label,
        )
        geometry = self._center_guard_metrics(candidate, seeds_by_label[label])
        if not bool(geometry["passes_center_guard"]):
            merged = dict(local)
            merged.update(penalties)
            merged.update(geometry)
            merged["total_score"] = -1_000_000.0
            return -1_000_000.0, merged
        total_score = (
            local["local_quality"]
            - (0.10 * self._clamp(penalties["row_penalty"], 0.0, 1.5))
            - (0.10 * self._clamp(penalties["col_penalty"], 0.0, 1.5))
            - (0.06 * self._clamp(penalties["size_penalty"], 0.0, 1.5))
            - (0.10 * self._clamp(penalties["seed_drift_penalty"], 0.0, 1.8))
            - (0.25 * self._clamp(penalties["overlap_penalty"], 0.0, 1.0))
        )
        merged = dict(local)
        merged.update(penalties)
        merged.update(geometry)
        merged["total_score"] = float(total_score)
        return float(total_score), merged

    def _candidate_score_for_quad(
        self,
        label: str,
        candidate_quad: _PatchwiseDisplayQuad,
        candidate_box: _PatchwiseDisplayBox,
        boxes_by_label: dict[str, _PatchwiseDisplayBox],
        seeds_by_label: dict[str, dict[str, object]],
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        edge_map: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        local = self._evaluate_patch_quad(
            frame_bgr,
            gray,
            edge_map,
            candidate_quad,
            seeds_by_label[label]["seed_box"],
        )
        penalties = self._regularization_penalty(
            label,
            candidate_box,
            boxes_by_label,
            seeds_by_label,
        )
        geometry = self._center_guard_metrics(candidate_box, seeds_by_label[label])
        if not bool(geometry["passes_center_guard"]):
            merged = dict(local)
            merged.update(penalties)
            merged.update(geometry)
            merged["total_score"] = -1_000_000.0
            return -1_000_000.0, merged
        total_score = (
            local["local_quality"]
            - (0.10 * self._clamp(penalties["row_penalty"], 0.0, 1.5))
            - (0.10 * self._clamp(penalties["col_penalty"], 0.0, 1.5))
            - (0.06 * self._clamp(penalties["size_penalty"], 0.0, 1.5))
            - (0.10 * self._clamp(penalties["seed_drift_penalty"], 0.0, 1.8))
            - (0.25 * self._clamp(penalties["overlap_penalty"], 0.0, 1.0))
        )
        merged = dict(local)
        merged.update(penalties)
        merged.update(geometry)
        merged["total_score"] = float(total_score)
        return float(total_score), merged

    def _search_rescue_patch_box(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        edge_map: np.ndarray,
        current_box: _PatchwiseDisplayBox,
        seed_info: dict[str, object],
    ) -> tuple[_PatchwiseDisplayBox, dict[str, float]]:
        seed_box = seed_info["seed_box"]
        image_h, image_w = gray.shape
        min_box_size = max(
            8,
            int(seed_info.get("rescue_min_width", seed_box.width)),
            int(seed_info.get("rescue_min_height", seed_box.height)),
            int(round(min(seed_box.width, seed_box.height) * 0.45)),
        )
        search_radius = max(1, int(round(min(seed_box.width, seed_box.height) * 0.16)))
        max_center_shift = max(2.0, min(seed_box.width, seed_box.height) * 0.12)
        anchor_cx = float(seed_info["predicted_cx"])
        anchor_cy = float(seed_info["predicted_cy"])
        best_box = current_box
        best_metrics = self._evaluate_patch_box(frame_bgr, gray, edge_map, current_box, seed_box)
        best_metrics.update(self._center_guard_metrics(current_box, seed_info))
        best_key = (
            not bool(best_metrics["passes_center_guard"]),
            best_metrics["usable_flag"] < 1.0,
            best_metrics["area_ratio"] < SPYDERCHECKR_PATCHWISE_MIN_AREA_RATIO,
            best_metrics["contamination_ratio"],
            best_metrics["leakage_ratio"],
            best_metrics["rgb_std_norm"],
            -best_metrics["local_quality"],
            0,
        )
        for width in range(min_box_size, current_box.width + 1):
            for height in range(min_box_size, current_box.height + 1):
                for dx in range(-search_radius, search_radius + 1):
                    for dy in range(-search_radius, search_radius + 1):
                        left = round(anchor_cx - (width - 1) / 2 + dx)
                        top = round(anchor_cy - (height - 1) / 2 + dy)
                        candidate = self._normalize_patch_box(
                            _PatchwiseDisplayBox(left, top, left + width - 1, top + height - 1),
                            seed_info,
                            image_w,
                            image_h,
                            min_box_size,
                            max(current_box.width, current_box.height),
                            max_center_shift,
                        )
                        if candidate is None:
                            continue
                        metrics = self._evaluate_patch_box(frame_bgr, gray, edge_map, candidate, seed_box)
                        metrics.update(self._center_guard_metrics(candidate, seed_info))
                        key = (
                            not bool(metrics["passes_center_guard"]),
                            metrics["usable_flag"] < 1.0,
                            metrics["area_ratio"] < SPYDERCHECKR_PATCHWISE_MIN_AREA_RATIO,
                            metrics["contamination_ratio"],
                            metrics["leakage_ratio"],
                            metrics["rgb_std_norm"],
                            -metrics["local_quality"],
                            abs(dx) + abs(dy),
                        )
                        if key < best_key:
                            best_key = key
                            best_box = candidate
                            best_metrics = metrics
        return best_box, best_metrics

    @staticmethod
    def _should_accept_rescue(before: dict[str, float], after: dict[str, float]) -> tuple[bool, str]:
        if not bool(after.get("passes_center_guard", False)):
            return False, "center guard failed"
        if bool(before.get("passes_center_guard", True)) is False:
            return True, "center guard restored"
        if (
            after["usable_flag"] >= 1.0
            and before["usable_flag"] < 1.0
            and after["area_ratio"] >= SPYDERCHECKR_PATCHWISE_MIN_AREA_RATIO
        ):
            return True, "usable rescue achieved"
        if (
            after["contamination_ratio"] + 0.08 < before["contamination_ratio"]
            and after["rgb_std_norm"] < before["rgb_std_norm"]
            and after["area_ratio"] >= SPYDERCHECKR_PATCHWISE_MIN_AREA_RATIO
        ):
            return True, "contamination reduced with acceptable retained area"
        return False, "baseline kept"

    def _build_patchwise_summary(
        self,
        final_entries: list[dict[str, object]],
        *,
        refined_patch_count: int,
        rescue_patch_count: int,
        fallback_patch_count: int,
        weak_patch_count: int,
        geometry_model: str,
        direct_dark_panel_diagnostics: dict[str, object] | None,
        status: str = "ok",
        reason: str | None = None,
        direct_patch_count: int = 0,
        direct_panel_mode: str | None = None,
    ) -> dict[str, object]:
        center_ratios = [
            float(entry["geometry"]["center_deviation_ratio"])
            for entry in final_entries
        ]
        width_ratios = [
            float(entry["size"]["width_ratio_to_col_pitch"])
            for entry in final_entries
        ]
        height_ratios = [
            float(entry["size"]["height_ratio_to_row_pitch"])
            for entry in final_entries
        ]
        area_ratios_to_cell = [
            float(entry["size"]["area_ratio_to_cell_pitch_area"])
            for entry in final_entries
        ]
        failing_center_guard_patch_count = sum(
            1
            for entry in final_entries
            if not bool(entry["geometry"]["passes_center_guard"])
        )
        median_area_ratio_to_cell_pitch_area = round(
            float(np.median(area_ratios_to_cell)),
            6,
        ) if area_ratios_to_cell else 0.0
        geometry_viability = _evaluate_patchwise_calibration_geometry(
            status=status,
            patch_count=len(final_entries),
            rescue_patch_count=rescue_patch_count,
            fallback_patch_count=fallback_patch_count,
            weak_patch_count=weak_patch_count,
            median_area_ratio_to_cell_pitch_area=median_area_ratio_to_cell_pitch_area,
        )
        summary = {
            "status": status,
            "refined_patch_count": refined_patch_count,
            "rescue_patch_count": rescue_patch_count,
            "fallback_patch_count": fallback_patch_count,
            "weak_patch_count": weak_patch_count,
            "direct_patch_count": direct_patch_count,
            "geometry_model": geometry_model,
            "center_tolerance_ratio": SPYDERCHECKR_PATCHWISE_CENTER_TOLERANCE_RATIO,
            "max_center_deviation_ratio": round(max(center_ratios), 6) if center_ratios else 0.0,
            "mean_center_deviation_ratio": round(
                float(np.mean(center_ratios)),
                6,
            ) if center_ratios else 0.0,
            "median_width_ratio_to_col_pitch": round(
                float(np.median(width_ratios)),
                6,
            ) if width_ratios else 0.0,
            "median_height_ratio_to_row_pitch": round(
                float(np.median(height_ratios)),
                6,
            ) if height_ratios else 0.0,
            "median_area_ratio_to_cell_pitch_area": median_area_ratio_to_cell_pitch_area,
            "failing_center_guard_patch_count": int(
                failing_center_guard_patch_count
            ),
            "mean_roi_score": round(
                float(np.mean([entry["roi_score"] for entry in final_entries])),
                6,
            ) if final_entries else 0.0,
            "direct_dark_panel_diagnostics": (
                dict(direct_dark_panel_diagnostics)
                if direct_dark_panel_diagnostics is not None
                else {"status": "not_attempted"}
            ),
        }
        roi_sides: list[float] = []
        roi_areas: list[float] = []
        rasterized_areas: list[float] = []
        for entry in final_entries:
            quad_size = entry.get("quad_size") or {}
            if isinstance(quad_size, dict):
                try:
                    width = float(quad_size.get("width", 0.0) or 0.0)
                    height = float(quad_size.get("height", 0.0) or 0.0)
                    area = float(quad_size.get("area", 0.0) or 0.0)
                except (TypeError, ValueError):
                    width = height = area = 0.0
                if width > 0.0 and height > 0.0:
                    roi_sides.append(float((width + height) * 0.5))
                if area > 0.0:
                    roi_areas.append(area)
            roi_quad = entry.get("roi_quad") or {}
            if isinstance(roi_quad, dict):
                mask_area = self._rasterized_quad_mask_area(roi_quad.get("points"))
                if mask_area is not None and mask_area > 0:
                    rasterized_areas.append(float(mask_area))
        if roi_areas:
            side_delta = (
                float(max(roi_sides) - min(roi_sides))
                if roi_sides
                else 0.0
            )
            area_delta = float(max(roi_areas) - min(roi_areas))
            area_median = float(np.median(roi_areas))
            summary["roi_uniform_area_metrics"] = {
                "count": int(len(roi_areas)),
                "side_min_px": round(float(min(roi_sides)), 6) if roi_sides else None,
                "side_median_px": round(float(np.median(roi_sides)), 6) if roi_sides else None,
                "side_max_px": round(float(max(roi_sides)), 6) if roi_sides else None,
                "side_range_px": round(side_delta, 6),
                "continuous_area_min_px2": round(float(min(roi_areas)), 6),
                "continuous_area_median_px2": round(area_median, 6),
                "continuous_area_max_px2": round(float(max(roi_areas)), 6),
                "continuous_area_range_px2": round(area_delta, 6),
                "max_relative_area_error": round(
                    float(area_delta / max(1e-9, area_median)),
                    10,
                ),
            }
        if rasterized_areas:
            raster_median = float(np.median(rasterized_areas))
            raster_delta = float(max(rasterized_areas) - min(rasterized_areas))
            summary["rasterized_display_mask_area_metrics"] = {
                "count": int(len(rasterized_areas)),
                "area_min_px": int(min(rasterized_areas)),
                "area_median_px": round(raster_median, 6),
                "area_max_px": int(max(rasterized_areas)),
                "area_range_px": int(raster_delta),
                "relative_area_range": round(
                    float(raster_delta / max(1e-9, raster_median)),
                    6,
                ),
            }
        if direct_panel_mode is not None:
            summary["direct_panel_mode"] = direct_panel_mode
        if reason is not None:
            summary["reason"] = reason
        accepted_roi_policies = sorted(
            {
                str(entry.get("accepted_roi_policy", "")).strip()
                for entry in final_entries
                if str(entry.get("accepted_roi_policy", "")).strip()
            }
        )
        accepted_source_counts: dict[str, int] = {}
        raw_source_counts: dict[str, int] = {}
        adopted_current_blue_labels: list[str] = []
        for entry in final_entries:
            accepted_source = str(entry.get("accepted_roi_source", "")).strip()
            if accepted_source:
                accepted_source_counts[accepted_source] = accepted_source_counts.get(accepted_source, 0) + 1
            raw_source = str(entry.get("raw_direct_panel_cell_source", "")).strip()
            if raw_source:
                raw_source_counts[raw_source] = raw_source_counts.get(raw_source, 0) + 1
                if raw_source == "lattice":
                    adopted_current_blue_labels.append(str(entry.get("label", "")))
        if accepted_roi_policies:
            summary["accepted_roi_policy"] = (
                accepted_roi_policies[0] if len(accepted_roi_policies) == 1 else "mixed"
            )
        if accepted_source_counts:
            summary["accepted_source_counts"] = dict(sorted(accepted_source_counts.items()))
        if raw_source_counts:
            summary["raw_direct_panel_cell_source_counts"] = dict(sorted(raw_source_counts.items()))
            summary["adopted_current_blue_labels"] = sorted(
                label for label in adopted_current_blue_labels if label
            )
        summary.update(geometry_viability)
        return summary

    def _finalize_direct_dark_panel_entries(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        edge_map: np.ndarray,
        seeds_by_label: dict[str, dict[str, object]],
        ordered_labels: list[str],
        boxes_by_label: dict[str, _PatchwiseDisplayBox],
        quads_by_label: dict[str, _PatchwiseDisplayQuad],
        diagnostics: dict[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        final_entries: list[dict[str, object]] = []
        weak_patch_count = 0
        for label in ordered_labels:
            seed_info = seeds_by_label[label]
            box = boxes_by_label[label]
            quad = quads_by_label.get(label) or seed_info.get("roi_quad") or self._box_to_quad(box)
            if not isinstance(quad, _PatchwiseDisplayQuad):
                quad = self._box_to_quad(box)
            seed_quad = seed_info.get("seed_quad") or quad
            if not isinstance(seed_quad, _PatchwiseDisplayQuad):
                seed_quad = quad
            direct_panel_mode = str(seed_info.get("direct_panel_mode", "outer_box_fallback"))
            if direct_panel_mode == "inner_lattice_estimated":
                score, details = self._candidate_score_for_quad(
                    label,
                    quad,
                    box,
                    boxes_by_label,
                    seeds_by_label,
                    frame_bgr,
                    gray,
                    edge_map,
                )
            else:
                score, details = self._candidate_score(
                    label,
                    box,
                    boxes_by_label,
                    seeds_by_label,
                    frame_bgr,
                    gray,
                    edge_map,
                )
            if score < SPYDERCHECKR_PATCHWISE_SCORE_THRESHOLD or details["usable_flag"] < 1.0:
                weak_patch_count += 1
            geometry = {
                "predicted_center": [
                    round(float(details["predicted_cx"]), 3),
                    round(float(details["predicted_cy"]), 3),
                ],
                "dx_px": round(float(details["dx_px"]), 6),
                "dy_px": round(float(details["dy_px"]), 6),
                "dx_ratio": round(float(details["dx_ratio"]), 6),
                "dy_ratio": round(float(details["dy_ratio"]), 6),
                "center_deviation_ratio": round(
                    float(details["center_deviation_ratio"]),
                    6,
                ),
                "passes_center_guard": bool(details["passes_center_guard"]),
            }
            col_pitch = max(1.0, float(seed_info["col_pitch"]))
            row_pitch = max(1.0, float(seed_info["row_pitch"]))
            size = {
                "width_ratio_to_col_pitch": round(float(box.width) / col_pitch, 6),
                "height_ratio_to_row_pitch": round(float(box.height) / row_pitch, 6),
                "area_ratio_to_cell_pitch_area": round(
                    float(box.area) / max(1.0, col_pitch * row_pitch),
                    6,
                ),
            }
            raw_direct_panel_cell_source = str(seed_info.get("direct_panel_cell_source", "unknown"))
            accepted_roi_source = raw_direct_panel_cell_source
            bbox_area = float(max(1, box.area))
            quad_area = float(max(0.0, quad.area))
            entry = {
                "label": label,
                "row_index": int(seed_info["row_index"]),
                "col_index": int(seed_info["col_index"]),
                "seed_roi_box": seed_info["seed_box"].as_dict(),
                "seed_roi_quad": seed_quad.as_dict(),
                "roi_box": box.as_dict(),
                "roi_quad": quad.as_dict(),
                "roi_shape": "quad" if direct_panel_mode == "inner_lattice_estimated" else "box",
                "center": [round(float(quad.cx), 3), round(float(quad.cy), 3)],
                "roi_score": round(float(score), 6),
                "local_quality": round(float(details["local_quality"]), 6),
                "leakage_ratio": round(float(details["leakage_ratio"]), 6),
                "contamination_ratio": round(float(details["contamination_ratio"]), 6),
                "rgb_std_norm": round(float(details["rgb_std_norm"]), 6),
                "area_ratio": round(float(details["area_ratio"]), 6),
                "usable_flag": round(float(details["usable_flag"]), 6),
                "source": "direct",
                "geometry_model": "direct_dark_panel",
                "direct_panel_mode": direct_panel_mode,
                "direct_panel_cell_source": raw_direct_panel_cell_source,
                "raw_direct_panel_cell_source": raw_direct_panel_cell_source,
                "decision_reason": (
                    "direct_inner_lattice_mainline"
                    if direct_panel_mode == "inner_lattice_estimated"
                    else "direct_outer_box_fallback"
                ),
                "geometry": geometry,
                "size": size,
                "quad_size": {
                    "width": round(float(quad.width), 6),
                    "height": round(float(quad.height), 6),
                    "area": round(float(quad_area), 6),
                    "rotation_degrees": round(float(quad.rotation_degrees), 6),
                    "bbox_area": round(float(bbox_area), 6),
                    "bbox_to_quad_area_ratio": round(
                        float(bbox_area / max(1.0, quad_area)),
                        6,
                    ),
                },
                "rescue": {
                    "accepted": False,
                    "decision_reason": "not_applicable",
                },
            }
            if direct_panel_mode == "inner_lattice_estimated":
                accepted_roi_source = (
                    "adopted_current_blue"
                    if raw_direct_panel_cell_source == "lattice"
                    else raw_direct_panel_cell_source
                )
                entry["direct_panel_cell_source"] = accepted_roi_source
                entry["accepted_roi_policy"] = "current_blue"
                entry["accepted_roi_source"] = accepted_roi_source
            final_entries.append(entry)

        summary = self._build_patchwise_summary(
            final_entries,
            refined_patch_count=0,
            rescue_patch_count=0,
            fallback_patch_count=0,
            weak_patch_count=weak_patch_count,
            geometry_model="direct_dark_panel",
            direct_dark_panel_diagnostics=diagnostics,
            status="ok",
            direct_patch_count=len(final_entries),
            direct_panel_mode=str(diagnostics.get("direct_panel_mode", "outer_box_fallback")),
        )
        return final_entries, summary

    def _prepare_patchwise_rois_legacy_after_rigid_failure(
        self,
        frame_bgr: np.ndarray,
        *,
        reason: str,
        quads_count: int,
        boxes_count: int,
        default_boxes_count: int,
        expected_count: int,
        rigid_corners_source: str,
    ) -> dict[str, object]:
        original_source = self._rigid_corners_source
        try:
            self._rigid_corners_source = None
            summary = self.prepare_patchwise_rois_from_frame(frame_bgr)
        finally:
            self._rigid_corners_source = original_source

        diagnostics = {
            "rigid_path_attempted": True,
            "rigid_path_fallback_reason": reason,
            "rigid_path_quads_count": int(quads_count),
            "rigid_path_boxes_count": int(boxes_count),
            "rigid_path_default_boxes_count": int(default_boxes_count),
            "rigid_path_expected_count": int(expected_count),
            "rigid_corners_source": rigid_corners_source,
        }
        summary = dict(summary)
        summary.update(diagnostics)
        if self._patchwise_summary is not None:
            self._patchwise_summary = dict(self._patchwise_summary)
            self._patchwise_summary.update(diagnostics)
        else:
            self._patchwise_summary = dict(summary)
        return dict(summary)

    def _prepare_patchwise_rois_rigid_path(self, frame_bgr: np.ndarray) -> dict[str, object]:
        expected_count = self.n_rows * self.n_cols
        source = str(self._rigid_corners_source or "")
        quads = self._rigid_patch_quads_from_homography()
        boxes = self._rigid_patch_boxes_from_quads(quads)
        default_boxes = self._default_patch_display_boxes()
        if (
            len(quads) != expected_count
            or len(boxes) != expected_count
            or len(default_boxes) != expected_count
        ):
            return self._prepare_patchwise_rois_legacy_after_rigid_failure(
                frame_bgr,
                reason="rigid_helper_invalid_patch_count",
                quads_count=len(quads),
                boxes_count=len(boxes),
                default_boxes_count=len(default_boxes),
                expected_count=expected_count,
                rigid_corners_source=source,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 28, 90)
        seeds_by_label, ordered_labels = self._build_patchwise_seed_info(default_boxes)
        boxes_by_label = {
            label: boxes[idx]
            for idx, label in enumerate(ordered_labels)
        }
        quads_by_label = {
            label: quads[idx]
            for idx, label in enumerate(ordered_labels)
        }

        final_entries: list[dict[str, object]] = []
        weak_patch_count = 0
        for label in ordered_labels:
            seed_info = seeds_by_label[label]
            seed_box = seed_info["seed_box"]
            seed_quad = self._box_to_quad(seed_box)
            box = boxes_by_label[label]
            quad = quads_by_label[label]
            score, details = self._candidate_score(
                label,
                box,
                boxes_by_label,
                seeds_by_label,
                frame_bgr,
                gray,
                edge_map,
            )
            if score < SPYDERCHECKR_PATCHWISE_SCORE_THRESHOLD or details["usable_flag"] < 1.0:
                weak_patch_count += 1
            geometry = {
                "predicted_center": [
                    round(float(details["predicted_cx"]), 3),
                    round(float(details["predicted_cy"]), 3),
                ],
                "dx_px": round(float(details["dx_px"]), 6),
                "dy_px": round(float(details["dy_px"]), 6),
                "dx_ratio": round(float(details["dx_ratio"]), 6),
                "dy_ratio": round(float(details["dy_ratio"]), 6),
                "center_deviation_ratio": round(
                    float(details["center_deviation_ratio"]),
                    6,
                ),
                "passes_center_guard": bool(details["passes_center_guard"]),
            }
            col_pitch = max(1.0, float(seed_info["col_pitch"]))
            row_pitch = max(1.0, float(seed_info["row_pitch"]))
            size = {
                "width_ratio_to_col_pitch": round(float(box.width) / col_pitch, 6),
                "height_ratio_to_row_pitch": round(float(box.height) / row_pitch, 6),
                "area_ratio_to_cell_pitch_area": round(
                    float(box.area) / max(1.0, col_pitch * row_pitch),
                    6,
                ),
            }
            bbox_area = float(max(1, box.area))
            quad_area = float(max(0.0, quad.area))
            final_entries.append(
                {
                    "label": label,
                    "row_index": int(seed_info["row_index"]),
                    "col_index": int(seed_info["col_index"]),
                    "seed_roi_box": seed_box.as_dict(),
                    "seed_roi_quad": seed_quad.as_dict(),
                    "roi_box": box.as_dict(),
                    "roi_quad": quad.as_dict(),
                    "roi_shape": "quad",
                    "center": [round(float(quad.cx), 3), round(float(quad.cy), 3)],
                    "roi_score": round(float(score), 6),
                    "local_quality": round(float(details["local_quality"]), 6),
                    "leakage_ratio": round(float(details["leakage_ratio"]), 6),
                    "contamination_ratio": round(float(details["contamination_ratio"]), 6),
                    "rgb_std_norm": round(float(details["rgb_std_norm"]), 6),
                    "area_ratio": round(float(details["area_ratio"]), 6),
                    "usable_flag": round(float(details["usable_flag"]), 6),
                    "source": "rigid_lock",
                    "geometry_model": "rigid_grid_lock",
                    "rigid_corners_source": source,
                    "decision_reason": "homography_rigid_grid_lock",
                    "geometry": geometry,
                    "size": size,
                    "quad_size": {
                        "width": round(float(quad.width), 6),
                        "height": round(float(quad.height), 6),
                        "area": round(float(quad_area), 6),
                        "rotation_degrees": round(float(quad.rotation_degrees), 6),
                        "bbox_area": round(float(bbox_area), 6),
                        "bbox_to_quad_area_ratio": round(
                            float(bbox_area / max(1.0, quad_area)),
                            6,
                        ),
                    },
                    "rescue": {
                        "accepted": False,
                        "decision_reason": "not_applicable",
                    },
                }
            )

        self._patchwise_boxes = [boxes_by_label[label] for label in ordered_labels]
        self._patchwise_quads = [quads_by_label[label] for label in ordered_labels]
        self._patchwise_entries = final_entries
        summary = self._build_patchwise_summary(
            final_entries,
            refined_patch_count=0,
            rescue_patch_count=0,
            fallback_patch_count=0,
            weak_patch_count=weak_patch_count,
            geometry_model="rigid_grid_lock",
            direct_dark_panel_diagnostics={
                "status": "skipped",
                "reason": "rigid_grid_lock",
            },
            status="ok",
            direct_patch_count=len(quads),
        )
        summary.update(
            {
                "rigid_path_attempted": True,
                "rigid_path_fallback_reason": None,
                "rigid_corners_source": source,
                "rigid_patch_count": len(quads),
                "rigid_box_count": len(boxes),
                "rigid_expected_count": expected_count,
                "rigid_roi_shape": "quad",
                "rigid_lock_rule": "homography_center_square_face_no_per_patch_search",
            }
        )
        self._patchwise_summary = summary
        return dict(summary)

    def prepare_patchwise_rois_from_oriented_panel_payload(
        self,
        frame_bgr: np.ndarray,
        payload: SpyderCheckrOrientedPanelPayload,
    ) -> dict[str, object]:
        patch_count = self.n_rows * self.n_cols
        if (self.n_rows, self.n_cols) != SPYDERCHECKR_48_SHAPE:
            self.invalidate_patchwise_rois()
            summary = self._structured_oriented_payload_failure_summary(
                patch_count=patch_count,
                reason="oriented_payload_invalid",
            )
            self._patchwise_summary = summary
            return dict(summary)
        if (
            frame_bgr is None
            or frame_bgr.size == 0
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            self.invalidate_patchwise_rois()
            summary = self._structured_oriented_payload_failure_summary(
                patch_count=patch_count,
                reason="oriented_payload_invalid",
            )
            self._patchwise_summary = summary
            return dict(summary)

        centers = np.asarray(payload.centers_48_xy, dtype=np.float64)
        quads_raw = tuple(payload.sampling_quads_48_xy)
        if centers.shape != (patch_count, 2) or len(quads_raw) != patch_count:
            self.invalidate_patchwise_rois()
            summary = self._structured_oriented_payload_failure_summary(
                patch_count=patch_count,
                reason="oriented_payload_patch_count_invalid",
            )
            self._patchwise_summary = summary
            return dict(summary)
        if not np.all(np.isfinite(centers)):
            self.invalidate_patchwise_rois()
            summary = self._structured_oriented_payload_failure_summary(
                patch_count=patch_count,
                reason="oriented_payload_geometry_invalid",
            )
            self._patchwise_summary = summary
            return dict(summary)

        panels = tuple(payload.panels)
        panel_by_side = {
            str(panel.get("side")): panel
            for panel in panels
            if isinstance(panel, dict)
        }
        if {"left", "right"} - set(panel_by_side):
            self.invalidate_patchwise_rois()
            summary = self._structured_oriented_payload_failure_summary(
                patch_count=patch_count,
                reason="oriented_payload_invalid",
            )
            self._patchwise_summary = summary
            return dict(summary)

        quad_objects: list[_PatchwiseDisplayQuad] = []
        for quad_points in quads_raw:
            quad = self._coerce_oriented_payload_quad(quad_points)
            if quad is None:
                self.invalidate_patchwise_rois()
                summary = self._structured_oriented_payload_failure_summary(
                    patch_count=patch_count,
                    reason="oriented_payload_geometry_invalid",
                )
                self._patchwise_summary = summary
                return dict(summary)
            quad_objects.append(quad)

        centers_grid = centers.reshape(self.n_rows, self.n_cols, 2)
        left_centers = centers_grid[:, :4, :]
        right_centers = centers_grid[:, 4:, :]
        left_col_pitch, left_row_pitch = self._panel_pitch_medians(left_centers)
        right_col_pitch, right_row_pitch = self._panel_pitch_medians(right_centers)
        if min(left_col_pitch, left_row_pitch, right_col_pitch, right_row_pitch) <= 0.0:
            self.invalidate_patchwise_rois()
            summary = self._structured_oriented_payload_failure_summary(
                patch_count=patch_count,
                reason="oriented_payload_geometry_invalid",
            )
            self._patchwise_summary = summary
            return dict(summary)
        global_pitch_candidates = [
            float(left_col_pitch),
            float(left_row_pitch),
            float(right_col_pitch),
            float(right_row_pitch),
        ]
        if not all(math.isfinite(value) and value > 0.0 for value in global_pitch_candidates):
            self.invalidate_patchwise_rois()
            summary = self._structured_oriented_payload_failure_summary(
                patch_count=patch_count,
                reason="oriented_payload_geometry_invalid",
            )
            self._patchwise_summary = summary
            return dict(summary)
        global_pitch = float(min(global_pitch_candidates))
        global_pitch_median = float(np.median(np.asarray(global_pitch_candidates, dtype=np.float64)))
        global_roi_side = max(
            4.0,
            global_pitch
            * (1.0 - (2.0 * SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO)),
        )
        normalized_quads: list[_PatchwiseDisplayQuad] = []
        for idx, quad in enumerate(quad_objects):
            axes = self._axes_from_display_quad(quad)
            if axes is None:
                self.invalidate_patchwise_rois()
                summary = self._structured_oriented_payload_failure_summary(
                    patch_count=patch_count,
                    reason="oriented_payload_geometry_invalid",
                )
                self._patchwise_summary = summary
                return dict(summary)
            x_axis, y_axis = axes
            normalized_quads.append(
                self._centered_oriented_quad(
                    float(centers[idx, 0]),
                    float(centers[idx, 1]),
                    global_roi_side,
                    global_roi_side,
                    x_axis,
                    y_axis,
                )
            )
        quad_objects = normalized_quads

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 28, 90)
        edge_map = cv2.dilate(edge_map, np.ones((3, 3), dtype=np.uint8), iterations=1)

        ordered_labels: list[str] = []
        seeds_by_label: dict[str, dict[str, object]] = {}
        boxes_by_label: dict[str, _PatchwiseDisplayBox] = {}
        quads_by_label: dict[str, _PatchwiseDisplayQuad] = {}
        half = self.n_cols // 2
        for idx in range(patch_count):
            row = idx // self.n_cols
            col = idx % self.n_cols
            label = self._patch_label(row, col)
            quad = quad_objects[idx]
            box = quad.bounding_box
            center_x = float(centers[idx, 0])
            center_y = float(centers[idx, 1])
            is_right_panel = col >= half
            col_pitch = right_col_pitch if is_right_panel else left_col_pitch
            row_pitch = right_row_pitch if is_right_panel else left_row_pitch
            seeds_by_label[label] = {
                "row_index": row,
                "col_index": col,
                "panel_col_index": col - half if is_right_panel else col,
                "is_right_panel": is_right_panel,
                "legacy_seed_box": box,
                "seed_box": box,
                "seed_quad": quad,
                "roi_quad": quad,
                "predicted_cx": center_x,
                "predicted_cy": center_y,
                "col_pitch": float(col_pitch),
                "row_pitch": float(row_pitch),
                "rescue_min_width": max(
                    int(box.width),
                    int(round(float(col_pitch) * SPYDERCHECKR_PATCHWISE_RESCUE_SIZE_FLOOR_RATIO)),
                ),
                "rescue_min_height": max(
                    int(box.height),
                    int(round(float(row_pitch) * SPYDERCHECKR_PATCHWISE_RESCUE_SIZE_FLOOR_RATIO)),
                ),
                "direct_panel_mode": "inner_lattice_estimated",
                "direct_panel_cell_source": "oriented_panel_payload",
            }
            ordered_labels.append(label)
            boxes_by_label[label] = box
            quads_by_label[label] = quad

        diagnostics: dict[str, object] = {
            "status": "ok",
            "reason": "oriented_panel_payload",
            "direct_panel_mode": "inner_lattice_estimated",
            "order_name": str(payload.order_name),
            "color_score": round(float(payload.color_score), 6),
            "visible_patch_alignment": dict(payload.visible_patch_alignment),
            "pair_diagnostics": dict(payload.pair_diagnostics),
            "roi_square_side": round(float(global_roi_side), 6),
            "roi_square_side_policy": "chart_global_min_pitch_inset",
            "roi_pitch_candidate_count": int(len(global_pitch_candidates)),
            "roi_pitch_min_px": round(float(global_pitch), 6),
            "roi_pitch_median_px": round(float(global_pitch_median), 6),
            "panel_pair": {},
        }
        panel_pair = diagnostics["panel_pair"]
        assert isinstance(panel_pair, dict)
        for side, center_grid, col_pitch, row_pitch, col_start, col_stop in (
            ("left", left_centers, left_col_pitch, left_row_pitch, 0, half),
            ("right", right_centers, right_col_pitch, right_row_pitch, half, self.n_cols),
        ):
            panel = dict(panel_by_side[side])
            panel_hull = panel.get("panel_hull_xy")
            patch_face_hull = panel.get("patch_face_hull_xy")
            panel_box = self._panel_hull_box_from_points(panel_hull)
            try:
                u_axis = np.asarray(panel["u_axis"], dtype=np.float64)
                v_axis = np.asarray(panel["v_axis"], dtype=np.float64)
                panel_center = np.asarray(panel["center_xy"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                panel_box = None
                u_axis = np.asarray([], dtype=np.float64)
                v_axis = np.asarray([], dtype=np.float64)
                panel_center = np.asarray([], dtype=np.float64)
            if (
                panel_box is None
                or u_axis.shape != (2,)
                or v_axis.shape != (2,)
                or panel_center.shape != (2,)
                or not np.all(np.isfinite(u_axis))
                or not np.all(np.isfinite(v_axis))
                or not np.all(np.isfinite(panel_center))
            ):
                self.invalidate_patchwise_rois()
                summary = self._structured_oriented_payload_failure_summary(
                    patch_count=patch_count,
                    reason="oriented_payload_geometry_invalid",
                )
                self._patchwise_summary = summary
                return dict(summary)

            roi_side = float(global_roi_side)
            patch_face_side = roi_side / max(
                1e-6,
                1.0 - (2.0 * SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO),
            )
            origin_xy = center_grid[0, 0]
            layout = {
                "status": "ok",
                "mode": "inner_lattice_estimated",
                "reason": "oriented_panel_payload",
                "panel_side": side,
                "panel_box": panel_box.as_dict(),
                "center_xy": [round(float(panel_center[0]), 6), round(float(panel_center[1]), 6)],
                "grid_x_fracs": list(panel.get("grid_x_fracs") or ()),
                "grid_y_fracs": list(panel.get("grid_y_fracs") or ()),
                "panel_hull_xy": panel_hull,
                "patch_face_hull_xy": patch_face_hull,
                "column_step": round(float(col_pitch), 6),
                "row_step": round(float(row_pitch), 6),
                "roi_square_side": round(float(roi_side), 6),
                "roi_square_side_policy": "chart_global_min_pitch_inset",
                "roi_inset_ratio": round(float(SPYDERCHECKR_DIRECT_PANEL_ROI_INSET_RATIO), 6),
                "estimated_face_side": round(float(patch_face_side), 6),
                "estimated_face_width": round(float(patch_face_side), 6),
                "estimated_face_height": round(float(patch_face_side), 6),
                "lattice_rotation_degrees": round(
                    float(np.degrees(np.arctan2(u_axis[1], u_axis[0]))),
                    6,
                ),
                "lattice_origin_xy": [round(float(origin_xy[0]), 6), round(float(origin_xy[1]), 6)],
                "lattice_col_vector_xy": [
                    round(float(u_axis[0] * col_pitch), 6),
                    round(float(u_axis[1] * col_pitch), 6),
                ],
                "lattice_row_vector_xy": [
                    round(float(v_axis[0] * row_pitch), 6),
                    round(float(v_axis[1] * row_pitch), 6),
                ],
                "lattice_origin_local": [
                    round(float(origin_xy[0] - panel_box.left), 6),
                    round(float(origin_xy[1] - panel_box.top), 6),
                ],
                "lattice_col_vector_local": [
                    round(float(u_axis[0] * col_pitch), 6),
                    round(float(u_axis[1] * col_pitch), 6),
                ],
                "lattice_row_vector_local": [
                    round(float(v_axis[0] * row_pitch), 6),
                    round(float(v_axis[1] * row_pitch), 6),
                ],
                "cells_from_components_count": 0,
                "cells_from_lattice_only_count": 24,
                "direct_panel_cell_source": "oriented_panel_payload",
            }
            diagnostics[f"{side}_panel_layout"] = layout
            panel_pair[f"{side}_box"] = panel_box.as_dict()

        final_entries, summary = self._finalize_direct_dark_panel_entries(
            frame_bgr,
            gray,
            edge_map,
            seeds_by_label,
            ordered_labels,
            boxes_by_label,
            quads_by_label,
            diagnostics,
        )
        self._patchwise_boxes = [boxes_by_label[label] for label in ordered_labels]
        self._patchwise_quads = [quads_by_label[label] for label in ordered_labels]
        self._patchwise_entries = final_entries
        self._patchwise_summary = summary
        return dict(summary)

    def prepare_patchwise_rois_from_frame(self, frame_bgr: np.ndarray) -> dict[str, object]:
        """現在フレームから bounded patchwise ROI を準備する。"""
        if frame_bgr is None or frame_bgr.size == 0:
            self.invalidate_patchwise_rois()
            patch_count = self.n_rows * self.n_cols
            geometry_viability = _evaluate_patchwise_calibration_geometry(
                status="fallback",
                patch_count=patch_count,
                rescue_patch_count=0,
                fallback_patch_count=patch_count,
                weak_patch_count=0,
                median_area_ratio_to_cell_pitch_area=0.0,
            )
            summary = {
                "status": "fallback",
                "reason": "frame_or_geometry_unavailable",
                "refined_patch_count": 0,
                "rescue_patch_count": 0,
                "fallback_patch_count": patch_count,
                "weak_patch_count": 0,
                "direct_patch_count": 0,
                "geometry_model": "unavailable",
                "center_tolerance_ratio": SPYDERCHECKR_PATCHWISE_CENTER_TOLERANCE_RATIO,
                "max_center_deviation_ratio": 0.0,
                "mean_center_deviation_ratio": 0.0,
                "median_width_ratio_to_col_pitch": 0.0,
                "median_height_ratio_to_row_pitch": 0.0,
                "median_area_ratio_to_cell_pitch_area": 0.0,
                "failing_center_guard_patch_count": 0,
                "direct_dark_panel_diagnostics": {
                    "status": "failed",
                    "reason": "frame_or_geometry_unavailable",
                },
            }
            summary.update(geometry_viability)
            self._patchwise_summary = summary
            return dict(summary)

        if self.has_rigid_quad():
            return self._prepare_patchwise_rois_rigid_path(frame_bgr)

        default_boxes = self._default_patch_display_boxes()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 28, 90)
        edge_map = cv2.dilate(edge_map, np.ones((3, 3), dtype=np.uint8), iterations=1)
        image_h, image_w = gray.shape
        direct_layout = self._attempt_direct_dark_panel_layout(frame_bgr, default_boxes)
        direct_diagnostics = dict(direct_layout.get("diagnostics") or {})
        if bool(direct_layout.get("ok")):
            final_boxes = dict(direct_layout["boxes_by_label"])
            final_quads = dict(direct_layout.get("quads_by_label") or {})
            final_entries, summary = self._finalize_direct_dark_panel_entries(
                frame_bgr,
                gray,
                edge_map,
                direct_layout["seeds_by_label"],
                direct_layout["ordered_labels"],
                final_boxes,
                final_quads,
                direct_diagnostics,
            )
            self._patchwise_boxes = [final_boxes[label] for label in direct_layout["ordered_labels"]]
            self._patchwise_quads = [
                final_quads.get(label, self._box_to_quad(final_boxes[label]))
                for label in direct_layout["ordered_labels"]
            ]
            self._patchwise_entries = final_entries
            self._patchwise_summary = summary
            return dict(summary)

        if not default_boxes:
            self.invalidate_patchwise_rois()
            patch_count = self.n_rows * self.n_cols
            geometry_viability = _evaluate_patchwise_calibration_geometry(
                status="fallback",
                patch_count=patch_count,
                rescue_patch_count=0,
                fallback_patch_count=patch_count,
                weak_patch_count=0,
                median_area_ratio_to_cell_pitch_area=0.0,
            )
            summary = {
                "status": "fallback",
                "reason": "seed_geometry_unavailable",
                "refined_patch_count": 0,
                "rescue_patch_count": 0,
                "fallback_patch_count": patch_count,
                "weak_patch_count": 0,
                "direct_patch_count": 0,
                "geometry_model": "legacy_patchwise_fallback",
                "center_tolerance_ratio": SPYDERCHECKR_PATCHWISE_CENTER_TOLERANCE_RATIO,
                "max_center_deviation_ratio": 0.0,
                "mean_center_deviation_ratio": 0.0,
                "median_width_ratio_to_col_pitch": 0.0,
                "median_height_ratio_to_row_pitch": 0.0,
                "median_area_ratio_to_cell_pitch_area": 0.0,
                "failing_center_guard_patch_count": 0,
                "direct_dark_panel_diagnostics": direct_diagnostics,
            }
            summary.update(geometry_viability)
            self._patchwise_summary = summary
            return dict(summary)

        seeds_by_label, ordered_labels = self._build_patchwise_seed_info(default_boxes)
        boxes_by_label: dict[str, _PatchwiseDisplayBox] = {}
        seed_sizes = [
            min(int(seed["seed_box"].width), int(seed["seed_box"].height))
            for seed in seeds_by_label.values()
        ]
        base_size = float(np.median(seed_sizes)) if seed_sizes else 0.0
        step_values = [max(1, int(round(base_size * 0.08))), 1]
        if step_values[0] == step_values[1]:
            step_values = [step_values[0]]

        for label, seed_info in seeds_by_label.items():
            boxes_by_label[label] = seed_info["seed_box"]

        for step in step_values:
            for _ in range(6):
                improved = False
                for label in ordered_labels:
                    current = boxes_by_label[label]
                    seed_box = seeds_by_label[label]["seed_box"]
                    min_box_size = max(
                        8,
                        int(seeds_by_label[label].get("rescue_min_width", seed_box.width)),
                        int(seeds_by_label[label].get("rescue_min_height", seed_box.height)),
                        int(round(min(seed_box.width, seed_box.height) * 0.55)),
                    )
                    max_box_size = max(min_box_size, int(round(max(seed_box.width, seed_box.height) * 1.15)))
                    max_center_shift = max(2.0, min(seed_box.width, seed_box.height) * 0.12)
                    best_box = current
                    best_score, _ = self._candidate_score(
                        label,
                        current,
                        boxes_by_label,
                        seeds_by_label,
                        frame_bgr,
                        gray,
                        edge_map,
                    )
                    for candidate in self._generate_patch_candidates(
                        current,
                        seed_info,
                        image_w,
                        image_h,
                        step,
                        min_box_size,
                        max_box_size,
                        max_center_shift,
                    ):
                        if candidate == current:
                            continue
                        trial_boxes = dict(boxes_by_label)
                        trial_boxes[label] = candidate
                        score, _ = self._candidate_score(
                            label,
                            candidate,
                            trial_boxes,
                            seeds_by_label,
                            frame_bgr,
                            gray,
                            edge_map,
                        )
                        if score > best_score + 1e-6:
                            best_box = candidate
                            best_score = score
                    if best_box != current:
                        boxes_by_label[label] = best_box
                        improved = True
                if not improved:
                    break

        final_boxes = dict(boxes_by_label)
        final_entries: list[dict[str, object]] = []
        refined_patch_count = 0
        rescue_patch_count = 0
        fallback_patch_count = 0
        weak_patch_count = 0
        for label in ordered_labels:
            seed_info = seeds_by_label[label]
            seed_box = seed_info["seed_box"]
            box = final_boxes[label]
            score, details = self._candidate_score(
                label,
                box,
                final_boxes,
                seeds_by_label,
                frame_bgr,
                gray,
                edge_map,
            )
            source = "default" if box == seed_box else "refined"
            decision_reason = "default_centered_box" if source == "default" else "refined_guard_pass"
            rescue = {"accepted": False, "decision_reason": ""}
            if score < SPYDERCHECKR_PATCHWISE_SCORE_THRESHOLD or details["usable_flag"] < 1.0:
                weak_patch_count += 1
                candidate_box, candidate_metrics = self._search_rescue_patch_box(
                    frame_bgr,
                    gray,
                    edge_map,
                    box,
                    seed_info,
                )
                accepted, decision_reason = self._should_accept_rescue(details, candidate_metrics)
                rescue = {"accepted": bool(accepted), "decision_reason": decision_reason}
                if accepted:
                    box = candidate_box
                    final_boxes[label] = candidate_box
                    score, details = self._candidate_score(
                        label,
                        box,
                        final_boxes,
                        seeds_by_label,
                        frame_bgr,
                        gray,
                        edge_map,
                    )
                    source = "rescue"
                    rescue_patch_count += 1
                else:
                    box = seed_box
                    final_boxes[label] = seed_box
                    score, details = self._candidate_score(
                        label,
                        box,
                        final_boxes,
                        seeds_by_label,
                        frame_bgr,
                        gray,
                        edge_map,
                    )
                    source = "default"
                    if rescue["decision_reason"] == "center guard failed":
                        decision_reason = "center_guard_default_fallback"
                    else:
                        decision_reason = "weak_default_fallback"
            if not bool(details.get("passes_center_guard", False)):
                box = seed_box
                final_boxes[label] = seed_box
                score, details = self._candidate_score(
                    label,
                    box,
                    final_boxes,
                    seeds_by_label,
                    frame_bgr,
                    gray,
                    edge_map,
                )
                source = "default"
                rescue = {
                    "accepted": False,
                    "decision_reason": "center_guard_default_fallback",
                }
                decision_reason = "center_guard_default_fallback"
            if source == "default":
                fallback_patch_count += 1
            elif source == "refined":
                refined_patch_count += 1
            geometry = {
                "predicted_center": [
                    round(float(details["predicted_cx"]), 3),
                    round(float(details["predicted_cy"]), 3),
                ],
                "dx_px": round(float(details["dx_px"]), 6),
                "dy_px": round(float(details["dy_px"]), 6),
                "dx_ratio": round(float(details["dx_ratio"]), 6),
                "dy_ratio": round(float(details["dy_ratio"]), 6),
                "center_deviation_ratio": round(
                    float(details["center_deviation_ratio"]),
                    6,
                ),
                "passes_center_guard": bool(details["passes_center_guard"]),
            }
            col_pitch = max(1.0, float(seed_info["col_pitch"]))
            row_pitch = max(1.0, float(seed_info["row_pitch"]))
            size = {
                "width_ratio_to_col_pitch": round(float(box.width) / col_pitch, 6),
                "height_ratio_to_row_pitch": round(float(box.height) / row_pitch, 6),
                "area_ratio_to_cell_pitch_area": round(
                    float(box.area) / max(1.0, col_pitch * row_pitch),
                    6,
                ),
            }
            quad = self._box_to_quad(box)
            seed_quad = self._box_to_quad(seed_box)
            final_entries.append(
                {
                    "label": label,
                    "row_index": int(seed_info["row_index"]),
                    "col_index": int(seed_info["col_index"]),
                    "seed_roi_box": seed_box.as_dict(),
                    "seed_roi_quad": seed_quad.as_dict(),
                    "roi_box": box.as_dict(),
                    "roi_quad": quad.as_dict(),
                    "roi_shape": "box",
                    "center": [round(float(box.cx), 3), round(float(box.cy), 3)],
                    "roi_score": round(float(score), 6),
                    "local_quality": round(float(details["local_quality"]), 6),
                    "leakage_ratio": round(float(details["leakage_ratio"]), 6),
                    "contamination_ratio": round(float(details["contamination_ratio"]), 6),
                    "rgb_std_norm": round(float(details["rgb_std_norm"]), 6),
                    "area_ratio": round(float(details["area_ratio"]), 6),
                    "usable_flag": round(float(details["usable_flag"]), 6),
                    "source": source,
                    "geometry_model": "legacy_patchwise_fallback",
                    "decision_reason": decision_reason,
                    "geometry": geometry,
                    "size": size,
                    "quad_size": {
                        "width": round(float(quad.width), 6),
                        "height": round(float(quad.height), 6),
                        "area": round(float(quad.area), 6),
                        "rotation_degrees": round(float(quad.rotation_degrees), 6),
                        "bbox_area": round(float(box.area), 6),
                        "bbox_to_quad_area_ratio": 1.0,
                    },
                    "rescue": rescue,
                }
            )

        self._patchwise_boxes = [final_boxes[label] for label in ordered_labels]
        self._patchwise_quads = [self._box_to_quad(final_boxes[label]) for label in ordered_labels]
        self._patchwise_entries = final_entries
        summary = self._build_patchwise_summary(
            final_entries,
            refined_patch_count=refined_patch_count,
            rescue_patch_count=rescue_patch_count,
            fallback_patch_count=fallback_patch_count,
            weak_patch_count=weak_patch_count,
            geometry_model="legacy_patchwise_fallback",
            direct_dark_panel_diagnostics=direct_diagnostics,
            status="ok",
            direct_patch_count=0,
        )
        self._patchwise_summary = summary
        return dict(summary)

    def extract_all_patch_bayer_means(
        self,
        avg_raw_bayer: np.ndarray,
        bayer_extractor,
        metadata: dict,
        cfg,
        debug: bool = False,
        debug_save_dir: str = "",
    ) -> np.ndarray:
        """全パッチのBayer平均値を一括抽出する。

        Args:
          avg_raw_bayer: 複数フレーム平均済みの RAW Bayer 配列 (H, W) uint16。
          bayer_extractor: BayerROIExtractor インスタンス（duck-typing）。
          metadata: Picamera2 メタデータ辞書。
          cfg: CSIConfig インスタンス（display_to_raw_coords に渡す）。
          debug: True のとき全パッチの表示座標・RAW座標・raw値を出力する。
          debug_save_dir: 非空文字列なら RAW ROI 可視化画像をそのディレクトリに保存する。
        Returns:
          np.ndarray: shape=(n_patches, 3) の float32 RGB Bayer 平均値。
        """
        display_boxes = self.get_active_patch_display_boxes()
        display_quads = self.get_active_patch_display_quads()
        panel_offset_x = getattr(getattr(cfg, "camera", None), "left_panel_width", 0)
        patchwise_quads_active = (
            self._patchwise_quads is not None
            and len(self._patchwise_quads) == len(display_boxes)
        )
        summary = self._patchwise_summary or {}
        diagnostics = summary.get("direct_dark_panel_diagnostics") or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        force_canonical_quad_sampling = bool(
            patchwise_quads_active
            and summary.get("geometry_model") == "direct_dark_panel"
            and summary.get("direct_panel_mode") == "inner_lattice_estimated"
            and diagnostics.get("reason") == "oriented_panel_payload"
        )
        active_source = (
            "patchwise_quad"
            if patchwise_quads_active
            and (
                force_canonical_quad_sampling
                or any(abs(float(quad.rotation_degrees)) > 0.5 for quad in self._patchwise_quads)
            )
            else "patchwise"
            if self._patchwise_boxes is not None
            and len(self._patchwise_boxes) == len(display_boxes)
            else "default"
        )

        if debug:
            _S = "═" * 80
            print(f"\n{_S}")
            print("  [DBG] extract_all_patch_bayer_means — パッチ座標・RAW値")
            print(f"  grid: {self.n_rows}行 × {self.n_cols}列 = {self.n_rows*self.n_cols}パッチ")
            scaler_crop = metadata.get("ScalerCrop", "N/A")
            print(f"  ScalerCrop={scaler_crop}")
            print(f"  disp_size=({bayer_extractor.disp_w}, {bayer_extractor.disp_h})  "
                  f"raw_size=({bayer_extractor.raw_w}, {bayer_extractor.raw_h})  "
                  f"sensor_size=({bayer_extractor.sensor_w}, {bayer_extractor.sensor_h})")
            print(f"  flip_h={cfg.display.flip_horizontal}  flip_v={cfg.display.flip_vertical}")
            print(f"  hinge_gap={self.hinge_gap}  patch_margin={self.patch_margin}  panel_offset_x={panel_offset_x}")
            print(f"  col_x_norm_offsets={[round(v, 1) for v in self.col_x_norm_offsets]}")
            print(f"  active_roi_source={active_source}")
            print(f"  {'idx':>3}  {'row':>3}  {'col':>3}  {'disp_x':>7}  {'disp_y':>7}  "
                  f"{'disp_w':>6}  {'disp_h':>6}  {'raw_x':>7}  {'raw_y':>7}  {'raw_w':>6}  {'raw_h':>6}  "
                  f"{'raw_R':>7}  {'raw_G':>7}  {'raw_B':>7}")
            print("─" * 80)

        results = []
        raw_rois = []   # デバッグ可視化用に RAW ROI 座標を保持
        raw_roi_quads: list[list[tuple[float, float]] | None] = []
        canonical_sampling_records: list[dict[str, object]] = []
        for idx, box in enumerate(display_boxes):
            row = idx // self.n_cols
            col = idx % self.n_cols
            quad = display_quads[idx] if idx < len(display_quads) else self._box_to_quad(box)
            quad_result = None
            if force_canonical_quad_sampling or abs(float(quad.rotation_degrees)) > 0.5:
                quad_result = self._extract_raw_quad_bayer_means(
                    avg_raw_bayer,
                    bayer_extractor,
                    metadata,
                    cfg,
                    quad,
                    int(panel_offset_x),
                )
            if quad_result is not None:
                means, (rx, ry, rw, rh), raw_quad, sampling_meta = quad_result
                raw_roi_quads.append(raw_quad)
                record = dict(sampling_meta)
                record["patch_index"] = int(idx)
                record["label"] = self._patch_label(row, col)
                canonical_sampling_records.append(record)
            else:
                disp_x = box.left - panel_offset_x
                disp_y = box.top
                rx, ry, rw, rh = bayer_extractor.display_to_raw_coords(
                    disp_x,
                    disp_y,
                    box.width,
                    box.height,
                    metadata,
                    flip_horizontal=cfg.display.flip_horizontal,
                    flip_vertical=cfg.display.flip_vertical,
                )
                roi = bayer_extractor.extract_raw_roi(avg_raw_bayer, rx, ry, rw, rh)
                means = bayer_extractor._extract_bayer_channel_means(roi)
                raw_roi_quads.append(None)
            results.append(means)
            raw_rois.append((rx, ry, rw, rh))

            if debug:
                print(f"  {idx:>3}  {row:>3}  {col:>3}  {box.left:>7}  {box.top:>7}  "
                      f"{box.width:>6}  {box.height:>6}  "
                      f"{rx:>7}  {ry:>7}  {rw:>6}  {rh:>6}  "
                      f"{int(means[0]):>7}  {int(means[1]):>7}  {int(means[2]):>7}")

        if debug:
            print(_S + "\n")

        # デバッグ用 RAW ROI 可視化画像の保存
        if debug_save_dir:
            self._save_raw_roi_debug_image(
                avg_raw_bayer, raw_rois, debug_save_dir, raw_roi_quads,
            )

        if canonical_sampling_records:
            samples_per_channel = sorted(
                {
                    int(record.get("samples_per_channel", 0) or 0)
                    for record in canonical_sampling_records
                }
            )
            total_samples = sorted(
                {
                    int(record.get("total_channel_samples", 0) or 0)
                    for record in canonical_sampling_records
                }
            )
            self._last_raw_quad_sampling_summary = {
                "mode": "canonical_square_fixed_bayer_phase",
                "canonical_patch_count": int(len(canonical_sampling_records)),
                "canonical_grid_side": int(SPYDERCHECKR_RAW_CANONICAL_SAMPLE_GRID_SIDE),
                "samples_per_channel_values": samples_per_channel,
                "total_channel_sample_values": total_samples,
                "records": canonical_sampling_records,
            }
        else:
            self._last_raw_quad_sampling_summary = {
                "mode": "box_or_default",
                "canonical_patch_count": 0,
                "canonical_grid_side": int(SPYDERCHECKR_RAW_CANONICAL_SAMPLE_GRID_SIDE),
                "samples_per_channel_values": [],
                "total_channel_sample_values": [],
                "records": [],
            }

        return np.stack(results, axis=0)  # shape=(n_patches, 3)

    def _save_raw_roi_debug_image(
        self,
        avg_raw_bayer: np.ndarray,
        raw_rois: list,
        save_dir: str,
        raw_roi_quads: "list[list[tuple[float, float]] | None] | None" = None,
    ) -> None:
        """RAW空間上にROI矩形を重畳した可視化画像を保存する。

        display_to_raw_coords が返した実際の RAW 座標で矩形を描画するため、
        表示ROIとのズレがあれば一目瞭然になる。

        Args:
          avg_raw_bayer: dark+flat 補正済み RAW Bayer 配列 (H, W) uint16。
          raw_rois: 各パッチの (rx, ry, rw, rh) リスト。
          save_dir: 保存先ディレクトリ。
        """
        # RAW フレームを 8bit グレースケール化し BGR に変換（矩形描画用）
        raw_f = avg_raw_bayer.astype(np.float32)
        max_v = max(raw_f.max(), 1.0)
        gray_8 = (raw_f / max_v * 255).clip(0, 255).astype(np.uint8)
        vis = cv2.cvtColor(gray_8, cv2.COLOR_GRAY2BGR)

        half = self.n_cols // 2
        for idx, (rx, ry, rw, rh) in enumerate(raw_rois):
            row = idx // self.n_cols
            col = idx % self.n_cols
            # 色分け: 1E (idx=4) = 黄, 左パネル = 緑, 右パネル = シアン
            if idx == 4:
                color = (0, 255, 255)     # 黄 (1E Card White)
            elif col >= half:
                color = (255, 200, 0)     # シアン (右パネル E〜H)
            else:
                color = (0, 200, 0)       # 緑 (左パネル A〜D)
            raw_quad = None
            if raw_roi_quads is not None and idx < len(raw_roi_quads):
                raw_quad = raw_roi_quads[idx]
            if raw_quad:
                pts = np.asarray(
                    [[int(round(x)), int(round(y))] for x, y in raw_quad],
                    dtype=np.int32,
                )
                cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 1)
            label = f"{row + 1}{chr(ord('A') + col + self.col_label_offset)}"
            cv2.putText(
                vis, label, (rx, ry - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1,
            )

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "dbg_raw_roi_overlay.png")
        cv2.imwrite(path, vis)
        print(f"  [DBG] RAW ROI可視化画像を保存: {path}")

    def _display_quad_to_raw_polygon(
        self,
        quad: _PatchwiseDisplayQuad,
        bayer_extractor,
        metadata: dict,
        cfg,
        panel_offset_x: int,
    ) -> list[tuple[float, float]]:
        scaler_crop = metadata.get(
            "ScalerCrop",
            (0, 0, bayer_extractor.sensor_w, bayer_extractor.sensor_h),
        )
        crop_x, crop_y, crop_w, crop_h = scaler_crop
        s2r_x = bayer_extractor.raw_w / bayer_extractor.sensor_w
        s2r_y = bayer_extractor.raw_h / bayer_extractor.sensor_h
        scale_x = crop_w / bayer_extractor.disp_w
        scale_y = crop_h / bayer_extractor.disp_h
        flip_horizontal = bool(cfg.display.flip_horizontal)
        flip_vertical = bool(cfg.display.flip_vertical)
        raw_points: list[tuple[float, float]] = []
        for x, y in quad.points:
            disp_x = float(x) - float(panel_offset_x)
            disp_y = float(y)
            if flip_horizontal:
                disp_x = float(bayer_extractor.disp_w - 1) - disp_x
            if flip_vertical:
                disp_y = float(bayer_extractor.disp_h - 1) - disp_y
            disp_x = self._clamp(disp_x, 0.0, float(bayer_extractor.disp_w - 1))
            disp_y = self._clamp(disp_y, 0.0, float(bayer_extractor.disp_h - 1))
            raw_x = (float(crop_x) + (disp_x * float(scale_x))) * float(s2r_x)
            raw_y = (float(crop_y) + (disp_y * float(scale_y))) * float(s2r_y)
            raw_points.append((float(raw_x), float(raw_y)))
        return raw_points

    @staticmethod
    def _extract_masked_bayer_channel_means(
        roi_uint16: np.ndarray,
        mask: np.ndarray,
        bayer_pattern,
    ) -> np.ndarray:
        slices = bayer_pattern.channel_slices()

        def masked_mean(channel: str) -> float:
            values = roi_uint16[slices[channel]]
            channel_mask = mask[slices[channel]].astype(bool)
            if values.size == 0 or not channel_mask.any():
                return 0.0
            return float(values[channel_mask].mean())

        r = masked_mean("R")
        b = masked_mean("B")
        g_values = [masked_mean("G1"), masked_mean("G2")]
        g_valid = [value for value in g_values if value > 0.0]
        g = float(np.mean(g_valid)) if g_valid else 0.0
        return np.array([r, g, b], dtype=np.float32)

    @staticmethod
    def _canonical_raw_quad_points(
        raw_polygon: list[tuple[float, float]],
        grid_side: int,
    ) -> np.ndarray | None:
        try:
            points = np.asarray(raw_polygon, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            return None
        side = max(1, int(grid_side))
        coords = (np.arange(side, dtype=np.float64) + 0.5) / float(side)
        u_grid, v_grid = np.meshgrid(coords, coords)
        p00, p10, p11, p01 = points
        raw_xy = (
            ((1.0 - u_grid) * (1.0 - v_grid))[..., None] * p00
            + (u_grid * (1.0 - v_grid))[..., None] * p10
            + (u_grid * v_grid)[..., None] * p11
            + ((1.0 - u_grid) * v_grid)[..., None] * p01
        )
        return raw_xy.reshape(-1, 2)

    @staticmethod
    def _snap_raw_index_to_bayer_phase(
        value: float,
        parity: int,
        limit: int,
    ) -> int | None:
        limit_i = int(limit)
        parity_i = int(parity) & 1
        if limit_i <= parity_i:
            return None
        first = parity_i
        last = limit_i - 1
        if (last & 1) != parity_i:
            last -= 1
        if last < first:
            return None
        base = int(round(float(value)))
        candidates = [
            candidate
            for candidate in range(base - 3, base + 4)
            if first <= candidate <= last and (candidate & 1) == parity_i
        ]
        if not candidates:
            clamped = min(max(base, first), last)
            if (clamped & 1) != parity_i:
                clamped += 1 if clamped < first else -1
            return min(max(clamped, first), last)
        return min(candidates, key=lambda candidate: abs(float(candidate) - float(value)))

    def _extract_canonical_raw_quad_bayer_means(
        self,
        avg_raw_bayer: np.ndarray,
        raw_polygon: list[tuple[float, float]],
        bayer_pattern,
    ) -> tuple[np.ndarray, dict[str, object]] | None:
        canonical_points = self._canonical_raw_quad_points(
            raw_polygon,
            SPYDERCHECKR_RAW_CANONICAL_SAMPLE_GRID_SIDE,
        )
        if canonical_points is None or canonical_points.size == 0:
            return None
        height, width = avg_raw_bayer.shape[:2]
        if height <= 1 or width <= 1:
            return None
        slices = bayer_pattern.channel_slices()
        samples: dict[str, list[float]] = {channel: [] for channel in ("R", "G1", "G2", "B")}
        unique_pixels: dict[str, set[tuple[int, int]]] = {channel: set() for channel in samples}
        for raw_x, raw_y in canonical_points:
            for channel in ("R", "G1", "G2", "B"):
                y_slice, x_slice = slices[channel]
                y = self._snap_raw_index_to_bayer_phase(
                    float(raw_y),
                    int(y_slice.start or 0),
                    int(height),
                )
                x = self._snap_raw_index_to_bayer_phase(
                    float(raw_x),
                    int(x_slice.start or 0),
                    int(width),
                )
                if x is None or y is None:
                    return None
                samples[channel].append(float(avg_raw_bayer[y, x]))
                unique_pixels[channel].add((int(x), int(y)))

        sample_counts = {channel: int(len(values)) for channel, values in samples.items()}
        expected_count = int(SPYDERCHECKR_RAW_CANONICAL_SAMPLE_GRID_SIDE) ** 2
        if any(count != expected_count for count in sample_counts.values()):
            return None
        r = float(np.mean(samples["R"]))
        g = float((np.mean(samples["G1"]) + np.mean(samples["G2"])) * 0.5)
        b = float(np.mean(samples["B"]))
        metadata = {
            "mode": "canonical_square_fixed_bayer_phase",
            "canonical_grid_side": int(SPYDERCHECKR_RAW_CANONICAL_SAMPLE_GRID_SIDE),
            "samples_per_channel": int(expected_count),
            "sample_counts_by_channel": sample_counts,
            "total_channel_samples": int(expected_count * 4),
            "unique_pixels_by_channel": {
                channel: int(len(values))
                for channel, values in unique_pixels.items()
            },
            "bayer_pattern": str(getattr(bayer_pattern, "name", bayer_pattern)),
        }
        return np.asarray([r, g, b], dtype=np.float32), metadata

    def _extract_raw_quad_bayer_means(
        self,
        avg_raw_bayer: np.ndarray,
        bayer_extractor,
        metadata: dict,
        cfg,
        quad: _PatchwiseDisplayQuad,
        panel_offset_x: int,
    ) -> tuple[
        np.ndarray,
        tuple[int, int, int, int],
        list[tuple[float, float]],
        dict[str, object],
    ] | None:
        bayer_pattern = getattr(bayer_extractor, "bayer_pattern", None)
        if bayer_pattern is None:
            return None
        raw_polygon = self._display_quad_to_raw_polygon(
            quad,
            bayer_extractor,
            metadata,
            cfg,
            panel_offset_x,
        )
        xs = [point[0] for point in raw_polygon]
        ys = [point[1] for point in raw_polygon]
        left = int(math.floor(min(xs))) & ~1
        top = int(math.floor(min(ys))) & ~1
        right = int(math.ceil(max(xs)))
        bottom = int(math.ceil(max(ys)))
        raw_w = max(2, int(right - left + 1))
        raw_h = max(2, int(bottom - top + 1))
        raw_w = raw_w & ~1
        raw_h = raw_h & ~1
        left = max(0, min(left, bayer_extractor.raw_w - raw_w))
        top = max(0, min(top, bayer_extractor.raw_h - raw_h))
        if raw_w <= 0 or raw_h <= 0:
            return None
        canonical = self._extract_canonical_raw_quad_bayer_means(
            avg_raw_bayer,
            raw_polygon,
            bayer_pattern,
        )
        if canonical is None:
            return None
        means, sampling_meta = canonical
        sampling_meta["raw_bbox"] = {
            "left": int(left),
            "top": int(top),
            "width": int(raw_w),
            "height": int(raw_h),
        }
        return means, (left, top, raw_w, raw_h), raw_polygon, sampling_meta

    def draw_overlay(
        self,
        frame: np.ndarray,
        corners_so_far: list,
        white_local_idx: "int | None" = None,
    ) -> np.ndarray:
        """コーナー指定中またはプレビュー時のオーバーレイを描画する。

        Args:
          frame: 描画対象フレーム (BGR)。
          corners_so_far: 現在指定済みのコーナーリスト（0〜4点）。
          white_local_idx: 白点パッチのローカルインデックス（黄色強調）。
                           左パネルは None、右パネルは 0（= 1E）。
        Returns:
          np.ndarray: オーバーレイ描画済みフレーム。
        """
        out = frame.copy()
        n_corners = len(corners_so_far)

        if not self.is_ready:
            # コーナー指定中: 指定済み点と接続線を描画
            for pt in corners_so_far:
                cv2.circle(out, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)
            for i in range(1, n_corners):
                p1 = (int(corners_so_far[i - 1][0]), int(corners_so_far[i - 1][1]))
                p2 = (int(corners_so_far[i][0]), int(corners_so_far[i][1]))
                cv2.line(out, p1, p2, (0, 255, 0), 2)
        else:
            # コーナー確定後: chart 外を減光し、ROI geometry を優先表示する
            chart_polygons = self._active_panel_frame_hulls_for_overlay()
            base = out.copy()
            dimmed = cv2.addWeighted(
                base,
                0.42,
                np.zeros_like(base),
                0.58,
                0.0,
            )
            if chart_polygons:
                mask = np.zeros(base.shape[:2], dtype=np.uint8)
                for chart_polygon in chart_polygons:
                    cv2.fillConvexPoly(mask, chart_polygon, 255)
                out = dimmed
                out[mask == 255] = base[mask == 255]
                for chart_polygon in chart_polygons:
                    cv2.polylines(out, [chart_polygon], True, (0, 180, 255), 1)
            else:
                out = base

            # コーナー確定後: 全パッチROI geometry を描画
            display_quads = self.get_active_patch_display_quads()
            half = self.n_cols // 2
            idx = 0
            for row in range(self.n_rows):
                for col in range(self.n_cols):
                    quad = display_quads[idx]
                    # 白点パッチ（右パネルの 1E = local index 0）を黄色で強調
                    if idx == white_local_idx:
                        color = (0, 255, 255)
                    elif col >= half:
                        color = (255, 200, 0)
                    else:
                        color = (0, 200, 0)
                    points = np.asarray(
                        [
                            [int(round(x)), int(round(y))]
                            for x, y in quad.points
                        ],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        out,
                        [points],
                        True,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    idx += 1

            # コーナーハンドルを描画（ドラッグ可能であることを視覚的に示す）
            HANDLE_R = 8
            HANDLE_OUTLINE_COLOR = (255, 255, 255)
            HANDLE_CENTER_COLOR = (0, 255, 0)
            for cx, cy in self.corners:
                cx_i, cy_i = int(cx), int(cy)
                cv2.circle(
                    out,
                    (cx_i, cy_i),
                    HANDLE_R,
                    HANDLE_OUTLINE_COLOR,
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    out,
                    (cx_i, cy_i),
                    2,
                    HANDLE_CENTER_COLOR,
                    -1,
                    cv2.LINE_AA,
                )

            # 列アンカーハンドルを描画（ラベルは出さずマーカーのみ）
            COL_ANCHOR_COLOR = (180, 0, 255)  # 紫
            half = self.n_cols // 2
            for col in range(1, self.n_cols - 1):
                gap_offset = self.hinge_gap * 100 if col >= half else 0.0
                cx_n = (col + 0.5) * 100 + gap_offset + self.col_x_norm_offsets[col]
                cy_n = 10.0  # 行0の上端付近
                pt_n = np.float32([[[cx_n, cy_n]]])
                pt_d = cv2.perspectiveTransform(pt_n, self.homography_inv)[0][0]
                ax, ay = int(pt_d[0]), int(pt_d[1])
                s = 5  # 菱形サイズ
                diamond = np.array(
                    [[ax, ay - s], [ax + s, ay], [ax, ay + s], [ax - s, ay]], np.int32
                )
                cv2.polylines(out, [diamond], True, COL_ANCHOR_COLOR, 1)

        return out

    def detect_patch_margin_from_frame(self, frame_bgr: np.ndarray) -> float:
        """ホモグラフィで正規化空間に射影し、黒フレーム幅を測定して patch_margin を自動推定する。

        各パッチセルの中央輝度を基準に「黒フレーム」と「カラー領域」を判別し、
        カラー領域の内縁マージンをメジアンで推定する。
        中央が暗いパッチ（黒・ダークパッチ）は推定から除外する。

        Args:
          frame_bgr: 現在の表示フレーム (BGR)。homography と同じ座標系であること。
        Returns:
          float: 推定 patch_margin。失敗時は現在の self.patch_margin を返す。
        """
        if not self.is_ready:
            return self.patch_margin

        dst_W = int((self.n_cols + self.hinge_gap) * 100)
        dst_H = self.n_rows * 100
        warped = cv2.warpPerspective(frame_bgr, self.homography, (dst_W, dst_H))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)

        margins = []
        half = self.n_cols // 2

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                gap_offset = int(self.hinge_gap * 100) if col >= half else 0
                x0 = col * 100 + gap_offset
                y0 = row * 100
                cell = gray[y0: y0 + 100, x0: x0 + 100]
                if cell.shape[0] < 20 or cell.shape[1] < 20:
                    continue

                # セル中央 50% の平均輝度をパッチの代表値とする
                cy0, cy1 = cell.shape[0] // 4, 3 * cell.shape[0] // 4
                cx0, cx1 = cell.shape[1] // 4, 3 * cell.shape[1] // 4
                center_val = float(cell[cy0:cy1, cx0:cx1].mean())

                # 黒・ダークパッチ（中央が暗すぎる）はスキップ
                if center_val < 35.0:
                    continue

                # 黒フレームとカラー領域の境界閾値 = 中央輝度の 45%
                threshold = center_val * 0.45

                # 水平方向: 行平均プロファイルで左右マージンを検出
                h_profile = cell.mean(axis=0)
                h_bright = h_profile > threshold
                if h_bright.sum() > 5:
                    left_m = float(np.argmax(h_bright)) / cell.shape[1]
                    right_m = float(cell.shape[1] - 1 - np.argmax(h_bright[::-1])) / cell.shape[1]
                    margins.extend([left_m, right_m])

                # 垂直方向: 列平均プロファイルで上下マージンを検出
                v_profile = cell.mean(axis=1)
                v_bright = v_profile > threshold
                if v_bright.sum() > 5:
                    top_m = float(np.argmax(v_bright)) / cell.shape[0]
                    bot_m = float(cell.shape[0] - 1 - np.argmax(v_bright[::-1])) / cell.shape[0]
                    margins.extend([top_m, bot_m])

        if not margins:
            return self.patch_margin

        # メジアンでロバスト推定、妥当範囲 0.02〜0.25 にクランプ
        estimated = float(np.median(margins))
        return float(np.clip(estimated, 0.02, 0.25))


# ---------------------------------------------------------------------------
# I-1: CCMVerifier クラス
# ---------------------------------------------------------------------------


class CCMVerifier:
    """4D パッチを用いた CCM の日次簡易検証クラス。"""

    # 4D パッチの既知 Lab 値
    REFERENCE_LAB_4D = np.array([57.15, 0.57, 1.19], dtype=np.float64)

    # 判定閾値
    THRESH_OK = 1.0  # ΔE < 1.0 → OK
    THRESH_WARN = 3.0  # 1.0 ≤ ΔE < 3.0 → WARN

    def __init__(self):
        self.last_result: dict | None = None

    def verify(self, measured_Lab: np.ndarray) -> dict:
        """測定 Lab と既知 4D Lab の ΔE を算出し、判定結果を返す。

        Args:
          measured_Lab: (3,) 測定された Lab 値。
        Returns:
          dict: {"dE": float, "status": "OK"|"WARN"|"FAIL", "measured": [...], "reference": [...]}
        """
        measured = np.asarray(measured_Lab, dtype=np.float64).ravel()[:3]
        diff = measured - self.REFERENCE_LAB_4D
        dE = float(np.sqrt(np.sum(diff**2)))
        if dE < self.THRESH_OK:
            status = "OK"
        elif dE < self.THRESH_WARN:
            status = "WARN"
        else:
            status = "FAIL"
        result = {
            "dE": dE,
            "status": status,
            "measured": measured.tolist(),
            "reference": self.REFERENCE_LAB_4D.tolist(),
        }
        self.last_result = result
        return result

    def status_text(self) -> str:
        """最後の検証結果を1行テキストで返す。未実行時は空文字列。"""
        if self.last_result is None:
            return ""
        r = self.last_result
        if r.get("is_stale"):
            last_status = r.get("status", "--")
            return f"4D絶対: STALE (last={last_status}, ΔE={r['dE']:.2f})"
        return f"4D絶対: {r['status']} (ΔE={r['dE']:.2f})"


class RelativeGrayVerifier:
    """4D グレーパッチの LinearRGB 相対値でキャリブ品質を判定するクラス。

    18% グレーカード基準で 4D gray50% の R,G,B が 1.000 に
    どれだけ近いかを主指標とする。Lab a,b はゼロ必須としない。
    """

    THRESH_PASS = 0.01    # |ch - 1.0| < 0.01 → 合格
    THRESH_GOOD = 0.005   # |ch - 1.0| < 0.005 → 良好
    THRESH_SIGMA = 0.003  # σ < 0.003 → 安定性合格

    def __init__(self):
        self.last_result: dict | None = None

    def evaluate(self, ratios: np.ndarray) -> dict:
        """LinearRGB 比率配列から精度・安定性を判定する。

        Args:
          ratios: (N, 3) の LinearRGB 比率配列（blank 補正済み、期待値 ~1.0）。
        Returns:
          判定結果の dict。
        """
        ratios = np.asarray(ratios, dtype=np.float64)
        mean_rgb = ratios.mean(axis=0)
        std_rgb = ratios.std(axis=0, ddof=1) if len(ratios) > 1 else np.zeros(3)
        max_dev = float(np.max(np.abs(mean_rgb - 1.0)))
        max_sigma = float(np.max(std_rgb))
        if max_dev < self.THRESH_GOOD:
            accuracy_status = "GOOD"
        elif max_dev < self.THRESH_PASS:
            accuracy_status = "PASS"
        else:
            accuracy_status = "FAIL"
        stability_status = "PASS" if max_sigma < self.THRESH_SIGMA else "FAIL"
        result = {
            "mean_rgb": mean_rgb.tolist(),
            "std_rgb": std_rgb.tolist(),
            "max_dev": max_dev,
            "max_sigma": max_sigma,
            "accuracy_status": accuracy_status,
            "stability_status": stability_status,
            "n_samples": len(ratios),
        }
        self.last_result = result
        return result

    def status_text(self) -> str:
        """最後の検証結果を1行テキストで返す。未実行時は空文字列。"""
        if self.last_result is None:
            return ""
        r = self.last_result
        if r.get("is_stale"):
            return (
                f"4D相対: STALE "
                f"(last={r['accuracy_status']}, dev={r['max_dev']:.4f}, "
                f"s={r['max_sigma']:.4f}, n={r['n_samples']})"
            )
        return (
            f"4D相対: {r['accuracy_status']} "
            f"(dev={r['max_dev']:.4f}, s={r['max_sigma']:.4f}, n={r['n_samples']})"
        )


class CIEDE2000Calculator:
    """
    CIEDE2000 色差を計算するクラス。

    機能:
      - 2つのLab値間の視覚的色差 dE00 をSharma式で算出する。
    入力:
      - 比較対象のLab配列2点と重み係数 `kL/kC/kH`。
    出力:
      - スカラー色差値 `dE00`（float）。
    """

    @staticmethod
    def compute(
        Lab1: np.ndarray,
        Lab2: np.ndarray,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0,
    ) -> float:
        """
        2点のLabからCIEDE2000色差を計算する。

        Args:
          Lab1, Lab2: Lab配列。
          kL, kC, kH: 補正係数。
        Returns:
          float: dE00値。
        """
        L1, a1, b1 = float(Lab1[0]), float(Lab1[1]), float(Lab1[2])
        L2, a2, b2 = float(Lab2[0]), float(Lab2[1]), float(Lab2[2])

        # Step 1: C_ab (元のクロマ)
        C1 = math.sqrt(a1**2 + b1**2)
        C2 = math.sqrt(a2**2 + b2**2)
        C_ab_mean = (C1 + C2) / 2.0

        # Step 2: G因子
        C_ab_mean_7 = C_ab_mean**7
        G = 0.5 * (1.0 - math.sqrt(C_ab_mean_7 / (C_ab_mean_7 + 25.0**7)))

        # Step 3: a' (補正済みa)
        a1p = a1 * (1.0 + G)
        a2p = a2 * (1.0 + G)

        # Step 4: C' (補正済みクロマ)
        C1p = math.sqrt(a1p**2 + b1**2)
        C2p = math.sqrt(a2p**2 + b2**2)

        # Step 5: h' (色相角, degrees, 0-360)
        h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
        h2p = math.degrees(math.atan2(b2, a2p)) % 360.0

        # Step 6: delta L', delta C'
        dLp = L2 - L1
        dCp = C2p - C1p

        # Step 7: delta h' (ラップアラウンド処理)
        if C1p * C2p == 0.0:
            dhp = 0.0
        elif abs(h2p - h1p) <= 180.0:
            dhp = h2p - h1p
        elif h2p - h1p > 180.0:
            dhp = h2p - h1p - 360.0
        else:
            dhp = h2p - h1p + 360.0

        # Step 8: delta H'
        dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

        # Step 9: 平均値
        Lp_mean = (L1 + L2) / 2.0
        Cp_mean = (C1p + C2p) / 2.0

        # hp_mean (色相平均のラップアラウンド処理)
        if C1p * C2p == 0.0:
            hp_mean = h1p + h2p
        elif abs(h1p - h2p) <= 180.0:
            hp_mean = (h1p + h2p) / 2.0
        elif h1p + h2p < 360.0:
            hp_mean = (h1p + h2p + 360.0) / 2.0
        else:
            hp_mean = (h1p + h2p - 360.0) / 2.0

        # Step 10: T (色相依存の重み係数)
        T = (
            1.0
            - 0.17 * math.cos(math.radians(hp_mean - 30.0))
            + 0.24 * math.cos(math.radians(2.0 * hp_mean))
            + 0.32 * math.cos(math.radians(3.0 * hp_mean + 6.0))
            - 0.20 * math.cos(math.radians(4.0 * hp_mean - 63.0))
        )

        # Step 11: SL, SC, SH (重み関数)
        SL = 1.0 + 0.015 * (Lp_mean - 50.0) ** 2 / math.sqrt(
            20.0 + (Lp_mean - 50.0) ** 2
        )
        SC = 1.0 + 0.045 * Cp_mean
        SH = 1.0 + 0.015 * Cp_mean * T

        # Step 12: RT (回転項 — 青領域補正)
        Cp_mean_7 = Cp_mean**7
        RC = 2.0 * math.sqrt(Cp_mean_7 / (Cp_mean_7 + 25.0**7))
        d_theta = 30.0 * math.exp(-(((hp_mean - 275.0) / 25.0) ** 2))
        RT = -math.sin(math.radians(2.0 * d_theta)) * RC

        # Step 13: 最終 deltaE00
        dE00 = math.sqrt(
            (dLp / (kL * SL)) ** 2
            + (dCp / (kC * SC)) ** 2
            + (dHp / (kH * SH)) ** 2
            + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
        )

        return dE00


# ===========================================================================
# 計測・安定性
# ===========================================================================


class StabilityMonitor:
    """
    リングバッファによる時間積算と安定性評価。
    ratioバッファ: 時間平均ratio算出用
    Labバッファ:  安定性指標（CV_L, σ_a, σ_b）算出用

    機能:
      - ratio/Labの時系列を保持し、安定状態（UNSTABLE/SETTLING/READY）を判定する。
    入力:
      - ratio配列、Lab配列、評価閾値（CV/σ）、バッファサイズ設定。
    出力:
      - 平均ratio、標準偏差、CV_L、σ_a、σ_b、安定状態コードを返す。
    """

    UNSTABLE = 0
    SETTLING = 1
    READY = 2

    def __init__(
        self,
        buffer_size: int = 60,
        cv_threshold: float = 0.001,
        sigma_threshold: float = 0.05,
        jump_threshold: float = 0.15,
        jump_confirm_frames: int = 3,
        jump_center: str = "median",
    ):
        """
        安定性評価のバッファと閾値を初期化する。

        Args:
          buffer_size: リングバッファ長。
          cv_threshold: CVの閾値。
          sigma_threshold: σの閾値。
          jump_threshold: 急変検出の相対乖離閾値（例: 0.15 = 15%）。
          jump_confirm_frames: 急変確定に必要な連続フレーム数。
          jump_center: 急変検出の中心値。"median" または "mean"。
        """
        if jump_center not in {"median", "mean"}:
            raise ValueError("jump_center must be 'median' or 'mean'")
        self.buffer_size = buffer_size
        self.cv_threshold = cv_threshold
        self.sigma_threshold = sigma_threshold
        self.jump_threshold = jump_threshold
        self.jump_confirm_frames = jump_confirm_frames
        self.jump_center = jump_center
        self._jump_counter = 0

        # ratio リングバッファ
        self.buffer = np.zeros((buffer_size, 3), dtype=np.float64)
        self.index = 0
        self.count = 0

        # Lab リングバッファ
        self.lab_buffer = np.zeros((buffer_size, 3), dtype=np.float64)
        self.lab_index = 0
        self.lab_count = 0

    def clear(self) -> None:
        """急変検出時にratio/Labバッファを全クリアする。"""
        self.buffer[:] = 0
        self.index = 0
        self.count = 0
        self.lab_buffer[:] = 0
        self.lab_index = 0
        self.lab_count = 0
        self._jump_counter = 0

    def push(self, ratio: np.ndarray) -> None:
        """ratioバッファに新しい値を追加（急変検出付き）"""
        # 急変検出: バッファ中心値との乖離が閾値超過を連続したらクリア
        if self.count >= 1:
            current_center = self._get_jump_center_ratio()
            diff = np.abs(ratio - current_center).max()
            relative_diff = diff / max(current_center.max(), 1e-6)
            if relative_diff > self.jump_threshold:
                self._jump_counter += 1
            else:
                self._jump_counter = 0
            if self._jump_counter >= self.jump_confirm_frames:
                self.clear()
        self.buffer[self.index % self.buffer_size] = ratio
        self.index += 1
        self.count = min(self.count + 1, self.buffer_size)

    def _get_jump_center_ratio(self) -> np.ndarray:
        """急変検出に使うratio中心値を返す。"""
        valid = self.buffer[: self.count]
        if self.jump_center == "mean":
            return valid.mean(axis=0)
        return np.median(valid, axis=0)

    def get_mean_ratio(self) -> np.ndarray:
        """バッファの有効範囲の平均ratioを返す"""
        if self.count == 0:
            return np.zeros(3, dtype=np.float64)
        return self.buffer[: self.count].mean(axis=0)

    def get_median_ratio(self) -> np.ndarray:
        """バッファの有効範囲の中央値ratioを返す"""
        if self.count == 0:
            return np.zeros(3, dtype=np.float64)
        return np.median(self.buffer[: self.count], axis=0)

    def get_std_ratio(self) -> np.ndarray:
        """バッファの有効範囲の標準偏差を返す"""
        if self.count < 2:
            return np.full(3, float("inf"), dtype=np.float64)
        return self.buffer[: self.count].std(axis=0, ddof=1)

    def push_lab(self, Lab: np.ndarray) -> None:
        """Labバッファに新しい値を追加（循環）"""
        self.lab_buffer[self.lab_index % self.buffer_size] = Lab
        self.lab_index += 1
        self.lab_count = min(self.lab_count + 1, self.buffer_size)

    def get_cv_L(self) -> float:
        """L*のCV値（変動係数 = σ/μ）を返す"""
        if self.lab_count < 2:
            return float("inf")
        valid = self.lab_buffer[: self.lab_count, 0]
        mean_L = valid.mean()
        if abs(mean_L) < 1e-10:
            return float("inf")
        return float(valid.std(ddof=1) / mean_L)

    def get_sigma_a(self) -> float:
        """a*の絶対標準偏差を返す"""
        if self.lab_count < 2:
            return float("inf")
        return float(self.lab_buffer[: self.lab_count, 1].std(ddof=1))

    def get_sigma_b(self) -> float:
        """b*の絶対標準偏差を返す"""
        if self.lab_count < 2:
            return float("inf")
        return float(self.lab_buffer[: self.lab_count, 2].std(ddof=1))

    def get_stability_state(self) -> int:
        """安定性状態を返す: READY, SETTLING, or UNSTABLE"""
        cv_L = self.get_cv_L()
        sigma_a = self.get_sigma_a()
        sigma_b = self.get_sigma_b()

        if (
            cv_L < self.cv_threshold
            and sigma_a < self.sigma_threshold
            and sigma_b < self.sigma_threshold
        ):
            return self.READY
        elif self.is_buffer_full():
            return self.SETTLING
        else:
            return self.UNSTABLE

    def is_buffer_full(self) -> bool:
        """ratioバッファが満杯かどうかを返す。"""
        return self.count >= self.buffer_size


class MeasurementLogger:
    """
    CSV記録によるエビデンス確保を行うロガークラス。

    機能:
      - 定期ログとイベントログをCSVへ書き出し、計測トレースを保存する。
    入力:
      - 出力ディレクトリ、記録間隔、計測データ辞書、イベント種別。
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
        "quality_warnings",
    ]

    def __init__(self, log_dir: str, interval_frames: int = 30):
        """
        ログ出力の設定を初期化する。

        Args:
          log_dir: 出力ディレクトリ。
          interval_frames: 記録間隔（フレーム数）。
        """
        self.log_dir = log_dir
        self.interval_frames = interval_frames
        self.frame_count = 0
        self.csv_file = None
        self.csv_writer = None

    def _ensure_file_open(self) -> None:
        """初回呼び出し時にCSVファイルを開きヘッダーを書き込む"""
        if self.csv_file is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        csv_path = os.path.join(self.log_dir, f"{ts}_measurement.csv")
        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self._HEADER)
        print(f"- CSV log started: {csv_path}")

    def _write_row(self, data: dict, event_type: str) -> None:
        """1行のデータをCSVに書き込む"""
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
            data.get("quality_warnings", ""),
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def log_interval(self, data: dict) -> None:
        """インターバル記録（event_type='interval'）"""
        self._write_row(data, "interval")

    def log_event(self, data: dict, event_type: str = "measure") -> None:
        """イベント記録（event_type='measure'）"""
        self._write_row(data, event_type)

    def tick(self, data: dict) -> None:
        """毎フレーム呼び出し。interval_framesごとにインターバル記録を行う"""
        self.frame_count += 1
        if self.frame_count % self.interval_frames == 0:
            self.log_interval(data)

    def close(self) -> None:
        """CSVファイルをflush+closeする"""
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None


# ===========================================================================
# フレーム処理・UI
# ===========================================================================


class FrameTransformer:
    """
    反転とクロップなどの幾何変換を行うクラス。

    機能:
      - 表示設定に基づく反転処理と中央クロップ処理を提供する。
    入力:
      - 入力フレーム配列と表示設定（反転フラグ、出力サイズ）。
    出力:
      - 表示用に整形された変換後フレーム配列を返す。
    """

    def __init__(self, config):
        """表示設定を保持する。"""
        self.config = config

    @staticmethod
    def crop_center(frame, crop_width, crop_height):
        """
        中央クロップを行う。

        Args:
          frame: 入力フレーム。
          crop_width, crop_height: クロップサイズ。
        Returns:
          np.ndarray: クロップ済みフレーム。
        """
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        cropped_frame = frame[
            center_y - crop_height // 2 : center_y + crop_height // 2,
            center_x - crop_width // 2 : center_x + crop_width // 2,
        ]
        return cropped_frame

    @staticmethod
    def flip_image(frame, flip_horizontal=False, flip_vertical=False):
        """
        水平/垂直反転を適用する。

        Args:
          frame: 入力フレーム。
          flip_horizontal: 水平反転の有無。
          flip_vertical: 垂直反転の有無。
        Returns:
          np.ndarray: 反転後のフレーム。
        """
        if flip_horizontal:
            frame = cv2.flip(frame, 1)  # 水平反転
        if flip_vertical:
            frame = cv2.flip(frame, 0)  # 垂直反転
        return frame

    def transform(self, frame):
        """
        設定に従い反転と中央クロップを適用する。

        Args:
          frame: 入力フレーム。
        Returns:
          np.ndarray: 変換後フレーム。
        """
        frame = self.flip_image(
            frame,
            self.config.display.flip_horizontal,
            self.config.display.flip_vertical,
        )
        frame = self.crop_center(
            frame, self.config.display.width, self.config.display.height
        )
        return frame


class LiveColorCorrector:
    """
    表示用フレームに対するライブ色補正を行うクラス。

    機能:
      - ROI抽出、Lab平均算出、グレーカード基準のLab補正を提供する。
    入力:
      - BGRフレーム、ROI座標/サイズ、Lab参照値、補正係数。
    出力:
      - 補正後BGRフレーム、輝度スケール、ROI由来のLab統計値を返す。
    """

    @staticmethod
    def get_ROI(frame, spot_x, spot_y, spot_w, spot_h=None):
        """指定位置・サイズでROIを取得する。"""
        if frame is None or frame.size == 0:
            return None
        if spot_h is None:
            spot_h = spot_w
        x0 = max(int(spot_x), 0)
        y0 = max(int(spot_y), 0)
        w = max(int(spot_w), 1)
        h = max(int(spot_h), 1)
        x1 = min(x0 + w, frame.shape[1])
        y1 = min(y0 + h, frame.shape[0])
        if x0 >= x1 or y0 >= y1:
            return None
        return frame[y0:y1, x0:x1]

    @staticmethod
    def get_Lab_mean(roi_bgr):
        """BGR ROIをLabへ変換し平均値を返す。"""
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab)
        lab_val = cv2.mean(roi_lab)
        return [x for x in lab_val]

    @staticmethod
    def bgr_color_correct_Lab(image_bgr, lab_ref, adj_factor=1.00):
        """18%グレーカード基準で表示フレームをLab空間補正する。"""
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(lab_image)

        bright_scale = (128 / lab_ref[0] if lab_ref[0] != 0 else 1) * adj_factor
        L_corr = L * bright_scale
        L_corr = np.clip(L_corr, 0, 255).astype(np.uint8)

        a_corr = cv2.addWeighted(
            a.astype(np.float32),
            1.0,
            np.ones(a.shape, dtype=np.float32) * float(lab_ref[1]),
            -1.0,
            128,
        ).astype(np.uint8)
        b_corr = cv2.addWeighted(
            b.astype(np.float32),
            1.0,
            np.ones(b.shape, dtype=np.float32) * float(lab_ref[2]),
            -1.0,
            128,
        ).astype(np.uint8)

        corr_image_bgr = cv2.merge((L_corr, a_corr, b_corr))
        corr_image_bgr = cv2.cvtColor(corr_image_bgr, cv2.COLOR_Lab2BGR)
        return corr_image_bgr, bright_scale


class WindowManager:
    """
    表示ウィンドウの作成・表示・破棄を担当するクラス。

    機能:
      - OpenCVウィンドウの初期化、フレーム表示、終了処理を一元管理する。
    入力:
      - ウィンドウ名、表示設定、表示対象フレーム。
    出力:
      - 画面表示状態の更新とウィンドウリソースの解放処理を実行する。
    """

    def __init__(self, window_name: str, config):
        """ウィンドウ名と表示設定を保持する。"""
        self.window_name = window_name
        self.config = config
        self._screen_size = self._detect_screen_size()
        self._canvas_offset = (
            0,
            0,
        )  # スケール後フレームのキャンバス内オフセット (x, y)
        self._canvas_scale = 1.0  # フレームのスケール係数
        self._initial_resize_done = False

    @staticmethod
    def _detect_screen_size() -> tuple:
        """スクリーンサイズを (width, height) で返す。取得失敗時は (0, 0)。"""
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            pass
        return (0, 0)

    def _request_wmctrl_fullscreen(self):
        """wmctrl / xdotool で X11 タイトルバーを除去しフルスクリーンにする（別スレッドで実行）。"""
        time.sleep(0.6)  # ウィンドウが表示されるまで待つ
        for cmd in (
            ["wmctrl", "-r", self.window_name, "-b", "add,fullscreen"],
            ["xdotool", "search", "--name", self.window_name, "windowfullscreen"],
        ):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                if r.returncode == 0:
                    return
            except FileNotFoundError:
                continue
        print(
            "[WindowManager] wmctrl/xdotool not found.\n"
            "  Install: sudo apt install wmctrl"
        )

    def make_offset_callback(self, callback):
        """
        マウスコールバックをラップし、キャンバス座標をフレーム座標へ逆変換して渡す。

        show() でフレームをスケールしてキャンバスに配置した場合、
        マウス座標をスケール・オフセット補正してからフレーム基準の座標に変換する。

        Args:
          callback: 元のマウスコールバック関数。
        Returns:
          callable: 補正済みラッパー関数。
        """

        def _wrapped(event, x, y, flags, param):
            ox, oy = self._canvas_offset
            s = self._canvas_scale if self._canvas_scale > 0 else 1.0
            callback(event, int((x - ox) / s), int((y - oy) / s), flags, param)

        return _wrapped

    def setup(self):
        """ウィンドウの生成と表示モード設定を行う。"""
        _configure_qt_fontdir()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, frame):
        """
        フレームをスクリーンサイズにスケールしてウィンドウ全体に表示する。

        スクリーンサイズが取得できた場合:
          - フレームをアスペクト比を保ちながらスクリーンサイズに合わせて拡大縮小
          - 余白を黒で埋めたキャンバスを imshow
          - 初回のみ resizeWindow + moveWindow + wmctrl（タイトルバー除去）

        Args:
          frame: 表示フレーム (BGR)。
        """
        sw, sh = self._screen_size
        if sw > 0:
            fh, fw = frame.shape[:2]
            scale = min(sw / fw, sh / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            scaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            ox = (sw - nw) // 2
            oy = (sh - nh) // 2
            canvas[oy : oy + nh, ox : ox + nw] = scaled
            self._canvas_offset = (ox, oy)
            self._canvas_scale = scale
            cv2.imshow(self.window_name, canvas)
        else:
            self._canvas_offset = (0, 0)
            self._canvas_scale = 1.0
            cv2.imshow(self.window_name, frame)

        if not self._initial_resize_done:
            cv2.waitKey(1)
            if sw > 0:
                cv2.resizeWindow(self.window_name, sw, sh)
                cv2.moveWindow(self.window_name, 0, 0)
                threading.Thread(
                    target=self._request_wmctrl_fullscreen, daemon=True
                ).start()
            self._initial_resize_done = True

    def destroy(self):
        """OpenCVのウィンドウを破棄する。"""
        cv2.destroyAllWindows()


# ===========================================================================
# 補正
# ===========================================================================


class DarkFrameManager:
    """
    ダークフレームの取得・保存・適用を行うクラス。

    役割:
      - ダークフレームの読み込み/保存
      - ROI単位の減算補正

    機能:
      - ダークフレームの取得・永続化・ROI減算補正を実行する。
    入力:
      - カメラから取得したRAWフレーム、ROI座標、保存先パス、クリップ上限。
    出力:
      - 読み込み成否、抽出ダークROI、減算補正後ROI配列を返す。
    """

    def __init__(self, save_path: str = "", max_val: int = 4095):
        """ダークフレームの保存パスと状態を初期化する。

        Args:
          save_path: ダークフレームの保存パス。空文字列ならデフォルト。
          max_val: クリップ上限値（CSI 12bit: 4095, USB 8bit: 255）。
        """
        self.save_path = save_path or os.path.join(
            get_today_calibration_dir(), "dark_frame.npy"
        )
        self.max_val = max_val
        self.dark_frame = None
        self.is_loaded = False

    def load_if_exists(self) -> bool:
        """
        既存のダークフレームを読み込む。最新の日付フォルダを優先して検索する。

        Returns:
          bool: 読み込み成功時True。
        """
        path = _find_calibration_file("dark_frame.npy")
        if path is None:
            return False
        try:
            loaded = np.load(path)
            self.dark_frame = loaded
            self.is_loaded = True
            print(f"- Dark frame loaded: {path} shape={loaded.shape}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load dark frame: {e}")
            return False

    def get_dark_roi(self, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        """
        ダークフレームから指定ROIを取得する。

        Args:
          x, y: ROI左上座標。
          w, h: ROI幅・高さ。
        Returns:
          Optional[np.ndarray]: ROI配列。未ロード時はNone。
        """
        if self.dark_frame is None:
            return None
        y_end = min(y + h, self.dark_frame.shape[0])
        x_end = min(x + w, self.dark_frame.shape[1])
        return self.dark_frame[y:y_end, x:x_end]

    def subtract(self, raw_roi: np.ndarray, dark_roi: np.ndarray) -> np.ndarray:
        """
        RAW ROIからダークROIを減算してクリップする。

        Args:
          raw_roi: RAW ROI配列。
          dark_roi: ダークROI配列。
        Returns:
          np.ndarray: 減算後のROI配列。
        """
        return np.clip(
            raw_roi.astype(np.int32) - dark_roi.astype(np.int32), 0, self.max_val
        ).astype(raw_roi.dtype)

    def capture_dark_frame(self, source, bayer_extractor=None, n_frames: int = 64):
        """ダークフレームを複数枚平均して保存する。

        CSI版: capture_dark_frame(picam2, bayer_extractor, n_frames)
          - picam2.capture_array("raw") → bayer_extractor.parse_raw() で RAW Bayer を取得。
          - uint16 で保存。
        USB版: capture_dark_frame(cap, n_frames=64)
          - cap.read() で BGR フレームを取得。
          - uint8 で保存。

        Args:
          source: Picamera2 インスタンス(CSI) または cv2.VideoCapture(USB)。
          bayer_extractor: BayerROIExtractor (CSI版のみ)。USB版では省略。
          n_frames: 取得フレーム数。
        """
        print(f"Capturing {n_frames} dark frames...")
        accumulator = None

        if bayer_extractor is not None:
            # --- CSI版: RAW Bayer ---
            for i in range(n_frames):
                raw_array = source.capture_array("raw")
                raw_bayer = bayer_extractor.parse_raw(raw_array)
                if accumulator is None:
                    accumulator = raw_bayer.astype(np.float64)
                else:
                    accumulator += raw_bayer.astype(np.float64)
            if accumulator is not None:
                dark_avg = (accumulator / n_frames).astype(np.uint16)
                np.save(self.save_path, dark_avg)
                self.dark_frame = dark_avg
                self.is_loaded = True
                _remove_cleared_marker("dark_frame.npy")
                print(f"- Dark frame saved: {self.save_path} shape={dark_avg.shape}")
        else:
            # --- USB版: BGR フレーム ---
            for _ in range(n_frames):
                ret, frame = source.read()
                if not ret:
                    continue
                if accumulator is None:
                    accumulator = frame.astype(np.float64)
                else:
                    accumulator += frame.astype(np.float64)
            if accumulator is not None:
                dark_avg = (accumulator / n_frames).astype(np.uint8)
                np.save(self.save_path, dark_avg)
                self.dark_frame = dark_avg
                self.is_loaded = True
                _remove_cleared_marker("dark_frame.npy")
                print(f"- Dark frame saved: {self.save_path} shape={dark_avg.shape}")
            else:
                print("Warning: No frames captured for dark frame.")


class WhiteBalanceCalibrator:
    """
    ホワイトバランスゲインの算出・保存・復元を行うクラス。

    役割:
      - 白色面のRAW Bayerまたは BGR 画像から Red/Blue ゲインを算出する。
      - 暗室上部の穴など外れ値に対してロバストな median ベースの算出を行う。
      - 算出結果を wb_gains.json に保存し、次回起動時に自動復元する。

    機能:
      - calibrate_from_raw_bayer: CSI版（12bit RAW Bayer）用のゲイン算出。
      - calibrate_from_bgr: USB版（8bit BGR）用のゲイン算出。
      - save / load_if_exists: ゲインの永続化と復元。
    入力:
      - RAW Bayer 2D配列 または BGR 3D配列。
    出力:
      - (red_gain, blue_gain) タプル。
    """

    def __init__(self, save_path: str = ""):
        """ホワイトバランスゲインの保存パスと状態を初期化する。

        Args:
          save_path: ゲイン保存先JSONファイルパス。空文字列ならデフォルト。
        """
        self.save_path = save_path or os.path.join(
            get_today_calibration_dir(), "wb_gains.json"
        )
        self.red_gain = 1.0
        self.blue_gain = 1.0
        self.is_calibrated = False

    def save(self) -> None:
        """算出済みゲインを wb_gains.json に書き出す。"""
        import json

        data = {"red_gain": float(self.red_gain), "blue_gain": float(self.blue_gain)}
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        _remove_cleared_marker("wb_gains.json")
        print(f"- WB gains saved: {self.save_path}")

    def load_if_exists(self) -> bool:
        """保存済みゲインファイルが存在すれば読み込む。最新の日付フォルダを優先して検索する。

        Returns:
          bool: 読み込み成功時 True。
        """
        import json

        path = _find_calibration_file("wb_gains.json")
        if path is None:
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.red_gain = float(data["red_gain"])
            self.blue_gain = float(data["blue_gain"])
            self.is_calibrated = True
            print(f"- WB gains loaded: R={self.red_gain:.4f} B={self.blue_gain:.4f}")
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: WB gains file broken: {e}")
            return False

    def set_gains(self, red_gain: float, blue_gain: float) -> None:
        """外部で確定したゲインを設定する。"""
        self.red_gain = float(red_gain)
        self.blue_gain = float(blue_gain)
        self.is_calibrated = True
        self.save()

    def get_gains(self) -> tuple:
        """現在のゲインをタプルで返す。

        Returns:
          tuple: (red_gain, blue_gain)。
        """
        return (self.red_gain, self.blue_gain)

    def calibrate_from_raw_bayer(
        self,
        raw_bayer: np.ndarray,
        bayer_pattern: "BayerPattern" = None,
    ) -> tuple:
        """RAW Bayer 画像から median ベースで WB ゲインを算出する。

        Bayer パターンに応じて R/G1/G2/B を分離し、
        G チャンネルを基準として Red, Blue のゲインを算出する。

        Args:
          raw_bayer: 2D RAW Bayer 配列 (H, W)。
          bayer_pattern: Bayer パターン。None なら BGGR を仮定。
        Returns:
          tuple: (red_gain, blue_gain)。
        """
        if bayer_pattern is None:
            bayer_pattern = BayerPattern.BGGR
        sl = bayer_pattern.channel_slices()
        B = raw_bayer[sl["B"]]
        G1 = raw_bayer[sl["G1"]]
        G2 = raw_bayer[sl["G2"]]
        R = raw_bayer[sl["R"]]

        # median ベースで代表値算出（外れ値にロバスト）
        G_combined = (G1.astype(np.float64) + G2.astype(np.float64)) / 2.0
        G_median = np.median(G_combined)
        R_median = np.median(R.astype(np.float64))
        B_median = np.median(B.astype(np.float64))

        # ゼロ除算防止
        if R_median < 1.0:
            R_median = 1.0
        if B_median < 1.0:
            B_median = 1.0

        red_gain = G_median / R_median
        blue_gain = G_median / B_median

        self.red_gain = red_gain
        self.blue_gain = blue_gain
        self.is_calibrated = True
        self.save()

        print(
            f"- WB calibrated (RAW): R_gain={red_gain:.4f} B_gain={blue_gain:.4f}"
            f"  (R_med={R_median:.1f} G_med={G_median:.1f} B_med={B_median:.1f})"
        )
        return (red_gain, blue_gain)

    def calibrate_from_bgr(self, frame_bgr: np.ndarray) -> tuple:
        """BGR 画像から median ベースで WB ゲインを算出する（USB版用）。

        G チャンネルを基準として Red, Blue のゲインを算出する。

        Args:
          frame_bgr: BGR 画像配列 (H, W, 3), uint8。
        Returns:
          tuple: (red_gain, blue_gain)。
        """
        B = frame_bgr[:, :, 0].astype(np.float64)
        G = frame_bgr[:, :, 1].astype(np.float64)
        R = frame_bgr[:, :, 2].astype(np.float64)

        G_median = np.median(G)
        R_median = np.median(R)
        B_median = np.median(B)

        # ゼロ除算防止
        if R_median < 1.0:
            R_median = 1.0
        if B_median < 1.0:
            B_median = 1.0

        red_gain = G_median / R_median
        blue_gain = G_median / B_median

        self.red_gain = red_gain
        self.blue_gain = blue_gain
        self.is_calibrated = True
        self.save()

        print(
            f"- WB calibrated (BGR): R_gain={red_gain:.4f} B_gain={blue_gain:.4f}"
            f"  (R_med={R_median:.1f} G_med={G_median:.1f} B_med={B_median:.1f})"
        )
        return (red_gain, blue_gain)


class SessionRecorder:
    """
    CSV記録セッションを管理するクラス。

    役割:
      - m キーで記録開始、s キーで記録停止のワークフローを管理する。
      - 記録中は毎フレーム CSV に1行追記する。
      - 停止時にファイルを読み直し、ヘッダー部に中央値を埋め込む。

    CSVフォーマット:
      - ヘッダー 9行（# プレフィックス） + カラムヘッダー 1行 = 固定10行。
      - データは11行目から開始。
    入力:
      - 計測データ辞書（Lab or LinearRGB）。
    出力:
      - ヘッダー付きCSVファイル。
    """

    HEADER_LINES = 9  # # プレフィックス行数
    DATA_START_LINE = 11  # 1-indexed

    def __init__(self, log_dir: str):
        """記録セッションの状態を初期化する。

        Args:
          log_dir: CSVファイルの出力先ディレクトリ。
        """
        self.log_dir = log_dir
        self.is_recording = False
        self.file_name = None
        self.csv_path = None
        self.mode = None
        self.camera_name = None
        self.start_time = None
        self.sample_count = 0
        self._csv_file = None
        self._csv_writer = None
        self._flush_interval = 30
        self._frame_in_session = 0
        self._values_buf: list = (
            []
        )  # push() のたびに値をメモリ上に蓄積（stop() で中央値計算用）
        self._rewrite_thread: threading.Thread | None = None  # ヘッダー書き換えスレッド

    def start(self, file_name: str, mode: str, camera_name: str = "IMX477") -> str:
        """記録セッションを開始する。

        CSVファイルを作成し、プレースホルダヘッダーとカラムヘッダーを書き込む。

        Args:
          file_name: ユーザー指定のファイル名（拡張子なし）。
          mode: "Lab" or "LinearRGB"。
          camera_name: カメラ名。
        Returns:
          str: 作成したCSVファイルパス。
        """
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_name = file_name
        self.mode = mode
        self.camera_name = camera_name
        self.start_time = datetime.now()
        self.sample_count = 0
        self._frame_in_session = 0
        self._values_buf = []

        # ファイル名に .csv を付与
        if not file_name.endswith(".csv"):
            file_name = file_name + ".csv"
        self.csv_path = os.path.join(self.log_dir, file_name)

        # プレースホルダヘッダーを書き込み
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        # 9行のプレースホルダ（停止時に上書き）
        for i in range(self.HEADER_LINES):
            self._csv_file.write(f"# placeholder line {i + 1}\n")

        # カラムヘッダー行（10行目）
        if mode == "Lab":
            self._csv_file.write("timestamp,L_star,a_star,b_star\n")
        else:
            self._csv_file.write("timestamp,R_lin,G_lin,B_lin\n")

        self._csv_file.flush()
        self.is_recording = True
        print(f"- Session recording started: {self.csv_path}")
        return self.csv_path

    def push(self, meas_dict: dict) -> None:
        """計測データを1行追記する（is_recording 時のみ）。

        Args:
          meas_dict: 計測結果辞書。Lab モードでは 'Lab' キー、
                     LinearRGB モードでは 'ratio_validated' キーを参照。
        """
        if not self.is_recording or self._csv_file is None:
            return

        ts = datetime.now().strftime("%y/%m/%d %H:%M:%S")

        if self.mode == "Lab":
            lab = meas_dict.get("Lab", (0.0, 0.0, 0.0))
            self._csv_file.write(f"{ts},{lab[0]:.4f},{lab[1]:.4f},{lab[2]:.4f}\n")
            self._values_buf.append([float(lab[0]), float(lab[1]), float(lab[2])])
        else:
            ratio = meas_dict.get("ratio_validated", np.zeros(3))
            self._csv_file.write(f"{ts},{ratio[0]:.6f},{ratio[1]:.6f},{ratio[2]:.6f}\n")
            self._values_buf.append([float(ratio[0]), float(ratio[1]), float(ratio[2])])

        self.sample_count += 1
        self._frame_in_session += 1

        # 定期 flush（毎フレームではなく30フレーム毎）
        if self._frame_in_session % self._flush_interval == 0:
            self._csv_file.flush()

    def stop(self) -> str:
        """記録セッションを即座に停止し、ヘッダー書き換えをバックグラウンドで実施する。

        ファイルクローズと is_recording=False は同期的に完了させ、
        重いヘッダー書き換え処理はバックグラウンドスレッドで実施する。
        これにより s キー押下後の即時フィードバックを保証する。

        Returns:
          str: 保存先CSVファイルパス。
        """
        csv_path = self.csv_path or ""
        sample_count = self.sample_count

        # ① ファイルを即座に閉じる（同期・高速）
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        # ② is_recording を即座に False に（以降 push() は無視される）
        self.is_recording = False

        if not csv_path or not os.path.exists(csv_path):
            return csv_path

        # ③ ヘッダー書き換えをバックグラウンドスレッドで実施
        values_snapshot = list(self._values_buf)  # コピーしてスレッドに渡す
        mode_snapshot = self.mode

        def _rewrite_header_bg():
            try:
                # メモリ上のバッファから中央値を算出（ファイル再読込み不要）
                medians = self._compute_medians_from_values(
                    values_snapshot, mode_snapshot
                )
                header_lines = self._build_header_lines(medians)
                with open(csv_path, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                data_lines = all_lines[self.HEADER_LINES + 1 :]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    for line in header_lines:
                        f.write(line + "\n")
                    f.write(all_lines[self.HEADER_LINES])
                    for line in data_lines:
                        f.write(line)
                print(f"- Session finalized: {csv_path} ({sample_count} samples)")
            except Exception as e:
                print(f"Warning: header rewrite failed: {e}")

        self._rewrite_thread = threading.Thread(target=_rewrite_header_bg, daemon=True)
        self._rewrite_thread.start()

        print(f"- Session recording stopped: {csv_path} ({sample_count} samples)")
        return csv_path

    def _compute_medians_from_lines(self, data_lines: list) -> dict:
        """CSVデータ行リストからモード別の中央値を算出する。

        Args:
          data_lines: CSVデータ行のリスト。
        Returns:
          dict: 中央値辞書。
        """
        values = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                values.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except (ValueError, IndexError):
                continue

        if not values:
            if self.mode == "Lab":
                return {"L_median": 0.0, "a_median": 0.0, "b_median": 0.0}
            else:
                return {"R_median": 0.0, "G_median": 0.0, "B_median": 0.0}

        arr = np.array(values)
        medians = np.median(arr, axis=0)

        if self.mode == "Lab":
            return {
                "L_median": medians[0],
                "a_median": medians[1],
                "b_median": medians[2],
            }
        else:
            return {
                "R_median": medians[0],
                "G_median": medians[1],
                "B_median": medians[2],
            }

    def _compute_medians_from_values(self, values: list, mode: str) -> dict:
        """メモリ上の値リストからモード別の中央値を算出する。

        _compute_medians_from_lines のファイル不要版。
        push() で蓄積した _values_buf を渡すことでファイル再読込みを省略できる。

        Args:
          values: [[v0, v1, v2], ...] の数値リスト。
          mode: "Lab" or "LinearRGB"。
        Returns:
          dict: 中央値辞書。
        """
        if not values:
            if mode == "Lab":
                return {"L_median": 0.0, "a_median": 0.0, "b_median": 0.0}
            else:
                return {"R_median": 0.0, "G_median": 0.0, "B_median": 0.0}
        arr = np.array(values, dtype=np.float64)
        medians = np.median(arr, axis=0)
        if mode == "Lab":
            return {
                "L_median": float(medians[0]),
                "a_median": float(medians[1]),
                "b_median": float(medians[2]),
            }
        else:
            return {
                "R_median": float(medians[0]),
                "G_median": float(medians[1]),
                "B_median": float(medians[2]),
            }

    def _build_header_lines(self, medians: dict) -> list:
        """固定9行のヘッダー文字列リストを生成する。

        Args:
          medians: 中央値辞書。
        Returns:
          list: 9個の文字列（# プレフィックス付き）。
        """
        date_str = (
            self.start_time.strftime("%y/%m/%d %H:%M:%S") if self.start_time else ""
        )
        file_display = os.path.basename(self.csv_path) if self.csv_path else ""

        lines = [
            f"# File: {file_display}",
            f"# Date: {date_str}",
            f"# Camera: {self.camera_name or 'unknown'}",
            f"# Mode: {self.mode or 'unknown'}",
            f"# Samples: {self.sample_count}",
        ]

        if self.mode == "Lab":
            lines.append(f"# L* median: {medians.get('L_median', 0.0):.3f}")
            lines.append(f"# a* median: {medians.get('a_median', 0.0):.3f}")
            lines.append(f"# b* median: {medians.get('b_median', 0.0):.3f}")
        else:
            lines.append(f"# R median: {medians.get('R_median', 0.0):.6f}")
            lines.append(f"# G median: {medians.get('G_median', 0.0):.6f}")
            lines.append(f"# B median: {medians.get('B_median', 0.0):.6f}")

        lines.append("# =============================")

        return lines


class FileNameInputOverlay:
    """
    OpenCV画面上でファイル名を入力させるUIオーバーレイ。

    役割:
      - ターミナル操作なしで、カメラ表示ウィンドウ上にテキスト入力UIを表示する。
      - cv2.waitKey() で1文字ずつキャプチャし、Enter で確定、ESC でキャンセル。

    入力:
      - OpenCV ウィンドウ名とサイズ。
    出力:
      - ユーザーが入力したファイル名文字列、またはキャンセル時 None。
    """

    def __init__(self, window_name: str, width: int = 640, height: int = 200):
        """入力UIの表示先情報を保持する。

        Args:
          window_name: 描画先のOpenCVウィンドウ名。
          width: 入力画面の幅。
          height: 入力画面の高さ。
        """
        self.window_name = window_name
        self.width = width
        self.height = height

    def get_filename(self) -> str:
        """画面上でファイル名を入力させ、確定された文字列を返す。

        Returns:
          str or None: Enter で確定されたファイル名。ESC でキャンセル時は None。
        """
        text = ""
        blink_counter = 0

        while True:
            # 黒背景に描画
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)

            # タイトル
            cv2.putText(
                canvas,
                "Enter filename:",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            # サブテキスト
            cv2.putText(
                canvas,
                "[Enter] OK  [ESC] Cancel  [BS] Delete",
                (20, self.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

            # 入力テキスト + カーソル
            display_text = text
            blink_counter += 1
            if blink_counter % 10 < 5:
                display_text += "_"

            # 入力枠
            cv2.rectangle(canvas, (18, 55), (self.width - 18, 95), (80, 80, 80), 1)
            cv2.putText(
                canvas,
                display_text,
                (25, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(50) & 0xFF

            if key == 13 or key == 10:  # Enter
                if text.strip():
                    return text.strip()
                # 空文字なら無視してループ続行
            elif key == 27:  # ESC
                return None
            elif key == 8 or key == 127:  # Backspace / Delete
                text = text[:-1]
            elif 32 <= key <= 126:  # 印字可能ASCII文字
                text += chr(key)


class HistogramQualityChecker:
    """
    RAW ROIの品質をヒストグラム統計で評価するクラス。

    役割:
      - 飽和/露出不足/ノイズ過多の警告を生成する。

    機能:
      - RAW ROI統計値から露出異常と非一様性を判定し警告コードを生成する。
    入力:
      - RAW ROI配列と品質判定しきい値（飽和/露出不足/CV/中心外周差）。
    出力:
      - 警告コードのリストと中心-外周差分率を返す。
    """

    CENTER_REGION_RATIO = 0.60

    def __init__(
        self,
        saturated_threshold: int = 4090,
        saturated_fraction: float = 0.01,
        underexposed_threshold: int = 10,
        underexposed_fraction: float = 0.05,
        noise_cv_threshold: float = 0.15,
        center_edge_diff_fraction: float = 0.03,
        bayer_channel_check: bool = True,
    ):
        """閾値をコンストラクタで設定する。

        Args:
          saturated_threshold: 飽和判定の閾値（12bit: 4090, 8bit: 250）。
          saturated_fraction: 飽和ピクセルの割合閾値。
          underexposed_threshold: 露出不足判定の閾値（12bit: 10, 8bit: 3）。
          underexposed_fraction: 露出不足ピクセルの割合閾値。
          noise_cv_threshold: ノイズCVの閾値。
          center_edge_diff_fraction: center-edge差の閾値。
          bayer_channel_check: True=Bayerパターン分離チェック, False=BGR全体チェック。
        """
        self.SATURATED_THRESHOLD = saturated_threshold
        self.SATURATED_FRACTION = saturated_fraction
        self.UNDEREXPOSED_THRESHOLD = underexposed_threshold
        self.UNDEREXPOSED_FRACTION = underexposed_fraction
        self.NOISE_CV_THRESHOLD = noise_cv_threshold
        self.CENTER_EDGE_DIFF_FRACTION = center_edge_diff_fraction
        self.bayer_channel_check = bayer_channel_check

    def _is_underexposed(self, flat: np.ndarray) -> bool:
        """露出不足 ROI では空間一様性を有効な測定値として扱わない。"""
        if flat is None or flat.size == 0:
            return True
        return (
            (flat <= self.UNDEREXPOSED_THRESHOLD).mean()
            > self.UNDEREXPOSED_FRACTION
        )

    def center_edge_diff_fraction(self, raw_roi: np.ndarray) -> float:
        """
        ROI中心部と外周部の平均差(相対値)を返す。

        Returns:
          float: 0.0以上の差分率（例 0.025 = 2.5%）。
        """
        if raw_roi is None or raw_roi.size == 0:
            return 0.0
        flat = raw_roi.astype(np.float32)
        if self._is_underexposed(flat):
            return 0.0
        h, w = flat.shape[:2]
        if h < 8 or w < 8:
            return 0.0
        ch = max(int(h * self.CENTER_REGION_RATIO), 2)
        cw = max(int(w * self.CENTER_REGION_RATIO), 2)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        y1 = y0 + ch
        x1 = x0 + cw
        center = flat[y0:y1, x0:x1]
        if center.size == 0:
            return 0.0
        mask = np.ones((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = False
        outer = flat[mask]
        if outer.size == 0:
            return 0.0
        center_mean = float(center.mean())
        outer_mean = float(outer.mean())
        base = max(center_mean, outer_mean, 1.0)
        return abs(center_mean - outer_mean) / base

    def check(self, raw_roi: np.ndarray) -> list:
        """
        RAW ROIの品質を判定し警告リストを返す。

        Args:
          raw_roi: RAW ROI配列。
        Returns:
          list: 警告コードのリスト。
        """
        warnings = []
        flat = raw_roi.astype(np.float32)
        if (flat >= self.SATURATED_THRESHOLD).mean() > self.SATURATED_FRACTION:
            warnings.append("OVEREXPOSED")
        is_underexposed = self._is_underexposed(flat)
        if is_underexposed:
            warnings.append("UNDEREXPOSED")

        if self.bayer_channel_check:
            # Bayerパターン分離チェック (12bit RAW用)
            channels = {
                "R": flat[1::2, 1::2],
                "G1": flat[0::2, 1::2],
                "G2": flat[1::2, 0::2],
                "B": flat[0::2, 0::2],
            }
        else:
            # BGRチャンネルチェック (8bit USB用)
            if flat.ndim == 3 and flat.shape[2] >= 3:
                channels = {
                    "B": flat[:, :, 0],
                    "G": flat[:, :, 1],
                    "R": flat[:, :, 2],
                }
            else:
                channels = {"all": flat}

        if not is_underexposed:
            for _name, ch in channels.items():
                mean_val = ch.mean()
                if mean_val > 1.0:
                    cv = ch.std() / mean_val
                    if cv > self.NOISE_CV_THRESHOLD:
                        warnings.append("NON_UNIFORM")
                        break

            center_edge_diff = self.center_edge_diff_fraction(raw_roi)
            if (
                center_edge_diff > self.CENTER_EDGE_DIFF_FRACTION
                and "NON_UNIFORM" not in warnings
            ):
                warnings.append("NON_UNIFORM")
        return warnings


_REF_UNEVEN_STATUS_LABELS = ("OK", "WATCH", "WARN", "RECAL")
_REF_UNEVEN_STATUS_RANKS = {
    "OK": 0,
    "WATCH": 1,
    "WARN": 2,
    "RECAL": 3,
}
_REF_UNEVEN_STATUS_BY_RANK = {
    rank: label for label, rank in _REF_UNEVEN_STATUS_RANKS.items()
}
_REF_UNEVEN_METRICS = ("CE", "LR", "TB", "DIAG", "TILE", "MAXT")
_REF_UNEVEN_SIGNED_METRICS = {"LR", "TB", "DIAG"}
_REF_UNEVEN_CANONICAL_INDEX = {
    metric: index for index, metric in enumerate(_REF_UNEVEN_METRICS)
}
_REF_UNEVEN_ABS_SPECS = {
    "CE": {"watch": 2.0, "warn": 3.0, "recal": 5.0},
    "LR": {"watch": 2.0, "warn": 3.0, "recal": 5.0},
    "TB": {"watch": 2.0, "warn": 3.0, "recal": 5.0},
    "DIAG": {"watch": 2.0, "warn": 3.0, "recal": 5.0},
    "TILE": {"watch": 3.0, "warn": 5.0, "recal": 8.0},
    "MAXT": {"watch": 4.0, "warn": 6.0, "recal": 10.0},
}
_REF_UNEVEN_DRIFT_SPECS = {
    "CE": {"watch": 1.0, "warn": 2.0, "recal": 3.0},
    "LR": {"watch": 1.0, "warn": 2.0, "recal": 3.0},
    "TB": {"watch": 1.0, "warn": 2.0, "recal": 3.0},
    "DIAG": {"watch": 1.0, "warn": 2.0, "recal": 3.0},
    "TILE": {"watch": 1.5, "warn": 3.0, "recal": 5.0},
    "MAXT": {"watch": 2.0, "warn": 4.0, "recal": 6.0},
}


@dataclass
class RefSpatialAbsolute:
    """Current-frame Ref ROI spatial nonuniformity metrics."""

    valid: bool
    reason: str
    source_kind: str
    mean: float | None
    center_edge_pct: float | None = None
    left_right_pct: float | None = None
    top_bottom_pct: float | None = None
    diag_pct: float | None = None
    tile_p95_p05_pct: float | None = None
    tile_max_dev_pct: float | None = None
    cv_pct: float | None = None
    metric_valid: dict = field(default_factory=dict)
    metric_values_pct: dict = field(default_factory=dict)
    metric_signed_values_pct: dict = field(default_factory=dict)
    abs_dominant: str | None = None
    # Operator display value is always unsigned; signed direction is kept separately.
    abs_value_pct: float | None = None
    abs_limit_pct: float | None = None
    abs_risk: float | None = None
    abs_level_rank: int | None = None
    abs_status: str = "N/A"


@dataclass
class RefUnevenStatus:
    """Operator-facing Uneven status built from current/baseline spatial metrics."""

    status: str
    level_rank: int | None
    reason: str
    value: float | None
    unit: str | None
    value_kind: str
    display_metric: str | None = None
    display_direction: str | None = None
    abs_dominant: str | None = None
    abs_value_pct: float | None = None
    abs_limit_pct: float | None = None
    abs_risk: float | None = None
    drift_dominant: str | None = None
    drift_delta_pp: float | None = None
    drift_limit_pp: float | None = None
    drift_risk: float | None = None
    drift_level_rank: int = 0
    direction_reversal: bool = False
    graph_risk: float | None = None

    def as_monitor_item(self) -> dict:
        limit_value = self.drift_limit_pp if self.value_kind == "drift" else self.abs_limit_pct
        return {
            "axis": "U",
            "label": "Uneven",
            "status": self.status,
            "level": self.level_rank if self.level_rank is not None else 0,
            "value": self.value,
            "unit": self.unit or "",
            "value_kind": self.value_kind,
            "display_value": self.value,
            "display_unit": self.unit,
            "display_metric": self.display_metric,
            "display_direction": self.display_direction,
            "direction_reversal": self.direction_reversal,
            "reason": self.reason,
            "abs_dominant": self.abs_dominant,
            "abs_value_pct": self.abs_value_pct,
            "abs_limit_pct": self.abs_limit_pct,
            "abs_risk": self.abs_risk,
            "drift_dominant": self.drift_dominant,
            "drift_delta_pp": self.drift_delta_pp,
            "drift_limit_pp": self.drift_limit_pp,
            "drift_risk": self.drift_risk,
            "drift_level": self.drift_level_rank,
            "graph_risk": self.graph_risk,
            "risk": self.graph_risk if self.graph_risk is not None else 0.0,
            "limit_value": limit_value,
            "limit_label": "recal" if self.status == "RECAL" else "limit",
        }


def _ref_uneven_level(value, spec: dict) -> tuple[str | None, int | None]:
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(number):
        return None, None
    if number >= float(spec["recal"]):
        return "RECAL", 3
    if number >= float(spec["warn"]):
        return "WARN", 2
    if number >= float(spec["watch"]):
        return "WATCH", 1
    return "OK", 0


def _ref_uneven_next_limit(status: str, spec: dict) -> float | None:
    key = {
        "OK": "watch",
        "WATCH": "warn",
        "WARN": "recal",
        "RECAL": "recal",
    }.get(str(status or "").upper())
    if key is None:
        return None
    return float(spec[key])


def _ref_uneven_direction(metric: str | None, value: float | None) -> str | None:
    if metric not in _REF_UNEVEN_SIGNED_METRICS or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or abs(number) < 1e-9:
        return None
    if metric == "LR":
        return "L>R" if number > 0.0 else "R>L"
    if metric == "TB":
        return "T>B" if number > 0.0 else "B>T"
    if metric == "DIAG":
        return "TLBR>TRBL" if number > 0.0 else "TRBL>TLBR"
    return None


class RefSpatialAnalyzer:
    """Analyze current Ref ROI spatial nonuniformity without baseline drift."""

    MIN_REGION_VALID_FRACTION = 0.80
    MIN_REGION_FINITE_PIXELS = 16
    MIN_QUADRANT_FINITE_PIXELS = 9

    def __init__(
        self,
        *,
        saturated_threshold: float = 4090.0,
        saturated_fraction: float = 0.01,
        underexposed_threshold: float = 10.0,
        underexposed_fraction: float = 0.05,
    ):
        self.saturated_threshold = float(saturated_threshold)
        self.saturated_fraction = float(saturated_fraction)
        self.underexposed_threshold = float(underexposed_threshold)
        self.underexposed_fraction = float(underexposed_fraction)

    @staticmethod
    def _invalid(reason: str, source_kind: str = "unknown") -> RefSpatialAbsolute:
        return RefSpatialAbsolute(
            valid=False,
            reason=reason,
            source_kind=source_kind,
            mean=None,
            metric_valid={metric: False for metric in (*_REF_UNEVEN_METRICS, "CV")},
            metric_values_pct={metric: None for metric in (*_REF_UNEVEN_METRICS, "CV")},
            metric_signed_values_pct={metric: None for metric in (*_REF_UNEVEN_METRICS, "CV")},
            abs_status="N/A",
        )

    @staticmethod
    def _finite_mean(values: np.ndarray) -> float | None:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(np.mean(finite))

    @classmethod
    def _region_usable(cls, values: np.ndarray, min_pixels: int) -> bool:
        arr = np.asarray(values)
        total = int(arr.size)
        if total <= 0:
            return False
        finite_count = int(np.isfinite(arr).sum())
        return (
            finite_count >= int(min_pixels)
            and finite_count / max(total, 1) >= cls.MIN_REGION_VALID_FRACTION
        )

    @staticmethod
    def _denom(reference_luma: float | None, underexposed_threshold: float) -> float:
        ref = 0.0 if reference_luma is None else abs(float(reference_luma))
        return max(ref, 2.0 * abs(float(underexposed_threshold)), 1e-6)

    @staticmethod
    def _metric_status(value: float | None, metric: str) -> tuple[str | None, int | None, float | None]:
        spec = _REF_UNEVEN_ABS_SPECS[metric]
        status, rank = _ref_uneven_level(value, spec)
        if status is None or value is None:
            return None, None, None
        risk = abs(float(value)) / max(float(spec["warn"]), 1e-8)
        return status, rank, risk

    def _luma_map(self, raw_or_luma, *, source_kind: str, channel_order: str | None) -> tuple[np.ndarray | None, str]:
        source = str(source_kind or "").lower()
        arr = np.asarray(raw_or_luma, dtype=np.float64)
        if arr.size == 0:
            return None, "empty"
        if source == "bayer":
            if arr.ndim != 2:
                return None, "bayer_requires_2d"
            h, w = arr.shape
            h_even = h - (h % 2)
            w_even = w - (w % 2)
            if h_even < 2 or w_even < 2:
                return None, "bayer_too_small"
            even = arr[:h_even, :w_even]
            return even.reshape(h_even // 2, 2, w_even // 2, 2).mean(axis=(1, 3)), "ok"
        if source == "mono":
            if arr.ndim != 2:
                return None, "mono_requires_2d"
            return arr, "ok"
        if source == "rgb":
            if arr.ndim != 3 or arr.shape[2] < 3:
                return None, "rgb_requires_3d"
            if str(channel_order or "").upper() not in {"RGB", "BGR"}:
                return None, "rgb_channel_order_required"
            return arr[:, :, :3].mean(axis=2), "ok"
        return None, "source_kind_required"

    def _tile_metrics(
        self,
        luma: np.ndarray,
        *,
        underexposed_threshold: float,
    ) -> tuple[bool, float | None, float | None, float | None]:
        h, w = luma.shape[:2]
        selected_grid = None
        for grid, min_tile_px in ((6, 6), (4, 4)):
            if h >= grid * min_tile_px and w >= grid * min_tile_px:
                selected_grid = grid
                break
        if selected_grid is None:
            return False, None, None, None

        tile_means = []
        for gy in range(selected_grid):
            y0 = int(round(gy * h / selected_grid))
            y1 = int(round((gy + 1) * h / selected_grid))
            for gx in range(selected_grid):
                x0 = int(round(gx * w / selected_grid))
                x1 = int(round((gx + 1) * w / selected_grid))
                tile = luma[y0:y1, x0:x1]
                if not self._region_usable(tile, 9):
                    return False, None, None, None
                tile_mean = self._finite_mean(tile)
                if tile_mean is None:
                    return False, None, None, None
                tile_means.append(tile_mean)

        means = np.asarray(tile_means, dtype=np.float64)
        ref = float(np.median(means))
        denom = self._denom(ref, underexposed_threshold)
        spread = float((np.percentile(means, 95) - np.percentile(means, 5)) / denom * 100.0)
        max_dev = float(np.max(np.abs(means - ref)) / denom * 100.0)
        cv = float(np.std(means, ddof=0) / denom * 100.0)
        return True, spread, max_dev, cv

    def analyze_ref_roi(
        self,
        raw_or_luma,
        *,
        source_kind: str,
        channel_order: str | None = None,
        underexposed_threshold_luma: float | None = None,
        saturated_threshold_luma: float | None = None,
        correction_stage: str = "dark_flat_corrected",
    ) -> RefSpatialAbsolute:
        del correction_stage
        source = str(source_kind or "").lower()
        luma, reason = self._luma_map(raw_or_luma, source_kind=source, channel_order=channel_order)
        if luma is None:
            return self._invalid(reason, source)

        finite = luma[np.isfinite(luma)]
        if finite.size == 0:
            return self._invalid("non_finite", source)

        under = self.underexposed_threshold if underexposed_threshold_luma is None else float(underexposed_threshold_luma)
        sat = self.saturated_threshold if saturated_threshold_luma is None else float(saturated_threshold_luma)
        if float(np.mean(finite >= sat)) > self.saturated_fraction:
            return self._invalid("saturated", source)
        if float(np.mean(finite <= under)) > self.underexposed_fraction:
            return self._invalid("underexposed", source)

        h, w = luma.shape[:2]
        metric_valid = {metric: False for metric in (*_REF_UNEVEN_METRICS, "CV")}
        metric_values = {metric: None for metric in (*_REF_UNEVEN_METRICS, "CV")}
        metric_signed = {metric: None for metric in (*_REF_UNEVEN_METRICS, "CV")}
        full_mean = self._finite_mean(luma)
        denom_full = self._denom(full_mean, under)

        if h >= 8 and w >= 8:
            ch = max(int(h * HistogramQualityChecker.CENTER_REGION_RATIO), 2)
            cw = max(int(w * HistogramQualityChecker.CENTER_REGION_RATIO), 2)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            y1 = y0 + ch
            x1 = x0 + cw
            center = luma[y0:y1, x0:x1]
            outer_mask = np.ones((h, w), dtype=bool)
            outer_mask[y0:y1, x0:x1] = False
            outer = luma[outer_mask]
            if self._region_usable(center, self.MIN_REGION_FINITE_PIXELS) and self._region_usable(outer, self.MIN_REGION_FINITE_PIXELS):
                center_mean = self._finite_mean(center)
                outer_mean = self._finite_mean(outer)
                if center_mean is not None and outer_mean is not None:
                    value = abs(center_mean - outer_mean) / denom_full * 100.0
                    metric_valid["CE"] = True
                    metric_values["CE"] = float(value)
                    metric_signed["CE"] = float(value)

            left = luma[:, : w // 2]
            right = luma[:, w // 2 :]
            if self._region_usable(left, self.MIN_REGION_FINITE_PIXELS) and self._region_usable(right, self.MIN_REGION_FINITE_PIXELS):
                left_mean = self._finite_mean(left)
                right_mean = self._finite_mean(right)
                if left_mean is not None and right_mean is not None:
                    value = (left_mean - right_mean) / denom_full * 100.0
                    metric_valid["LR"] = True
                    metric_values["LR"] = abs(float(value))
                    metric_signed["LR"] = float(value)

            top = luma[: h // 2, :]
            bottom = luma[h // 2 :, :]
            if self._region_usable(top, self.MIN_REGION_FINITE_PIXELS) and self._region_usable(bottom, self.MIN_REGION_FINITE_PIXELS):
                top_mean = self._finite_mean(top)
                bottom_mean = self._finite_mean(bottom)
                if top_mean is not None and bottom_mean is not None:
                    value = (top_mean - bottom_mean) / denom_full * 100.0
                    metric_valid["TB"] = True
                    metric_values["TB"] = abs(float(value))
                    metric_signed["TB"] = float(value)

            tl = luma[: h // 2, : w // 2]
            tr = luma[: h // 2, w // 2 :]
            bl = luma[h // 2 :, : w // 2]
            br = luma[h // 2 :, w // 2 :]
            quads = (tl, tr, bl, br)
            if all(self._region_usable(q, self.MIN_QUADRANT_FINITE_PIXELS) for q in quads):
                tl_mean, tr_mean, bl_mean, br_mean = [self._finite_mean(q) for q in quads]
                if None not in (tl_mean, tr_mean, bl_mean, br_mean):
                    value = (((tl_mean + br_mean) * 0.5) - ((tr_mean + bl_mean) * 0.5)) / denom_full * 100.0
                    metric_valid["DIAG"] = True
                    metric_values["DIAG"] = abs(float(value))
                    metric_signed["DIAG"] = float(value)

        tile_valid, tile_spread, tile_max_dev, cv_pct = self._tile_metrics(luma, underexposed_threshold=under)
        if tile_valid:
            metric_valid["TILE"] = True
            metric_valid["MAXT"] = True
            metric_valid["CV"] = True
            metric_values["TILE"] = tile_spread
            metric_values["MAXT"] = tile_max_dev
            metric_values["CV"] = cv_pct
            metric_signed["TILE"] = tile_spread
            metric_signed["MAXT"] = tile_max_dev
            metric_signed["CV"] = cv_pct

        valid_metrics = [m for m in _REF_UNEVEN_METRICS if metric_valid.get(m)]
        if not valid_metrics:
            return RefSpatialAbsolute(
                valid=False,
                reason="no_valid_metrics",
                source_kind=source,
                mean=full_mean,
                cv_pct=metric_values.get("CV"),
                metric_valid=metric_valid,
                metric_values_pct=metric_values,
                metric_signed_values_pct=metric_signed,
                abs_status="N/A",
            )

        best = None
        for metric in valid_metrics:
            status, rank, risk = self._metric_status(metric_values.get(metric), metric)
            if status is None or rank is None or risk is None:
                continue
            key = (rank, risk, -_REF_UNEVEN_CANONICAL_INDEX[metric])
            if best is None or key > best[0]:
                best = (key, metric, status, rank, risk)
        if best is None:
            return self._invalid("no_status_metric", source)
        _key, dominant, status, rank, risk = best
        limit = _ref_uneven_next_limit(status, _REF_UNEVEN_ABS_SPECS[dominant])

        return RefSpatialAbsolute(
            valid=True,
            reason="ok",
            source_kind=source,
            mean=full_mean,
            center_edge_pct=metric_values.get("CE"),
            left_right_pct=metric_signed.get("LR"),
            top_bottom_pct=metric_signed.get("TB"),
            diag_pct=metric_signed.get("DIAG"),
            tile_p95_p05_pct=metric_values.get("TILE"),
            tile_max_dev_pct=metric_values.get("MAXT"),
            cv_pct=metric_values.get("CV"),
            metric_valid=metric_valid,
            metric_values_pct=metric_values,
            metric_signed_values_pct=metric_signed,
            abs_dominant=dominant,
            abs_value_pct=metric_values.get(dominant),
            abs_limit_pct=limit,
            abs_risk=risk,
            abs_level_rank=rank,
            abs_status=status,
        )


def build_ref_uneven_status(
    current_abs: RefSpatialAbsolute | None,
    baseline_abs: RefSpatialAbsolute | None = None,
) -> RefUnevenStatus:
    """Combine current spatial state and baseline spatial drift into HUD U status."""
    if current_abs is None or not current_abs.valid:
        return RefUnevenStatus(
            status="N/A",
            level_rank=None,
            reason="invalid",
            value=None,
            unit=None,
            value_kind="invalid",
            graph_risk=None,
        )

    abs_rank = int(current_abs.abs_level_rank or 0)
    abs_status = current_abs.abs_status or _REF_UNEVEN_STATUS_BY_RANK.get(abs_rank, "OK")
    abs_metric = current_abs.abs_dominant
    abs_value = current_abs.abs_value_pct
    abs_signed_value = current_abs.metric_signed_values_pct.get(abs_metric) if abs_metric else None
    abs_direction = _ref_uneven_direction(abs_metric, abs_signed_value)

    drift_metric = None
    drift_delta = None
    drift_limit = None
    drift_risk = None
    drift_rank = 0
    direction_reversal = False
    if (
        baseline_abs is not None
        and baseline_abs.valid
        and baseline_abs.source_kind == current_abs.source_kind
    ):
        best = None
        for metric in _REF_UNEVEN_METRICS:
            if not current_abs.metric_valid.get(metric) or not baseline_abs.metric_valid.get(metric):
                continue
            cur = current_abs.metric_signed_values_pct.get(metric)
            base = baseline_abs.metric_signed_values_pct.get(metric)
            if cur is None or base is None:
                continue
            delta = float(cur) - float(base)
            level_value = abs(delta) if metric in _REF_UNEVEN_SIGNED_METRICS else max(0.0, delta)
            status, rank = _ref_uneven_level(level_value, _REF_UNEVEN_DRIFT_SPECS[metric])
            if status is None or rank is None:
                continue
            risk = level_value / max(float(_REF_UNEVEN_DRIFT_SPECS[metric]["warn"]), 1e-8)
            key = (rank, risk, -_REF_UNEVEN_CANONICAL_INDEX[metric])
            if best is None or key > best[0]:
                best = (key, metric, status, rank, risk, delta, level_value)
        if best is not None:
            _key, best_metric, drift_status, best_rank, best_risk, best_delta, _level_value = best
            drift_risk = best_risk
            if int(best_rank) > 0:
                drift_metric = best_metric
                drift_rank = int(best_rank)
                drift_delta = best_delta
                drift_limit = _ref_uneven_next_limit(drift_status, _REF_UNEVEN_DRIFT_SPECS[drift_metric])
            if drift_metric in _REF_UNEVEN_SIGNED_METRICS:
                base = baseline_abs.metric_signed_values_pct.get(drift_metric)
                cur = current_abs.metric_signed_values_pct.get(drift_metric)
                if base is not None and cur is not None:
                    reversal_floor = float(_REF_UNEVEN_DRIFT_SPECS[drift_metric]["watch"])
                    direction_reversal = (
                        float(base) * float(cur) < 0.0
                        and abs(float(drift_delta)) >= float(_REF_UNEVEN_DRIFT_SPECS[drift_metric]["watch"])
                        and max(abs(float(base)), abs(float(cur))) >= reversal_floor
                    )

    combined_rank = max(abs_rank, int(drift_rank))
    status = _REF_UNEVEN_STATUS_BY_RANK.get(combined_rank, "OK")
    if combined_rank <= 0:
        reason = "abs"
    elif abs_rank > drift_rank:
        reason = "abs"
    elif drift_rank > abs_rank:
        reason = "drift"
    else:
        reason = "both" if drift_rank > 0 else "abs"

    if reason == "drift":
        value = drift_delta
        unit = "pp"
        value_kind = "drift"
        display_metric = drift_metric
        display_direction = None
    else:
        value = abs_value
        unit = "%"
        value_kind = "absolute"
        display_metric = abs_metric
        display_direction = abs_direction

    graph_risk = max(float(current_abs.abs_risk or 0.0), float(drift_risk or 0.0))
    return RefUnevenStatus(
        status=status,
        level_rank=combined_rank,
        reason=reason,
        value=value,
        unit=unit,
        value_kind=value_kind,
        display_metric=display_metric,
        display_direction=display_direction,
        abs_dominant=abs_metric,
        abs_value_pct=abs_value,
        abs_limit_pct=current_abs.abs_limit_pct,
        abs_risk=current_abs.abs_risk,
        drift_dominant=drift_metric,
        drift_delta_pp=drift_delta,
        drift_limit_pp=drift_limit,
        drift_risk=drift_risk,
        drift_level_rank=int(drift_rank),
        direction_reversal=direction_reversal,
        graph_risk=graph_risk,
    )


class DarkFlatStabilityTracker:
    """
    Dark/Flat補正後のリアルタイム安定性を監視するトラッカー。

    役割:
      - Ref RAW mean（照明強度）と center-edge（空間一様性）を時系列記録
      - 起動直後 warmup_frames で初期基準値を確定
      - 初期値からの偏差が閾値を超えたらアラート

    機能:
      - Dark/Flat適用後の照明強度と一様性を監視し、アラート判定を提供する。
    入力:
      - フレームごとのRef RAW平均値、uniformity値、しきい値設定。
    出力:
      - 時系列履歴、基準からの偏差[%]、アラート判定結果を返す。
    """

    STATE_NORMAL = "NORMAL"
    STATE_LED_DRIFT_WARN = "LED_DRIFT_WARN"
    STATE_RECALIB_REQUIRED = "RECALIB_REQUIRED"

    DRIFT_OK = "OK"
    DRIFT_WATCH = "WATCH"
    DRIFT_WARN = "WARN"
    DRIFT_RECAL = "RECAL"

    _STATE_LEVELS = {
        DRIFT_OK: 0,
        DRIFT_WATCH: 1,
        DRIFT_WARN: 2,
        DRIFT_RECAL: 3,
    }
    _LEVEL_STATES = {value: key for key, value in _STATE_LEVELS.items()}

    _WATCH_THRESHOLDS = {
        "I": 5.0,
        "C": 0.006,
        "U": 0.75,
        "CLIP": 0.01,
        "FLOOR": 0.01,
        "J": 0.50,
    }
    _WARN_THRESHOLDS = {
        "I": 8.0,
        "C": 0.010,
        "U": 1.0,
        "CLIP": 0.10,
        "FLOOR": 0.10,
        "J": 1.00,
    }
    _RECAL_THRESHOLDS = {
        "I": 12.0,
        "C": 0.015,
        "U": 1.5,
        "CLIP": 1.00,
        "FLOOR": 1.00,
        "J": 2.00,
    }
    _REF_MONITOR_AXES = ("I", "C", "U", "J", "CLIP", "FLOOR")
    _REF_MONITOR_LABELS = {
        "I": "Bright",
        "C": "Color",
        "U": "Uneven",
        "J": "Jitter",
        "CLIP": "Clip",
        "FLOOR": "Floor",
    }
    _REF_MONITOR_UNITS = {
        "I": "%",
        "C": "",
        "U": "%",
        "J": "%",
        "CLIP": "%",
        "FLOOR": "%",
    }
    _REF_MONITOR_AXIS_CYCLE = ("AUTO", "I", "C", "U", "J", "CLIP", "FLOOR")
    _JITTER_WINDOW = 60
    _PROMOTE_SEC = 2.0
    _DEMOTE_SEC = 6.0
    _PROMOTE_FRAMES = 30
    _DEMOTE_FRAMES = 90

    def __init__(
        self,
        buffer_size: int = 300,
        warmup_frames: int = 10,
        raw_mean_threshold_pct: float = 5.0,
        uniformity_threshold_pct: float = 15.0,
    ):
        """
        安定性トラッカーを初期化する。

        Args:
          buffer_size: リングバッファ長（フレーム数）。
          warmup_frames: 初期基準値の平均に使うフレーム数。
          raw_mean_threshold_pct: Ref RAW平均の許容偏差[%]。
          uniformity_threshold_pct: center-edge差の上限[%]。
        """
        self.buffer_size = buffer_size
        self.warmup_frames = warmup_frames
        self.raw_mean_threshold_pct = raw_mean_threshold_pct
        self.uniformity_threshold_pct = uniformity_threshold_pct

        self.raw_mean_buf = np.zeros(buffer_size, dtype=np.float32)
        self.uniformity_buf = np.zeros(buffer_size, dtype=np.float32)
        self.chromaticity_buf = np.zeros((buffer_size, 3), dtype=np.float32)
        self.clip_high_buf = np.zeros(buffer_size, dtype=np.float32)
        self.clip_low_buf = np.zeros(buffer_size, dtype=np.float32)
        self.uneven_abs_value_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.uneven_abs_limit_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.uneven_abs_risk_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.uneven_abs_level_buf = np.zeros(buffer_size, dtype=np.int16)
        self.uneven_drift_delta_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.uneven_drift_limit_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.uneven_drift_risk_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.uneven_drift_level_buf = np.zeros(buffer_size, dtype=np.int16)
        self.uneven_display_value_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.uneven_graph_risk_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.ref_center_edge_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.ref_left_right_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.ref_top_bottom_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.ref_diag_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.ref_tile_p95_p05_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.ref_tile_max_dev_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.ref_tile_cv_buf = np.full(buffer_size, np.nan, dtype=np.float32)
        self.spatial_valid_buf = np.zeros(buffer_size, dtype=np.bool_)
        self.spatial_abs_buf: list[RefSpatialAbsolute | None] = [None] * buffer_size
        self.uneven_status_buf: list[RefUnevenStatus | None] = [None] * buffer_size
        self.uneven_abs_metric_buf: list[str | None] = [None] * buffer_size
        self.uneven_drift_metric_buf: list[str | None] = [None] * buffer_size
        self.uneven_display_metric_buf: list[str | None] = [None] * buffer_size
        self.uneven_display_unit_buf: list[str | None] = [None] * buffer_size
        self.uneven_value_kind_buf: list[str | None] = [None] * buffer_size
        self.uneven_reason_buf: list[str | None] = [None] * buffer_size
        self.spatial_source_kind_buf: list[str | None] = [None] * buffer_size
        self.index = 0
        self.count = 0

        self._warmup_raw = []
        self._warmup_unif = []
        self._warmup_chr = []
        self._warmup_clip_high = []
        self._warmup_clip_low = []
        self._warmup_spatial_abs: list[RefSpatialAbsolute] = []
        self.ref_raw_mean = None
        self.ref_uniformity = None
        self.ref_chromaticity = None
        self.ref_clip_high_pct = 0.0
        self.ref_clip_low_pct = 0.0
        self.baseline_spatial_abs: RefSpatialAbsolute | None = None
        self.baseline_source = "none"
        self.baseline_kind = "none"
        self.baseline_timestamp = None
        self.hysteresis_frames = 30
        self._led_drift_counter = 0
        self._recalib_counter = 0
        self._recover_counter = 0
        self.guard_state = self.STATE_NORMAL
        self.guard_reason = ""
        self.drift_state = self.DRIFT_OK
        self.drift_reason_axes = ""
        self.ref_monitor_axis_mode = "AUTO"
        self._candidate_level = 0
        self._candidate_frames = 0
        self._candidate_started_at = None
        self._recover_target_level = 0
        self._recover_frames = 0
        self._recover_started_at = None
        self.recalib_raw_threshold_pct = 12.0
        self.recalib_uniformity_threshold_pct = 1.5
        self.recalib_chromaticity_threshold = 0.015
        self.led_drift_raw_threshold_pct = 8.0
        self.led_drift_uniformity_threshold_pct = 1.0
        self.led_drift_chromaticity_threshold = 0.010

    @staticmethod
    def _finite_float(value: float, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _normalize_chroma(chromaticity: Optional[np.ndarray]) -> np.ndarray:
        if chromaticity is None:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        chroma = np.asarray(chromaticity, dtype=np.float32).reshape(-1)
        if chroma.size < 3 or not np.all(np.isfinite(chroma[:3])):
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return chroma[:3].astype(np.float32, copy=True)

    @staticmethod
    def _ordered_from_buffer(buf: np.ndarray, count: int, index: int, buffer_size: int) -> np.ndarray:
        if count < buffer_size:
            return buf[:count].copy()
        start = index % buffer_size
        return np.concatenate([buf[start:], buf[:start]])

    def _current_uneven_status(self) -> RefUnevenStatus | None:
        if self.count == 0:
            return None
        return self.uneven_status_buf[(self.index - 1) % self.buffer_size]

    @staticmethod
    def _store_optional_float(buf: np.ndarray, idx: int, value) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            buf[idx] = np.nan
            return
        buf[idx] = number if math.isfinite(number) else np.nan

    def _store_spatial_status(
        self,
        idx: int,
        spatial_abs: RefSpatialAbsolute | None,
        status: RefUnevenStatus | None,
    ) -> None:
        self.spatial_abs_buf[idx] = spatial_abs
        self.uneven_status_buf[idx] = status
        self.spatial_valid_buf[idx] = bool(spatial_abs is not None and spatial_abs.valid)
        self.spatial_source_kind_buf[idx] = (
            None if spatial_abs is None else str(spatial_abs.source_kind)
        )
        if spatial_abs is not None:
            self._store_optional_float(self.ref_center_edge_buf, idx, spatial_abs.center_edge_pct)
            self._store_optional_float(self.ref_left_right_buf, idx, spatial_abs.left_right_pct)
            self._store_optional_float(self.ref_top_bottom_buf, idx, spatial_abs.top_bottom_pct)
            self._store_optional_float(self.ref_diag_buf, idx, spatial_abs.diag_pct)
            self._store_optional_float(self.ref_tile_p95_p05_buf, idx, spatial_abs.tile_p95_p05_pct)
            self._store_optional_float(self.ref_tile_max_dev_buf, idx, spatial_abs.tile_max_dev_pct)
            self._store_optional_float(self.ref_tile_cv_buf, idx, spatial_abs.cv_pct)
            self._store_optional_float(self.uneven_abs_value_buf, idx, spatial_abs.abs_value_pct)
            self._store_optional_float(self.uneven_abs_limit_buf, idx, spatial_abs.abs_limit_pct)
            self._store_optional_float(self.uneven_abs_risk_buf, idx, spatial_abs.abs_risk)
            self.uneven_abs_level_buf[idx] = int(spatial_abs.abs_level_rank or 0)
            self.uneven_abs_metric_buf[idx] = spatial_abs.abs_dominant
        else:
            self._store_optional_float(self.ref_center_edge_buf, idx, None)
            self._store_optional_float(self.ref_left_right_buf, idx, None)
            self._store_optional_float(self.ref_top_bottom_buf, idx, None)
            self._store_optional_float(self.ref_diag_buf, idx, None)
            self._store_optional_float(self.ref_tile_p95_p05_buf, idx, None)
            self._store_optional_float(self.ref_tile_max_dev_buf, idx, None)
            self._store_optional_float(self.ref_tile_cv_buf, idx, None)
            self._store_optional_float(self.uneven_abs_value_buf, idx, None)
            self._store_optional_float(self.uneven_abs_limit_buf, idx, None)
            self._store_optional_float(self.uneven_abs_risk_buf, idx, None)
            self.uneven_abs_level_buf[idx] = 0
            self.uneven_abs_metric_buf[idx] = None

        if status is not None:
            self._store_optional_float(self.uneven_drift_delta_buf, idx, status.drift_delta_pp)
            self._store_optional_float(self.uneven_drift_limit_buf, idx, status.drift_limit_pp)
            self._store_optional_float(self.uneven_drift_risk_buf, idx, status.drift_risk)
            self._store_optional_float(self.uneven_display_value_buf, idx, status.value)
            self._store_optional_float(self.uneven_graph_risk_buf, idx, status.graph_risk)
            self.uneven_drift_level_buf[idx] = int(status.drift_level_rank or 0)
            self.uneven_drift_metric_buf[idx] = status.drift_dominant
            self.uneven_display_metric_buf[idx] = status.display_metric
            self.uneven_display_unit_buf[idx] = status.unit
            self.uneven_value_kind_buf[idx] = status.value_kind
            self.uneven_reason_buf[idx] = status.reason
        else:
            self._store_optional_float(self.uneven_drift_delta_buf, idx, None)
            self._store_optional_float(self.uneven_drift_limit_buf, idx, None)
            self._store_optional_float(self.uneven_drift_risk_buf, idx, None)
            self._store_optional_float(self.uneven_display_value_buf, idx, None)
            self._store_optional_float(self.uneven_graph_risk_buf, idx, None)
            self.uneven_drift_level_buf[idx] = 0
            self.uneven_drift_metric_buf[idx] = None
            self.uneven_display_metric_buf[idx] = None
            self.uneven_display_unit_buf[idx] = None
            self.uneven_value_kind_buf[idx] = None
            self.uneven_reason_buf[idx] = None

    @staticmethod
    def _format_baseline_time(timestamp: float | None) -> str:
        if timestamp is None:
            return ""
        try:
            return datetime.fromtimestamp(float(timestamp)).strftime("%H:%M")
        except Exception:
            return ""

    @staticmethod
    def _source_label(source: str | None) -> str:
        value = str(source or "").strip().lower()
        if value in {"w", "white", "master_ref"}:
            return "W"
        if value in {"n", "neutral"}:
            return "N"
        if value in {"p", "ccm", "ref_train"}:
            return "P"
        if value in {"wnp", "calib", "calibration"}:
            return "WNP"
        if value == "warmup":
            return "WARMUP"
        if value in {"unver", "unverified"}:
            return "UNVER"
        return "UNVER"

    def push(
        self,
        raw_mean: float,
        uniformity_pct: float,
        chromaticity: Optional[np.ndarray] = None,
        clip_high_pct: float = 0.0,
        clip_low_pct: float = 0.0,
        spatial_abs: RefSpatialAbsolute | None = None,
        timestamp: Optional[float] = None,
    ):
        """新しいフレームの値を記録する。"""
        raw_mean = self._finite_float(raw_mean)
        uniformity_pct = self._finite_float(uniformity_pct)
        clip_high_pct = max(0.0, self._finite_float(clip_high_pct))
        clip_low_pct = max(0.0, self._finite_float(clip_low_pct))
        chroma = self._normalize_chroma(chromaticity)
        if self.ref_raw_mean is None:
            self._warmup_raw.append(raw_mean)
            self._warmup_unif.append(uniformity_pct)
            self._warmup_chr.append(chroma.copy())
            self._warmup_clip_high.append(clip_high_pct)
            self._warmup_clip_low.append(clip_low_pct)
            if spatial_abs is not None and spatial_abs.valid:
                self._warmup_spatial_abs.append(spatial_abs)
            if len(self._warmup_raw) >= self.warmup_frames:
                self.ref_raw_mean = float(np.mean(self._warmup_raw))
                self.ref_uniformity = float(np.mean(self._warmup_unif))
                self.ref_chromaticity = np.mean(
                    np.asarray(self._warmup_chr, dtype=np.float32),
                    axis=0,
                )
                self.ref_clip_high_pct = float(np.mean(self._warmup_clip_high))
                self.ref_clip_low_pct = float(np.mean(self._warmup_clip_low))
                if self._warmup_spatial_abs:
                    self.baseline_spatial_abs = self._warmup_spatial_abs[-1]
                self.baseline_source = "warmup"
                self.baseline_kind = "warmup"
                self.baseline_timestamp = time.time()

        idx = self.index % self.buffer_size
        self.raw_mean_buf[idx] = raw_mean
        self.uniformity_buf[idx] = uniformity_pct
        self.chromaticity_buf[idx] = chroma
        self.clip_high_buf[idx] = clip_high_pct
        self.clip_low_buf[idx] = clip_low_pct
        uneven_status = (
            build_ref_uneven_status(spatial_abs, self.baseline_spatial_abs)
            if spatial_abs is not None
            else None
        )
        self._store_spatial_status(idx, spatial_abs, uneven_status)
        self.index += 1
        self.count = min(self.count + 1, self.buffer_size)
        self._update_guard_state(timestamp=timestamp)

    def get_history(self, which: str = "raw_mean") -> np.ndarray:
        """時系列データを古い順に返す。"""
        if which == "raw_mean":
            buf = self.raw_mean_buf
        elif which == "uniformity":
            buf = self.uniformity_buf
        elif which == "clip_high_pct":
            buf = self.clip_high_buf
        elif which == "clip_low_pct":
            buf = self.clip_low_buf
        else:
            buf = self.raw_mean_buf
        if self.count < self.buffer_size:
            history = buf[: self.count].copy()
        else:
            start = self.index % self.buffer_size
            history = np.concatenate([buf[start:], buf[:start]])
        if which == "intensity_drift_pct":
            if self.ref_raw_mean is None or abs(self.ref_raw_mean) < 1e-8:
                return np.zeros_like(history, dtype=np.float32)
            return ((history.astype(np.float64) - self.ref_raw_mean) / self.ref_raw_mean * 100.0).astype(np.float32)
        return history

    def _ordered_chromaticity_history(self) -> np.ndarray:
        """chromaticity 履歴を古い順に返す。"""
        if self.count < self.buffer_size:
            return self.chromaticity_buf[: self.count].copy()
        start = self.index % self.buffer_size
        return np.concatenate([self.chromaticity_buf[start:], self.chromaticity_buf[:start]])

    def get_ref_monitor_axis_history(self, axis: str) -> np.ndarray:
        """Ref Stability Monitor 用に指定軸の時系列を返す。"""
        axis = str(axis or "I").upper()
        if axis == "I":
            return self.get_history("intensity_drift_pct").astype(np.float32)
        if axis == "U":
            if any(item is not None for item in self.uneven_status_buf):
                return self._ordered_from_buffer(
                    self.uneven_graph_risk_buf,
                    self.count,
                    self.index,
                    self.buffer_size,
                ).astype(np.float32)
            history = self.get_history("uniformity").astype(np.float64)
            if self.ref_uniformity is None:
                return np.zeros_like(history, dtype=np.float32)
            if abs(float(self.ref_uniformity)) < 1e-8:
                return (history - float(self.ref_uniformity)).astype(np.float32)
            return ((history - float(self.ref_uniformity)) / float(self.ref_uniformity) * 100.0).astype(np.float32)
        if axis == "C":
            history = self._ordered_chromaticity_history().astype(np.float64)
            if self.ref_chromaticity is None or history.size == 0:
                return np.zeros((history.shape[0],), dtype=np.float32)
            ref = np.asarray(self.ref_chromaticity, dtype=np.float64).reshape(3)
            eps = 1e-8
            r0 = max(abs(float(ref[0])), eps)
            b0 = max(abs(float(ref[2])), eps)
            dr = (history[:, 0] - float(ref[0])) / r0
            db = (history[:, 2] - float(ref[2])) / b0
            return np.sqrt(dr * dr + db * db).astype(np.float32)
        if axis == "J":
            raw_history = self.get_history("raw_mean").astype(np.float64)
            if self.ref_raw_mean is None or abs(float(self.ref_raw_mean)) < 1e-8:
                return np.zeros_like(raw_history, dtype=np.float32)
            values = np.zeros_like(raw_history, dtype=np.float64)
            for idx in range(raw_history.size):
                window = raw_history[max(0, idx + 1 - self._JITTER_WINDOW): idx + 1]
                if window.size >= 2:
                    values[idx] = np.std(window, ddof=0) / max(abs(float(self.ref_raw_mean)), 1e-8) * 100.0
            return values.astype(np.float32)
        if axis == "CLIP":
            return self.get_history("clip_high_pct").astype(np.float32)
        if axis == "FLOOR":
            return self.get_history("clip_low_pct").astype(np.float32)
        return self.get_ref_monitor_axis_history("I")

    def get_dev_I_pct(self) -> Optional[float]:
        """照明強度偏差 dev_I_pct [%] を返す。"""
        if self.ref_raw_mean is None or self.count == 0:
            return None
        curr = float(self.raw_mean_buf[(self.index - 1) % self.buffer_size])
        if abs(self.ref_raw_mean) < 1e-8:
            return None
        return (curr - self.ref_raw_mean) / self.ref_raw_mean * 100.0

    def get_raw_mean_deviation_pct(self) -> Optional[float]:
        """後方互換: 照明強度偏差[%]を返す。"""
        return self.get_dev_I_pct()

    def is_raw_mean_alert(self) -> bool:
        """Ref RAW平均が閾値を超えたか。"""
        dev = self.get_raw_mean_deviation_pct()
        if dev is None:
            return False
        return abs(dev) > self.raw_mean_threshold_pct

    def get_dev_uniformity_pct(self) -> Optional[float]:
        """空間一様性偏差 dev_uniformity_pct [%] を返す。"""
        if self.ref_uniformity is None or self.count == 0:
            return None
        curr = float(self.uniformity_buf[(self.index - 1) % self.buffer_size])
        if abs(self.ref_uniformity) < 1e-8:
            return curr - self.ref_uniformity
        return (curr - self.ref_uniformity) / self.ref_uniformity * 100.0

    def get_dev_chromaticity(self) -> Optional[float]:
        """色バランス偏差 dev_chromaticity を返す。"""
        if self.ref_chromaticity is None or self.count == 0:
            return None
        curr = self.chromaticity_buf[(self.index - 1) % self.buffer_size].astype(
            np.float64
        )
        ref = np.asarray(self.ref_chromaticity, dtype=np.float64).reshape(3)
        eps = 1e-8
        r0 = max(abs(float(ref[0])), eps)
        b0 = max(abs(float(ref[2])), eps)
        dr = (float(curr[0]) - float(ref[0])) / r0
        db = (float(curr[2]) - float(ref[2])) / b0
        return float(math.sqrt(dr * dr + db * db))

    def get_intensity_jitter_pct(self) -> Optional[float]:
        """Ref intensity の短期ジッタ[%]を返す。"""
        if self.ref_raw_mean is None or abs(self.ref_raw_mean) < 1e-8 or self.count < 2:
            return None
        history = self.get_history("raw_mean").astype(np.float64)
        if history.size < 2:
            return None
        window = history[-min(self._JITTER_WINDOW, history.size):]
        return float(np.std(window, ddof=0) / max(abs(self.ref_raw_mean), 1e-8) * 100.0)

    def get_clip_high_pct(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.clip_high_buf[(self.index - 1) % self.buffer_size])

    def get_clip_low_pct(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.clip_low_buf[(self.index - 1) % self.buffer_size])

    def _axis_values(self) -> dict[str, float]:
        uneven_status = self._current_uneven_status()
        if uneven_status is not None:
            uneven_value = float(uneven_status.graph_risk or 0.0)
        else:
            uneven_value = abs(float(self.get_dev_uniformity_pct() or 0.0))
        return {
            "I": abs(float(self.get_dev_I_pct() or 0.0)),
            "C": abs(float(self.get_dev_chromaticity() or 0.0)),
            "U": uneven_value,
            "CLIP": self.get_clip_high_pct(),
            "FLOOR": self.get_clip_low_pct(),
            "J": abs(float(self.get_intensity_jitter_pct() or 0.0)),
        }

    def _axis_level(self, axis: str, value: float) -> int:
        if str(axis).upper() == "U":
            uneven_status = self._current_uneven_status()
            if uneven_status is not None:
                return int(uneven_status.level_rank or 0)
        if value >= self._RECAL_THRESHOLDS[axis]:
            return 3
        if value >= self._WARN_THRESHOLDS[axis]:
            return 2
        if value >= self._WATCH_THRESHOLDS[axis]:
            return 1
        return 0

    def _monitor_axis_level(self, axis: str, value: float) -> int:
        """Ref monitor 表示用の軸レベルを返す。warmup baseline は現行 guard と同じく WATCH に抑える。"""
        level = self._axis_level(axis, value)
        if self.baseline_kind == "warmup" and level > 1:
            return 1
        return level

    def get_ref_monitor_axis_statuses(self) -> tuple[dict, ...]:
        """Ref Stability Monitor 用の operator-facing 軸ステータスを返す。"""
        values = self._axis_values()
        result = []
        for axis in self._REF_MONITOR_AXES:
            if axis == "U":
                uneven_status = self._current_uneven_status()
                if uneven_status is not None:
                    item = uneven_status.as_monitor_item()
                    if self.baseline_kind == "warmup" and int(item.get("level") or 0) > 1:
                        item["status"] = self.DRIFT_WATCH
                        item["level"] = 1
                    item["watch"] = 1.0
                    item["warn"] = 1.0
                    item["recal"] = 1.0
                    item["history_kind"] = "risk"
                    if uneven_status.value_kind == "drift" and uneven_status.drift_dominant:
                        spec = _REF_UNEVEN_DRIFT_SPECS[uneven_status.drift_dominant]
                        item["watch"] = float(spec["watch"])
                        item["warn"] = float(spec["warn"])
                        item["recal"] = float(spec["recal"])
                        item["limit_value"] = uneven_status.drift_limit_pp
                        item["limit_label"] = (
                            "recal" if item.get("status") == self.DRIFT_RECAL else "limit"
                        )
                    elif uneven_status.abs_dominant:
                        spec = _REF_UNEVEN_ABS_SPECS[uneven_status.abs_dominant]
                        item["watch"] = float(spec["watch"])
                        item["warn"] = float(spec["warn"])
                        item["recal"] = float(spec["recal"])
                        item["limit_value"] = uneven_status.abs_limit_pct
                        item["limit_label"] = (
                            "recal" if item.get("status") == self.DRIFT_RECAL else "limit"
                        )
                    result.append(item)
                    continue
            value = float(values.get(axis, 0.0))
            level = self._monitor_axis_level(axis, value)
            warn = float(self._WARN_THRESHOLDS[axis])
            result.append(
                {
                    "axis": axis,
                    "label": self._REF_MONITOR_LABELS[axis],
                    "status": self._LEVEL_STATES[level],
                    "level": level,
                    "value": value,
                    "unit": self._REF_MONITOR_UNITS[axis],
                    "watch": float(self._WATCH_THRESHOLDS[axis]),
                    "warn": warn,
                    "recal": float(self._RECAL_THRESHOLDS[axis]),
                    "risk": value / max(warn, 1e-8),
                }
            )
        return tuple(result)

    def get_ref_monitor_series(self, axis: str | None = None) -> dict:
        """Ref Stability Monitor 用に選択軸とその履歴を返す。"""
        statuses = self.get_ref_monitor_axis_statuses()
        status_by_axis = {item["axis"]: item for item in statuses}
        selected_axis = str(axis or self.ref_monitor_axis_mode or "").upper()
        if selected_axis == "AUTO":
            selected_axis = ""
        if selected_axis not in status_by_axis:
            reason_axes = [
                part for part in str(self.drift_reason_axes or "").split("+")
                if part in status_by_axis
            ]
            if reason_axes:
                selected_axis = reason_axes[0]
            else:
                worst = max(statuses, key=lambda item: (int(item["level"]), float(item["risk"])))
                selected_axis = "I" if int(worst["level"]) == 0 else str(worst["axis"])
        selected = dict(status_by_axis[selected_axis])
        selected["history"] = self.get_ref_monitor_axis_history(selected_axis)
        selected["axis_statuses"] = statuses
        selected["mode"] = self.ref_monitor_axis_mode
        return selected

    def cycle_ref_monitor_axis(self, direction: int = 1) -> dict:
        """Ref Stability Monitor の表示軸モードを切り替え、現在 series を返す。"""
        try:
            current_idx = self._REF_MONITOR_AXIS_CYCLE.index(
                str(self.ref_monitor_axis_mode or "AUTO").upper()
            )
        except ValueError:
            current_idx = 0
        next_idx = (current_idx + int(direction)) % len(self._REF_MONITOR_AXIS_CYCLE)
        self.ref_monitor_axis_mode = self._REF_MONITOR_AXIS_CYCLE[next_idx]
        return self.get_ref_monitor_series()

    def set_ref_monitor_axis_mode(self, axis: str | None = None) -> dict:
        """Ref Stability Monitor の表示軸モードを直接設定し、現在 series を返す。"""
        mode = str(axis or "AUTO").upper()
        if mode not in self._REF_MONITOR_AXIS_CYCLE:
            mode = "AUTO"
        self.ref_monitor_axis_mode = mode
        return self.get_ref_monitor_series()

    def _instant_level_and_axes(self) -> tuple[int, list[str]]:
        values = self._axis_values()
        levels = {axis: self._axis_level(axis, value) for axis, value in values.items()}
        warn_axes = [axis for axis, level in levels.items() if level >= 2]
        max_level = max(levels.values()) if levels else 0
        if len(warn_axes) >= 2:
            max_level = max(max_level, 3)
        axes = [axis for axis, level in levels.items() if level == max_level and level > 0]
        if max_level == 3 and len(warn_axes) >= 2:
            axes = warn_axes
        if self.baseline_kind == "warmup" and max_level > 1:
            max_level = 1
            axes = [axis for axis, level in levels.items() if level > 0]
        return max_level, axes

    def _elapsed_or_frames_met(
        self,
        *,
        started_at: Optional[float],
        frames: int,
        timestamp: Optional[float],
        seconds: float,
        fallback_frames: int,
    ) -> bool:
        if started_at is not None and timestamp is not None and timestamp >= started_at:
            if timestamp - started_at >= seconds:
                return True
        return frames >= fallback_frames

    def _set_drift_level(self, level: int, axes: list[str]) -> None:
        level = int(np.clip(level, 0, 3))
        self.drift_state = self._LEVEL_STATES[level]
        self.drift_reason_axes = "+".join(axes[:3])
        if self.drift_state == self.DRIFT_RECAL:
            self.guard_state = self.STATE_RECALIB_REQUIRED
        elif self.drift_state == self.DRIFT_WARN:
            self.guard_state = self.STATE_LED_DRIFT_WARN
        else:
            self.guard_state = self.STATE_NORMAL
        self.guard_reason = self.drift_reason_axes or ("STABLE" if level == 0 else "")

    def _below_recover_threshold(self, target_level: int) -> bool:
        values = self._axis_values()
        if target_level <= 0:
            thresholds = self._WATCH_THRESHOLDS
        elif target_level == 1:
            thresholds = self._WARN_THRESHOLDS
        else:
            thresholds = self._RECAL_THRESHOLDS
        return all(values[axis] < thresholds[axis] * 0.8 for axis in values)

    def _update_guard_state(self, timestamp: Optional[float] = None) -> None:
        """偏差とヒステリシスから guard_state を更新する。"""
        dev_i = self.get_dev_I_pct()
        dev_u = self.get_dev_uniformity_pct()
        dev_c = self.get_dev_chromaticity()
        if dev_i is None or dev_u is None or dev_c is None:
            self.guard_state = self.STATE_NORMAL
            self.guard_reason = "WARMUP"
            self.drift_state = self.DRIFT_OK
            self.drift_reason_axes = ""
            return

        timestamp = self._finite_float(timestamp, time.monotonic())
        instant_level, axes = self._instant_level_and_axes()
        current_level = self._STATE_LEVELS.get(self.drift_state, 0)

        if self.drift_state == self.DRIFT_RECAL:
            self._set_drift_level(3, axes or self.drift_reason_axes.split("+"))
            return

        if instant_level > current_level:
            if self._candidate_level != instant_level:
                self._candidate_level = instant_level
                self._candidate_frames = 0
                self._candidate_started_at = timestamp
            self._candidate_frames += 1
            if self._elapsed_or_frames_met(
                started_at=self._candidate_started_at,
                frames=self._candidate_frames,
                timestamp=timestamp,
                seconds=self._PROMOTE_SEC,
                fallback_frames=self._PROMOTE_FRAMES,
            ):
                self._set_drift_level(instant_level, axes)
                self._candidate_level = 0
                self._candidate_frames = 0
                self._candidate_started_at = None
            return

        self._candidate_level = 0
        self._candidate_frames = 0
        self._candidate_started_at = None

        if instant_level < current_level and self._below_recover_threshold(instant_level):
            if self._recover_target_level != instant_level:
                self._recover_target_level = instant_level
                self._recover_frames = 0
                self._recover_started_at = timestamp
            self._recover_frames += 1
            if self._elapsed_or_frames_met(
                started_at=self._recover_started_at,
                frames=self._recover_frames,
                timestamp=timestamp,
                seconds=self._DEMOTE_SEC,
                fallback_frames=self._DEMOTE_FRAMES,
            ):
                self._set_drift_level(instant_level, axes)
                self._recover_frames = 0
                self._recover_started_at = None
            return

        self._recover_frames = 0
        self._recover_started_at = None
        if instant_level == current_level:
            self._set_drift_level(current_level, axes)

    def get_guard_state(self) -> str:
        """既存 key_handler 互換のガード状態を返す。"""
        return self.guard_state

    def get_guard_reason(self) -> str:
        """現在のガード理由を返す。"""
        return self.guard_reason

    def get_drift_state(self) -> str:
        """Ref Drift Guard の operator-facing 状態を返す。"""
        return self.drift_state

    def get_drift_reason_axes(self) -> str:
        return self.drift_reason_axes

    def get_baseline_label(self) -> str:
        label = self._source_label(self.baseline_source)
        if label == "WARMUP":
            return "Base WARMUP"
        if label == "UNVER":
            return "Base UNVER"
        suffix = self._format_baseline_time(self.baseline_timestamp)
        return f"Base {label} {suffix}".rstrip()

    def get_baseline_kind(self) -> str:
        return self.baseline_kind or "none"

    def get_drift_snapshot(self) -> dict:
        """HUD/log 用の Ref Drift Guard snapshot を返す。"""
        uneven_status = self._current_uneven_status()
        return {
            "drift_state": self.drift_state,
            "drift_reason_axes": self.drift_reason_axes,
            "baseline_label": self.get_baseline_label(),
            "baseline_source": self.baseline_source,
            "baseline_kind": self.get_baseline_kind(),
            "baseline_timestamp": self.baseline_timestamp,
            "raw_mean": None if self.count == 0 else float(self.raw_mean_buf[(self.index - 1) % self.buffer_size]),
            "raw_baseline": self.ref_raw_mean,
            "intensity_drift_pct": self.get_dev_I_pct(),
            "chroma_drift": self.get_dev_chromaticity(),
            "uniformity_drift_pct": self.get_dev_uniformity_pct(),
            "intensity_jitter_pct": self.get_intensity_jitter_pct(),
            "clip_high_pct": self.get_clip_high_pct(),
            "clip_low_pct": self.get_clip_low_pct(),
            "ref_uneven_status": None if uneven_status is None else uneven_status.as_monitor_item(),
        }

    def mark_recalibrated(
        self,
        source: str = "calib",
        timestamp: Optional[float] = None,
    ) -> None:
        """再キャリブ完了を通知し、ガード状態を即時解除する。"""
        if self.count > 0:
            idx = (self.index - 1) % self.buffer_size
            self.ref_raw_mean = float(self.raw_mean_buf[idx])
            self.ref_uniformity = float(self.uniformity_buf[idx])
            self.ref_chromaticity = self.chromaticity_buf[idx].copy()
            self.ref_clip_high_pct = float(self.clip_high_buf[idx])
            self.ref_clip_low_pct = float(self.clip_low_buf[idx])
            self.baseline_spatial_abs = self.spatial_abs_buf[idx]
            self.baseline_source = source or "calib"
            self.baseline_kind = "wnp" if self._source_label(source) != "WARMUP" else "warmup"
            self.baseline_timestamp = time.time() if timestamp is None else float(timestamp)
        self._recalib_counter = 0
        self._led_drift_counter = 0
        self._recover_counter = 0
        self._candidate_level = 0
        self._candidate_frames = 0
        self._candidate_started_at = None
        self._recover_target_level = 0
        self._recover_frames = 0
        self._recover_started_at = None
        self.drift_state = self.DRIFT_OK
        self.drift_reason_axes = ""
        self.guard_state = self.STATE_NORMAL
        self.guard_reason = "RECALIB_DONE"

    def is_uniformity_alert(self) -> bool:
        """center-edge差が閾値を超えたか。"""
        if self.count == 0:
            return False
        curr = float(self.uniformity_buf[(self.index - 1) % self.buffer_size])
        return curr > self.uniformity_threshold_pct


class SpectralDriftTracker:
    """
    Phase 0 動的補正係数 f_R, f_B の経時変化を記録するトラッカー。

    機能:
      - ニュートラルキャリブレーション（n キー）後からの f_R, f_B を全量蓄積する。
      - リングバッファではなくセッション全区間を保持し、フル X 軸グラフを実現する。
      - 再キャリブ時（clear 操作）にリセットされる。
    入力:
      - CIELABConverter._dynamic_correction から毎フレーム取得した f_R, f_B。
    出力:
      - get_plot_data() でダウンサンプリング済み描画用配列を返す。
    """

    _CHUNK = 5000  # numpy 配列の1回増量サイズ

    def __init__(self, max_samples: int = 500_000):
        """
        Args:
          max_samples: 蓄積上限サンプル数（メモリ過大防止）。
        """
        self.max_samples = max_samples
        self._capacity = self._CHUNK
        self._fR = np.zeros(self._CHUNK, dtype=np.float32)
        self._fB = np.zeros(self._CHUNK, dtype=np.float32)
        self._times = np.zeros(self._CHUNK, dtype=np.float32)
        self._n: int = 0
        self._start_time: float | None = None
        self.is_active: bool = False  # activate() が呼ばれた後のみ push を受け付ける
        # Y軸レンジ記憶（一度広がったら縮まない）
        self._y_min_seen: float = 0.95
        self._y_max_seen: float = 1.05

    def reset(self) -> None:
        """バッファをクリアして非アクティブ状態に戻す。Y軸レンジもデフォルトに戻す。"""
        self._n = 0
        self._start_time = None
        self.is_active = False
        self._y_min_seen = 0.95
        self._y_max_seen = 1.05

    def activate(self) -> None:
        """ニュートラルキャリブ完了時に呼び出す。バッファをリセットして記録を開始する。"""
        self.reset()
        self.is_active = True

    def push(self, f_R: float, f_B: float) -> None:
        """f_R, f_B を経過時間とともに記録する。

        Args:
          f_R: R チャンネルの動的補正係数。
          f_B: B チャンネルの動的補正係数。
        """
        if not self.is_active:
            return
        now = time.time()
        if self._start_time is None:
            self._start_time = now
        if self._n >= self.max_samples:
            return
        if self._n >= self._capacity:
            add = min(self._CHUNK, self.max_samples - self._capacity)
            self._fR = np.concatenate([self._fR, np.zeros(add, dtype=np.float32)])
            self._fB = np.concatenate([self._fB, np.zeros(add, dtype=np.float32)])
            self._times = np.concatenate([self._times, np.zeros(add, dtype=np.float32)])
            self._capacity += add
        self._fR[self._n] = f_R
        self._fB[self._n] = f_B
        self._times[self._n] = now - self._start_time
        self._n += 1

    def get_plot_data(self, max_points: int = 250):
        """描画用にダウンサンプリングした (times, fR, fB) を返す。

        Args:
          max_points: 最大プロット点数（グラフ幅 px に合わせる）。
        Returns:
          tuple[np.ndarray, np.ndarray, np.ndarray]: 経過秒・fR・fB の配列。
        """
        n = self._n
        if n == 0:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty
        times = self._times[:n]
        fR = self._fR[:n]
        fB = self._fB[:n]
        if n > max_points:
            idx = np.round(np.linspace(0, n - 1, max_points)).astype(np.int32)
            return times[idx], fR[idx], fB[idx]
        return times, fR, fB

    def get_current(self) -> tuple:
        """最新の (f_R, f_B) を返す。データなしなら (1.0, 1.0)。"""
        if self._n == 0:
            return (1.0, 1.0)
        return (float(self._fR[self._n - 1]), float(self._fB[self._n - 1]))

    def get_elapsed_seconds(self) -> float:
        """セッション開始からの経過秒を返す。"""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def update_y_range(self, v_min: float, v_max: float) -> None:
        """データの min/max で Y軸レンジを広げる（縮小はしない）。

        Args:
          v_min: 現フレームのデータ最小値。
          v_max: 現フレームのデータ最大値。
        """
        if v_min < self._y_min_seen:
            self._y_min_seen = v_min
        if v_max > self._y_max_seen:
            self._y_max_seen = v_max

    def get_y_range(self) -> tuple:
        """現在の Y軸レンジ (y_min, y_max) を返す。"""
        return (self._y_min_seen, self._y_max_seen)

    @property
    def count(self) -> int:
        """蓄積サンプル数を返す。"""
        return self._n


# ===========================================================================
# ROI操作
# ===========================================================================


class ROIMouseHandler:
    """
    ROI移動・サイズ変更のマウス入力を処理するクラス。

    機能:
      - マウス操作でRef/Target ROIの移動、サイズ変更、保存トリガーを管理する。
    入力:
      - マウスイベント、カーソル座標、ROI設定、表示幅情報、保存ハンドラ。
    出力:
      - 更新済みROI座標/サイズと必要に応じた永続化処理を反映する。
    """

    def __init__(self, config, persistence=None):
        self.config = config
        self.persistence = persistence
        self.dragging = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.active_roi = "ref"
        self.resize_axis = "w"
        self._on_ref_roi_moved: "Callable[[], None] | None" = None
        self._on_roi_changed: "Callable[[], None] | None" = None

    def _trigger_save(self):
        """ROI変更時に永続化を実行する。"""
        if self.persistence is not None:
            self.persistence.save()

    def _current_roi_dims(self):
        ref_w = max(int(self.config.processing.spot_size_ref), 2)
        ref_h = max(int(ref_w * self.config.processing.aspect_ref), 2)
        tar_w = max(int(self.config.processing.spot_size_tar), 2)
        tar_h = max(int(tar_w * self.config.processing.aspect_tar), 2)
        return ref_w, ref_h, tar_w, tar_h

    @staticmethod
    def _is_inside_roi(mx, my, pos, w, h):
        return pos[0] <= mx <= pos[0] + w and pos[1] <= my <= pos[1] + h

    def _clamp_pos(self, pos, w, h):
        max_x = max(self.config.display.width - w, 0)
        max_y = max(self.config.display.height - h, 0)
        pos[0] = max(0, min(int(pos[0]), max_x))
        pos[1] = max(0, min(int(pos[1]), max_y))

    def _resize_roi_by_key(self, key):
        """j/lキーでactive_roiの幅または高さを変更する（kで軸切替）。"""
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

        if self.persistence is not None:
            self.persistence.save()

    def callback(self, event, x, y, flags, param):
        # マウス座標を左パネル分オフセットしてカメラ座標に変換
        ox = getattr(self.config, "camera", None)
        if ox is not None:
            ox = getattr(ox, "left_panel_width", 0)
        else:
            ox = 0
        cam_x = x - ox
        cam_w = self.config.display.width
        if cam_x < 0 or cam_x >= cam_w:
            return  # カメラ領域外のクリックは無視
        ref_w, ref_h, tar_w, tar_h = self._current_roi_dims()

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._is_inside_roi(
                cam_x, y, self.config.processing.posi_ref, ref_w, ref_h
            ):
                self.dragging = "ref"
                self.drag_offset_x = cam_x - self.config.processing.posi_ref[0]
                self.drag_offset_y = y - self.config.processing.posi_ref[1]
            elif self._is_inside_roi(
                cam_x, y, self.config.processing.posi_tar, tar_w, tar_h
            ):
                self.dragging = "tar"
                self.drag_offset_x = cam_x - self.config.processing.posi_tar[0]
                self.drag_offset_y = y - self.config.processing.posi_tar[1]

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            if self.dragging == "ref":
                self.config.processing.posi_ref[0] = cam_x - self.drag_offset_x
                self.config.processing.posi_ref[1] = y - self.drag_offset_y
                self._clamp_pos(self.config.processing.posi_ref, ref_w, ref_h)
            elif self.dragging == "tar":
                self.config.processing.posi_tar[0] = cam_x - self.drag_offset_x
                self.config.processing.posi_tar[1] = y - self.drag_offset_y
                self._clamp_pos(self.config.processing.posi_tar, tar_w, tar_h)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging is not None:
                self._trigger_save()
                if self._on_roi_changed is not None:
                    self._on_roi_changed()
                if self.dragging == "ref" and self._on_ref_roi_moved is not None:
                    self._on_ref_roi_moved()
            self.dragging = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 10 if flags > 0 else -10
            cx_ref = self.config.processing.posi_ref[0] + ref_w // 2
            cy_ref = self.config.processing.posi_ref[1] + ref_h // 2
            cx_tar = self.config.processing.posi_tar[0] + tar_w // 2
            cy_tar = self.config.processing.posi_tar[1] + tar_h // 2
            dist_ref = math.hypot(cam_x - cx_ref, y - cy_ref)
            dist_tar = math.hypot(cam_x - cx_tar, y - cy_tar)

            if dist_ref <= dist_tar:
                self.active_roi = "ref"
            else:
                self.active_roi = "tar"
            self._resize_roi_by_key(ord("l") if delta > 0 else ord("j"))
            ref_w, ref_h, tar_w, tar_h = self._current_roi_dims()
            if self.active_roi == "ref":
                self._clamp_pos(self.config.processing.posi_ref, ref_w, ref_h)
            else:
                self._clamp_pos(self.config.processing.posi_tar, tar_w, tar_h)
            self._trigger_save()
            if self._on_roi_changed is not None:
                self._on_roi_changed()


class ROIConfigPersistence:
    """
    ROI位置・サイズの永続化を担当するクラス。

    機能:
      - ROI設定をJSONへ保存し、次回起動時に復元する。
    入力:
      - 設定オブジェクト、保存パス、保存済みJSONデータ。
    出力:
      - ファイル保存/復元結果と設定オブジェクトへの反映状態を返す。
    """

    DEFAULT_PATH = os.path.join(CALIBRATION_DIR, "roi_config.json")

    def __init__(self, config, path: str = None):
        """設定オブジェクトと保存パスを保持する。"""
        self.config = config
        self.path = path or self.DEFAULT_PATH

    def load(self) -> bool:
        """
        JSONからROI設定を読み込みconfig.processingに適用する。

        Returns:
          bool: 読み込み成功時True。
        """
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            p = self.config.processing
            if "ref_pos" in data:
                p.posi_ref = list(data["ref_pos"])
            if "ref_spot_size" in data:
                p.spot_size_ref = int(data["ref_spot_size"])
            if "ref_aspect" in data:
                p.aspect_ref = float(data["ref_aspect"])
            if "tar_pos" in data:
                p.posi_tar = list(data["tar_pos"])
            if "tar_spot_size" in data:
                p.spot_size_tar = int(data["tar_spot_size"])
            if "tar_aspect" in data:
                p.aspect_tar = float(data["tar_aspect"])
            print(f"- ROI config loaded: {self.path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load ROI config: {e}")
            return False

    def save(self) -> None:
        """現在のROI設定をJSONに書き出す。"""
        p = self.config.processing
        data = {
            "ref_pos": list(p.posi_ref),
            "ref_spot_size": int(p.spot_size_ref),
            "ref_aspect": float(p.aspect_ref),
            "tar_pos": list(p.posi_tar),
            "tar_spot_size": int(p.spot_size_tar),
            "tar_aspect": float(p.aspect_tar),
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save ROI config: {e}")


# ===========================================================================
# ブランク測定（ゼロ点補正）
# ===========================================================================


class BlankRatioManager:
    """
    ブランク測定（ゼロ点補正）の取得・保存・適用を行うクラス。

    機能:
      - 試料なし状態での Tar/Ref 比率を記録し、空間的照明不均一をキャンセルする。
    入力:
      - ブランク状態で算出した ratio 配列 [R, G, B]。
    出力:
      - ブランク比で除算した補正済み ratio 配列を返す。
    """

    def __init__(self, save_path: str = ""):
        """
        保存先パスと内部状態を初期化する。

        Args:
          save_path: ブランク比の保存先JSONファイルパス。空文字列ならデフォルト。
        """
        self.save_path = save_path or os.path.join(
            get_today_calibration_dir(), "blank_ratio.json"
        )
        self.blank_ratio = None
        self.is_loaded = False

    def load_if_exists(self) -> bool:
        """
        既存のブランク比を読み込む。最新の日付フォルダを優先して検索する。

        Returns:
          bool: 読み込み成功時True。
        """
        path = _find_calibration_file("blank_ratio.json")
        if path is None:
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.blank_ratio = np.array(
                [data["R"], data["G"], data["B"]], dtype=np.float64
            )
            self.is_loaded = True
            print(
                f"- ブランク比を読み込み: R={data['R']:.4f}"
                f" G={data['G']:.4f} B={data['B']:.4f}"
            )
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: ブランク比の読み込みに失敗: {e}")
            return False

    def save(self, ratio: np.ndarray) -> None:
        """
        ブランク比を保存する。

        Args:
          ratio: (3,) [R, G, B] のブランク比配列。
        """
        self.blank_ratio = ratio.astype(np.float64)
        self.is_loaded = True
        data = {
            "R": float(ratio[0]),
            "G": float(ratio[1]),
            "B": float(ratio[2]),
            "created": datetime.now().isoformat(),
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        _remove_cleared_marker("blank_ratio.json")
        # バックアップ
        import shutil
        import glob

        date_str = datetime.now().strftime("%Y-%m-%d")
        backup_dir = os.path.dirname(self.save_path)
        backup_path = os.path.join(backup_dir, f"blank_ratio_{date_str}.json")
        shutil.copy2(self.save_path, backup_path)
        # 古いバックアップの自動削除（5世代保持）
        pattern = os.path.join(backup_dir, "blank_ratio_*.json")
        backups = sorted(glob.glob(pattern), reverse=True)
        for old in backups[5:]:
            os.remove(old)
        print(
            f"- ブランク比を保存: R={data['R']:.4f}"
            f" G={data['G']:.4f} B={data['B']:.4f}"
        )

    def correct(self, ratio: np.ndarray) -> np.ndarray:
        """
        ブランク比で除算して空間的照明不均一を補正する。

        Args:
          ratio: (3,) [R, G, B] の測定比配列。
        Returns:
          np.ndarray: 補正済み ratio 配列。
        """
        if self.blank_ratio is None:
            return ratio
        safe_blank = np.where(self.blank_ratio < 1e-8, 1e-8, self.blank_ratio)
        return ratio / safe_blank

    def clear(self) -> None:
        """ブランク比を無効化しファイルを削除する。"""
        self.blank_ratio = None
        self.is_loaded = False
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            print(f"- ブランク比を削除: {self.save_path}")


# ===========================================================================
# マスターRef正規化
# ===========================================================================


class MasterRefManager:
    """
    マスターRef値を保存し、セッション間の照明ドリフトを吸収するクラス。

    機能:
      - キャリブレーション時の Ref RGB 平均値をマスター基準として保存する。
      - 毎フレームの Ref RGB とマスター値の比をスケール係数として返す。
    入力:
      - キャリブレーション時の Ref RGB 平均値、毎フレームの Ref RGB 平均値。
    出力:
      - スケール係数 (3,) [R, G, B] を返す。
    """

    def __init__(self, save_path: str = ""):
        """
        保存先パスと内部状態を初期化する。

        Args:
          save_path: マスターRef値の保存先JSONファイルパス。空文字列ならデフォルト。
        """
        self.save_path = save_path or os.path.join(
            get_today_calibration_dir(), "master_ref.json"
        )
        self.master_ref_rgb = None
        self.is_loaded = False

    def load_if_exists(self) -> bool:
        """
        既存のマスターRef値を読み込む。最新の日付フォルダを優先して検索する。

        ロード後にバリデーションを行い、ゼロ・NaN・inf を含む無効データは
        拒否して is_loaded=False のまま返す。

        Returns:
          bool: 読み込み成功かつ有効値の場合True。
        """
        path = _find_calibration_file("master_ref.json")
        if path is None:
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = np.array(
                [data["R"], data["G"], data["B"]], dtype=np.float64
            )
            # 無効値ガード: NaN / inf / ゼロ近傍は拒否する
            if not np.all(np.isfinite(loaded)):
                print(
                    f"Warning: マスターRef値に NaN/inf が含まれるため拒否: {loaded}"
                )
                return False
            if np.any(np.abs(loaded) < 1e-6):
                print(
                    f"Warning: マスターRef値がゼロに近いため拒否"
                    f" (再キャリブレーションが必要): {loaded}"
                    f" → 12bit換算: {loaded * 4095}"
                )
                return False
            self.master_ref_rgb = loaded
            self.is_loaded = True
            # 表示: 0~1正規化値を 12bit カウント換算で表示
            print(
                f"- マスターRef値を読み込み:"
                f" R={data['R']:.4f}({data['R']*4095:.0f}cnt)"
                f" G={data['G']:.4f}({data['G']*4095:.0f}cnt)"
                f" B={data['B']:.4f}({data['B']*4095:.0f}cnt)"
            )
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: マスターRef値の読み込みに失敗: {e}")
            return False

    def save(self, ref_rgb: np.ndarray) -> None:
        """
        マスターRef値を保存する。

        無効値（ゼロ・NaN・inf）は保存を拒否しエラーログを出す。
        保存前バリデーションにより、後続の compute_scale での
        scale=0 → NaN 伝播を根本的に防止する。

        Args:
          ref_rgb: (3,) [R, G, B] のRef RGB平均値（0~1正規化）。
        """
        rgb = np.asarray(ref_rgb, dtype=np.float64).ravel()[:3]
        # 無効値ガード: NaN / inf / ゼロ近傍は保存しない
        if not np.all(np.isfinite(rgb)):
            print(
                f"ERROR: マスターRef値に NaN/inf が含まれるため保存をスキップ: {rgb}"
            )
            return
        if np.any(np.abs(rgb) < 1e-6):
            print(
                f"ERROR: マスターRef値がゼロに近いため保存をスキップ"
                f" (Ref ROI が暗点を示している可能性): {rgb}"
                f" → 12bit換算: {rgb * 4095}"
            )
            return
        self.master_ref_rgb = rgb
        self.is_loaded = True
        data = {"R": float(rgb[0]), "G": float(rgb[1]), "B": float(rgb[2])}
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        _remove_cleared_marker("master_ref.json")
        # 表示: 0~1正規化値を 12bit カウント換算で表示（:.1f では 0.0x が "0.0" に見える問題を回避）
        print(
            f"- マスターRef値を保存:"
            f" R={data['R']:.4f}({data['R']*4095:.0f}cnt)"
            f" G={data['G']:.4f}({data['G']*4095:.0f}cnt)"
            f" B={data['B']:.4f}({data['B']*4095:.0f}cnt)"
        )

    def compute_scale(self, current_ref_rgb: np.ndarray) -> np.ndarray:
        """
        マスターRefとのスケール比を算出する。

        Args:
          current_ref_rgb: (3,) [R, G, B] の現在のRef RGB平均値。
        Returns:
          np.ndarray: (3,) スケール係数。
        """
        if self.master_ref_rgb is None:
            return np.ones(3, dtype=np.float64)
        # master_ref_rgb がゼロまたは NaN の場合: 破滅的な NaN 伝播を防ぐため
        # そのチャンネルのスケールを 1.0 にフォールバックする。
        # （ゼロ master_ref → scale=0 → ratio/0=NaN が根本原因）
        safe_master = np.where(
            np.isfinite(self.master_ref_rgb) & (np.abs(self.master_ref_rgb) >= 1e-6),
            self.master_ref_rgb,
            np.ones(3, dtype=np.float64),
        )
        safe_current = np.where(
            np.abs(current_ref_rgb) < 1e-8, 1e-8, current_ref_rgb
        )
        return safe_master / safe_current

    def clear(self) -> None:
        """マスターRef値を無効化しファイルを削除する。"""
        self.master_ref_rgb = None
        self.is_loaded = False
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            print(f"- マスターRef値を削除: {self.save_path}")
