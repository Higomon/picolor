"""
CSIオーバーレイ描画モジュール。

サイドパネルUI（測定値・ヒストグラム・安定性グラフ・警告）の描画を提供する。
"""

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401 - ImageFont kept for import compatibility

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from .colorimeter_common import (
    _ascii_to_fullwidth,
    _find_japanese_font,
    StabilityMonitor,
)


_ASCII_SAFE_MAP = {
    "ΔE": "dE",
    "最低合格": "MINIMUM",
    "要再判定": "RECHECK",
    "失格": "REJECTED",
    "合格": "ACCEPTED",
}
_ASCII_SAFE_KEYS_SORTED = sorted(
    _ASCII_SAFE_MAP.keys(), key=len, reverse=True,
)


def _to_ascii_safe(text: str) -> str:
    """日本語・特殊文字を ASCII 代替に変換する。cv2.putText フォールバック用。

    変換マップのキーを文字列長降順でソートし str.replace() を適用することで、
    ``"最低合格"`` が ``"合格"`` より先に処理され部分置換を防ぐ。
    """
    result = text
    for key in _ASCII_SAFE_KEYS_SORTED:
        result = result.replace(key, _ASCII_SAFE_MAP[key])
    return result


@dataclass(frozen=True)
class LeftPanelFlowStep:
    label: str
    is_done: bool
    phase: int


@dataclass(frozen=True)
class LeftPanelShortcutSection:
    title: "str | None"
    ascii_title: "str | None"
    items: "tuple[tuple[str, str], ...]"


@dataclass(frozen=True)
class LeftPanelContent:
    current_phase: int
    flow_header: str
    flow_header_ascii: str
    keys_header: str
    keys_header_ascii: str
    rules_header: str
    rules_header_ascii: str
    version_header: str
    version_header_ascii: str
    flow_steps: "tuple[LeftPanelFlowStep, ...]"
    shortcut_sections: "tuple[LeftPanelShortcutSection, ...]"
    operation_rules: "tuple[str, ...]"
    operation_rules_ascii: "tuple[str, ...]"
    gray_summary_text: "str | None"
    gray_summary_color: "tuple[int, int, int]"
    version_lines: "tuple[str, ...]"


@dataclass(frozen=True)
class LeftPanelLayout:
    width: int
    height: int
    circle_x: int
    step_start_y: int
    step_gap: int
    circle_radius: int
    key_header_y: int
    shortcut_start_gap: int
    shortcut_section_top_gap: int
    shortcut_section_title_gap: int
    shortcut_item_gap: int
    rules_header_gap: int
    rules_line_gap: int
    version_line_gap: int
    version_bottom_margin: int
    step_label_x: int
    step_label_max_width: int
    shortcut_desc_x: int
    shortcut_desc_max_width: int
    rule_max_width: int
    version_header_y: int
    version_value_y: int
    show_version: bool


class CSIOverlayRenderer:
    """
    CSI向けのサイドパネルUIを描画するクラス。

    機能:
      - 左フローパネル/中央カメラ/右パネルからなるUIを合成描画する。
      - 測定値（Lab/ΔE）、露光情報、安定性時系列、品質警告を右パネルへ描画する。
    入力:
      - `CSIConfig`、カメラフレーム、Lab/ΔE、ROI座標・サイズ、安定性状態。
      - 品質警告リスト、時系列トラッカー、Ref/TarのROI画像データ。
    出力:
      - 描画済みの合成フレーム、左パネルキャッシュ画像を生成する。
    """

    HIST_BINS = 32
    HIST_RANGE_12BIT = (0, 4096)

    def __init__(self, config):
        """
        フォント定数とUIパネルの色・幅を初期化する。

        Args:
          config: パネル寸法・レイアウト値を保持するCSI設定。
        """
        self.config = config
        self.ly = config.panel_layout  # PanelLayoutSettings ショートカット
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.LINE_TYPE = cv2.LINE_AA
        self.PANEL_WIDTH = 280
        self.LEFT_PANEL_WIDTH = config.camera.left_panel_width
        self.PANEL_BG = (30, 30, 30)
        self._jp_font = _find_japanese_font(14)
        self._jp_font_small = _find_japanese_font(12)
        self._left_panel_cache = None
        self._left_panel_cache_key = None

    @staticmethod
    def _extract_status_keyword(status_text: "str | None") -> "str | None":
        """状態文字列から主要ステータス語を抽出する。"""
        if not status_text:
            return None
        upper = str(status_text).upper()
        for token in ("STALE", "FAIL", "WARN", "GOOD", "PASS", "OK"):
            if token in upper:
                return token
        return None

    def _build_gray_summary_text(
        self,
        gray_verify_status: "str | None",
        gray_absolute_status: "str | None",
    ) -> "tuple[str | None, tuple[int, int, int]]":
        """左パネル用の4D結果サマリを1行ASCIIで返す。"""
        rel = self._extract_status_keyword(gray_verify_status)
        abs_status = self._extract_status_keyword(gray_absolute_status)
        parts = []
        if rel is not None:
            parts.append(f"rel {rel}")
        if abs_status is not None:
            parts.append(f"abs {abs_status}")
        if not parts:
            return None, (160, 160, 160)

        color = (0, 200, 0)
        if abs_status == "STALE" or rel == "STALE":
            color = (0, 170, 220)
        elif abs_status == "FAIL" or rel == "FAIL":
            color = (0, 0, 200)
        elif abs_status == "WARN" or rel == "WARN":
            color = (0, 200, 220)
        return f"4D {' / '.join(parts)}", color

    @staticmethod
    def _left_panel_shortcuts() -> "list[tuple[str, str]]":
        """左パネルの操作キー一覧を返す。"""
        return [
            ("m", "記録開始"),
            ("s", "記録停止"),
            ("V", "Gray Check"),
            ("c", "ゼロリセット"),
            ("r/t", "参照/測定を選択"),
            ("k", "幅/高さ切替"),
            ("j/l", "縮小/拡大"),
            ("i", "Ref表示切替"),
            ("TAB", "表示切替"),
            ("q", "終了"),
        ]

    @staticmethod
    def _left_panel_version_label(software_version: "str | None") -> str:
        """左パネル下部に表示する software_version を返す。"""
        text = str(software_version or "").strip()
        return text or "unknown"

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
        transform=None,
        ellipsis: str = "...",
    ) -> str:
        """PIL 描画用にテキストを最大幅へ収める。"""
        if max_width_px <= 0:
            return ""
        transform_fn = transform or (lambda value: value)
        if cls._measure_text_width_pil(transform_fn(text), font) <= max_width_px:
            return text
        if cls._measure_text_width_pil(transform_fn(ellipsis), font) > max_width_px:
            return ""
        clipped = text
        while clipped:
            clipped = clipped[:-1].rstrip()
            candidate = f"{clipped}{ellipsis}" if clipped else ellipsis
            if cls._measure_text_width_pil(transform_fn(candidate), font) <= max_width_px:
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
        if max_width_px <= 0:
            return ""

        def _text_width(value: str) -> int:
            return cv2.getTextSize(value, font, font_scale, thickness)[0][0]

        if _text_width(text) <= max_width_px:
            return text
        if _text_width(ellipsis) > max_width_px:
            return ""
        clipped = text
        while clipped:
            clipped = clipped[:-1].rstrip()
            candidate = f"{clipped}{ellipsis}" if clipped else ellipsis
            if _text_width(candidate) <= max_width_px:
                return candidate
        return ellipsis

    @classmethod
    def _wrap_text_to_two_lines_pil(
        cls,
        text: str,
        *,
        max_width_px: int,
        font,
        transform=None,
    ) -> "list[str]":
        """PIL 描画向けに文字列を最大 2 行へ収める。"""
        if not text:
            return [""]
        transform_fn = transform or (lambda value: value)
        if cls._measure_text_width_pil(transform_fn(text), font) <= max_width_px:
            return [text]

        split_idx = len(text)
        while split_idx > 0 and cls._measure_text_width_pil(
            transform_fn(text[:split_idx]), font,
        ) > max_width_px:
            split_idx -= 1
        if split_idx <= 0:
            return [cls._fit_text_to_width_pil(
                text,
                max_width_px=max_width_px,
                font=font,
                transform=transform_fn,
            )]

        preferred_break = max((text.rfind(ch, 1, split_idx + 1) for ch in ("_", "-", ":")), default=-1)
        if preferred_break >= max(1, split_idx // 2):
            split_idx = preferred_break + 1

        first_line = text[:split_idx].rstrip()
        remainder = text[split_idx:].lstrip()
        if not remainder:
            return [first_line]
        if cls._measure_text_width_pil(transform_fn(remainder), font) <= max_width_px:
            return [first_line, remainder]
        return [
            first_line,
            cls._fit_text_to_width_pil(
                remainder,
                max_width_px=max_width_px,
                font=font,
                transform=transform_fn,
            ),
        ]

    @classmethod
    def _wrap_text_to_two_lines_cv2(
        cls,
        text: str,
        *,
        max_width_px: int,
        font,
        font_scale: float,
        thickness: int = 1,
    ) -> "list[str]":
        """cv2 描画向けに文字列を最大 2 行へ収める。"""
        if not text:
            return [""]

        def _text_width(value: str) -> int:
            return cv2.getTextSize(value, font, font_scale, thickness)[0][0]

        if _text_width(text) <= max_width_px:
            return [text]

        split_idx = len(text)
        while split_idx > 0 and _text_width(text[:split_idx]) > max_width_px:
            split_idx -= 1
        if split_idx <= 0:
            return [cls._fit_text_to_width_cv2(
                text,
                max_width_px=max_width_px,
                font=font,
                font_scale=font_scale,
                thickness=thickness,
            )]

        preferred_break = max((text.rfind(ch, 1, split_idx + 1) for ch in ("_", "-", ":")), default=-1)
        if preferred_break >= max(1, split_idx // 2):
            split_idx = preferred_break + 1

        first_line = text[:split_idx].rstrip()
        remainder = text[split_idx:].lstrip()
        if not remainder:
            return [first_line]
        if _text_width(remainder) <= max_width_px:
            return [first_line, remainder]
        return [
            first_line,
            cls._fit_text_to_width_cv2(
                remainder,
                max_width_px=max_width_px,
                font=font,
                font_scale=font_scale,
                thickness=thickness,
            ),
        ]

    def _build_left_panel_version_lines(
        self,
        software_version: "str | None",
        *,
        max_width_px: int,
    ) -> "list[str]":
        """左パネル下部に出す software_version 行群を返す。"""
        label = self._left_panel_version_label(software_version)
        compact_lines = self._compact_deploy_version_lines(label)
        if compact_lines is not None:
            return compact_lines
        if _PIL_AVAILABLE and self._jp_font_small is not None:
            return self._wrap_text_to_two_lines_pil(
                label,
                max_width_px=max_width_px,
                font=self._jp_font_small,
                transform=_ascii_to_fullwidth,
            )
        return self._wrap_text_to_two_lines_cv2(
            label,
            max_width_px=max_width_px,
            font=self.FONT,
            font_scale=0.28,
            thickness=1,
        )

    @staticmethod
    def _compact_deploy_version_lines(label: str) -> "list[str] | None":
        """`deploy:YYYY-MM-DD_HH:MM:SS` を左パネル向けの短い2行にする。"""
        prefix = "deploy:"
        if not label.startswith(prefix):
            return None
        payload = label[len(prefix) :].strip()
        if "_" not in payload:
            return None
        date_part, time_part = payload.split("_", 1)
        if len(date_part) != 10 or len(time_part) < 5:
            return None
        return ["deploy", f"{date_part[5:]} {time_part[:5]}"]

    @staticmethod
    def _format_verified_time(verified_at: "str | None") -> str:
        """ISO時刻をUI向けの短い文字列へ整形する。"""
        if not verified_at:
            return "--:--:--"
        text = str(verified_at).strip()
        if "T" in text:
            _, time_part = text.split("T", 1)
        elif " " in text:
            _, time_part = text.split(" ", 1)
        else:
            return text[:8]
        time_part = time_part.split("+", 1)[0]
        time_part = time_part.split("Z", 1)[0]
        return time_part[:8] if len(time_part) >= 8 else time_part

    @staticmethod
    def _gray_absolute_status_color(
        gray_absolute_result: "dict | None",
    ) -> "tuple[int, int, int]":
        """4D絶対結果の描画色を返す。"""
        if gray_absolute_result is None:
            return (160, 160, 160)
        status = str(gray_absolute_result.get("status", "--"))
        is_stale = bool(gray_absolute_result.get("is_stale"))
        if is_stale:
            return (0, 170, 220)
        if status == "FAIL":
            return (0, 0, 255)
        if status == "WARN":
            return (0, 220, 220)
        return (0, 200, 0)

    def _draw_right_panel_info_pil(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        lines_with_colors: "list[tuple[str, tuple[int,int,int], float]]",
    ) -> int:
        """右パネルの情報テキストを PIL (または cv2 フォールバック) で描画する。

        Parameters:
            frame: 合成フレーム (BGR, H×W×3)。直接書き換える。
            x: 描画開始 x 座標。
            y: 描画開始 y 座標（テキスト上端）。
            lines_with_colors: ``[(text, color_bgr, font_scale), ...]``。

        Returns:
            描画後の y 座標（次行の開始位置）。
        """
        if not lines_with_colors:
            return y

        use_pil = _PIL_AVAILABLE and self._jp_font is not None

        if use_pil:
            # --- PIL パス ---
            h_frame, w_frame = frame.shape[:2]
            x0 = max(x - 4, 0)
            x1 = w_frame

            # 各行の高さを実測して ROI 高さを決定
            line_heights = []
            for text, _color, font_scale in lines_with_colors:
                font = self._jp_font if font_scale >= 0.34 else self._jp_font_small
                try:
                    bbox = font.getbbox(text)
                    lh = bbox[3] - bbox[1] + 4  # 上下マージン
                except Exception:
                    lh = 16
                line_heights.append(max(lh, 12))

            total_h = sum(line_heights) + 4
            y0 = max(y - 2, 0)
            y1 = min(y0 + total_h, h_frame)
            if y1 <= y0 or x1 <= x0:
                return y + total_h

            roi = frame[y0:y1, x0:x1].copy()
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            draw_y = 2  # ROI 内相対座標
            for i, (text, color_bgr, font_scale) in enumerate(lines_with_colors):
                font = self._jp_font if font_scale >= 0.34 else self._jp_font_small
                # BGR → RGB for PIL
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                draw.text((x - x0, draw_y), _ascii_to_fullwidth(_to_ascii_safe(text)), font=font, fill=color_rgb)
                draw_y += line_heights[i]

            frame[y0:y1, x0:x1] = cv2.cvtColor(
                np.array(pil_img), cv2.COLOR_RGB2BGR,
            )
            return y0 + total_h
        else:
            # --- cv2 フォールバック (ASCII 変換) ---
            current_y = y
            for text, color_bgr, font_scale in lines_with_colors:
                safe_text = _to_ascii_safe(text)
                cv2.putText(
                    frame, safe_text, (x, current_y),
                    self.FONT, font_scale, color_bgr, 1, self.LINE_TYPE,
                )
                current_y += 14
            return current_y

    def _build_gray_absolute_detail_lines(
        self,
        gray_absolute_result: "dict | None",
    ) -> "list[str]":
        """右パネルの4D絶対補助表示は通常HUDでは出さない。"""
        del gray_absolute_result
        return []

    @staticmethod
    def _acceptance_status_color(
        acceptance_result: "dict | None",
    ) -> "tuple[int, int, int]":
        """acceptance_result の描画色を返す。"""
        if acceptance_result is None:
            return (160, 160, 160)
        if acceptance_result.get("is_stale") or not acceptance_result.get("is_current", False):
            return (0, 170, 220)
        gate_state = str(acceptance_result.get("gate_state", "blocked"))
        result_status = str(acceptance_result.get("result_status", ""))
        if gate_state == "blocked" or result_status in ("失格", "要再判定"):
            return (0, 0, 255)
        if gate_state == "provisional" or result_status == "最低合格":
            return (0, 220, 220)
        return (0, 200, 0)

    def _build_acceptance_detail_lines(
        self,
        acceptance_result: "dict | None",
    ) -> "list[str]":
        """右パネルの acceptance 表示は廃止。左パネルに集約。常に空リストを返す。"""
        return []

    def _build_workflow_detail_lines(
        self,
        workflow_status: "dict | None",
    ) -> "list[str]":
        """右パネルの workflow 補助表示を返す。"""
        if not workflow_status or not workflow_status.get("active"):
            return []
        message = str(workflow_status.get("message", "")).strip()
        detail = str(workflow_status.get("detail", "")).strip()
        stage = str(workflow_status.get("stage", "")).strip()
        source = str(workflow_status.get("source", "")).strip()
        orientation_order = str(
            workflow_status.get("chart_orientation_order", "")
        ).strip()
        progress = workflow_status.get("progress")
        total = workflow_status.get("total")
        orientation_lines = []
        if orientation_order:
            if orientation_order == "visual_180":
                orientation_lines.append("ORIENT: UPSIDE-DOWN")
                orientation_lines.append("remap applied")
            elif orientation_order == "visual":
                orientation_lines.append("ORIENT: UPRIGHT")
            else:
                orientation_lines.append(f"ORIENT: {orientation_order}")
        if stage == "preview_loading":
            lines = [message or "色基準48色 自動検出中"]
            if detail:
                lines.append(detail)
            lines.extend(orientation_lines)
            return lines
        if stage == "preview":
            meta = f"source={source or 'preview'}  P/Enter start  G save  ESC/q cancel"
            adjust = "[+/-] hinge  [,/.] ROI"
            lines = [message or "色基準48色 プレビュー中", meta]
            lines.extend(orientation_lines)
            lines.append(adjust)
            return lines
        meta_parts = []
        if stage:
            meta_parts.append(stage.upper())
        if source:
            meta_parts.append(source)
        if progress is not None and total:
            meta_parts.append(f"{progress}/{total}")
        lines = [message or "P workflow"]
        if detail:
            lines.append(detail)
        lines.extend(orientation_lines)
        if meta_parts:
            lines.append(" / ".join(meta_parts))
        return lines

    def _left_panel_gray_summary_y(self, key_header_y: int) -> int:
        """フロー末尾とキーセクションの間に置く4DサマリのベースラインY。

        5A.2 (Phase 5 overlay UI fix): `[m] 測定開始`(shortcut 末尾行)と
        gray_summary (`4D rel GOOD / abs OK`) が重なる問題を解消するため、
        オフセットを -10 -> -20 に拡大。shortcut_item_gap=17 との余白合計
        約 6px を確保。
        """
        return key_header_y - 20

    def _build_left_panel_state_key(
        self,
        *,
        dark_loaded,
        flat_loaded,
        wb_calibrated,
        neutral_calibrated,
        blank_loaded,
        master_ref_loaded,
        ccm_loaded,
        flat_global_p90,
        flat_ref_p95,
        flat_ref_roi_gain_max,
        flat_has_valid_mask,
        gray_verify_status,
        gray_absolute_status,
        software_version,
    ) -> tuple:
        """left panel cache key を生成する。"""
        return (
            dark_loaded,
            flat_loaded,
            wb_calibrated,
            neutral_calibrated,
            blank_loaded,
            master_ref_loaded,
            ccm_loaded,
            flat_global_p90,
            flat_ref_p95,
            flat_ref_roi_gain_max,
            flat_has_valid_mask,
            gray_verify_status,
            gray_absolute_status,
            str(software_version or ""),
        )

    @staticmethod
    def _compute_left_panel_phase(
        *,
        dark_loaded,
        flat_loaded,
        wb_calibrated,
        neutral_calibrated,
        ccm_loaded,
    ) -> int:
        """現在の left panel phase を返す。"""
        if not dark_loaded:
            return 0
        if not flat_loaded:
            return 1
        if not wb_calibrated:
            return 2
        if not (ccm_loaded and neutral_calibrated):
            return 3
        return 4

    @staticmethod
    def _build_left_panel_flow_steps(
        *,
        dark_loaded,
        flat_loaded,
        wb_calibrated,
        neutral_calibrated,
        ccm_loaded,
    ) -> "tuple[LeftPanelFlowStep, ...]":
        """left panel の測定フロー定義を返す。"""
        return (
            LeftPanelFlowStep("レンズキャップ装着", False, 0),
            LeftPanelFlowStep("[D] ノイズ補正", dark_loaded, 0),
            LeftPanelFlowStep("レンズキャップ外す", False, 1),
            LeftPanelFlowStep("[F] 照明ムラ補正", flat_loaded, 1),
            LeftPanelFlowStep("灰色カード設置", False, 2),
            LeftPanelFlowStep("[W] ホワイト固定", wb_calibrated, 2),
            LeftPanelFlowStep("色基準48色の設置", False, 3),
            LeftPanelFlowStep("[P] 色基準取得", ccm_loaded and neutral_calibrated, 3),
            LeftPanelFlowStep("色基準48色を取出す", False, 4),
            LeftPanelFlowStep("[V] Tar灰カード確認", False, 4),
            LeftPanelFlowStep("試料セット", False, 4),
            LeftPanelFlowStep("[m] 測定開始", False, 4),
        )

    def _build_left_panel_shortcut_sections(self) -> "tuple[LeftPanelShortcutSection, ...]":
        """left panel shortcut を section 単位で返す。"""
        shortcuts = self._left_panel_shortcuts()
        roi_index = next(
            (index for index, (key, _desc) in enumerate(shortcuts) if key == "r/t"),
            len(shortcuts),
        )
        sections = []
        if roi_index > 0:
            sections.append(
                LeftPanelShortcutSection(
                    title=None,
                    ascii_title=None,
                    items=tuple(shortcuts[:roi_index]),
                )
            )
        if roi_index < len(shortcuts):
            sections.append(
                LeftPanelShortcutSection(
                    title="ROI操作",
                    ascii_title="ROI",
                    items=tuple(shortcuts[roi_index:]),
                )
            )
        return tuple(sections)

    @staticmethod
    def _build_left_panel_rule_lines() -> "tuple[tuple[str, ...], tuple[str, ...]]":
        """left panel の運用ルール文言を返す."""
        return (
            (
                "・測定前に D→F→W→P→V",
                "・背景変更後は c→D/F/W/P",
                "・初回Pは自動で始業登録",
            ),
            (
                "Run D->F->W->P->V",
                "After change: c then D/F/W/P",
                "First P auto-registers day start",
            ),
        )

    @staticmethod
    def _build_left_panel_section_labels() -> dict:
        """left panel section header 文言を返す。"""
        return {
            "flow_header": "測定フロー",
            "flow_header_ascii": "Flow",
            "keys_header": "操作キー",
            "keys_header_ascii": "Keys",
            "rules_header": "運用ルール",
            "rules_header_ascii": "Rules",
            "version_header": "動作版",
            "version_header_ascii": "Version",
        }

    def _assemble_left_panel_content(
        self,
        *,
        dark_loaded,
        flat_loaded,
        wb_calibrated,
        neutral_calibrated,
        ccm_loaded,
        gray_verify_status,
        gray_absolute_status,
        software_version,
    ) -> LeftPanelContent:
        """renderer 非依存の left panel content を組み立てる。"""
        operation_rules, operation_rules_ascii = self._build_left_panel_rule_lines()
        section_labels = self._build_left_panel_section_labels()
        del gray_verify_status, gray_absolute_status
        gray_summary_text, gray_summary_color = None, (160, 160, 160)
        return LeftPanelContent(
            current_phase=self._compute_left_panel_phase(
                dark_loaded=dark_loaded,
                flat_loaded=flat_loaded,
                wb_calibrated=wb_calibrated,
                neutral_calibrated=neutral_calibrated,
                ccm_loaded=ccm_loaded,
            ),
            flow_header=section_labels["flow_header"],
            flow_header_ascii=section_labels["flow_header_ascii"],
            keys_header=section_labels["keys_header"],
            keys_header_ascii=section_labels["keys_header_ascii"],
            rules_header=section_labels["rules_header"],
            rules_header_ascii=section_labels["rules_header_ascii"],
            version_header=section_labels["version_header"],
            version_header_ascii=section_labels["version_header_ascii"],
            flow_steps=self._build_left_panel_flow_steps(
                dark_loaded=dark_loaded,
                flat_loaded=flat_loaded,
                wb_calibrated=wb_calibrated,
                neutral_calibrated=neutral_calibrated,
                ccm_loaded=ccm_loaded,
            ),
            shortcut_sections=self._build_left_panel_shortcut_sections(),
            operation_rules=operation_rules,
            operation_rules_ascii=operation_rules_ascii,
            gray_summary_text=gray_summary_text,
            gray_summary_color=gray_summary_color,
            version_lines=tuple(
                self._build_left_panel_version_lines(
                    software_version,
                    max_width_px=max(0, self.LEFT_PANEL_WIDTH - 20),
                )
            ),
        )

    def _build_left_panel_layout(self, content: LeftPanelContent, panel: np.ndarray) -> LeftPanelLayout:
        """left panel レイアウト定数を返す。"""
        width = panel.shape[1]
        height = panel.shape[0]
        circle_x = 18
        step_start_y = 42
        step_gap = 20
        circle_radius = 5
        shortcut_start_gap = 18
        shortcut_section_top_gap = 4
        shortcut_section_title_gap = 10
        shortcut_item_gap = 15
        rules_header_gap = 14
        version_line_gap = 14
        version_bottom_margin = 28
        key_header_y = step_start_y + len(content.flow_steps) * step_gap
        step_label_x = circle_x + circle_radius + 8
        shortcut_desc_x = 58
        shortcut_y = key_header_y + shortcut_start_gap
        for section in content.shortcut_sections:
            if section.title:
                shortcut_y += shortcut_section_top_gap
                shortcut_y += shortcut_section_title_gap
            shortcut_y += shortcut_item_gap * len(section.items)

        rules_header_y = shortcut_y + 4
        version_line_count = max(1, len(content.version_lines))
        version_value_y = max(
            0,
            height - version_bottom_margin - version_line_gap * (version_line_count - 1),
        )
        version_header_y = max(0, version_value_y - 18)
        available_rule_height = max(0, version_header_y - 8 - (rules_header_y + rules_header_gap))
        min_rules_line_gap = 12
        max_rules_line_gap = 16
        if len(content.operation_rules) > 0:
            rules_line_gap = max(
                min_rules_line_gap,
                min(max_rules_line_gap, available_rule_height // len(content.operation_rules)),
            )
        else:
            rules_line_gap = min_rules_line_gap
        rules_bottom_y = rules_header_y + rules_header_gap + rules_line_gap * len(content.operation_rules)
        show_version = version_header_y >= rules_bottom_y + 8
        return LeftPanelLayout(
            width=width,
            height=height,
            circle_x=circle_x,
            step_start_y=step_start_y,
            step_gap=step_gap,
            circle_radius=circle_radius,
            key_header_y=key_header_y,
            shortcut_start_gap=shortcut_start_gap,
            shortcut_section_top_gap=shortcut_section_top_gap,
            shortcut_section_title_gap=shortcut_section_title_gap,
            shortcut_item_gap=shortcut_item_gap,
            rules_header_gap=rules_header_gap,
            rules_line_gap=rules_line_gap,
            version_line_gap=version_line_gap,
            version_bottom_margin=version_bottom_margin,
            step_label_x=step_label_x,
            step_label_max_width=max(0, width - step_label_x - 10),
            shortcut_desc_x=shortcut_desc_x,
            shortcut_desc_max_width=max(0, width - shortcut_desc_x - 12),
            rule_max_width=max(0, width - 24),
            version_header_y=version_header_y,
            version_value_y=version_value_y,
            show_version=show_version,
        )

    def _draw_left_panel_base_shapes(
        self,
        panel: np.ndarray,
        content: LeftPanelContent,
        layout: LeftPanelLayout,
    ) -> "list[int]":
        """left panel の円・線・区切り線だけを描画する。"""
        green_bgr = (0, 180, 0)
        yellow_bgr = (0, 220, 255)
        gray_bgr = (80, 80, 80)
        dim_bgr = (60, 60, 60)

        cv2.line(panel, (10, 28), (layout.width - 10, 28), (80, 80, 80), 1)

        step_centers = []
        for index, step in enumerate(content.flow_steps):
            cy = layout.step_start_y + index * layout.step_gap
            step_centers.append(cy)

            if step.is_done:
                cv2.circle(panel, (layout.circle_x, cy), layout.circle_radius, green_bgr, -1, cv2.LINE_AA)
                cv2.line(
                    panel,
                    (layout.circle_x - 3, cy),
                    (layout.circle_x - 1, cy + 3),
                    green_bgr,
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    panel,
                    (layout.circle_x - 1, cy + 3),
                    (layout.circle_x + 4, cy - 3),
                    green_bgr,
                    2,
                    cv2.LINE_AA,
                )
            elif step.phase == content.current_phase:
                cv2.circle(panel, (layout.circle_x, cy), layout.circle_radius, yellow_bgr, -1, cv2.LINE_AA)
            elif step.phase < content.current_phase:
                cv2.circle(panel, (layout.circle_x, cy), layout.circle_radius, green_bgr, -1, cv2.LINE_AA)
            else:
                cv2.circle(panel, (layout.circle_x, cy), layout.circle_radius, gray_bgr, 1, cv2.LINE_AA)

            if index > 0:
                prev_cy = step_centers[index - 1]
                line_color = green_bgr if step.phase <= content.current_phase else dim_bgr
                cv2.line(
                    panel,
                    (layout.circle_x, prev_cy + layout.circle_radius + 2),
                    (layout.circle_x, cy - layout.circle_radius - 2),
                    line_color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.line(
            panel,
            (10, layout.key_header_y + 16),
            (layout.width - 10, layout.key_header_y + 16),
            (80, 80, 80),
            1,
        )
        return step_centers

    def _render_left_panel_pil(
        self,
        panel: np.ndarray,
        content: LeftPanelContent,
        layout: LeftPanelLayout,
        step_centers: "list[int]",
    ) -> np.ndarray:
        """PIL backend で left panel text を描画する。"""
        fw = _ascii_to_fullwidth
        pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        draw.text((10, 10), content.flow_header, font=self._jp_font, fill=(220, 220, 220))

        for step, cy in zip(content.flow_steps, step_centers):
            text_y = cy - 7
            if step.is_done:
                color = (0, 200, 0)
            elif step.phase == content.current_phase:
                color = (255, 220, 0)
            elif step.phase < content.current_phase:
                color = (0, 160, 0)
            else:
                color = (100, 100, 100)
            fitted_label = self._fit_text_to_width_pil(
                step.label,
                max_width_px=layout.step_label_max_width,
                font=self._jp_font_small,
                transform=fw,
            )
            draw.text(
                (layout.step_label_x, text_y),
                fw(fitted_label),
                font=self._jp_font_small,
                fill=color,
            )

        draw.text((10, layout.key_header_y), content.keys_header, font=self._jp_font, fill=(220, 220, 220))
        ky = layout.key_header_y + layout.shortcut_start_gap
        for section in content.shortcut_sections:
            if section.title:
                ky += layout.shortcut_section_top_gap
                draw.text((14, ky), section.title, font=self._jp_font_small, fill=(180, 180, 180))
                ky += layout.shortcut_section_title_gap
            for key, desc in section.items:
                fitted_desc = self._fit_text_to_width_pil(
                    desc,
                    max_width_px=layout.shortcut_desc_max_width,
                    font=self._jp_font_small,
                    transform=fw,
                )
                draw.text((14, ky), fw(key), font=self._jp_font_small, fill=(0, 220, 220))
                draw.text(
                    (layout.shortcut_desc_x, ky),
                    fw(fitted_desc),
                    font=self._jp_font_small,
                    fill=(160, 160, 160),
                )
                ky += layout.shortcut_item_gap

        rule_header_y = ky + 4
        draw.text((10, rule_header_y), content.rules_header, font=self._jp_font, fill=(220, 220, 220))
        ry = rule_header_y + layout.rules_header_gap
        for rule in content.operation_rules:
            fitted_rule = self._fit_text_to_width_pil(
                rule,
                max_width_px=layout.rule_max_width,
                font=self._jp_font_small,
                transform=fw,
            )
            draw.text((14, ry), fw(fitted_rule), font=self._jp_font_small, fill=(160, 160, 160))
            ry += layout.rules_line_gap

        if layout.show_version:
            draw.line(
                (10, layout.version_header_y - 6, layout.width - 10, layout.version_header_y - 6),
                fill=(80, 80, 80),
                width=1,
            )
            draw.text((10, layout.version_header_y), content.version_header, font=self._jp_font_small, fill=(180, 180, 180))
            for idx, version_line in enumerate(content.version_lines):
                draw.text(
                    (14, layout.version_value_y + idx * layout.version_line_gap),
                    fw(version_line),
                    font=self._jp_font_small,
                    fill=(160, 160, 160),
                )

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _render_left_panel_cv2(
        self,
        panel: np.ndarray,
        content: LeftPanelContent,
        layout: LeftPanelLayout,
        step_centers: "list[int]",
    ) -> None:
        """cv2 backend で left panel text を描画する。"""
        cv2.putText(panel, content.flow_header_ascii, (10, 22), self.FONT, 0.4, (220, 220, 220), 1, self.LINE_TYPE)

        for step, cy in zip(content.flow_steps, step_centers):
            if step.is_done:
                color = (0, 200, 0)
            elif step.phase == content.current_phase:
                color = (0, 220, 255)
            elif step.phase < content.current_phase:
                color = (0, 160, 0)
            else:
                color = (100, 100, 100)
            fitted_label = self._fit_text_to_width_cv2(
                step.label,
                max_width_px=layout.step_label_max_width,
                font=self.FONT,
                font_scale=0.3,
                thickness=1,
            )
            cv2.putText(
                panel,
                fitted_label,
                (layout.step_label_x, cy + 4),
                self.FONT,
                0.3,
                color,
                1,
                self.LINE_TYPE,
            )

        cv2.putText(
            panel,
            content.keys_header_ascii,
            (10, layout.key_header_y + 12),
            self.FONT,
            0.4,
            (220, 220, 220),
            1,
            self.LINE_TYPE,
        )
        ky = layout.key_header_y + layout.shortcut_start_gap + 6
        for section in content.shortcut_sections:
            if section.ascii_title:
                ky += layout.shortcut_section_top_gap
                cv2.putText(
                    panel,
                    section.ascii_title,
                    (14, ky),
                    self.FONT,
                    0.28,
                    (180, 180, 180),
                    1,
                    self.LINE_TYPE,
                )
                ky += layout.shortcut_section_title_gap
            for key, desc in section.items:
                fitted_desc = self._fit_text_to_width_cv2(
                    desc,
                    max_width_px=layout.shortcut_desc_max_width,
                    font=self.FONT,
                    font_scale=0.28,
                    thickness=1,
                )
                cv2.putText(panel, key, (14, ky), self.FONT, 0.3, (0, 220, 220), 1, self.LINE_TYPE)
                cv2.putText(
                    panel,
                    fitted_desc,
                    (layout.shortcut_desc_x, ky),
                    self.FONT,
                    0.28,
                    (160, 160, 160),
                    1,
                    self.LINE_TYPE,
                )
                ky += layout.shortcut_item_gap

        cv2.putText(
            panel,
            content.rules_header_ascii,
            (10, ky + 10),
            self.FONT,
            0.35,
            (220, 220, 220),
            1,
            self.LINE_TYPE,
        )
        ry = ky + 10 + layout.rules_header_gap
        for rule in content.operation_rules_ascii:
            fitted_rule = self._fit_text_to_width_cv2(
                rule,
                max_width_px=layout.rule_max_width,
                font=self.FONT,
                font_scale=0.27,
                thickness=1,
            )
            cv2.putText(
                panel,
                fitted_rule,
                (14, ry),
                self.FONT,
                0.27,
                (160, 160, 160),
                1,
                self.LINE_TYPE,
            )
            ry += layout.rules_line_gap

        if layout.show_version:
            cv2.line(
                panel,
                (10, layout.version_header_y - 8),
                (layout.width - 10, layout.version_header_y - 8),
                (80, 80, 80),
                1,
            )
            cv2.putText(
                panel,
                content.version_header_ascii,
                (10, layout.version_header_y),
                self.FONT,
                0.3,
                (180, 180, 180),
                1,
                self.LINE_TYPE,
            )
            for idx, version_line in enumerate(content.version_lines):
                cv2.putText(
                    panel,
                    version_line,
                    (14, layout.version_value_y + idx * layout.version_line_gap),
                    self.FONT,
                    0.28,
                    (160, 160, 160),
                    1,
                    self.LINE_TYPE,
                )

    def _draw_left_panel_gray_summary(
        self,
        panel: np.ndarray,
        content: LeftPanelContent,
        layout: LeftPanelLayout,
    ) -> None:
        """4D summary を共通位置へ描画する。"""
        if content.gray_summary_text is None:
            return
        cv2.putText(
            panel,
            content.gray_summary_text,
            (42, self._left_panel_gray_summary_y(layout.key_header_y)),
            self.FONT,
            0.28,
            content.gray_summary_color,
            1,
            self.LINE_TYPE,
        )

    def _info_line_height(self) -> int:
        """情報行 1 行あたりの高さ (px) を返す。PIL 利用可ならフォント実測値を使う。"""
        use_pil = _PIL_AVAILABLE and self._jp_font_small is not None
        if use_pil:
            try:
                bbox = self._jp_font_small.getbbox("Ag絶対")
                return bbox[3] - bbox[1] + 4
            except Exception:
                pass
        return 14

    def _measurement_extra_line_count(
        self,
        gray_absolute_result: "dict | None",
        acceptance_result: "dict | None" = None,
        lab_rel_l_warn: bool = False,
        workflow_status: "dict | None" = None,
    ) -> int:
        """Measurement ブロックで追加表示する補助行数を返す。"""
        return (
            len(self._build_gray_absolute_detail_lines(gray_absolute_result))
            + len(self._build_acceptance_detail_lines(acceptance_result))
            + len(self._build_workflow_detail_lines(workflow_status))
            + int(lab_rel_l_warn)
        )

    def _measurement_block_height(
        self,
        gray_absolute_result: "dict | None",
        acceptance_result: "dict | None" = None,
        lab_rel_l_warn: bool = False,
        workflow_status: "dict | None" = None,
    ) -> int:
        """Measurement ブロックの必要高さを返す。

        通常時は Lab 3行の直後に Ref Exposure を寄せ、detail 行がある時だけ
        `row_spacing * 3 + 16 + extra_lines * line_height` を使う。
        detail 末端 = info_y + extra_lines * line_height = sp*3 + 8 + extra_lines * line_height。
        block_height - detail_end = 8px を維持し、さらに section gap で見出し衝突を防ぐ。
        """
        ly = self.ly
        extra_lines = self._measurement_extra_line_count(
            gray_absolute_result,
            acceptance_result,
            lab_rel_l_warn,
            workflow_status,
        )
        if extra_lines <= 0:
            return ly.hdr_gap_text + ly.row_spacing * 2 + 22
        return ly.hdr_gap_text + ly.row_spacing * 3 + 16 + extra_lines * self._info_line_height()

    def _side_panel_top(self, has_mode_bar: bool) -> int:
        """右パネルの最上段開始位置を返す。"""
        if not has_mode_bar:
            return self.ly.margin_top
        mode_bar_gap = getattr(self.ly, "mode_bar_gap", 14)
        return max(
            self.ly.margin_top,
            self.ly.mode_bar_top + self.ly.mode_bar_height + mode_bar_gap,
        )

    def _compute_side_panel_layout(
        self,
        panel_h: int,
        has_mode_bar: bool,
        has_ref_signal: bool,
        gray_absolute_result: "dict | None",
        acceptance_result: "dict | None" = None,
        lab_rel_l_warn: bool = False,
        workflow_status: "dict | None" = None,
    ) -> dict:
        """右パネル各セクションのY位置を計算する。"""
        ly = self.ly
        top_y = self._side_panel_top(has_mode_bar)
        blocks = [
            (
                "measurement_y",
                self._measurement_block_height(
                    gray_absolute_result,
                    acceptance_result,
                    lab_rel_l_warn,
                    workflow_status,
                ),
            ),
            ("ref_exposure_y", ly.hdr_gap_graph + ly.hist_height),
        ]
        if has_ref_signal:
            blocks.append(("ref_signal_y", ly.hdr_gap_graph + ly.ts_height))
        blocks.append(("spectral_drift_y", ly.hdr_gap_graph + ly.drift_height))

        total_h = sum(height for _, height in blocks)
        gap_count = max(len(blocks) - 1, 1)
        natural_gap = (panel_h - top_y - ly.margin_bot - total_h) // gap_count
        # Section titles are drawn on their baseline and extend upward. A 10-12px
        # graph-to-title gap lets the next title collide with the previous graph.
        gap = min(max(natural_gap, 24), 28)

        y = top_y
        layout = {"gap": gap, "top_y": top_y, "ref_signal_y": None}
        for idx, (name, height) in enumerate(blocks):
            layout[name] = y
            if idx < len(blocks) - 1:
                y += height + gap
        return layout

    def draw_stability_indicator(self, frame, x, y, state):
        """
        測定安定性インジケータを描画する。

        Args:
          frame: 描画対象フレーム。
          x: インジケータ中心X座標。
          y: インジケータ中心Y座標。
          state: `StabilityMonitor` の状態定数。
        """
        colors = {
            StabilityMonitor.READY: (0, 200, 0),
            StabilityMonitor.SETTLING: (0, 200, 200),
            StabilityMonitor.UNSTABLE: (0, 0, 200),
        }
        labels = {
            StabilityMonitor.READY: "Meas. OK",
            StabilityMonitor.SETTLING: "Stabilizing...",
            StabilityMonitor.UNSTABLE: "Unstable",
        }
        color = colors.get(state, (0, 0, 200))
        label = labels.get(state, "Unstable")
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.putText(
            frame,
            label,
            (x + 14, y + 5),
            self.FONT,
            self.ly.font_body,
            color,
            2,
            self.LINE_TYPE,
        )

    def draw_roi_rectangles(
        self,
        frame,
        pos_ref,
        w_ref,
        h_ref,
        pos_tar,
        w_tar,
        h_tar,
        ref_raw_mean_12bit=None,
        ref_raw_in_band=False,
        ref_center_edge_diff_pct=None,
        center_edge_warn_pct=3.0,
        tar_raw_mean_12bit=None,
    ):
        """
        REF/TAR ROIの矩形枠と注釈を描画する。

        Args:
          frame: 描画対象フレーム。
          pos_ref: Ref ROI左上座標 (x, y)。
          w_ref: Ref ROI幅。
          h_ref: Ref ROI高さ。
          pos_tar: Target ROI左上座標 (x, y)。
          w_tar: Target ROI幅。
          h_tar: Target ROI高さ。
          ref_raw_mean_12bit: Ref ROIの12bit平均RAW値。
          ref_raw_in_band: Ref平均RAW値が目標帯域内ならTrue。
          ref_center_edge_diff_pct: Ref ROI中心-外周差[%]。
          center_edge_warn_pct: 非一様警告の閾値[%]。
          tar_raw_mean_12bit: Target ROIの12bit平均RAW値。
        """
        ox = self.LEFT_PANEL_WIDTH  # 左パネル分のXオフセット
        # Reference 枠（青）+ ラベル
        rx, ry = pos_ref[0] + ox, pos_ref[1]
        cv2.rectangle(
            frame,
            (rx, ry),
            (rx + w_ref, ry + h_ref),
            (255, 100, 0),
            2,
        )
        cv2.putText(
            frame,
            "Ref",
            (rx, ry - 8),
            self.FONT,
            0.5,
            (255, 100, 0),
            1,
            self.LINE_TYPE,
        )
        # Ref ROI 直下に RAW / Non-uniform を表示
        fs = self.ly.font_roi_annotation
        roi_sp = self.ly.roi_annotation_spacing
        info_y = ry + h_ref + roi_sp
        if ref_raw_mean_12bit is not None:
            raw_color = (0, 220, 0) if ref_raw_in_band else (0, 200, 230)
            cv2.putText(
                frame,
                f"RAW:{ref_raw_mean_12bit:4.0f}",
                (rx, info_y),
                self.FONT,
                fs,
                raw_color,
                1,
                self.LINE_TYPE,
            )
            info_y += roi_sp
        has_nonuniform = (
            ref_center_edge_diff_pct is not None
            and ref_center_edge_diff_pct > center_edge_warn_pct
        )
        if has_nonuniform:
            cv2.putText(
                frame,
                f"NU:{ref_center_edge_diff_pct:.1f}%",
                (rx, info_y),
                self.FONT,
                fs,
                (0, 200, 230),
                1,
                self.LINE_TYPE,
            )
        # Target 枠（赤）+ ラベル
        tx, ty = pos_tar[0] + ox, pos_tar[1]
        cv2.rectangle(
            frame,
            (tx, ty),
            (tx + w_tar, ty + h_tar),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Target",
            (tx, ty - 8),
            self.FONT,
            0.5,
            (0, 0, 255),
            1,
            self.LINE_TYPE,
        )
        # Target ROI 直下に RAW を表示
        if tar_raw_mean_12bit is not None:
            cv2.putText(
                frame,
                f"RAW:{tar_raw_mean_12bit:4.0f}",
                (tx, ty + h_tar + roi_sp),
                self.FONT,
                fs,
                (0, 100, 255),
                1,
                self.LINE_TYPE,
            )

    def create_composite_frame(
        self,
        camera_frame: np.ndarray,
        left_panel: np.ndarray = None,
    ) -> np.ndarray:
        """
        左パネル＋カメラ＋右パネルの合成フレームを作成する。

        Args:
          camera_frame: カメラフレーム(BGR)。
          left_panel: 左パネル画像。Noneなら空パネルを生成。
        Returns:
          np.ndarray: 合成フレーム。
        """
        h = camera_frame.shape[0]
        right = np.zeros((h, self.PANEL_WIDTH, 3), dtype=np.uint8)
        right[:] = self.PANEL_BG
        if left_panel is None:
            left_panel = np.zeros((h, self.LEFT_PANEL_WIDTH, 3), dtype=np.uint8)
            left_panel[:] = self.PANEL_BG
        return np.hstack([left_panel, camera_frame, right])

    def build_left_panel(
        self,
        dark_loaded,
        flat_loaded,
        wb_calibrated=False,
        neutral_calibrated=False,
        blank_loaded=False,
        master_ref_loaded=False,
        ccm_loaded=False,
        flat_global_p90=None,
        flat_ref_p95=None,
        flat_ref_roi_gain_max=None,
        flat_has_valid_mask=False,
        gray_verify_status=None,
        gray_absolute_status=None,
        software_version=None,
    ):
        """
        測定フロー＋ショートカット一覧の左パネル画像を生成する。

        状態変化時のみ再構築し、キャッシュを返す。

        Args:
          dark_loaded: ダークフレーム取得済みか。
          flat_loaded: フラットフィールド取得済みか。
          wb_calibrated: WBキャリブレーション済みか。
          neutral_calibrated: ニュートラル補正済みか。
          blank_loaded: ブランク測定済みか。
          master_ref_loaded: マスターRef取得済みか。
          ccm_loaded: CCMキャリブレーション済みか（呼び出し元から渡す）。
        Returns:
          np.ndarray: BGR画像 (h, w, 3)。
        """
        state_key = self._build_left_panel_state_key(
            dark_loaded=dark_loaded,
            flat_loaded=flat_loaded,
            wb_calibrated=wb_calibrated,
            neutral_calibrated=neutral_calibrated,
            blank_loaded=blank_loaded,
            master_ref_loaded=master_ref_loaded,
            ccm_loaded=ccm_loaded,
            flat_global_p90=flat_global_p90,
            flat_ref_p95=flat_ref_p95,
            flat_ref_roi_gain_max=flat_ref_roi_gain_max,
            flat_has_valid_mask=flat_has_valid_mask,
            gray_verify_status=gray_verify_status,
            gray_absolute_status=gray_absolute_status,
            software_version=software_version,
        )
        if (
            self._left_panel_cache_key == state_key
            and self._left_panel_cache is not None
        ):
            return self._left_panel_cache

        h = self.config.camera.display_size[1]
        panel = np.zeros((h, self.LEFT_PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = self.PANEL_BG
        content = self._assemble_left_panel_content(
            dark_loaded=dark_loaded,
            flat_loaded=flat_loaded,
            wb_calibrated=wb_calibrated,
            neutral_calibrated=neutral_calibrated,
            ccm_loaded=ccm_loaded,
            gray_verify_status=gray_verify_status,
            gray_absolute_status=gray_absolute_status,
            software_version=software_version,
        )
        layout = self._build_left_panel_layout(content, panel)
        step_centers = self._draw_left_panel_base_shapes(panel, content, layout)
        if _PIL_AVAILABLE and self._jp_font is not None:
            panel = self._render_left_panel_pil(panel, content, layout, step_centers)
        else:
            self._render_left_panel_cv2(panel, content, layout, step_centers)
        self._draw_left_panel_gray_summary(panel, content, layout)

        self._left_panel_cache = panel
        self._left_panel_cache_key = state_key
        return panel

    def draw_section_header(self, frame, x, y, title):
        """
        セクション見出しと区切り線を描画する。

        Args:
          frame: 描画対象フレーム。
          x, y: 描画位置。
          title: セクションタイトル。
        """
        ly = self.ly
        cv2.putText(
            frame,
            title,
            (x, y),
            self.FONT,
            ly.font_section_title,
            (230, 230, 230),
            2,
            self.LINE_TYPE,
        )
        cv2.line(frame, (x, y + 6), (x + ly.graph_width + 5, y + 6), (100, 100, 100), 1)

    def draw_target_histogram(self, frame, x, y, tar_luma, width=250, height=72):
        """
        ターゲットROIの輝度Y(12bit相当)ヒストグラムを描画する。

        Args:
          frame: 描画対象フレーム。
          x, y: 描画位置(左上)。
          tar_luma: ターゲットROIから算出した輝度Y配列。
          width, height: ヒストグラム描画サイズ。
        """
        gray = (100, 100, 100)
        cyan = (0, 230, 230)
        cv2.rectangle(frame, (x, y), (x + width, y + height), gray, 1)
        if tar_luma is None or tar_luma.size == 0:
            cv2.putText(
                frame, "No histogram data", (x + 6, y + 20), self.FONT, 0.35, gray, 1
            )
            return
        # 白色ターゲットで空間一様なら、ヒストグラムは単峰かつ狭帯域になる。
        hist, _ = np.histogram(
            tar_luma.ravel(), bins=self.HIST_BINS, range=self.HIST_RANGE_12BIT
        )
        hist = hist.astype(np.float32)
        max_v = float(hist.max()) if hist.size > 0 else 0.0
        if max_v <= 0.0:
            return
        bar_w = max(width // self.HIST_BINS, 1)
        for i, v in enumerate(hist):
            bar_h = int((v / max_v) * (height - 4))
            x0 = x + i * bar_w
            x1 = x + min((i + 1) * bar_w - 1, width - 1)
            y1 = y + height - 2
            y0 = y1 - bar_h
            cv2.rectangle(frame, (x0, y0), (x1, y1), cyan, -1)
        cv2.putText(
            frame,
            "0",
            (x + 2, y + height - 4),
            self.FONT,
            0.30,
            gray,
            1,
            self.LINE_TYPE,
        )
        max_label = "4095"
        max_label_size, _ = cv2.getTextSize(max_label, self.FONT, 0.30, 1)
        cv2.putText(
            frame,
            max_label,
            (x + width - max_label_size[0] - 2, y + height - 4),
            self.FONT,
            0.30,
            gray,
            1,
            self.LINE_TYPE,
        )

    def draw_ref_histogram(self, frame, x, y, ref_roi_raw, width=250, height=72):
        """
        Ref ROIのR/G/Bチャンネル別ヒストグラムと最適露出ゾーンを描画する。

        Args:
          frame: 描画対象フレーム。
          x, y: 描画位置(左上)。
          ref_roi_raw: Ref ROIの12bit RAW Bayer配列(uint16)。
          width, height: ヒストグラム描画サイズ。
        """
        gray = (100, 100, 100)
        cv2.rectangle(frame, (x, y), (x + width, y + height), gray, 1)

        if ref_roi_raw is None or ref_roi_raw.size == 0:
            cv2.putText(
                frame, "No histogram data", (x + 6, y + 20), self.FONT, 0.35, gray, 1
            )
            return

        # 最適露出ゾーン (500-1000 on 0-4095 scale)
        zone_x0 = x + int(500 / 4096 * width)
        zone_x1 = x + int(1000 / 4096 * width)
        cv2.rectangle(
            frame, (zone_x0, y + 1), (zone_x1, y + height - 1), (0, 45, 0), -1
        )

        # BGGRパターンからR/G/Bチャンネル抽出
        B = ref_roi_raw[0::2, 0::2].ravel().astype(np.float32)
        G1 = ref_roi_raw[0::2, 1::2].ravel().astype(np.float32)
        G2 = ref_roi_raw[1::2, 0::2].ravel().astype(np.float32)
        R = ref_roi_raw[1::2, 1::2].ravel().astype(np.float32)
        G = np.concatenate([G1, G2])

        bins = self.HIST_BINS
        hist_range = self.HIST_RANGE_12BIT
        hist_r, _ = np.histogram(R, bins=bins, range=hist_range)
        hist_g, _ = np.histogram(G, bins=bins, range=hist_range)
        hist_b, _ = np.histogram(B, bins=bins, range=hist_range)

        # G はピクセル数が2倍なので正規化
        hist_r = hist_r.astype(np.float32)
        hist_g = (hist_g / 2.0).astype(np.float32)
        hist_b = hist_b.astype(np.float32)

        max_v = max(float(hist_r.max()), float(hist_g.max()), float(hist_b.max()), 1.0)

        # R/G/Bをライン描画
        channel_data = [
            (hist_r, (0, 0, 200)),
            (hist_g, (0, 180, 0)),
            (hist_b, (200, 100, 0)),
        ]
        for hist, color in channel_data:
            pts = []
            for i, v in enumerate(hist):
                px_x = x + int((i + 0.5) * width / bins)
                px_y = y + height - 2 - int((v / max_v) * (height - 4))
                pts.append((px_x, px_y))
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], color, 2, cv2.LINE_AA)

        # R/G/B 凡例
        cv2.putText(frame, "R", (x + width - 48, y + 12), self.FONT, 0.30, (0, 0, 200), 1, self.LINE_TYPE)
        cv2.putText(frame, "G", (x + width - 32, y + 12), self.FONT, 0.30, (0, 180, 0), 1, self.LINE_TYPE)
        cv2.putText(frame, "B", (x + width - 16, y + 12), self.FONT, 0.30, (200, 100, 0), 1, self.LINE_TYPE)

        # 軸ラベル（矩形内側の下端に描画）
        cv2.putText(frame, "0", (x + 2, y + height - 3), self.FONT, 0.26, gray, 1, self.LINE_TYPE)
        cv2.putText(frame, "4095", (x + width - 34, y + height - 3), self.FONT, 0.26, gray, 1, self.LINE_TYPE)

    @staticmethod
    def _ref_monitor_decimals(axis: str) -> int:
        axis = str(axis or "").upper()
        if axis == "C":
            return 3
        if axis == "I":
            return 1
        return 2

    @staticmethod
    def _ref_monitor_metric_suffix(item: dict, axis: str) -> str:
        if str(axis or "").upper() != "U":
            return ""
        metric = str(item.get("display_metric") or "").strip()
        direction = str(item.get("display_direction") or "").strip()
        if metric and direction:
            return f" {metric} {direction}"
        if metric:
            return f" {metric}"
        return ""

    @classmethod
    def _format_ref_monitor_quantity(
        cls,
        axis: str,
        value,
        unit: str,
        *,
        signed: bool = False,
    ) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not np.isfinite(number):
            return "N/A"
        if not signed:
            number = abs(number)
        decimals = cls._ref_monitor_decimals(axis)
        return f"{number:.{decimals}f}{unit}"

    @staticmethod
    def _ref_monitor_limit_for_status(item: dict, status: str):
        try:
            direct = item.get("limit_value")
        except AttributeError:
            direct = None
        if direct is not None:
            try:
                value = float(direct)
            except (TypeError, ValueError):
                value = None
            if value is not None and np.isfinite(value):
                return value
        status = str(status or "OK").upper()
        key = {
            "OK": "watch",
            "WATCH": "warn",
            "WARN": "recal",
            "RECAL": "recal",
        }.get(status)
        if key is None:
            return None
        try:
            value = float(item.get(key))
        except (TypeError, ValueError, AttributeError):
            return None
        if np.isfinite(value):
            return value
        return None

    @classmethod
    def _build_ref_monitor_summary_text(
        cls,
        item: dict,
        *,
        selected_axis: str,
        selected_value,
    ) -> str:
        axis = str(item.get("axis") or selected_axis or "I").upper()
        label = str(item.get("label") or axis)
        status = str(item.get("status") or "OK").upper()
        unit = str(item.get("display_unit") or item.get("unit") or "")
        signed = axis == "I"
        value = item.get("display_value") if "display_value" in item else None
        if value is None:
            value = selected_value
        if value is None:
            value = item.get("value")
        value_kind = str(item.get("value_kind") or "")
        if axis == "U" and value_kind == "drift":
            signed = True
        value_text = cls._format_ref_monitor_quantity(axis, value, unit, signed=signed)
        if axis == "U" and value_kind == "drift":
            metric = str(item.get("display_metric") or item.get("drift_dominant") or "").strip()
            prefix = f"d:{metric} " if metric else "d:"
            value_text = f"{prefix}{value_text}"
            if item.get("direction_reversal"):
                value_text = f"{value_text} REV"
        else:
            value_text = f"{value_text}{cls._ref_monitor_metric_suffix(item, axis)}"
        limit_value = cls._ref_monitor_limit_for_status(item, status)
        limit_text = cls._format_ref_monitor_quantity(axis, limit_value, unit, signed=False)
        limit_label = str(item.get("limit_label") or "limit")
        return f"{label} {status} {value_text} / {limit_label} {limit_text}  i"

    @classmethod
    def _build_ref_monitor_summary_text_variants(
        cls,
        item: dict,
        *,
        selected_axis: str,
        selected_value,
    ) -> tuple[str, ...]:
        axis = str(item.get("axis") or selected_axis or "I").upper()
        label = str(item.get("label") or axis)
        status = str(item.get("status") or "OK").upper()
        unit = str(item.get("display_unit") or item.get("unit") or "")
        signed = axis in {"I", "U"}
        value = item.get("display_value") if "display_value" in item else None
        if value is None:
            value = selected_value if selected_value is not None else item.get("value")
        value_kind = str(item.get("value_kind") or "")
        if axis == "U" and value_kind == "drift":
            signed = True
        value_text = cls._format_ref_monitor_quantity(axis, value, unit, signed=signed)
        if axis == "U" and value_kind == "drift":
            metric = str(item.get("display_metric") or item.get("drift_dominant") or "").strip()
            prefix = f"d:{metric} " if metric else "d:"
            value_text = f"{prefix}{value_text}"
            if item.get("direction_reversal"):
                value_text = f"{value_text} REV"
        else:
            value_text = f"{value_text}{cls._ref_monitor_metric_suffix(item, axis)}"
        limit_value = cls._ref_monitor_limit_for_status(item, status)
        limit_text = cls._format_ref_monitor_quantity(axis, limit_value, unit, signed=False)
        limit_label = str(item.get("limit_label") or "limit")
        op = ">=" if status == "RECAL" else "<"
        variants = (
            f"{label} {status} {value_text} / {limit_label} {limit_text}  i",
            f"{label} {status} {value_text}/{op}{limit_text} i",
            f"{label} {status} {value_text} i",
        )
        return tuple(dict.fromkeys(variants))

    def _fit_ref_monitor_summary_text(
        self,
        item: dict,
        *,
        selected_axis: str,
        selected_value,
        max_width_px: int,
        font_scale: float = 0.38,
    ) -> str:
        """Ref summary を幅内に収める。狭い場合も末尾の i は残す。"""
        if max_width_px <= 0:
            return ""

        def text_width(value: str) -> int:
            return cv2.getTextSize(value, self.FONT, font_scale, 1)[0][0]

        variants = self._build_ref_monitor_summary_text_variants(
            item,
            selected_axis=selected_axis,
            selected_value=selected_value,
        )
        for text in variants:
            if text_width(text) <= max_width_px:
                return text

        suffix = " i"
        minimal = variants[-1]
        if minimal.endswith(suffix):
            suffix_width = text_width(suffix)
            prefix_width = max(0, max_width_px - suffix_width - 2)
            prefix = self._fit_text_to_width_cv2(
                minimal[: -len(suffix)].rstrip(),
                max_width_px=prefix_width,
                font=self.FONT,
                font_scale=font_scale,
                ellipsis="",
            )
            return f"{prefix}{suffix}" if prefix else "i"
        return self._fit_text_to_width_cv2(
            minimal,
            max_width_px=max_width_px,
            font=self.FONT,
            font_scale=font_scale,
            ellipsis="",
        )

    @staticmethod
    def _ref_stability_plot_range(
        history,
        axis: str,
        selected_status: str,
        warn_value: float,
        recal_value: float,
    ) -> tuple[float, float]:
        """Ref Stability graph の表示用レンジを返す。

        判定閾値は status 行で伝える。グラフは時系列の形を読むため、
        OK 域では WARN/RECAL 閾値まで含めずデータ周辺に自動ズームする。
        """
        values = np.asarray(history, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return -1.0, 1.0

        axis = str(axis or "I").upper()
        state = str(selected_status or "OK").upper()
        data_min = float(np.min(values))
        data_max = float(np.max(values))
        data_span = max(data_max - data_min, 0.0)
        signed_axis = axis in {"I", "U"}

        if signed_axis:
            data_abs = max(abs(data_min), abs(data_max))
            floor = max(abs(float(warn_value)) * 0.04, 0.02)
            max_abs = max(data_abs * 1.6, floor)
            if state == "WATCH":
                max_abs = max(max_abs, abs(float(warn_value)) * 1.15)
            elif state == "WARN":
                max_abs = max(max_abs, abs(float(warn_value)) * 1.25)
            elif state == "RECAL":
                max_abs = max(max_abs, abs(float(recal_value)) * 1.10)
            return -max_abs, max_abs

        center = (data_min + data_max) * 0.5
        floor = max(abs(float(warn_value)) * 0.08, abs(center) * 0.35, 1e-6)
        visible_span = max(data_span * 1.8, floor)
        y_min = center - visible_span * 0.5
        y_max = center + visible_span * 0.5

        if state in {"WATCH", "WARN", "RECAL"}:
            y_min = min(0.0, data_min - visible_span * 0.2)
            y_max = max(y_max, data_max + visible_span * 0.2)
            if state == "WATCH":
                y_max = max(y_max, float(warn_value) * 1.15)
            elif state == "WARN":
                y_max = max(y_max, float(warn_value) * 1.25)
            else:
                y_max = max(y_max, float(recal_value) * 1.10)

        if y_max - y_min < 1e-9:
            y_min -= 0.5
            y_max += 0.5
        return float(y_min), float(y_max)

    def draw_stability_timeseries(self, frame, x, y, tracker, width=250, height=55):
        """
        Ref Stability の選択軸時系列グラフと軸ステータスを描画する。

        Args:
          frame: 描画対象フレーム。
          x, y: 描画位置(左上)。
          tracker: DarkFlatStabilityTracker インスタンス。
          width, height: グラフ描画サイズ。
        """
        gray = (100, 100, 100)
        status_rows_height = 64
        draw_status_rows = height >= 108
        plot_h = max(44, height - status_rows_height if draw_status_rows else height)
        cv2.rectangle(frame, (x, y), (x + width, y + plot_h), gray, 1)

        if tracker is None or tracker.count < 2:
            cv2.putText(
                frame, "Collecting data...", (x + 6, y + height // 2 + 4),
                self.FONT, 0.33, gray, 1, self.LINE_TYPE,
            )
            return

        series = None
        if hasattr(tracker, "get_ref_monitor_series"):
            series = tracker.get_ref_monitor_series()
            history = np.asarray(series.get("history"), dtype=np.float32).ravel()
            axis = str(series.get("axis") or "I").upper()
            selected_status = str(series.get("status") or "OK").upper()
            statuses = tuple(series.get("axis_statuses") or ())
            warn_value = float(series.get("warn") or 0.0)
            recal_value = float(series.get("recal") or 0.0)
            history_kind = str(series.get("history_kind") or "").lower()
        else:
            history = tracker.get_history("intensity_drift_pct")
            axis = "I"
            selected_status = "OK"
            statuses = ()
            warn_value, recal_value = 8.0, 12.0
            history_kind = ""
        n = len(history)
        if n < 2:
            return

        risk_axis = axis == "U" and history_kind == "risk"
        signed_axis = axis in {"I", "U"} and not risk_axis
        if risk_axis:
            warn_value = 1.0
            recal_value = 1.0
        plot_axis = "RISK" if risk_axis else axis
        y_min, y_max = self._ref_stability_plot_range(
            history,
            plot_axis,
            selected_status,
            warn_value,
            recal_value,
        )
        y_range = y_max - y_min

        def val_to_py(v):
            """値をピクセルY座標に変換（上が大、下が小）。"""
            frac = (v - y_min) / y_range
            return y + plot_h - 2 - int(frac * (plot_h - 4))

        # 0 line
        zero_py = val_to_py(0.0)
        cv2.line(frame, (x + 1, zero_py), (x + width - 1, zero_py), (150, 150, 150), 1)

        # Axis-specific threshold lines. OK域では時系列形状を優先し、閾値はstatus行に任せる。
        threshold_lines = []
        if str(selected_status).upper() in {"WATCH", "WARN", "RECAL"}:
            threshold_lines = [
                (warn_value, (0, 160, 255)),
            ]
            if not risk_axis:
                threshold_lines.append((recal_value, (0, 0, 200)))
            if signed_axis:
                threshold_lines = threshold_lines + [
                    (-warn_value, (0, 160, 255)),
                    (-recal_value, (0, 0, 200)),
                ]
        for th_val, color in threshold_lines:
            th_py = val_to_py(th_val)
            if y + 1 <= th_py <= y + plot_h - 1:
                for dx in range(0, width, 8):
                    x0 = x + dx
                    x1 = min(x + dx + 4, x + width)
                    cv2.line(frame, (x0, th_py), (x1, th_py), color, 1)

        # 時系列ライン
        state_colors = {
            "OK": (0, 200, 0),
            "WATCH": (0, 200, 255),
            "WARN": (0, 140, 255),
            "RECAL": (0, 0, 255),
        }
        line_color = state_colors.get(selected_status, (0, 200, 0))
        pts = []
        for i in range(n):
            px_x = x + 1 + int(i * (width - 2) / max(n - 1, 1))
            value = float(history[i])
            if not np.isfinite(value):
                pts.append(None)
                continue
            px_y = np.clip(val_to_py(value), y + 1, y + plot_h - 1)
            pts.append((px_x, int(px_y)))
        for j in range(len(pts) - 1):
            if pts[j] is None or pts[j + 1] is None:
                continue
            cv2.line(frame, pts[j], pts[j + 1], line_color, 1, cv2.LINE_AA)

        selected_value = None
        if n > 0 and np.isfinite(float(history[-1])):
            selected_value = float(history[-1])
        if draw_status_rows:
            self._draw_ref_monitor_status_rows(
                frame,
                x,
                y + plot_h + 12,
                width,
                statuses,
                selected_axis=axis,
                selected_value=selected_value,
                selected_series=series,
                state_colors=state_colors,
            )

    def _draw_ref_monitor_status_rows(
        self,
        frame,
        x: int,
        y: int,
        width: int,
        statuses: tuple,
        *,
        selected_axis: str,
        selected_value=None,
        selected_series=None,
        state_colors: dict,
    ) -> None:
        """Ref monitor の Bright/Color/... 状態を3行2列で描画する。"""
        selected_axis = str(selected_axis or "I").upper()
        status_by_axis = {
            str(item.get("axis") or "").upper(): item
            for item in statuses
            if isinstance(item, dict)
        }
        axes_rows = (("I", "C"), ("U", "J"), ("CLIP", "FLOOR"))
        fallback_labels = {
            "I": "Bright",
            "C": "Color",
            "U": "Uneven",
            "J": "Jitter",
            "CLIP": "Clip",
            "FLOOR": "Floor",
        }
        selected_item = dict(status_by_axis.get(selected_axis, {}))
        if isinstance(selected_series, dict):
            selected_item.update(selected_series)
        selected_label = str(selected_item.get("label") or fallback_labels.get(selected_axis, selected_axis))
        selected_status = str(selected_item.get("status") or "OK").upper()
        selected_color = state_colors.get(selected_status, (0, 200, 0))
        selected_item["axis"] = selected_axis
        selected_item["label"] = selected_label
        selected_item["status"] = selected_status
        mode_text = self._fit_ref_monitor_summary_text(
            selected_item,
            selected_axis=selected_axis,
            selected_value=selected_value,
            max_width_px=max(80, width - 4),
            font_scale=0.38,
        )
        cv2.putText(
            frame,
            mode_text,
            (x + 3, y),
            self.FONT,
            0.38,
            selected_color,
            1,
            self.LINE_TYPE,
        )

        col_w = max(1, width // 2)
        row_base_y = y + 17
        for row_index, axes in enumerate(axes_rows):
            text_y = row_base_y + row_index * 15
            for col_index, axis in enumerate(axes):
                item = status_by_axis.get(axis, {})
                label = str(item.get("label") or fallback_labels[axis])
                status = str(item.get("status") or "OK").upper()
                cell_x = x + col_index * col_w
                if axis == selected_axis:
                    color_for_border = state_colors.get(status, (0, 200, 0))
                    cv2.rectangle(
                        frame,
                        (cell_x + 1, text_y - 12),
                        (cell_x + col_w - 5, text_y + 4),
                        (34, 34, 34),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (cell_x + 1, text_y - 12),
                        (cell_x + col_w - 5, text_y + 4),
                        color_for_border,
                        1,
                    )
                text = self._fit_text_to_width_cv2(
                    f"{label} {status}",
                    max_width_px=max(80, col_w - 9),
                    font=self.FONT,
                    font_scale=0.34,
                    ellipsis="",
                )
                color = state_colors.get(status, (0, 200, 0))
                thickness = 2 if axis == selected_axis else 1
                cv2.putText(
                    frame,
                    text,
                    (cell_x + 5, text_y),
                    self.FONT,
                    0.34,
                    color,
                    thickness,
                    self.LINE_TYPE,
                )

    def draw_spectral_drift_graph(
        self,
        frame,
        x,
        y,
        tracker,
        width=250,
        height=65,
        ref_scale: "float | None" = None,
        ref_scale_warn: bool = False,
    ):
        """
        スペクトルドリフト補正係数 f_R・f_B の時系列グラフを描画する。

        Args:
          frame: 描画対象フレーム。
          x, y: 描画位置（左上）。
          tracker: SpectralDriftTracker インスタンス。
          width, height: グラフ描画サイズ。
        """
        gray = (100, 100, 100)
        value_strip_h = 20 if height >= 48 else 0
        plot_h = max(24, height - value_strip_h)
        cv2.rectangle(frame, (x, y), (x + width, y + plot_h), gray, 1)

        if tracker is None or not tracker.is_active:
            cv2.putText(
                frame, "Awaiting calib...", (x + 6, y + plot_h // 2 + 4),
                self.FONT, 0.33, gray, 1, self.LINE_TYPE,
            )
            return

        times, fR, fB = tracker.get_plot_data(max_points=width)
        n = len(times)

        if n < 2:
            cv2.putText(
                frame, "Collecting...", (x + 6, y + plot_h // 2 + 4),
                self.FONT, 0.33, gray, 1, self.LINE_TYPE,
            )
            return

        # --- Y軸レンジ ---
        all_vals = np.concatenate([fR, fB])
        data_min = float(all_vals.min())
        data_max = float(all_vals.max())
        tracker.update_y_range(data_min, data_max)
        y_min, y_max = tracker.get_y_range()
        y_range = y_max - y_min
        if y_range < 0.01:
            y_range = 0.01

        def val_to_py(v):
            frac = (v - y_min) / y_range
            return y + plot_h - 2 - int(frac * (plot_h - 4))

        # --- 1.0 基準線（白破線）---
        ref_py = val_to_py(1.0)
        if y + 2 <= ref_py <= y + plot_h - 2:
            for dx in range(0, width, 8):
                x0 = x + dx
                x1 = min(x + dx + 4, x + width)
                cv2.line(frame, (x0, ref_py), (x1, ref_py), (160, 160, 160), 1)

        # --- 折れ線グラフ描画 ---
        def draw_line(values, color):
            pts = []
            for i in range(n):
                px_x = x + 1 + int(i * (width - 2) / max(n - 1, 1))
                px_y = int(np.clip(val_to_py(float(values[i])), y + 1, y + plot_h - 1))
                pts.append((px_x, px_y))
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], color, 1, cv2.LINE_AA)

        draw_line(fR, (0, 0, 200))  # f_R: 赤
        draw_line(fB, (200, 80, 0))  # f_B: 青

        # --- 現在値テキスト（グラフ外の値ストリップ） ---
        f_R_cur, f_B_cur = tracker.get_current()
        elapsed = tracker.get_elapsed_seconds()
        if elapsed < 3600:
            elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"
        else:
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            elapsed_str = f"{h}h{m:02d}m"
        value_text = f"{elapsed_str} R{f_R_cur:.2f} B{f_B_cur:.2f}"
        value_color = (150, 150, 150)
        if ref_scale is not None:
            value_text += f" Y{ref_scale:.2f}"
            if ref_scale_warn:
                value_text += " WARN"
                value_color = (0, 200, 220)
            else:
                value_color = (0, 180, 0)
        cv2.putText(
            frame,
            value_text,
            (x + 3, y + plot_h + 15),
            self.FONT,
            0.38,
            value_color,
            1,
            self.LINE_TYPE,
        )

    def draw_correction_applied_warning_banner(self, frame, warning: "dict | None"):
        """Correction Applied の異常をカメラ領域上部中央に大きく描画する。"""
        if not isinstance(warning, dict):
            return frame
        axes = [str(axis).strip().upper() for axis in warning.get("axes", [])]
        axes = [axis for axis in axes if axis]
        if not axes:
            return frame

        h_img, w_img = frame.shape[:2]
        left_panel_w = int(getattr(self, "LEFT_PANEL_WIDTH", 0) or 0)
        right_panel_w = int(getattr(self, "PANEL_WIDTH", 0) or 0)
        camera_left = min(max(left_panel_w, 0), max(w_img - 1, 0))
        camera_right = max(camera_left + 1, w_img - max(right_panel_w, 0))
        camera_w = max(1, camera_right - camera_left)
        if camera_w < 120 or h_img < 80:
            return frame

        banner_w = min(camera_w - 24, max(360, int(camera_w * 0.76)))
        banner_w = max(96, banner_w)
        banner_h = min(92, max(64, h_img // 4))
        x1 = camera_left + max(12, (camera_w - banner_w) // 2)
        x2 = min(camera_right - 12, x1 + banner_w)
        x1 = max(camera_left + 8, x2 - banner_w)
        y1 = 10
        y2 = min(h_img - 8, y1 + banner_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 170), cv2.FILLED)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3, self.LINE_TYPE)

        values = warning.get("values", {}) if isinstance(warning.get("values"), dict) else {}
        detail = str(warning.get("detail") or "").strip()
        if not detail:
            detail = " ".join(
                f"{axis}{float(values[axis]):.2f}"
                for axis in ("R", "B", "Y")
                if axis in values
            )
        axis_text = "/".join(axes)
        title_text = str(warning.get("title") or "補正ドリフト警告")
        body_text = str(warning.get("message") or "色または明るさが基準からずれています")
        action_text = str(warning.get("action") or "vでGray Check / 必要なら再校正")
        detail_text = f"{axis_text}: {detail}" if detail else axis_text

        if _PIL_AVAILABLE and self._jp_font is not None:
            from PIL import Image, ImageDraw

            title_font = _find_japanese_font(25) or self._jp_font
            body_font = _find_japanese_font(17) or self._jp_font_small
            detail_font = _find_japanese_font(15) or self._jp_font_small
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            lines = (
                (title_text, title_font, (255, 255, 255)),
                (body_text, body_font, (255, 244, 210)),
                (f"{action_text}   {detail_text}", detail_font, (255, 255, 170)),
            )
            line_boxes = [draw.textbbox((0, 0), text, font=font) for text, font, _ in lines]
            line_heights = [box[3] - box[1] for box in line_boxes]
            total_h = sum(line_heights) + 8 * (len(lines) - 1)
            y = y1 + max(8, (y2 - y1 - total_h) // 2)
            for (text, font, color), box, line_h in zip(lines, line_boxes, line_heights):
                text_w = box[2] - box[0]
                tx = x1 + max(8, (x2 - x1 - text_w) // 2)
                for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                    draw.text((tx + dx, y + dy), text, font=font, fill=(0, 0, 0))
                draw.text((tx, y), text, font=font, fill=color)
                y += line_h + 8
            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

        font = self.FONT
        lines = (
            ("CORRECTION DRIFT WARNING", 0.76, (255, 255, 255), 2),
            ("Color/brightness changed from baseline", 0.52, (255, 244, 210), 1),
            (f"Press V Gray Check / recalibrate  {detail_text}", 0.46, (255, 255, 170), 1),
        )
        total_h = 0
        metrics = []
        for text, scale, color, thickness in lines:
            size = cv2.getTextSize(text, font, scale, thickness)[0]
            metrics.append((text, scale, color, thickness, size))
            total_h += size[1]
        total_h += 8 * (len(lines) - 1)
        baseline_y = y1 + max(12, (y2 - y1 - total_h) // 2)
        for text, scale, color, thickness, size in metrics:
            baseline_y += size[1]
            tx = x1 + max(8, (x2 - x1 - size[0]) // 2)
            cv2.putText(frame, text, (tx, baseline_y), font, scale, (0, 0, 0), thickness + 3, self.LINE_TYPE)
            cv2.putText(frame, text, (tx, baseline_y), font, scale, color, thickness, self.LINE_TYPE)
            baseline_y += 8
        return frame

    def _draw_lab_with_uncertainty(self, frame, x, y, Lab, U_L, U_a, U_b, lh):
        """READY状態時のLab値と拡張不確かさ(+/-)を描画する。"""
        green = (0, 230, 0)
        cv2.putText(frame, f"L*={Lab[0]:8.3f} +/-{U_L:.3f}", (x, y), self.FONT, 0.38, green, 1, self.LINE_TYPE)
        cv2.putText(frame, f"a*={Lab[1]:8.3f} +/-{U_a:.3f}", (x, y + lh), self.FONT, 0.38, green, 1, self.LINE_TYPE)
        cv2.putText(frame, f"b*={Lab[2]:8.3f} +/-{U_b:.3f}", (x, y + lh * 2), self.FONT, 0.38, green, 1, self.LINE_TYPE)

    def _draw_lab_values(self, frame, x, y, Lab, lh):
        """非READY状態時のLab値のみを描画する。"""
        green = (0, 230, 0)
        cv2.putText(frame, f"L*={Lab[0]:8.3f}", (x, y), self.FONT, 0.38, green, 1, self.LINE_TYPE)
        cv2.putText(frame, f"a*={Lab[1]:8.3f}", (x, y + lh), self.FONT, 0.38, green, 1, self.LINE_TYPE)
        cv2.putText(frame, f"b*={Lab[2]:8.3f}", (x, y + lh * 2), self.FONT, 0.38, green, 1, self.LINE_TYPE)

    def _draw_delta_e_and_patch(self, frame, x, y, dE00, lh):
        """dE00値と補助ラベルを描画する。"""
        cyan = (0, 230, 230)
        gray = (160, 160, 160)
        cv2.putText(frame, f"dE00={dE00:7.3f}", (x, y + lh * 3), self.FONT, 0.38, cyan, 1, self.LINE_TYPE)
        cv2.putText(frame, "(k=2)", (x + 160, y + lh * 3), self.FONT, 0.3, gray, 1, self.LINE_TYPE)

    def draw_measurement(
        self,
        frame,
        x,
        y,
        Lab_abs,
        Lab_rel,
        dE00_ref,
        dE00_target,
        mode=None,
        ratio=None,
        gray_absolute_result=None,
        acceptance_result=None,
        lab_rel_l_warn: bool = False,
        workflow_status=None,
    ):
        """L/a/b の値を強調して描画する。"""
        del Lab_abs, dE00_ref, dE00_target  # 直感UIでは非表示
        ly = self.ly
        scale = ly.font_body
        sp = ly.row_spacing
        right_edge = x + ly.graph_width

        # LinearRGBモードの描画
        if mode is not None and mode.is_linear() and ratio is not None:
            cyan = (230, 200, 0)
            outline = (255, 255, 255)
            ot = ly.lab_outline_thickness
            rows_lin = [
                ("R", ratio[0], y),
                ("G", ratio[1], y + sp),
                ("B", ratio[2], y + sp * 2),
            ]
            for label, val, ty in rows_lin:
                if ot > 0:
                    cv2.putText(frame, label, (x, ty), self.FONT, scale, outline, ot, self.LINE_TYPE)
                cv2.putText(frame, label, (x, ty), self.FONT, scale, cyan, 2, self.LINE_TYPE)
                val_str = f"{val:.4f}"
                sz, _ = cv2.getTextSize(val_str, self.FONT, scale, 2)
                vx = right_edge - sz[0]
                if ot > 0:
                    cv2.putText(frame, val_str, (vx, ty), self.FONT, scale, outline, ot, self.LINE_TYPE)
                cv2.putText(frame, val_str, (vx, ty), self.FONT, scale, cyan, 2, self.LINE_TYPE)
            return

        green = (0, 230, 0)
        rows = [
            ("L", Lab_rel[0], y),
            ("a", Lab_rel[1], y + sp),
            ("b", Lab_rel[2], y + sp * 2),
        ]
        outline = (255, 255, 255)
        ot = ly.lab_outline_thickness
        for label, val, ty in rows:
            row_color = green
            if label == "L" and lab_rel_l_warn:
                row_color = (0, 220, 220)
            if ot > 0:
                cv2.putText(frame, label, (x, ty), self.FONT, scale, outline, ot, self.LINE_TYPE)
            cv2.putText(frame, label, (x, ty), self.FONT, scale, row_color, 2, self.LINE_TYPE)
            val_str = f"{val:+.2f}"
            sz, _ = cv2.getTextSize(val_str, self.FONT, scale, 2)
            vx = right_edge - sz[0]
            if ot > 0:
                cv2.putText(frame, val_str, (vx, ty), self.FONT, scale, outline, ot, self.LINE_TYPE)
            cv2.putText(frame, val_str, (vx, ty), self.FONT, scale, row_color, 2, self.LINE_TYPE)
        info_y = y + sp * 3 + 8
        # --- 4D 絶対情報行 (PIL / cv2 フォールバック) ---
        detail_lines = self._build_gray_absolute_detail_lines(gray_absolute_result)
        if detail_lines:
            color = self._gray_absolute_status_color(gray_absolute_result)
            meta_color = (185, 185, 185)
            lwc = [
                (line, color if idx == 0 else meta_color, 0.34 if idx == 0 else 0.30)
                for idx, line in enumerate(detail_lines)
            ]
            info_y = self._draw_right_panel_info_pil(frame, x, info_y, lwc)

        # --- acceptance 情報行 (PIL / cv2 フォールバック) ---
        acceptance_lines = self._build_acceptance_detail_lines(acceptance_result)
        if acceptance_lines:
            color = self._acceptance_status_color(acceptance_result)
            meta_color = (185, 185, 185)
            lwc = [
                (line, color if idx == 0 else meta_color, 0.34 if idx == 0 else 0.30)
                for idx, line in enumerate(acceptance_lines)
            ]
            info_y = self._draw_right_panel_info_pil(frame, x, info_y, lwc)

        workflow_lines = self._build_workflow_detail_lines(workflow_status)
        if workflow_lines:
            workflow_color = (0, 220, 220)
            meta_color = (185, 185, 185)
            lwc = [
                (
                    line,
                    workflow_color if idx == 0 else meta_color,
                    0.34 if idx == 0 else 0.30,
                )
                for idx, line in enumerate(workflow_lines)
            ]
            info_y = self._draw_right_panel_info_pil(frame, x, info_y, lwc)

        # --- lab_rel_l 警告行 (PIL / cv2 フォールバック) ---
        if lab_rel_l_warn:
            info_y = self._draw_right_panel_info_pil(
                frame, x, info_y,
                [("L WARN", (0, 220, 220), 0.30)],
            )

    def draw_quality_warnings(self, frame, x, y, warnings_ref, warnings_tar):
        """品質警告を短い自然言語で描画する。"""
        colors = {
            "OVEREXPOSED": (0, 0, 255),
            "UNDEREXPOSED": (255, 100, 0),
            "NON_UNIFORM": (0, 230, 230),
        }
        labels = {
            "OVEREXPOSED": "Too bright",
            "UNDEREXPOSED": "Too dark",
            "NON_UNIFORM": "Non-uniform",
        }
        ly = self.ly
        dy = 0
        for w in warnings_ref[:1]:
            color = colors.get(w, (0, 0, 255))
            cv2.putText(frame, f"! Ref: {labels.get(w, w)}", (x, y + dy), self.FONT, ly.font_warning, color, 1, self.LINE_TYPE)
            dy += ly.warning_row_spacing
        for w in warnings_tar[:1]:
            color = colors.get(w, (0, 0, 255))
            cv2.putText(frame, f"! Target: {labels.get(w, w)}", (x, y + dy), self.FONT, ly.font_warning, color, 1, self.LINE_TYPE)
            dy += ly.warning_row_spacing

    def draw_side_panel(
        self, frame, Lab_abs, Lab_rel, U_L, U_a, U_b, dE00_target, dE00_ref,
        state, dark_loaded, exposure_us, gain, ae_on, cv_L, sigma_L, sigma_a, sigma_b,
        warnings_ref, warnings_tar, tar_luma,
        ref_raw_mean_12bit=None, ref_raw_in_band=False, ref_center_edge_diff_pct=None,
        ref_L_raw=None, ref_L_corr=None, bright_scale=1.0, ref_roi_raw=None,
        flat_loaded=False, stability_tracker=None, ref_rgb=None,
        center_edge_warn_pct: float = 3.0, mode=None, ratio=None, drift_tracker=None,
        ref_scale: "float | None" = None,
        ref_scale_warn: bool = False,
        gray_absolute_result: "dict | None" = None,
        acceptance_result: "dict | None" = None,
        lab_rel_l_warn: bool = False,
        flat_global_p90: "float | None" = None,
        flat_ref_p95: "float | None" = None,
        workflow_status: "dict | None" = None,
    ):
        """右側パネル全体を描画する。"""
        px = self.LEFT_PANEL_WIDTH + 810
        del dark_loaded, exposure_us, gain, ae_on
        del ref_L_raw, ref_L_corr, bright_scale
        del U_L, U_a, U_b, sigma_L, sigma_a, sigma_b
        del flat_loaded, ref_rgb, cv_L
        del ref_raw_mean_12bit, ref_raw_in_band
        del ref_center_edge_diff_pct, center_edge_warn_pct, state
        del flat_global_p90, flat_ref_p95

        ly = self.ly
        panel_h = self.config.camera.display_size[1]
        ht = ly.hdr_gap_text
        hg = ly.hdr_gap_graph
        hist_h = ly.hist_height
        ts_h = ly.ts_height
        drift_h = ly.drift_height
        gw = ly.graph_width
        has_mode_bar = mode is not None
        has_ref_signal = stability_tracker is not None and stability_tracker.count > 0
        layout = self._compute_side_panel_layout(
            panel_h=panel_h,
            has_mode_bar=has_mode_bar,
            has_ref_signal=has_ref_signal,
            gray_absolute_result=gray_absolute_result,
            acceptance_result=acceptance_result,
            lab_rel_l_warn=lab_rel_l_warn,
            workflow_status=workflow_status,
        )
        y = layout["top_y"]

        # --- Mode indicator bar ---
        if mode is not None:
            y = ly.mode_bar_top
            bar_w = ly.graph_width
            bar_h = ly.mode_bar_height
            if mode.is_lab():
                bar_color = (180, 80, 40)
                mode_text = "Mode: Lab"
            else:
                bar_color = (40, 160, 40)
                mode_text = "Mode: LinearRGB"
            cv2.rectangle(frame, (px, y), (px + bar_w, y + bar_h), bar_color, -1)
            cv2.putText(frame, mode_text, (px + 6, y + bar_h - 8), self.FONT, 0.55, (255, 255, 255), 1, self.LINE_TYPE)
            y = layout["top_y"]

        # --- Block A: Measurement ---
        y = layout["measurement_y"]
        self.draw_section_header(frame, px, y, "Measurement")
        self.draw_measurement(
            frame,
            px,
            y + ht,
            Lab_abs,
            Lab_rel,
            dE00_ref,
            dE00_target,
            mode=mode,
            ratio=ratio,
            gray_absolute_result=gray_absolute_result,
            acceptance_result=acceptance_result,
            lab_rel_l_warn=lab_rel_l_warn,
            workflow_status=workflow_status,
        )

        # --- Block B: Ref Exposure ---
        y = layout["ref_exposure_y"]
        self.draw_section_header(frame, px, y, "Ref Exposure")
        if ref_roi_raw is not None:
            self.draw_ref_histogram(frame, px, y + hg, ref_roi_raw, width=gw, height=hist_h)
        else:
            self.draw_target_histogram(frame, px, y + hg, tar_luma, width=gw, height=hist_h)

        # --- Block C: Ref Drift Guard ---
        if layout["ref_signal_y"] is not None:
            y = layout["ref_signal_y"]
            self.draw_section_header(frame, px, y, "Ref Stability")
            self.draw_stability_timeseries(frame, px, y + hg, stability_tracker, width=gw, height=ts_h)

        # --- Block D: Correction Applied ---
        y = layout["spectral_drift_y"]
        self.draw_section_header(frame, px, y, "Correction Applied")
        self.draw_spectral_drift_graph(
            frame,
            px,
            y + hg,
            drift_tracker,
            width=gw,
            height=drift_h,
            ref_scale=ref_scale,
            ref_scale_warn=ref_scale_warn,
        )
