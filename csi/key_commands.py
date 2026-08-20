"""Preview-freeze command helpers for KeyHandler."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreviewFreezeDecision:
    """Side-effect instructions for the preview-freeze key branch."""

    set_pending: bool
    overlay_title: str | None = None
    overlay_message: str | None = None
    overlay_wait_sec: float = 1.2


def is_preview_freeze_key(key: int) -> bool:
    """Return whether key is the preview-freeze hotkey."""
    return key == ord("g")


__all__ = ["PreviewFreezeDecision", "is_preview_freeze_key"]
