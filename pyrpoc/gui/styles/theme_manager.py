from __future__ import annotations

from typing import Literal

import qdarktheme
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

ThemeMode = Literal["system", "dark", "light"]

_SETTINGS_ORG = "pyrpoc"
_SETTINGS_APP = "pyrpoc"
_SETTINGS_KEY_THEME_MODE = "ui/theme_mode"


class ThemeController:
    """Single source of truth for app theme mode and application."""

    AVAILABLE_MODES: tuple[ThemeMode, ...] = ("system", "dark", "light")

    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    def get_available_modes(self) -> list[ThemeMode]:
        return list(self.AVAILABLE_MODES)

    def get_saved_mode(self) -> ThemeMode:
        raw = self.settings.value(_SETTINGS_KEY_THEME_MODE, "system")
        mode = str(raw).strip().lower()
        if mode in self.AVAILABLE_MODES:
            return mode  # type: ignore[return-value]
        return "system"

    def apply_saved_or_default(self) -> ThemeMode:
        return self.apply(self.get_saved_mode(), persist=False)

    def apply(self, mode: str, persist: bool = True) -> ThemeMode:
        normalized = mode.strip().lower()
        if normalized not in self.AVAILABLE_MODES:
            normalized = "system"
        selected_mode: ThemeMode = normalized  # type: ignore[assignment]

        if persist:
            self.settings.setValue(_SETTINGS_KEY_THEME_MODE, selected_mode)

        qdarktheme_mode = "auto" if selected_mode == "system" else selected_mode
        qdarktheme.setup_theme(qdarktheme_mode)
