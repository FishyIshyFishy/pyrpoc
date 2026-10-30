from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication

from pyrpoc.app.theme.mfruit import ACCENTS
from pyrpoc.app.theme.palette import (
    DEFAULT_ACCENT,
    DEFAULT_MODE,
    Mode,
    Theme,
    build_palette,
)
from pyrpoc.app.theme.qss import build_qss

_SETTINGS_ORG = "pyrpoc"
_SETTINGS_APP = "pyrpoc"
_KEY_MODE = "ui/mode"
_KEY_ACCENT = "ui/accent"
_BASE_POINT_SIZE = 10.0


class ThemeController:
    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        cache = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
        self.glyph_dir = Path(cache) / "theme_glyphs"
        # One size for the app; the stylesheet scales headings and captions from it.
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        font.setPointSizeF(_BASE_POINT_SIZE)
        app.setFont(font)

    def get_saved_theme(self) -> Theme:
        """The saved theme, with the default for anything settings do not hold
        validly (including the old Breeze names)."""
        mode_name = str(self.settings.value(_KEY_MODE, DEFAULT_MODE.value))
        accent = str(self.settings.value(_KEY_ACCENT, DEFAULT_ACCENT))
        mode = Mode(mode_name) if mode_name in {m.value for m in Mode} else DEFAULT_MODE
        return Theme(mode, accent if accent in ACCENTS else DEFAULT_ACCENT)

    def apply_saved_or_default(self) -> Theme:
        return self.apply(self.get_saved_theme(), persist=False)

    def apply(self, theme: Theme, *, persist: bool) -> Theme:
        self.app.setPalette(build_palette(theme))
        self.app.setStyleSheet(build_qss(theme, self.glyph_dir))
        if persist:
            self.settings.setValue(_KEY_MODE, theme.mode.value)
            self.settings.setValue(_KEY_ACCENT, theme.accent)
        return theme
