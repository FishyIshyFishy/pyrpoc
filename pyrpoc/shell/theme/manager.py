from __future__ import annotations

import re

from PyQt6.QtCore import QFile, QSettings, QTextStream
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from . import breeze_all

_SETTINGS_ORG = 'pyrpoc'
_SETTINGS_APP = 'pyrpoc'
_SETTINGS_KEY_THEME_MODE = 'ui/theme_mode'
DEFAULT_THEME = 'dark-pink'
available_breeze_themes = [
    'dark-blue', 
    'dark-blue-alt', 
    'dark-cyan', 
    'dark-cyan-alt', 
    'dark-green', 
    'dark-green-alt', 
    'dark-pink', 
    'dark-pink-alt', 
    'dark-purple', 
    'dark-purple-alt', 
    'dark-red', 
    'dark-red-alt', 
    'light-blue', 
    'light-blue-alt', 
    'light-cyan', 
    'light-cyan-alt', 
    'light-green', 
    'light-green-alt', 
    'light-pink', 
    'light-pink-alt', 
    'light-purple', 
    'light-purple-alt', 
    'light-red', 
    'light-red-alt'
]



_RGBA_RE = re.compile(
    r'rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'
)


def _parse_qss_color(value: str) -> QColor:
    """Parse a QSS color literal (``#rrggbb`` or ``rgba(r, g, b, a)``)."""
    value = value.strip()
    match = _RGBA_RE.match(value)
    if match:
        r, g, b, a = match.groups()
        return QColor(int(float(r)), int(float(g)), int(float(b)), round(float(a) * 255))
    return QColor(value)


def _require(pattern: str, text: str, qss_path: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f'breeze stylesheet {qss_path} missing expected rule: {pattern}')
    return match.group(1)


def _derive_palette(qss_text: str, qss_path: str) -> QPalette:
    """Build a QPalette matching the colors baked into *qss_text*.

    The bundled Breeze stylesheets recolor most widgets with literal hex
    values on type selectors, so those widgets re-theme on their own via
    ``QApplication.setStyleSheet``. Anything styled against ``palette(...)``
    tokens (e.g. the card widgets in ``shell/cards.py``) only re-themes if
    the app's QPalette is kept in sync -- which this derives directly from
    the same stylesheet text so there is a single source of truth per theme.
    """
    widget_block = _require(r'QWidget\s*\{([^}]*)\}', qss_text, qss_path)
    lineedit_block = _require(r'QLineEdit\s*\{([^}]*)\}', qss_text, qss_path)

    window_bg = _parse_qss_color(_require(r'background-color:\s*([^;]+);', widget_block, qss_path))
    window_fg = _parse_qss_color(_require(r'(?<!-)color:\s*([^;]+);', widget_block, qss_path))
    highlight = _parse_qss_color(
        _require(r'selection-background-color:\s*([^;]+);', widget_block, qss_path)
    )
    highlighted_text = _parse_qss_color(
        _require(r'selection-color:\s*([^;]+);', widget_block, qss_path)
    )

    base = _parse_qss_color(_require(r'background-color:\s*([^;]+);', lineedit_block, qss_path))
    text = _parse_qss_color(_require(r'(?<!-)color:\s*([^;]+);', lineedit_block, qss_path))
    border = _parse_qss_color(
        _require(r'border:\s*[^;]*solid\s+([^;\s]+);', lineedit_block, qss_path)
    )

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, window_bg)
    palette.setColor(QPalette.ColorRole.WindowText, window_fg)
    palette.setColor(QPalette.ColorRole.Button, window_bg)
    palette.setColor(QPalette.ColorRole.ButtonText, window_fg)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Mid, border)
    palette.setColor(QPalette.ColorRole.Midlight, border)
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, highlighted_text)
    return palette


class ThemeController:
    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    def get_saved_mode(self) -> str:
        raw = self.settings.value(_SETTINGS_KEY_THEME_MODE, DEFAULT_THEME)
        theme = str(raw).strip().lower()
        if theme in available_breeze_themes:
            return theme
        return DEFAULT_THEME

    def apply_saved_or_default(self) -> str:
        return self.apply(self.get_saved_mode(), persist=False)

    def load_breeze_stylesheet(self, theme: str) -> str:
        qss_path = f':/{theme}/stylesheet.qss'

        file = QFile(qss_path)
        ok = file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
        if not ok:
            raise RuntimeError(f'failed to open breeze stylesheet: {qss_path}')

        stream = QTextStream(file)
        return stream.readAll()

    def apply(self, theme: str, persist: bool = True) -> str:
        normalized = theme.strip().lower()
        if normalized not in available_breeze_themes:
            normalized = DEFAULT_THEME
        selected_theme = normalized

        if persist:
            self.settings.setValue(_SETTINGS_KEY_THEME_MODE, selected_theme)

        base_qss = self.load_breeze_stylesheet(selected_theme)
        self.app.setPalette(_derive_palette(base_qss, f':/{selected_theme}/stylesheet.qss'))
        self.app.setStyleSheet(base_qss)
        return selected_theme