from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QFile, QSettings, QStandardPaths, QTextStream
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from . import breeze_all
from .mcolors import MCOLORS, MColor, recolor

_SETTINGS_ORG = "pyrpoc"
_SETTINGS_APP = "pyrpoc"
_SETTINGS_KEY_THEME_MODE = "ui/theme_mode"
DEFAULT_THEME = "dark-blueberry"
available_themes = [f"{mode}-{color.key}" for mode in ("dark", "light") for color in MCOLORS]
# The bundled stylesheets are recolored from these pink ones.
_TEMPLATES = {"dark": "dark-pink", "light": "light-pink"}


_ICON_URL_RE = re.compile(r"url\(:/([\w-]+)/([\w.-]+\.svg)\)")
_RGBA_RE = re.compile(r"rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")


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
        raise RuntimeError(f"breeze stylesheet {qss_path} missing expected rule: {pattern}")
    return match.group(1)


@dataclass(frozen=True)
class _SheetColors:
    """The colors a stylesheet gives the window, its inputs, disabled text and selections."""

    window: QColor
    window_text: QColor
    base: QColor
    text: QColor
    border: QColor
    disabled_text: QColor
    highlight: QColor
    highlighted_text: QColor


def _sheet_colors(qss_text: str, qss_path: str) -> _SheetColors:
    """Read the window, input, disabled and selection colors out of *qss_text*."""
    widget_block = _require(r"QWidget\s*\{([^}]*)\}", qss_text, qss_path)
    lineedit_block = _require(r"QLineEdit\s*\{([^}]*)\}", qss_text, qss_path)
    disabled_block = _require(r"QWidget:disabled\s*\{([^}]*)\}", qss_text, qss_path)

    def color(pattern: str, block: str) -> QColor:
        return _parse_qss_color(_require(pattern, block, qss_path))

    return _SheetColors(
        window=color(r"background-color:\s*([^;]+);", widget_block),
        window_text=color(r"(?<!-)color:\s*([^;]+);", widget_block),
        base=color(r"background-color:\s*([^;]+);", lineedit_block),
        text=color(r"(?<!-)color:\s*([^;]+);", lineedit_block),
        border=color(r"border:\s*[^;]*solid\s+([^;\s]+);", lineedit_block),
        disabled_text=color(r"(?<!-)color:\s*([^;]+);", disabled_block),
        highlight=color(r"selection-background-color:\s*([^;]+);", widget_block),
        highlighted_text=color(r"selection-color:\s*([^;]+);", widget_block),
    )


def _role_colors(sheet: _SheetColors) -> dict[QPalette.ColorRole, QColor]:
    """A color for every palette role, from the stylesheet's colors."""
    # Light themes select in a see-through accent; links and the accent itself are solid.
    accent = QColor(sheet.highlight)
    accent.setAlpha(255)
    role = QPalette.ColorRole
    return {
        role.Window: sheet.window,
        role.WindowText: sheet.window_text,
        role.Button: sheet.window,
        role.ButtonText: sheet.window_text,
        role.Base: sheet.base,
        # The stylesheet gives item views the window color as their alternate rows.
        role.AlternateBase: sheet.window,
        role.Text: sheet.text,
        role.BrightText: QColor("#ffffff"),
        role.PlaceholderText: sheet.disabled_text,
        role.ToolTipBase: sheet.window,
        role.ToolTipText: sheet.window_text,
        role.Light: sheet.window.lighter(150),
        role.Midlight: sheet.border,
        role.Mid: sheet.border,
        role.Dark: sheet.window.darker(150),
        role.Shadow: sheet.window.darker(300),
        role.Highlight: sheet.highlight,
        role.HighlightedText: sheet.highlighted_text,
        role.Accent: accent,
        role.Link: accent,
        role.LinkVisited: accent,
    }


def _derive_palette(qss_text: str, qss_path: str) -> QPalette:
    """A QPalette with every role set from the colors in *qss_text*."""
    # Widgets styled with ``palette(...)`` tokens only re-theme if the palette tracks the
    # stylesheet, so it is derived from the same text. Every role is set: one left out falls
    # through to the OS palette, whose dark- or light-mode colors then show in the other mode.
    sheet = _sheet_colors(qss_text, qss_path)
    palette = QPalette()
    for role, color in _role_colors(sheet).items():
        palette.setColor(role, color)
    # Icons drawn from the palette dim with their button only if this is set.
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.Text,
    ):
        palette.setColor(disabled, role, sheet.disabled_text)
    return palette


class ThemeController:
    def __init__(self, app: QApplication):
        self.app = app
        breeze_all.qInitResources()
        self.settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    def get_saved_mode(self) -> str:
        """The saved theme, or the default if settings hold something unknown."""
        raw = self.settings.value(_SETTINGS_KEY_THEME_MODE, DEFAULT_THEME)
        theme = str(raw).strip().lower()
        if theme in available_themes:
            return theme
        return DEFAULT_THEME

    def apply_saved_or_default(self) -> str:
        return self.apply(self.get_saved_mode(), persist=False)

    def load_stylesheet(self, theme: str) -> str:
        mode, key = theme.split("-")
        color = next(c for c in MCOLORS if c.key == key)
        qss_path = f":/{_TEMPLATES[mode]}/stylesheet.qss"

        file = QFile(qss_path)
        if not file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            raise RuntimeError(f"failed to open breeze stylesheet: {qss_path}")

        qss = recolor(QTextStream(file).readAll(), color, dark=mode == "dark")
        return self._with_recolored_icons(qss, theme, color, dark=mode == "dark")

    def _with_recolored_icons(self, qss: str, theme: str, color: MColor, *, dark: bool) -> str:
        """Point each icon url at an accent-colored copy of it in the cache dir."""
        cache = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation))
        icon_dir = cache / "icons" / theme
        icon_dir.mkdir(parents=True, exist_ok=True)

        def swap(match: re.Match[str]) -> str:
            resource = QFile(f":/{match.group(1)}/{match.group(2)}")
            # The bundled stylesheet names a few icons it does not ship; leave those as they are.
            if not resource.open(QFile.OpenModeFlag.ReadOnly):
                return match.group(0)
            svg = bytes(resource.readAll().data()).decode()
            target = icon_dir / match.group(2)
            target.write_text(recolor(svg, color, dark=dark))
            return f"url({target.as_posix()})"

        return _ICON_URL_RE.sub(swap, qss)

    def apply(self, theme: str, *, persist: bool) -> str:
        if persist:
            self.settings.setValue(_SETTINGS_KEY_THEME_MODE, theme)
        base_qss = self.load_stylesheet(theme)
        self.app.setPalette(_derive_palette(base_qss, f"{theme} stylesheet"))
        self.app.setStyleSheet(base_qss)
        return theme
