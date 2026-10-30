"""A theme is a mode (neutral light or dark surfaces) and an accent; the
QPalette and the stylesheet are both derived from it, so they cannot disagree.

Palette roles carry a fixed meaning that ``palette(...)`` stylesheets rely on:
Midlight is a border, Mid is muted text, Base is an input or card fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6.QtGui import QColor, QPalette

from pyrpoc.app.theme.mfruit import ACCENTS


class Mode(str, Enum):
    LIGHT = "light"
    DARK = "dark"


@dataclass(frozen=True)
class Surfaces:
    window: str
    base: str
    raised: str
    border: str
    text: str
    muted: str
    disabled: str


# Neutral grays, with no blue cast: a dark mode should read as dark, not slate.
SURFACES: dict[Mode, Surfaces] = {
    Mode.DARK: Surfaces(
        window="#1a1a1a",
        base="#121212",
        raised="#2a2a2a",
        border="#383838",
        text="#e8e8e8",
        muted="#9a9a9a",
        disabled="#5e5e5e",
    ),
    Mode.LIGHT: Surfaces(
        window="#f3f3f3",
        base="#ffffff",
        raised="#e6e6e6",
        border="#c9c9c9",
        text="#1b1b1b",
        muted="#6a6a6a",
        disabled="#a6a6a6",
    ),
}

DEFAULT_MODE = Mode.DARK
DEFAULT_ACCENT = "mStrawberry"


@dataclass(frozen=True)
class Theme:
    mode: Mode
    accent: str

    @property
    def surfaces(self) -> Surfaces:
        return SURFACES[self.mode]

    @property
    def accent_color(self) -> QColor:
        color = QColor(ACCENTS[self.accent])
        # The bright dark-mode shades wash out on a light window.
        return color.darker(125) if self.mode is Mode.LIGHT else color

    @property
    def on_accent(self) -> QColor:
        """Text readable on the accent fill."""
        accent = self.accent_color
        luma = 0.299 * accent.red() + 0.587 * accent.green() + 0.114 * accent.blue()
        return QColor("#111111") if luma > 150 else QColor("#ffffff")


def build_palette(theme: Theme) -> QPalette:
    s = theme.surfaces
    accent = theme.accent_color
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: s.window,
        QPalette.ColorRole.WindowText: s.text,
        QPalette.ColorRole.Base: s.base,
        QPalette.ColorRole.AlternateBase: s.raised,
        QPalette.ColorRole.Text: s.text,
        QPalette.ColorRole.Button: s.raised,
        QPalette.ColorRole.ButtonText: s.text,
        QPalette.ColorRole.BrightText: s.text,
        QPalette.ColorRole.ToolTipBase: s.raised,
        QPalette.ColorRole.ToolTipText: s.text,
        QPalette.ColorRole.PlaceholderText: s.muted,
        QPalette.ColorRole.Light: s.raised,
        QPalette.ColorRole.Midlight: s.border,
        QPalette.ColorRole.Mid: s.muted,
        QPalette.ColorRole.Dark: s.border,
        QPalette.ColorRole.Shadow: "#000000",
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, theme.on_accent)
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.LinkVisited, accent)
    # Icons and text dim with their widget only if the disabled group differs.
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.Text,
    ):
        palette.setColor(disabled, role, QColor(s.disabled))
    return palette
