from __future__ import annotations

import re
from dataclasses import dataclass

from PyQt6.QtGui import QColor


@dataclass(frozen=True)
class MColor:
    """A named accent color and the text color that stays readable on top of it."""

    key: str
    label: str
    accent: str
    on_accent: str


MCOLORS = [
    MColor("banana", "mBanana", "#f2c744", "#1e1f22"),
    MColor("tangerine", "mTangerine", "#f28c38", "#1e1f22"),
    MColor("cherry", "mCherry", "#e5484d", "#ffffff"),
    MColor("mint", "mMint", "#3ecf8e", "#1e1f22"),
    MColor("blueberry", "mBlueberry", "#4f8df5", "#ffffff"),
    MColor("grape", "mGrape", "#a66be8", "#ffffff"),
]

# The bundled stylesheets are pink; these are the pink values that get swapped for an accent.
_PINK_ACCENT = ("#e639ba", "#df34df", "#e641e6")
_PINK_ACCENT_DARK = ("#b832b8", "#bb3d9a", "#682768", "#b82db8")
_PINK_ACCENT_LIGHT = ("#d044ac", "#da7cda", "#ff57ff")
_PINK_RGB = ("255, 0, 255", "218, 60, 218", "223, 52, 223", "184, 45, 184", "243, 73, 243")
_PINK_RGB_RE = re.compile(
    r"rgba\(\s*(?:" + "|".join(r.replace(", ", r"\s*,\s*") for r in _PINK_RGB) + r")\s*,"
)

# A conventional dark palette: the window is the deepest layer, and inputs and cards sit
# one step lighter so they read as raised against it.
_DARK_NEUTRALS = {
    "#31363b": "#1e1f22",
    "#2c3034": "#25272b",
    "#1d2023": "#2b2d31",
    "#76797c": "#70747b",
    "#626568": "#5a5d63",
    "#454a4f": "#3a3c40",
    "#454545": "#55575c",
    "#eff0f1": "#e3e5e8",
}


def recolor(qss: str, color: MColor, *, dark: bool) -> str:
    """The pink bundled stylesheet *qss* with *color* as its accent (and neutral dark greys)."""
    accent = QColor(color.accent)
    swaps = {pink: color.accent for pink in _PINK_ACCENT}
    swaps |= {pink: accent.darker(125).name() for pink in _PINK_ACCENT_DARK}
    swaps |= {pink: accent.lighter(115).name() for pink in _PINK_ACCENT_LIGHT}
    if dark:
        swaps |= _DARK_NEUTRALS
    pattern = re.compile("|".join(re.escape(old) for old in swaps), re.IGNORECASE)
    qss = pattern.sub(lambda m: swaps[m.group(0).lower()], qss)
    rgb = f"rgba({accent.red()}, {accent.green()}, {accent.blue()},"
    qss = _PINK_RGB_RE.sub(rgb, qss)
    return qss.replace("selection-color: #e3e5e8;", f"selection-color: {color.on_accent};")
