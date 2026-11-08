"""The panel's colours and type: one colour per phase and per unit state, and the overlay's
glass."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QFont

from pyrpoc.plugins.program_io.outputs import UnitState

from .state.history import Tone
from .state.phase import Phase

PHASE = {
    Phase.MOVING: QColor(90, 200, 255),
    Phase.FOCUSING: QColor(255, 190, 70),
    Phase.SCANNING: QColor(100, 235, 140),
}
STATE = {
    UnitState.WAITING: QColor(235, 238, 245),
    UnitState.PAUSED: QColor(140, 145, 160),
    UnitState.LOST: QColor(255, 130, 70),
    UnitState.DONE: QColor(105, 110, 120),
    UnitState.REMOVED: QColor(80, 80, 80),
}
# Kept for how things went, never for a channel or a phase, and always shown with its mark.
TONE = {
    Tone.GOOD: QColor(110, 210, 150),
    Tone.WARNING: QColor(255, 175, 60),
    Tone.BAD: QColor(255, 105, 105),
    Tone.CHANGE: QColor(110, 175, 255),
}
PLAIN_OUTLINE = QColor(235, 238, 245)
IDLE = QColor(170, 175, 190)
WARNING = QColor(255, 130, 70)
ACCENT = QColor(110, 175, 255)

GLASS = QColor(16, 18, 24, 172)
GLASS_EDGE = QColor(255, 255, 255, 38)
# Laid over the dish while the overlay is up, just enough to say the dish is not taking input.
VEIL = QColor(0, 0, 0, 46)
TEXT = QColor(236, 239, 246)
TEXT_DIM = QColor(150, 157, 172)
SELECTED_ROW = QColor(110, 175, 255, 58)


def font(points: float, bold: bool) -> QFont:
    face = QFont()
    face.setPointSizeF(points)
    face.setBold(bold)
    return face


def faded(colour: QColor, alpha: int) -> QColor:
    return QColor(colour.red(), colour.green(), colour.blue(), alpha)
