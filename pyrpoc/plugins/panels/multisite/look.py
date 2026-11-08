"""How units are drawn on the dish. The dish is a fluorescence image, dark in every theme, so
outlines take fixed colours that read on it; the unit being worked on and new ones take the
theme's accent."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor

from .state.standing import Standing

WAITING = QColor(232, 234, 240)
LOST = QColor(255, 150, 60)
PAUSED = QColor(150, 152, 162)
DONE = QColor(105, 108, 118)
REFUSED = QColor(240, 90, 90)


@dataclass(frozen=True)
class Stroke:
    colour: QColor
    dashed: bool


def stroke_of(standing: Standing, accent: QColor) -> Stroke:
    match standing:
        case Standing.RUNNING:
            return Stroke(accent, False)
        case Standing.WAITING:
            return Stroke(WAITING, False)
        case Standing.LOST:
            return Stroke(LOST, False)
        case Standing.PAUSED:
            return Stroke(PAUSED, True)
        case Standing.DONE:
            return Stroke(DONE, False)
        case Standing.DRAWN:
            return Stroke(accent, True)
        case Standing.REFUSED:
            return Stroke(REFUSED, True)


def faded(colour: QColor, alpha: int) -> QColor:
    return QColor(colour.red(), colour.green(), colour.blue(), alpha)
