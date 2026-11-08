"""The corner of the dish that says in words what the loop is doing, any news, and the keys to
start from."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPaintEvent
from PyQt6.QtWidgets import QWidget

from .. import keys, look
from ..state.narration import doing_text, notice
from ..state.phase import phase_of
from ..state.snapshot import Snapshot

HINT = (
    f"Hold {keys.OVERLAY_NAME}: queue   ·   {keys.DRAW_NAME}: draw a unit   ·   "
    f"{keys.FIT_NAME}: whole dish"
)
PAD = 10
MAIN = look.font(10.0, True)
SMALL = look.font(8.5, False)


class StatusLine(QWidget):
    """Sized to its text; the canvas places it."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.lines: list[tuple[str, QColor, bool]] = []
        self.dot: QColor | None = None

    def show_snapshot(self, snapshot: Snapshot, drawn_notice: str | None, on: bool) -> None:
        news = drawn_notice or notice(snapshot)
        lines: list[tuple[str, QColor, bool]] = []
        if on:
            lines.append((doing_text(snapshot), look.TEXT, True))
        if news is not None:
            lines.append((news, look.WARNING, False))
        lines.append((HINT, look.TEXT_DIM, False))
        doing = snapshot.doing
        dot = look.PHASE[phase_of(doing)] if on and doing is not None else None
        if (lines, dot) != (self.lines, self.dot):
            self.lines, self.dot = lines, dot
            self.fit()
            self.update()

    def fit(self) -> None:
        width, height = 0, 0
        for text, _colour, bold in self.lines:
            metrics = QFontMetrics(MAIN if bold else SMALL)
            width = max(width, metrics.horizontalAdvance(text) + (16 if bold else 0))
            height += metrics.height() + 2
        self.resize(width + 2 * PAD, height + 2 * PAD - 2)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(look.GLASS)
        painter.drawRoundedRect(QRectF(self.rect()), 8, 8)
        y = PAD
        for text, colour, bold in self.lines:
            face = MAIN if bold else SMALL
            height = QFontMetrics(face).height()
            x = PAD
            if bold and self.dot is not None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self.dot)
                painter.drawEllipse(QRectF(x, y + height / 2 - 4, 8, 8))
            if bold:
                x += 16
            painter.setFont(face)
            painter.setPen(colour)
            painter.drawText(x, y, self.width() - x, height, Qt.AlignmentFlag.AlignLeft, text)
            y += height + 2
