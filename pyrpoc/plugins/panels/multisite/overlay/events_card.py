"""``EventsCard``: what the loop has just done, newest first. Clicking an event about a unit
chooses that unit."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetricsF, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QSizePolicy, QWidget

from pyrpoc.plugins.program_io.outputs import LoopEvent

from .. import look
from ..state.history import notable, tone_of, unit_of
from ..state.narration import duration
from ..state.snapshot import Snapshot
from .card import Anchor, Card

ROW_H = 24
# More than a tall card shows, so it is always full.
KEPT = 60
TEXT = look.font(8.5, False)
MARK = look.font(9.0, True)


def age(seconds: float) -> str:
    return "now" if seconds < 1 else f"{duration(seconds)} ago"


class EventList(QWidget):
    picked = pyqtSignal(int)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.events: list[LoopEvent] = []
        self.now_s = 0.0
        self.hovered: int | None = None

    def show_events(self, events: list[LoopEvent], now_s: float) -> None:
        self.events, self.now_s = events, now_s
        self.update()

    def row_at(self, y: float) -> int | None:
        index = int(y // ROW_H)
        return index if 0 <= index < len(self.events) else None

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        index = self.row_at(event.position().y())
        clickable = index is not None and unit_of(self.events[index]) is not None
        self.hovered = index if clickable else None
        cursor = Qt.CursorShape.PointingHandCursor if clickable else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
        self.hovered = None
        self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        index = self.row_at(event.position().y())
        unit = None if index is None else unit_of(self.events[index])
        if unit is not None:
            self.picked.emit(unit)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rows = min(len(self.events), self.height() // ROW_H)
        for index in range(rows):
            self.paint_row(painter, QRectF(0, index * ROW_H, self.width(), ROW_H), index)

    def paint_row(self, painter: QPainter, box: QRectF, index: int) -> None:
        event = self.events[index]
        if index == self.hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(look.SELECTED_ROW)
            painter.drawRoundedRect(box, 6, 6)
        tone = tone_of(event)
        painter.setFont(MARK)
        painter.setPen(look.TONE[tone])
        painter.drawText(box.adjusted(6, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, tone.value)
        painter.setFont(TEXT)
        when = age(self.now_s - event.time_s)
        metrics = QFontMetricsF(TEXT)
        when_w = metrics.horizontalAdvance(when) + 8
        painter.setPen(look.TEXT_DIM)
        painter.drawText(
            box.adjusted(0, 0, -6, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            when,
        )
        text_box = box.adjusted(24, 0, -when_w - 6, 0)
        text = metrics.elidedText(event.text(), Qt.TextElideMode.ElideRight, text_box.width())
        painter.setPen(look.TEXT)
        painter.drawText(text_box, Qt.AlignmentFlag.AlignVCenter, text)


class EventsCard(Card):
    # A unit whose event was clicked.
    picked = pyqtSignal(int)

    def __init__(self, parent: QWidget, width: int):
        super().__init__(parent, "Recent", Anchor.RIGHT, width, 2)
        self.list = EventList(self)
        self.list.picked.connect(self.picked)
        self.content.addWidget(self.list, 1)

    def show_snapshot(self, snapshot: Snapshot) -> None:
        self.list.show_events(notable(snapshot.events, KEPT), snapshot.now_s)
