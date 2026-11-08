"""The next few routines to run, each marked on its unit with its place in line and a ring that
closes as it comes due."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPen, QTransform
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsScene, QStyleOptionGraphicsItem, QWidget

from ... import look
from ...state.narration import duration
from ...state.run_order import QueueRow, Section
from ...state.snapshot import Snapshot

NEXT_UP_Z = 25.0
# How many routines are marked.
COUNT = 3
# The ring is full this long before a routine is due, and closes from there.
HORIZON_S = 60.0
RADIUS = 11.0
FONT = look.font(8.0, True)
SMALL = look.font(7.0, False)


class Countdown(QGraphicsItem):
    """One routine's marker, the same size at any zoom, centred on its position."""

    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.place = 1
        self.left_s = 0.0
        self.ready = False

    def boundingRect(self) -> QRectF:
        return QRectF(-RADIUS - 3, -RADIUS - 3, 2 * RADIUS + 6, 2 * RADIUS + 20)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ) -> None:
        if painter is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ring = QRectF(-RADIUS, -RADIUS, 2 * RADIUS, 2 * RADIUS)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(look.faded(look.GLASS, 210))
        painter.drawEllipse(ring)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(look.faded(look.ACCENT, 70), 2.0))
        painter.drawEllipse(ring)
        closed = 1.0 if self.ready else 1.0 - min(self.left_s, HORIZON_S) / HORIZON_S
        painter.setPen(QPen(look.ACCENT, 2.4))
        painter.drawArc(ring, 90 * 16, -int(closed * 360 * 16))
        painter.setFont(FONT)
        painter.setPen(look.TEXT)
        painter.drawText(ring, Qt.AlignmentFlag.AlignCenter, str(self.place))
        if not self.ready:
            painter.setFont(SMALL)
            painter.setPen(look.TEXT_DIM)
            under = QRectF(-RADIUS - 12, RADIUS + 1, 2 * RADIUS + 24, 14)
            painter.drawText(under, Qt.AlignmentFlag.AlignCenter, duration(self.left_s))


class NextUp:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.marks: list[Countdown] = []

    def show(self, snapshot: Snapshot, on: bool) -> None:
        coming = snapshot.coming(COUNT) if on else ()
        while len(self.marks) < len(coming):
            mark = Countdown()
            mark.setZValue(NEXT_UP_Z)
            self.scene.addItem(mark)
            self.marks.append(mark)
        # Several routines of one unit sit side by side rather than on top of each other.
        beside: dict[int, int] = {}
        for index, mark in enumerate(self.marks):
            mark.setVisible(index < len(coming))
            if index < len(coming):
                self.place(mark, coming[index], index + 1, snapshot, beside)

    def place(
        self, mark: Countdown, row: QueueRow, place: int, snapshot: Snapshot, beside: dict[int, int]
    ) -> None:
        unit = snapshot.unit(row.status.unit)
        if unit is None:
            mark.hide()
            return
        right = max(x for x, _ in unit.outline)
        top = min(y for _, y in unit.outline)
        offset = beside.get(unit.number, 0)
        beside[unit.number] = offset + 1
        mark.setPos(QPointF(right, top))
        # An item that ignores the zoom applies its own transform in screen pixels.
        mark.setTransform(QTransform.fromTranslate(RADIUS + 4 + offset * (2 * RADIUS + 6), 0))
        mark.place = place
        mark.ready = row.section is not Section.UPCOMING
        mark.left_s = max(0.0, row.status.due_s - snapshot.now_s)
        mark.update()
