"""``ChannelTrace``: one channel of one measurement over time, small. The channel's name sits on
the left and its latest value on the right; hovering reads off the point under the cursor."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PyQt6.QtWidgets import QWidget

from .. import look
from ..state.narration import duration

HEIGHT = 34
NAME_W, VALUE_W = 46, 108
NAME = look.font(8.0, True)
VALUE = look.font(8.5, False)
# The newest point is marked this wide, so where the trace ends is plain.
DOT = 8.0


def value_text(value: float) -> str:
    return f"{value:.3g}"


class ChannelTrace(QWidget):
    def __init__(self, parent: QWidget, name: str, colour: QColor):
        super().__init__(parent)
        self.setFixedHeight(HEIGHT)
        self.setMouseTracking(True)
        self.name, self.colour = name, colour
        self.points: list[tuple[float, float]] = []
        self.hover_x: float | None = None

    def show_points(self, points: list[tuple[float, float]]) -> None:
        """``(seconds since the loop started, value)`` pairs, oldest first."""
        if points != self.points:
            self.points = points
            self.update()

    def plot_rect(self) -> QRectF:
        return QRectF(NAME_W, 6, self.width() - NAME_W - VALUE_W - 8, HEIGHT - 12)

    def place(self, box: QRectF) -> list[QPointF]:
        """Each point in the plot, time across and value up, the range filling the box."""
        times = [t for t, _ in self.points]
        values = [v for _, v in self.points]
        t0, t1 = min(times), max(times)
        v0, v1 = min(values), max(values)
        span_t, span_v = (t1 - t0) or 1.0, (v1 - v0) or 1.0
        return [
            QPointF(
                box.left() + (t - t0) / span_t * box.width(),
                box.bottom() - (v - v0) / span_v * box.height(),
            )
            for t, v in self.points
        ]

    def nearest(self, placed: list[QPointF]) -> int:
        """The point nearest the cursor across, or the newest when not hovered."""
        if self.hover_x is None:
            return len(placed) - 1
        x = self.hover_x
        return min(range(len(placed)), key=lambda i: abs(placed[i].x() - x))

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self.hover_x = event.position().x()
            self.update()

    def leaveEvent(self, event: QEvent | None) -> None:
        self.hover_x = None
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(NAME)
        painter.setPen(look.TEXT_DIM)
        painter.drawText(QRectF(0, 0, NAME_W, HEIGHT), Qt.AlignmentFlag.AlignVCenter, self.name)
        box = self.plot_rect()
        painter.setPen(QPen(look.GLASS_EDGE, 1.0))
        painter.drawLine(box.bottomLeft(), box.bottomRight())
        if not self.points:
            return
        placed = self.place(box)
        self.paint_line(painter, placed)
        self.paint_reading(painter, placed, box)

    def paint_line(self, painter: QPainter, placed: list[QPointF]) -> None:
        path = QPainterPath(placed[0])
        for point in placed[1:]:
            path.lineTo(point)
        line = QPen(self.colour, 2.0)
        line.setCapStyle(Qt.PenCapStyle.RoundCap)
        line.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(line)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def paint_reading(self, painter: QPainter, placed: list[QPointF], box: QRectF) -> None:
        """The marked point and its value: the newest, or the one under the cursor."""
        index = self.nearest(placed)
        point = placed[index]
        if self.hover_x is not None:
            painter.setPen(QPen(look.faded(look.TEXT, 90), 1.0))
            painter.drawLine(QPointF(point.x(), box.top()), QPointF(point.x(), box.bottom()))
        # A ring of the card's glass round the dot keeps it apart from the line.
        painter.setPen(QPen(look.faded(look.GLASS, 255), 2.0))
        painter.setBrush(self.colour)
        painter.drawEllipse(point, DOT / 2, DOT / 2)
        t, value = self.points[index]
        text = value_text(value)
        if self.hover_x is not None:
            text = f"{text} · at {duration(t)}"
        painter.setFont(VALUE)
        painter.setPen(look.TEXT)
        reading = QRectF(self.width() - VALUE_W, 0, VALUE_W, HEIGHT)
        painter.drawText(reading, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, text)
