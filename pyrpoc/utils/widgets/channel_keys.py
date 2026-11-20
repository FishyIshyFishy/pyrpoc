"""``ChannelKeys``: a joined block of numbered keys, one per channel, lit when chosen."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PyQt6.QtGui import QPalette as P
from PyQt6.QtWidgets import QWidget

PER_ROW = 8
KEY_W = 24
KEY_H = 22
RADIUS = 5


class ChannelKeys(QWidget):
    """Channels as keys in rows of eight: click a key to turn its channel on or off, or drag
    across keys to set them all the same way. The last lit key stays lit."""

    toggled = pyqtSignal()

    def __init__(self, num_channels: int, parent: QWidget):
        super().__init__(parent)
        self.num_channels = num_channels
        self.lit: set[int] = set()
        self.hovered: int | None = None
        # What a press made its key, so a drag sets the keys it crosses the same way.
        self.painting: bool | None = None
        self.rows = -(-num_channels // PER_ROW)
        self.setFixedSize(QSize(min(num_channels, PER_ROW) * KEY_W + 1, self.rows * KEY_H + 1))
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_lit(self, channels: tuple[int, ...]) -> None:
        self.lit = set(channels)
        self.update()

    def channels(self) -> tuple[int, ...]:
        return tuple(sorted(self.lit))

    def key_rect(self, index: int) -> QRectF:
        row, column = divmod(index, PER_ROW)
        return QRectF(column * KEY_W + 0.5, row * KEY_H + 0.5, KEY_W, KEY_H)

    def key_at(self, point: QPointF) -> int | None:
        column, row = int(point.x() // KEY_W), int(point.y() // KEY_H)
        index = row * PER_ROW + column
        inside = 0 <= column < PER_ROW and 0 <= row < self.rows
        return index if inside and index < self.num_channels else None

    def set_key(self, index: int, on: bool) -> None:
        # A scan reads at least one input, so the last lit key cannot go dark.
        if on == (index in self.lit) or (not on and self.lit == {index}):
            return
        self.lit ^= {index}
        self.update()
        self.toggled.emit()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is None or a0.button() != Qt.MouseButton.LeftButton:
            return
        index = self.key_at(a0.position())
        if index is None:
            return
        self.painting = index not in self.lit
        self.set_key(index, self.painting)

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is None:
            return
        index = self.key_at(a0.position())
        if index != self.hovered:
            self.hovered = index
            self.update()
        if self.painting is not None and index is not None:
            self.set_key(index, self.painting)

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        self.painting = None

    def leaveEvent(self, a0: QEvent | None) -> None:
        self.hovered = None
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outline = self.outline()
        painter.setClipPath(outline)
        self.paint_keys(painter)
        self.paint_dividers(painter)
        painter.setClipping(False)
        painter.setPen(QPen(self.palette().color(P.ColorRole.Mid), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(outline)
        self.paint_numbers(painter)

    def outline(self) -> QPainterPath:
        """The keys' outline: rounded outside, and stepped in where a last row is short."""
        path = QPainterPath()
        for index in range(self.num_channels):
            key = QPainterPath()
            key.addRect(self.key_rect(index))
            path = path.united(key)
        rounded = QPainterPath()
        rounded.addRoundedRect(path.boundingRect(), RADIUS, RADIUS)
        return path.intersected(rounded).simplified()

    def paint_keys(self, painter: QPainter) -> None:
        palette = self.palette()
        for index in range(self.num_channels):
            if index in self.lit:
                fill = palette.color(P.ColorRole.Highlight)
            elif index == self.hovered:
                fill = palette.color(P.ColorRole.Button)
            else:
                fill = palette.color(P.ColorRole.Base)
            painter.fillRect(self.key_rect(index), fill)

    def paint_dividers(self, painter: QPainter) -> None:
        """A line between keys; darker accent between two lit ones, so a run reads as one."""
        mid = self.palette().color(P.ColorRole.Mid)
        joined = self.palette().color(P.ColorRole.Highlight).darker(130)
        for index in range(self.num_channels):
            rect = self.key_rect(index)
            if index % PER_ROW:
                both = {index, index - 1} <= self.lit
                painter.setPen(QPen(joined if both else mid, 1))
                painter.drawLine(rect.topLeft(), rect.bottomLeft())
            if index >= PER_ROW:
                both = {index, index - PER_ROW} <= self.lit
                painter.setPen(QPen(joined if both else mid, 1))
                painter.drawLine(rect.topLeft(), rect.topRight())

    def paint_numbers(self, painter: QPainter) -> None:
        palette = self.palette()
        font = QFont(self.font())
        for index in range(self.num_channels):
            lit = index in self.lit
            font.setBold(lit)
            painter.setFont(font)
            painter.setPen(palette.color(P.ColorRole.HighlightedText if lit else P.ColorRole.Text))
            painter.drawText(self.key_rect(index), Qt.AlignmentFlag.AlignCenter, str(index))
