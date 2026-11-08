"""``Badge``: a small label on the dish that stays the same size at any zoom."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFontMetricsF, QPainter
from PyQt6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from ... import look

PAD_X, PAD_Y = 5.0, 2.0
FONT = look.font(8.5, True)


class Badge(QGraphicsItem):
    """Text on a dark rounded tag, its top-left at its position."""

    def __init__(self, text: str, colour: QColor):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.text, self.colour = text, colour
        self.size = self.measure(text)

    @staticmethod
    def measure(text: str) -> QRectF:
        width = QFontMetricsF(FONT).horizontalAdvance(text)
        height = QFontMetricsF(FONT).height()
        return QRectF(0, 0, width + 2 * PAD_X, height + 2 * PAD_Y)

    def set_look(self, text: str, colour: QColor) -> None:
        if (text, colour) == (self.text, self.colour):
            return
        self.prepareGeometryChange()
        self.text, self.colour, self.size = text, colour, self.measure(text)
        self.update()

    def boundingRect(self) -> QRectF:
        return self.size

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ) -> None:
        if painter is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 190))
        painter.drawRoundedRect(self.size, 4, 4)
        painter.setFont(FONT)
        painter.setPen(self.colour)
        painter.drawText(self.size, Qt.AlignmentFlag.AlignCenter, self.text)
