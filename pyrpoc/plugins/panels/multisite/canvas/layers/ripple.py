"""A ring that spreads once from a unit just located, so the eye lands on it."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsScene, QStyleOptionGraphicsItem, QWidget

from ... import look

RIPPLE_Z = 40.0
LASTS_S = 0.7
SMALLEST, LARGEST = 10.0, 70.0


class RippleItem(QGraphicsItem):
    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.share = 0.0

    def boundingRect(self) -> QRectF:
        return QRectF(-LARGEST - 2, -LARGEST - 2, 2 * LARGEST + 4, 2 * LARGEST + 4)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ) -> None:
        if painter is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = SMALLEST + (LARGEST - SMALLEST) * self.share
        painter.setPen(QPen(look.faded(look.ACCENT, int(255 * (1 - self.share))), 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0, 0), radius, radius)


class Ripple:
    def __init__(self, scene: QGraphicsScene):
        self.item = RippleItem()
        self.item.setZValue(RIPPLE_Z)
        self.item.hide()
        scene.addItem(self.item)
        self.began_s: float | None = None

    def start(self, at: QPointF, now_s: float) -> None:
        self.item.setPos(at)
        self.began_s = now_s
        self.item.show()

    def tick(self, now_s: float) -> None:
        if self.began_s is None:
            return
        share = (now_s - self.began_s) / LASTS_S
        if share >= 1.0:
            self.began_s = None
            self.item.hide()
            return
        self.item.share = share
        self.item.update()
