"""Drawing a unit freehand: press, drag round it, release. While drawing, a dotted square shows
one field of view, and the line turns red once the shape no longer fits in it."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene

from pyrpoc.plugins.program_io.outputs import Outline

from . import look
from .state.outline import middle, thinned, too_small, within_field

LASSO_Z = 50.0


def pen(colour: QColor, width: float, style: Qt.PenStyle) -> QPen:
    line = QPen(colour, width, style)
    line.setCosmetic(True)
    return line


class Lasso:
    def __init__(self, scene: QGraphicsScene, field_um: float):
        self.field_um = field_um
        self.points: list[tuple[float, float]] = []
        self.accent = QColor()
        self.path = QGraphicsPathItem()
        self.guide = QGraphicsRectItem()
        self.guide.setPen(pen(look.faded(look.WAITING, 140), 1.0, Qt.PenStyle.DotLine))
        for item in (self.path, self.guide):
            item.setZValue(LASSO_Z)
            item.hide()
            scene.addItem(item)

    @property
    def drawing(self) -> bool:
        return bool(self.points)

    def begin(self, at: QPointF, accent: QColor) -> None:
        self.points, self.accent = [(at.x(), at.y())], accent
        self.path.show()
        self.guide.show()
        self.redraw()

    def extend(self, at: QPointF) -> None:
        self.points.append((at.x(), at.y()))
        self.redraw()

    def redraw(self) -> None:
        path = QPainterPath(QPointF(*self.points[0]))
        for x, y in self.points[1:]:
            path.lineTo(QPointF(x, y))
        self.path.setPath(path)
        fits = within_field(tuple(self.points), self.field_um)
        colour = self.accent if fits else look.REFUSED
        self.path.setPen(pen(colour, 2.0, Qt.PenStyle.SolidLine))
        x, y = middle(tuple(self.points))
        side = self.field_um
        self.guide.setRect(QRectF(x - side / 2, y - side / 2, side, side))

    def finish(self, tolerance_um: float) -> tuple[Outline | None, str | None]:
        """The outline drawn, or None and why it was not kept: a stray drag too small to be a
        unit says nothing."""
        outline = thinned(self.points, tolerance_um)
        self.cancel()
        if too_small(outline, self.field_um):
            return None, None
        if not within_field(outline, self.field_um):
            return None, "Not added: a unit must fit inside the dotted square of one field."
        return outline, None

    def cancel(self) -> None:
        self.points = []
        self.path.hide()
        self.guide.hide()
