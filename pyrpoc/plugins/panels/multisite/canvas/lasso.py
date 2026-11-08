"""Drawing a unit freehand: press, drag round it, release. While drawing, a square shows the most a
unit may span, and the line turns to a warning once the shape is too large."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene

from pyrpoc.plugins.program_io.outputs import Outline

from .. import look
from ..state.outline import LARGEST, fits, middle, thinned, too_small

LASSO_Z = 50.0


def pen(colour: QColor, width: float, style: Qt.PenStyle) -> QPen:
    line = QPen(colour, width, style)
    line.setCosmetic(True)
    return line


class Lasso:
    def __init__(self, scene: QGraphicsScene, field_um: float):
        self.scene = scene
        self.field_um = field_um
        self.points: list[tuple[float, float]] = []
        self.path = QGraphicsPathItem()
        self.path.setZValue(LASSO_Z)
        self.guide = QGraphicsRectItem()
        self.guide.setZValue(LASSO_Z)
        self.guide.setPen(pen(look.faded(look.TEXT, 90), 1.0, Qt.PenStyle.DotLine))
        for item in (self.path, self.guide):
            item.hide()
            scene.addItem(item)

    @property
    def drawing(self) -> bool:
        return bool(self.points)

    def begin(self, at: QPointF) -> None:
        self.points = [(at.x(), at.y())]
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
        large = len(self.points) > 2 and not fits(tuple(self.points), self.field_um)
        colour = look.WARNING if large else look.ACCENT
        self.path.setPen(pen(colour, 2.0, Qt.PenStyle.SolidLine))
        side = LARGEST * self.field_um
        x, y = middle(tuple(self.points))
        self.guide.setRect(QRectF(x - side / 2, y - side / 2, side, side))

    def finish(self, tolerance_um: float) -> tuple[Outline | None, str | None]:
        """The outline drawn, or None and why it was not kept."""
        outline = thinned(self.points, tolerance_um)
        self.cancel()
        if too_small(outline, self.field_um):
            return None, None
        if not fits(outline, self.field_um):
            return None, "Too large: a unit must fit inside the dotted square"
        return outline, None

    def cancel(self) -> None:
        self.points = []
        self.path.hide()
        self.guide.hide()
