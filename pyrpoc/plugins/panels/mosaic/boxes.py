"""The regions on the mosaic, each a square one field across with its name on a tag, and the
square that follows a region being drawn."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QStyleOptionGraphicsItem,
    QWidget,
)

from .regions import Region, bounds_middle, span

BOX_Z, TAG_Z, DRAWING_Z = 10.0, 20.0, 30.0
TAG_PAD = 4.0
# The mosaic is a fluorescence image, dark in every theme, so these read on it whatever the theme.
BOX = QColor(232, 234, 240)
REFUSED = QColor(240, 90, 90)
GUIDE = QColor(232, 234, 240, 140)


def pen(colour: QColor, width: float, style: Qt.PenStyle) -> QPen:
    line = QPen(colour, width, style)
    line.setCosmetic(True)
    return line


def square(x_um: float, y_um: float, side_um: float) -> QRectF:
    return QRectF(x_um - side_um / 2, y_um - side_um / 2, side_um, side_um)


class Tag(QGraphicsItem):
    """A region's name on a dark tag, the same size at any zoom, its top-left at its
    position."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.face = QFont()
        self.face.setBold(True)
        self.text = text
        self.colour = BOX
        metrics = QFontMetricsF(self.face)
        self.box = QRectF(0, 0, metrics.horizontalAdvance(text) + 2 * TAG_PAD, metrics.height())

    def set_colour(self, colour: QColor) -> None:
        if colour != self.colour:
            self.colour = colour
            self.update()

    def boundingRect(self) -> QRectF:
        return self.box

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
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(self.box, 3, 3)
        painter.setFont(self.face)
        painter.setPen(self.colour)
        painter.drawText(self.box, Qt.AlignmentFlag.AlignCenter, self.text)


class RegionBoxes:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.drawn: dict[int, tuple[QGraphicsRectItem, Tag]] = {}

    def show(self, regions: Sequence[Region], selected: int | None, accent: QColor) -> None:
        numbers = {region.number for region in regions}
        for number in [n for n in self.drawn if n not in numbers]:
            for item in self.drawn.pop(number):
                self.scene.removeItem(item)
        for region in regions:
            box, tag = self.drawn.get(region.number) or self.add(region)
            chosen = region.number == selected
            colour = accent if chosen else BOX
            box.setPen(pen(colour, 2.4 if chosen else 1.4, Qt.PenStyle.SolidLine))
            # A chosen region is filled lightly, so it stands out from those round it.
            fill = QColor(colour.red(), colour.green(), colour.blue(), 35 if chosen else 0)
            box.setBrush(QBrush(fill))
            tag.set_colour(colour)

    def add(self, region: Region) -> tuple[QGraphicsRectItem, Tag]:
        box = QGraphicsRectItem(square(region.x_um, region.y_um, region.side_um))
        box.setZValue(BOX_Z)
        tag = Tag(f"Region {region.number}")
        tag.setZValue(TAG_Z)
        rect = box.rect()
        tag.setPos(rect.left(), rect.top())
        for item in (box, tag):
            self.scene.addItem(item)
        self.drawn[region.number] = (box, tag)
        return box, tag


class Drawing:
    """A region being drawn: the line dragged so far, and the square one field across round its
    middle, which is the region kept. The line turns red once it no longer fits the square."""

    def __init__(self, scene: QGraphicsScene, field_um: float):
        self.field_um = field_um
        self.points: list[tuple[float, float]] = []
        self.accent = QColor()
        self.path = QGraphicsPathItem()
        self.guide = QGraphicsRectItem()
        self.guide.setPen(pen(GUIDE, 1.0, Qt.PenStyle.DotLine))
        for item in (self.path, self.guide):
            item.setZValue(DRAWING_Z)
            item.hide()
            scene.addItem(item)

    @property
    def active(self) -> bool:
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
        colour = self.accent if self.fits() else REFUSED
        self.path.setPen(pen(colour, 2.0, Qt.PenStyle.SolidLine))
        self.guide.setRect(square(*bounds_middle(self.points), self.field_um))

    def fits(self) -> bool:
        return span(self.points) <= self.field_um

    def finish(self) -> tuple[tuple[float, float] | None, str | None]:
        """The centre of the region drawn, or None and why it was not kept."""
        fits, centre = self.fits(), bounds_middle(self.points)
        self.cancel()
        if not fits:
            return None, "Not added: a region must fit inside the dotted square of one field."
        return centre, None

    def cancel(self) -> None:
        self.points = []
        self.path.hide()
        self.guide.hide()
