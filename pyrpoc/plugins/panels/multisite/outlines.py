"""The units on the dish: each outline where it was last tracked, tagged with its name and
drawn by where it stands, and a box one field across where the loop is working."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QStyleOptionGraphicsItem,
    QWidget,
)

from pyrpoc.plugins.program_io.outputs import Outline

from .look import Stroke, faded
from .state.outline import middle, span_um

OUTLINE_Z, FIELD_Z, TAG_Z = 10.0, 20.0, 30.0
# A unit narrower than this on screen is shown by a ring this size, in pixels.
SMALLEST_PX = 18.0
RING_PX = 10.0
TAG_PAD = 4.0


def path_of(outline: Outline) -> QPainterPath:
    path = QPainterPath(QPointF(*outline[0]))
    for x, y in outline[1:]:
        path.lineTo(QPointF(x, y))
    path.closeSubpath()
    return path


def pen_of(stroke: Stroke, width: float) -> QPen:
    pen = QPen(stroke.colour, width)
    pen.setCosmetic(True)
    if stroke.dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    return pen


def bold_font() -> QFont:
    face = QFont()
    face.setBold(True)
    return face


class Tag(QGraphicsItem):
    """A unit's name on a dark tag, the same size at any zoom, its top-left at its position."""

    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.face = bold_font()
        self.text, self.colour = "", QColor()
        self.box = QRectF()

    def set_look(self, text: str, colour: QColor) -> None:
        if (text, colour) == (self.text, self.colour):
            return
        self.prepareGeometryChange()
        self.text, self.colour = text, colour
        metrics = QFontMetricsF(self.face)
        self.box = QRectF(0, 0, metrics.horizontalAdvance(text) + 2 * TAG_PAD, metrics.height())
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


@dataclass(frozen=True)
class UnitLook:
    """One unit as the dish draws it: its key, outline, name and stroke."""

    key: str
    outline: Outline
    name: str
    stroke: Stroke


@dataclass
class Drawn:
    path: QGraphicsPathItem
    ring: QGraphicsEllipseItem
    tag: Tag
    outline: Outline


class UnitOutlines:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.drawn: dict[str, Drawn] = {}
        self.px_per_um = 1.0

    def show(self, looks: list[UnitLook], selected: str | None, px_per_um: float) -> None:
        self.px_per_um = px_per_um
        keys = {look.key for look in looks}
        for key in list(self.drawn):
            if key not in keys:
                self.drop(key)
        for look in looks:
            self.draw(look, look.key == selected)

    def draw(self, look: UnitLook, selected: bool) -> None:
        drawn = self.drawn.get(look.key) or self.add(look)
        if drawn.outline != look.outline:
            drawn.path.setPath(path_of(look.outline))
            drawn.outline = look.outline
        pen = pen_of(look.stroke, 2.6 if selected else 1.4)
        # A chosen unit is filled lightly, so it stands out from those round it.
        fill = QBrush(faded(look.stroke.colour, 35 if selected else 0))
        drawn.path.setPen(pen)
        drawn.path.setBrush(fill)
        drawn.ring.setPen(pen)
        drawn.ring.setBrush(fill)
        drawn.ring.setPos(QPointF(*middle(look.outline)))
        drawn.ring.setVisible(self.small(look.outline))
        left = min(x for x, _ in look.outline)
        top = min(y for _, y in look.outline)
        drawn.tag.setPos(left, top)
        drawn.tag.set_look(look.name, look.stroke.colour)

    def small(self, outline: Outline) -> bool:
        return span_um(outline) * self.px_per_um < SMALLEST_PX

    def add(self, look: UnitLook) -> Drawn:
        path = QGraphicsPathItem(path_of(look.outline))
        path.setZValue(OUTLINE_Z)
        ring = QGraphicsEllipseItem(-RING_PX, -RING_PX, 2 * RING_PX, 2 * RING_PX)
        ring.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        ring.setZValue(OUTLINE_Z)
        tag = Tag()
        tag.setZValue(TAG_Z)
        for item in (path, ring, tag):
            self.scene.addItem(item)
        drawn = Drawn(path, ring, tag, look.outline)
        self.drawn[look.key] = drawn
        return drawn

    def drop(self, key: str) -> None:
        drawn = self.drawn.pop(key)
        for item in (drawn.path, drawn.ring, drawn.tag):
            self.scene.removeItem(item)

    def unit_at(self, point: QPointF) -> str | None:
        """The unit whose outline, or ring when it is drawn as one, holds ``point``."""
        reach_um = RING_PX / self.px_per_um
        for key, drawn in reversed(self.drawn.items()):
            if drawn.path.path().contains(point):
                return key
            x, y = middle(drawn.outline)
            if self.small(drawn.outline) and math.hypot(point.x() - x, point.y() - y) <= reach_um:
                return key
        return None

    def outline_of(self, key: str) -> Outline | None:
        drawn = self.drawn.get(key)
        return None if drawn is None else drawn.outline


class FieldBox:
    """A box one scan field across, centred where the loop is working: dashed while it moves
    there or finds focus, solid while it scans."""

    def __init__(self, scene: QGraphicsScene, field_um: float):
        self.item = QGraphicsRectItem(-field_um / 2, -field_um / 2, field_um, field_um)
        self.item.setZValue(FIELD_Z)
        self.item.hide()
        scene.addItem(self.item)

    def show_at(self, at: tuple[float, float] | None, scanning: bool, accent: QColor) -> None:
        self.item.setVisible(at is not None)
        if at is None:
            return
        self.item.setPos(QPointF(*at))
        self.item.setPen(pen_of(Stroke(accent, not scanning), 1.6))
