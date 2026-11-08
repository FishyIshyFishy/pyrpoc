"""Each unit's outline where it was last tracked, with its number, drawn by its state, and the
unit being worked on drawn by the phase. A unit too small to see at the zoom gets a ring of
fixed size, so it can be seen and clicked from the whole dish."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene

from pyrpoc.plugins.program_io.outputs import Outline, UnitState, UnitView

from ... import look
from ...state.outline import middle, span_um
from ...state.phase import Phase, phase_of
from ...state.snapshot import Snapshot
from .badge import Badge

OUTLINE_Z, BADGE_Z = 10.0, 30.0
# How fast the active unit's outline breathes, in cycles a second.
PULSE_HZ = 1.2
# A unit narrower than this on screen is shown by a ring this size, in pixels.
SMALLEST_PX = 18.0
RING_PX = 10.0


@dataclass
class Drawn:
    path: QGraphicsPathItem
    ring: QGraphicsEllipseItem
    badge: Badge
    outline: Outline


@dataclass(frozen=True)
class Style:
    colour: QColor
    width: float
    dashed: bool
    filled: bool


def path_of(outline: Outline) -> QPainterPath:
    path = QPainterPath(QPointF(*outline[0]))
    for x, y in outline[1:]:
        path.lineTo(QPointF(x, y))
    path.closeSubpath()
    return path


def pen(style: Style, cosmetic: bool) -> QPen:
    line = QPen(style.colour, style.width)
    line.setCosmetic(cosmetic)
    if style.dashed:
        line.setStyle(Qt.PenStyle.DashLine)
    return line


def style_of(
    unit: UnitView, phase: Phase | None, selected: bool, styled: bool, now_s: float
) -> Style:
    """How a unit is drawn: by its state, by the phase while worked on, bolder when chosen."""
    width = 2.6 if selected or phase is not None else 1.4
    if not styled:
        return Style(look.PLAIN_OUTLINE, width, False, selected)
    colour = look.PHASE[phase] if phase is not None else look.STATE[unit.state]
    if phase is not None:
        # The active unit breathes, so the eye finds it at a glance.
        colour = look.faded(colour, int(170 + 85 * math.sin(now_s * PULSE_HZ * math.tau)))
    dashed = unit.state in (UnitState.PAUSED, UnitState.LOST)
    return Style(colour, width, dashed, selected)


class UnitOutlines:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.drawn: dict[int, Drawn] = {}
        self.px_per_um = 1.0

    def show(
        self,
        snapshot: Snapshot,
        selected: int | None,
        styled: bool,
        now_s: float,
        px_per_um: float,
    ) -> None:
        self.px_per_um = px_per_um
        units = {u.number: u for u in snapshot.view.units if u.state is not UnitState.REMOVED}
        for number in list(self.drawn):
            if number not in units:
                self.drop(number)
        doing = snapshot.doing
        active = None if doing is None else (doing.unit, phase_of(doing))
        for number, unit in units.items():
            phase = active[1] if active is not None and active[0] == number else None
            self.draw(unit, style_of(unit, phase, number == selected, styled, now_s))

    def draw(self, unit: UnitView, style: Style) -> None:
        drawn = self.drawn.get(unit.number) or self.add(unit)
        if drawn.outline != unit.outline:
            drawn.path.setPath(path_of(unit.outline))
            drawn.outline = unit.outline
        fill = QBrush(look.faded(style.colour, 44 if style.filled else 0))
        drawn.path.setPen(pen(style, True))
        drawn.path.setBrush(fill)
        drawn.ring.setPen(pen(style, False))
        drawn.ring.setBrush(fill)
        drawn.ring.setPos(QPointF(*middle(unit.outline)))
        drawn.ring.setVisible(self.small(unit.outline))
        left, top = min(x for x, _ in unit.outline), min(y for _, y in unit.outline)
        drawn.badge.setPos(left, top)
        drawn.badge.set_look(str(unit.number), style.colour)

    def small(self, outline: Outline) -> bool:
        return span_um(outline) * self.px_per_um < SMALLEST_PX

    def add(self, unit: UnitView) -> Drawn:
        path = QGraphicsPathItem(path_of(unit.outline))
        path.setZValue(OUTLINE_Z)
        ring = QGraphicsEllipseItem(-RING_PX, -RING_PX, 2 * RING_PX, 2 * RING_PX)
        ring.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        ring.setZValue(OUTLINE_Z)
        badge = Badge(str(unit.number), look.PLAIN_OUTLINE)
        badge.setZValue(BADGE_Z)
        for item in (path, ring, badge):
            self.scene.addItem(item)
        drawn = Drawn(path, ring, badge, unit.outline)
        self.drawn[unit.number] = drawn
        return drawn

    def drop(self, number: int) -> None:
        drawn = self.drawn.pop(number)
        for item in (drawn.path, drawn.ring, drawn.badge):
            self.scene.removeItem(item)

    def unit_at(self, point: QPointF) -> int | None:
        """The unit whose outline, or ring when it is drawn as one, holds ``point``."""
        reach_um = RING_PX / self.px_per_um
        for number, drawn in reversed(self.drawn.items()):
            if drawn.path.path().contains(point):
                return number
            x, y = middle(drawn.outline)
            if self.small(drawn.outline) and math.hypot(point.x() - x, point.y() - y) <= reach_um:
                return number
        return None
