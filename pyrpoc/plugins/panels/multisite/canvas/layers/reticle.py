"""The reticle: a box one scan field across that travels to the unit being worked on, its look
saying the phase: brackets while moving, a breathing square while focusing, a sweep while
scanning."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsScene, QStyleOptionGraphicsItem, QWidget

from pyrpoc.plugins.program_io.outputs import MovingToUnit

from ... import look
from ...state.outline import middle
from ...state.phase import Phase, phase_of
from ...state.snapshot import Snapshot
from ..motion import Travel, distance

RETICLE_Z = 20.0
# How long one sweep of a scan takes on screen, in seconds; it shows scanning, not its progress.
SWEEP_S = 1.6
# How fast the reticle travels, in µm a second, kept between these times for any distance.
SPEED_UM_S = 2500.0
SHORTEST_S, LONGEST_S = 0.25, 0.9


def target_of(snapshot: Snapshot) -> QPointF | None:
    """Where the reticle goes: where the stage is moving to, or the unit being worked on."""
    doing = snapshot.doing
    if doing is None:
        return None
    if isinstance(doing, MovingToUnit):
        return QPointF(doing.x_um, doing.y_um)
    unit = snapshot.unit(doing.unit)
    if unit is None:
        return None
    return QPointF(*middle(unit.outline))


class ReticleItem(QGraphicsItem):
    """The box itself, centred on its position."""

    def __init__(self, field_um: float):
        super().__init__()
        self.half = field_um / 2
        self.phase: Phase | None = None
        self.phase_s = 0.0
        self.now_s = 0.0

    def boundingRect(self) -> QRectF:
        grow = self.half * 1.1
        return QRectF(-grow, -grow, 2 * grow, 2 * grow)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ) -> None:
        if painter is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = look.IDLE if self.phase is None else look.PHASE[self.phase]
        box = QRectF(-self.half, -self.half, 2 * self.half, 2 * self.half)
        alpha = 110 if self.phase is None else 235
        self.brackets(painter, box, look.faded(colour, alpha))
        since = self.now_s - self.phase_s
        if self.phase is Phase.FOCUSING:
            self.breathe(painter, box, colour, since)
        elif self.phase is Phase.SCANNING:
            self.sweep(painter, box, colour, since)
        elif self.phase is Phase.MOVING:
            painter.setPen(self.line(look.faded(colour, 120), 1.0, Qt.PenStyle.DashLine))
            painter.drawRect(box)

    def line(self, colour: QColor, width: float, style: Qt.PenStyle) -> QPen:
        pen = QPen(colour, width, style)
        pen.setCosmetic(True)
        return pen

    def brackets(self, painter: QPainter, box: QRectF, colour: QColor) -> None:
        painter.setPen(self.line(colour, 2.4, Qt.PenStyle.SolidLine))
        arm = box.width() * 0.18
        for corner, dx, dy in (
            (box.topLeft(), 1, 1),
            (box.topRight(), -1, 1),
            (box.bottomLeft(), 1, -1),
            (box.bottomRight(), -1, -1),
        ):
            painter.drawLine(corner, corner + QPointF(dx * arm, 0))
            painter.drawLine(corner, corner + QPointF(0, dy * arm))

    def breathe(self, painter: QPainter, box: QRectF, colour: QColor, since: float) -> None:
        share = 0.55 + 0.2 * (0.5 + 0.5 * math.sin(since * math.tau * 1.4))
        inner = QRectF(box.center(), box.center())
        grow = box.width() * share / 2
        inner.adjust(-grow, -grow, grow, grow)
        painter.setPen(self.line(look.faded(colour, 200), 1.4, Qt.PenStyle.SolidLine))
        painter.drawRect(inner)

    def sweep(self, painter: QPainter, box: QRectF, colour: QColor, since: float) -> None:
        y = box.top() + box.height() * ((since / SWEEP_S) % 1.0)
        done = QRectF(box.topLeft(), QPointF(box.right(), y))
        painter.fillRect(done, look.faded(colour, 34))
        painter.setPen(self.line(colour, 2.0, Qt.PenStyle.SolidLine))
        painter.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))


class Reticle:
    def __init__(self, scene: QGraphicsScene, field_um: float):
        self.item = ReticleItem(field_um)
        self.item.setZValue(RETICLE_Z)
        self.item.hide()
        scene.addItem(self.item)
        self.travel: Travel | None = None

    def show(self, snapshot: Snapshot, on: bool, now_s: float) -> None:
        doing = snapshot.doing
        phase = None if doing is None else phase_of(doing)
        if phase != self.item.phase:
            self.item.phase, self.item.phase_s = phase, now_s
        target = target_of(snapshot)
        if target is not None:
            self.head_to(target, now_s)
        self.item.setVisible(on and self.travel is not None)

    def head_to(self, target: QPointF, now_s: float) -> None:
        if self.travel is None:
            self.travel = Travel(target, target, now_s, SHORTEST_S)
            return
        if self.travel.end == target:
            return
        start = self.travel.at(now_s)
        takes = min(LONGEST_S, max(SHORTEST_S, distance(start, target) / SPEED_UM_S))
        self.travel = Travel(start, target, now_s, takes)

    def tick(self, now_s: float) -> None:
        if self.travel is None:
            return
        self.item.now_s = now_s
        self.item.setPos(self.travel.at(now_s))
        self.item.update()
