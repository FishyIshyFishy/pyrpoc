"""``DishCanvas``: the whole dish in µm, zoomed with the wheel round the cursor, panned with the
right or middle button, units drawn with Shift and the left button, and picked with a click."""

from __future__ import annotations

from collections.abc import Mapping

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QPainter, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView, QWidget

from pyrpoc.plugins.program_io.outputs import Unit, UnitState

from .. import keys, look
from ..feed import Backdrop, UnitFrame
from ..state.drafts import Drafts
from ..state.outline import middle
from ..state.snapshot import Snapshot
from .backdrop import BackdropLayer
from .indicators import Indicator
from .lasso import Lasso
from .layers.drafts import DraftOutlines
from .layers.next_up import NextUp
from .layers.reticle import Reticle
from .layers.ripple import Ripple
from .layers.unit_outlines import UnitOutlines
from .motion import Travel
from .status_line import StatusLine

ZOOM_STEP = 1.15
# How long a drawn shape's refusal stays on the status line, in seconds.
NOTICE_S = 5.0
# A located unit is zoomed to at least this many screen pixels across one field.
LOCATE_FIELD_PX = 220.0
GLIDE_S = 0.35
# The drawn outline keeps every turn larger than this many screen pixels.
THIN_PX = 1.5
PAN_BUTTONS = (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton)


class DishCanvas(QGraphicsView):
    # The unit clicked, or None for empty dish.
    picked = pyqtSignal(object)
    # A unit just drawn, not yet sent to the loop.
    drawn = pyqtSignal(object)
    remove_asked = pyqtSignal(int)

    def __init__(self, parent: QWidget, field_um: float):
        self.dish = QGraphicsScene()
        super().__init__(self.dish, parent)
        self.field_um = field_um
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(look.faded(look.GLASS, 255))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.backdrop = BackdropLayer(self.dish)
        self.outlines = UnitOutlines(self.dish)
        self.reticle = Reticle(self.dish, field_um)
        self.next_up = NextUp(self.dish)
        self.ripple = Ripple(self.dish)
        self.lasso = Lasso(self.dish, field_um)
        self.drafts = Drafts()
        self.draft_outlines = DraftOutlines(self.dish)
        self.loop_now_s = 0.0
        self.status = StatusLine(self)
        self.selected: int | None = None
        self.dish_rect = QRectF()
        self.fitted = False
        self.panning: QPoint | None = None
        self.glide: Travel | None = None
        self.drawn_notice: tuple[str, float] | None = None
        self.now_s = 0.0

    def show_loop(
        self,
        snapshot: Snapshot,
        backdrop: Backdrop,
        frames: Mapping[int, UnitFrame],
        indicators: Indicator,
        now_s: float,
    ) -> None:
        """Bring the dish up to ``snapshot``; ``now_s`` is the wall clock animations run on."""
        self.now_s, self.loop_now_s = now_s, snapshot.now_s
        self.take_backdrop(backdrop)
        self.drafts.settle(snapshot.view, snapshot.now_s)
        self.draft_outlines.show(self.drafts.waiting)
        present = {u.number for u in snapshot.view.units if u.state is not UnitState.REMOVED}
        self.backdrop.show_frames(frames, present, now_s)
        styled = Indicator.OUTLINE_STATE in indicators
        self.outlines.show(snapshot, self.selected, styled, now_s, self.transform().m11())
        self.reticle.show(snapshot, Indicator.RETICLE in indicators, now_s)
        self.next_up.show(snapshot, Indicator.NEXT_UP in indicators)
        self.status.show_snapshot(snapshot, self.notice(), Indicator.STATUS_LINE in indicators)
        self.place_status()

    def take_backdrop(self, backdrop: Backdrop) -> None:
        self.backdrop.show_backdrop(backdrop)
        height, width = backdrop.planes.shape[1:]
        placement = backdrop.placement
        rect = QRectF(
            placement.left_um,
            placement.top_um,
            width * placement.um_per_px,
            height * placement.um_per_px,
        )
        if rect != self.dish_rect:
            self.dish_rect = rect
            # Room past the edges, so a unit at the rim can still be brought to the middle.
            grow = max(rect.width(), rect.height())
            self.dish.setSceneRect(rect.adjusted(-grow, -grow, grow, grow))
        if not self.fitted and self.view_rect().width() > 1:
            self.fit_dish()

    def notice(self) -> str | None:
        if self.drawn_notice is None or self.now_s - self.drawn_notice[1] > NOTICE_S:
            return None
        return self.drawn_notice[0]

    def tick(self, now_s: float) -> None:
        """Carry every animation on."""
        self.backdrop.tick(now_s)
        self.reticle.tick(now_s)
        self.ripple.tick(now_s)
        if self.glide is not None:
            self.centerOn(self.glide.at(now_s))
            if self.glide.done(now_s):
                self.glide = None

    def view_rect(self) -> QRect:
        viewport = self.viewport()
        if viewport is None:
            raise RuntimeError("the dish has no viewport")
        return viewport.rect()

    def fit_dish(self) -> None:
        self.fitted = True
        self.glide = None
        self.fitInView(self.dish_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def select(self, unit: int | None) -> None:
        self.selected = unit

    def locate(self, snapshot: Snapshot, free: QRect) -> None:
        """Glide the selected unit to the middle of ``free``, the part of the view no card
        covers, zooming in if it would be too small to see, and ripple."""
        unit = None if self.selected is None else snapshot.unit(self.selected)
        if unit is None:
            return
        target = QPointF(*middle(unit.outline))
        field_px = self.field_um * self.transform().m11()
        if field_px < LOCATE_FIELD_PX:
            factor = LOCATE_FIELD_PX / field_px
            self.scale(factor, factor)
        start = self.mapToScene(self.view_rect().center())
        off = free.center() - self.view_rect().center()
        scale = self.transform().m11()
        shift = QPointF(off.x() / scale, off.y() / scale)
        self.glide = Travel(start, target - shift, self.now_s, GLIDE_S)
        self.ripple.start(target, self.now_s)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.place_status()
        if not self.fitted and not self.dish_rect.isEmpty():
            self.fit_dish()

    def place_status(self) -> None:
        # Bottom right, clear of the queue card the overlay docks on the left.
        corner = self.rect().bottomRight() - QPoint(self.status.width(), self.status.height())
        self.status.move(corner - QPoint(12, 12))

    # Qt types every event as optional; the None checks narrow for that.
    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        self.glide = None
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() in PAN_BUTTONS:
            self.panning, self.glide = event.position().toPoint(), None
        elif event.button() == Qt.MouseButton.LeftButton:
            at = self.mapToScene(event.position().toPoint())
            if event.modifiers() & keys.DRAW:
                self.lasso.begin(at)
            else:
                self.picked.emit(self.outlines.unit_at(at))

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self.panning is not None:
            now = event.position().toPoint()
            self.pan_by(now - self.panning)
            self.panning = now
        elif self.lasso.drawing:
            self.lasso.extend(self.mapToScene(event.position().toPoint()))

    def pan_by(self, moved: QPoint) -> None:
        centre = self.view_rect().center() - moved
        self.centerOn(self.mapToScene(centre))

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() in PAN_BUTTONS:
            self.panning = None
        elif event.button() == Qt.MouseButton.LeftButton and self.lasso.drawing:
            outline, refused = self.lasso.finish(THIN_PX / self.transform().m11())
            if refused is not None:
                self.drawn_notice = (refused, self.now_s)
            if outline is not None:
                # Groups are not drawn yet, so every unit takes the one plain group.
                unit = Unit.new(outline, "")
                self.drafts.add(unit, self.loop_now_s)
                self.drawn.emit(unit)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.fit_dish()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if event.key() == keys.FIT:
            self.fit_dish()
        elif event.key() in keys.REMOVE and self.selected is not None:
            self.remove_asked.emit(self.selected)
        elif event.key() == keys.CLOSE:
            if self.lasso.drawing:
                self.lasso.cancel()
            else:
                self.picked.emit(None)
        else:
            super().keyPressEvent(event)
