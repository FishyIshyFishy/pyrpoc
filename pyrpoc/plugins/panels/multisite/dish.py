"""``DishCanvas``: the whole dish in µm. The wheel zooms in and out, dragging pans, a click
chooses a unit and a right-click asks for its menu. While drawing, a drag draws a unit."""

from __future__ import annotations

from collections.abc import Mapping

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QContextMenuEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPalette,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView, QWidget

from .backdrop import BackdropLayer
from .feed import Backdrop, UnitFrame
from .lasso import Lasso
from .outlines import FieldBox, UnitLook, UnitOutlines
from .state.outline import middle

ZOOM_STEP = 1.15
# A press must travel this far, in pixels, before it pans rather than clicks.
DRAG_PX = 4
# A unit shown on request is zoomed to at least this many screen pixels across one field.
LOCATE_FIELD_PX = 220.0
# The drawn outline keeps every turn larger than this many screen pixels.
THIN_PX = 1.5


class DishCanvas(QGraphicsView):
    # The unit clicked, by key, or None for the empty dish.
    chose = pyqtSignal(object)
    # A right-click on a unit, by key, or on the empty dish (None), at a global position.
    menu_asked = pyqtSignal(object, QPoint)
    # An outline just drawn, and why one drawn was not kept.
    drawn = pyqtSignal(object)
    refused = pyqtSignal(str)
    drawing_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget, backdrop: Backdrop, field_um: float):
        self.dish = QGraphicsScene()
        super().__init__(self.dish, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.show_backdrop()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.drawing = False
        self.selected: str | None = None
        self.pressed: QPoint | None = None
        self.panning = False
        self.fitted = False
        self.field_um = field_um
        self.backdrop = BackdropLayer(self.dish)
        self.backdrop.show_backdrop(backdrop)
        self.outlines = UnitOutlines(self.dish)
        self.field = FieldBox(self.dish, field_um)
        self.lasso = Lasso(self.dish, field_um)
        self.dish_rect = dish_rect_of(backdrop)
        # Room past the edges, so a unit at the rim can still be brought to the middle.
        grow = max(self.dish_rect.width(), self.dish_rect.height())
        self.dish.setSceneRect(self.dish_rect.adjusted(-grow, -grow, grow, grow))

    def show_backdrop(self) -> None:
        self.setBackgroundBrush(self.palette().color(QPalette.ColorRole.Base))

    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        # Around the dish is the theme's input color, so a new theme fills it again.
        if event is not None and event.type() == QEvent.Type.PaletteChange:
            self.show_backdrop()

    @property
    def accent(self) -> QColor:
        return self.palette().color(QPalette.ColorRole.Highlight)

    def show_units(
        self,
        looks: list[UnitLook],
        frames: Mapping[int, UnitFrame],
        present: set[int],
        working: tuple[tuple[float, float], bool] | None,
    ) -> None:
        """Draw ``looks``, the frames of the units ``present``, and the field the loop is
        ``working`` on, with whether it is scanning there."""
        self.backdrop.show_frames(frames, present)
        self.outlines.show(looks, self.selected, self.transform().m11())
        at, scanning = (None, False) if working is None else working
        self.field.show_at(at, scanning, self.accent)

    def select(self, key: str | None) -> None:
        self.selected = key

    def set_drawing(self, on: bool) -> None:
        if on == self.drawing:
            return
        self.drawing = on
        if not on:
            self.lasso.cancel()
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.drawing_changed.emit(on)

    def fit_dish(self) -> None:
        self.fitted = True
        self.fitInView(self.dish_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        # Fitted once it has a size: before it is shown, it has none to fit to.
        if not self.fitted and self.view_rect().width() > 1:
            self.fit_dish()

    def locate(self, key: str, free: QRect) -> None:
        """Bring a unit to the middle of ``free``, the part of the view nothing covers,
        zooming in when one field would be too small to see."""
        outline = self.outlines.outline_of(key)
        if outline is None:
            return
        field_px = self.field_um * self.transform().m11()
        if field_px < LOCATE_FIELD_PX:
            factor = LOCATE_FIELD_PX / field_px
            self.scale(factor, factor)
        off = free.center() - self.view_rect().center()
        scale = self.transform().m11()
        self.centerOn(QPointF(*middle(outline)) - QPointF(off.x() / scale, off.y() / scale))

    def view_rect(self) -> QRect:
        viewport = self.viewport()
        if viewport is None:
            raise RuntimeError("the dish has no viewport")
        return viewport.rect()

    # Qt types every event as optional; the None checks narrow for that.
    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        at = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.lasso.begin(self.mapToScene(at), self.accent)
        elif event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self.pressed, self.panning = at, event.button() == Qt.MouseButton.MiddleButton

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        at = event.position().toPoint()
        if self.lasso.drawing:
            self.lasso.extend(self.mapToScene(at))
        elif self.pressed is not None:
            moved = at - self.pressed
            self.panning = self.panning or moved.manhattanLength() >= DRAG_PX
            if self.panning:
                self.centerOn(self.mapToScene(self.view_rect().center() - moved))
                self.pressed = at

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self.lasso.drawing:
            self.finish_drawing()
        elif self.pressed is not None and not self.panning:
            self.chose.emit(self.outlines.unit_at(self.mapToScene(self.pressed)))
        self.pressed, self.panning = None, False

    def finish_drawing(self) -> None:
        outline, refusal = self.lasso.finish(THIN_PX / self.transform().m11())
        self.set_drawing(False)
        if refusal is not None:
            self.refused.emit(refusal)
        if outline is not None:
            self.drawn.emit(outline)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton or self.drawing:
            return
        if self.outlines.unit_at(self.mapToScene(event.position().toPoint())) is None:
            self.fit_dish()

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        if event is None or self.drawing:
            return
        key = self.outlines.unit_at(self.mapToScene(event.pos()))
        if key is not None:
            self.chose.emit(key)
        self.menu_asked.emit(key, event.globalPos())

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape and self.drawing:
            self.set_drawing(False)
        else:
            super().keyPressEvent(event)


def dish_rect_of(backdrop: Backdrop) -> QRectF:
    height, width = backdrop.planes.shape[1:]
    placement = backdrop.placement
    um = placement.um_per_px
    return QRectF(placement.left_um, placement.top_um, width * um, height * um)
