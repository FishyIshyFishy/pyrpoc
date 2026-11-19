"""``MosaicCanvas``: the mosaic and its regions on the dish, in µm. The wheel zooms, dragging
pans, a click chooses a region and a right-click asks for its menu. While drawing, a drag marks
a region."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QContextMenuEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPalette,
    QWheelEvent,
)
from PyQt6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView, QWidget

from pyrpoc.plugins.program_io.outputs import MosaicOutput

from .boxes import Drawing, RegionBoxes, square
from .regions import Regions
from .tiles import Colouring, TileLayer

ZOOM_STEP = 1.15
# A press must travel this far, in pixels, before it pans rather than clicks.
DRAG_PX = 4


class MosaicCanvas(QGraphicsView):
    # The region clicked, by number, or None for the empty dish.
    chose = pyqtSignal(object)
    # A right-click on a region, by number, or on the empty dish (None), at a global position.
    menu_asked = pyqtSignal(object, QPoint)
    # The centre of a region just drawn, and why one drawn was not kept.
    drawn = pyqtSignal(float, float)
    refused = pyqtSignal(str)
    drawing_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget, regions: Regions):
        self.dish = QGraphicsScene()
        super().__init__(self.dish, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show_background()
        self.regions = regions
        self.drawing = False
        self.selected: int | None = None
        self.pressed: QPoint | None = None
        self.panning = False
        self.tiles = TileLayer(self.dish)
        self.boxes = RegionBoxes(self.dish)
        # One field across; known once the mosaic shown holds a tile.
        self.field_um: float | None = None
        self.sketch: Drawing | None = None

    def show_background(self) -> None:
        self.setBackgroundBrush(self.palette().color(QPalette.ColorRole.Base))

    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        # Around the mosaic is the theme's input color, so a new theme fills it again.
        if event is not None and event.type() == QEvent.Type.PaletteChange:
            self.show_background()

    @property
    def accent(self) -> QColor:
        return self.palette().color(QPalette.ColorRole.Highlight)

    def show_mosaic(self, mosaic: MosaicOutput, colour: Colouring) -> None:
        """Lay what ``mosaic`` holds, fitting the view to it the first time anything shows."""
        was_empty = not self.tiles.laid
        self.tiles.show(mosaic, colour)
        if len(mosaic) and mosaic.field_um != self.field_um:
            self.field_um = mosaic.field_um
            self.set_drawing(False)
            if self.sketch is None:
                self.sketch = Drawing(self.dish, mosaic.field_um)
            self.sketch.field_um = mosaic.field_um
        if was_empty and self.tiles.laid:
            self.fit_all()

    def recolour(self, colour: Colouring) -> None:
        self.tiles.recolour(colour)

    def clear_mosaic(self) -> None:
        self.tiles.clear()

    def show_regions(self) -> None:
        self.boxes.show(self.regions.marked, self.selected, self.accent)

    def select(self, number: int | None) -> None:
        self.selected = number
        self.show_regions()

    def set_drawing(self, on: bool) -> None:
        """Start or stop drawing; there is nothing to draw on until a tile shows."""
        self.drawing = on and self.sketch is not None
        if not self.drawing and self.sketch is not None:
            self.sketch.cancel()
        cursor = Qt.CursorShape.CrossCursor if self.drawing else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        # Told even when unchanged, so a button asking to draw with nothing to draw on unchecks.
        self.drawing_changed.emit(self.drawing)

    def fit_all(self) -> None:
        """Show the whole mosaic and every region."""
        rect = self.tiles.bounds()
        for region in self.regions.marked:
            rect = rect.united(square(region.x_um, region.y_um, region.side_um))
        if rect.isEmpty():
            return
        # Room past the edges, so what is at the rim can still be brought to the middle.
        grow = max(rect.width(), rect.height())
        self.dish.setSceneRect(rect.adjusted(-grow, -grow, grow, grow))
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

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
        if event.button() == Qt.MouseButton.LeftButton and self.drawing and self.sketch:
            self.sketch.begin(self.mapToScene(at), self.accent)
        elif event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self.pressed, self.panning = at, event.button() == Qt.MouseButton.MiddleButton

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        at = event.position().toPoint()
        if self.sketch is not None and self.sketch.active:
            self.sketch.extend(self.mapToScene(at))
        elif self.pressed is not None:
            moved = at - self.pressed
            self.panning = self.panning or moved.manhattanLength() >= DRAG_PX
            if self.panning:
                self.centerOn(self.mapToScene(self.viewport_rect().center() - moved))
                self.pressed = at

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self.sketch is not None and self.sketch.active:
            self.finish_drawing(self.sketch)
        elif self.pressed is not None and not self.panning:
            self.chose.emit(self.region_number_at(self.pressed))
        self.pressed, self.panning = None, False

    def finish_drawing(self, sketch: Drawing) -> None:
        centre, refusal = sketch.finish()
        self.set_drawing(False)
        if refusal is not None:
            self.refused.emit(refusal)
        if centre is not None:
            self.drawn.emit(*centre)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton or self.drawing:
            return
        if self.region_number_at(event.position().toPoint()) is None:
            self.fit_all()

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        if event is None or self.drawing:
            return
        number = self.region_number_at(event.pos())
        if number is not None:
            self.chose.emit(number)
        self.menu_asked.emit(number, event.globalPos())

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape and self.drawing:
            self.set_drawing(False)
        else:
            super().keyPressEvent(event)

    def region_number_at(self, at: QPoint) -> int | None:
        point = self.mapToScene(at)
        region = self.regions.at(point.x(), point.y())
        return None if region is None else region.number

    def viewport_rect(self) -> QRect:
        viewport = self.viewport()
        if viewport is None:
            raise RuntimeError("the mosaic view has no viewport")
        return viewport.rect()
