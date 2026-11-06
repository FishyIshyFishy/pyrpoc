"""``UnitCanvas``: the survey's backdrop in dish µm, with every unit's outline over it, where
outlines are drawn, selected, moved and reshaped."""

# Each unit is a closed PolyLineROI: dragging inside moves it, dragging a corner reshapes it,
# and double-clicking an edge adds a corner. A new outline is drawn by clicking its corners and
# closing it with a double-click or a click on its first corner.

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from pyrpoc.plugins.program_io.inputs import Unit
from pyrpoc.utils.widgets.colors import color_for_index
from pyrpoc.utils.widgets.colour_image import ColourImage

# A click this close to the first corner, in screen pixels, closes the outline.
CLOSE_PX = 10


def group_colour(group: str, groups: list[str]) -> tuple[int, int, int]:
    """Ungrouped units are white; each group takes the next colour in the palette."""
    return (235, 235, 235) if not group else color_for_index(groups.index(group) + 2)


class UnitCanvas(ColourImage):
    """Shows the units and reports what the user does to them; it never changes them
    itself."""

    drawn = pyqtSignal(object)  # the new outline, a tuple of (x, y) in µm
    reshaped = pyqtSignal(int, object)  # a unit's id and its new outline
    selected = pyqtSignal(object)  # a unit's id, or None

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.units: tuple[Unit, ...] = ()
        self.chosen: int | None = None
        self.rois: dict[int, pg.PolyLineROI] = {}
        self.labels: list[pg.TextItem] = []
        self.drawing = False
        self.corners: list[tuple[float, float]] = []
        self.sketch = pg.PlotDataItem(
            pen=pg.mkPen((255, 220, 0), width=2), symbol="o", symbolSize=5
        )
        self.addItem(self.sketch)
        self.tracked: list[pg.PlotDataItem] = []
        scene = self.sceneObj
        if scene is None:
            raise RuntimeError("a new PlotWidget has no scene")
        scene.sigMouseClicked.connect(self.on_clicked)

    def show_backdrop(self, rgb: np.ndarray, rect: QRectF) -> None:
        self.show_rgb(rgb)
        self.place(rect)

    def show_units(self, units: tuple[Unit, ...], chosen: int | None) -> None:
        """Draw ``units``, rebuilding only when they changed, so a drag in progress is kept."""
        if units == self.units and chosen == self.chosen:
            return
        self.units, self.chosen = units, chosen
        for roi in self.rois.values():
            self.removeItem(roi)
        for label in self.labels:
            self.removeItem(label)
        self.rois, self.labels = {}, []
        groups = sorted({unit.group for unit in units if unit.group})
        for unit in units:
            self.add_unit(unit, group_colour(unit.group, groups), unit.id == chosen)

    def add_unit(self, unit: Unit, colour: tuple[int, int, int], chosen: bool) -> None:
        pen = pg.mkPen(colour, width=3 if chosen else 1.5)
        roi = pg.PolyLineROI(list(unit.outline), closed=True, pen=pen, movable=chosen)
        # Corners can be dragged only on the chosen unit, so a stray drag never reshapes another.
        for handle in roi.getHandles():
            handle.setVisible(chosen)
        roi.sigRegionChangeFinished.connect(lambda r, uid=unit.id: self.on_reshaped(uid, r))
        self.addItem(roi)
        self.rois[unit.id] = roi
        label = pg.TextItem(str(unit.id), color=colour, anchor=(0.5, 0.5))
        label.setPos(*unit.centre)
        self.addItem(label)
        self.labels.append(label)

    def on_reshaped(self, unit_id: int, roi: pg.PolyLineROI) -> None:
        outline = tuple(
            (float(point.x()), float(point.y()))
            for point in (roi.mapToParent(handle.pos()) for handle in roi.getHandles())
        )
        self.reshaped.emit(unit_id, outline)

    def view_point(self, scene_position: QPointF) -> tuple[float, float]:
        point = self.getViewBox().mapSceneToView(scene_position)
        return float(point.x()), float(point.y())

    def on_clicked(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.view_point(event.scenePos())
        if self.drawing:
            self.add_corner(point, event.scenePos(), bool(event.double()))
        else:
            self.selected.emit(self.unit_at(point))

    def unit_at(self, point: tuple[float, float]) -> int | None:
        """The id of the unit whose outline holds ``point``, the last drawn first."""
        for unit in reversed(self.units):
            contour = np.asarray(unit.outline, dtype=np.float32)
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return unit.id
        return None

    def show_tracked(self, outlines: list[tuple[tuple[float, float], ...]]) -> None:
        """Where a running loop has tracked each unit to, dashed over the drawn outlines."""
        for item in self.tracked:
            self.removeItem(item)
        pen = pg.mkPen((255, 220, 0), width=1.5, style=Qt.PenStyle.DashLine)
        self.tracked = []
        for outline in outlines:
            xs = [x for x, _y in outline] + [outline[0][0]]
            ys = [y for _x, y in outline] + [outline[0][1]]
            item = pg.PlotDataItem(xs, ys, pen=pen)
            self.addItem(item)
            self.tracked.append(item)

    def start_drawing(self) -> None:
        self.drawing = True
        self.corners = []
        self.sketch.setData([], [])

    def stop_drawing(self) -> None:
        self.drawing = False
        self.corners = []
        self.sketch.setData([], [])

    def add_corner(self, point: tuple[float, float], scene_position: QPointF, double: bool) -> None:
        """Add a corner, or close the outline on a double-click or a click on its first corner."""
        if self.corners and (double or self.near_first(scene_position)):
            if len(self.corners) >= 3:
                self.drawn.emit(tuple(self.corners))
            self.start_drawing()
            return
        self.corners.append(point)
        xs, ys = zip(*self.corners, strict=True)
        self.sketch.setData(list(xs), list(ys))

    def near_first(self, scene_position: QPointF) -> bool:
        first = self.getViewBox().mapViewToScene(QPointF(*self.corners[0]))
        return (first - scene_position).manhattanLength() <= CLOSE_PX
