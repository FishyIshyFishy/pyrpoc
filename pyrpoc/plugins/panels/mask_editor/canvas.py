"""What an ROI is, and the surface it is drawn on."""

# The view asks the panel (``editor``) the two things it cannot decide itself: where a point may
# land, and what a finished polygon becomes.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

if TYPE_CHECKING:  # pragma: no cover
    from .panel import MaskEditorPanel


@dataclass
class Threshold:
    """Which pixels of an ROI's shape count: those within ``low..high`` on any
    active channel."""

    low: float
    high: float
    active_channels: list[bool]


# ``eq=False`` because comparing arrays in a dataclass's ``__eq__`` raises.
@dataclass(eq=False)
class MaskRoi:
    """One region and its threshold; without a threshold, every pixel of the shape counts. The
    shape is a drawn polygon, ``points``, or the lit ``pixels`` of a loaded mask."""

    points: list[tuple[float, float]]
    threshold: Threshold | None
    # On the image's grid; None for a drawn polygon.
    pixels: np.ndarray | None


class MaskImageView(QGraphicsView):
    """The image, fitted to the view until the wheel zooms it, with every ROI outlined."""

    def __init__(self, scene: QGraphicsScene, editor: MaskEditorPanel):
        super().__init__(scene)
        self.gscene = scene
        self.editor = editor
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.show_backdrop()
        # Off once the wheel zooms, so a resize does not undo the user's zoom.
        self.fitting = True

        self._drawing = False
        self._current_points: list[QPointF] = []
        self._live_path_item: QGraphicsPathItem | None = None
        # Kept here, not read back from the item: the item drops a path that is
        # only its start point, so the first segment would start at the origin.
        self._live_path = QPainterPath()
        self._path_pen = QPen(
            QColor(255, 80, 80), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
        )
        self._roi_items: list[QGraphicsPathItem] = []
        self._roi_labels: list[QGraphicsTextItem] = []

    # Qt types every event as optional; the None checks narrow for that.
    def show_backdrop(self) -> None:
        self.setBackgroundBrush(self.palette().color(QPalette.ColorRole.Base))

    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        # The backdrop and the mask's tint are the theme's colors, so a new theme redraws both.
        if event is not None and event.type() == QEvent.Type.PaletteChange:
            self.show_backdrop()
            self.editor.update_view_image()

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self.fitting = False
        self.scale(factor, factor)

    def fit(self) -> None:
        """Show the whole image, as large as the view allows, and keep it fitted."""
        self.fitting = True
        self.fitInView(self.gscene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if self.fitting:
            self.fitInView(self.gscene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        scene_pos = self.editor.clamp_scene_point(self.mapToScene(event.pos()))
        self._drawing = True
        self._current_points = [scene_pos]
        self.clear_live_path()
        self._live_path = QPainterPath(scene_pos)
        self._live_path_item = QGraphicsPathItem()
        self._live_path_item.setPen(self._path_pen)
        self.gscene.addItem(self._live_path_item)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None or self._live_path_item is None:
            super().mouseMoveEvent(event)
            return
        scene_pos = self.editor.clamp_scene_point(self.mapToScene(event.pos()))
        self._current_points.append(scene_pos)
        self._live_path.lineTo(scene_pos)
        self._live_path_item.setPath(self._live_path)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            super().mouseReleaseEvent(event)
            return
        self._drawing = False
        points = [(p.x(), p.y()) for p in self._current_points]
        self.clear_live_path()
        self._current_points = []
        self.editor.add_roi(points)

    def clear_live_path(self) -> None:
        if self._live_path_item is not None:
            self.gscene.removeItem(self._live_path_item)
            self._live_path_item = None

    def set_rois(self, rois: list[MaskRoi]) -> None:
        """Redraw every ROI."""
        # Deleting one renumbers every ROI after it, so all of them are redrawn on any edit.
        for item in self._roi_items:
            self.gscene.removeItem(item)
        for label in self._roi_labels:
            self.gscene.removeItem(label)
        self._roi_items = []
        self._roi_labels = []
        for index, roi in enumerate(rois):
            self.draw_roi(index, roi)

    def draw_roi(self, index: int, roi: MaskRoi) -> None:
        color = color_for_index(index)
        outline = QGraphicsPathItem(outline_path(roi))
        outline.setPen(QPen(color, 2))
        outline.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
        self.gscene.addItem(outline)
        self._roi_items.append(outline)

        label = QGraphicsTextItem(str(index + 1))
        label.setDefaultTextColor(Qt.GlobalColor.white)
        bounds = label.boundingRect()
        cx, cy = roi_center(roi)
        label.setPos(cx - bounds.width() / 2, cy - bounds.height() / 2)
        label.setZValue(10_000)
        self.gscene.addItem(label)
        self._roi_labels.append(label)


def outline_path(roi: MaskRoi) -> QPainterPath:
    """The region's outline: its polygon, or every edge of its pixels, holes included."""
    if roi.pixels is None:
        path = QPainterPath(QPointF(*roi.points[0]))
        for x, y in roi.points[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()
        return path
    contours, _ = cv2.findContours(
        roi.pixels.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    path = QPainterPath()
    # Odd-even, so a hole inside a region is left unfilled.
    path.setFillRule(Qt.FillRule.OddEvenFill)
    for contour in contours:
        # Through the pixels' centres, where the scene puts them.
        points = [QPointF(x + 0.5, y + 0.5) for x, y in contour.reshape(-1, 2)]
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()
    return path


def roi_center(roi: MaskRoi) -> tuple[float, float]:
    """Where the region's number goes: the mean of its points, or of its lit pixels."""
    if roi.pixels is None:
        xs, ys = zip(*roi.points, strict=True)
        return sum(xs) / len(xs), sum(ys) / len(ys)
    rows, cols = np.nonzero(roi.pixels)
    return float(cols.mean()) + 0.5, float(rows.mean()) + 0.5


def color_for_index(index: int) -> QColor:
    """A stable, distinct colour per ROI number."""
    rng = random.Random(index * 17 + 11)
    return QColor(rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))
