"""What an ROI is, and the surface it is drawn on.

The view asks the panel (``editor``) the two things it cannot decide itself:
where a point may land, and what a finished polygon becomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView

if TYPE_CHECKING:  # pragma: no cover
    from .panel import MaskEditorPanel


@dataclass
class MaskRoi:
    """One drawn region. Its number is its position in the editor's list; a
    stored id would be a second numbering that deleting an ROI puts out of step."""

    points: list[tuple[float, float]]
    threshold_low: float
    threshold_high: float
    active_channels: list[bool]


class MaskImageView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, editor: MaskEditorPanel):
        super().__init__(scene)
        self.gscene = scene
        self.editor = editor
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        self._drawing = False
        self._current_points: list[QPointF] = []
        self._live_path_item: QGraphicsPathItem | None = None
        self._path_pen = QPen(
            QColor(255, 80, 80), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
        )
        self._roi_items: list[QGraphicsPathItem] = []
        self._roi_labels: list[QGraphicsTextItem] = []

    # Qt types every event as optional; the None checks narrow for that.
    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        scene_pos = self.editor.clamp_scene_point(self.mapToScene(event.pos()))
        self._drawing = True
        self._current_points = [scene_pos]
        self.clear_live_path()
        self._live_path_item = QGraphicsPathItem(QPainterPath(scene_pos))
        self._live_path_item.setPen(self._path_pen)
        self.gscene.addItem(self._live_path_item)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None or self._live_path_item is None:
            super().mouseMoveEvent(event)
            return
        scene_pos = self.editor.clamp_scene_point(self.mapToScene(event.pos()))
        self._current_points.append(scene_pos)
        path = self._live_path_item.path()
        path.lineTo(scene_pos)
        self._live_path_item.setPath(path)

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
        """Redraw every ROI. Deleting one renumbers every ROI after it, so all
        of them are redrawn on any edit."""
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
        path = QPainterPath(QPointF(*roi.points[0]))
        for x, y in roi.points[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()

        outline = QGraphicsPathItem(path)
        outline.setPen(QPen(color, 2))
        outline.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
        self.gscene.addItem(outline)
        self._roi_items.append(outline)

        label = QGraphicsTextItem(str(index + 1))
        label.setDefaultTextColor(Qt.GlobalColor.white)
        bounds = label.boundingRect()
        cx = sum(p[0] for p in roi.points) / len(roi.points)
        cy = sum(p[1] for p in roi.points) / len(roi.points)
        label.setPos(cx - bounds.width() / 2, cy - bounds.height() / 2)
        label.setZValue(10_000)
        self.gscene.addItem(label)
        self._roi_labels.append(label)


def color_for_index(index: int) -> QColor:
    """A stable, distinct colour per ROI number."""
    rng = random.Random(index * 17 + 11)
    return QColor(rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))
