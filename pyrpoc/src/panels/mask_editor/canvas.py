"""What an ROI is, and the surface it is drawn on.

Split out of the panel because both are self-contained: ``MaskRoi`` is a
plain data model, and ``MaskImageView`` is a real widget with its own mouse
handling and its own drawn state (the live path, the outlines, the number
labels). It talks back to the panel through ``editor`` for the two things it
cannot decide on its own: where a point is allowed to land
(``clamp_scene_point``) and what a finished polygon becomes (``add_roi``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView

if TYPE_CHECKING:  # pragma: no cover
    from .panel import MaskEditorPanel


@dataclass
class MaskRoi:
    """One drawn region. Its number is its position in the editor's list.

    There is no stored id: a stable id and a displayed row number are two
    numberings of the same thing, and deleting an ROI made them disagree --
    the table renumbered, the labels drawn on the image did not.
    """

    points: list[tuple[float, float]]
    threshold_low: float
    threshold_high: float
    active_channels: list[bool]


class MaskImageView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, editor: MaskEditorPanel):
        super().__init__(scene)
        self.gscene: QGraphicsScene = scene
        self.editor = editor
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        self._drawing = False
        self._current_points: list[QPointF] = []
        self._live_path: QPainterPath | None = None
        self._live_path_item: QGraphicsPathItem | None = None
        self._path_pen = QPen(
            QColor(255, 80, 80), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
        )

        self._roi_items: list[QGraphicsPathItem] = []
        self._roi_labels: list[QGraphicsTextItem] = []
        self._zoom_level = 0

    def wheelEvent(self, event) -> None:
        if event is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self._zoom_level += 1 if factor > 1.0 else -1
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            scene_pos = self.editor.clamp_scene_point(scene_pos)
            self._drawing = True
            self._current_points = [scene_pos]
            self._live_path = QPainterPath(scene_pos)
            self.clear_live_path()
            self._live_path_item = self.gscene.addPath(self._live_path, self._path_pen)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event is None:
            return
        if self._drawing and self._live_path is not None:
            scene_pos = self.mapToScene(event.pos())
            scene_pos = self.editor.clamp_scene_point(scene_pos)
            self._current_points.append(scene_pos)
            self._live_path.lineTo(scene_pos)
            if self._live_path_item is not None:
                self._live_path_item.setPath(self._live_path)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            points = [(float(p.x()), float(p.y())) for p in self._current_points]
            self.clear_live_path()
            self._current_points = []
            self._live_path = None
            self.editor.add_roi(points)
            return
        super().mouseReleaseEvent(event)

    def clear_live_path(self) -> None:
        if self._live_path_item is not None:
            self.gscene.removeItem(self._live_path_item)
            self._live_path_item = None

    def clear_rois(self) -> None:
        for item in self._roi_items:
            self.gscene.removeItem(item)
        for label in self._roi_labels:
            self.gscene.removeItem(label)
        self._roi_items = []
        self._roi_labels = []

    def set_rois(self, rois: list[MaskRoi]) -> None:
        """Redraw every ROI. The only way the drawn numbers change.

        Redrawing all of them on any edit is what keeps the labels honest:
        deleting one shifts the number of every ROI after it, so there is no
        such thing as removing one drawing and leaving the rest alone.
        """
        self.clear_rois()
        for index, roi in enumerate(rois):
            self.draw_roi(index, roi)

    def draw_roi(self, index: int, roi: MaskRoi) -> None:
        if len(roi.points) < 3:
            return

        color = self.color_for_index(index)
        path = QPainterPath(QPointF(roi.points[0][0], roi.points[0][1]))
        for x, y in roi.points[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()

        outline = cast(QGraphicsPathItem, self.gscene.addPath(path, QPen(color, 2)))
        outline.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
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

    def color_for_index(self, index: int) -> QColor:
        rng = random.Random(index * 17 + 11)
        return QColor(rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))
