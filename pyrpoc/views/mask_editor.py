"""Authoring a mask from an acquired dataset.

Was ``gui/main_widgets/opto_control_mgr/mask_editor.py``. It is a view now: it
reads a bound dataset rather than reaching into a display widget's
``_data_chw``.

A finished mask leaves by being added to the dataset library as a ``Mask2D``
entry, which is the whole of how it reaches a modality. This view does not know
what a modality is, and the Modulation parameter that consumes masks does not
know this view exists -- it asks the library for entries matching ``Mask2D`` the
same way every view asks for its own sources. The alternative, and the reason
this is worth stating, was for one of the two to name the other.

``Save mask...`` stays, and is now only what it says: writing a PNG for
something outside this application to read. It is not how a mask gets used.

The threshold and polygon-ROI machinery is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
import random

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.streams import Image2D, Mask2D
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.data.io import utc_now
from pyrpoc.data.transforms import normalize_channels

from .base import View
from .range_slider import RangeSlider
from .registry import view_registry


def write_mask(path: Path | str, mask: np.ndarray) -> Path:
    """Write a 2-D mask to disk. Returns the path written.

    An export, not a step in using a mask -- nothing in this application reads
    the file back. It lives here because this editor is the only thing that
    authors a mask at all.
    """
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={array.shape}")
    resolved = Path(str(path)).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(resolved), array):
        raise OSError(f"failed to write a mask to '{resolved}'")
    return resolved


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
    def __init__(self, scene: QGraphicsScene, editor: "MaskEditorView"):
        super().__init__(scene)
        self.gscene: QGraphicsScene = scene
        self.editor = editor
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        self._drawing = False
        self._current_points: list[QPointF] = []
        self._live_path: QPainterPath | None = None
        self._live_path_item: QGraphicsPathItem | None = None
        self._path_pen = QPen(QColor(255, 80, 80), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

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


class MaskPreviewLabel(QLabel):
    """Shows a mask scaled to fit, and keeps fitting it as it is resized.

    Scaling once when the mask changes is not enough: the label is stretched
    by its layout after that, and a pixmap sized to the old geometry is
    silently clipped rather than re-fitted.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap
        self.rescale()

    def rescale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        # Nearest-neighbour: a mask is two values, and smoothing invents
        # greys that no pixel of it has.
        super().setPixmap(
            self._source.scaled(
                self.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.rescale()


class RoiThresholdDialog(QDialog):
    """Re-threshold one ROI against a live preview of the resulting mask.

    The preview is the same array ``Preview`` shows -- the whole mask, every
    ROI -- recomputed on each change. Showing only the edited ROI's own pixels
    would answer a question nobody asked: a threshold is chosen for how the
    finished mask looks, and the other ROIs are the context that decision is
    made in.

    The editor's stored ROI is untouched until the dialog is accepted, so the
    preview runs against a substituted copy and Cancel needs no undo.
    """

    def __init__(self, editor: "MaskEditorView", index: int):
        super().__init__(editor)
        self.editor = editor
        self.index = index
        roi = editor.rois()[index]
        self.setWindowTitle(f"ROI {index + 1} thresholds")

        int_min = int(np.floor(editor.data_min()))
        int_max = int(np.ceil(editor.data_max()))
        low = max(int_min, min(int_max, int(round(roi.threshold_low))))
        high = max(low, min(int_max, int(round(roi.threshold_high))))

        layout = QVBoxLayout(self)

        self.preview_label = MaskPreviewLabel(self)
        layout.addWidget(self.preview_label, 1)

        threshold_row = QHBoxLayout()
        self.low_spin = QSpinBox(self)
        self.high_spin = QSpinBox(self)
        for spin in (self.low_spin, self.high_spin):
            spin.setRange(int_min, int_max)
            spin.setFixedWidth(74)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.low_spin.valueChanged.connect(self.on_spin_changed)
        self.high_spin.valueChanged.connect(self.on_spin_changed)

        self.slider = RangeSlider(self)
        self.slider.setRange(int_min, int_max)
        self.slider.setValues(low, high)
        self.slider.values_changed.connect(self.on_slider_changed)

        threshold_row.addWidget(self.low_spin)
        threshold_row.addWidget(self.slider, 1)
        threshold_row.addWidget(self.high_spin)
        layout.addLayout(threshold_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(420, 520)
        self.update_preview()

    def values(self) -> tuple[int, int]:
        low = int(self.low_spin.value())
        high = int(self.high_spin.value())
        if high < low:
            low, high = high, low
        return low, high

    def write_values(self, low: int, high: int) -> None:
        controls = (self.low_spin, self.high_spin, self.slider)
        for control in controls:
            control.blockSignals(True)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.slider.setValues(low, high)
        for control in controls:
            control.blockSignals(False)

    def on_spin_changed(self, _value: int) -> None:
        low, high = self.values()
        self.write_values(low, high)
        self.update_preview()

    def on_slider_changed(self, low: int, high: int) -> None:
        self.write_values(low, high)
        self.update_preview()

    def previewed_rois(self) -> list[MaskRoi]:
        low, high = self.values()
        rois = list(self.editor.rois())
        if 0 <= self.index < len(rois):
            rois[self.index] = replace(
                rois[self.index], threshold_low=float(low), threshold_high=float(high)
            )
        return rois

    def update_preview(self) -> None:
        mask = self.editor.generate_mask(self.previewed_rois())
        if mask is None:
            self.preview_label.setText("No ROI to preview.")
            return
        height, width = mask.shape
        qimg = QImage(
            mask.tobytes(), width, height, width, QImage.Format.Format_Grayscale8
        ).copy()
        self.preview_label.set_source(QPixmap.fromImage(qimg))


@view_registry.register("mask_editor")
class MaskEditorView(View):
    """Draw thresholded polygon ROIs over an acquired image and file the mask."""

    display_name = "Mask Editor"
    renders = [Image2D]

    dirty_state_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.setObjectName("maskEditorRoot")
        self.setStyleSheet(
            "#maskEditorRoot, #maskEditorRoot QWidget { background: transparent; }"
            "#maskEditorRoot QGraphicsView, #maskEditorRoot QTableWidget { background: transparent; }"
        )
        self._rois: list[MaskRoi] = []
        self._dirty = False
        self._display_qimage: QImage | None = None

        self._data = np.zeros((1, 1, 1), dtype=np.float32)
        self._h = 1
        self._w = 1
        self._channel_visibility = [True]
        self._data_min = 0.0
        self._data_max = 1.0
        self.apply_new_data(None)

        self.build_ui()
        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.update_view_image()

    def build_ui(self) -> None:
        root = QHBoxLayout(self.body)

        left = QVBoxLayout()
        self.channels_row = QHBoxLayout()
        self.channels_row.addWidget(QLabel("Channels:", self))
        self.channel_boxes: list[QCheckBox] = []
        left.addLayout(self.channels_row)

        # One span, read left to right: the number at each end is the handle
        # beside it, and the accented groove between them is what is kept. No
        # "Low"/"High" labels, because the arrangement already says it.
        threshold_row = QHBoxLayout()
        int_min = int(np.floor(self._data_min))
        int_max = int(np.ceil(self._data_max))
        low_default, high_default = self.default_thresholds()

        self.low_spin = QSpinBox(self)
        self.high_spin = QSpinBox(self)
        for spin in (self.low_spin, self.high_spin):
            spin.setRange(int_min, int_max)
            spin.setFixedWidth(74)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.low_spin.setValue(low_default)
        self.high_spin.setValue(high_default)
        self.low_spin.valueChanged.connect(self.on_threshold_changed)
        self.high_spin.valueChanged.connect(self.on_threshold_changed)

        self.threshold_slider = RangeSlider(self)
        self.threshold_slider.setRange(int_min, int_max)
        self.threshold_slider.setValues(low_default, high_default)
        self.threshold_slider.setToolTip(
            "Pixels inside the accented span count toward the mask."
        )
        self.threshold_slider.values_changed.connect(self.on_slider_changed)

        threshold_row.addWidget(self.low_spin)
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.high_spin)
        left.addLayout(threshold_row)

        self.scene = QGraphicsScene(self)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.image_view = MaskImageView(self.scene, self)
        self.image_view.setMinimumSize(480, 320)
        left.addWidget(self.image_view, 1)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:", self))
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("what this mask is of")
        self.name_edit.returnPressed.connect(self.add_to_library)
        name_row.addWidget(self.name_edit, 1)
        add_btn = QPushButton("Add to library", self)
        add_btn.setToolTip(
            "File this mask as data. It then appears in the Modulation table's "
            "mask list, and in the data panel."
        )
        add_btn.clicked.connect(self.add_to_library)
        name_row.addWidget(add_btn)
        left.addLayout(name_row)

        button_row = QHBoxLayout()
        preview_btn = QPushButton("Preview", self)
        save_btn = QPushButton("Save mask...", self)
        save_btn.setToolTip("Export a PNG. Not needed to use the mask here.")
        preview_btn.clicked.connect(self.preview_mask)
        save_btn.clicked.connect(self.save_mask)
        button_row.addWidget(preview_btn)
        button_row.addWidget(save_btn)
        button_row.addStretch(1)
        left.addLayout(button_row)

        root.addLayout(left, 3)

        right = QVBoxLayout()
        self.roi_table = QTableWidget(0, 3, self)
        self.roi_table.setHorizontalHeaderLabels(["Low", "High", "Channels"])
        self.roi_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.roi_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        # Editing an ROI is done on the row that names it, rather than through
        # a button elsewhere that acts on whatever happens to be selected.
        self.roi_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.roi_table.customContextMenuRequested.connect(self.show_roi_menu)
        right.addWidget(self.roi_table)
        root.addLayout(right, 2)

    def apply_new_data(self, image_data: np.ndarray | None) -> None:
        self._data = self.coerce_input_data(image_data)
        self._h = int(self._data.shape[1])
        self._w = int(self._data.shape[2])
        self._channel_visibility = [True] * int(self._data.shape[0])
        self._data_min = float(np.min(self._data))
        self._data_max = float(np.max(self._data))
        if self._data_max <= self._data_min:
            self._data_max = self._data_min + 1e-9

    def rebuild_channel_boxes(self) -> None:
        while self.channels_row.count() > 1:
            item = self.channels_row.takeAt(1)
            if item is None:
                continue
            child = item.widget()
            if child is not None:
                child.deleteLater()
        self.channel_boxes = []
        for idx in range(self._data.shape[0]):
            cb = QCheckBox(f"C{idx + 1}", self)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, i=idx: self.on_channel_toggled(i, checked))
            self.channel_boxes.append(cb)
            self.channels_row.addWidget(cb)
        self.channels_row.addStretch(1)

    def default_thresholds(self) -> tuple[int, int]:
        low = int(round(self._data_min + 0.2 * (self._data_max - self._data_min)))
        high = int(round(self._data_min + 0.8 * (self._data_max - self._data_min)))
        return low, max(high, low)

    def reset_threshold_controls(self) -> None:
        int_min = int(np.floor(self._data_min))
        int_max = int(np.ceil(self._data_max))
        low_default, high_default = self.default_thresholds()
        self.low_spin.setRange(int_min, int_max)
        self.high_spin.setRange(int_min, int_max)
        self.threshold_slider.setRange(int_min, int_max)
        self.write_thresholds(low_default, high_default)

    def refresh(self) -> None:
        """Re-read the bound dataset.

        Editing state survives a same-shape frame: rebuilding the ROI table on
        every published frame would make the editor unusable during a live run,
        so it only resets when the image changes shape.
        """
        dataset = self.dataset()
        latest = dataset.latest() if dataset is not None else None
        scaled = None
        if latest is not None:
            normalized = normalize_channels(latest)
            if normalized is not None:
                scaled = normalized * 255.0

        if scaled is not None and self._data is not None and self._data.shape == scaled.shape:
            self.apply_new_data(scaled)
            self.update_view_image()
            return
        self.set_image_data(scaled)

    def clear(self) -> None:
        self.set_image_data(None)

    def set_image_data(self, image_data: np.ndarray | None) -> None:
        self.apply_new_data(image_data)
        self._rois.clear()
        self.sync_rois()
        self.rebuild_channel_boxes()
        self.reset_threshold_controls()
        self.set_dirty(False)
        self.update_view_image()

    def coerce_input_data(self, image_data: np.ndarray | None) -> np.ndarray:
        if image_data is None:
            return self.generate_default_data()
        arr = np.asarray(image_data, dtype=np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        if arr.ndim == 3:
            if arr.shape[0] <= 8:
                return arr
            if arr.shape[-1] <= 8:
                return np.moveaxis(arr, -1, 0)
        raise ValueError("image_data must be [H,W], [C,H,W], or [H,W,C] with <= 8 channels")

    def generate_default_data(self) -> np.ndarray:
        """A blank frame when nothing is bound.

        v3.0 synthesised a plausible-looking microscope image here, which is
        indistinguishable from real data at a glance. Blank is honest.
        """
        return np.zeros((1, 256, 256), dtype=np.float32)

    def clamp_scene_point(self, point: QPointF) -> QPointF:
        x = min(max(point.x(), 0.0), float(self._w - 1))
        y = min(max(point.y(), 0.0), float(self._h - 1))
        return QPointF(x, y)

    def on_channel_toggled(self, idx: int, checked: bool) -> None:
        if idx < 0 or idx >= len(self._channel_visibility):
            return
        self._channel_visibility[idx] = bool(checked)
        self.update_view_image()

    def write_thresholds(self, low: int, high: int) -> None:
        """Push one pair of values into every threshold control at once.

        Each control would otherwise echo the change back to the one that
        caused it, so they are all written with their signals blocked and the
        redraw is done once, here.
        """
        controls = (self.low_spin, self.high_spin, self.threshold_slider)
        for control in controls:
            control.blockSignals(True)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.threshold_slider.setValues(low, high)
        for control in controls:
            control.blockSignals(False)

    def on_threshold_changed(self, _value: int) -> None:
        low, high = self.coerced_thresholds()
        self.write_thresholds(low, high)
        self.update_view_image()

    def on_slider_changed(self, low: int, high: int) -> None:
        self.write_thresholds(low, high)
        self.update_view_image()

    def coerced_thresholds(self) -> tuple[int, int]:
        low = int(self.low_spin.value())
        high = int(self.high_spin.value())
        if high < low:
            low, high = high, low
        return low, high

    def update_view_image(self) -> None:
        display = np.zeros((self._h, self._w, 3), dtype=np.float32)
        low, high = self.coerced_thresholds()
        active = np.zeros((self._h, self._w), dtype=bool)
        channel_colors = (
            np.array([255.0, 64.0, 64.0], dtype=np.float32),
            np.array([64.0, 180.0, 255.0], dtype=np.float32),
            np.array([255.0, 180.0, 64.0], dtype=np.float32),
            np.array([180.0, 64.0, 255.0], dtype=np.float32),
            np.array([64.0, 255.0, 160.0], dtype=np.float32),
            np.array([255.0, 64.0, 160.0], dtype=np.float32),
            np.array([200.0, 255.0, 64.0], dtype=np.float32),
            np.array([64.0, 160.0, 255.0], dtype=np.float32),
        )

        for idx, channel in enumerate(self._data):
            if idx >= len(self._channel_visibility) or not self._channel_visibility[idx]:
                continue
            span = max(float(np.max(channel) - np.min(channel)), 1e-9)
            norm = (channel - float(np.min(channel))) / span
            color = channel_colors[idx % len(channel_colors)]
            display += norm[..., None] * color * 0.65
            active |= (channel >= low) & (channel <= high)

        display = np.clip(display, 0, 255)
        display[active] = 255

        rgb = display.astype(np.uint8)
        qimg = QImage(rgb.tobytes(), self._w, self._h, 3 * self._w, QImage.Format.Format_RGB888)
        self._display_qimage = qimg.copy()
        self.image_item.setPixmap(QPixmap.fromImage(self._display_qimage))
        self.scene.setSceneRect(QRectF(self._display_qimage.rect()))

    def add_roi(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return
        low, high = self.coerced_thresholds()
        self._rois.append(
            MaskRoi(
                points=[(float(x), float(y)) for x, y in points],
                threshold_low=float(low),
                threshold_high=float(high),
                active_channels=self._channel_visibility.copy(),
            )
        )
        self.sync_rois()
        self.set_dirty(True)

    def sync_rois(self) -> None:
        """Rebuild the table and the drawn outlines from ``self._rois``.

        Both are numbered by position, so both are rebuilt together rather
        than edited in place: a table row is numbered by Qt's vertical header,
        which is the row index, and the label on the image is that same index.
        """
        self.roi_table.setRowCount(len(self._rois))
        for row, roi in enumerate(self._rois):
            channels = ",".join(
                str(i + 1) for i, active in enumerate(roi.active_channels) if active
            )
            self.roi_table.setItem(row, 0, QTableWidgetItem(f"{roi.threshold_low:.1f}"))
            self.roi_table.setItem(row, 1, QTableWidgetItem(f"{roi.threshold_high:.1f}"))
            self.roi_table.setItem(row, 2, QTableWidgetItem(channels if channels else "-"))
        self.image_view.set_rois(self._rois)

    def show_roi_menu(self, pos) -> None:
        row = self.roi_table.rowAt(pos.y())
        if row < 0 or row >= len(self._rois):
            return
        self.roi_table.selectRow(row)
        menu = QMenu(self.roi_table)
        # Connected rather than compared against exec()'s return value: Qt
        # hides the menu before it emits triggered, so the dialog opens with
        # the menu already gone.
        menu.addAction("Change thresholds...").triggered.connect(
            lambda _checked=False, r=row: self.change_roi_thresholds(r)
        )
        menu.addAction("Delete").triggered.connect(
            lambda _checked=False, r=row: self.delete_roi(r)
        )
        menu.exec(self.roi_table.viewport().mapToGlobal(pos))

    def change_roi_thresholds(self, row: int) -> None:
        if row < 0 or row >= len(self._rois):
            return
        dialog = RoiThresholdDialog(self, row)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if row >= len(self._rois):
            return
        low, high = dialog.values()
        self._rois[row] = replace(
            self._rois[row], threshold_low=float(low), threshold_high=float(high)
        )
        self.sync_rois()
        self.set_dirty(True)

    def delete_roi(self, row: int) -> None:
        if row < 0 or row >= len(self._rois):
            return
        del self._rois[row]
        self.sync_rois()
        self.set_dirty(True)

    def rois(self) -> list[MaskRoi]:
        return self._rois

    def data_min(self) -> float:
        return self._data_min

    def data_max(self) -> float:
        return self._data_max

    def has_rois(self) -> bool:
        return len(self._rois) > 0

    def is_dirty(self) -> bool:
        return self._dirty

    def set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        self._dirty = dirty
        self.dirty_state_changed.emit(dirty)

    def generate_mask(self, rois: list[MaskRoi] | None = None) -> np.ndarray | None:
        """The mask *rois* would produce, defaulting to the drawn ones.

        Taking the list is what lets the threshold dialog preview an edit
        without writing it first: it passes a copy with one ROI substituted.
        """
        rois = self._rois if rois is None else rois
        if not rois:
            return None
        final_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        for roi in rois:
            if len(roi.points) < 3:
                continue
            polygon = np.array([[int(round(x)), int(round(y))] for x, y in roi.points], dtype=np.int32).reshape(-1, 1, 2)
            roi_mask = np.zeros((self._h, self._w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [polygon], 255)

            active = np.zeros((self._h, self._w), dtype=bool)
            for idx, channel in enumerate(self._data):
                if idx >= len(roi.active_channels) or not roi.active_channels[idx]:
                    continue
                active |= (channel >= roi.threshold_low) & (channel <= roi.threshold_high)

            final_mask[(roi_mask == 255) & active] = 255
        return final_mask

    def preview_mask(self) -> None:
        mask = self.generate_mask()
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before previewing.")
            return
        qimg = QImage(mask.tobytes(), self._w, self._h, self._w, QImage.Format.Format_Grayscale8).copy()
        dlg = QDialog(self)
        dlg.setWindowTitle("Mask Preview")
        layout = QVBoxLayout(dlg)
        label = QLabel(dlg)
        label.setPixmap(QPixmap.fromImage(qimg))
        layout.addWidget(label)
        dlg.resize(max(320, self._w), max(240, self._h))
        dlg.exec()

    def add_to_library(self) -> None:
        """File the drawn mask as a ``Mask2D`` dataset. The way a mask is used.

        A dataset with no run behind it: ``Provenance`` needs only a
        ``program_key``, and the ``run_id`` of 0 it defaults to is what says
        nothing acquired this. One ``append`` and it is done -- a mask is drawn
        once, not streamed.
        """
        library = self.library()
        if library is None:
            QMessageBox.warning(
                self, "No Library", "This view is not attached to the open data yet."
            )
            return
        mask = self.generate_mask()
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before adding.")
            return

        dataset = Dataset(
            stream="mask",
            spec=Mask2D,
            provenance=Provenance(
                program_key="mask_editor",
                started_at=utc_now(),
                name=self.name_edit.text().strip(),
            ),
        )
        try:
            dataset.append(mask)
        except Exception as exc:  # noqa: BLE001 - reported, not raised: Qt slot
            QMessageBox.critical(self, "Add Failed", f"Could not file that mask: {exc}")
            return
        library.add(dataset)
        self.set_dirty(False)
        self.name_edit.clear()

    def save_mask(self) -> None:
        mask = self.generate_mask()
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            "",
            "PNG (*.png);;TIFF (*.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        try:
            written = write_mask(path, mask)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to save mask to {path}: {exc}")
            return
        self.set_dirty(False)
        QMessageBox.information(self, "Mask Saved", f"Wrote a mask to {written}.")


