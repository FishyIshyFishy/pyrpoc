from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class MaskRoi:
    roi_id: int
    points: list[tuple[float, float]]
    threshold_low: float
    threshold_high: float
    active_channels: list[bool]


class _MaskImageView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, editor: "MaskEditorWidget"):
        super().__init__(scene)
        self.editor = editor
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        self._drawing = False
        self._current_points: list[QPointF] = []
        self._live_path: QPainterPath | None = None
        self._live_path_item: QGraphicsPathItem | None = None
        self._path_pen = QPen(QColor(255, 80, 80), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

        self._roi_items: dict[int, QGraphicsPathItem] = {}
        self._roi_labels: dict[int, QGraphicsTextItem] = {}
        self._zoom_level = 0

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self._zoom_level += 1 if factor > 1.0 else -1
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            scene_pos = self.editor.clamp_scene_point(scene_pos)
            self._drawing = True
            self._current_points = [scene_pos]
            self._live_path = QPainterPath(scene_pos)
            self._clear_live_path()
            self._live_path_item = self.scene().addPath(self._live_path, self._path_pen)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
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
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            points = [(float(p.x()), float(p.y())) for p in self._current_points]
            self._clear_live_path()
            self._current_points = []
            self._live_path = None
            self.editor.add_roi(points)
            return
        super().mouseReleaseEvent(event)

    def _clear_live_path(self) -> None:
        if self._live_path_item is not None:
            self.scene().removeItem(self._live_path_item)
            self._live_path_item = None

    def clear_rois(self) -> None:
        for roi_id in list(self._roi_items.keys()):
            self.remove_roi(roi_id)

    def remove_roi(self, roi_id: int) -> None:
        item = self._roi_items.pop(roi_id, None)
        if item is not None:
            self.scene().removeItem(item)
        label = self._roi_labels.pop(roi_id, None)
        if label is not None:
            self.scene().removeItem(label)

    def add_roi(self, roi: MaskRoi) -> None:
        self.remove_roi(roi.roi_id)
        if len(roi.points) < 3:
            return

        color = self._color_for_roi(roi.roi_id)
        path = QPainterPath(QPointF(roi.points[0][0], roi.points[0][1]))
        for x, y in roi.points[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()

        outline = self.scene().addPath(path, QPen(color, 2))
        outline.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
        self._roi_items[roi.roi_id] = outline

        label = QGraphicsTextItem(str(roi.roi_id))
        label.setDefaultTextColor(Qt.GlobalColor.white)
        bounds = label.boundingRect()
        cx = sum(p[0] for p in roi.points) / len(roi.points)
        cy = sum(p[1] for p in roi.points) / len(roi.points)
        label.setPos(cx - bounds.width() / 2, cy - bounds.height() / 2)
        label.setZValue(10_000)
        self.scene().addItem(label)
        self._roi_labels[roi.roi_id] = label

    def _color_for_roi(self, roi_id: int) -> QColor:
        rng = random.Random(roi_id * 17 + 11)
        return QColor(rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))


class MaskEditorWidget(QWidget):
    create_mask_requested = pyqtSignal(object)
    mask_saved = pyqtSignal(object, object)
    cancel_requested = pyqtSignal()
    dirty_state_changed = pyqtSignal(bool)

    def __init__(self, image_data: np.ndarray | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("maskEditorRoot")
        self.setStyleSheet(
            "#maskEditorRoot, #maskEditorRoot QWidget { background: transparent; }"
            "#maskEditorRoot QGraphicsView, #maskEditorRoot QTableWidget { background: transparent; }"
        )
        self._rois: list[MaskRoi] = []
        self._next_roi_id = 1
        self._dirty = False
        self._display_qimage: QImage | None = None

        self._data = np.zeros((1, 1, 1), dtype=np.float32)
        self._h = 1
        self._w = 1
        self._channel_visibility = [True]
        self._data_min = 0.0
        self._data_max = 1.0
        self._apply_new_data(image_data)

        self._build_ui()
        self._rebuild_channel_boxes()
        self._reset_threshold_controls()
        self._update_view_image()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.channels_row = QHBoxLayout()
        self.channels_row.addWidget(QLabel("Channels:", self))
        self.channel_boxes: list[QCheckBox] = []
        left.addLayout(self.channels_row)

        threshold_form = QFormLayout()
        self.low_spin = QSpinBox(self)
        self.high_spin = QSpinBox(self)
        int_min = int(np.floor(self._data_min))
        int_max = int(np.ceil(self._data_max))
        self.low_spin.setRange(int_min, int_max)
        self.high_spin.setRange(int_min, int_max)
        low_default = int(round(self._data_min + 0.2 * (self._data_max - self._data_min)))
        high_default = int(round(self._data_min + 0.8 * (self._data_max - self._data_min)))
        if high_default < low_default:
            high_default = low_default
        self.low_spin.setValue(low_default)
        self.high_spin.setValue(high_default)
        self.low_spin.valueChanged.connect(self._on_threshold_changed)
        self.high_spin.valueChanged.connect(self._on_threshold_changed)

        self.low_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.high_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.low_slider.setRange(int_min, int_max)
        self.high_slider.setRange(int_min, int_max)
        self.low_slider.setValue(low_default)
        self.high_slider.setValue(high_default)
        self.low_slider.valueChanged.connect(self._on_low_slider_changed)
        self.high_slider.valueChanged.connect(self._on_high_slider_changed)

        threshold_form.addRow("Low", self.low_spin)
        threshold_form.addRow("High", self.high_spin)
        threshold_form.addRow("Low Slider", self.low_slider)
        threshold_form.addRow("High Slider", self.high_slider)
        left.addLayout(threshold_form)

        self.scene = QGraphicsScene(self)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.image_view = _MaskImageView(self.scene, self)
        self.image_view.setMinimumSize(480, 320)
        left.addWidget(self.image_view, 1)

        button_row = QHBoxLayout()
        preview_btn = QPushButton("Preview", self)
        save_btn = QPushButton("Save", self)
        create_btn = QPushButton("Create Mask", self)
        cancel_btn = QPushButton("Cancel", self)
        delete_btn = QPushButton("Delete Selected ROI", self)
        preview_btn.clicked.connect(self.preview_mask)
        save_btn.clicked.connect(self.save_mask)
        create_btn.clicked.connect(self._emit_create)
        cancel_btn.clicked.connect(self.cancel_requested.emit)
        delete_btn.clicked.connect(self.delete_selected_roi)
        button_row.addWidget(preview_btn)
        button_row.addWidget(save_btn)
        button_row.addWidget(delete_btn)
        button_row.addStretch(1)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(create_btn)
        left.addLayout(button_row)

        root.addLayout(left, 3)

        right = QVBoxLayout()
        self.roi_table = QTableWidget(0, 4, self)
        self.roi_table.setHorizontalHeaderLabels(["ROI", "Low", "High", "Channels"])
        self.roi_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.roi_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        right.addWidget(self.roi_table)
        root.addLayout(right, 2)

    def _apply_new_data(self, image_data: np.ndarray | None) -> None:
        self._data = self._coerce_input_data(image_data)
        self._h = int(self._data.shape[1])
        self._w = int(self._data.shape[2])
        self._channel_visibility = [True] * int(self._data.shape[0])
        self._data_min = float(np.min(self._data))
        self._data_max = float(np.max(self._data))
        if self._data_max <= self._data_min:
            self._data_max = self._data_min + 1e-9

    def _rebuild_channel_boxes(self) -> None:
        while self.channels_row.count() > 1:
            item = self.channels_row.takeAt(1)
            child = item.widget()
            if child is not None:
                child.deleteLater()
        self.channel_boxes = []
        for idx in range(self._data.shape[0]):
            cb = QCheckBox(f"C{idx + 1}", self)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, i=idx: self._on_channel_toggled(i, checked))
            self.channel_boxes.append(cb)
            self.channels_row.addWidget(cb)
        self.channels_row.addStretch(1)

    def _reset_threshold_controls(self) -> None:
        int_min = int(np.floor(self._data_min))
        int_max = int(np.ceil(self._data_max))
        low_default = int(round(self._data_min + 0.2 * (self._data_max - self._data_min)))
        high_default = int(round(self._data_min + 0.8 * (self._data_max - self._data_min)))
        if high_default < low_default:
            high_default = low_default
        self.low_spin.blockSignals(True)
        self.high_spin.blockSignals(True)
        self.low_slider.blockSignals(True)
        self.high_slider.blockSignals(True)
        self.low_spin.setRange(int_min, int_max)
        self.high_spin.setRange(int_min, int_max)
        self.low_slider.setRange(int_min, int_max)
        self.high_slider.setRange(int_min, int_max)
        self.low_spin.setValue(low_default)
        self.high_spin.setValue(high_default)
        self.low_slider.setValue(low_default)
        self.high_slider.setValue(high_default)
        self.low_spin.blockSignals(False)
        self.high_spin.blockSignals(False)
        self.low_slider.blockSignals(False)
        self.high_slider.blockSignals(False)

    def set_image_data(self, image_data: np.ndarray | None) -> None:
        self._apply_new_data(image_data)
        self._rois.clear()
        self._next_roi_id = 1
        self.roi_table.setRowCount(0)
        self.image_view.clear_rois()
        self._rebuild_channel_boxes()
        self._reset_threshold_controls()
        self._set_dirty(False)
        self._update_view_image()

    def _coerce_input_data(self, image_data: np.ndarray | None) -> np.ndarray:
        if image_data is None:
            return self._generate_default_data()
        arr = np.asarray(image_data, dtype=np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        if arr.ndim == 3:
            if arr.shape[0] <= 8:
                return arr
            if arr.shape[-1] <= 8:
                return np.moveaxis(arr, -1, 0)
        raise ValueError("image_data must be [H,W], [C,H,W], or [H,W,C] with <= 8 channels")

    def _generate_default_data(self) -> np.ndarray:
        rng = np.random.default_rng(20260217)
        h, w = 256, 256
        channels = []
        yy, xx = np.mgrid[0:h, 0:w]
        for i in range(3):
            base = rng.normal(loc=35 + 20 * i, scale=20, size=(h, w)).astype(np.float32)
            cx = 65 + i * 45
            cy = 80 + i * 55
            blob = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2.0 * (20 + i * 8) ** 2))
            wave = 30.0 * np.sin((xx + i * 11) * 0.05) * np.cos((yy + i * 7) * 0.04)
            channel = np.clip(base + blob * 160 + wave + i * 30, 0, 255)
            channels.append(channel.astype(np.float32))
        return np.stack(channels, axis=0)

    def clamp_scene_point(self, point: QPointF) -> QPointF:
        x = min(max(point.x(), 0.0), float(self._w - 1))
        y = min(max(point.y(), 0.0), float(self._h - 1))
        return QPointF(x, y)

    def _on_channel_toggled(self, idx: int, checked: bool) -> None:
        if idx < 0 or idx >= len(self._channel_visibility):
            return
        self._channel_visibility[idx] = bool(checked)
        self._update_view_image()

    def _on_threshold_changed(self, _value: int) -> None:
        low, high = self._coerced_thresholds()
        self.low_spin.blockSignals(True)
        self.high_spin.blockSignals(True)
        self.low_slider.blockSignals(True)
        self.high_slider.blockSignals(True)
        self.low_spin.setValue(low)
        self.high_spin.setValue(high)
        self.low_slider.setValue(low)
        self.high_slider.setValue(high)
        self.low_spin.blockSignals(False)
        self.high_spin.blockSignals(False)
        self.low_slider.blockSignals(False)
        self.high_slider.blockSignals(False)
        self._update_view_image()

    def _on_low_slider_changed(self, value: int) -> None:
        self.low_spin.setValue(value)

    def _on_high_slider_changed(self, value: int) -> None:
        self.high_spin.setValue(value)

    def _coerced_thresholds(self) -> tuple[int, int]:
        low = int(self.low_spin.value())
        high = int(self.high_spin.value())
        if high < low:
            low, high = high, low
        return low, high

    def _update_view_image(self) -> None:
        display = np.zeros((self._h, self._w, 3), dtype=np.float32)
        low, high = self._coerced_thresholds()
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
        qimg = QImage(rgb.data, self._w, self._h, 3 * self._w, QImage.Format.Format_RGB888)
        self._display_qimage = qimg.copy()
        self.image_item.setPixmap(QPixmap.fromImage(self._display_qimage))
        self.scene.setSceneRect(QRectF(self._display_qimage.rect()))

    def add_roi(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return
        low, high = self._coerced_thresholds()
        roi = MaskRoi(
            roi_id=self._next_roi_id,
            points=[(float(x), float(y)) for x, y in points],
            threshold_low=float(low),
            threshold_high=float(high),
            active_channels=self._channel_visibility.copy(),
        )
        self._next_roi_id += 1
        self._rois.append(roi)
        self.image_view.add_roi(roi)
        self._upsert_roi_row(roi)
        self._set_dirty(True)

    def _upsert_roi_row(self, roi: MaskRoi) -> None:
        row = self.roi_table.rowCount()
        self.roi_table.insertRow(row)
        id_item = QTableWidgetItem(f"ROI {roi.roi_id}")
        id_item.setData(Qt.ItemDataRole.UserRole, roi.roi_id)
        self.roi_table.setItem(row, 0, id_item)
        self.roi_table.setItem(row, 1, QTableWidgetItem(f"{roi.threshold_low:.1f}"))
        self.roi_table.setItem(row, 2, QTableWidgetItem(f"{roi.threshold_high:.1f}"))
        channels_text = ",".join(str(i + 1) for i, active in enumerate(roi.active_channels) if active)
        self.roi_table.setItem(row, 3, QTableWidgetItem(channels_text if channels_text else "-"))

    def delete_selected_roi(self) -> None:
        row = self.roi_table.currentRow()
        if row < 0:
            return
        item = self.roi_table.item(row, 0)
        if item is None:
            return
        roi_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(roi_id, int):
            return
        self.roi_table.removeRow(row)
        self._rois = [roi for roi in self._rois if roi.roi_id != roi_id]
        self.image_view.remove_roi(roi_id)
        self._set_dirty(True)

    def has_rois(self) -> bool:
        return len(self._rois) > 0

    def is_dirty(self) -> bool:
        return self._dirty

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        self._dirty = dirty
        self.dirty_state_changed.emit(dirty)

    def generate_mask(self) -> np.ndarray | None:
        if not self._rois:
            return None
        final_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        for roi in self._rois:
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
        qimg = QImage(mask.data, self._w, self._h, self._w, QImage.Format.Format_Grayscale8).copy()
        dlg = QDialog(self)
        dlg.setWindowTitle("Mask Preview")
        layout = QVBoxLayout(dlg)
        label = QLabel(dlg)
        label.setPixmap(QPixmap.fromImage(qimg))
        layout.addWidget(label)
        dlg.resize(max(320, self._w), max(240, self._h))
        dlg.exec()

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
        ok = cv2.imwrite(str(Path(path)), mask)
        if not ok:
            QMessageBox.critical(self, "Save Failed", f"Failed to save mask to '{path}'.")
            return
        self.mask_saved.emit(str(path), mask.astype(np.uint8))

    def _emit_create(self) -> None:
        mask = self.generate_mask()
        if mask is None:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI before creating the mask.")
            return
        self.create_mask_requested.emit(mask.astype(np.uint8))

