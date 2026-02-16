from __future__ import annotations

import random

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from pyrpoc.rpoc.segmentation_methods import labels_to_contours, segment
from pyrpoc.rpoc.types import RPOCEditorState, RPOCImageInput, RPOCRoi


class _ImageViewer(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, editor: "RPOCMaskEditor"):
        super().__init__(scene)
        self.editor = editor
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.live_path_item: QGraphicsPathItem | None = None
        self.drawing = False
        self.current_path: QPainterPath | None = None
        self.current_points: list[QPointF] = []
        self.path_pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

        self._zoom = 0
        self.roi_opacity = 0.4
        self.roi_items: dict[int, QGraphicsPathItem] = {}
        self.roi_label_items: dict[int, QGraphicsTextItem] = {}

        self.cursor_brush = QGraphicsEllipseItem(0, 0, 8, 8)
        self.cursor_brush.setBrush(QColor(100, 170, 255))
        self.cursor_brush.setPen(QPen(Qt.PenStyle.NoPen))
        self.cursor_brush.setZValue(1000)
        self.cursor_brush.setVisible(False)
        self.scene().addItem(self.cursor_brush)

    def wheelEvent(self, event):
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom += 1 if zoom_factor > 1 else -1
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            pos = self.mapToScene(event.pos())
            if not self.drawing:
                self.drawing = True
                self.current_points = [pos]
                self.current_path = QPainterPath()
                self.current_path.moveTo(pos)
                if self.live_path_item:
                    self.scene().removeItem(self.live_path_item)
                self.live_path_item = self.scene().addPath(self.current_path, self.path_pen)
            else:
                self._finish_path(pos)
            return

        if event.button() == Qt.MouseButton.LeftButton and not self.drawing:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_path is not None:
            pos = self.mapToScene(event.pos())
            self.current_points.append(pos)
            self.current_path.lineTo(pos)
            if self.live_path_item:
                self.live_path_item.setPath(self.current_path)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.drawing:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _finish_path(self, end_pos: QPointF) -> None:
        if self.current_path is None:
            return
        self.drawing = False
        self.current_points.append(end_pos)
        self.current_path.lineTo(end_pos)
        self.current_path.closeSubpath()
        if self.live_path_item:
            self.scene().removeItem(self.live_path_item)
            self.live_path_item = None

        points = [(float(p.x()), float(p.y())) for p in self.current_points]
        self.editor.add_roi(points)
        self.current_path = None
        self.current_points = []

    def add_roi_graphics(self, roi_id: int, points: list[tuple[float, float]]) -> None:
        if roi_id in self.roi_items:
            self.remove_roi_graphics(roi_id)
        if len(points) < 3:
            return

        path = QPainterPath()
        path.moveTo(QPointF(points[0][0], points[0][1]))
        for x, y in points[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()

        color = self._random_color()
        roi_item = self.scene().addPath(path, QPen(color, 2))
        roi_item.setBrush(color)
        roi_item.setOpacity(self.roi_opacity if self.editor.state.show_rois else 0.0)
        self.roi_items[roi_id] = roi_item

        lbl = self._create_roi_label(roi_id, points)
        self.roi_label_items[roi_id] = lbl

    def remove_roi_graphics(self, roi_id: int) -> None:
        roi_item = self.roi_items.pop(roi_id, None)
        if roi_item is not None:
            self.scene().removeItem(roi_item)
        lbl = self.roi_label_items.pop(roi_id, None)
        if lbl is not None:
            self.scene().removeItem(lbl)

    def clear_roi_graphics(self) -> None:
        for roi_id in list(self.roi_items.keys()):
            self.remove_roi_graphics(roi_id)

    def _create_roi_label(self, roi_id: int, points: list[tuple[float, float]]) -> QGraphicsTextItem:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        label_item = QGraphicsTextItem(str(roi_id))
        label_item.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        label_item.setDefaultTextColor(Qt.GlobalColor.white)
        bounds = label_item.boundingRect()
        label_item.setPos(cx - bounds.width() / 2, cy - bounds.height() / 2)
        label_item.setZValue(999)
        label_item.setVisible(self.editor.state.show_rois and self.editor.state.show_labels)
        self.scene().addItem(label_item)
        return label_item

    def _random_color(self) -> QColor:
        return QColor(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def update_roi_visibility(self) -> None:
        for roi_item in self.roi_items.values():
            roi_item.setOpacity(self.roi_opacity if self.editor.state.show_rois else 0.0)
        for item in self.roi_label_items.values():
            item.setVisible(self.editor.state.show_rois and self.editor.state.show_labels)


class RPOCMaskEditor(QMainWindow):
    mask_created = pyqtSignal(object)

    _SLIDER_MIN = 0
    _SLIDER_MAX = 10000

    def __init__(
        self,
        image_data: np.ndarray | RPOCImageInput | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("RPOC Mask Editor")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.state = RPOCEditorState()
        self._emitted = False
        self._closing_via_create = False
        self.threshold_mode = "normalized_0_1"
        self.data_min = 0.0
        self.data_max = 1.0

        self.roi_table = QTableWidget(0, 5)
        self.roi_table.setHorizontalHeaderLabels(
            ["ROI Name", "Coordinates", "Lower Threshold", "Upper Threshold", "Modulation Level"]
        )
        self.roi_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.roi_table.customContextMenuRequested.connect(self._show_table_context_menu)

        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self._load_image)
        segment_button = QPushButton("Segment")
        segment_button.clicked.connect(self._segment)
        preview_button = QPushButton("Preview")
        preview_button.clicked.connect(self._preview_mask)
        save_button = QPushButton("Save Mask")
        save_button.clicked.connect(self._save_mask)
        create_button = QPushButton("Create Mask")
        create_button.clicked.connect(self.create_and_close)

        self.mask_checkbox = QCheckBox("Mask [M]")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.stateChanged.connect(self._toggle_mask_visibility)
        self.label_checkbox = QCheckBox("Labels [N]")
        self.label_checkbox.setChecked(True)
        self.label_checkbox.stateChanged.connect(self._toggle_label_visibility)

        self.threshold_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(self._SLIDER_MIN)
        self.threshold_slider.setMaximum(self._SLIDER_MAX)
        self.threshold_slider.setValue((2000, 8000))
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider_changed)

        top_row = QHBoxLayout()
        top_row.addWidget(load_button)
        top_row.addWidget(segment_button)
        top_row.addWidget(preview_button)
        top_row.addWidget(save_button)
        top_row.addWidget(create_button)
        top_row.addStretch(1)

        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.mask_checkbox)
        toggle_row.addWidget(self.label_checkbox)
        toggle_row.addWidget(QLabel("Threshold"))
        toggle_row.addWidget(self.threshold_slider, 1)

        self.channels_row = QHBoxLayout()
        self.channel_checkboxes: list[QCheckBox] = []
        self.image_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]

        self.image_scene = QGraphicsScene()
        self.image_item = self.image_scene.addPixmap(QPixmap())
        self.image_view = _ImageViewer(self.image_scene, self)

        left = QVBoxLayout()
        left.addLayout(self.channels_row)
        left.addLayout(top_row)
        left.addLayout(toggle_row)
        left.addWidget(self.image_view, 1)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addWidget(self.roi_table, 2)

        central = QWidget(self)
        central.setLayout(root)
        self.setCentralWidget(central)

        if image_data is not None:
            self.set_image_data(image_data)
        else:
            self._update_displayed_image()

    def set_image_data(self, image_data: np.ndarray | RPOCImageInput) -> None:
        if isinstance(image_data, RPOCImageInput):
            image_input = image_data
        else:
            image_input = RPOCImageInput.from_array(image_data)

        self.state.image_input = image_input
        self.state.channel_visibility = [True] * image_input.data.shape[0]
        self.state.rois = []
        self.state.next_roi_id = 1
        self.state.show_rois = True
        self.state.show_labels = True
        self.mask_checkbox.setChecked(True)
        self.label_checkbox.setChecked(True)

        self.roi_table.setRowCount(0)
        self.image_view.clear_roi_graphics()
        self._clear_channel_widgets()
        self._build_channel_widgets()
        self._adjust_threshold_slider()
        self._update_displayed_image()

    def _clear_channel_widgets(self) -> None:
        while self.channels_row.count():
            item = self.channels_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.channel_checkboxes.clear()

    def _build_channel_widgets(self) -> None:
        image_input = self.state.image_input
        if image_input is None:
            self.channels_row.addStretch(1)
            return
        for i, label in enumerate(image_input.channel_labels):
            cb = QCheckBox(f"{label} [{i + 1}]")
            cb.setChecked(self.state.channel_visibility[i])
            cb.stateChanged.connect(lambda state, idx=i: self._on_channel_toggle(idx, state))
            self.channel_checkboxes.append(cb)
            self.channels_row.addWidget(cb)
        self.channels_row.addStretch(1)

    def _on_channel_toggle(self, idx: int, state: int) -> None:
        if idx < 0 or idx >= len(self.state.channel_visibility):
            return
        self.state.channel_visibility[idx] = bool(state)
        self._update_displayed_image()

    def _adjust_threshold_slider(self) -> None:
        image_input = self.state.image_input
        if image_input is None:
            return
        global_min = float(np.min(image_input.data))
        global_max = float(np.max(image_input.data))
        self._set_threshold_domain(global_min, global_max)

        span = max(self.data_max - self.data_min, 1e-9)
        low_data = self.data_min + 0.2 * span
        high_data = self.data_min + 0.8 * span
        if self.threshold_mode == "normalized_0_1":
            low_data = max(0.0, min(1.0, low_data))
            high_data = max(0.0, min(1.0, high_data))
            if high_data - low_data < 0.1:
                high_data = min(1.0, low_data + 0.5)

        low_i, high_i = self.data_to_slider(low_data, high_data)
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue((low_i, high_i))
        self.threshold_slider.blockSignals(False)

    def _set_threshold_domain(self, data_min: float, data_max: float) -> None:
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1e-9
        if self.data_min >= 0.0 and self.data_max <= 1.0:
            self.threshold_mode = "normalized_0_1"
        else:
            self.threshold_mode = "raw_range"

    def _coerce_slider_bounds(self, low_i: int, high_i: int) -> tuple[int, int]:
        low = int(max(self._SLIDER_MIN, min(self._SLIDER_MAX, low_i)))
        high = int(max(self._SLIDER_MIN, min(self._SLIDER_MAX, high_i)))
        if low > high:
            low, high = high, low
        if low == high:
            if high < self._SLIDER_MAX:
                high += 1
            elif low > self._SLIDER_MIN:
                low -= 1
        return low, high

    def slider_to_data(self, low_i: int, high_i: int) -> tuple[float, float]:
        low_i, high_i = self._coerce_slider_bounds(low_i, high_i)
        if self.threshold_mode == "normalized_0_1":
            low = low_i / float(self._SLIDER_MAX)
            high = high_i / float(self._SLIDER_MAX)
        else:
            frac_low = low_i / float(self._SLIDER_MAX)
            frac_high = high_i / float(self._SLIDER_MAX)
            span = self.data_max - self.data_min
            low = self.data_min + frac_low * span
            high = self.data_min + frac_high * span
        if high < low:
            low, high = high, low
        return float(low), float(high)

    def data_to_slider(self, low: float, high: float) -> tuple[int, int]:
        low_f = float(low)
        high_f = float(high)
        if high_f < low_f:
            low_f, high_f = high_f, low_f
        if self.threshold_mode == "normalized_0_1":
            low_i = int(round(max(0.0, min(1.0, low_f)) * self._SLIDER_MAX))
            high_i = int(round(max(0.0, min(1.0, high_f)) * self._SLIDER_MAX))
        else:
            span = max(self.data_max - self.data_min, 1e-9)
            low_i = int(round((low_f - self.data_min) / span * self._SLIDER_MAX))
            high_i = int(round((high_f - self.data_min) / span * self._SLIDER_MAX))
        return self._coerce_slider_bounds(low_i, high_i)

    def _get_threshold_bounds(self) -> tuple[float, float]:
        low_i, high_i = self.threshold_slider.value()
        return self.slider_to_data(int(low_i), int(high_i))

    def _on_threshold_slider_changed(self, _value: tuple[int, int]) -> None:
        self._update_displayed_image()

    def _update_displayed_image(self) -> None:
        image_input = self.state.image_input
        if image_input is None:
            self.image_item.setPixmap(QPixmap())
            return
        data = image_input.data
        _, h, w = data.shape
        rgb_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        low, high = self._get_threshold_bounds()
        high = max(high, low + 1e-9)

        for i, img in enumerate(data):
            visible = i < len(self.state.channel_visibility) and self.state.channel_visibility[i]
            if not visible:
                continue
            color = np.array(self.image_colors[i % len(self.image_colors)], dtype=np.float32)
            mask = (img >= low) & (img <= high)
            norm = np.zeros_like(img, dtype=np.float32)
            if np.any(mask):
                clipped = np.clip(img.astype(np.float32), low, high)
                norm[mask] = (clipped[mask] - low) / (high - low)
            channel = (norm[..., None] * color).astype(np.uint8)
            rgb_overlay = np.clip(rgb_overlay + channel, 0, 255)

        qimg = QImage(rgb_overlay.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.image_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        self.image_scene.setSceneRect(QRectF(qimg.rect()))

    def _find_roi(self, roi_id: int) -> RPOCRoi | None:
        for roi in self.state.rois:
            if roi.roi_id == roi_id:
                return roi
        return None

    def _find_row_by_roi_id(self, roi_id: int) -> int:
        for row in range(self.roi_table.rowCount()):
            item = self.roi_table.item(row, 0)
            if item is None:
                continue
            rid = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(rid, int) and rid == roi_id:
                return row
        return -1

    def _upsert_roi_row(self, roi: RPOCRoi) -> None:
        row = self._find_row_by_roi_id(roi.roi_id)
        if row < 0:
            row = self.roi_table.rowCount()
            self.roi_table.insertRow(row)

        item = QTableWidgetItem(f"ROI {roi.roi_id}")
        item.setData(Qt.ItemDataRole.UserRole, roi.roi_id)
        self.roi_table.setItem(row, 0, item)
        self.roi_table.setItem(
            row,
            1,
            QTableWidgetItem(", ".join(f"({x:.1f}, {y:.1f})" for x, y in roi.points)),
        )
        self.roi_table.setItem(row, 2, QTableWidgetItem(f"{roi.threshold_low:.6f}"))
        self.roi_table.setItem(row, 3, QTableWidgetItem(f"{roi.threshold_high:.6f}"))
        self.roi_table.setItem(row, 4, QTableWidgetItem(f"{roi.modulation_level:.6f}"))

    def add_roi(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return
        low, high = self._get_threshold_bounds()
        roi_id = self.state.next_roi_id
        self.state.next_roi_id += 1
        roi = RPOCRoi(
            roi_id=roi_id,
            points=[(float(x), float(y)) for x, y in points],
            threshold_low=low,
            threshold_high=high,
            modulation_level=0.5,
            active_channels=self.state.channel_visibility.copy(),
        )
        self.state.rois.append(roi)
        self.image_view.add_roi_graphics(roi.roi_id, roi.points)
        self._upsert_roi_row(roi)

    def _show_table_context_menu(self, pos):
        row = self.roi_table.indexAt(pos).row()
        if row < 0:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("Delete ROI")
        action = menu.exec(self.roi_table.mapToGlobal(pos))
        if action == delete_action:
            self._delete_roi_row(row)

    def _delete_roi_row(self, row: int) -> None:
        idx_item = self.roi_table.item(row, 0)
        if idx_item is None:
            return
        roi_id = idx_item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(roi_id, int):
            return
        self.roi_table.removeRow(row)
        self.state.rois = [roi for roi in self.state.rois if roi.roi_id != roi_id]
        self.image_view.remove_roi_graphics(roi_id)

    def _toggle_mask_visibility(self, state: int) -> None:
        self.state.show_rois = bool(state)
        self.image_view.update_roi_visibility()

    def _toggle_label_visibility(self, state: int) -> None:
        self.state.show_labels = bool(state)
        self.image_view.update_roi_visibility()

    def _get_composite(self) -> np.ndarray | None:
        image_input = self.state.image_input
        if image_input is None:
            return None
        visible = [
            img for i, img in enumerate(image_input.data) if i < len(self.state.channel_visibility) and self.state.channel_visibility[i]
        ]
        if not visible:
            return None
        composite = np.mean(visible, axis=0).astype(np.float32)
        max_val = float(np.max(composite)) if composite.size else 0.0
        if max_val > 1.0:
            composite = composite / max_val
        return np.clip(composite, 0, 1)

    def _segment(self) -> None:
        comp = self._get_composite()
        if comp is None:
            return
        low, high = self._get_threshold_bounds()
        if self.threshold_mode == "raw_range":
            span = max(self.data_max - self.data_min, 1e-9)
            low = (low - self.data_min) / span
            high = (high - self.data_min) / span
        low = float(max(0.0, min(1.0, low)))
        high = float(max(0.0, min(1.0, high)))

        labels = segment(comp, "threshold + components", low=low, high=high, min_area=120)
        contours = labels_to_contours(labels, min_area=120)
        if not contours:
            return
        for contour in contours:
            points = [(float(x), float(y)) for x, y in contour]
            self.add_roi(points)

    def _sync_roi_values_from_table(self) -> None:
        for row in range(self.roi_table.rowCount()):
            idx_item = self.roi_table.item(row, 0)
            if idx_item is None:
                continue
            roi_id = idx_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(roi_id, int):
                continue
            roi = self._find_roi(roi_id)
            if roi is None:
                continue
            try:
                low_val = float(self.roi_table.item(row, 2).text())
                high_val = float(self.roi_table.item(row, 3).text())
                mod_val = float(self.roi_table.item(row, 4).text())
            except Exception:
                continue
            if high_val < low_val:
                low_val, high_val = high_val, low_val
            roi.threshold_low = float(low_val)
            roi.threshold_high = float(high_val)
            roi.modulation_level = float(np.clip(mod_val, 0.0, 1.0))

    def generate_final_mask(self) -> np.ndarray | None:
        image_input = self.state.image_input
        if image_input is None:
            return None
        self._sync_roi_values_from_table()
        _, h, w = image_input.data.shape
        final_mask = np.zeros((h, w), dtype=np.uint8)

        for roi in self.state.rois:
            pts = np.array(
                [[int(round(x)), int(round(y))] for x, y in roi.points],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            if pts.shape[0] < 3:
                continue
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [pts], 255)

            combined = np.zeros((h, w), dtype=bool)
            for img, active in zip(image_input.data, roi.active_channels):
                if not active:
                    continue
                combined |= (img >= roi.threshold_low) & (img <= roi.threshold_high)

            final_mask[combined & (roi_mask == 255)] = int(np.clip(roi.modulation_level, 0.0, 1.0) * 255)
        return final_mask

    def _preview_mask(self) -> None:
        mask = self.generate_final_mask()
        if mask is None:
            return
        h, w = mask.shape
        qimg = QImage(mask.data, w, h, w, QImage.Format.Format_Grayscale8)
        dlg = QDialog(self)
        dlg.setWindowTitle("Mask Preview")
        layout = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(QPixmap.fromImage(qimg.copy()))
        layout.addWidget(lbl)
        dlg.resize(w, h)
        dlg.exec()

    def _save_mask(self) -> None:
        mask = self.generate_final_mask()
        if mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            "",
            "PNG (*.png);;TIFF (*.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        cv2.imwrite(path, mask)

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        if img.ndim == 2:
            data = img.astype(np.float32)
        elif img.ndim == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data = np.moveaxis(rgb, -1, 0).astype(np.float32)
        else:
            return
        self.set_image_data(data)

    def create_and_close(self) -> None:
        if self._closing_via_create:
            return
        self._closing_via_create = True
        mask = self.generate_final_mask()
        if mask is not None:
            self._emitted = True
            self.mask_created.emit(mask.astype(np.uint8))
        self.close()

    def closeEvent(self, event):
        if not self._emitted and not self._closing_via_create:
            mask = self.generate_final_mask()
            if mask is not None and np.any(mask):
                self._emitted = True
                self.mask_created.emit(mask.astype(np.uint8))
        super().closeEvent(event)
