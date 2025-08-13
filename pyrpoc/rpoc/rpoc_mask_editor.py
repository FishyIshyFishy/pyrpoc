import sys
import numpy as np
from PIL import Image, ImageDraw 
import cv2 
import random
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QCheckBox, QLabel, QMenu, QGraphicsTextItem, 
    QDialog, QSpinBox, QGroupBox
)
from PyQt6.QtGui import QPixmap, QPainterPath, QPen, QBrush, QPainter, QFont, QColor, QPalette, QImage
from PyQt6.QtCore import Qt, QPointF, QRectF, QPoint, QVariant, pyqtSignal
from superqt import QRangeSlider

class ImageViewer(QGraphicsView):
    def __init__(self, scene, roi_table, params, main_window, update_mask_cb=None, update_label_cb=None):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.roi_table = roi_table
        self.params = params
        self.main_window = main_window
        self.update_mask_cb = update_mask_cb
        self.update_label_cb = update_label_cb

        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.live_path_item = None

        self._zoom = 0
        self._empty = True
        self._scene = scene

        self.drawing = False
        self.current_path = None
        self.temp_path_item = None
        self.current_points = []
        self.show_rois = True
        self.show_labels = True
        self.roi_items = []
        self.roi_label_items = []
        self.path_pen = QPen(Qt.red, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.roi_opacity = 0.4

        self.brush_radius = 3
        self.cursor_brush = QGraphicsEllipseItem(0, 0, self.brush_radius * 2, self.brush_radius * 2)
        self.cursor_brush.setBrush(QBrush(Qt.blue))
        self.cursor_brush.setPen(QPen(Qt.NoPen))
        self.cursor_brush.setZValue(1000)
        self.cursor_brush.setVisible(False)
        self.scene().addItem(self.cursor_brush)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            if not self.drawing:
                self.drawing = True
                self.current_points = []
                self.current_path = QPainterPath()
                start_pos = self.mapToScene(event.pos())
                self.current_path.moveTo(start_pos)
                self.current_points.append(start_pos)

                if self.live_path_item:
                    self.scene().removeItem(self.live_path_item)
                self.live_path_item = self.scene().addPath(self.current_path, self.path_pen)
            else:
                self.drawing = False
                end_pos = self.mapToScene(event.pos())
                self.current_points.append(end_pos)
                self.current_path.lineTo(end_pos)
                self.current_path.closeSubpath()

                self.scene().removeItem(self.live_path_item)
                self.live_path_item = None

                new_index = len(self.roi_items) + 1
                roi_item = self.scene().addPath(self.current_path)
                color = self.get_random_color()
                roi_item.setPen(QPen(color, 2))
                roi_item.setBrush(QBrush(color))
                roi_item.setOpacity(self.roi_opacity if self.show_rois else 0.0)

                self.roi_items.append(roi_item)

                roi_label = self.create_roi_label(new_index, self.current_points)
                self.roi_label_items.append(roi_label)

                self.add_roi_to_table(new_index, self.current_points)

                self.main_window.roi_channel_flags.append(self.main_window.image_visibility.copy())

                self.current_path = None
                self.current_points = []

        elif event.button() == Qt.LeftButton and not self.drawing:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.drawing:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def is_inside_any_roi(self, point):
        for roi_item in self.roi_items:
            if roi_item.contains(point):
                return True
        return False

    def find_boundary_point(self, p1, p2):
        steps = 50
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()

        last_outside = p1
        for i in range(1, steps+1):
            alpha = i / steps
            x = x1 + alpha * (x2 - x1)
            y = y1 + alpha * (y2 - y1)
            candidate = QPointF(x, y)
            if self.is_inside_any_roi(candidate):
                return last_outside
            last_outside = candidate
        return p2

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_path is not None:
            new_pos = self.mapToScene(event.pos())

            if not self.current_points:
                self.current_points.append(new_pos)
                self.current_path.lineTo(new_pos)
            else:
                last_valid = self.current_points[-1]
                if self.is_inside_any_roi(new_pos):
                    boundary = self.find_boundary_point(last_valid, new_pos)
                    self.current_points.append(boundary)
                    self.current_path.lineTo(boundary)
                else:
                    self.current_points.append(new_pos)
                    self.current_path.lineTo(new_pos)

            if self.live_path_item:
                self.live_path_item.setPath(self.current_path)

        super().mouseMoveEvent(event)

    def create_roi_label(self, roi_index, points):
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        label_text = f"{roi_index}"
        label_item = QGraphicsTextItem(label_text)
        
        font = QFont("Arial", 14, QFont.Bold)
        label_item.setFont(font)
        label_item.setDefaultTextColor(Qt.white)

        text_rect = label_item.boundingRect()
        label_item.setPos(cx - text_rect.width() / 2, cy - text_rect.height() / 2)
        label_item.setZValue(999)

        label_item.setVisible(self.show_rois and self.show_labels)
        self.scene().addItem(label_item)
        return label_item

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_M:
            self.show_rois = not self.show_rois
            if self.update_mask_cb:
                self.update_mask_cb(self.show_rois)
            self.update_roi_visibility()
        elif event.key() == Qt.Key_N:
            self.show_labels = not self.show_labels
            if self.update_label_cb:
                self.update_label_cb(self.show_labels)
            self.update_roi_visibility()
        else:
            super().keyPressEvent(event)

    def update_roi_visibility(self):
        for roi_item in self.roi_items:
            roi_item.setOpacity(self.roi_opacity if self.show_rois else 0)
        for lbl in self.roi_label_items:
            lbl.setVisible(self.show_rois and self.show_labels)

    def add_roi_to_table(self, idx, points):
        row = self.roi_table.rowCount()
        self.roi_table.insertRow(row)

        item = QTableWidgetItem(f'ROI {idx}')
        item.setData(Qt.UserRole, idx)
        self.roi_table.setItem(row, 0, item)

        coords_str = ', '.join([f'({p.x():.1f}, {p.y():.1f})' for p in points])
        self.roi_table.setItem(row, 1, QTableWidgetItem(coords_str))

        # Get current threshold values from the main window's slider (0-1000 range)
        low, high = self.main_window.threshold_slider.value()
        
        # Convert to normalized 0-1 range
        low = low / 1000.0
        high = high / 1000.0
        
        # Ensure values are in valid range
        low = max(0.0, min(1.0, low))
        high = max(0.0, min(1.0, high))
        
        # Ensure low <= high
        if low > high:
            low, high = high, low
        
        low_item = QTableWidgetItem(f"{low:.3f}")
        high_item = QTableWidgetItem(f"{high:.3f}")
        self.roi_table.setItem(row, 2, low_item)
        self.roi_table.setItem(row, 3, high_item)

        self.roi_table.setItem(row, 4, QTableWidgetItem(str(0.5)))

    def get_random_color(self):
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        return QColor(r, g, b)

class Params:
    def __init__(self):
        self.low = 80
        self.high = 200

class RPOCMaskEditor(QMainWindow):
    mask_created = pyqtSignal(object)  # emits the generated mask
    
    def __init__(self, image_data=None, channel_names=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('RPOC Mask Editor')
        self.params = Params()

        self.channel_names = channel_names if channel_names else []
        self.loaded_img = None
        self.image_layers = []
        self.image_visibility = []
        self.roi_channel_flags = []

        self.roi_table = QTableWidget(0, 5)
        self.roi_table.setHorizontalHeaderLabels(['ROI Name', 'Coordinates', 'Lower Threshold', 'Upper Threshold', 'Modulation Level'])
        self.roi_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_table.customContextMenuRequested.connect(self.show_table_context_menu)

        # top left controls
        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)

        self.mask_checkbox = QCheckBox("Mask [M]")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.stateChanged.connect(lambda state: self.toggle_mask_visibility(state == Qt.Checked))

        self.label_checkbox = QCheckBox("Labels [N]")
        self.label_checkbox.setChecked(True)
        self.label_checkbox.stateChanged.connect(lambda state: self.toggle_label_visibility(state == Qt.Checked))

        self.preview_button = QPushButton("Preview [P]")
        self.preview_button.clicked.connect(self.preview_mask)

        self.threshold_slider = QRangeSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(1000)  # Will be adjusted based on data range
        self.threshold_slider.setValue((200, 800))  # Will be adjusted based on data range
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        self.save_button = QPushButton("Save Mask")
        self.save_button.clicked.connect(self.save_mask)

        self.create_mask_button = QPushButton("Create Mask")
        self.create_mask_button.clicked.connect(self.create_and_close)

        self.cellpose_button = QPushButton("Segment with Cellpose")
        self.cellpose_button.clicked.connect(self.run_cellpose_segmentation)

        middle_controls_layout = QHBoxLayout()
        middle_controls_layout.addWidget(self.mask_checkbox)
        middle_controls_layout.addWidget(self.label_checkbox)
        middle_controls_layout.addWidget(self.preview_button)
        middle_controls_layout.addWidget(self.threshold_slider)
        middle_controls_layout.addWidget(self.save_button)
        middle_controls_layout.addWidget(self.create_mask_button)

        self.toggle_layout = QHBoxLayout()
        self.channel_checkboxes = []
        self.image_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        self.image_scene = QGraphicsScene()
        self.image_item = self.image_scene.addPixmap(QPixmap())
        self.image_view = ImageViewer(self.image_scene, self.roi_table, self.params, self)

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(self.toggle_layout)
        left_layout.addLayout(middle_controls_layout)
        left_layout.addWidget(self.image_view)
        left_layout.addWidget(self.cellpose_button)

        layout.addLayout(left_layout)
        layout.addWidget(self.roi_table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        if image_data is not None:
            self.set_image_data(image_data)
        else:
            self.update_displayed_image()

    def set_image_data(self, image_data):
        """set image data from the main application"""
        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 3:
                # multi-channel data
                for i in range(image_data.shape[0]):
                    self.image_layers.append(image_data[i])
                    self.image_visibility.append(True)
                    
                    checkbox = QCheckBox(f"Channel {i+1} [{i+1}]")
                    checkbox.setChecked(True)
                    checkbox.stateChanged.connect(lambda state, ch=i: self.on_channel_toggle(ch, state))
                    self.channel_checkboxes.append(checkbox)
                    self.toggle_layout.addWidget(checkbox)
            else:
                # single channel data
                self.image_layers.append(image_data)
                self.image_visibility.append(True)
                
                checkbox = QCheckBox("Channel 1 [1]")
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(lambda state, ch=0: self.on_channel_toggle(ch, state))
                self.channel_checkboxes.append(checkbox)
                self.toggle_layout.addWidget(checkbox)
            
            # Adjust threshold slider based on data range
            self._adjust_threshold_slider()
            
            # Update the display with the new data
            self.update_displayed_image()

    def on_channel_toggle(self, idx, state):
        self.image_visibility[idx] = bool(state)
        self.update_displayed_image()
    
    def _adjust_threshold_slider(self):
        """Adjust threshold slider range and values based on the data range"""
        if not self.image_layers:
            return
        
        # Always use 0-1 range for the slider
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(1000)  # 0-1000 for precision, maps to 0-1
        # Set default values to include the full range
        self.threshold_slider.setValue((0, 1000))  # Full range by default

    def keyPressEvent(self, event):
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1
            if 0 <= idx < len(self.image_layers):
                new_state = not self.image_visibility[idx]
                self.image_visibility[idx] = new_state
                cb = self.channel_checkboxes[idx]
                cb.blockSignals(True)
                cb.setChecked(new_state)
                cb.blockSignals(False)
                self.update_displayed_image()

        elif key == Qt.Key_M:
            new_state = not self.mask_checkbox.isChecked()
            self.mask_checkbox.blockSignals(True)
            self.mask_checkbox.setChecked(new_state)
            self.mask_checkbox.blockSignals(False)
            self.toggle_mask_visibility(new_state)

        elif key == Qt.Key_N:
            new_state = not self.label_checkbox.isChecked()
            self.label_checkbox.blockSignals(True)
            self.label_checkbox.setChecked(new_state)
            self.label_checkbox.blockSignals(False)
            self.toggle_label_visibility(new_state)

        elif key == Qt.Key_P:
            self.preview_mask()

        else:
            super().keyPressEvent(event)

    def run_cellpose_segmentation(self):
        try:
            from cellpose import models
        except ImportError:
            return

        if not self.image_layers:
            return

        self.cellpose_button.setText("Segmenting...")
        self.cellpose_button.setEnabled(False)
        QApplication.processEvents()

        visible_imgs = [
            img for img, visible in zip(self.image_layers, self.image_visibility) if visible
        ]
        if not visible_imgs:
            return

        # Normalize the composite image to [0, 255] for cellpose
        composite = np.mean(visible_imgs, axis=0)
        if composite.max() <= 1.0:
            # Data is in [0,1] range, scale to [0,255]
            composite = (composite * 255).astype(np.uint8)
        else:
            # Data is already in [0,255] range or higher, clip to [0,255]
            composite = np.clip(composite, 0, 255).astype(np.uint8)

        model = models.Cellpose(model_type='cyto3')
        masks, _, _, _ = model.eval([composite], diameter=None, channels=[0, 0])
        masks = masks[0]

        n_rois = len(self.image_view.roi_items)

        for mask_val in range(1, masks.max() + 1):
            binary_mask = np.uint8(masks == mask_val) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = contours[0]
            path = QPainterPath()
            points = []

            for i, pt in enumerate(contour):
                x, y = pt[0]
                qpt = QPointF(x, y)
                points.append(qpt)
                if i == 0:
                    path.moveTo(qpt)
                else:
                    path.lineTo(qpt)
            path.closeSubpath()

            color = self.image_view.get_random_color()
            roi_item = self.image_scene.addPath(path, QPen(color, 2), QBrush(color))
            roi_item.setOpacity(self.image_view.roi_opacity if self.image_view.show_rois else 0)
            self.image_view.roi_items.append(roi_item)

            label_item = self.image_view.create_roi_label(n_rois + 1, points)
            self.image_view.roi_label_items.append(label_item)

            self.image_view.add_roi_to_table(n_rois + 1, points)
            self.roi_channel_flags.append(self.image_visibility.copy())
            n_rois += 1

        self.cellpose_button.setText("Segment with Cellpose")
        self.cellpose_button.setEnabled(True)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            img_array = cv2.imread(file_path)
            if img_array is None:
                return
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            self.original_rgb_img = img_array
            self.loaded_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            self.update_displayed_image()

    def update_displayed_image(self):
        if not self.image_layers:
            # Clear the image if no data
            self.image_item.setPixmap(QPixmap())
            return

        height, width = self.image_layers[0].shape
        rgb_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        low, high = self.threshold_slider.value()

        # Convert slider values (0-1000) to normalized range (0-1)
        low = low / 1000.0
        high = high / 1000.0

        for i, (img, visible) in enumerate(zip(self.image_layers, self.image_visibility)):
            if not visible:
                continue
            color = np.array(self.image_colors[i % len(self.image_colors)])
            
            # Normalize image data to 0-1 range
            img_min = np.min(img)
            img_max = np.max(img)
            if img_max > img_min:
                img_normalized = (img - img_min) / (img_max - img_min)
            else:
                img_normalized = img * 0  # All zeros if no range
            
            # Apply threshold to normalized data
            mask = (img_normalized >= low) & (img_normalized <= high)
            normalized = np.zeros_like(img_normalized, dtype=np.float32)
            if np.any(mask):
                clipped = np.clip(img_normalized, low, high)
                normalized[mask] = (clipped[mask] - low) / max((high - low), 1e-9)
            channel_img = (normalized[..., None] * color).astype(np.uint8)
            rgb_overlay = np.clip(rgb_overlay + channel_img, 0, 255)

        qimage = QImage(rgb_overlay.data, width, height, 3 * width, QImage.Format_RGB888)
        self.image_item.setPixmap(QPixmap.fromImage(qimage))
        self.image_scene.setSceneRect(QRectF(qimage.rect()))

    def show_table_context_menu(self, pos):
        row = self.roi_table.indexAt(pos).row()
        if row < 0:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("Delete ROI")
        action = menu.exec_(self.roi_table.mapToGlobal(pos))
        if action == delete_action:
            self.delete_roi_row(row)

    def delete_roi_row(self, row):
        idx_item = self.roi_table.item(row, 0)
        if not idx_item:
            return
        roi_idx = idx_item.data(Qt.UserRole)
        if roi_idx is None:
            return

        self.roi_table.removeRow(row)
        viewer = self.image_view

        label_to_remove = None
        roi_to_remove = None
        remove_i = None
        for i, lbl in enumerate(viewer.roi_label_items):
            if lbl.toPlainText() == str(roi_idx):
                label_to_remove = lbl
                roi_to_remove = viewer.roi_items[i]
                remove_i = i
                break

        if remove_i is not None:
            viewer.scene().removeItem(roi_to_remove)
            viewer.scene().removeItem(label_to_remove)
            viewer.roi_items.pop(remove_i)
            viewer.roi_label_items.pop(remove_i)

        # relabel the remaining rois so they match table order
        for i, (roi, label) in enumerate(zip(viewer.roi_items, viewer.roi_label_items)):
            idx = i + 1
            label.setPlainText(str(idx))

            path = roi.path()
            poly = path.toSubpathPolygons()[0]
            cx = sum(p.x() for p in poly) / len(poly)
            cy = sum(p.y() for p in poly) / len(poly)
            text_rect = label.boundingRect()
            label.setPos(cx - text_rect.width() / 2, cy - text_rect.height() / 2)

        # rebuild table
        viewer.roi_table.setRowCount(0)
        for i, roi in enumerate(viewer.roi_items):
            path = roi.path()
            points = path.toSubpathPolygons()[0]
            viewer.add_roi_to_table(i + 1, points)

    def on_threshold_changed(self, values):
        self.update_displayed_image()
        # Store the raw slider values in params for compatibility
        self.params.low, self.params.high = values

    def toggle_mask_visibility(self, visible):
        self.image_view.show_rois = visible
        self.image_view.update_roi_visibility()

    def toggle_label_visibility(self, visible):
        self.image_view.show_labels = visible
        self.image_view.update_roi_visibility()

    def generate_final_mask(self):
        if not self.image_layers:
            return None

        height, width = self.image_layers[0].shape
        final_mask = np.zeros((height, width), dtype=np.uint8)

        for row in range(self.roi_table.rowCount()):
            idx_item = self.roi_table.item(row, 0)
            if not idx_item:
                continue

            roi_index = idx_item.data(Qt.UserRole)
            if roi_index is None or roi_index - 1 >= len(self.roi_channel_flags):
                continue

            try:
                low_val = float(self.roi_table.item(row, 2).text())
                high_val = float(self.roi_table.item(row, 3).text())
                mod_val = float(self.roi_table.item(row, 4).text())
            except (ValueError, TypeError):
                continue

            active_channels = self.roi_channel_flags[roi_index - 1]
            roi_path_item = self.image_view.roi_items[roi_index - 1]
            polygons = roi_path_item.path().toSubpathPolygons()
            if not polygons:
                continue

            polygon = polygons[0]
            pts = np.array([[int(p.x()), int(p.y())] for p in polygon], dtype=np.int32).reshape(-1, 1, 2)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [pts], 255)

            combined_mask = np.zeros((height, width), dtype=bool)

            for i, (img, active) in enumerate(zip(self.image_layers, active_channels)):
                if not active:
                    continue
                
                # Normalize image data to 0-1 range
                img_min = np.min(img)
                img_max = np.max(img)
                if img_max > img_min:
                    img_normalized = (img - img_min) / (img_max - img_min)
                else:
                    img_normalized = img * 0  # All zeros if no range
                
                # Apply threshold to normalized data (threshold values are already 0-1)
                valid_range = (img_normalized >= low_val) & (img_normalized <= high_val)
                combined_mask |= valid_range

            valid_pixels = combined_mask & (roi_mask == 255)
            final_mask[valid_pixels] = int(mod_val * 255)

        return final_mask

    def preview_mask(self):
        mask = self.generate_final_mask()
        if mask is None:
            return

        height, width = mask.shape
        bytes_per_line = width
        qimg = QImage(mask.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        dialog = QDialog(self)
        dialog.setWindowTitle("Mask Preview")
        layout = QVBoxLayout()
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.resize(width, height)
        dialog.exec_()

    def save_mask(self):
        mask = self.generate_final_mask()
        if mask is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Mask', '', 'PNG (*.png);;TIFF (*.tif);;All Files (*)'
        )
        if not save_path:
            return

        cv2.imwrite(save_path, mask)

    def create_and_close(self):
        """create the mask and emit it, then close the window"""
        mask = self.generate_final_mask()
        if mask is not None:
            self.mask_created.emit(mask)
        self.close() 