import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem, QGridLayout
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont
from .base_display import BaseImageDisplayWidget

class ImageDisplayWidget(BaseImageDisplayWidget):
    def __init__(self, app_state, signals):
        super().__init__(app_state, signals)
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setStyleSheet('''
            QGraphicsView {
                border: 2px dashed #cccccc;
                background-color: #f0f0f0;
            }
        ''')
        layout.addWidget(self.graphics_view)

        frame_controls_layout = QHBoxLayout()
        self.frame_label = QLabel('Frame: 0/0')
        frame_controls_layout.addWidget(self.frame_label)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        frame_controls_layout.addWidget(self.frame_slider)
        layout.addLayout(frame_controls_layout)

        self._dragging_endpoint = None  # (line_index, endpoint_idx, offset_x, offset_y)
        self._add_mode = False
        self._temp_line_start = None  # (x, y) for temporary line during creation
        self._lines = []  # [(x1, y1, x2, y2, color)]

        self.graphics_view.viewport().installEventFilter(self)

    def connect_lines_widget(self, lines_widget):
        super().connect_lines_widget(lines_widget)

    def eventFilter(self, obj, event):
        if obj is self.graphics_view.viewport():
            if event.type() == event.Type.MouseButtonPress:
                return self.handle_mouse_press(event)
            elif event.type() == event.Type.MouseMove:
                return self.handle_mouse_move(event)
            elif event.type() == event.Type.MouseButtonRelease:
                return self.handle_mouse_release(event)
        return super().eventFilter(obj, event)

    def handle_mouse_press(self, event):
        pos = event.position() if hasattr(event, 'position') else event.pos()
        scene_pos = self.graphics_view.mapToScene(int(pos.x()), int(pos.y()))
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        if self._add_mode:
            if self._temp_line_start is None:
                # First click - start the line
                self._temp_line_start = (x, y)
                return True
            else:
                # Second click - complete the line
                x1, y1 = self._temp_line_start
                # Send all channel data for proper trace plotting
                all_channel_data = self.get_all_channel_data()
                self.line_add_requested.emit(x1, y1, x, y, all_channel_data, self.get_channel_names())
                self._add_mode = False
                self._temp_line_start = None
                self.update_display()
                return True
        
        # check if clicking on an existing line endpoint for dragging
        for idx, (x1, y1, x2, y2, color) in enumerate(self.get_lines()):
            # Check first endpoint
            if abs(x - x1) < 8 and abs(y - y1) < 8:
                self._dragging_endpoint = (idx, 0, x - x1, y - y1)
                return True
            # Check second endpoint
            if abs(x - x2) < 8 and abs(y - y2) < 8:
                self._dragging_endpoint = (idx, 1, x - x2, y - y2)
                return True
        return False

    def handle_mouse_move(self, event):
        """Handle mouse move for line drawing"""
        if self._add_mode and self._temp_line_start is not None:
            # Force a redraw to show the temporary line
            self.update_display()
            return True
        return False

    def handle_mouse_release(self, event):
        if self._dragging_endpoint is not None:
            self._dragging_endpoint = None
            return True
        return False

    @pyqtSlot()
    def enter_add_mode(self):
        print(f"ImageDisplayWidget.enter_add_mode called")
        self._add_mode = True
        self._temp_line_start = None
        print(f"ImageDisplayWidget._add_mode set to {self._add_mode}")

    @pyqtSlot(int)
    def remove_line_overlay(self, index):
        '''
        remove a given line 

        index: Index of the line to remove
        '''
        self.line_remove_requested.emit(index)
        self.update_display()

    @pyqtSlot(int, int, int, int, object)
    def move_line_endpoint(self, line_index, endpoint_idx, x, y, image_data):
        self.line_endpoint_move_requested.emit(line_index, endpoint_idx, x, y, image_data)
        self.update_display()

    def handle_frame_acquired(self, data_unit, idx, total):
        if self._acq_buffer is None or self._acq_total != total:
            # allocate buffer for the expected number of data units
            if isinstance(data_unit, np.ndarray):
                shape = (total,) + data_unit.shape
                self._acq_buffer = np.zeros(shape, dtype=data_unit.dtype)
            else:
                self._acq_buffer = [None] * total
            self._acq_total = total
        # store data unit
        if isinstance(self._acq_buffer, np.ndarray):
            self._acq_buffer[idx] = data_unit
        else:
            self._acq_buffer[idx] = data_unit
        self.total_frames = total
        self.current_frame = idx
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.frame_slider.setValue(idx)
        self.update_frame_label()
        self.display_frame(idx)
        self.update_overlays()
        self.update_display()

    def handle_data_updated(self, data):
        '''
        replace buffer with full data
        '''

        # Store the data in the buffer instead of setting it to None
        self._acq_buffer = data
        self._acq_total = None
        self.app_state.current_data = data
       
        if isinstance(data, np.ndarray) and data.ndim == 3:
            self.total_frames = data.shape[0]
            self.current_frame = 0
            self.frame_slider.setMaximum(max(0, self.total_frames - 1))
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(self.total_frames > 1)
            self.update_frame_label()
            self.display_frame(self.current_frame)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self.total_frames = 1
            self.current_frame = 0
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(False)
            self.update_frame_label()
            self.display_frame(self.current_frame)
        else:
            self.graphics_scene.clear()
            self.total_frames = 0
            self.current_frame = 0
            self.update_frame_label()
        self.update_display()
        self.traces_update_requested.emit(self.get_all_channel_data())

    def update_frame_label(self):
        self.frame_label.setText(f'Frame: {self.current_frame + 1}/{self.total_frames}')

    def on_frame_slider_changed(self, value):
        self.current_frame = value
        self.update_frame_label()
        self.display_frame(self.current_frame)
        self.update_overlays()
        self.update_display()
        self.traces_update_requested.emit(self.get_all_channel_data())

    def display_frame(self, frame_idx):
        data = self.app_state.current_data
        # if acquisition is ongoing, use buffer
        if self._acq_buffer is not None:
            data = self._acq_buffer
        if data is None:
            self.graphics_scene.clear()
            return
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                if 0 <= frame_idx < data.shape[0]:
                    frame_data = data[frame_idx]
                else:
                    self.graphics_scene.clear()
                    return
            elif data.ndim == 2:
                frame_data = data
            else:
                self.graphics_scene.clear()
                return
 
            height, width = frame_data.shape
            data_norm = ((frame_data - frame_data.min()) / (frame_data.max() - frame_data.min() + 1e-9) * 255).astype(np.uint8)
            qimage = QImage(data_norm.data, width, height, width, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            self.graphics_scene.clear()
            self.graphics_scene.addPixmap(pixmap)
            scene_rect = QRectF(pixmap.rect())
            self.graphics_scene.setSceneRect(scene_rect)
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.graphics_scene.clear()
        self.update_display()

    def update_display(self):
        frame = self.get_current_frame_data()
        if frame is None:
            self.graphics_scene.clear()
            return
        height, width = frame.shape
        data_norm = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-9) * 255).astype(np.uint8)
        qimage = QImage(data_norm.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)
        scene_rect = QRectF(pixmap.rect())
        self.graphics_scene.setSceneRect(scene_rect)
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # draw lines overlays
        for idx, (x1, y1, x2, y2, color) in enumerate(self.get_lines()):
            # Draw the line
            pen = QPen(QColor(color), 2)
            self.graphics_scene.addLine(x1, y1, x2, y2, pen)
            
            # Draw endpoints as circles
            endpoint_pen = QPen(QColor(color), 1)
            endpoint_brush = QBrush(QColor(color))
            self.graphics_scene.addEllipse(x1-3, y1-3, 6, 6, endpoint_pen, endpoint_brush)
            self.graphics_scene.addEllipse(x2-3, y2-3, 6, 6, endpoint_pen, endpoint_brush)
            
            # Add line label at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            text = QGraphicsSimpleTextItem(f"L{idx+1}")
            text.setBrush(QBrush(QColor(color)))
            text.setPos(mid_x + 5, mid_y - 10)
            self.graphics_scene.addItem(text)
        
        # Draw temporary line during creation
        if self._add_mode and self._temp_line_start is not None:
            # Get current mouse position for temporary line
            cursor_pos = self.graphics_view.mapFromGlobal(self.graphics_view.cursor().pos())
            scene_pos = self.graphics_view.mapToScene(cursor_pos)
            temp_x, temp_y = int(scene_pos.x()), int(scene_pos.y())
            x1, y1 = self._temp_line_start
            
            # Draw temporary line
            temp_pen = QPen(QColor('#FF6B6B'), 2, Qt.PenStyle.DashLine)
            self.graphics_scene.addLine(x1, y1, temp_x, temp_y, temp_pen)
            
            # Draw temporary endpoint
            temp_endpoint_pen = QPen(QColor('#FF6B6B'), 1)
            temp_endpoint_brush = QBrush(QColor('#FF6B6B'))
            self.graphics_scene.addEllipse(temp_x-3, temp_y-3, 6, 6, temp_endpoint_pen, temp_endpoint_brush)

    def get_current_frame_index(self):
        return self.current_frame

    def get_current_frame_data(self):
        if self._acq_buffer is None:
            return None
        
        if self._acq_buffer.ndim == 3:
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame]
            else:
                return None
        else:
            return self._acq_buffer
    
    def get_image_data_for_rpoc(self):
        '''
        get image data in a format suitable for RPOC mask creation
        '''
        if self._acq_buffer is None:
            return None
            
        # if we have 3D data (frames x height x width), return the current frame as a single channel
        if self._acq_buffer.ndim == 3:
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame][np.newaxis, :, :]
            else:
                return None
        # if we have 2D data, return it as a single channel
        elif self._acq_buffer.ndim == 2:
            return self._acq_buffer[np.newaxis, :, :]
        else:
            return None

    def get_lines(self):
        if hasattr(self, '_lines_widget'):
            return self._lines_widget.get_lines()
        return self._lines

    def get_all_channel_data(self):
        """Get all channel data for the current frame - for confocal mode"""
        if self._acq_buffer is None:
            return None
        
        if self._acq_buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame]  # Return all channels
            else:
                return None
        elif self._acq_buffer.ndim == 3:  # frames x height x width (single channel)
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame][np.newaxis, :, :]  # Add channel dimension
            else:
                return None
        else:
            return self._acq_buffer[np.newaxis, :, :]  # Add channel dimension