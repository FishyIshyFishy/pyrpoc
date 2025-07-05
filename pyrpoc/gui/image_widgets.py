import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont

class BaseImageDisplayWidget(QWidget):
    '''
    defines the interface that all image display widgets must implement to work with the acquisition system and dockable widgets

    some image widgets will need to say no to the widgets, like raise a statusbar update (lines incompatible with zscan)
    '''
    
    # all widgets must deal with these signals
    line_add_requested = pyqtSignal(int, int, int, int, object)  # x1, y1, x2, y2, image_data
    line_endpoint_move_requested = pyqtSignal(int, int, int, int, object)  # line_index, endpoint_idx, x, y, image_data
    line_remove_requested = pyqtSignal(int)  # index
    traces_update_requested = pyqtSignal(object)  # image_data
    
    def __init__(self, app_state, signals):
        super().__init__()
        self.app_state = app_state
        self.signals = signals
        self.current_frame = 0
        self.total_frames = 0
        
        # Buffer for incoming data units during acquisition
        self._acq_buffer = None
        self._acq_total = None
        
        # Overlay callbacks for future extensibility
        self.overlay_callbacks = []
        
        self.signals.data_updated.connect(self.handle_data_updated)
    
    def handle_frame_acquired(self, data_unit, idx, total):
        '''
        called as new data units are acquired during acquisition
        
    
        data_unit: The data unit (e.g., single frame) that was just acquired
        idx: Index of the acquired data unit
        total: Total number of expected data units
        '''
        raise NotImplementedError("Subclasses must implement handle_frame_acquired")
    
    def handle_data_updated(self, data):
        '''
        calls when acquisition is complete and full data is available.

        data: complete dataset (could be 2D or 3D numpy array)
        '''
        raise NotImplementedError("Subclasses must implement handle_data_updated")
    
    def get_current_frame_data(self):
        '''
        get the data for the current displayed unit of data
        
        returns: numpy.ndarray or None: current frame data, or None if no data available
        '''
        raise NotImplementedError("Subclasses must implement get_current_frame_data")
    
    def get_image_data_for_rpoc(self):
        '''
        get image data in a format suitable for RPOC mask creation
        
        returns: numpy.ndarray or None: array of 2D images (channels x height x width) or None if no data
        '''
        current_data = self.get_current_frame_data()
        if current_data is None:
            return None
            
        # if it's already 3D (channels x height x width), return as is
        if current_data.ndim == 3:
            return current_data
        # if it's 2D, add a channel dimension
        elif current_data.ndim == 2:
            return current_data[np.newaxis, :, :]
        else:
            return None
    
    def get_current_frame_index(self):
        '''
        get the current displayed unit of data index.
        
        returns: int: Current frame index
        '''
        raise NotImplementedError("Subclasses must implement get_current_frame_index")
    
    def update_display(self):
        '''
        update the visual display with current data and overlays.
        '''
        raise NotImplementedError("Subclasses must implement update_display")
    
    def display_frame(self, frame_idx):
        '''
        display a specific frame.
        
        frame_idx: Index of the frame to display
        '''
        raise NotImplementedError("Subclasses must implement display_frame")
    
    def connect_lines_widget(self, lines_widget):
        '''
        connect signals between this widget and a lines widget.
        
        lines_widget: The lines widget to connect to
        '''
        # lines --> image display
        lines_widget.add_mode_requested.connect(self.enter_add_mode)
        lines_widget.remove_line_requested.connect(self.remove_line_overlay)
        lines_widget.line_endpoint_moved.connect(self.move_line_endpoint)
        
        # image display --> lines
        self.line_add_requested.connect(lines_widget.add_line)
        self.line_endpoint_move_requested.connect(lines_widget.update_line_endpoint)
        self.line_remove_requested.connect(lines_widget.remove_line)
        self.traces_update_requested.connect(lines_widget.update_all_traces)
        
        # for drawing overlays
        self._lines_widget = lines_widget
    
    def enter_add_mode(self):
        '''
        enter interactive tool mode with clicks and stuff
        '''
        raise NotImplementedError("Subclasses must implement enter_add_mode")
    
    def remove_line_overlay(self, index):
        '''
        remove a given line 

        index: Index of the line to remove
        '''
        raise NotImplementedError("Subclasses must implement remove_line_overlay")
    
    def move_line_endpoint(self, line_index, endpoint_idx, x, y, image_data):
        '''
        move a line endpoint
        
        line_index: Index of the line
        endpoint_idx: 0 for first endpoint, 1 for second endpoint
        x, y: New coordinates
        image_data: Current image data
        '''
        raise NotImplementedError("Subclasses must implement move_line_endpoint")
    
    def get_lines(self):
        '''
        get current line positions and colors.
        
        returns a list of (x1, y1, x2, y2, color) tuples for each line
        '''
        raise NotImplementedError("Subclasses must implement get_lines")
    
    def register_overlay_callback(self, callback):
        '''
        holds the callback function to be called when overlays need updating.
        
        callback: Function to call with current frame index
        '''
        self.overlay_callbacks.append(callback)
    
    def update_overlays(self):
        '''
        call all the overlay callbacks (may be more than 1)
        '''
        for cb in self.overlay_callbacks:
            cb(self.current_frame)
    
    def update_frame_controls(self, total_frames, current_frame):
        '''
        Update frame navigation controls.
        
        Args:
            total_frames: Total number of frames
            current_frame: Current frame index
        '''
        self.total_frames = total_frames
        self.current_frame = current_frame


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
                self.line_add_requested.emit(x1, y1, x, y, self.get_current_frame_data())
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
        if self._dragging_endpoint is not None:
            pos = event.position() if hasattr(event, 'position') else event.pos()
            scene_pos = self.graphics_view.mapToScene(int(pos.x()), int(pos.y()))
            x, y = int(scene_pos.x()), int(scene_pos.y())
            line_idx, endpoint_idx, offset_x, offset_y = self._dragging_endpoint
            new_x = x - offset_x
            new_y = y - offset_y
            frame = self.get_current_frame_data()
            if frame is not None:
                h, w = frame.shape
                new_x = max(0, min(new_x, w - 1))
                new_y = max(0, min(new_y, h - 1))
            self.line_endpoint_move_requested.emit(line_idx, endpoint_idx, new_x, new_y, self.get_current_frame_data())
            self.update_display()
            return True
        elif self._add_mode and self._temp_line_start is not None:
            # Update temporary line display during creation
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
        self._add_mode = True
        self._temp_line_start = None

    @pyqtSlot(int)
    def remove_line_overlay(self, index):
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

        self._acq_buffer = None
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
        self.traces_update_requested.emit(self.get_current_frame_data())

    def update_frame_label(self):
        self.frame_label.setText(f'Frame: {self.current_frame + 1}/{self.total_frames}')

    def on_frame_slider_changed(self, value):
        self.current_frame = value
        self.update_frame_label()
        self.display_frame(self.current_frame)
        self.update_overlays()
        self.update_display()
        self.traces_update_requested.emit(self.get_current_frame_data())

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
