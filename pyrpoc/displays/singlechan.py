import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem, QGridLayout
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot, QTimer
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
        
        # Connect resize event to ensure proper sizing
        self.graphics_view.resizeEvent = self.on_graphics_view_resize
        
        # Timer for delayed viewport updates to fix sizing issues
        # Initialize frame counter
        self._current_frame_idx = 0
        
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.delayed_viewport_update)

    def handle_data_frame_received(self, data):
        """
        New method for handling individual data frames during acquisition.
        This is part of the uniform acquisition pipeline.
        
        data: The data frame that was just received
        """
        if self.acq_buffer is None or self.acq_total != self.total_frames:
            # allocate buffer for the expected number of data units
            if isinstance(data, np.ndarray):
                shape = (self.total_frames,) + data.shape
                self.acq_buffer = np.zeros(shape, dtype=data.dtype)
            else:
                self.acq_buffer = [None] * self.total_frames
            self.acq_total = self.total_frames
        
        # store data unit
        if isinstance(self.acq_buffer, np.ndarray):
            # grow buffer if needed
            if self._current_frame_idx >= self.acq_buffer.shape[0]:
                new_len = max(self._current_frame_idx + 1, self.acq_buffer.shape[0] * 2 if self.acq_buffer.shape[0] > 0 else 1)
                new_shape = (new_len,) + self.acq_buffer.shape[1:]
                new_buf = np.zeros(new_shape, dtype=self.acq_buffer.dtype)
                new_buf[:self.acq_buffer.shape[0]] = self.acq_buffer
                self.acq_buffer = new_buf
            self.acq_buffer[self._current_frame_idx] = data
        else:
            self.acq_buffer[self._current_frame_idx] = data
        
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.frame_slider.setValue(self._current_frame_idx)
        self.update_frame_label()
        self.display_frame(self._current_frame_idx)
        
        # Schedule delayed viewport update on first frame to fix sizing issues
        if self._current_frame_idx == 0:
            self.resize_timer.start(100)  # 100ms delay
        
        self.update_overlays()
        self.update_display()
        
        # Increment frame counter
        self._current_frame_idx += 1

    def prepare_for_acquisition(self, context_or_total_frames):
        """
        Prepare the display widget for acquisition.
        Accepts either an AcquisitionContext or a total_frames integer for backward compatibility.
        """
        # Determine total frames and reset optional context
        if isinstance(context_or_total_frames, int):
            total_frames = context_or_total_frames
            self.acquisition_context = None
        else:
            ctx = context_or_total_frames
            total_frames = int(getattr(ctx, 'total_frames', 0) or 0)
            self.acquisition_context = ctx
            # store shape info if provided
            self.frame_shape = getattr(ctx, 'frame_shape', None)
            self.channel_info = getattr(ctx, 'channel_info', {})

        # Reset frame counter for new acquisition
        self._current_frame_idx = 0
        
        # Update internal state
        self.total_frames = max(1, int(total_frames))
        self.current_frame = 0
        
        # Clear acquisition buffer
        self.acq_buffer = None
        self.acq_total = self.total_frames
        
        # Reset frame controls
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.update_frame_label()
        
        # Clear the graphics scene
        self.graphics_scene.clear()
        
        # Update frame label
        self.frame_label.setText(f'Frame: 1/{self.total_frames}')
        
        # Schedule delayed viewport update to fix sizing issues
        self.resize_timer.start(100)  # 100ms delay

    def handle_data_updated(self, data):
        '''
        replace buffer with full data
        '''

        # Store the data in the buffer instead of setting it to None
        self.acq_buffer = data
        self.acq_total = None
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
        
        # Schedule delayed viewport update to fix sizing issues
        self.resize_timer.start(100)  # 100ms delay
        
        self.update_display()

    def update_frame_label(self):
        self.frame_label.setText(f'Frame: {self.current_frame + 1}/{self.total_frames}')

    def on_frame_slider_changed(self, value):
        self.current_frame = value
        self.update_frame_label()
        self.display_frame(self.current_frame)
        self.update_overlays()
        self.update_display()

    def display_frame(self, frame_idx):
        data = self.app_state.current_data
        # if acquisition is ongoing, use buffer
        if self.acq_buffer is not None:
            data = self.acq_buffer
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
        

    def get_current_frame_index(self):
        return self.current_frame

    def get_current_frame_data(self):
        if self.acq_buffer is None:
            return None
        
        if self.acq_buffer.ndim == 3:
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame]
            else:
                return None
        else:
            return self.acq_buffer
    
    def get_image_data_for_rpoc(self):
        '''
        get image data in a format suitable for RPOC mask creation
        '''
        if self.acq_buffer is None:
            return None
            
        # if we have 3D data (frames x height x width), return the current frame as a single channel
        if self.acq_buffer.ndim == 3:
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame][np.newaxis, :, :]
            else:
                return None
        # if we have 2D data, return it as a single channel
        elif self.acq_buffer.ndim == 2:
            return self.acq_buffer[np.newaxis, :, :]
        else:
            return None

    def get_all_channel_data(self):
        """Get all channel data for the current frame - for confocal mode"""
        if self.acq_buffer is None:
            return None
        
        if self.acq_buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame]  # Return all channels
            else:
                return None
        elif self.acq_buffer.ndim == 3:  # frames x height x width (single channel)
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame][np.newaxis, :, :]  # Add channel dimension
            else:
                return None
        else:
            return self.acq_buffer[np.newaxis, :, :]  # Add channel dimension
    
    def on_graphics_view_resize(self, event):
        """Handle graphics view resize to maintain proper sizing"""
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def delayed_viewport_update(self):
        """Delayed viewport update to fix sizing issues after GUI updates"""
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)