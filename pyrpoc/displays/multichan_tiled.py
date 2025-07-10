import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                             QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem, 
                             QGridLayout, QSpinBox, QFrame, QSizePolicy, QCheckBox, QGroupBox, QDoubleSpinBox)
from superqt import QRangeSlider
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from .base_display import BaseImageDisplayWidget
import math

class MultichannelImageDisplayWidget(BaseImageDisplayWidget):
    def __init__(self, app_state, signals):
        super().__init__(app_state, signals)
        self.num_channels = 1
        self.channel_names = ['Channel 1']
        self.channel_scenes = []
        self.channel_views = []
        
        # Display parameters for intensity control
        self.intensity_params = {}  # {channel_idx: {'min': float, 'max': float, 'auto': bool}}
        
        self.add_mode = False
        self.temp_line_start = None
        self.dragging_endpoint = None

        self.lines_widget = None
        
        self.setup_ui()
        self.update_channel_names()
    
    def get_display_parameters(self):
        """Get current display parameters for GUI coupling"""
        return {
            'intensity_params': self.intensity_params.copy(),
            'num_channels': self.num_channels,
            'channel_names': self.channel_names.copy()
        }
    
    def set_display_parameters(self, params):
        """Set display parameters from GUI"""
        if 'intensity_params' in params:
            self.intensity_params = params['intensity_params'].copy()
            # Update colorbars for all channels
            for ch in range(self.num_channels):
                self.update_colorbar(ch)
            self.update_display()
    

    

    
    def create_colorbar(self, channel_idx):
        """Create a colorbar widget for a channel"""
        widget = QWidget()
        widget.setFixedWidth(30)
        widget.setMinimumHeight(150)
        widget.setMaximumHeight(300)
        widget.colorbar_pixmap = None
        widget.channel_idx = channel_idx
        
        # Override paintEvent to draw the colorbar
        def paintEvent(event):
            if widget.colorbar_pixmap:
                painter = QPainter(widget)
                painter.drawPixmap(0, 0, widget.colorbar_pixmap)
            else:
                # Draw a default colorbar if no pixmap
                painter = QPainter(widget)
                painter.fillRect(0, 0, widget.width(), widget.height(), QColor(240, 240, 240))
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawRect(0, 0, widget.width()-1, widget.height()-1)
                painter.drawText(5, widget.height() - 5, "No data")
        
        widget.paintEvent = paintEvent
        return widget
    
    def update_colorbar(self, channel_idx):
        """Update the colorbar for a channel"""
        if channel_idx >= len(self.channel_views):
            return
            
        # Find the colorbar widget by looking at the channel layout structure
        # Each channel has 3 rows: label, image+colorbar, controls
        # The colorbar is in the second row, first item of the image container
        row = channel_idx // math.ceil(math.sqrt(self.num_channels))
        col = channel_idx % math.ceil(math.sqrt(self.num_channels))
        start_row = row * 3
        
        # Get the image container (second row)
        image_container_item = self.channel_layout.itemAtPosition(start_row + 1, col)
        if not image_container_item or not image_container_item.widget():
            return
            
        image_container = image_container_item.widget()
        if not hasattr(image_container, 'layout'):
            return
            
        layout = image_container.layout()
        if layout.count() < 2:
            return
            
        # Colorbar is the first item in the image container layout
        colorbar_item = layout.itemAt(0)
        if not colorbar_item or not colorbar_item.widget():
            return
            
        colorbar_widget = colorbar_item.widget()
        
        # Create colorbar pixmap
        width = 30
        height = colorbar_widget.height()
        if height <= 0:
            height = 100
            
        pixmap = QPixmap(width, height)
        painter = QPainter(pixmap)
        
        # Get current intensity parameters
        params = self.intensity_params.get(channel_idx, {'min': 0.0, 'max': 1.0, 'auto': True})
        
        # Create gradient
        gradient = QLinearGradient(0, height, 0, 0)  # Bottom to top
        gradient.setColorAt(0.0, QColor(0, 0, 0))    # Black at bottom
        gradient.setColorAt(1.0, QColor(255, 255, 255))  # White at top
        
        # Fill with gradient
        painter.fillRect(0, 0, width, height, gradient)
        
        # Draw border
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawRect(0, 0, width-1, height-1)
        
        # Draw min/max labels
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 8))
        
        if params['auto']:
            # Show "Auto" label
            painter.drawText(5, height - 5, "Auto")
            painter.drawText(5, 15, "Auto")
        else:
            # Show actual min/max values
            min_val = params['min']
            max_val = params['max']
            painter.drawText(5, height - 5, f"{min_val:.3f}")
            painter.drawText(5, 15, f"{max_val:.3f}")
        
        painter.end()
        
        # Store the pixmap
        colorbar_widget.colorbar_pixmap = pixmap
        colorbar_widget.update()
    

    
    def update_channel_names(self):
        modality = self.app_state.modality.lower()
        channel_names = self.get_channel_names_from_instruments()
        if channel_names and len(channel_names) >= self.num_channels:
            self.channel_names = channel_names[:self.num_channels]
        else:
            if modality == 'confocal':
                self.channel_names = [f'Channel {i+1}' for i in range(self.num_channels)]
            elif modality == 'split data stream':
                # Use real channel names from data input instrument
                input_names = channel_names if channel_names else [f'Input {i+1}' for i in range(self.num_channels // 2)]
                self.channel_names = []
                for input_ch, base_name in enumerate(input_names):
                    self.channel_names.extend([
                        f'{base_name} - First Portion',
                        f'{base_name} - Second Portion',
                    ])
                    
                remaining = self.num_channels - len(self.channel_names)
                if remaining > 0:
                    for i in range(remaining):
                        self.channel_names.append(f'Channel {len(self.channel_names) + 1}')
            else:
                self.channel_names = [f'Channel {i+1}' for i in range(self.num_channels)]
    
    def get_channel_names_from_instruments(self):
        channel_names = []
        
        if hasattr(self.app_state, 'instruments'):
            for instrument in self.app_state.instruments:
                if instrument.instrument_type == "data input" and hasattr(instrument, 'parameters'):
                    # Get channel names from the instrument parameters
                    input_channels = instrument.parameters.get('input_channels', [])
                    channel_names_param = instrument.parameters.get('channel_names', {})
                    
                    for ch in input_channels:
                        # Use the channel name if available, otherwise use default
                        channel_name = channel_names_param.get(str(ch), f"ch{ch}")
                        channel_names.append(channel_name)
        
        return channel_names

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Frame controls
        frame_controls = QHBoxLayout()
        frame_controls.addWidget(QLabel('Frame:'))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        frame_controls.addWidget(self.frame_slider)
        
        self.frame_label = QLabel('Frame: 1/1')
        frame_controls.addWidget(self.frame_label)
        layout.addLayout(frame_controls)
        

        
        # Channel grid display
        self.channel_grid = QWidget()
        self.channel_layout = QGridLayout()
        self.channel_grid.setLayout(self.channel_layout)
        layout.addWidget(self.channel_grid)
        
        # Graphics views for each channel
        self.channel_views = []
        self.channel_scenes = []
        
        # Initialize with single channel
        self.setup_channel_display(1)
        
        # Timer for delayed viewport updates to fix sizing issues
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.delayed_viewport_update)
    
    def setup_channel_display(self, num_channels):
        # Clean up existing views and controls
        for view in self.channel_views:
            view.deleteLater()
        self.channel_views.clear()
        self.channel_scenes.clear()
        

        
        # Clean up channel layout
        while self.channel_layout.count():
            child = self.channel_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        

        
        self.update_channel_names()
        
        # Square grid layout
        cols = math.ceil(math.sqrt(num_channels))
        rows = math.ceil(num_channels / cols)
        
        for i in range(num_channels):
            # Create channel display
            view = QGraphicsView()
            scene = QGraphicsScene()
            view.setScene(scene)
            view.setStyleSheet('''
                QGraphicsView {
                    border: 2px dashed #cccccc;
                    background-color: #f0f0f0;
                }
            ''')
            
            channel_name = self.channel_names[i] if i < len(self.channel_names) else f'Channel {i+1}'
            label = QLabel(channel_name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('font-weight: bold; color: #333;')
            

            
            # Create colorbar
            colorbar_widget = self.create_colorbar(i)
            
            # Layout: 3 rows per channel (label, image+colorbar, controls)
            row = i // cols
            col = i % cols
            start_row = row * 3
            
            self.channel_layout.addWidget(label, start_row, col)
            
            # Create horizontal layout for image and colorbar
            image_colorbar_layout = QHBoxLayout()
            image_colorbar_layout.addWidget(view, 1)
            image_colorbar_layout.addWidget(colorbar_widget, 0)
            image_colorbar_layout.setStretch(0, 1)  # colorbar gets less space
            image_colorbar_layout.setStretch(1, 4)  # image gets more space
            
            # Create container for image+colorbar
            image_container = QWidget()
            image_container.setLayout(image_colorbar_layout)
            self.channel_layout.addWidget(image_container, start_row + 1, col)
            

            
            self.channel_views.append(view)
            self.channel_scenes.append(scene)
            view.viewport().installEventFilter(self)
            view.resizeEvent = lambda event, v=view: self.on_channel_view_resize(v, event)
    
    def handle_frame_acquired(self, data_unit, idx, total):
        """Handle frame acquired signal for confocal multi-channel data"""
        # Reset buffer if this is the first frame of a new acquisition
        if idx == 0:
            self.acq_buffer = None
        
        # Store the acquired frame in the buffer
        if self.acq_buffer is None:
            # Initialize buffer based on data shape
            if isinstance(data_unit, np.ndarray):
                if data_unit.ndim == 3:  # channels x height x width
                    self.acq_buffer = np.zeros((total, data_unit.shape[0], data_unit.shape[1], data_unit.shape[2]))
                    self.num_channels = data_unit.shape[0]
                    # Update channel names and display
                    self.update_channel_names()
                    self.setup_channel_display(self.num_channels)
                elif data_unit.ndim == 2:  # height x width (single channel)
                    self.acq_buffer = np.zeros((total, 1, data_unit.shape[0], data_unit.shape[1]))
                    self.num_channels = 1
                    self.update_channel_names()
                else:
                    self.acq_buffer = np.zeros((total, 1, data_unit.shape[0], data_unit.shape[1]))
                    self.num_channels = 1
                    self.update_channel_names()
            else:
                # Fallback for non-array data
                self.acq_buffer = np.zeros((total, 1, 512, 512))
                self.num_channels = 1
                self.update_channel_names()
        
        # Store the frame data
        if isinstance(data_unit, np.ndarray):
            if data_unit.ndim == 3:  # channels x height x width
                if idx < self.acq_buffer.shape[0]:  # Safety check
                    self.acq_buffer[idx] = data_unit
            elif data_unit.ndim == 2:  # height x width (single channel)
                if idx < self.acq_buffer.shape[0]:  # Safety check
                    self.acq_buffer[idx, 0] = data_unit
            else:
                if idx < self.acq_buffer.shape[0]:  # Safety check
                    self.acq_buffer[idx, 0] = data_unit
        
        # Update current frame and frame controls
        self.current_frame = idx
        self.total_frames = total
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.frame_slider.setValue(idx)
        self.update_frame_label()
        
        # Update display if this is the current frame
        if idx == self.current_frame:
            self.display_frame(idx)
        
        # Update frame controls if this is the first frame
        if idx == 0:
            self.update_frame_controls(total, 0)
        
        # Schedule delayed viewport update on first frame to fix sizing issues
        if idx == 0:
            self.resize_timer.start(100)  # 100ms delay
        
        # Update overlays and display consistently with ImageDisplayWidget
        self.update_overlays()
        self.update_display()
        
        # Emit traces update signal
        self.traces_update_requested.emit(self.get_all_channel_data())
    
    def handle_data_updated(self, data):
        """Handle data updated signal - determine number of channels and update display"""
        # Store the complete dataset
        self.acq_buffer = data
        self.app_state.current_data = data
        
        # Determine number of channels from data
        if isinstance(data, np.ndarray):
            if data.ndim == 4:  # frames x channels x height x width
                self.num_channels = data.shape[1]
                self.total_frames = data.shape[0]
                self.current_frame = 0
            elif data.ndim == 3:  # frames x height x width (single channel)
                self.num_channels = 1
                self.total_frames = data.shape[0]
                self.current_frame = 0
            else:
                self.num_channels = 1
                self.total_frames = 1
                self.current_frame = 0
        else:
            self.num_channels = 1
            self.total_frames = 0
            self.current_frame = 0
        
        # Update channel names and display
        self.update_channel_names()
        self.setup_channel_display(self.num_channels)
        
        # Update frame controls consistently with ImageDisplayWidget
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.update_frame_label()
        
        # Update display
        self.display_frame(self.current_frame)
        self.update_display()
        
        # Schedule delayed viewport update to fix sizing issues
        self.resize_timer.start(100)  # 100ms delay
        
        # Emit traces update signal consistently with ImageDisplayWidget
        self.traces_update_requested.emit(self.get_all_channel_data())
    
    def on_frame_slider_changed(self, value):
        """Handle frame slider value changes"""
        self.current_frame = value
        self.update_frame_label()
        self.display_frame(self.current_frame)
        self.update_overlays()
        self.update_display()
        self.traces_update_requested.emit(self.get_all_channel_data())
    
    def update_frame_label(self):
        """Update the frame label text"""
        self.frame_label.setText(f'Frame: {self.current_frame + 1}/{self.total_frames}')
    
    def display_frame(self, frame_idx):
        """Display the specified frame across all channels"""
        data = self.app_state.current_data
        
        # If acquisition is ongoing, use buffer
        if self.acq_buffer is not None:
            data = self.acq_buffer
        
        if data is None:
            for scene in self.channel_scenes:
                scene.clear()
            return
        
        if isinstance(data, np.ndarray):
            if data.ndim == 4:  # frames x channels x height x width
                if 0 <= frame_idx < data.shape[0]:
                    frame_data = data[frame_idx]  # channels x height x width
                    for ch in range(min(self.num_channels, frame_data.shape[0])):
                        self._display_channel(ch, frame_data[ch])
                else:
                    for scene in self.channel_scenes:
                        scene.clear()
            elif data.ndim == 3:  # frames x height x width (single channel)
                if 0 <= frame_idx < data.shape[0]:
                    frame_data = data[frame_idx]  # height x width
                    self._display_channel(0, frame_data)
                else:
                    for scene in self.channel_scenes:
                        scene.clear()
            elif data.ndim == 2:  # height x width (single frame, single channel)
                self._display_channel(0, data)
            else:
                for scene in self.channel_scenes:
                    scene.clear()
        else:
            for scene in self.channel_scenes:
                scene.clear()
    
    def _display_channel(self, channel_idx, channel_data):
        """Display a single channel's data"""
        if channel_idx >= len(self.channel_scenes):
            return
        
        scene = self.channel_scenes[channel_idx]
        view = self.channel_views[channel_idx]
        
        if channel_data is None:
            scene.clear()
            return
        
        # Update auto intensity parameters if needed
        self.update_auto_intensity_params(channel_idx, channel_data)
        
        # Get intensity parameters for this channel
        params = self.intensity_params.get(channel_idx, {'min': 0.0, 'max': 1.0, 'auto': True})
        
        # Normalize data based on intensity parameters
        if params['auto']:
            # Auto scaling - use data min/max
            data_min = channel_data.min()
            data_max = channel_data.max()
            if data_max > data_min:
                data_norm = ((channel_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data_norm = np.zeros_like(channel_data, dtype=np.uint8)
        else:
            # Manual scaling - use user-set min/max
            user_min = params['min']
            user_max = params['max']
            
            # Clip data to user range
            clipped_data = np.clip(channel_data, user_min, user_max)
            
            # Normalize to 0-255
            if user_max > user_min:
                data_norm = ((clipped_data - user_min) / (user_max - user_min) * 255).astype(np.uint8)
            else:
                data_norm = np.zeros_like(channel_data, dtype=np.uint8)
        
        # Create QImage and QPixmap
        height, width = channel_data.shape
        qimage = QImage(data_norm.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        
        # Update scene
        scene.clear()
        scene.addPixmap(pixmap)
        scene_rect = QRectF(pixmap.rect())
        scene.setSceneRect(scene_rect)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # Update colorbar
        self.update_colorbar(channel_idx)
    
    def update_auto_intensity_params(self, channel_idx, channel_data):
        """Update intensity parameters when in auto mode based on data"""
        if channel_idx not in self.intensity_params:
            self.intensity_params[channel_idx] = {'min': 0.0, 'max': 1.0, 'auto': True}
        
        params = self.intensity_params[channel_idx]
        if params['auto'] and channel_data is not None:
            # Update min/max based on actual data range
            data_min = float(channel_data.min())
            data_max = float(channel_data.max())
            
            # Only update if we have valid data
            if data_max > data_min:
                params['min'] = data_min
                params['max'] = data_max
    
    def update_display(self):
        """Update all channel displays"""
        self.display_frame(self.current_frame)
        
        # Draw line overlays on all channels
        for scene in self.channel_scenes:
            self._draw_line_overlays(scene)
        
        # Draw temporary line during creation on all channels
        if self.add_mode and self.temp_line_start is not None:
            # Get current mouse position for temporary line
            if self.channel_views:
                cursor_pos = self.channel_views[0].mapFromGlobal(self.channel_views[0].cursor().pos())
                scene_pos = self.channel_views[0].mapToScene(cursor_pos)
                temp_x, temp_y = int(scene_pos.x()), int(scene_pos.y())
                x1, y1 = self.temp_line_start
                
                # Draw temporary line on all channels
                for scene in self.channel_scenes:
                    # Draw temporary line
                    temp_pen = QPen(QColor('#FF6B6B'), 2, Qt.PenStyle.DashLine)
                    scene.addLine(x1, y1, temp_x, temp_y, temp_pen)
                    
                    # Draw temporary endpoint
                    temp_endpoint_pen = QPen(QColor('#FF6B6B'), 1)
                    temp_endpoint_brush = QBrush(QColor('#FF6B6B'))
                    scene.addEllipse(temp_x-3, temp_y-3, 6, 6, temp_endpoint_pen, temp_endpoint_brush)
    
    def _draw_line_overlays(self, scene):
        """Draw line overlays on a scene"""
        for idx, (x1, y1, x2, y2, color) in enumerate(self.get_lines()):
            # Draw the line
            pen = QPen(QColor(color), 2)
            scene.addLine(x1, y1, x2, y2, pen)
            
            # Draw endpoints as circles
            endpoint_pen = QPen(QColor(color), 1)
            endpoint_brush = QBrush(QColor(color))
            scene.addEllipse(x1-3, y1-3, 6, 6, endpoint_pen, endpoint_brush)
            scene.addEllipse(x2-3, y2-3, 6, 6, endpoint_pen, endpoint_brush)
            
            # Add line label at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            text = QGraphicsSimpleTextItem(f"L{idx+1}")
            text.setBrush(QBrush(QColor(color)))
            text.setPos(mid_x + 5, mid_y - 10)
            scene.addItem(text)

    def get_current_frame_data(self):
        """Get current frame data - return first channel for compatibility"""
        if self.acq_buffer is None:
            return None
        
        if self.acq_buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame, 0]  # Return first channel
            else:
                return None
        elif self.acq_buffer.ndim == 3:  # frames x height x width
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame]
            else:
                return None
        else:
            return self.acq_buffer
    
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
    
    def get_image_data_for_rpoc(self):
        """Get image data for RPOC - return all channels"""
        if self.acq_buffer is None:
            return None
        
        if self.acq_buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame]  # Return all channels
            else:
                return None
        elif self.acq_buffer.ndim == 3:  # frames x height x width
            if 0 <= self.current_frame < self.acq_buffer.shape[0]:
                return self.acq_buffer[self.current_frame][np.newaxis, :, :]  # Add channel dimension
            else:
                return None
        else:
            return self.acq_buffer[np.newaxis, :, :]  # Add channel dimension
    
    def get_current_frame_index(self):
        """Get current frame index"""
        return self.current_frame
    
    def update_frame_controls(self, total_frames, current_frame):
        """Update frame navigation controls"""
        self.total_frames = total_frames
        self.current_frame = current_frame
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(current_frame)
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.update_frame_label()

    def eventFilter(self, obj, event):
        """Handle mouse events for line drawing"""
        # Check if this event is from one of our channel viewports
        if obj not in [view.viewport() for view in self.channel_views]:
            return False
        
        if not self.add_mode:
            return False
        
        if event.type() == event.Type.MouseButtonPress:
            return self.handle_mouse_press(obj, event)
        elif event.type() == event.Type.MouseMove:
            return self.handle_mouse_move(event)
        elif event.type() == event.Type.MouseButtonRelease:
            return self.handle_mouse_release(event)
        
        return False
    
    def handle_mouse_press(self, viewport_obj, event):
        """Handle mouse press for line drawing"""
        if event.button() == Qt.MouseButton.LeftButton and self.add_mode:
            # Find which view this event came from
            view = None
            for ch_view in self.channel_views:
                if ch_view.viewport() == viewport_obj:
                    view = ch_view
                    break
            
            if view is None:
                return False
            
            # Convert viewport coordinates to scene coordinates
            pos = event.position() if hasattr(event, 'position') else event.pos()
            scene_pos = view.mapToScene(int(pos.x()), int(pos.y()))
            x, y = int(scene_pos.x()), int(scene_pos.y())
            
            if self.temp_line_start is None:
                # First click - start the line
                self.temp_line_start = (x, y)
                return True
            else:
                # Second click - complete the line
                x1, y1 = self.temp_line_start
                # Send all channel data for proper trace plotting
                all_channel_data = self.get_all_channel_data()
                self.line_add_requested.emit(x1, y1, x, y, all_channel_data, self.get_channel_names())
                self.temp_line_start = None
                self.add_mode = False
                self.update_display()
                return True
        
        return False
    
    def handle_mouse_move(self, event):
        """Handle mouse move for line drawing"""
        if self.add_mode and self.temp_line_start is not None:
            # Force a redraw to show the temporary line
            self.update_display()
            return True
        return False
    
    def handle_mouse_release(self, event):
        """Handle mouse release for line drawing"""
        return False

    @pyqtSlot()
    def enter_add_mode(self):
        """Enter line drawing mode"""
        self.add_mode = True
        self.temp_line_start = None
    
    @pyqtSlot(int)
    def remove_line_overlay(self, index):
        """Remove a line overlay"""
        if hasattr(self, 'lines_widget') and self.lines_widget is not None:
            self.lines_widget.remove_line(index)
        self.update_display()
    
    @pyqtSlot(int, int, int, int, object)
    def move_line_endpoint(self, line_index, endpoint_idx, x, y, image_data):
        """Move a line endpoint"""
        self.line_endpoint_move_requested.emit(line_index, endpoint_idx, x, y, image_data)
        self.update_display()
    
    def get_lines(self):
        """Get current lines"""
        if hasattr(self, 'lines_widget') and self.lines_widget is not None:
            return self.lines_widget.get_lines()
        return []
    
    def connect_lines_widget(self, lines_widget):
        """Connect to the lines widget"""
        super().connect_lines_widget(lines_widget)  # This establishes the signal connections
        self.lines_widget = lines_widget
    
    def get_channel_names(self):
        """Get channel names for legend display in lines widget"""
        return self.channel_names
    
    def on_channel_view_resize(self, view, event):
        """Handle channel view resize to maintain proper sizing"""
        super(view.__class__, view).resizeEvent(event)
        if view.scene() and view.scene().sceneRect().isValid():
            view.fitInView(view.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def delayed_viewport_update(self):
        """Delayed viewport update to fix sizing issues after GUI updates"""
        for view in self.channel_views:
            if view.scene() and view.scene().sceneRect().isValid():
                view.viewport().update()
                view.fitInView(view.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def refresh_channel_labels(self):
        """Refresh channel labels when instruments are updated"""
        if hasattr(self, 'channel_layout'):
            # Find all QLabel widgets in the channel layout and update them
            for i in range(self.channel_layout.count()):
                item = self.channel_layout.itemAt(i)
                if item.widget() and isinstance(item.widget(), QLabel):
                    # This is a channel label, update it
                    row = i // 6  # 6 items per row (label + view for 3 columns)
                    col = (i % 6) // 2  # Every other item is a label
                    channel_idx = row * 3 + col
                    if channel_idx < len(self.channel_names):
                        item.widget().setText(self.channel_names[channel_idx])

class MultichannelDisplayParametersWidget(QWidget):
    def __init__(self, display_widget):
        super().__init__()
        self.display_widget = display_widget
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.channel_controls = []
        self.build_controls()

    def build_controls(self):
        # Clear existing controls
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.channel_controls.clear()

        params = self.display_widget.get_display_parameters()
        num_channels = params.get('num_channels', 1)
        channel_names = params.get('channel_names', [f'Channel {i+1}' for i in range(num_channels)])
        intensity_params = params.get('intensity_params', {})

        for ch in range(num_channels):
            group = QGroupBox(channel_names[ch] if ch < len(channel_names) else f'Channel {ch+1}')
            group_layout = QVBoxLayout()
            group.setLayout(group_layout)

            # Auto checkbox
            auto_checkbox = QCheckBox('Auto')
            auto_checkbox.setChecked(intensity_params.get(ch, {}).get('auto', True))
            auto_checkbox.toggled.connect(lambda checked, ch=ch: self.on_auto_toggled(ch, checked))
            group_layout.addWidget(auto_checkbox)

            # Range slider
            range_slider = QRangeSlider()
            range_slider.setRange(0, 1000)  # Use 0-1000 for precision
            min_val = intensity_params.get(ch, {}).get('min', 0.0)
            max_val = intensity_params.get(ch, {}).get('max', 1.0)
            range_slider.setValue((int(min_val * 1000), int(max_val * 1000)))
            range_slider.valueChanged.connect(lambda val, ch=ch: self.on_range_changed(ch, val))
            group_layout.addWidget(range_slider)

            # Min/Max labels
            labels_layout = QHBoxLayout()
            min_label = QLabel(f'Min: {min_val:.3f}')
            max_label = QLabel(f'Max: {max_val:.3f}')
            labels_layout.addWidget(min_label)
            labels_layout.addStretch()
            labels_layout.addWidget(max_label)
            group_layout.addLayout(labels_layout)

            self.channel_controls.append({
                'auto': auto_checkbox,
                'range_slider': range_slider,
                'min_label': min_label,
                'max_label': max_label
            })
            self.layout.addWidget(group)
        self.layout.addStretch()

    def on_auto_toggled(self, ch, checked):
        params = self.display_widget.get_display_parameters()
        if 'intensity_params' not in params:
            params['intensity_params'] = {}
        if ch not in params['intensity_params']:
            params['intensity_params'][ch] = {'min': 0.0, 'max': 1.0, 'auto': True}
        params['intensity_params'][ch]['auto'] = checked
        self.display_widget.set_display_parameters(params)

    def on_range_changed(self, ch, value):
        min_val = value[0] / 1000.0
        max_val = value[1] / 1000.0
        
        params = self.display_widget.get_display_parameters()
        if 'intensity_params' not in params:
            params['intensity_params'] = {}
        if ch not in params['intensity_params']:
            params['intensity_params'][ch] = {'min': 0.0, 'max': 1.0, 'auto': False}
        
        params['intensity_params'][ch]['min'] = min_val
        params['intensity_params'][ch]['max'] = max_val
        params['intensity_params'][ch]['auto'] = False
        
        # Update labels
        if ch < len(self.channel_controls):
            controls = self.channel_controls[ch]
            controls['min_label'].setText(f'Min: {min_val:.3f}')
            controls['max_label'].setText(f'Max: {max_val:.3f}')
        
        self.display_widget.set_display_parameters(params)

    def refresh(self):
        self.build_controls()
