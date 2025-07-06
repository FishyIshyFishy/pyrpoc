import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem, QGridLayout
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont
from .base_display import BaseImageDisplayWidget


class MultichannelImageDisplayWidget(BaseImageDisplayWidget):
    def __init__(self, app_state, signals):
        super().__init__(app_state, signals)
        self.num_channels = 1
        self.channel_names = ['Channel 1']
        self.channel_scenes = []
        self.channel_views = []
        
        # Line drawing state
        self._add_mode = False
        self._temp_line_start = None
        self._dragging_endpoint = None
        
        # Lines widget reference
        self._lines_widget = None
        
        self.setup_ui()
        self.update_channel_names()
    
    def update_channel_names(self):
        """Update channel names based on modality and data input instruments"""
        modality = self.app_state.modality.lower()
        
        # Try to get channel names from data input instruments first
        channel_names = self._get_channel_names_from_instruments()
        
        if channel_names and len(channel_names) >= self.num_channels:
            # Use the actual channel names from instruments
            self.channel_names = channel_names[:self.num_channels]
        else:
            # Fallback to modality-specific naming
            if modality == 'confocal':
                # For confocal, use generic channel names
                self.channel_names = [f'Channel {i+1}' for i in range(self.num_channels)]
            elif modality == 'split data stream':
                # For split data stream, use descriptive channel names
                # Each input channel creates 3 output channels
                input_channels = self.num_channels // 3
                self.channel_names = []
                for input_ch in range(input_channels):
                    base_name = f'Input {input_ch + 1}'
                    self.channel_names.extend([
                        f'{base_name} - First Portion',
                        f'{base_name} - Second Portion', 
                        f'{base_name} - Full Data'
                    ])
                # Handle any remaining channels (in case num_channels is not divisible by 3)
                remaining = self.num_channels % 3
                if remaining > 0:
                    for i in range(remaining):
                        self.channel_names.append(f'Channel {input_channels * 3 + i + 1}')
            else:
                # Default fallback
                self.channel_names = [f'Channel {i+1}' for i in range(self.num_channels)]
    
    def _get_channel_names_from_instruments(self):
        """Get channel names from data input instruments in the app state"""
        channel_names = []
        
        if hasattr(self.app_state, 'instruments'):
            for instrument in self.app_state.instruments:
                if instrument.instrument_type == "Data Input" and hasattr(instrument, 'parameters'):
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
        self._setup_channel_display(1)
        
        # Timer for delayed viewport updates to fix sizing issues
        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._delayed_viewport_update)
    
    def _setup_channel_display(self, num_channels):
        """Setup the channel grid display"""
        # Clear existing views
        for view in self.channel_views:
            view.deleteLater()
        self.channel_views.clear()
        self.channel_scenes.clear()
        
        # Clear layout
        while self.channel_layout.count():
            child = self.channel_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Update channel names before creating display
        self.update_channel_names()
        
        # Calculate grid dimensions
        cols = min(3, num_channels)  # Max 3 columns
        rows = (num_channels + cols - 1) // cols
        
        # Create views for each channel
        for i in range(num_channels):
            # Create graphics view and scene
            view = QGraphicsView()
            scene = QGraphicsScene()
            view.setScene(scene)
            view.setStyleSheet('''
                QGraphicsView {
                    border: 2px dashed #cccccc;
                    background-color: #f0f0f0;
                }
            ''')
            
            # Add channel label
            channel_name = self.channel_names[i] if i < len(self.channel_names) else f'Channel {i+1}'
            label = QLabel(channel_name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('font-weight: bold; color: #333;')
            
            # Add to layout
            row = i // cols
            col = i % cols
            self.channel_layout.addWidget(label, row*2, col)
            self.channel_layout.addWidget(view, row*2+1, col)
            
            self.channel_views.append(view)
            self.channel_scenes.append(scene)
            
            # Install event filter for mouse interaction
            view.viewport().installEventFilter(self)
            
            # Connect resize event to ensure proper sizing
            view.resizeEvent = lambda event, v=view: self.on_channel_view_resize(v, event)
    
    def handle_frame_acquired(self, data_unit, idx, total):
        """Handle frame acquired signal for confocal multi-channel data"""
        # Reset buffer if this is the first frame of a new acquisition
        if idx == 0:
            self._acq_buffer = None
        
        # Store the acquired frame in the buffer
        if self._acq_buffer is None:
            # Initialize buffer based on data shape
            if isinstance(data_unit, np.ndarray):
                if data_unit.ndim == 3:  # channels x height x width
                    self._acq_buffer = np.zeros((total, data_unit.shape[0], data_unit.shape[1], data_unit.shape[2]))
                    self.num_channels = data_unit.shape[0]
                    # Update channel names and display
                    self.update_channel_names()
                    self._setup_channel_display(self.num_channels)
                elif data_unit.ndim == 2:  # height x width (single channel)
                    self._acq_buffer = np.zeros((total, 1, data_unit.shape[0], data_unit.shape[1]))
                    self.num_channels = 1
                    self.update_channel_names()
                else:
                    self._acq_buffer = np.zeros((total, 1, data_unit.shape[0], data_unit.shape[1]))
                    self.num_channels = 1
                    self.update_channel_names()
            else:
                # Fallback for non-array data
                self._acq_buffer = np.zeros((total, 1, 512, 512))
                self.num_channels = 1
                self.update_channel_names()
        
        # Store the frame data
        if isinstance(data_unit, np.ndarray):
            if data_unit.ndim == 3:  # channels x height x width
                if idx < self._acq_buffer.shape[0]:  # Safety check
                    self._acq_buffer[idx] = data_unit
            elif data_unit.ndim == 2:  # height x width (single channel)
                if idx < self._acq_buffer.shape[0]:  # Safety check
                    self._acq_buffer[idx, 0] = data_unit
            else:
                if idx < self._acq_buffer.shape[0]:  # Safety check
                    self._acq_buffer[idx, 0] = data_unit
        
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
            self._resize_timer.start(100)  # 100ms delay
        
        # Update overlays and display consistently with ImageDisplayWidget
        self.update_overlays()
        self.update_display()
        
        # Emit traces update signal
        self.traces_update_requested.emit(self.get_all_channel_data())
    
    def handle_data_updated(self, data):
        """Handle data updated signal - determine number of channels and update display"""
        # Store the complete dataset
        self._acq_buffer = data
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
        self._setup_channel_display(self.num_channels)
        
        # Update frame controls consistently with ImageDisplayWidget
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.update_frame_label()
        
        # Update display
        self.display_frame(self.current_frame)
        self.update_display()
        
        # Schedule delayed viewport update to fix sizing issues
        self._resize_timer.start(100)  # 100ms delay
        
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
        if self._acq_buffer is not None:
            data = self._acq_buffer
        
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
        
        # Normalize data
        data_norm = ((channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-9) * 255).astype(np.uint8)
        
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
    
    def update_display(self):
        """Update all channel displays"""
        self.display_frame(self.current_frame)
        
        # Draw line overlays on all channels
        for scene in self.channel_scenes:
            self._draw_line_overlays(scene)
        
        # Draw temporary line during creation on all channels
        if self._add_mode and self._temp_line_start is not None:
            # Get current mouse position for temporary line
            if self.channel_views:
                cursor_pos = self.channel_views[0].mapFromGlobal(self.channel_views[0].cursor().pos())
                scene_pos = self.channel_views[0].mapToScene(cursor_pos)
                temp_x, temp_y = int(scene_pos.x()), int(scene_pos.y())
                x1, y1 = self._temp_line_start
                
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
        if self._acq_buffer is None:
            return None
        
        if self._acq_buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame, 0]  # Return first channel
            else:
                return None
        elif self._acq_buffer.ndim == 3:  # frames x height x width
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame]
            else:
                return None
        else:
            return self._acq_buffer
    
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
    
    def get_image_data_for_rpoc(self):
        """Get image data for RPOC - return all channels"""
        if self._acq_buffer is None:
            return None
        
        if self._acq_buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame]  # Return all channels
            else:
                return None
        elif self._acq_buffer.ndim == 3:  # frames x height x width
            if 0 <= self.current_frame < self._acq_buffer.shape[0]:
                return self._acq_buffer[self.current_frame][np.newaxis, :, :]  # Add channel dimension
            else:
                return None
        else:
            return self._acq_buffer[np.newaxis, :, :]  # Add channel dimension
    
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
        
        if not self._add_mode:
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
        if event.button() == Qt.MouseButton.LeftButton and self._add_mode:
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
                self._temp_line_start = None
                self._add_mode = False
                self.update_display()
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
        """Handle mouse release for line drawing"""
        return False

    @pyqtSlot()
    def enter_add_mode(self):
        """Enter line drawing mode"""
        self._add_mode = True
        self._temp_line_start = None
    
    @pyqtSlot(int)
    def remove_line_overlay(self, index):
        """Remove a line overlay"""
        if hasattr(self, '_lines_widget') and self._lines_widget is not None:
            self._lines_widget.remove_line(index)
        self.update_display()
    
    @pyqtSlot(int, int, int, int, object)
    def move_line_endpoint(self, line_index, endpoint_idx, x, y, image_data):
        """Move a line endpoint"""
        self.line_endpoint_move_requested.emit(line_index, endpoint_idx, x, y, image_data)
        self.update_display()
    
    def get_lines(self):
        """Get current lines"""
        if hasattr(self, '_lines_widget') and self._lines_widget is not None:
            return self._lines_widget.get_lines()
        return []
    
    def connect_lines_widget(self, lines_widget):
        """Connect to the lines widget"""
        super().connect_lines_widget(lines_widget)  # This establishes the signal connections
        self._lines_widget = lines_widget
    
    def get_channel_names(self):
        """Get channel names for legend display in lines widget"""
        return self.channel_names
    
    def on_channel_view_resize(self, view, event):
        """Handle channel view resize to maintain proper sizing"""
        super(view.__class__, view).resizeEvent(event)
        if view.scene() and view.scene().sceneRect().isValid():
            view.fitInView(view.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def _delayed_viewport_update(self):
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
