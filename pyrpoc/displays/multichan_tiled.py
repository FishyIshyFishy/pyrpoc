import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QSlider, QHBoxLayout, QComboBox, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from .base_display import BaseImageDisplayWidget
import math

def make_gray_to_red_colormap():
    # 0.0 to just below 1.0: grayscale
    # 1.0: red
    positions = np.array([0.0, 0.99, 1.0])
    colors = np.array([
        [0, 0, 0, 255],      # black
        [255, 255, 255, 255],# white
        [255, 0, 0, 255],    # red
    ])
    return pg.ColorMap(positions, colors)

class MultichannelImageDisplayWidget(BaseImageDisplayWidget):
    """Multi-channel image display using pyqtgraph.ImageView per channel."""
    display_data_changed = pyqtSignal()

    def __init__(self, app_state, signals):
        super().__init__(app_state, signals)
        self.channel_views = []           # List[ImageView]
        self._buffer = None               # numpy array: (T, C, H, W)
        self.num_channels = 1
        self.channel_names = ['Channel 1']
        
        # Line drawing state
        self.add_mode = False
        self.temp_line_start = None
        self.dragging_endpoint = None
        self.lines_widget = None
        
        # Line overlays for each channel
        self.line_overlays = {}  # {channel_idx: [line_items]}

        self._build_ui()
        # Base class wires up `self.signals.data_updated` → `handle_data_updated`

    def _build_ui(self):
        """Construct frame slider + channel grid + (optional) LUT controls."""
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # — Frame selector —
        frame_ctrl = QWidget()
        fc = QHBoxLayout(frame_ctrl); fc.setContentsMargins(0,0,0,0)
        fc.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        fc.addWidget(self.frame_slider)
        self.frame_label = QLabel("0/0")
        fc.addWidget(self.frame_label)
        layout.addWidget(frame_ctrl)

        # — Grid for channels —
        self.grid = QGridLayout()
        self.grid.setSpacing(4)  # Add some spacing between grid items
        
        # Create a widget to hold the grid layout with proper size policy
        grid_widget = QWidget()
        grid_widget.setLayout(self.grid)
        grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout.addWidget(grid_widget)

        # Initial channel
        self._setup_channel_views(1)

    def _setup_channel_views(self, num_channels):
        """(Re)build one ImageView + LUT combo per channel."""
        # clear old
        for iv in self.channel_views:
            iv.setParent(None)
        self.channel_views.clear()
        self.line_overlays.clear()

        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # update names
        self.num_channels = num_channels
        self.update_channel_names()

        cols = int(math.ceil(math.sqrt(num_channels)))
        rows = int(math.ceil(num_channels/cols))
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= num_channels:
                    break

                # 1) Title
                title = QLabel(self.channel_names[idx])
                title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                title.setStyleSheet("font-weight:bold;")
                self.grid.addWidget(title, 2*r, c)

                # 2) ImageView
                iv = ImageView()
                iv.ui.roiBtn.hide()       # hide ROI if unused
                iv.ui.menuBtn.show()      # show context menu (export, auto-levels…)
                iv.ui.histogram.show()    # dock histogram + LUT controls
                iv.getView().setAspectLocked(True)
                iv.getView().invertY(True)
                iv.getView().setBackgroundColor('w')
                
                # Hide the timeline and plot below the image
                if hasattr(iv.ui, 'timeLine'):
                    iv.ui.timeLine.hide()
                if hasattr(iv.ui, 'roiPlot'):
                    iv.ui.roiPlot.hide()
                
                # Install event filter for mouse handling
                iv.ui.graphicsView.viewport().installEventFilter(self)

                # Set size policies to prevent compression
                iv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                cmap = make_gray_to_red_colormap()
                iv.setColorMap(cmap)

                self.grid.addWidget(iv, 2*r+1, c)
                
                # Set stretch factors for the grid to ensure proper distribution
                self.grid.setRowStretch(2*r+1, 1)  # ImageView rows get stretch
                self.grid.setColumnStretch(c, 1)    # All columns get equal stretch

                self.channel_views.append(iv)
                self.line_overlays[idx] = []
                idx += 1

    def update_channel_names(self):
        """Update channel names based on modality and instruments."""
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
        """Get channel names from data input instruments."""
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

    def handle_frame_acquired(self, data_unit, idx, total):
        """Handle frame acquired signal for confocal multi-channel data."""
        # Reset buffer if this is the first frame of a new acquisition
        if idx == 0:
            self._buffer = None
        
        # Store the acquired frame in the buffer
        if self._buffer is None:
            # Initialize buffer based on data shape
            if isinstance(data_unit, np.ndarray):
                if data_unit.ndim == 3:  # channels x height x width
                    self._buffer = np.zeros((total, data_unit.shape[0], data_unit.shape[1], data_unit.shape[2]))
                    self.num_channels = data_unit.shape[0]
                    # Only rebuild if channel count actually changed
                    if self.num_channels != len(self.channel_views):
                        self.update_channel_names()
                        self._setup_channel_views(self.num_channels)
                        self._refresh_channel_labels()
                elif data_unit.ndim == 2:  # height x width (single channel)
                    self._buffer = np.zeros((total, 1, data_unit.shape[0], data_unit.shape[1]))
                    self.num_channels = 1
                    self.update_channel_names()
                else:
                    self._buffer = np.zeros((total, 1, data_unit.shape[0], data_unit.shape[1]))
                    self.num_channels = 1
                    self.update_channel_names()
            else:
                # Fallback for non-array data
                self._buffer = np.zeros((total, 1, 512, 512))
                self.num_channels = 1
                self.update_channel_names()
        
        # Store the frame data
        if isinstance(data_unit, np.ndarray):
            if data_unit.ndim == 3:  # channels x height x width
                if idx < self._buffer.shape[0]:  # Safety check
                    self._buffer[idx] = data_unit
            elif data_unit.ndim == 2:  # height x width (single channel)
                if idx < self._buffer.shape[0]:  # Safety check
                    self._buffer[idx, 0] = data_unit
            else:
                if idx < self._buffer.shape[0]:  # Safety check
                    self._buffer[idx, 0] = data_unit
        
        # Update current frame and frame controls
        self.current_frame = idx
        self.total_frames = total
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.frame_slider.setValue(idx)
        self._update_frame_label()
        
        # Update display if this is the current frame
        if idx == self.current_frame:
            self._display_frame(idx)
        
        # Update frame controls if this is the first frame
        if idx == 0:
            self._update_frame_controls(total, 0)
        
        # Only update overlays and emit signals if this is the current frame
        if idx == self.current_frame:
            self.update_overlays()
            # Emit traces update signal
            self.traces_update_requested.emit(self.get_all_channel_data())
        
        # Emit display data changed after data and channels are set up
        self.display_data_changed.emit()

    def handle_data_updated(self, data: np.ndarray):
        """New 3D/4D array available from acquisition or load."""
        # normalize to (T, C, H, W)
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]
        if data.ndim != 4:
            raise ValueError("Expected 3D or 4D array")

        self._buffer = data
        self.app_state.current_data = data
        T, C, H, W = data.shape

        # Only rebuild if channel count actually changed and we don't have the right number of views
        if C != len(self.channel_views):
            self._setup_channel_views(C)
            # Update channel names after rebuilding
            self.update_channel_names()
            # Refresh channel labels
            self._refresh_channel_labels()

        # configure frame slider
        self.frame_slider.setEnabled(T > 1)
        self.frame_slider.setRange(0, max(0, T-1))
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"1/{T}")
        
        # Update current frame tracking
        self.current_frame = 0
        self.total_frames = T

        # hand off entire time-series to each ImageView
        for ch, iv in enumerate(self.channel_views):
            iv.setImage(data[:,ch,:,:], xvals=np.arange(T))
            iv.autoLevels()  # initial auto-scale
            self._hide_imageview_timeline(iv)

        self.display_data_changed.emit()

    def _on_frame_slider(self, idx: int):
        """User changed frame → update all views."""
        self.current_frame = idx
        T = self._buffer.shape[0] if self._buffer is not None else 0
        self.frame_label.setText(f"{idx+1}/{T}")
        for iv in self.channel_views:
            iv.setCurrentIndex(idx)
        # redraw overlays if any
        self.update_overlays()
        self.traces_update_requested.emit(self.get_all_channel_data())

    def _update_frame_label(self):
        """Update the frame label text."""
        self.frame_label.setText(f"{self.current_frame + 1}/{self.total_frames}")

    def _display_frame(self, frame_idx):
        """Display the specified frame across all channels."""
        if self._buffer is None:
            for iv in self.channel_views:
                iv.clear()
            return

        if isinstance(self._buffer, np.ndarray):
            if self._buffer.ndim == 4:  # frames x channels x height x width
                if 0 <= frame_idx < self._buffer.shape[0]:
                    frame_data = self._buffer[frame_idx]  # channels x height x width
                    for ch in range(min(self.num_channels, frame_data.shape[0])):
                        if ch < len(self.channel_views):
                            self.channel_views[ch].setImage(frame_data[ch])
                else:
                    for iv in self.channel_views:
                        iv.clear()
            elif self._buffer.ndim == 3:  # frames x height x width (single channel)
                if 0 <= frame_idx < self._buffer.shape[0]:
                    frame_data = self._buffer[frame_idx]  # height x width
                    if self.channel_views:
                        self.channel_views[0].setImage(frame_data)
                else:
                    for iv in self.channel_views:
                        iv.clear()
            elif self._buffer.ndim == 2:  # height x width (single frame, single channel)
                if self.channel_views:
                    self.channel_views[0].setImage(self._buffer)
            else:
                for iv in self.channel_views:
                    iv.clear()
        else:
            for iv in self.channel_views:
                iv.clear()

    def display_frame(self, frame_idx):
        """Display a specific frame - required by base class interface."""
        self._display_frame(frame_idx)

    def update_display(self):
        """Update all channel displays."""
        self._display_frame(self.current_frame)
        self._draw_line_overlays()

    def _draw_line_overlays(self):
        """Draw line overlays on all channels."""
        # Clear existing overlays
        for channel_idx in self.line_overlays:
            for item in self.line_overlays[channel_idx]:
                if hasattr(item, 'scene') and item.scene():
                    item.scene().removeItem(item)
            self.line_overlays[channel_idx].clear()
        
        # Draw lines on all channels
        for channel_idx in range(self.num_channels):
            if channel_idx >= len(self.channel_views):
                continue
                
            view = self.channel_views[channel_idx]
            plot_item = view.getView()
            
            for idx, (x1, y1, x2, y2, color) in enumerate(self.get_lines()):
                # Draw the line
                line = pg.PlotDataItem(
                    x=[x1, x2], 
                    y=[y1, y2], 
                    pen=pg.mkPen(color=color, width=2)
                )
                plot_item.addItem(line)
                self.line_overlays[channel_idx].append(line)
                
                # Draw endpoints as circles
                endpoint1 = pg.ScatterPlotItem(
                    x=[x1], y=[y1], 
                    pen=pg.mkPen(color=color), 
                    brush=pg.mkBrush(color=color),
                    size=6
                )
                endpoint2 = pg.ScatterPlotItem(
                    x=[x2], y=[y2], 
                    pen=pg.mkPen(color=color), 
                    brush=pg.mkBrush(color=color),
                    size=6
                )
                plot_item.addItem(endpoint1)
                plot_item.addItem(endpoint2)
                self.line_overlays[channel_idx].extend([endpoint1, endpoint2])
                
                # Add line label at midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                text = pg.TextItem(
                    text=f"L{idx+1}", 
                    color=color,
                    anchor=(0, 0)
                )
                text.setPos(mid_x + 5, mid_y - 10)
                plot_item.addItem(text)
                self.line_overlays[channel_idx].append(text)
        
        # Draw temporary line during creation
        if self.add_mode and self.temp_line_start is not None and self.channel_views:
            # Get current mouse position for temporary line
            cursor_pos = self.channel_views[0].mapFromGlobal(self.channel_views[0].cursor().pos())
            scene_pos = self.channel_views[0].getView().mapSceneToView(cursor_pos)
            temp_x, temp_y = int(scene_pos.x()), int(scene_pos.y())
            x1, y1 = self.temp_line_start
            
            # Draw temporary line on all channels
            for channel_idx in range(self.num_channels):
                if channel_idx >= len(self.channel_views):
                    continue
                    
                view = self.channel_views[channel_idx]
                plot_item = view.getView()
                
                # Draw temporary line
                temp_line = pg.PlotDataItem(
                    x=[x1, temp_x], 
                    y=[y1, temp_y], 
                    pen=pg.mkPen(color='#FF6B6B', width=2, style=Qt.PenStyle.DashLine)
                )
                plot_item.addItem(temp_line)
                self.line_overlays[channel_idx].append(temp_line)
                
                # Draw temporary endpoint
                temp_endpoint = pg.ScatterPlotItem(
                    x=[temp_x], y=[temp_y], 
                    pen=pg.mkPen(color='#FF6B6B'), 
                    brush=pg.mkBrush(color='#FF6B6B'),
                    size=6
                )
                plot_item.addItem(temp_endpoint)
                self.line_overlays[channel_idx].append(temp_endpoint)

    def get_current_frame_data(self):
        """Get current frame data - return first channel for compatibility."""
        if self._buffer is None:
            return None
        
        if self._buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self._buffer.shape[0]:
                return self._buffer[self.current_frame, 0]  # Return first channel
            else:
                return None
        elif self._buffer.ndim == 3:  # frames x height x width
            if 0 <= self.current_frame < self._buffer.shape[0]:
                return self._buffer[self.current_frame]
            else:
                return None
        else:
            return self._buffer

    def get_all_channel_data(self):
        """Get all channel data for the current frame - for confocal mode."""
        if self._buffer is None:
            return None
        
        if self._buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self._buffer.shape[0]:
                return self._buffer[self.current_frame]  # Return all channels
            else:
                return None
        elif self._buffer.ndim == 3:  # frames x height x width (single channel)
            if 0 <= self.current_frame < self._buffer.shape[0]:
                return self._buffer[self.current_frame][np.newaxis, :, :]  # Add channel dimension
            else:
                return None
        else:
            return self._buffer[np.newaxis, :, :]  # Add channel dimension

    def get_image_data_for_rpoc(self):
        """Get image data for RPOC - return all channels."""
        if self._buffer is None:
            return None
        
        if self._buffer.ndim == 4:  # frames x channels x height x width
            if 0 <= self.current_frame < self._buffer.shape[0]:
                return self._buffer[self.current_frame]  # Return all channels
            else:
                return None
        elif self._buffer.ndim == 3:  # frames x height x width
            if 0 <= self.current_frame < self._buffer.shape[0]:
                return self._buffer[self.current_frame][np.newaxis, :, :]  # Add channel dimension
            else:
                return None
        else:
            return self._buffer[np.newaxis, :, :]  # Add channel dimension

    def get_current_frame_index(self):
        """Get current frame index."""
        return self.current_frame

    def _update_frame_controls(self, total_frames, current_frame):
        """Update frame navigation controls."""
        self.total_frames = total_frames
        self.current_frame = current_frame
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(current_frame)
        self.frame_slider.setEnabled(self.total_frames > 1)
        self._update_frame_label()

    def eventFilter(self, obj, event):
        """Handle mouse events for line drawing."""
        # Check if this event is from one of our channel views
        if obj not in [iv.ui.graphicsView.viewport() for iv in self.channel_views]:
            return False
        
        if not self.add_mode:
            return False
        
        if event.type() == event.Type.MouseButtonPress:
            return self._handle_mouse_press(obj, event)
        elif event.type() == event.Type.MouseMove:
            return self._handle_mouse_move(event)
        elif event.type() == event.Type.MouseButtonRelease:
            return self._handle_mouse_release(event)
        
        return False
    
    def _handle_mouse_press(self, viewport_obj, event):
        """Handle mouse press for line drawing."""
        if event.button() == Qt.MouseButton.LeftButton and self.add_mode:
            # Find which view this event came from
            view = None
            for ch_view in self.channel_views:
                if ch_view.ui.graphicsView.viewport() == viewport_obj:
                    view = ch_view
                    break
            
            if view is None:
                return False
            
            # Convert viewport coordinates to scene coordinates
            pos = event.position() if hasattr(event, 'position') else event.pos()
            scene_pos = view.getView().mapSceneToView(pos)
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
    
    def _handle_mouse_move(self, event):
        """Handle mouse move for line drawing."""
        if self.add_mode and self.temp_line_start is not None:
            # Force a redraw to show the temporary line
            self.update_display()
            return True
        return False
    
    def _handle_mouse_release(self, event):
        """Handle mouse release for line drawing."""
        return False

    @pyqtSlot()
    def enter_add_mode(self):
        """Enter line drawing mode."""
        self.add_mode = True
        self.temp_line_start = None
    
    @pyqtSlot(int)
    def remove_line_overlay(self, index):
        """Remove a line overlay."""
        if hasattr(self, 'lines_widget') and self.lines_widget is not None:
            self.lines_widget.remove_line(index)
        self.update_display()
    
    @pyqtSlot(int, int, int, int, object)
    def move_line_endpoint(self, line_index, endpoint_idx, x, y, image_data):
        """Move a line endpoint."""
        self.line_endpoint_move_requested.emit(line_index, endpoint_idx, x, y, image_data)
        self.update_display()
    
    def get_lines(self):
        """Get current lines."""
        if hasattr(self, 'lines_widget') and self.lines_widget is not None:
            return self.lines_widget.get_lines()
        return []
    
    def connect_lines_widget(self, lines_widget):
        """Connect to the lines widget."""
        super().connect_lines_widget(lines_widget)  # This establishes the signal connections
        self.lines_widget = lines_widget
    
    def get_channel_names(self):
        """Get channel names for legend display in lines widget."""
        return self.channel_names

    def get_display_parameters(self):
        """Get current display parameters for GUI coupling."""
        return {
            'num_channels': self.num_channels,
            'channel_names': self.channel_names.copy(),
            'current_frame': self.current_frame
        }
    
    def set_display_parameters(self, params):
        """Set display parameters from GUI."""
        # This method can be extended if needed for parameter setting
        pass

    def _refresh_channel_labels(self):
        """Refresh channel labels without rebuilding the entire UI."""
        # Update the title labels in the grid
        for idx in range(self.num_channels):
            if idx < len(self.channel_names):
                # Find the title label in the grid and update it
                for i in range(self.grid.count()):
                    item = self.grid.itemAt(i)
                    if item and item.widget() and isinstance(item.widget(), QLabel):
                        # Check if this is a title label (even row)
                        row = i // self.grid.columnCount()
                        col = i % self.grid.columnCount()
                        if row % 2 == 0 and col == idx % self.grid.columnCount():
                            item.widget().setText(self.channel_names[idx])
                            break

    def refresh_channel_labels(self):
        """Refresh channel labels when instruments are updated."""
        self.update_channel_names()
        self._refresh_channel_labels()

    def _hide_imageview_timeline(self, iv):
        """Hide the timeline and roiPlot below the image for a given ImageView."""
        if hasattr(iv.ui, 'timeLine'):
            iv.ui.timeLine.hide()
        if hasattr(iv.ui, 'roiPlot'):
            iv.ui.roiPlot.hide()


class MultichannelDisplayParametersWidget(QWidget):
    """Widget for controlling multichannel display parameters."""
    
    def __init__(self, display_widget):
        super().__init__()
        self.display_widget = display_widget
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.channel_controls = []
        self.last_num_channels = 0
        self.build_controls()

    def build_controls(self):
        """Build the control widgets."""
        # Clear existing controls
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.channel_controls.clear()
        
        params = self.display_widget.get_display_parameters()
        num_channels = params.get('num_channels', 1)
        channel_names = params.get('channel_names', [f'Channel {i+1}' for i in range(num_channels)])
        
        for ch in range(num_channels):
            group = QLabel(channel_names[ch] if ch < len(channel_names) else f'Channel {ch+1}')
            group.setStyleSheet('font-weight: bold; color: #333;')
            self.layout.addWidget(group)
        
        self.layout.addStretch()
        self.last_num_channels = num_channels

    def refresh(self):
        """Refresh the controls."""
        self.build_controls() 