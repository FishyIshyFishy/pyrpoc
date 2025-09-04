import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QSlider, QHBoxLayout, QComboBox, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from .base_display import BaseImageDisplayWidget
import math
from utils import *

class MultichannelImageDisplayWidget(BaseImageDisplayWidget):
    """Multi-channel image display using pyqtgraph.ImageView per channel."""
    display_data_changed = pyqtSignal()

    def __init__(self, app_state, signals):
        super().__init__(app_state, signals)
        self.channel_views = []           # List[ImageView]
        self._buffer = None               # numpy array: (T, C, H, W)
        self.num_channels = 1
        self.channel_names = ['Channel 1']

        # Initialize frame counter
        self._current_frame_idx = 0
        
        self._build_ui()
        # Base class provides interface for data handling

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

    def handle_data_received(self, data):
        """
        New method for handling individual data frames during acquisition.
        This is part of the uniform acquisition pipeline.
        
        data: The data frame that was just received
        """
        # Reset buffer if this is the first frame of a new acquisition
        if not hasattr(self, '_current_frame_idx') or self._current_frame_idx == 0:
            self._buffer = None
        
        # Store the acquired frame in the buffer
        if self._buffer is None:
            # Initialize buffer based on data shape
            if isinstance(data, np.ndarray):
                if data.ndim == 3:  # channels x height x width
                    self._buffer = np.zeros((self.total_frames, data.shape[0], data.shape[1], data.shape[2]))
                    self.num_channels = data.shape[0]
                    # Only rebuild if channel count actually changed
                    if self.num_channels != len(self.channel_views):
                        self.update_channel_names()
                        self._setup_channel_views(self.num_channels)
                        self._refresh_channel_labels()
                elif data.ndim == 2:  # height x width (single channel)
                    self._buffer = np.zeros((self.total_frames, 1, data.shape[0], data.shape[1]))
                    self.num_channels = 1
                    self.update_channel_names()
                else:
                    # unexpected shape; coerce to single channel 2D per frame
                    flat = data.reshape(-1)
                    size = int(np.sqrt(flat.size))
                    self._buffer = np.zeros((self.total_frames, 1, size, size))
                    self.num_channels = 1
                    self.update_channel_names()
            else:
                # Fallback for non-array data
                self._buffer = np.zeros((self.total_frames, 1, 512, 512))
                self.num_channels = 1
                self.update_channel_names()
        
        # Store the frame data
        if isinstance(data, np.ndarray):
            # grow buffer if needed
            if self._current_frame_idx >= self._buffer.shape[0]:
                new_len = max(self._current_frame_idx + 1, self._buffer.shape[0] * 2 if self._buffer.shape[0] > 0 else 1)
                T, C, H, W = self._buffer.shape
                new_buf = np.zeros((new_len, C, H, W), dtype=self._buffer.dtype)
                new_buf[:T] = self._buffer
                self._buffer = new_buf
            if data.ndim == 3:  # channels x height x width
                self._buffer[self._current_frame_idx] = data
            elif data.ndim == 2:  # height x width (single channel)
                self._buffer[self._current_frame_idx, 0] = data
            else:
                # unexpected, attempt first channel 2D
                self._buffer[self._current_frame_idx, 0] = data.reshape(self._buffer.shape[2], self._buffer.shape[3])
        
        # Update current frame and frame controls
        self.current_frame = self._current_frame_idx
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.frame_slider.setValue(self._current_frame_idx)
        self._update_frame_label()
        
        # Update display if this is the current frame
        if self._current_frame_idx == self.current_frame:
            self._display_frame(self._current_frame_idx)
        
        # Update frame controls if this is the first frame
        if self._current_frame_idx == 0:
            self._update_frame_controls(self.total_frames, 0)
        
        # Emit display data changed after data and channels are set up
        self.display_data_changed.emit()
        
        # Increment frame counter
        self._current_frame_idx += 1

    def prepare_for_acquisition(self, context_or_total_frames):
        if isinstance(context_or_total_frames, int):
            total_frames = context_or_total_frames
            self.acquisition_context = None

        # Reset frame counter for new acquisition
        self._current_frame_idx = 0
        
        # Update internal state
        self.total_frames = max(1, int(total_frames))
        self.current_frame = 0
        
        # Clear acquisition buffer
        self._buffer = None
        self.acq_total = self.total_frames
        
        # Reset frame controls
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(self.total_frames > 1)
        self._update_frame_label()
        
        # Clear all channel views
        for iv in self.channel_views:
            iv.clear()
        
        # Update frame label
        self.frame_label.setText(f"1/{self.total_frames}")
        
        # Emit display data changed to notify other components
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
            # swap height<->width
            iv.setImage(
                np.transpose(data[:, ch, :, :], (0, 2, 1)),
                xvals=np.arange(T)
            )
            iv.autoLevels()  # initial auto-scale
            self._hide_imageview_timeline(iv)

        self.display_data_changed.emit()

    def _on_frame_slider(self, idx: int):
        """User changed frame → update all views."""
        self.current_frame = idx
        T = self._buffer.shape[0] if self._buffer is not None else 0
        self.frame_label.setText(f"{idx+1}/{T}")
        # redraw overlays if any
        self.update_overlays()

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
            if self._buffer.ndim == 4:  # frames x channels x H x W
                if 0 <= frame_idx < self._buffer.shape[0]:
                    frame_data = self._buffer[frame_idx]  # C x H x W
                    for ch in range(min(self.num_channels, frame_data.shape[0])):
                        if ch < len(self.channel_views):
                            # transpose each 2D channel
                            self.channel_views[ch].setImage(frame_data[ch].T)
                else:
                    for iv in self.channel_views:
                        iv.clear()
            elif self._buffer.ndim == 3:  # frames x height x width (single channel)
                if 0 <= frame_idx < self._buffer.shape[0]:
                    frame_data = self._buffer[frame_idx]  # height x width
                    if self.channel_views:
                        # transpose each 2D channel
                        self.channel_views[0].setImage(frame_data.T)
                else:
                    for iv in self.channel_views:
                        iv.clear()
            elif self._buffer.ndim == 2:  # height x width (single frame, single channel)
                if self.channel_views:
                    # transpose each 2D channel
                    self.channel_views[0].setImage(self._buffer.T)
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