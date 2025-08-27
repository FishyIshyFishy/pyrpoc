import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem, QGridLayout
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont

class BaseImageDisplayWidget(QWidget):
    '''
    Abstract base class defining the interface that all image display widgets must implement
    to work with the uniform acquisition pipeline and dockable widgets.

    The new uniform acquisition pipeline follows this sequence:
    1. User presses start → display_setup_requested signal emitted
    2. Display widget is set up (if different from current) → acquisition_setup_complete signal emitted
    3. During acquisition → data_frame_received signal emitted for each frame
    4. Acquisition complete → acquisition_complete signal emitted

    This eliminates modality-specific logic in the GUI, making displays modality-agnostic.
    '''
    
    # all widgets must deal with these signals
    line_add_requested = pyqtSignal(int, int, int, int, object, list)  # x1, y1, x2, y2, image_data, channel_names
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
        self.acq_buffer = None
        self.acq_total = None
        
        # Overlay callbacks for future extensibility
        self.overlay_callbacks = []
    
    def handle_data_frame_received(self, data):
        '''
        New method for handling individual data frames during acquisition.
        This is part of the uniform acquisition pipeline.
        
        data: The data frame that was just received
        '''
        # Default implementation: update current frame
        # Subclasses should override this for specific frame handling
        if hasattr(self, '_current_frame_idx'):
            self._current_frame_idx += 1
        else:
            self._current_frame_idx = 1
    
    def prepare_for_acquisition(self, total_frames):
        '''
        Prepare the display widget for acquisition.
        This is called when acquisition_setup_complete is emitted.
        
        total_frames: Total number of frames expected in this acquisition
        '''
        # Default implementation: update internal state
        self.total_frames = total_frames
        self.current_frame = 0
        
        # Clear acquisition buffer
        self.acq_buffer = None
        self.acq_total = total_frames
        
        # Initialize frame counter
        self._current_frame_idx = 0
        
        # Subclasses can override this for more specific preparation
    

    

    
    def get_current_frame_data(self):
        '''
        Get the data for the current displayed unit of data.
        
        returns: numpy.ndarray or None: current frame data, or None if no data available
        '''
        raise NotImplementedError("Subclasses must implement get_current_frame_data")
    
    def get_image_data_for_rpoc(self):
        '''
        Get image data in a format suitable for RPOC mask creation.
        
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
        Get the current displayed unit of data index.
        
        returns: int: Current frame index
        '''
        raise NotImplementedError("Subclasses must implement get_current_frame_index")
    
    def update_display(self):
        '''
        Update the visual display with current data and overlays.
        '''
        raise NotImplementedError("Subclasses must implement update_display")
    
    def display_frame(self, frame_idx):
        '''
        Display a specific frame.
        
        frame_idx: Index of the frame to display
        '''
        raise NotImplementedError("Subclasses must implement display_frame")
    
    def connect_lines_widget(self, lines_widget):
        '''
        Connect signals between this widget and a lines widget.
        
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
        self.lines_widget = lines_widget
    
    def enter_add_mode(self):
        '''
        Enter interactive tool mode with clicks and stuff.
        '''
        raise NotImplementedError("Subclasses must implement enter_add_mode")
    
    def remove_line_overlay(self, index):
        '''
        Remove a given line.

        index: Index of the line to remove
        '''
        raise NotImplementedError("Subclasses must implement remove_line_overlay")
    
    def move_line_endpoint(self, line_index, endpoint_idx, x, y, image_data):
        '''
        Move a line endpoint.
        
        line_index: Index of the line
        endpoint_idx: 0 for first endpoint, 1 for second endpoint
        x, y: New coordinates
        image_data: Current image data
        '''
        raise NotImplementedError("Subclasses must implement move_line_endpoint")
    
    def get_lines(self):
        '''
        Get current line positions and colors.
        
        returns a list of (x1, y1, x2, y2, color) tuples for each line
        '''
        raise NotImplementedError("Subclasses must implement get_lines")
    
    def register_overlay_callback(self, callback):
        '''
        Hold the callback function to be called when overlays need updating.
        
        callback: Function to call with current frame index
        '''
        self.overlay_callbacks.append(callback)
    
    def update_overlays(self):
        '''
        Call all the overlay callbacks (may be more than 1).
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
    
    def get_channel_names(self):
        '''
        Get channel names for legend display in lines widget.
        
        Returns:
            list: List of channel names
        '''
        # Default implementation for single channel
        return ['Channel 1']
