import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem, QGridLayout
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont

class BaseImageDisplayWidget(QWidget):
    '''
    defines the interface that all image display widgets must implement to work with the acquisition system and dockable widgets

    some image widgets will need to say no to the widgets, like raise a statusbar update (lines incompatible with zscan)
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
        self._acq_buffer = None
        self._acq_total = None
        
        # Overlay callbacks for future extensibility
        self.overlay_callbacks = []
        
        # Connect to unified data signal
        self.signals.data_signal.connect(self.handle_data_signal)
    
    def handle_data_signal(self, data, idx, total, is_final):
        '''
        Unified handler for both frame updates and final data
        
        data: The data unit (could be single frame or complete dataset)
        idx: Index of the data unit
        total: Total number of expected data units
        is_final: True if this is the final data signal, False for frame updates
        '''
        if is_final:
            # This is the final data signal - handle as completion
            self.handle_data_updated(data)
        else:
            # This is a frame update during acquisition
            self.handle_frame_acquired(data, idx, total)
    
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
        print(f"Connecting lines widget to {self.__class__.__name__}")
        
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
        
        print(f"Lines widget connection completed for {self.__class__.__name__}")
    
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
    
    def get_channel_names(self):
        '''
        Get channel names for legend display in lines widget.
        
        Returns:
            list: List of channel names
        '''
        # Default implementation for single channel
        return ['Channel 1']
