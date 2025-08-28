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
    
    def prepare_for_acquisition(self, acquisition_context):
        """
        Prepare the display widget for acquisition using the provided context.
        This replaces the old total_frames parameter approach.
        
        Args:
            acquisition_context: AcquisitionContext object containing acquisition information
        """
        # Store the context for reference
        self.acquisition_context = acquisition_context
        
        # Update internal state
        self.total_frames = acquisition_context.total_frames
        self.current_frame = 0
        
        # Clear acquisition buffer
        self.acq_buffer = None
        self.acq_total = acquisition_context.total_frames
        
        # Initialize frame counter
        self._current_frame_idx = 0
        
        # Store frame shape information
        self.frame_shape = acquisition_context.frame_shape
        self.channel_info = acquisition_context.channel_info
        
        # Subclasses can override this for more specific preparation
        self._prepare_for_acquisition_impl(acquisition_context)
    
    def _prepare_for_acquisition_impl(self, acquisition_context):
        """
        Subclass-specific preparation logic. Override this method for custom preparation.
        
        Args:
            acquisition_context: AcquisitionContext object
        """
        # Default implementation does nothing
        pass
    
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
