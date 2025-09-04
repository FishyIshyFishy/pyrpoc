import numpy as np
import nidaqmx
import abc
from pyrpoc.instruments.instrument_manager import *
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
import tifffile
from pathlib import Path
from .base_acquisition import Acquisition
from datetime import datetime

class Simulated(Acquisition):
    def __init__(self, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.signal_bus = signal_bus
        self.acquisition_parameters = acquisition_parameters or {}
        
        # Extract parameters from acquisition_parameters dict
        self.x_pixels = self.acquisition_parameters.get('x_pixels', 512)
        self.y_pixels = self.acquisition_parameters.get('y_pixels', 512)
        self.num_frames = self.acquisition_parameters.get('num_frames', 1)

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # Save metadata before starting acquisition
        self.save_metadata()
        

        
        frames = []
        for frame in range(self.num_frames):
            if self._stop_flag and self._stop_flag():
                break
            
            frame_data = self.generate_simulated_frame()
            frames.append(frame_data)
            
            if self.signal_bus:
                # Use new uniform pipeline instead of legacy data_signal
                self.emit_data(self.signal_bus, frame_data)
        
        if frames:
            final_data = np.stack(frames)
            if self.signal_bus:
                # Use new uniform pipeline instead of legacy data_signal
                self.emit_acquisition_complete(self.signal_bus)
            return final_data
        else:
            return None

    def generate_simulated_frame(self):
        """
        Generate a simulated frame with random data.
        
        Returns:
            numpy.ndarray: Random frame data with shape (y_pixels, x_pixels)
        """
        frame_data = np.random.rand(self.y_pixels, self.x_pixels)
        return frame_data
    
    def save_data(self, data):
        """
        Save simulated data as a single TIFF file
        data shape: (num_frames, height, width) or (height, width)
        """
        if not self.save_enabled or not self.save_path:
            return
        
        try:
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Path(self.save_path).stem}_{timestamp}.tiff"
            filepath = save_dir / filename
            
            # Save as TIFF
            tifffile.imwrite(filepath, data)
            
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Saved simulated data to {filepath}")
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving simulated data: {e}")
    