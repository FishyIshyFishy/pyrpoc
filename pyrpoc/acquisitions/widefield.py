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

class Widefield(Acquisition):
    def __init__(self, data_inputs=None, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.data_inputs = data_inputs or []
        self.signal_bus = signal_bus
        self.acquisition_parameters = acquisition_parameters or {}
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # Save metadata before starting acquisition
        self.save_metadata()
        
        # Get parameters from acquisition_parameters
        x_pixels = self.acquisition_parameters.get('x_pixels', 512)
        y_pixels = self.acquisition_parameters.get('y_pixels', 512)
        num_frames = self.acquisition_parameters.get('num_frames', 1)
        
        # TODO: Implement widefield acquisition
        # For now, return simulated data with frame emission
        frames = []
        for frame in range(num_frames):
            if self._stop_flag and self._stop_flag():
                break
            frame_data = np.random.rand(y_pixels, x_pixels)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, num_frames, False)
        
        if frames:
            if num_frames > 1:
                final_data = np.stack(frames)
            else:
                final_data = frames[0]
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(frames)-1, num_frames, True)
            # Save data if enabled
            self.save_data(final_data)
            return final_data
        else:
            return None
    
    def save_data(self, data):
        """
        Save widefield data as a single TIFF file
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
                self.signal_bus.console_message.emit(f"Saved widefield data to {filepath}")
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving widefield data: {e}")
            else:
                print(f"Error saving widefield data: {e}")