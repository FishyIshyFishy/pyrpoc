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


class ZScan(Acquisition):
    def __init__(self, stages=None, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.stages = stages or []
        self.signal_bus = signal_bus
        self.acquisition_parameters = acquisition_parameters or {}
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # Save metadata before starting acquisition
        self.save_metadata()
        
        # TODO: Implement ZScan acquisition
        # For now, return simulated data with frame emission
        frames = []
        for frame in range(1):  # Single frame for now
            if self._stop_flag and self._stop_flag():
                break
            frame_data = np.random.rand(512, 512)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, 1, False)
        
        if frames:
            final_data = frames[0]  # Return single frame
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, 0, 1, True)
            # Save data if enabled
            self.save_data(final_data)
            return final_data
        else:
            return None
    
    def save_data(self, data):
        """
        Save zscan data as a single TIFF file
        data shape: (height, width)
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
                self.signal_bus.console_message.emit(f"Saved zscan data to {filepath}")
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving zscan data: {e}")
            else:
                print(f"Error saving zscan data: {e}")