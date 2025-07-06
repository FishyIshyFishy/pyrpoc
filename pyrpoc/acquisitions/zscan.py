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


class ZScan(Acquisition):
    def __init__(self, stages=None, signal_bus=None, **kwargs):
        super().__init__(**kwargs)
        self.stages = stages or []
        self.signal_bus = signal_bus
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
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