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

class Simulated(Acquisition):
    def __init__(self, x_pixels: int, y_pixels: int, num_frames: int, signal_bus=None, **kwargs):
        super().__init__(**kwargs)
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.num_frames = num_frames
        self.signal_bus = signal_bus

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        frames = []
        for frame in range(self.num_frames):
            if self._stop_flag and self._stop_flag():
                break
                
            frame_data = np.random.rand(self.y_pixels, self.x_pixels)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, self.num_frames, False)
            time.sleep(1)
        
        if frames:
            final_data = np.stack(frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(frames)-1, self.num_frames, True)

            self.save_data(final_data)
            return final_data
        else:
            return None
    