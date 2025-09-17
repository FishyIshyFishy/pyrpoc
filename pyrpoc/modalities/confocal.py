from .base_modality import BaseModality
from .mod_registry import modality_registry
from pyrpoc.backend_utils.data import DataImage
from pyrpoc.instruments import BaseInstrument, Galvo, DAQInput
from pyrpoc.gui.signals import AcquisitionSignals

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Type, List, Dict, Any
import nidaqmx
import numpy as np

@modality_registry.register('confocal')
class ConfocalModality(BaseModality):
    REQUIRED_PARAMETERS: Dict[str, Dict[str, Dict[str, Any]]] = {
        'scan': {
            'x_pixels':      {'type': int,   'default': 512,  'min': 1, 'tooltip': 'Number of pixels in X'},
            'y_pixels':      {'type': int,   'default': 512,  'min': 1, 'tooltip': 'Number of pixels in Y'},
            'extra_left':    {'type': int,   'default': 50,   'min': 0, 'tooltip': 'Pre-scan padding in X'},
            'extra_right':   {'type': int,   'default': 50,   'min': 0, 'tooltip': 'Post-scan padding in X'},
            'dwell_time': {'type': float, 'default': 10.0, 'min': 0.1, 'tooltip': 'Dwell time per pixel (µs)'},
            'frames':        {'type': int,   'default': 1,    'min': 1, 'tooltip': 'Number of frames to acquire'},
        }, # i think i want savestuff always present, but i need to keep that implementation standardized
    }

    REQUIRED_INSTRUMENTS: List[Type[BaseInstrument]] = [
        Galvo,
        DAQInput
    ]

    ALLOWED_DISPLAYS: List[str] = [
        'tiled_image',
        'stacked_image'
    ]

    DATA_TYPE = DataImage
    def __init__(self, galvo: Galvo, inputs: list[DAQInput], acq_signals: AcquisitionSignals, 
                 num_frames: int, x_pixels: int, y_pixels: int, extrapixels_left: int, extrapixels_right: int, dwell_time: float, 
                 save_enabled: bool, save_path: str):
        super().__init__(name='confocal', data_type=self.DATA_TYPE)

    def do_something(self):
        print('do something')

