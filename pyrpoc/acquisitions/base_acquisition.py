import numpy as np
import nidaqmx
import abc
from pyrpoc.instruments.instrument_manager import *
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
import tifffile
from pathlib import Path

class Acquisition(abc.ABC):
    def __init__(self, save_enabled=False, save_path='', **kwargs):
        self._stop_flag = None
        self.save_enabled = save_enabled
        self.save_path = save_path

    def set_stop_flag(self, stop_flag_func):
        '''
        stop button in main gui for in any given acquisition sets this flag
        '''
        self._stop_flag = stop_flag_func
    
    def set_worker(self, worker):
        '''
        Set reference to worker for signal emission
        '''
        self.worker = worker

    @abc.abstractmethod
    def configure_rpoc(self, rpoc_enabled, **kwargs):
        '''
        check global rpoc flag (whatever that ends up being)
        if not, set up rpoc for the acquisition process
        '''

    @abc.abstractmethod
    def perform_acquisition(self): 
        '''
        yield each lowest-level data unit (e.g., a single image, a single tile, etc.) as it is acquired, and finally return a list or array of all such data units
        '''
        pass
    
    def save_data(self, data):
        if not self.save_enabled or not self.save_path:
            return
        
        try:            
            save_dir = Path(self.save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = self.save_path
            if not save_path.lower().endswith(('.tiff', '.tif')):
                save_path = save_path + '.tiff'

            tifffile.imwrite(save_path, data)
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving data: {e}")    