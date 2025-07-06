import numpy as np
import nidaqmx
import abc
from pyrpoc.instruments.instrument_manager import *
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
import tifffile
from pathlib import Path
import json
from datetime import datetime

class Acquisition(abc.ABC):
    def __init__(self, save_enabled=False, save_path='', **kwargs):
        self._stop_flag = None
        self.save_enabled = save_enabled
        self.save_path = save_path
        self.metadata = {}

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

    def save_metadata(self):
        """
        Save metadata JSON file to the folder determined by the base filename
        """
        if not self.save_enabled or not self.save_path:
            return
        
        try:
            # Get the directory from the save_path
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata with ALL acquisition parameters
            metadata = {
                'acquisition_type': self.__class__.__name__,
                'timestamp': datetime.now().isoformat(),
                'acquisition_parameters': getattr(self, 'acquisition_parameters', {}),
                'rpoc_enabled': getattr(self, 'rpoc_enabled', False),
                'rpoc_masks': len(getattr(self, 'rpoc_masks', {})),
                'rpoc_channels': len(getattr(self, 'rpoc_channels', {})),
                'custom_metadata': self.metadata
            }
            
            # Save metadata to JSON file
            metadata_path = save_dir / f"{Path(self.save_path).stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            if hasattr(self, 'signal_bus') and self.signal_bus:
                self.signal_bus.console_message.emit(f"Metadata saved to {metadata_path}")
                
        except Exception as e:
            if hasattr(self, 'signal_bus') and self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving metadata: {e}")



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
    
    @abc.abstractmethod
    def save_data(self, data):
        """
        Save data in modality-specific format. Each modality should implement this method.
        """
        pass    