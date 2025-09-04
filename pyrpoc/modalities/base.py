from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
from pyrpoc.displays.base_display import BaseImageDisplayWidget
from pyrpoc.instruments.base_instrument import Instrument
import numpy as np
import abc
from pyrpoc.instruments.instrument_manager import *
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass

class BaseModality(ABC):
    """Base class for all acquisition modalities"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the modality"""
        pass
    
    @property
    @abstractmethod
    def key(self) -> str:
        """Internal key used to identify the modality"""
        pass
    
    @property
    @abstractmethod
    def required_instruments(self) -> List[str]:
        """List of required instrument types for this modality"""
        pass
    
    @property
    @abstractmethod
    def compatible_displays(self) -> List[Type[BaseImageDisplayWidget]]:
        """List of display widget classes compatible with this modality"""
        pass
    
    @property
    @abstractmethod
    def required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Dictionary of required parameters with their metadata
        
        Format:
        {
            'param_name': {
                'type': 'int|float|bool|str|choice',
                'default': default_value,
                'range': (min, max),  # for numeric types
                'choices': ['choice1', 'choice2'],  # for choice type
                'unit': 'unit_string',  # optional
                'description': 'Human readable description'
            }
        }
        """
        pass
    
    @property
    def parameter_groups(self) -> Dict[str, List[str]]:
        """Dictionary defining parameter groupings for better UI organization
        
        Format:
        {
            'group_name': ['param1', 'param2', 'param3'],
            'another_group': ['param4', 'param5']
        }
        
        If not overridden, all parameters will be in a single 'General' group.
        """
        # Default implementation: put all parameters in a 'General' group
        return {'General': list(self.required_parameters.keys())}
    
    @property
    @abstractmethod
    def acquisition_class(self) -> Type:
        """The acquisition class to use for this modality"""
        pass
    
    def validate_instruments(self, instruments: List[Instrument]) -> bool:
        """Validate that all required instruments are present"""
        instrument_types = [inst.instrument_type for inst in instruments]
        return all(req_type in instrument_types for req_type in self.required_instruments)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate that all required parameters are present and valid"""
        for param_name, param_meta in self.required_parameters.items():
            if param_name not in parameters:
                return False
            # Add more validation logic here
        return True


class Acquisition(abc.ABC):
    def __init__(self, save_enabled=False, save_path='', **kwargs):
        self._stop_flag = None
        self.save_enabled = save_enabled
        self.save_path = save_path
        self.metadata = {}

    def set_stop_flag(self, stop_flag_func):
        '''
        stop button in main gui for in any given acquisition sets this flag

        in general we do something like:
        if self._stop_flag and self._stop_flag()
        so that we can check if _stop_flag has a callback, and then actually get the callback
        '''
        self._stop_flag = stop_flag_func
    
    def set_worker(self, worker):
        '''
        Set reference to worker for signal emission
        '''
        self.worker = worker

    def emit_data(self, signal_bus, data):
        """
        Emit a data frame using the new uniform pipeline.
        
        Args:
            signal_bus: The StateSignalBus instance for emitting signals
            data: The data frame to emit
        """
        if signal_bus:
            signal_bus.data_received.emit(data)

    def emit_acquisition_complete(self, signal_bus):
        """
        Emit acquisition complete signal using the new uniform pipeline.
        
        Args:
            signal_bus: The StateSignalBus instance for emitting signals
        """
        if signal_bus:
            signal_bus.acquisition_complete.emit()

    def save_metadata(self):
        '''
        Save metadata JSON file to the folder determined by the base filename
        '''
        if not self.save_enabled or not self.save_path:
            return
        
        try:
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'acquisition_type': self.__class__.__name__,
                'timestamp': datetime.now().isoformat(),
                'acquisition_parameters': getattr(self, 'acquisition_parameters', {}),
                'rpoc_enabled': getattr(self, 'rpoc_enabled', False),
                'rpoc_mask_channels': len(getattr(self, 'rpoc_mask_channels', {})),
                'rpoc_static_channels': len(getattr(self, 'rpoc_static_channels', {})),
                'rpoc_script_channels': len(getattr(self, 'rpoc_script_channels', {})),
                'custom_metadata': self.metadata
            }
            
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
        '''
        save data in modality-specific format
        '''
        pass    

@dataclass
class ChannelData: 
    def __init__(self, name, data, metadata):
        self.name = name
        self.data = data
        self.metadata = metadata