from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
from pyrpoc.displays.base_display import BaseImageDisplayWidget
from pyrpoc.instruments.base_instrument import Instrument

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
