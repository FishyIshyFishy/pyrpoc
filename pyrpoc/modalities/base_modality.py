from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
from pyrpoc.displays.base_display import BaseImageDisplayWidget
from pyrpoc.instruments.base_instrument import Instrument

class AcquisitionContext:
    """Context object containing information needed by displays to prepare for acquisition"""
    
    def __init__(self, modality_key: str, total_frames: int, frame_shape: tuple, 
                 channel_info: dict = None, metadata: dict = None):
        self.modality_key = modality_key
        self.total_frames = total_frames
        self.frame_shape = frame_shape  # (height, width) or (channels, height, width)
        self.channel_info = channel_info or {}
        self.metadata = metadata or {}
    
    def get_frame_dimensions(self):
        """Get the dimensions of a single frame"""
        if len(self.frame_shape) == 2:
            return self.frame_shape  # (height, width)
        elif len(self.frame_shape) == 3:
            return self.frame_shape[1:]  # (height, width) from (channels, height, width)
        return self.frame_shape
    
    def get_channel_count(self):
        """Get the number of channels"""
        if len(self.frame_shape) == 3:
            return self.frame_shape[0]
        return 1

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
    
    def create_acquisition_context(self, parameters: Dict[str, Any]) -> AcquisitionContext:
        """
        Create an acquisition context for the display to prepare for acquisition.
        This method should be implemented by subclasses to provide modality-specific context.
        
        Args:
            parameters: Acquisition parameters from app_state
            
        Returns:
            AcquisitionContext object with information needed by displays
        """
        raise NotImplementedError("Subclasses must implement create_acquisition_context")
    
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
