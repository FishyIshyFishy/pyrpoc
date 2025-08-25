from .base_modality import BaseModality
from pyrpoc.displays.singlechan import ImageDisplayWidget
from pyrpoc.acquisitions.simulated import Simulated
from typing import List, Type, Dict, Any

class SimulatedModality(BaseModality):
    @property
    def name(self) -> str:
        return "Simulated"
    
    @property
    def key(self) -> str:
        return "simulated"
    
    @property
    def required_instruments(self) -> List[str]:
        return []  # No instruments required for simulated modality
    
    @property
    def compatible_displays(self):
        return [ImageDisplayWidget]
    
    @property
    def required_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'x_pixels': {
                'type': 'int',
                'default': 512,
                'range': (64, 4096),
                'description': 'Number of X pixels'
            },
            'y_pixels': {
                'type': 'int',
                'default': 512,
                'range': (64, 4096),
                'description': 'Number of Y pixels'
            }
        }
    
    @property
    def acquisition_class(self) -> Type:
        return Simulated
