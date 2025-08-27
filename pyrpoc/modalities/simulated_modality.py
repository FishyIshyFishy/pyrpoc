from .base_modality import BaseModality, AcquisitionContext
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
    def parameter_groups(self) -> Dict[str, List[str]]:
        return {
            'Image Dimensions': ['x_pixels', 'y_pixels']
        }
    
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

    def create_acquisition_context(self, parameters: Dict[str, Any]) -> AcquisitionContext:
        """
        Provide modality-agnostic context for displays.
        Simulated frames are single-channel 2D arrays of size (y_pixels, x_pixels).
        """
        total_frames = int(parameters.get('num_frames', 1))
        x_pixels = int(parameters.get('x_pixels', 512))
        y_pixels = int(parameters.get('y_pixels', 512))

        frame_shape = (y_pixels, x_pixels)
        channel_info = {
            'count': 1,
            'names': ['Channel 1']
        }
        metadata = {}

        return AcquisitionContext(
            modality_key=self.key,
            total_frames=total_frames,
            frame_shape=frame_shape,
            channel_info=channel_info,
            metadata=metadata
        )
