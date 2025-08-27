from .base_modality import BaseModality, AcquisitionContext
from pyrpoc.displays.multichan_tiled import MultichannelImageDisplayWidget
from pyrpoc.acquisitions.confocal import Confocal
from typing import List, Type, Dict, Any
from .base_modality import AcquisitionContext

class ConfocalModality(BaseModality):
    @property
    def name(self) -> str:
        return "Confocal"
    
    @property
    def key(self) -> str:
        return "confocal"
    
    @property
    def required_instruments(self):
        return ["galvo", "data input"]
    
    @property
    def compatible_displays(self):
        return [MultichannelImageDisplayWidget]
    
    @property
    def parameter_groups(self) -> Dict[str, List[str]]:
        return {
            'Image Dimensions': ['x_pixels', 'y_pixels'],
            'Scanning': ['dwell_time', 'extrasteps_left', 'extrasteps_right'],
            'Galvo Control': ['amplitude_x', 'amplitude_y', 'offset_x', 'offset_y']
        }
    
    @property
    def required_parameters(self):
        return {
            'dwell_time': {
                'type': 'float',
                'default': 10.0,
                'range': (1.0, 1000.0),
                'unit': 'μs',
                'description': 'Per pixel dwell time'
            },
            'extrasteps_left': {
                'type': 'int',
                'default': 50,
                'range': (0, 10000),
                'description': 'Extra steps left in fast direction'
            },
            'extrasteps_right': {
                'type': 'int',
                'default': 50,
                'range': (0, 10000),
                'description': 'Extra steps right in fast direction'
            },
            'amplitude_x': {
                'type': 'float',
                'default': 0.5,
                'range': (0.0, 10.0),
                'unit': 'V',
                'description': 'Amplitude for X axis'
            },
            'amplitude_y': {
                'type': 'float',
                'default': 0.5,
                'range': (0.0, 10.0),
                'unit': 'V',
                'description': 'Amplitude for Y axis'
            },
            'offset_x': {
                'type': 'float',
                'default': 0.0,
                'range': (-10.0, 10.0),
                'unit': 'V',
                'description': 'Offset for X axis'
            },
            'offset_y': {
                'type': 'float',
                'default': 0.0,
                'range': (-10.0, 10.0),
                'unit': 'V',
                'description': 'Offset for Y axis'
            },
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
        return Confocal

    def create_acquisition_context(self, parameters: Dict[str, Any]) -> AcquisitionContext:
        """Create acquisition context for confocal imaging.
        Data is typically 2D (H, W) per channel; multiple channels possible.
        """
        total_frames = int(parameters.get('num_frames', 1))
        x_pixels = int(parameters.get('x_pixels', 512))
        y_pixels = int(parameters.get('y_pixels', 512))

        # Channel count may depend on connected data input instruments; fall back to 1
        channel_count = 1
        # The display can refine names later from instruments
        channel_info = {
            'count': channel_count,
            'names': [f'Channel {i+1}' for i in range(channel_count)]
        }

        # If multiple channels, shape is (C, H, W); else (H, W)
        frame_shape = (y_pixels, x_pixels) if channel_count == 1 else (channel_count, y_pixels, x_pixels)

        metadata = {
            'dwell_time': parameters.get('dwell_time'),
            'extrasteps_left': parameters.get('extrasteps_left'),
            'extrasteps_right': parameters.get('extrasteps_right'),
            'amplitude_x': parameters.get('amplitude_x'),
            'amplitude_y': parameters.get('amplitude_y'),
            'offset_x': parameters.get('offset_x'),
            'offset_y': parameters.get('offset_y'),
        }

        return AcquisitionContext(
            modality_key=self.key,
            total_frames=total_frames,
            frame_shape=frame_shape,
            channel_info=channel_info,
            metadata=metadata
        )

    def create_acquisition_context(self, parameters: Dict[str, Any]) -> AcquisitionContext:
        """Create acquisition context for confocal modality"""
        # Calculate total frames (for confocal, this is typically 1)
        total_frames = parameters.get('num_frames', 1)
        
        # Get frame dimensions from parameters
        x_pixels = parameters.get('x_pixels', 512)
        y_pixels = parameters.get('y_pixels', 512)
        
        # For confocal, we typically have multiple channels from data input
        # This could be determined by the actual data input instrument
        # For now, we'll assume a default channel count
        channel_count = 1  # This could be made dynamic based on instrument configuration
        
        if channel_count == 1:
            frame_shape = (y_pixels, x_pixels)
        else:
            frame_shape = (channel_count, y_pixels, x_pixels)
        
        # Create channel info
        channel_info = {
            'count': channel_count,
            'names': [f'Channel {i+1}' for i in range(channel_count)]
        }
        
        # Create metadata
        metadata = {
            'dwell_time': parameters.get('dwell_time'),
            'scan_parameters': {
                'extrasteps_left': parameters.get('extrasteps_left'),
                'extrasteps_right': parameters.get('extrasteps_right'),
                'amplitude_x': parameters.get('amplitude_x'),
                'amplitude_y': parameters.get('amplitude_y'),
                'offset_x': parameters.get('offset_x'),
                'offset_y': parameters.get('offset_y')
            }
        }
        
        return AcquisitionContext(
            modality_key=self.key,
            total_frames=total_frames,
            frame_shape=frame_shape,
            channel_info=channel_info,
            metadata=metadata
        )
