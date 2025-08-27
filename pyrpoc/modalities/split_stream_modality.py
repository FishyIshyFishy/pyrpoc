from .base_modality import BaseModality, AcquisitionContext
from pyrpoc.displays.multichan_tiled import MultichannelImageDisplayWidget
from pyrpoc.acquisitions.split_stream import SplitDataStream
from typing import List, Type, Dict, Any

class SplitStreamModality(BaseModality):
    @property
    def name(self) -> str:
        return "Split data stream"
    
    @property
    def key(self) -> str:
        return "split data stream"
    
    @property
    def required_instruments(self) -> List[str]:
        return ["galvo", "data input", "prior stage"]
    
    @property
    def compatible_displays(self):
        return [MultichannelImageDisplayWidget]
    
    @property
    def parameter_groups(self) -> Dict[str, List[str]]:
        return {
            'Image Dimensions': ['x_pixels', 'y_pixels'],
            'Scanning': ['dwell_time', 'extrasteps_left', 'extrasteps_right'],
            'Galvo Control': ['amplitude_x', 'amplitude_y', 'offset_x', 'offset_y'],
            'Tiling': ['numtiles_x', 'numtiles_y', 'numtiles_z'],
            'Tile Sizes': ['tile_size_x', 'tile_size_y', 'tile_size_z'],
            'Split Stream': ['split_percentage', 'aom_delay']
        }
    
    @property
    def required_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'split_percentage': {
                'type': 'int',
                'default': 50,
                'range': (1, 99),
                'unit': '%',
                'description': 'Split percentage for data stream'
            },
            'aom_delay': {
                'type': 'int',
                'default': 0,
                'range': (0, 1000),
                'unit': 'μs',
                'description': 'AOM delay in microseconds'
            },
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
            },
            'numtiles_x': {
                'type': 'int',
                'default': 10,
                'range': (1, 1000),
                'description': 'Number of X tiles'
            },
            'numtiles_y': {
                'type': 'int',
                'default': 10,
                'range': (1, 1000),
                'description': 'Number of Y tiles'
            },
            'numtiles_z': {
                'type': 'int',
                'default': 5,
                'range': (1, 1000),
                'description': 'Number of Z tiles'
            },
            'tile_size_x': {
                'type': 'float',
                'default': 100.0,
                'range': (-10000, 10000),
                'unit': 'μm',
                'description': 'X tile size'
            },
            'tile_size_y': {
                'type': 'float',
                'default': 100.0,
                'range': (-10000, 10000),
                'unit': 'μm',
                'description': 'Y tile size'
            },
            'tile_size_z': {
                'type': 'float',
                'default': 50.0,
                'range': (-10000, 10000),
                'unit': 'μm',
                'description': 'Z tile size'
            }
        }
    
    @property
    def acquisition_class(self) -> Type:
        return SplitDataStream

    def create_acquisition_context(self, parameters: Dict[str, Any]) -> AcquisitionContext:
        """Create acquisition context for Split Data Stream modality.
        Expects interleaved or paired channel outputs; displays can show 2x inputs per physical input.
        """
        total_frames = int(parameters.get('num_frames', 1))
        x_pixels = int(parameters.get('x_pixels', 512))
        y_pixels = int(parameters.get('y_pixels', 512))

        # Default to 2 channels to represent split portions for one input.
        # If more inputs are active, display can expand based on instruments.
        channel_count = 2
        channel_info = {
            'count': channel_count,
            'names': [f'Channel {i+1}' for i in range(channel_count)]
        }
        frame_shape = (channel_count, y_pixels, x_pixels)

        metadata = {
            'split_stream': {
                'split_percentage': parameters.get('split_percentage'),
                'aom_delay': parameters.get('aom_delay'),
            }
        }

        return AcquisitionContext(
            modality_key=self.key,
            total_frames=total_frames,
            frame_shape=frame_shape,
            channel_info=channel_info,
            metadata=metadata
        )
