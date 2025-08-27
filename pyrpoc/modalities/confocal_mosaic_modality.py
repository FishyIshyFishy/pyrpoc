from .base_modality import BaseModality, AcquisitionContext
from pyrpoc.displays.multichan_tiled import MultichannelImageDisplayWidget
from pyrpoc.acquisitions.confocal_mosaic import ConfocalMosaic
from typing import List, Type, Dict, Any

class ConfocalMosaicModality(BaseModality):
    @property
    def name(self) -> str:
        return "Confocal mosaic"
    
    @property
    def key(self) -> str:
        return "confocal mosaic"
    
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
            'Tile Sizes': ['tile_size_x', 'tile_size_y', 'tile_size_z']
        }
    
    @property
    def required_parameters(self) -> Dict[str, Dict[str, Any]]:
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
        return ConfocalMosaic

    def create_acquisition_context(self, parameters: Dict[str, Any]) -> AcquisitionContext:
        """Create acquisition context for confocal mosaic imaging.
        Each data unit is a tile image; the display treats it as a frame.
        """
        # total frames equals frames per acquisition times number of tile positions
        num_frames = int(parameters.get('num_frames', 1))
        numtiles_x = int(parameters.get('numtiles_x', 1))
        numtiles_y = int(parameters.get('numtiles_y', 1))
        numtiles_z = int(parameters.get('numtiles_z', 1))
        total_frames = max(1, num_frames * numtiles_x * numtiles_y * numtiles_z)
        x_pixels = int(parameters.get('x_pixels', 512))
        y_pixels = int(parameters.get('y_pixels', 512))

        # Channel count depends on data inputs; default 1
        channel_count = 1
        channel_info = {
            'count': channel_count,
            'names': [f'Channel {i+1}' for i in range(channel_count)]
        }
        frame_shape = (y_pixels, x_pixels) if channel_count == 1 else (channel_count, y_pixels, x_pixels)

        metadata = {
            'tiling': {
                'numtiles_x': numtiles_x,
                'numtiles_y': numtiles_y,
                'numtiles_z': numtiles_z,
                'tile_size_x': parameters.get('tile_size_x'),
                'tile_size_y': parameters.get('tile_size_y'),
                'tile_size_z': parameters.get('tile_size_z'),
            }
        }

        return AcquisitionContext(
            modality_key=self.key,
            total_frames=total_frames,
            frame_shape=frame_shape,
            channel_info=channel_info,
            metadata=metadata
        )
