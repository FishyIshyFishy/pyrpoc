from .base_modality import BaseModality
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
