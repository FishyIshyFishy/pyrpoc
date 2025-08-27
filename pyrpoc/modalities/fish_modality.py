from .base_modality import BaseModality
from pyrpoc.displays.multichan_tiled import MultichannelImageDisplayWidget
from pyrpoc.acquisitions.fish import Fish
from typing import List, Type, Dict, Any

class FishModality(BaseModality):
    @property
    def name(self) -> str:
        return "Fish"
    
    @property
    def key(self) -> str:
        return "fish"
    
    @property
    def required_instruments(self):
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
            'Local RPOC': ['local_rpoc_dwell_time', 'local_extrasteps_left', 'local_extrasteps_right'],
            'Treatment': ['repetitions', 'offset_drift_x', 'offset_drift_y'],
            'TTL Configuration': ['ttl_device', 'ttl_port_line', 'pfi_line']
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
            },
            'local_rpoc_dwell_time': {
                'type': 'int',
                'default': 10,
                'range': (1, 10000),
                'unit': 'μs',
                'description': 'Local RPOC dwell time'
            },
            'offset_drift_x': {
                'type': 'float',
                'default': 0.0,
                'range': (-10.0, 10.0),
                'unit': 'V',
                'description': 'Drift offset for X axis'
            },
            'offset_drift_y': {
                'type': 'float',
                'default': 0.0,
                'range': (-10.0, 10.0),
                'unit': 'V',
                'description': 'Drift offset for Y axis'
            },
            'repetitions': {
                'type': 'int',
                'default': 1,
                'range': (1, 1000),
                'description': 'Number of treatment repetitions'
            },
            'ttl_device': {
                'type': 'choice',
                'default': 'Dev1',
                'choices': ['Dev1', 'Dev2', 'Dev3', 'Dev4'],
                'description': 'TTL device for local RPOC'
            },
            'ttl_port_line': {
                'type': 'choice',
                'default': 'port0/line0',
                'choices': [
                    'port0/line0', 'port0/line1', 'port0/line2', 'port0/line3', 'port0/line4', 'port0/line5', 'port0/line6', 'port0/line7',
                    'port0/line8', 'port0/line9', 'port0/line10', 'port0/line11', 'port0/line12', 'port0/line13', 'port0/line14', 'port0/line15',
                    'port1/line0', 'port1/line1', 'port1/line2', 'port1/line3', 'port1/line4', 'port1/line5', 'port1/line6', 'port1/line7',
                    'port1/line8', 'port1/line9', 'port1/line10', 'port1/line11', 'port1/line12', 'port1/line13', 'port1/line14', 'port1/line15'
                ],
                'description': 'TTL port/line for local RPOC'
            },
            'pfi_line': {
                'type': 'choice',
                'default': 'None',
                'choices': ['None', 'PFI0', 'PFI1', 'PFI2', 'PFI3', 'PFI4', 'PFI5', 'PFI6', 'PFI7', 'PFI8', 'PFI9', 'PFI10', 'PFI11', 'PFI12', 'PFI13', 'PFI14', 'PFI15'],
                'description': 'PFI line for timing (optional)'
            },
            'local_extrasteps_left': {
                'type': 'int',
                'default': 50,
                'range': (0, 1000),
                'description': 'Local RPOC extra steps left'
            },
            'local_extrasteps_right': {
                'type': 'int',
                'default': 50,
                'range': (0, 1000),
                'description': 'Local RPOC extra steps right'
            }
        }
    
    @property
    def acquisition_class(self) -> Type:
        return Fish
