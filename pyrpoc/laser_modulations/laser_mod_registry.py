from pyrpoc.backend_utils.registry import Registry
from .base_laser_mod import BaseLaserModulation

class ModalityRegistry(Registry):
    def __init__(self):
        super().__init__(name='ModalityRegistry', base_class=BaseLaserModulation)

modality_registry = ModalityRegistry()