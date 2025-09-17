from pyrpoc.backend_utils.registry import Registry
from .base_modality import BaseModality

class ModalityRegistry(Registry):
    def __init__(self):
        super().__init__(name='ModalityRegistry', base_class=BaseModality)

modality_registry = ModalityRegistry()

