from .modality_registry import ModalityRegistry
from .base_modality import BaseModality
from .confocal_modality import ConfocalModality
from .fish_modality import FishModality
from .simulated_modality import SimulatedModality
from .split_stream_modality import SplitStreamModality
from .confocal_mosaic_modality import ConfocalMosaicModality

# Global registry instance
modality_registry = ModalityRegistry()

# Register all modalities
modality_registry.register(ConfocalModality())
modality_registry.register(FishModality())
modality_registry.register(SimulatedModality())
modality_registry.register(SplitStreamModality())
modality_registry.register(ConfocalMosaicModality())
