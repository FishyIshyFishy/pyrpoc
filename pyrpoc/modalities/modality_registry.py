from typing import Dict, List, Optional
from .base_modality import BaseModality

class ModalityRegistry:
    """Central registry for all available modalities"""
    
    def __init__(self):
        self._modalities: Dict[str, BaseModality] = {}
    
    def register(self, modality: BaseModality):
        """Register a new modality"""
        self._modalities[modality.key] = modality
    
    def get_modality(self, key: str) -> Optional[BaseModality]:
        """Get a modality by its key"""
        return self._modalities.get(key)
    
    def get_all_modalities(self) -> List[BaseModality]:
        """Get all registered modalities"""
        return list(self._modalities.values())
    
    def get_modality_names(self) -> List[str]:
        """Get list of modality names for UI dropdowns"""
        return [mod.name for mod in self._modalities.values()]
    
    def get_modality_keys(self) -> List[str]:
        """Get list of modality keys"""
        return list(self._modalities.keys())
    
    def get_modality_by_name(self, name: str) -> Optional[BaseModality]:
        """Get a modality by its display name"""
        for modality in self._modalities.values():
            if modality.name == name:
                return modality
        return None
