"""Acquisition modalities. Importing registers every concrete modality."""

from __future__ import annotations

from pyrpoc_next.acquisition.modalities.base import Modality, modality_registry
from pyrpoc_next.acquisition.modalities.scanner import ScannerModality
from pyrpoc_next.acquisition.modalities.confocal import ConfocalModality
from pyrpoc_next.acquisition.modalities.split_confocal import SplitConfocalModality
from pyrpoc_next.acquisition.modalities.flim import FlimModality
from pyrpoc_next.acquisition.modalities.simulated import SimulatedFlimModality, SimulatedModality

__all__ = [
    "Modality",
    "modality_registry",
    "ScannerModality",
    "ConfocalModality",
    "SplitConfocalModality",
    "FlimModality",
    "SimulatedModality",
    "SimulatedFlimModality",
]
