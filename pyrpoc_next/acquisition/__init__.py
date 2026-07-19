"""The acquisition subsystem: how data actually gets acquired.

Runs one configured modality via the shared async runner, emitting parcels. Knows
nothing of routines or the GUI.
"""

from __future__ import annotations

from pyrpoc_next.acquisition.modalities import (
    ConfocalModality,
    FlimModality,
    Modality,
    ScannerModality,
    SimulatedFlimModality,
    SimulatedModality,
    SplitConfocalModality,
    modality_registry,
)
from pyrpoc_next.acquisition.modifiers import MaskModifier, Modifier, modifier_registry
from pyrpoc_next.acquisition.runner import RunContext, Runner

__all__ = [
    "Modality",
    "ScannerModality",
    "ConfocalModality",
    "SplitConfocalModality",
    "FlimModality",
    "SimulatedModality",
    "SimulatedFlimModality",
    "modality_registry",
    "Modifier",
    "MaskModifier",
    "modifier_registry",
    "RunContext",
    "Runner",
]
