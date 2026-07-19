"""Hardware control. Each device wraps its SDK and exposes convenience methods.

Importing this package registers every instrument. Real instruments run hardware
only; the simulated ones produce toy data for a hardware-free machine.
"""

from __future__ import annotations

from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.instruments.ni_daq import DaqError, NIDAQ, mask_to_ttl, raster_waveform
from pyrpoc_next.instruments.registry import instrument_registry
from pyrpoc_next.instruments.scanning import RasterScan
from pyrpoc_next.instruments.simulated_daq import SimulatedDAQ, SimulatedTagger
from pyrpoc_next.instruments.stages import PriorStage, Stage, ZaberStage
from pyrpoc_next.instruments.time_tagger import TimeTagger

__all__ = [
    "Instrument",
    "instrument_registry",
    "RasterScan",
    "NIDAQ",
    "DaqError",
    "raster_waveform",
    "mask_to_ttl",
    "SimulatedDAQ",
    "SimulatedTagger",
    "TimeTagger",
    "Stage",
    "PriorStage",
    "ZaberStage",
]
