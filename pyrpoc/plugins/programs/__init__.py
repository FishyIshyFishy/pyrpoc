"""One file per experiment, plus the components they are assembled from.

A program's scan code lives in its own file, since that code is what the
modality is. Programs running similar scans hold copies of the waveform
arithmetic that are meant to stay identical: change one, change the others.
"""

from __future__ import annotations

from pyrpoc.structs.program import program_registry

from .confocal import Confocal
from .flim import FLIM
from .pinpoint_raman import PinpointRaman
from .simulation import Simulation
from .split_confocal import SplitConfocal

__all__ = [
    "program_registry",
    "Confocal",
    "SplitConfocal",
    "FLIM",
    "PinpointRaman",
    "Simulation",
]
