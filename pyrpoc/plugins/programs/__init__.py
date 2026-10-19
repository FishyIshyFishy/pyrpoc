"""Experiments, one folder each, holding the program and the code only it
needs. Everything shared (parameter groups, runners, and functions like the
galvo raster) is in ``building_blocks/``. Programs build from those and never
import each other. Importing this package registers every program.
"""

from __future__ import annotations

from pyrpoc.structs.plugins.programs.program import program_registry

from .autofocus.program import Autofocus
from .confocal.program import Confocal
from .flim.program import FLIM
from .galvo_pinpoint.program import GalvoPinpoint
from .mosaic.program import Mosaic
from .mosaic_simulation.program import MosaicSimulation
from .pinpoint_raman.program import PinpointRaman
from .raman.program import Raman
from .roi_spectral_series.program import RoiSpectralSeries
from .simulation.program import Simulation
from .split_confocal.program import SplitConfocal

__all__ = [
    "program_registry",
    "Confocal",
    "SplitConfocal",
    "FLIM",
    "GalvoPinpoint",
    "Mosaic",
    "Autofocus",
    "MosaicSimulation",
    "PinpointRaman",
    "Raman",
    "RoiSpectralSeries",
    "Simulation",
]
