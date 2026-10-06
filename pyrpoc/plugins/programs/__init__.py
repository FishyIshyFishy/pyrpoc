"""Experiments, one folder each, holding the program and the code only it
needs. Everything shared (parameter groups, runners, and functions like the
galvo raster) is in ``building_blocks/``. Programs build from those and never
import each other. Importing this package registers every program.
"""

from __future__ import annotations

from pyrpoc.structs.plugins.programs.program import program_registry

from .confocal.program import Confocal
from .dual_galvo_confocal.program import DualGalvoConfocal
from .flim.program import FLIM
from .mosaic.program import Mosaic
from .mosaic_simulation.program import MosaicSimulation
from .pinpoint_raman.program import PinpointRaman
from .simulation.program import Simulation
from .split_confocal.program import SplitConfocal

__all__ = [
    "program_registry",
    "Confocal",
    "SplitConfocal",
    "DualGalvoConfocal",
    "FLIM",
    "Mosaic",
    "MosaicSimulation",
    "PinpointRaman",
    "Simulation",
]
