"""Experiments, one folder each, holding the program and the code only it needs. Importing
this package registers every program."""

# Everything shared is in ``building_blocks/``; programs never import each other.

from __future__ import annotations

from pyrpoc.core.program import program_registry

from .autofocus.program import Autofocus
from .brightfield.program import Brightfield
from .confocal.program import Confocal
from .flim.program import FLIM
from .galvo_pinpoint.program import GalvoPinpoint
from .mosaic.program import StageMosaic
from .pinpoint_raman.program import PinpointRaman
from .raman.program import Raman
from .roi_spectral_series.program import RoiSpectralSeries
from .split_confocal.program import SplitConfocal

__all__ = [
    "program_registry",
    "Confocal",
    "SplitConfocal",
    "FLIM",
    "GalvoPinpoint",
    "StageMosaic",
    "Autofocus",
    "PinpointRaman",
    "Raman",
    "RoiSpectralSeries",
    "Brightfield",
]
