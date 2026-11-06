"""Experiments, one folder each, holding the program and the code only it needs. Importing
this package registers every program."""

# Everything shared is in ``building_blocks/``; programs never import each other.

from __future__ import annotations

from pyrpoc.core.program import program_registry

from .autofocus.program import Autofocus
from .brightfield.program import Brightfield
from .confocal.program import Confocal
from .flim.program import FLIM
from .galvo_pair_imaging.program import GalvoPairImaging
from .galvo_pair_pinpoint.program import GalvoPairPinpoint
from .mosaic.program import StageMosaic
from .raman.program import Raman
from .split_confocal.program import SplitConfocal

__all__ = [
    "program_registry",
    "Confocal",
    "SplitConfocal",
    "FLIM",
    "GalvoPairImaging",
    "GalvoPairPinpoint",
    "StageMosaic",
    "Autofocus",
    "Raman",
    "Brightfield",
]
