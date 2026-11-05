"""The types of program output programs fill, each with the files it is saved in. Importing
this package registers them all."""

from __future__ import annotations

from .focus_stack import FocusResult, FocusStackOutput, FocusStep
from .histogram import HistogramOutput
from .image import ImageOutput
from .mosaic import MosaicOutput, MosaicTile
from .samples import SamplesOutput
from .spectrum import SpectrumOutput
from .spectrum_series import SpectrumSeriesOutput

__all__ = [
    "FocusResult",
    "FocusStackOutput",
    "FocusStep",
    "HistogramOutput",
    "ImageOutput",
    "MosaicOutput",
    "MosaicTile",
    "SamplesOutput",
    "SpectrumOutput",
    "SpectrumSeriesOutput",
]
