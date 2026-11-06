"""The types of program output programs fill, each with the files it is saved in. Importing
this package registers them all."""

from __future__ import annotations

from .focus_stack import FocusResult, FocusStackOutput, FocusStep
from .galvo_calibration import CalibrationPoint, GalvoCalibration, GalvoCalibrationOutput
from .histogram import HistogramOutput
from .image import ImageOutput
from .loop_log import LoopEvent, LoopLogOutput, LoopView, QueueEntry, UnitView
from .mosaic import MosaicOutput, MosaicTile
from .placed_image import PlacedImageOutput, Placement
from .samples import SamplesOutput
from .spectrum import SpectrumOutput
from .spectrum_series import SpectrumSeriesOutput
from .survey import FocusPlane, MotionRate, SurveyFinding, SurveyOutput, SurveyTile
from .unit_frames import FrameTag, UnitFramesOutput

__all__ = [
    "CalibrationPoint",
    "FocusPlane",
    "FrameTag",
    "GalvoCalibration",
    "GalvoCalibrationOutput",
    "LoopEvent",
    "LoopLogOutput",
    "LoopView",
    "MotionRate",
    "PlacedImageOutput",
    "Placement",
    "QueueEntry",
    "SurveyFinding",
    "SurveyOutput",
    "SurveyTile",
    "UnitFramesOutput",
    "UnitView",
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
