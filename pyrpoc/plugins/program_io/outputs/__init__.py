"""The types of program output programs fill, each with the files it is saved in. Importing
this package registers them all."""

from __future__ import annotations

from .focus_stack import FocusResult, FocusStackOutput, FocusStep
from .galvo_calibration import CalibrationPoint, GalvoCalibration, GalvoCalibrationOutput
from .histogram import HistogramOutput
from .image import ImageOutput
from .loop_log import LoopEvent, LoopLogOutput, LoopView, QueueEntry, UnitView
from .mosaic import MosaicOutput, MosaicTile
from .samples import SamplesOutput
from .spectrum import SpectrumOutput
from .spectrum_series import SpectrumSeriesOutput
from .survey import FocusPlane, MotionRate, Placement, SurveyFinding, SurveyOutput, SurveyTile
from .unit_frames import FrameTag, UnitFramesOutput
from .units import Unit, UnitsOutput

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
    "Placement",
    "QueueEntry",
    "SurveyFinding",
    "SurveyOutput",
    "SurveyTile",
    "Unit",
    "UnitFramesOutput",
    "UnitsOutput",
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
