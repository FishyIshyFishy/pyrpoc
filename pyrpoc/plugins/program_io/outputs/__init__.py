"""The types of program output programs fill, each with the files it is saved in. Importing
this package registers them all."""

from __future__ import annotations

from .focus_stack import FocusResult, FocusStackOutput, FocusStep
from .galvo_calibration import CalibrationPoint, GalvoCalibration, GalvoCalibrationOutput
from .histogram import HistogramOutput
from .image import ImageOutput
from .loop_events import (
    CommandRefused,
    FocusChecked,
    LoopEvent,
    LoopFinished,
    LoopResumed,
    LoopStarted,
    Measured,
    RoutineEnded,
    RoutineFailed,
    RunNowAsked,
    StepDue,
    UnitEdited,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitResumed,
    UnitScanned,
    event_from_dict,
    event_to_dict,
)
from .loop_log import LoopLogOutput
from .loop_view import LoopView, QueueEntry, Series, UnitState, UnitView
from .mosaic import MosaicOutput, MosaicTile
from .samples import SamplesOutput
from .spectrum import SpectrumOutput
from .spectrum_series import SpectrumSeriesOutput
from .survey import FocusPlane, MotionRate, Placement, SurveyFinding, SurveyOutput, SurveyTile
from .unit_frames import FrameTag, UnitFramesOutput
from .units import Outline, Unit, UnitRecord, UnitsOutput

__all__ = [
    "event_from_dict",
    "event_to_dict",
    "CommandRefused",
    "FocusChecked",
    "LoopFinished",
    "LoopResumed",
    "LoopStarted",
    "Measured",
    "Outline",
    "RoutineEnded",
    "RoutineFailed",
    "RunNowAsked",
    "Series",
    "StepDue",
    "UnitEdited",
    "UnitJoined",
    "UnitLeft",
    "UnitPaused",
    "UnitRecord",
    "UnitResumed",
    "UnitScanned",
    "UnitState",
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
