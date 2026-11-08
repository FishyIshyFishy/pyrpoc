"""The types of program output programs fill, each with the files it is saved in. Importing
this package registers them all."""

from __future__ import annotations

from .calibration_log import (
    AxisMeasured,
    AxisMeasuring,
    CalibrationEvent,
    CalibrationLogOutput,
    FeatureFound,
    StageMoved,
)
from .focus_stack import FocusResult, FocusStackOutput, FocusStep
from .galvo_calibration import CalibrationPoint, GalvoCalibration, GalvoCalibrationOutput
from .histogram import HistogramOutput
from .image import ImageOutput
from .log import LogEvent, LogOutput, event_from_dict, event_to_dict
from .loop_events import (
    LOOP_EVENTS,
    FocusChecked,
    FocusingAtUnit,
    LoopActivity,
    LoopEvent,
    LoopFinished,
    LoopResumed,
    LoopStarted,
    Measured,
    MovingToUnit,
    RoutineEnded,
    RoutineFailed,
    RunNowAsked,
    ScanningUnit,
    StepDue,
    UnitEdited,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitRefused,
    UnitResumed,
    UnitScanned,
)
from .loop_log import LoopLogOutput
from .loop_view import LoopFold, LoopView, QueueEntry, Series, UnitState, UnitView
from .mosaic import MosaicOutput, MosaicTile
from .samples import SamplesOutput
from .spectrum import SpectrumOutput
from .spectrum_series import SpectrumSeriesOutput
from .survey import FocusPlane, MotionRate, Placement, SurveyFinding, SurveyOutput, SurveyTile
from .survey_log import (
    MotionMeasured,
    MotionRevisiting,
    Stitching,
    SurveyEnded,
    SurveyEvent,
    SurveyLogOutput,
    TileFocusing,
    TileTaken,
    TileVisiting,
)
from .unit_frames import FrameTag, UnitFramesOutput
from .units import Outline, Unit, UnitRecord, UnitsOutput

__all__ = [
    "LOOP_EVENTS",
    "AxisMeasured",
    "AxisMeasuring",
    "CalibrationEvent",
    "CalibrationLogOutput",
    "FeatureFound",
    "FocusingAtUnit",
    "LogEvent",
    "LogOutput",
    "LoopActivity",
    "MotionMeasured",
    "MotionRevisiting",
    "MovingToUnit",
    "ScanningUnit",
    "StageMoved",
    "Stitching",
    "SurveyEnded",
    "SurveyEvent",
    "SurveyLogOutput",
    "TileFocusing",
    "TileTaken",
    "TileVisiting",
    "event_from_dict",
    "event_to_dict",
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
    "UnitRefused",
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
    "LoopFold",
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
