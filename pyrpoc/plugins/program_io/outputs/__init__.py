"""The kinds of data programs publish, and the file formats each is saved in. Importing this
package registers them all."""

from __future__ import annotations

from .arrays import ArrayData, DecayHistogram, Image, RawSamples, Spectrum
from .loop import (
    LoopEvent,
    LoopState,
    LoopUnits,
    QueueEntry,
    UnitFrame,
    UnitSnapshot,
    UnitState,
)
from .mosaic import Mosaic, Overview, PlacedTile
from .records import (
    CalibrationPoint,
    FocusPlane,
    GalvoCalibration,
    MotionRate,
    Record,
    SurveyResult,
    SurveyTile,
)
from .writers import (
    EventLogWriter,
    JsonWriter,
    MosaicWriter,
    NpzWriter,
    TiffWriter,
    UnitFramesWriter,
)

__all__ = [
    "ArrayData",
    "CalibrationPoint",
    "DecayHistogram",
    "EventLogWriter",
    "FocusPlane",
    "GalvoCalibration",
    "Image",
    "JsonWriter",
    "LoopEvent",
    "LoopState",
    "LoopUnits",
    "Mosaic",
    "MotionRate",
    "MosaicWriter",
    "NpzWriter",
    "Overview",
    "PlacedTile",
    "QueueEntry",
    "RawSamples",
    "Record",
    "Spectrum",
    "SurveyResult",
    "SurveyTile",
    "TiffWriter",
    "UnitFrame",
    "UnitFramesWriter",
    "UnitSnapshot",
    "UnitState",
]
