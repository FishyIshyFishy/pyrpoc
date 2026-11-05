"""The kinds of data programs publish, and the file formats each is saved in. Importing this
package registers them all."""

from __future__ import annotations

from .arrays import ArrayData, DecayHistogram, Image, RawSamples, Spectrum
from .loop import LoopEvent, LoopUnits, UnitFrame, UnitSnapshot
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
    "LoopUnits",
    "Mosaic",
    "MotionRate",
    "MosaicWriter",
    "NpzWriter",
    "Overview",
    "PlacedTile",
    "RawSamples",
    "Record",
    "Spectrum",
    "SurveyResult",
    "SurveyTile",
    "TiffWriter",
    "UnitFrame",
    "UnitFramesWriter",
    "UnitSnapshot",
]
