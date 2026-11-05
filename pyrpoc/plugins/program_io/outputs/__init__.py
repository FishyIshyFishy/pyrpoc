"""The kinds of data programs publish, and the file formats each is saved in. Importing this
package registers them all."""

from __future__ import annotations

from .arrays import ArrayData, DecayHistogram, Image, RawSamples, Spectrum
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
from .writers import JsonWriter, MosaicWriter, NpzWriter, TiffWriter

__all__ = [
    "ArrayData",
    "CalibrationPoint",
    "DecayHistogram",
    "FocusPlane",
    "GalvoCalibration",
    "Image",
    "JsonWriter",
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
]
