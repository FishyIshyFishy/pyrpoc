"""The kinds of data programs publish, and the file formats each is saved in. Importing this
package registers them all."""

from __future__ import annotations

from .arrays import ArrayData, DecayHistogram, Image, RawSamples, Spectrum
from .mosaic import Mosaic, MosaicTile
from .records import CalibrationPoint, GalvoCalibration, Record
from .writers import JsonWriter, MosaicWriter, NpzWriter, TiffWriter

__all__ = [
    "ArrayData",
    "CalibrationPoint",
    "DecayHistogram",
    "GalvoCalibration",
    "Image",
    "JsonWriter",
    "Mosaic",
    "MosaicTile",
    "MosaicWriter",
    "NpzWriter",
    "RawSamples",
    "Record",
    "Spectrum",
    "TiffWriter",
]
