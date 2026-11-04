"""The kinds of data programs publish, and the file formats each is saved in. Importing this
package registers them all."""

from __future__ import annotations

from .arrays import ArrayData, DecayHistogram, Image, RawSamples, Spectrum
from .mosaic import Mosaic, MosaicTile
from .writers import MosaicWriter, NpzWriter, TiffWriter

__all__ = [
    "ArrayData",
    "DecayHistogram",
    "Image",
    "Mosaic",
    "MosaicTile",
    "MosaicWriter",
    "NpzWriter",
    "RawSamples",
    "Spectrum",
    "TiffWriter",
]
