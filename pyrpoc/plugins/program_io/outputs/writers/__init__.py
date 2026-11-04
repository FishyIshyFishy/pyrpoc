"""File formats a run's outputs are saved in, one per file, each registered
for the kinds of ``Data`` it saves."""

from __future__ import annotations

from .mosaic import MosaicWriter
from .npz import NpzWriter
from .tiff import TiffWriter

__all__ = ["MosaicWriter", "NpzWriter", "TiffWriter"]
