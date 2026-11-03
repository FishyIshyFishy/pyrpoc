"""The data library: every dataset open in this session, and recordings on
disk. Importing this package registers the file formats."""

from __future__ import annotations

from .writers import MosaicWriter, NpzWriter, TiffWriter

__all__ = ["MosaicWriter", "NpzWriter", "TiffWriter"]
