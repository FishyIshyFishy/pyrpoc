"""Acquired data: datasets, the library of open runs, and saving.

Exists regardless of how the data was acquired, which is why saving is not a
program's job. May import core/ and nothing else.
"""

from .dataset import Dataset, Provenance
from .library import DatasetLibrary
from .io import RunSaver, SaveTarget, read_metadata
from .transforms import channel_levels, normalize_channels

__all__ = [
    "Dataset",
    "Provenance",
    "DatasetLibrary",
    "RunSaver",
    "SaveTarget",
    "read_metadata",
    "normalize_channels",
    "channel_levels",
]
