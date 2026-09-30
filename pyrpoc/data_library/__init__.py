"""The data library: every dataset open in this session, and recordings on disk.

    store.py      LibraryStore: the open datasets, the size limit, what may be purged (no Qt)
    model.py      LibraryModel: the commands the UI issues, and change signals (Qt)
    saving.py     RecordingSaver: one recording's writers and metadata file (no Qt)
    format.py     the recording format, written and read in one place (no Qt)
    writers/      file formats per kind of data (TIFF, NPZ), registered in writer_registry
    panel.py      the Data Library panel
    details.py    the Details dialog: metadata and notes

Importing this package registers the writers.
"""

from __future__ import annotations

from .writers import NpzWriter, TiffWriter

__all__ = ["NpzWriter", "TiffWriter"]
