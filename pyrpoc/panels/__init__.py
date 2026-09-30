"""The app's built-in panels, always present and built once by the window.
Each moves into the subsystem it shows."""

from __future__ import annotations

from pyrpoc.data_library.panel import DataLibraryPanel

from .acquisition.panel import AcquisitionPanel
from .devices.panel import DevicesPanel

__all__ = ["AcquisitionPanel", "DataLibraryPanel", "DevicesPanel"]
