"""The app's built-in panels, always present and built once by the window.
Each moves into the subsystem it shows."""

from __future__ import annotations

from pyrpoc.data_library.panel import DataLibraryPanel
from pyrpoc.device_inventory.panel import DevicesPanel

from .acquisition.panel import AcquisitionPanel

__all__ = ["AcquisitionPanel", "DataLibraryPanel", "DevicesPanel"]
