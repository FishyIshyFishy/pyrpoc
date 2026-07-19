"""Displays: parcel-consuming Qt widgets. Importing registers each one."""

from __future__ import annotations

from pyrpoc_next.gui.displays.base import DisplayWidget, display_registry
from pyrpoc_next.gui.displays.flim_display import FlimDisplay
from pyrpoc_next.gui.displays.image_displays import (
    MultiChannelDisplay,
    StreamedDisplay,
    TiledDisplay,
)

__all__ = [
    "DisplayWidget",
    "display_registry",
    "StreamedDisplay",
    "TiledDisplay",
    "MultiChannelDisplay",
    "FlimDisplay",
]
