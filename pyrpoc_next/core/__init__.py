"""Orchestration: interpret the routine, check compatibility, route parcels, store data.

The application layer above acquisition. Qt-free — the gui drives it and observes via
callbacks.
"""

from __future__ import annotations

from pyrpoc_next.core.app_state import AppState
from pyrpoc_next.core.compatibility import check_routine
from pyrpoc_next.core.controller import Controller
from pyrpoc_next.core.router import DisplaySink, route_parcel
from pyrpoc_next.core.storage import FrameStorage

__all__ = [
    "AppState",
    "check_routine",
    "Controller",
    "DisplaySink",
    "route_parcel",
    "FrameStorage",
]
