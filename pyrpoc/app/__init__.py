"""The business logic: what exists, what is running, what a session means.

No pixels here -- ``Application`` and its collaborators are ``QObject``s for
their signals, not ``QWidget``s. ``panels/`` and ``shell/`` call into this
layer; this layer never imports either back.
"""

from __future__ import annotations

from .application import Application
from .catalog import CATALOG, Entry
from .picking import PickingController
from .run_bridge import RunBridge
from .session_io import Autosave, apply, capture, seed_defaults

__all__ = [
    "Application",
    "CATALOG",
    "Entry",
    "PickingController",
    "RunBridge",
    "Autosave",
    "apply",
    "capture",
    "seed_defaults",
]
