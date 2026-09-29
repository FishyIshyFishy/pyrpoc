"""Addressable pieces of the instrument: driver and panel in one folder.

A device has configuration, calibration, a panel and persistence. Two properties
vary: whether it owns a connection, and whether it is backed by another device
rather than having one of its own.

Each folder implements ``structs.device.Device``. May import structs/. Qt
appears only in devices/*/panel.py, imported lazily inside
Device.panel() so the headless layers stay importable without Qt.
"""

from __future__ import annotations

from pyrpoc.src.structs.device import Device
from pyrpoc.src.structs.registries import device_registry

from .daq.device import DAQ, DaqError
from .galvo.device import Galvo
from .time_tagger.device import FlimMeasurement, TaggerError, TimeTagger

__all__ = [
    "Device",
    "device_registry",
    "DAQ",
    "DaqError",
    "FlimMeasurement",
    "Galvo",
    "TaggerError",
    "TimeTagger",
]
