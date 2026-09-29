"""Addressable pieces of the instrument: driver and panel in one folder.

A device has configuration, calibration, a panel and persistence. Two properties
vary: whether it owns a connection, and whether it is backed by another device
rather than having one of its own.

May import core/. Qt appears only in devices/*/panel.py, imported lazily inside
Device.panel() so the headless layers stay importable without Qt.
"""

from pyrpoc.structs.device import Device
from pyrpoc.structs.registries import device_registry
from .daq.device import DAQ, DaqError
from .galvo.device import Galvo
from .time_tagger.device import TaggerError, TimeTagger

__all__ = ["Device", "device_registry", "DAQ", "DaqError", "Galvo", "TaggerError", "TimeTagger"]
