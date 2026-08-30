"""Addressable pieces of the instrument: driver and panel in one folder.

A device has configuration, calibration, a panel and persistence. Two properties
vary: whether it owns a connection, and whether it is backed by another device
rather than having one of its own.

May import core/. Qt appears only in devices/*/panel.py, imported lazily inside
Device.panel() so the headless layers stay importable without Qt.
"""

from .base import Device
from .registry import device_registry
from .daq.device import DAQ
from .galvo.device import Galvo
from .time_tagger.device import TimeTagger

__all__ = ["Device", "device_registry", "DAQ", "Galvo", "TimeTagger"]
