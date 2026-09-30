"""Device implementations, one folder each: driver and panel together.

Qt appears only in ``*/panel.py``, imported inside ``Device.panel()`` so this
package stays importable without a display.
"""

from __future__ import annotations

from pyrpoc.structs.device import Device, device_registry

from .daq.device import DAQ, DaqError
from .galvo.device import Galvo
from .prior_stage.device import PriorStage
from .prior_stage.sdk import PriorError
from .time_tagger.device import FlimMeasurement, TaggerError, TimeTagger

__all__ = [
    "Device",
    "device_registry",
    "DAQ",
    "DaqError",
    "FlimMeasurement",
    "Galvo",
    "PriorError",
    "PriorStage",
    "TaggerError",
    "TimeTagger",
]
