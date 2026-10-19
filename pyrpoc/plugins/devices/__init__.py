"""Device implementations, one folder each: driver and panel together.

Qt appears only in ``*/panel.py``, imported inside ``Device.panel()`` so this
package stays importable without a display.
"""

from __future__ import annotations

from pyrpoc.structs.plugins.devices import Device, device_registry

from .andor_ccd.device import AndorCCD, SpectrumReadout
from .andor_ccd.sdk import AndorError
from .daq.device import DAQ, DaqError
from .galvo.device import Galvo
from .prior_stage.device import PriorStage
from .prior_stage.sdk import PriorError
from .time_tagger.device import FlimMeasurement, TaggerError, TimeTagger
from .touptek_camera.device import ToupcamCamera
from .touptek_camera.sdk import ToupcamError

__all__ = [
    "Device",
    "device_registry",
    "AndorCCD",
    "AndorError",
    "DAQ",
    "DaqError",
    "FlimMeasurement",
    "Galvo",
    "PriorError",
    "PriorStage",
    "SpectrumReadout",
    "TaggerError",
    "TimeTagger",
    "ToupcamCamera",
    "ToupcamError",
]
