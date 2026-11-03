"""Device implementations, one folder each: the driver and, for a device with live
controls, its panel. Importing this package registers them all."""

# Qt appears only in ``*/panel.py``, so a driver can be used without it.

from __future__ import annotations

from pyrpoc.core.device import Device, device_registry
from pyrpoc.core.panels import device_panel_registry

from .andor_ccd.device import AndorCCD, SpectrumReadout
from .andor_ccd.panel import AndorCcdPanel
from .andor_ccd.sdk import AndorError
from .daq.device import DAQ, DaqError
from .galvo.device import Galvo
from .prior_stage.device import PriorStage
from .prior_stage.panel import PriorStagePanel
from .prior_stage.sdk import PriorError
from .time_tagger.device import FlimMeasurement, TaggerError, TimeTagger
from .touptek_camera.device import ToupcamCamera
from .touptek_camera.sdk import ToupcamError

__all__ = [
    "Device",
    "device_registry",
    "device_panel_registry",
    "AndorCCD",
    "AndorCcdPanel",
    "AndorError",
    "DAQ",
    "DaqError",
    "FlimMeasurement",
    "Galvo",
    "PriorError",
    "PriorStage",
    "PriorStagePanel",
    "SpectrumReadout",
    "TaggerError",
    "TimeTagger",
    "ToupcamCamera",
    "ToupcamError",
]
