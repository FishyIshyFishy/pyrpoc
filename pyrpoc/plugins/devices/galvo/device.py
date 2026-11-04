"""The scanner: two AO channels on the DAQ's card, ``backed_by`` the DAQ."""

# It has no connection of its own, so claiming it claims the card; its wiring is still
# configuration worth keeping.

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core import device_config as C
from pyrpoc.core.device import Device, device_registry
from pyrpoc.utils.value_rules import int_field

from ..daq.device import DAQ


@dataclass
class GalvoConfig(C.DeviceConfig):
    fast_ao: int = int_field(
        "Fast Axis AO",
        0,
        minimum=0,
        maximum=31,
        tooltip="Analog output channel for the fast (X) galvo",
    )
    slow_ao: int = int_field(
        "Slow Axis AO",
        1,
        minimum=0,
        maximum=31,
        tooltip="Analog output channel for the slow (Y) galvo",
    )


@device_registry.register("galvo")
class Galvo(Device):
    display_name = "Galvo"
    backed_by = DAQ
    config_cls = GalvoConfig

    config: GalvoConfig

    def summary(self) -> str:
        return f"fast ao{self.config.fast_ao}, slow ao{self.config.slow_ao}"
