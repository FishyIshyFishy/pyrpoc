"""The scanner: two AO channels on the DAQ's card.

It has no connection of its own, so it is ``backed_by`` the DAQ and claiming it
claims the card. Its wiring is still configuration worth persisting.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.structs.plugins import params as P
from pyrpoc.structs.plugins.devices import Device, device_registry

from ..daq.device import DAQ


@dataclass
class GalvoConfig(P.Group):
    fast_ao: int = P.int_field(
        "Fast Axis AO",
        0,
        minimum=0,
        maximum=31,
        tooltip="Analog output channel for the fast (X) galvo",
    )
    slow_ao: int = P.int_field(
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
