"""The scanner: two AO channels on the DAQ's card, ``backed_by`` the DAQ."""

# It has no connection of its own, so claiming it claims the card; its wiring is still
# configuration worth keeping. A card may drive several galvos, each its own Galvo on its own
# channels. The channel is the card's own AO number, which on a second BNC board is not the
# number printed beside the jack.

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.device import Device, DeviceConfig, device_registry
from pyrpoc.utils.value_rules import int_field

from ..daq.device import DAQ

# Said beside each channel, since a wrong one moves the other galvo.
AO_TOOLTIP = (
    "The card's AO channel number. A BNC board on the card's second connector carries ao2 and "
    "ao3 on the jacks labelled AO 0 and AO 1."
)


@dataclass
class GalvoConfig(DeviceConfig):
    fast_ao: int = int_field(
        "Fast Axis AO",
        0,
        minimum=0,
        maximum=31,
        tooltip=f"Fast (X) axis. {AO_TOOLTIP}",
    )
    slow_ao: int = int_field(
        "Slow Axis AO",
        1,
        minimum=0,
        maximum=31,
        tooltip=f"Slow (Y) axis. {AO_TOOLTIP}",
    )


@device_registry.register("galvo")
class Galvo(Device):
    display_name = "Galvo"
    backed_by = DAQ
    config_cls = GalvoConfig

    config: GalvoConfig

    def summary(self) -> str:
        return f"fast ao{self.config.fast_ao}, slow ao{self.config.slow_ao}"
