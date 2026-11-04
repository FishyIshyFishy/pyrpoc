"""The NI-DAQ card: its device name, and which analog inputs are wired."""

# Analog inputs are a property of the card, not the scanner, so they live here.
# ``owns_connection`` means the card can be verified, not that it holds a handle: NI tasks are
# created per scan by the programs that use them.

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core import device_config as C
from pyrpoc.core.device import Device, DeviceError, device_registry
from pyrpoc.utils.value_rules import channels_field, text_field


class DaqError(DeviceError):
    """An NI-DAQ operation failed."""


@dataclass
class DaqConfig(C.DeviceConfig):
    device_name: str = text_field("DAQ Device", "Dev1", tooltip="NI-DAQ device name (e.g. Dev1)")
    ai_channels: tuple[int, ...] = channels_field(
        "Active AI Channels",
        num_channels=9,
        tooltip="Which analog input channels are connected and should be read",
    )


@device_registry.register("daq")
class DAQ(Device):
    display_name = "NI-DAQ"
    owns_connection = True
    config_cls = DaqConfig

    config: DaqConfig

    def connection_info(self) -> str:
        return self.config.device_name

    def check_reachable(self) -> bool:
        import nidaqmx.system

        names = {device.name for device in nidaqmx.system.System.local().devices}
        return self.config.device_name in names
