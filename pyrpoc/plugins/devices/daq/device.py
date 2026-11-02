"""The NI-DAQ card: its device name, and which analog inputs are wired.

Analog inputs are a property of the card, not the scanner, so they live here.
``owns_connection`` means the card can be verified, not that it holds a
handle: NI tasks are created per scan by the programs that use them.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.structs.plugins import params as P
from pyrpoc.structs.plugins.devices import Device, DeviceError, device_registry


class DaqError(DeviceError):
    """An NI-DAQ operation failed."""


@dataclass
class DaqConfig(P.Group):
    device_name: str = P.text_field("DAQ Device", "Dev1", tooltip="NI-DAQ device name (e.g. Dev1)")
    ai_channels: tuple[int, ...] = P.channels_field(
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
