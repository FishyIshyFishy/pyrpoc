"""The NI-DAQ card, known by its device name."""

# ``TESTABLE``, not ``SESSION``: the card can be verified, but holds no handle. NI tasks are
# created per scan by the programs that use them.

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.device import Connection, Device, DeviceConfig, DeviceError, device_registry
from pyrpoc.utils.value_rules import text_field


class DaqError(DeviceError):
    """An NI-DAQ operation failed."""


@dataclass
class DaqConfig(DeviceConfig):
    device_name: str = text_field("DAQ Device", "Dev1", tooltip="NI-DAQ device name (e.g. Dev1)")


@device_registry.register("daq")
class DAQ(Device):
    display_name = "NI-DAQ"
    connection = Connection.TESTABLE
    config_cls = DaqConfig

    config: DaqConfig

    def connection_info(self) -> str:
        return self.config.device_name

    def check_reachable(self) -> bool:
        import nidaqmx.system

        names = {device.name for device in nidaqmx.system.System.local().devices}
        return self.config.device_name in names
