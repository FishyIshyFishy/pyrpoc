"""The NI-DAQ card: its device name, and which analog inputs are wired.

Analog inputs are a property of the card, not the scanner, so they live here.
``owns_connection`` means the card can be verified, not that it holds a
handle: NI tasks are created per scan by the programs that use them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrpoc.src.structs import params as P
from pyrpoc.src.structs.device import Device, DeviceError
from pyrpoc.src.structs.registries import device_registry

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget


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

    def summary(self) -> str:
        channels = ", ".join(f"AI{n}" for n in self.config.ai_channels) or "no inputs"
        return f"{self.config.device_name} - {channels}"

    def check_reachable(self) -> bool:
        import nidaqmx.system

        names = {device.name for device in nidaqmx.system.System.local().devices}
        return self.config.device_name in names

    def panel(self, parent: QWidget, on_change: Callable[[], None]) -> QWidget:
        from .panel import DaqPanel

        return DaqPanel(self, parent, on_change)
