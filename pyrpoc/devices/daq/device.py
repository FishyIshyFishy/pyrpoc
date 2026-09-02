"""The NI-DAQ card: which device name, and which analog inputs are wired.

``active_ai_channels`` lands here rather than on the galvo. Section 9's table
lumps "galvo/AI channels" into one row whose point is that they stop being loose
per-modality parameters; analog input is a property of the card, not the
scanner, so it is filed with the card. One definition instead of three, which is
what the row asks for.

``owns_connection`` means the device can verify it exists, not that it holds an
open handle: an NI task *is* the clock domain, so tasks are created per scan
inside ``programs/hardware/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrpoc.core import params as P

from ..base import Device
from ..registry import device_registry

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget


@dataclass
class DaqConfig(P.Group):
    device_name: str = P.text_field(
        "DAQ Device", "Dev1", tooltip="NI-DAQ device name (e.g. Dev1)"
    )
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

    def panel(self, parent: "QWidget | None" = None, on_change=None) -> "QWidget | None":
        from .panel import DaqPanel

        return DaqPanel(self, parent=parent, on_change=on_change)
