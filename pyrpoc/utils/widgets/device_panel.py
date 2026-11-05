"""``DevicePanel``: one device's controls and configuration, docked on their own, and its
registry."""

from __future__ import annotations

from typing import Protocol

from pyrpoc.core.device import Device
from pyrpoc.core.panels import Panel
from pyrpoc.core.registry import Registry


class DeviceLink(Protocol):
    """What a device panel is given by whoever holds its device: the device's name, connecting
    it with any fault kept, and reporting that it changed."""

    @property
    def name(self) -> str: ...

    def connect(self) -> None: ...

    def fault(self) -> str | None: ...

    def changed(self) -> None: ...


class DevicePanel(Panel):
    """One device's controls and configuration, docked on their own."""

    def __init__(self, device: Device, link: DeviceLink):
        super().__init__()
        # The device's identity, so the dock keeps its place across launches.
        self.instance_id = device.instance_id
        self.link = link

    @property
    def title(self) -> str:
        return self.link.name

    def sync(self) -> None:
        """Re-read the device, which may have changed elsewhere."""

    def set_in_use(self, in_use: bool) -> None:
        """Disable the controls while a run holds the device."""
        self.setEnabled(not in_use)


device_panel_registry: Registry[DevicePanel] = Registry("DevicePanelRegistry", DevicePanel)


def has_panel(device: Device) -> bool:
    return device.registry_key in device_panel_registry.entries
