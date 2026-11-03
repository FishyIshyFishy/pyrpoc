"""The devices this workbench has: adding and removing them, connecting them,
and which are in use by a run or have their panel open."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core.device import Device, DeviceError, device_registry

log = logging.getLogger(__name__)


def close_if_open(device: Device) -> None:
    """Close a device's session, logging a fault rather than raising it."""
    # Removing a device or quitting must not depend on the hardware.
    if not device.session_open:
        return
    try:
        device.close_session()
    except DeviceError:
        log.warning("could not cleanly disconnect %s", device.name, exc_info=True)


class DeviceInventory(QObject):
    # Which devices exist or are connected changed: what a program can use.
    changed = pyqtSignal()
    # Something worth remembering changed, a device's configuration included.
    edited = pyqtSignal()
    # Which devices a run holds changed.
    in_use_changed = pyqtSignal()
    # Which devices have their panel open changed.
    panels_changed = pyqtSignal()
    # Asked to open (True) or close (False) a device's panel; the window does it.
    panel_requested = pyqtSignal(object, bool)

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self.devices: list[Device] = []
        self.in_use: set[Device] = set()
        self.open_panels: set[Device] = set()

    def add(self, key: str, instance_id: str | None = None) -> Device:
        device = device_registry.get(key)(instance_id=instance_id)
        device.number = self.free_number(type(device))
        self.devices.append(device)
        self.changed.emit()
        self.edited.emit()
        return device

    def free_number(self, cls: type[Device]) -> int:
        """The lowest number this device type has free."""
        # Assigned once, so a device never renames itself when an earlier one is removed.
        taken = {device.number for device in self.devices if type(device) is cls}
        number = 1
        while number in taken:
            number += 1
        return number

    def remove(self, device: Device) -> None:
        close_if_open(device)
        self.devices.remove(device)
        self.changed.emit()
        self.edited.emit()

    def clear(self) -> None:
        self.close_sessions()
        self.devices.clear()
        self.changed.emit()

    def device_changed(self) -> None:
        """A device's configuration or connection changed in place."""
        self.changed.emit()
        self.edited.emit()

    def connect_device(self, device: Device) -> None:
        device.try_open_session()
        self.device_changed()

    def disconnect_device(self, device: Device) -> None:
        """Close the device's session, keeping any fault on the device to show."""
        # Kept here, unlike elsewhere, because the user asked for it.
        try:
            device.close_session()
            device.last_error = None
        except DeviceError as exc:
            device.last_error = str(exc)
        self.device_changed()

    def test_device(self, device: Device) -> None:
        device.test_connection()
        self.device_changed()

    def request_panel(self, device: Device, show: bool) -> None:
        self.panel_requested.emit(device, show)

    def set_panel_open(self, device: Device, is_open: bool) -> None:
        if is_open:
            self.open_panels.add(device)
        else:
            self.open_panels.discard(device)
        self.panels_changed.emit()

    def set_in_use(self, devices: set[Device]) -> None:
        if devices != self.in_use:
            self.in_use = devices
            self.in_use_changed.emit()

    def open_sessions(self) -> list[Device]:
        """Connect every device that keeps a session, returning the ones that failed."""
        # Failed devices stay in the inventory, disconnected, with ``last_error``.
        failed = [
            device
            for device in self.devices
            if device.holds_session and not device.session_open and not device.try_open_session()
        ]
        self.changed.emit()
        return failed

    def close_sessions(self) -> None:
        for device in self.devices:
            close_if_open(device)
