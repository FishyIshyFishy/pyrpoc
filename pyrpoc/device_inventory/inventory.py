"""The devices this workbench has: adding and removing them, and opening and
closing the connections the ones that keep a session hold."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.structs.plugins.devices import Device, DeviceError, device_registry

log = logging.getLogger(__name__)


def close_if_open(device: Device) -> None:
    """Close a device's session. A fault while disconnecting is logged, never
    raised: removing a card or quitting must not depend on the hardware."""
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

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self.devices: list[Device] = []

    def add(
        self, key: str, instance_id: str | None = None, user_label: str | None = None
    ) -> Device:
        device = device_registry.get(key)(instance_id=instance_id, user_label=user_label)
        self.devices.append(device)
        self.changed.emit()
        self.edited.emit()
        return device

    def remove(self, device: Device) -> None:
        close_if_open(device)
        self.devices.remove(device)
        self.changed.emit()
        self.edited.emit()

    def clear(self) -> None:
        self.close_sessions()
        self.devices.clear()
        self.changed.emit()

    def mark_edited(self) -> None:
        """A device's configuration was edited in place."""
        self.edited.emit()

    def open_sessions(self) -> list[Device]:
        """Connect every device that keeps a session, returning the ones that
        failed; they stay in the inventory, disconnected, with ``last_error``."""
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
