"""Device panels in the dock area: at most one per device, and a Panels menu
entry per device that opens or closes it.

A panel exists exactly as long as its dock; closing it leaves the device in
the inventory, connected, and reopening builds a fresh panel.
"""

from __future__ import annotations

import PyQt6Ads as qtads
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QAction

from pyrpoc.core.device import Device
from pyrpoc.engine.device_inventory.inventory import DeviceInventory
from pyrpoc.utils.widgets.device_panel import DevicePanel, device_panel_registry, has_panel


def dock_object_name(device: Device) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in device.instance_id)
    # ADS restores docks by object name, so renaming this strands saved layouts.
    return f"dock.device.{safe}"


class InventoryLink:
    """A device panel's way to its device's place in the inventory."""

    def __init__(self, inventory: DeviceInventory, device: Device):
        self.inventory = inventory
        self.device = device

    @property
    def name(self) -> str:
        return self.inventory.name(self.device)

    def connect(self) -> None:
        self.inventory.try_open_session(self.device)

    def fault(self) -> str | None:
        return self.inventory.fault(self.device)

    def changed(self) -> None:
        self.inventory.device_changed()


class DeviceDocks(QObject):
    # The per-device menu entries changed; the window rebuilds the Panels menu.
    actions_changed = pyqtSignal()

    def __init__(
        self, inventory: DeviceInventory, dock_manager: qtads.CDockManager, parent: QObject
    ):
        super().__init__(parent)
        self.inventory = inventory
        self.dock_manager = dock_manager
        # Set while ADS closes docks itself, restoring a layout or tearing down,
        # so those closes are not taken for the user closing a panel.
        self.frozen = False
        self.docks: dict[Device, qtads.CDockWidget] = {}
        self.panels: dict[Device, DevicePanel] = {}
        self.actions: dict[Device, QAction] = {}

        inventory.panel_requested.connect(self.on_panel_requested)
        inventory.changed.connect(self.on_inventory_changed)
        inventory.in_use_changed.connect(self.on_in_use_changed)
        self.on_inventory_changed()

    def on_panel_requested(self, device: Device, show: bool) -> None:
        if show:
            self.open(device)
        else:
            self.discard(device)

    def open(self, device: Device) -> None:
        """Show the device's panel, bringing an open one to the front."""
        if device in self.docks:
            self.docks[device].raise_()
            return
        panel = device_panel_registry.get(device.registry_key)(
            device, InventoryLink(self.inventory, device)
        )
        panel.set_in_use(device in self.inventory.in_use)
        dock = qtads.CDockWidget(panel.title)
        # ADS keys save/restore by the object name at add time, so set it first.
        dock.setObjectName(dock_object_name(device))
        dock.setWidget(panel)
        self.dock_manager.addDockWidgetFloating(dock)
        dock.closed.connect(lambda *_args, d=device: self.discard(d))
        self.docks[device] = dock
        self.panels[device] = panel
        self.inventory.set_panel_open(device, True)
        self.show_checked(device)

    def close(self, device: Device) -> None:
        dock = self.docks.pop(device)
        panel = self.panels.pop(device)
        self.dock_manager.removeDockWidget(dock)
        dock.takeWidget()
        panel.setParent(None)
        panel.deleteLater()
        dock.deleteLater()
        self.inventory.set_panel_open(device, False)
        self.show_checked(device)

    def discard(self, device: Device) -> None:
        """Close a device panel, from its dock's close button, its menu entry or the
        inventory."""
        # Deferred, because a dock's close arrives in a signal from the dock about to be deleted;
        # a second request for one gesture finds the panel already gone.
        if self.frozen or device not in self.docks:
            return
        QTimer.singleShot(0, lambda d=device: self.close(d) if d in self.docks else None)

    def show_checked(self, device: Device) -> None:
        """Show on the device's menu entry whether its panel is open, if the entry still
        exists."""
        action = self.actions.get(device)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(device in self.docks)
            action.blockSignals(False)

    def on_inventory_changed(self) -> None:
        """Close the panels of removed devices, keep one menu entry per device,
        and let open panels re-read their devices."""
        devices = self.inventory.devices
        for device in [d for d in self.docks if d not in devices]:
            self.close(device)
        for device in [d for d in self.actions if d not in devices]:
            self.actions.pop(device).deleteLater()
        for device in devices:
            if device not in self.actions:
                self.actions[device] = self.make_action(device)
        # In inventory order, which removals and additions above can disturb.
        self.actions = {device: self.actions[device] for device in devices}
        for panel in self.panels.values():
            panel.sync()
        self.actions_changed.emit()

    def make_action(self, device: Device) -> QAction:
        action = QAction(self.inventory.name(device), self)
        action.setCheckable(True)
        action.setChecked(device in self.docks)
        # Listed even without a panel, so the menu shows every device there is.
        action.setEnabled(has_panel(device))
        action.toggled.connect(lambda checked, d=device: self.on_panel_requested(d, checked))
        return action

    def on_in_use_changed(self) -> None:
        for device, panel in self.panels.items():
            panel.set_in_use(device in self.inventory.in_use)
