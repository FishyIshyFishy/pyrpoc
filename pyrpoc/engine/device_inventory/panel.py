"""The Devices panel: what this workbench has, whether each is reachable, and the actions
on a device other than controlling it."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMenu,
    QPushButton,
    QVBoxLayout,
)

from pyrpoc.core.device import Device, device_registry
from pyrpoc.core.panels import Panel
from pyrpoc.utils.widgets.device_panel import has_panel
from pyrpoc.utils.widgets.table import ListTable, horizontal_header, viewport_of

from .configure import ConfigureDialog
from .inventory import DeviceInventory, DeviceStatus

NAME, STATUS, CONNECTION = range(3)
COLUMNS = ["Device", "Status", "Connection"]


def status_of(device: Device, status: DeviceStatus, in_use: bool) -> str:
    if in_use:
        return "In use"
    if device.holds_session:
        if device.session_open:
            return "Connected"
        return "Error" if status.last_error else "Disconnected"
    if device.owns_connection:
        if status.last_test_ok is None:
            return "Not tested"
        return "OK" if status.last_test_ok else "Failed"
    if device.backed_by is not None:
        return f"via {device.backed_by.display_name}"
    return "—"


class DevicesPanel(Panel):
    display_name = "Devices"

    def __init__(self, inventory: DeviceInventory):
        super().__init__()
        self.inventory = inventory

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addLayout(self.build_add_row())

        self.table = ListTable(COLUMNS, self)
        header = horizontal_header(self.table)
        header.setSectionResizeMode(NAME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(CONNECTION, QHeaderView.ResizeMode.Stretch)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_row_menu)
        root.addWidget(self.table, 1)

        for signal in (
            inventory.changed,
            inventory.edited,
            inventory.in_use_changed,
            inventory.panels_changed,
        ):
            signal.connect(self.refresh)
        self.refresh()

    def build_add_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.type_combo = QComboBox(self)
        for key in device_registry.keys():
            self.type_combo.addItem(device_registry.get(key).display_name, key)
        add_btn = QPushButton("Add", self)
        add_btn.clicked.connect(self.add_chosen)
        row.addWidget(self.type_combo, 1)
        row.addWidget(add_btn)
        return row

    def add_chosen(self) -> None:
        device = self.inventory.add(self.type_combo.currentData())
        if has_panel(device):
            self.inventory.request_panel(device, True)

    def refresh(self) -> None:
        devices = self.inventory.devices
        self.table.setRowCount(len(devices))
        for row, device in enumerate(devices):
            self.table.set_cell(row, NAME, self.inventory.name(device))
            status = self.table.set_cell(
                row,
                STATUS,
                status_of(device, self.inventory.status[device], device in self.inventory.in_use),
            )
            status.setToolTip(self.inventory.fault(device) or "")
            self.table.set_cell(row, CONNECTION, device.connection_info())

    def show_row_menu(self, position: QPoint) -> None:
        row = self.table.rowAt(position.y())
        if row < 0:
            return
        self.table.selectRow(row)
        menu = self.build_row_menu(self.inventory.devices[row])
        menu.exec(viewport_of(self.table).mapToGlobal(position))

    def build_row_menu(self, device: Device) -> QMenu:
        """Only the actions that apply to this kind of device; the ones that
        would disturb a run are greyed while it holds the device."""
        inventory = self.inventory
        idle = device not in inventory.in_use
        menu = QMenu(self)
        if has_panel(device):
            showing = device in inventory.open_panels
            menu.addAction(
                "Close Panel" if showing else "Open Panel",
                lambda: inventory.request_panel(device, not showing),
            )
        else:
            menu.addAction("Configure…", lambda: self.configure(device))
        if device.holds_session:
            if device.session_open:
                add_action(menu, "Disconnect", lambda: inventory.disconnect_device(device), idle)
            else:
                menu.addAction("Connect", lambda: inventory.connect_device(device))
        elif device.owns_connection:
            add_action(menu, "Test Connection", lambda: inventory.test_device(device), idle)
        menu.addAction("Rename…", lambda: self.rename(device))
        menu.addSeparator()
        add_action(menu, "Remove", lambda: inventory.remove(device), idle)
        return menu

    def rename(self, device: Device) -> None:
        label, accepted = QInputDialog.getText(
            self,
            "Rename device",
            f"Name for this {device.display_name} (blank for the default):",
            text=self.inventory.status[device].label,
        )
        if accepted:
            self.inventory.rename(device, label)

    def configure(self, device: Device) -> None:
        ConfigureDialog(device, self.inventory, self).show()


def add_action(menu: QMenu, text: str, slot: Callable[[], None], enabled: bool) -> None:
    action = QAction(text, menu)
    action.setEnabled(enabled)
    action.triggered.connect(slot)
    menu.addAction(action)
