"""``InventoryStage``: the inventory's XY stage, as a panel may move it outside a run."""

# A panel is never handed the device itself: it may only send the stage somewhere, and only
# while no run holds it, so a panel can never pull the stage from under a run.

from __future__ import annotations

from pyrpoc.core.device import Connection, Device, DeviceError
from pyrpoc.utils.stage import XYStage

from .inventory import DeviceInventory


def reachable(device: Device) -> bool:
    return device.connection is not Connection.SESSION or device.session_open


class InventoryStage:
    """The first XY stage in the inventory that is connected."""

    def __init__(self, inventory: DeviceInventory):
        self.inventory = inventory

    def stage(self) -> XYStage | None:
        for device in self.inventory.devices:
            if isinstance(device, XYStage) and reachable(device):
                return device
        return None

    def blocked(self) -> str | None:
        stage = self.stage()
        if stage is None:
            return "no XY stage is connected"
        if stage in self.inventory.in_use:
            return "a run is using the stage"
        return None

    def move_to(self, x_um: float, y_um: float) -> str | None:
        stage = self.stage()
        if stage is None or stage in self.inventory.in_use:
            return self.blocked()
        # The hardware is a boundary: a controller fault is told, not raised.
        try:
            stage.move_xy_to(x_um, y_um)
        except DeviceError as exc:
            return str(exc)
        return None
