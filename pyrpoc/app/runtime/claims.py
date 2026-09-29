"""Resolving a program's declared devices against what is configured, and
holding them while a run uses them.

Claims propagate up ``backed_by``: claiming the galvo claims its DAQ. Runs may
overlap, so what stops two of them is a device held twice, not a second run.
"""

from __future__ import annotations

from collections.abc import Iterable

from pyrpoc.structs.device import Device, DeviceError, MissingDevice


class DeviceBusy(DeviceError):
    """A program needs devices that another run holds."""

    def __init__(self, busy: list[str]):
        self.busy = list(busy)
        super().__init__("in use by another run: " + ", ".join(self.busy))


def expand(uses: list[type[Device]]) -> list[type[Device]]:
    """Every device class implied by ``uses``, following ``backed_by``:
    declaration order first, then each backing device, with no duplicates."""
    ordered: list[type[Device]] = []

    def add(cls: type[Device]) -> None:
        if cls in ordered:
            return
        ordered.append(cls)
        if cls.backed_by is not None:
            add(cls.backed_by)

    for cls in uses:
        add(cls)
    return ordered


def missing(uses: list[type[Device]], inventory: list[Device]) -> list[type[Device]]:
    """Which required device classes have no instance configured."""
    return [cls for cls in expand(uses) if not any(isinstance(device, cls) for device in inventory)]


def resolve(uses: list[type[Device]], inventory: list[Device]) -> dict[type[Device], Device]:
    """Bind each required class to an instance, or raise naming what is absent."""
    absent = missing(uses, inventory)
    if absent:
        raise MissingDevice([cls.display_name for cls in absent])
    return {
        cls: next(device for device in inventory if isinstance(device, cls)) for cls in expand(uses)
    }


class Leases:
    """Which run holds each device. Not locked: the executor holds its own
    lock around every call."""

    def __init__(self) -> None:
        self._held: dict[Device, int] = {}

    def check(self, devices: Iterable[Device]) -> None:
        busy = [device.name for device in devices if device in self._held]
        if busy:
            raise DeviceBusy(busy)

    def take(self, run_id: int, devices: Iterable[Device]) -> None:
        for device in devices:
            self._held[device] = run_id

    def release(self, run_id: int) -> None:
        self._held = {device: held for device, held in self._held.items() if held != run_id}
