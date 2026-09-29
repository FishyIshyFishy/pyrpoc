"""Resolving a program's declared devices against what is configured.

Claims propagate up ``backed_by``: claiming the galvo claims its DAQ.
"""

from __future__ import annotations

from pyrpoc.src.structs.device import Device, MissingDevice


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
