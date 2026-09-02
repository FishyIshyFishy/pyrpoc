"""Resolving a program's declared devices against what is configured.

Claims propagate up ``backed_by``: claiming the galvo claims its DAQ, because
the galvo is voltages on the DAQ's AO channels. Only one program runs at a time
today, so this does not yet arbitrate between competing claims -- but it is the
place that would, and it is where v3.0's ``validate_required_instruments``
lands.
"""

from __future__ import annotations

from pyrpoc.core.errors import MissingDevice
from pyrpoc.devices.base import Device


def expand(uses: list[type[Device]]) -> list[type[Device]]:
    """Every device class implied by ``uses``, following ``backed_by``.

    Declaration order first, then each backing device, with no duplicates.
    """
    ordered: list[type[Device]] = []

    def add(cls: type[Device]) -> None:
        if cls in ordered:
            return
        ordered.append(cls)
        backing = getattr(cls, "backed_by", None)
        if backing is not None:
            add(backing)

    for cls in uses:
        add(cls)
    return ordered


def missing(uses: list[type[Device]], inventory: list[Device]) -> list[type[Device]]:
    """Which required device classes have no instance configured."""
    return [
        cls
        for cls in expand(uses)
        if not any(isinstance(device, cls) for device in inventory)
    ]


def resolve(uses: list[type[Device]], inventory: list[Device]) -> dict[type[Device], Device]:
    """Bind each required class to an instance, or raise naming what is absent."""
    absent = missing(uses, inventory)
    if absent:
        raise MissingDevice([cls.display_name for cls in absent])

    bound: dict[type[Device], Device] = {}
    for cls in expand(uses):
        bound[cls] = next(device for device in inventory if isinstance(device, cls))
    return bound
