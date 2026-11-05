"""``DeviceChoices``: which device the user chose for each name programs choose a device under."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pyrpoc.core.device import Device


class DeviceChoices:
    """The device picked for each name, by its id: one choice per name, shared by every program
    that chooses under it."""

    def __init__(self) -> None:
        self._picked: dict[str, str] = {}

    def pick(self, name: str, device: Device) -> None:
        self._picked[name] = device.instance_id

    def chosen(
        self, chooses: Mapping[str, type[Device]], inventory: list[Device]
    ) -> dict[str, Device]:
        """The device each of ``chooses`` stands for now: the one picked for it, or else the
        first of its class that none of the others has. A name with neither is left out."""
        # Picks first, so a name left to its default never takes a device picked for another.
        chosen: dict[str, Device] = {}
        for name, device_type in chooses.items():
            picked = next(
                (
                    device
                    for device in inventory
                    if isinstance(device, device_type)
                    and device.instance_id == self._picked.get(name)
                ),
                None,
            )
            if picked is not None:
                chosen[name] = picked
        for name, device_type in chooses.items():
            if name in chosen:
                continue
            free = next(
                (
                    device
                    for device in inventory
                    if isinstance(device, device_type) and device not in chosen.values()
                ),
                None,
            )
            if free is not None:
                chosen[name] = free
        return {name: chosen[name] for name in chooses if name in chosen}

    def to_dict(self) -> dict[str, str]:
        return dict(self._picked)

    def load_dict(self, raw: Mapping[str, Any]) -> None:
        """Take the picks from saved JSON, skipping any that is not a device id."""
        self._picked = {str(name): value for name, value in raw.items() if isinstance(value, str)}
