"""Masks driving digital lines during a scan. Shared across modalities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block

from .mask_field import Mask, masks_field


@block
@dataclass
class ModulationGroup(Group):
    label: ClassVar[str] = "Modulation"

    masks: tuple[Mask, ...] = masks_field(
        "Masks",
        tooltip="Masks driving digital output lines during the scan. "
        "Draw one in the Mask Editor to add it here",
    )
