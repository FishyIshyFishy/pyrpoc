"""Where a point acquisition parks the galvos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block

from .point_field import Point, point_field


@block
@dataclass
class PointGroup(Group):
    """Where a point acquisition happens. Separate from the detector's block
    because the galvos and the detector are configured independently."""

    label: ClassVar[str] = "Point"

    target: Point = point_field(
        "Target",
        tooltip="Galvo position to park at. Pick it off an image, or type volts directly",
    )
