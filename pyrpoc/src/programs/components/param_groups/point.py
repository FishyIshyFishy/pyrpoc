"""Where a point acquisition parks the galvos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.src.structs.params import Group
from pyrpoc.src.structs.registries import block

from .point_field import Point, point_field


@block
@dataclass
class PointGroup(Group):
    """Where a point acquisition happens.

    Its own block rather than a field on the spectrum block, because what the
    galvos do and what the detector does are configured independently -- and a
    future point program that is not spectroscopy declares this one and not the
    other.
    """

    label: ClassVar[str] = "Point"

    target: Point = point_field(
        "Target",
        tooltip="Galvo position to park at. Pick it off an image, or type volts directly",
    )
