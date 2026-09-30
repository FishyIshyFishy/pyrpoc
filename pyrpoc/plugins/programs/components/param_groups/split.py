"""Split confocal's subpixel windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.params import Group, block, int_field


@block
@dataclass
class SplitGroup(Group):
    label: ClassVar[str] = "Split"

    t0_samples: int = int_field(
        "t0 Samples", 1, minimum=1, tooltip="Number of samples in the first subpixel window"
    )
    t1_samples: int = int_field(
        "t1 Samples", 0, minimum=0, tooltip="Number of samples to discard between t0 and t2"
    )
