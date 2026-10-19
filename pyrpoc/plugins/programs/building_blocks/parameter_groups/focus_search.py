"""How an autofocus searches in z: how far it may go, the steps of its coarse
and fine passes, and how much a peak must stand out to count as focus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, float_field


@block
@dataclass
class FocusSearchGroup(Group):
    label: ClassVar[str] = "Focus Search"

    search_range_um: float = float_field(
        "Search Range (um)",
        50.0,
        minimum=0.001,
        step=10.0,
        decimals=3,
        tooltip="The search never moves z further than this from where it started",
    )
    coarse_step_um: float = float_field(
        "Coarse Step (um)",
        5.0,
        minimum=0.001,
        step=1.0,
        decimals=3,
        tooltip="Step of the outward search; keep it under the depth over which "
        "the sample looks in focus, or a narrow focus can be stepped over",
    )
    fine_step_um: float = float_field(
        "Fine Step (um)",
        0.5,
        minimum=0.001,
        step=0.1,
        decimals=3,
        tooltip="Step of the final sweep across the peak; sets the precision",
    )
    min_contrast: float = float_field(
        "Min Contrast",
        2.0,
        minimum=1.0,
        step=0.5,
        decimals=2,
        tooltip="A peak counts as focus only if its score is at least this many "
        "times the lowest score seen",
    )
