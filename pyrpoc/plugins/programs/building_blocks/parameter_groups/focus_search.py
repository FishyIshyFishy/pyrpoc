"""How an autofocus searches in z: how far it may go, how it starts stepping,
and when the gain is too small to keep refining."""

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
    initial_step_um: float = float_field(
        "Initial Step (um)",
        5.0,
        minimum=0.001,
        step=1.0,
        decimals=3,
        tooltip="The first z step; it halves each time the climb overshoots",
    )
    plateau_pct: float = float_field(
        "Plateau (%)",
        1.0,
        minimum=0.0,
        step=0.5,
        decimals=3,
        tooltip="Stop once a halving of the step raises the focus metric by less than this",
    )
