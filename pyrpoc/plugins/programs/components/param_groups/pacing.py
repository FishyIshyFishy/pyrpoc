"""How long a simulated acquisition takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, int_field


@block
@dataclass
class PacingGroup(Group):
    """Simulation's stand-in for how long an acquisition takes."""

    label: ClassVar[str] = "Pacing"

    frame_interval_ms: int = int_field(
        "Frame Interval (ms)",
        100,
        minimum=0,
        tooltip="Pause between frames, standing in for acquisition time",
    )
