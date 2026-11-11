"""The delay sweep of hyperspectral imaging: where the delay stage starts and stops, and how many
images it takes on the way, and ``DelaySweep``, which lays out the stops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.utils.value_rules import float_input, int_input

DELAY_START_UM = float_input(
    "delay_start_um",
    "Start (um)",
    20000.0,
    step=100.0,
    decimals=3,
    tooltip="Delay stage position of the first image",
)
DELAY_STOP_UM = float_input(
    "delay_stop_um",
    "Stop (um)",
    30000.0,
    step=100.0,
    decimals=3,
    tooltip="Delay stage position of the last image; below Start, the sweep runs backwards",
)
DELAY_STEPS = int_input(
    "delay_steps",
    "Steps",
    10,
    minimum=1,
    tooltip="Images in the sweep, evenly spaced from Start to Stop, both included",
)


@dataclass(frozen=True)
class DelaySweep:
    """Where the delay stage starts and stops, and how many images it takes."""

    start_um: float
    stop_um: float
    steps: int

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> DelaySweep:
        return cls(values[DELAY_START_UM], values[DELAY_STOP_UM], values[DELAY_STEPS])

    def positions_um(self) -> tuple[float, ...]:
        """Each image's stage position, in sweep order; one step is the start alone."""
        return tuple(float(p) for p in np.linspace(self.start_um, self.stop_um, self.steps))
