"""How the autofocus search moves z."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.utils.value_rules import float_input

FOCUS_SEARCH_RANGE_UM = float_input(
    "focus_search_range_um",
    "Search Range (um)",
    50.0,
    minimum=0.001,
    step=10.0,
    decimals=3,
    tooltip="The search never moves z further than this from where it started",
)
FOCUS_INITIAL_STEP_UM = float_input(
    "focus_initial_step_um",
    "Initial Step (um)",
    5.0,
    minimum=0.001,
    step=1.0,
    decimals=3,
    tooltip="The first z step; it halves each time neither direction improves focus",
)
FOCUS_FINAL_STEP_UM = float_input(
    "focus_final_step_um",
    "Final Step (um)",
    0.5,
    minimum=0.001,
    step=0.1,
    decimals=3,
    tooltip="The search stops once the step would halve below this",
)


@dataclass(frozen=True)
class FocusSearch:
    """How far the search may move z, and its first and last step."""

    search_range_um: float
    initial_step_um: float
    final_step_um: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> FocusSearch:
        return cls(
            values[FOCUS_SEARCH_RANGE_UM],
            values[FOCUS_INITIAL_STEP_UM],
            values[FOCUS_FINAL_STEP_UM],
        )
