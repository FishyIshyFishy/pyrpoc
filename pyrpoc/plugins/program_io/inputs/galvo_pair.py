"""Two galvos on one sample: which of them scans or parks, and ``Registration``, where the
second points relative to the reference."""

# Every scan setting, image and target is in the reference galvo's volts, so images from
# either galvo line up and a pixel on one names the same spot for the other.

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.utils.value_rules import choice_input, float_input

REFERENCE = "Reference"
SECOND = "Second"
BOTH = "Both"

SCAN_GALVOS = choice_input(
    "scan_galvos",
    "Scan with",
    REFERENCE,
    choices=(REFERENCE, SECOND, BOTH),
    tooltip=(
        "Which galvo scans. With Both, the second follows the reference, shifted by its offset. "
        "The galvo that does not scan stays where it was parked."
    ),
)
PINPOINT_GALVO = choice_input(
    "pinpoint_galvo",
    "Galvo",
    SECOND,
    choices=(SECOND, REFERENCE),
    tooltip="Which galvo to park. Pick its target on an image taken with the other",
)
SECOND_OFFSET_FAST_V = float_input(
    "second_offset_fast_v",
    "Second Offset Fast (V)",
    0.0,
    step=0.001,
    decimals=4,
    tooltip="The second galvo's fast-axis volts minus the reference's, at the same spot",
)
SECOND_OFFSET_SLOW_V = float_input(
    "second_offset_slow_v",
    "Second Offset Slow (V)",
    0.0,
    step=0.001,
    decimals=4,
    tooltip="The second galvo's slow-axis volts minus the reference's, at the same spot",
)

# The inputs ``Registration`` is made from.
REGISTRATION_INPUTS = (SECOND_OFFSET_FAST_V, SECOND_OFFSET_SLOW_V)


@dataclass(frozen=True)
class Registration:
    """How far the second galvo's volts are from the reference's at the same spot on the
    sample."""

    fast_v: float
    slow_v: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> Registration:
        return cls(values[SECOND_OFFSET_FAST_V], values[SECOND_OFFSET_SLOW_V])

    def offset_of(self, galvo: str) -> tuple[float, float]:
        """What to add to reference volts to point ``galvo`` (``REFERENCE`` or ``SECOND``) at
        the same spot."""
        return (0.0, 0.0) if galvo == REFERENCE else (self.fast_v, self.slow_v)
