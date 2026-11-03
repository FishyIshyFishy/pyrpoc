"""The shape of a simulated frame."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_inputs import ProgramInputValues, int_input

from ..building_blocks.program_inputs import X_PIXELS, Y_PIXELS

SIMULATED_CHANNELS = int_input(
    "simulated_channels",
    "Channels",
    2,
    minimum=1,
    maximum=16,
    tooltip="How many detector channels to fake",
)


@dataclass(frozen=True)
class FrameShape:
    """A simulated frame's width, height and channel count."""

    x_pixels: int
    y_pixels: int
    channels: int

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> FrameShape:
        return cls(values[X_PIXELS], values[Y_PIXELS], values[SIMULATED_CHANNELS])
