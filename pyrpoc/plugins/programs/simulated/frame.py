"""The shape of a simulated frame."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import ProgramInputValues, int_input

X_PIXELS = int_input("frame.x_pixels", "X Pixels", 256, minimum=8, tooltip="Frame width in pixels")
Y_PIXELS = int_input("frame.y_pixels", "Y Pixels", 256, minimum=8, tooltip="Frame height in pixels")
CHANNELS = int_input(
    "frame.channels",
    "Channels",
    2,
    minimum=1,
    maximum=16,
    tooltip="How many detector channels to fake",
)

FRAME_SECTION = InputSection("Frame", (X_PIXELS, Y_PIXELS, CHANNELS))


@dataclass(frozen=True)
class FrameShape:
    """A simulated frame's width, height and channel count."""

    x_pixels: int
    y_pixels: int
    channels: int

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> FrameShape:
        return cls(values[X_PIXELS], values[Y_PIXELS], values[CHANNELS])
