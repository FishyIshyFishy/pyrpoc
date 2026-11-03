"""How many frames a run takes."""

from __future__ import annotations

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import int_input

# Its own section rather than part of the scan, so a program that images once
# per position, like a mosaic, can declare the scan without it.
NUM_FRAMES = int_input(
    "frame_count.num_frames", "Frames", 1, minimum=1, tooltip="Number of frames to capture"
)

FRAME_COUNT_SECTION = InputSection("Frames", (NUM_FRAMES,))
