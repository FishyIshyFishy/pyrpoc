"""How many frames a run takes."""

from __future__ import annotations

from pyrpoc.core.program_inputs import int_input

NUM_FRAMES = int_input("num_frames", "Frames", 1, minimum=1, tooltip="Number of frames to capture")
