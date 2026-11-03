"""How long a simulated frame takes."""

from __future__ import annotations

from pyrpoc.core.program_inputs import int_input

FRAME_INTERVAL_MS = int_input(
    "frame_interval_ms",
    "Frame Interval (ms)",
    100,
    minimum=0,
    tooltip="Pause between frames, standing in for acquisition time",
)
