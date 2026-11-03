"""The DAQ's sample rate."""

from __future__ import annotations

from pyrpoc.core.program_inputs import float_input

SAMPLE_RATE_HZ = float_input(
    "sample_rate_hz",
    "Sample Rate (Hz)",
    1_000_000.0,
    minimum=1.0,
    maximum=5_000_000.0,
    step=1_000.0,
    tooltip="DAQ AO sample rate in Hz; the FLIM pixel clock divides down from it",
)
