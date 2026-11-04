"""The DAQ's sample rate, in MHz."""

from __future__ import annotations

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.utils.value_rules import float_input

SAMPLE_RATE_MHZ = float_input(
    "sample_rate_mhz",
    "Sample Rate (MHz)",
    1.0,
    minimum=0.000001,
    maximum=5.0,
    step=0.1,
    tooltip="DAQ AO sample rate in MHz; the FLIM pixel clock divides down from it",
)


def rate_hz(inputs: ProgramInputValues) -> float:
    """The sample rate input, in the Hz the DAQ is given."""
    return inputs[SAMPLE_RATE_MHZ] * 1e6
