"""The DAQ lines that carry FLIM's frame marker and pixel clock."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import ProgramInputValues, int_input

FRAME_TRIGGER_PFI = int_input(
    "trigger.frame_trigger_pfi",
    "Frame Trigger PFI Line",
    0,
    minimum=0,
    tooltip="PFI line that exports the AO start trigger (frame marker)",
)
PIXEL_CLOCK_CTR = int_input(
    "trigger.pixel_clock_ctr",
    "Pixel Clock Counter",
    0,
    minimum=0,
    tooltip="Counter used to generate the pixel clock",
)
PIXEL_CLOCK_PFI = int_input(
    "trigger.pixel_clock_pfi",
    "Pixel Clock PFI Line",
    1,
    minimum=0,
    tooltip="PFI line that outputs the pixel clock",
)

TRIGGER_SECTION = InputSection("Triggers", (FRAME_TRIGGER_PFI, PIXEL_CLOCK_CTR, PIXEL_CLOCK_PFI))


@dataclass(frozen=True)
class TriggerLines:
    """The PFI line for the frame marker, and the counter and PFI line for the
    pixel clock."""

    frame_trigger_pfi: int
    pixel_clock_ctr: int
    pixel_clock_pfi: int

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> TriggerLines:
        return cls(values[FRAME_TRIGGER_PFI], values[PIXEL_CLOCK_CTR], values[PIXEL_CLOCK_PFI])
