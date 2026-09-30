"""FLIM's frame trigger and pixel clock lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.params import Group, block, int_field


@block
@dataclass
class TriggerGroup(Group):
    label: ClassVar[str] = "Triggers"

    frame_trigger_pfi: int = int_field(
        "Frame Trigger PFI Line",
        0,
        minimum=0,
        tooltip="PFI line that exports the AO start trigger (frame marker)",
    )
    pixel_clock_ctr: int = int_field(
        "Pixel Clock Counter", 0, minimum=0, tooltip="Counter used to generate the pixel clock"
    )
    pixel_clock_pfi: int = int_field(
        "Pixel Clock PFI Line", 1, minimum=0, tooltip="PFI line that outputs the pixel clock"
    )
