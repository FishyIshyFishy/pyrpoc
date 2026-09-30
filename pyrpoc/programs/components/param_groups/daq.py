"""The DAQ's sample clock. Shared across modalities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.params import Group, block, float_field


@block
@dataclass
class DaqGroup(Group):
    label: ClassVar[str] = "DAQ"

    sample_rate_hz: float = float_field(
        "Sample Rate (Hz)",
        1_000_000.0,
        minimum=1.0,
        maximum=5_000_000.0,
        step=1_000.0,
        tooltip="DAQ AO sample rate in Hz; the FLIM pixel clock divides down from it",
    )
