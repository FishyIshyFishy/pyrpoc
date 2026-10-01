"""What the second galvo pair plays while the first scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, choice_field, float_field

# "circle" plays sine and cosine on the pair; "raster" copies the scan exactly.
SECOND_PATTERNS = ("circle", "raster")


@block
@dataclass
class SecondGalvosGroup(Group):
    label: ClassVar[str] = "Second Galvos"

    pattern: str = choice_field(
        "Pattern",
        "circle",
        choices=SECOND_PATTERNS,
        tooltip="circle: sine on fast, cosine on slow. raster: a copy of the scan",
    )
    amplitude: float = float_field(
        "Amplitude (V)", 1.0, minimum=0.0, tooltip="Circle radius in volts"
    )
    offset: float = float_field("Offset (V)", 0.0, tooltip="Added to both circle channels")
    frequency_hz: float = float_field(
        "Frequency (Hz)", 100.0, minimum=1e-3, tooltip="Circle revolutions per second"
    )
