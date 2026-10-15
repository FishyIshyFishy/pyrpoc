"""How a camera exposes each frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, bool_field, float_field, int_field


@block
@dataclass
class CameraGroup(Group):
    """Run parameters rather than device configuration, because they change
    with the sample, not with how the camera is wired."""

    label: ClassVar[str] = "Camera"

    exposure_ms: float = float_field(
        "Exposure (ms)", 10.0, minimum=0.001, decimals=3, tooltip="Exposure time per frame"
    )
    gain_pct: int = int_field(
        "Gain (%)", 100, minimum=100, tooltip="Analog gain; 100 is unity, the camera sets the top"
    )
    binned: bool = bool_field(
        "Bin 2x2", False, tooltip="Half the width and height, for faster and brighter frames"
    )
