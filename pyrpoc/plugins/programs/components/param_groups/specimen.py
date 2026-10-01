"""What a simulated mosaic images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, float_field


@block
@dataclass
class SpecimenGroup(Group):
    """A simulated slide: how large a pixel is on it, standing in for the
    objective, and how noisy the detector is."""

    label: ClassVar[str] = "Specimen"

    um_per_pixel: float = float_field(
        "Pixel Size (um)",
        0.5,
        minimum=0.001,
        step=0.1,
        decimals=4,
        tooltip="Stage distance one pixel spans; with the spacing, sets the tile overlap",
    )
    noise_level: float = float_field(
        "Noise Level", 0.03, minimum=0.0, step=0.01, tooltip="Gaussian noise added per pixel"
    )
