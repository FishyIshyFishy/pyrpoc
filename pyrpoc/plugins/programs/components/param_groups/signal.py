"""What the simulated detector sees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, choice_field, float_field

# Pattern names offered by the simulated program, in menu order.
PATTERNS = ("cells", "rings", "gradient", "checkerboard", "flat")


@block
@dataclass
class SignalGroup(Group):
    """What the fake detector sees."""

    label: ClassVar[str] = "Signal"

    pattern: str = choice_field(
        "Pattern",
        "cells",
        choices=PATTERNS,
        tooltip="cells drift like a sample, the rest are test targets",
    )
    signal_level: float = float_field(
        "Signal Level", 1.0, minimum=0.0, tooltip="Peak brightness before noise"
    )
    noise_level: float = float_field(
        "Noise Level", 0.03, minimum=0.0, step=0.01, tooltip="Gaussian noise added per pixel"
    )
    drift_pixels_per_frame: float = float_field(
        "Drift (px/frame)", 1.5, minimum=0.0, tooltip="How far the pattern moves each frame"
    )
    mask_gain: float = float_field(
        "Mask Gain",
        0.5,
        minimum=0.0,
        tooltip="Extra brightness inside bound masks, standing in for stimulation",
    )
