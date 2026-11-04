"""What the simulated detector sees."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.utils.value_rules import choice_input, float_input

from .synthetic_detector import NOISE_LEVEL

# Pattern names offered by the simulated program, in menu order; the simulation
# keys its plane functions by these.
PATTERNS = ("cells", "rings", "gradient", "checkerboard", "flat")

SIGNAL_PATTERN = choice_input(
    "signal_pattern",
    "Pattern",
    "cells",
    choices=PATTERNS,
    tooltip="cells drift like a sample, the rest are test targets",
)
SIGNAL_LEVEL = float_input(
    "signal_level", "Signal Level", 1.0, minimum=0.0, tooltip="Peak brightness before noise"
)
DRIFT_PIXELS_PER_FRAME = float_input(
    "drift_pixels_per_frame",
    "Drift (px/frame)",
    1.5,
    minimum=0.0,
    tooltip="How far the pattern moves each frame",
)
MASK_GAIN = float_input(
    "mask_gain",
    "Mask Gain",
    0.5,
    minimum=0.0,
    tooltip="Extra brightness inside bound masks, standing in for stimulation",
)


@dataclass(frozen=True)
class SyntheticSignal:
    """The fake detector's pattern, brightness, noise, drift, and the extra
    brightness under a mask."""

    pattern: str
    signal_level: float
    noise_level: float
    drift_pixels_per_frame: float
    mask_gain: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> SyntheticSignal:
        return cls(
            values[SIGNAL_PATTERN],
            values[SIGNAL_LEVEL],
            values[NOISE_LEVEL],
            values[DRIFT_PIXELS_PER_FRAME],
            values[MASK_GAIN],
        )
