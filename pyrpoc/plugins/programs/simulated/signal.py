"""What the simulated detector sees."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import ProgramInputValues, choice_input, float_input

# Pattern names offered by the simulated program, in menu order; the simulation
# keys its plane functions by these.
PATTERNS = ("cells", "rings", "gradient", "checkerboard", "flat")

PATTERN = choice_input(
    "signal.pattern",
    "Pattern",
    "cells",
    choices=PATTERNS,
    tooltip="cells drift like a sample, the rest are test targets",
)
SIGNAL_LEVEL = float_input(
    "signal.signal_level", "Signal Level", 1.0, minimum=0.0, tooltip="Peak brightness before noise"
)
NOISE_LEVEL = float_input(
    "signal.noise_level",
    "Noise Level",
    0.03,
    minimum=0.0,
    step=0.01,
    tooltip="Gaussian noise added per pixel",
)
DRIFT_PIXELS_PER_FRAME = float_input(
    "signal.drift_pixels_per_frame",
    "Drift (px/frame)",
    1.5,
    minimum=0.0,
    tooltip="How far the pattern moves each frame",
)
MASK_GAIN = float_input(
    "signal.mask_gain",
    "Mask Gain",
    0.5,
    minimum=0.0,
    tooltip="Extra brightness inside bound masks, standing in for stimulation",
)

SIGNAL_SECTION = InputSection(
    "Signal", (PATTERN, SIGNAL_LEVEL, NOISE_LEVEL, DRIFT_PIXELS_PER_FRAME, MASK_GAIN)
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
            values[PATTERN],
            values[SIGNAL_LEVEL],
            values[NOISE_LEVEL],
            values[DRIFT_PIXELS_PER_FRAME],
            values[MASK_GAIN],
        )
