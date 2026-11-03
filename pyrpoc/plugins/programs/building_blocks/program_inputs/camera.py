"""The camera's run settings: exposure, gain and binning."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import ProgramInputValues, bool_input, float_input, int_input

# Program inputs rather than device configuration: they change with the
# sample, not with how the camera is wired.
EXPOSURE_MS = float_input(
    "camera.exposure_ms",
    "Exposure (ms)",
    10.0,
    minimum=0.001,
    decimals=3,
    tooltip="Exposure time per frame",
)
GAIN_PCT = int_input(
    "camera.gain_pct",
    "Gain (%)",
    100,
    minimum=100,
    tooltip="Analog gain; 100 is unity, the camera sets the top",
)
BINNED = bool_input(
    "camera.binned",
    "Bin 2x2",
    False,
    tooltip="Half the width and height, for faster and brighter frames",
)

CAMERA_SECTION = InputSection("Camera", (EXPOSURE_MS, GAIN_PCT, BINNED))


@dataclass(frozen=True)
class CameraSettings:
    """How the camera takes each frame."""

    exposure_ms: float
    gain_pct: int
    binned: bool

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> CameraSettings:
        return cls(values[EXPOSURE_MS], values[GAIN_PCT], values[BINNED])
