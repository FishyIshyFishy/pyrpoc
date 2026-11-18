"""``CameraSettings``: how the camera takes each frame."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.program_input import InputReader
from pyrpoc.plugins.program_io.inputs import CAMERA_BINNED, CAMERA_EXPOSURE_MS, CAMERA_GAIN_PCT


@dataclass(frozen=True)
class CameraSettings:
    """How the camera takes each frame."""

    exposure_ms: float
    gain_pct: int
    binned: bool

    @classmethod
    def from_inputs(cls, values: InputReader) -> CameraSettings:
        return cls(values[CAMERA_EXPOSURE_MS], values[CAMERA_GAIN_PCT], values[CAMERA_BINNED])
