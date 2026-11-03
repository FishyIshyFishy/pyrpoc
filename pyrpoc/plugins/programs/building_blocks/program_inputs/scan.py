"""The scan: how the beam crosses the sample, and ``ScanGeometry``, which turns
those inputs into volts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.program_inputs import (
    ParameterError,
    ProgramInputValues,
    float_input,
    int_input,
    values_from_parameters,
)

X_PIXELS = int_input("scan.x_pixels", "X Pixels", 512, minimum=8, tooltip="Number of pixels in X")
Y_PIXELS = int_input("scan.y_pixels", "Y Pixels", 512, minimum=8, tooltip="Number of pixels in Y")
EXTRA_LEFT = int_input(
    "scan.extra_left",
    "Extra Steps Left",
    300,
    minimum=0,
    tooltip="Extra scan steps at the left edge",
)
EXTRA_RIGHT = int_input(
    "scan.extra_right",
    "Extra Steps Right",
    20,
    minimum=0,
    tooltip="Extra scan steps at the right edge",
)
FAST_AXIS_OFFSET = float_input(
    "scan.fast_axis_offset", "Fast Axis Offset", 0.0, tooltip="Fast-axis offset"
)
FAST_AXIS_AMPLITUDE = float_input(
    "scan.fast_axis_amplitude",
    "Fast Axis Amplitude",
    1.0,
    minimum=1e-6,
    tooltip="Fast-axis amplitude",
)
SLOW_AXIS_OFFSET = float_input(
    "scan.slow_axis_offset", "Slow Axis Offset", 0.0, tooltip="Slow-axis offset"
)
SLOW_AXIS_AMPLITUDE = float_input(
    "scan.slow_axis_amplitude",
    "Slow Axis Amplitude",
    1.0,
    minimum=1e-6,
    tooltip="Slow-axis amplitude",
)
DWELL_TIME_US = float_input(
    "scan.dwell_time_us", "Dwell Time (us)", 2.0, minimum=0.1, tooltip="Pixel dwell time"
)

SCAN_SECTION = InputSection(
    "Scan",
    (
        X_PIXELS,
        Y_PIXELS,
        EXTRA_LEFT,
        EXTRA_RIGHT,
        FAST_AXIS_OFFSET,
        FAST_AXIS_AMPLITUDE,
        SLOW_AXIS_OFFSET,
        SLOW_AXIS_AMPLITUDE,
        DWELL_TIME_US,
    ),
)


@dataclass(frozen=True)
class ScanGeometry:
    """How the beam is scanned: the pixel grid, the overscan either side, the
    volts each axis spans, and the dwell per pixel."""

    x_pixels: int
    y_pixels: int
    extra_left: int
    extra_right: int
    fast_axis_offset: float
    fast_axis_amplitude: float
    slow_axis_offset: float
    slow_axis_amplitude: float
    dwell_time_us: float

    @classmethod
    def from_inputs(cls, values: ProgramInputValues) -> ScanGeometry:
        return cls(
            x_pixels=values[X_PIXELS],
            y_pixels=values[Y_PIXELS],
            extra_left=values[EXTRA_LEFT],
            extra_right=values[EXTRA_RIGHT],
            fast_axis_offset=values[FAST_AXIS_OFFSET],
            fast_axis_amplitude=values[FAST_AXIS_AMPLITUDE],
            slow_axis_offset=values[SLOW_AXIS_OFFSET],
            slow_axis_amplitude=values[SLOW_AXIS_AMPLITUDE],
            dwell_time_us=values[DWELL_TIME_US],
        )

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any], source: str) -> ScanGeometry:
        """The scan a recording's ``parameters`` describe. Raises ``ParameterError``
        naming ``source`` if they describe none."""
        if any(each.key not in parameters for each in SCAN_SECTION.inputs):
            raise ParameterError(
                f"{source} was not acquired with a scan geometry, "
                "so a pixel does not name a position"
            )
        return cls.from_inputs(values_from_parameters(SCAN_SECTION.inputs, parameters))

    @property
    def total_x(self) -> int:
        return self.x_pixels + self.extra_left + self.extra_right

    @property
    def kept_columns(self) -> slice:
        """The scanned columns that are displayed, between the overscan."""
        return slice(self.extra_left, self.extra_left + self.x_pixels)

    def fast_volts(self, columns: np.ndarray) -> np.ndarray:
        """Fast-axis volts at displayed ``columns``, where columns left of 0 or
        past ``x_pixels`` are overscan at the same pitch."""
        fast_step = 2.0 * self.fast_axis_amplitude / self.x_pixels
        return self.fast_axis_offset - self.fast_axis_amplitude + columns * fast_step

    def slow_volts(self, rows: np.ndarray) -> np.ndarray:
        """Slow-axis volts at displayed ``rows``."""
        relative = -1.0 + 2.0 * rows / self.y_pixels
        return self.slow_axis_offset + relative * self.slow_axis_amplitude

    def voltage_at(self, x: int, y: int) -> tuple[float, float]:
        """The (fast, slow) volts at which displayed pixel ``(x, y)`` was sampled."""
        # Displayed frames are already cropped of overscan, and each pixel is held
        # at one voltage for its whole dwell, so there is no half-pixel centre to add.
        return float(self.fast_volts(np.asarray(x))), float(self.slow_volts(np.asarray(y)))
