"""The scan, read from its program inputs: ``ScanGeometry``, which turns them into volts, the
DAQ's rate and FLIM's trigger lines, and the focus search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyrpoc.core.program_input import (
    InputReader,
    ParameterError,
    ProgramInput,
    ProgramInputValues,
)
from pyrpoc.plugins.program_io.inputs import (
    DWELL_TIME_US,
    EXTRA_LEFT,
    EXTRA_RIGHT,
    FAST_AXIS_AMPLITUDE,
    FAST_AXIS_OFFSET,
    FOCUS_FINAL_STEP_UM,
    FOCUS_INITIAL_STEP_UM,
    FOCUS_SEARCH_RANGE_UM,
    FRAME_TRIGGER_PFI,
    PIXEL_CLOCK_CTR,
    PIXEL_CLOCK_PFI,
    SAMPLE_RATE_MHZ,
    SLOW_AXIS_AMPLITUDE,
    SLOW_AXIS_OFFSET,
    X_PIXELS,
    Y_PIXELS,
)


def values_from_parameters(
    inputs: Sequence[ProgramInput[Any]], parameters: Mapping[str, Any]
) -> ProgramInputValues:
    """``inputs`` as a recording's ``parameters`` hold them, which must name every one."""
    return ProgramInputValues({each.key: each.rule.decode(parameters[each.key]) for each in inputs})


def rate_hz(inputs: InputReader) -> float:
    """The sample rate input, in the Hz the DAQ is given."""
    return inputs[SAMPLE_RATE_MHZ] * 1e6


# The inputs ``ScanGeometry`` is made from.
SCAN_GEOMETRY_INPUTS = (
    X_PIXELS,
    Y_PIXELS,
    EXTRA_LEFT,
    EXTRA_RIGHT,
    FAST_AXIS_OFFSET,
    FAST_AXIS_AMPLITUDE,
    SLOW_AXIS_OFFSET,
    SLOW_AXIS_AMPLITUDE,
    DWELL_TIME_US,
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
    def from_inputs(cls, values: InputReader) -> ScanGeometry:
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
        if any(each.key not in parameters for each in SCAN_GEOMETRY_INPUTS):
            raise ParameterError(
                f"{source} was not acquired with a scan geometry, "
                "so a pixel does not name a position"
            )
        return cls.from_inputs(values_from_parameters(SCAN_GEOMETRY_INPUTS, parameters))

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


@dataclass(frozen=True)
class TriggerLines:
    """The PFI line for the frame marker, and the counter and PFI line for the
    pixel clock."""

    frame_trigger_pfi: int
    pixel_clock_ctr: int
    pixel_clock_pfi: int

    @classmethod
    def from_inputs(cls, values: InputReader) -> TriggerLines:
        return cls(values[FRAME_TRIGGER_PFI], values[PIXEL_CLOCK_CTR], values[PIXEL_CLOCK_PFI])


@dataclass(frozen=True)
class FocusSearch:
    """How far the search may move z, and its first and last step."""

    search_range_um: float
    initial_step_um: float
    final_step_um: float

    @classmethod
    def from_inputs(cls, values: InputReader) -> FocusSearch:
        return cls(
            values[FOCUS_SEARCH_RANGE_UM],
            values[FOCUS_INITIAL_STEP_UM],
            values[FOCUS_FINAL_STEP_UM],
        )
