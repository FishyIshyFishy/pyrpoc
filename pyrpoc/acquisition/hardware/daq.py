from __future__ import annotations

import numpy as np


class DaqUnavailableError(RuntimeError):
    """Raised when a DAQ-backed acquisition cannot run on this machine."""


def generate_raster_waveform(
    x_pixels: int,
    extra_left: int,
    extra_right: int,
    y_pixels: int,
    pixel_samples: int,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
) -> np.ndarray:
    """Build the (2, N) galvo waveform for one raster frame.

    1. total pixels per line includes the extra left/right overscan
    2. per-line amplitude pads the voltage step to the left/right of the
       offset-amp and offset+amp points
    3. tile the fast axis per line, repeat the slow axis per line
    """
    total_x = extra_left + x_pixels + extra_right
    fast_amp = max(float(fast_axis_amplitude), 1e-6)
    slow_amp = max(float(slow_axis_amplitude), 1e-6)
    fast_step = (2.0 * fast_amp) / float(x_pixels)
    fast_start = -fast_amp - (float(extra_left) * fast_step)
    fast_axis = fast_start + (np.arange(total_x, dtype=np.float32) * fast_step) + float(fast_axis_offset)
    slow_axis = (
        np.linspace(-1.0, 1.0, y_pixels, endpoint=False, dtype=np.float32) * slow_amp
        + float(slow_axis_offset)
    )
    fast_raster = np.tile(np.repeat(fast_axis, pixel_samples), y_pixels)
    slow_raster = np.repeat(slow_axis, total_x * pixel_samples)
    return np.vstack((fast_raster, slow_raster)).astype(np.float64)
