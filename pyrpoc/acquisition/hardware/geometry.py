from __future__ import annotations


def pixel_to_voltage(
    px: float,
    py: float,
    *,
    x_pixels: int,
    y_pixels: int,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
) -> tuple[float, float]:
    """Map a displayed-image pixel (px, py) to galvo (fast, slow) volts.

    Inverse of the per-pixel mapping baked into ``generate_raster_waveform``:
    the displayed frame has the overscan columns trimmed, so a displayed
    column ``px`` sits at fast voltage ``offset - amp + px * (2*amp / x_pixels)``
    and likewise for the slow axis. This is the single owner of the
    pixel->voltage transform (used by the click-to-point source); nothing
    else should re-derive galvo geometry.
    """
    fast_amp = max(float(fast_axis_amplitude), 1e-6)
    slow_amp = max(float(slow_axis_amplitude), 1e-6)
    vx = float(fast_axis_offset) - fast_amp + (float(px) * (2.0 * fast_amp / float(x_pixels)))
    vy = float(slow_axis_offset) - slow_amp + (float(py) * (2.0 * slow_amp / float(y_pixels)))
    return vx, vy
