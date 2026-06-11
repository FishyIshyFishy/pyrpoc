from __future__ import annotations

import numpy as np

from pyrpoc.modalities.helpers.daq import generate_raster_waveform


def make(**overrides):
    kwargs = dict(
        x_pixels=4,
        extra_left=1,
        extra_right=2,
        y_pixels=3,
        pixel_samples=2,
        fast_axis_offset=0.0,
        fast_axis_amplitude=1.0,
        slow_axis_offset=0.0,
        slow_axis_amplitude=1.0,
    )
    kwargs.update(overrides)
    return generate_raster_waveform(**kwargs)


def test_waveform_shape_and_dtype():
    wf = make()
    total_x = 1 + 4 + 2
    expected_len = total_x * 2 * 3  # total_x * pixel_samples * y_pixels
    assert wf.shape == (2, expected_len)
    assert wf.dtype == np.float64


def test_slow_axis_is_constant_within_each_line():
    total_x, pixel_samples, y_pixels = 7, 2, 3
    wf = make()
    slow = wf[1]
    line_len = total_x * pixel_samples
    for line in range(y_pixels):
        segment = slow[line * line_len:(line + 1) * line_len]
        assert np.allclose(segment, segment[0])


def test_slow_axis_increases_across_lines():
    wf = make()
    total_x, pixel_samples = 7, 2
    line_len = total_x * pixel_samples
    first_line = wf[1][0]
    second_line = wf[1][line_len]
    assert second_line > first_line


def test_zero_amplitude_is_clamped(monkeypatch):
    # amplitude 0 would divide by zero in the step; the function clamps to 1e-6.
    wf = make(fast_axis_amplitude=0.0, slow_axis_amplitude=0.0)
    assert np.all(np.isfinite(wf))


def test_offset_shifts_fast_axis():
    base = make(fast_axis_offset=0.0)
    shifted = make(fast_axis_offset=5.0)
    assert np.allclose(shifted[0] - base[0], 5.0)
