"""Waveform generation, sample extraction and frame reshaping.

Moved with the code from tests/modalities/test_daq.py and
test_confocal_acquisition_core.py. The arithmetic is unchanged; only its home
is, and tests/reference/ pins that.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.core.params import ScanGroup
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.programs.hardware import raster
from pyrpoc.programs.hardware.raster import (
    extract_kept_samples,
    generate_raster_waveform,
    reshape_to_frame,
)


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


def test_extract_kept_samples_clips_overscan():
    total_y, total_x, pixel_samples, extra_left, x_pixels = 2, 4, 2, 1, 2
    data = np.arange(total_y * total_x * pixel_samples, dtype=np.float32)
    kept = extract_kept_samples(data, total_y, total_x, pixel_samples, extra_left, x_pixels)
    assert kept.shape == (total_y, x_pixels * pixel_samples)


def test_reshape_to_frame_averages_pixel_samples():
    # one channel, total_y=2, x_pixels=2, pixel_samples=3, all-ones -> mean is 1.0
    scan = np.ones((1, 2, 2 * 3), dtype=np.float32)
    frame = reshape_to_frame(scan, total_y=2, x_pixels=2, pixel_samples=3)
    assert frame.shape == (1, 2, 2)
    assert np.allclose(frame, 1.0)


# --------------------------------------------------------------------------- #
# extract_mask_contexts
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# raster_scan: device config -> NI arguments
# --------------------------------------------------------------------------- #

def wired_devices():
    daq, galvo = DAQ(), Galvo()
    daq.config.device_name = "Dev7"
    daq.config.ai_channels = (2, 5)
    galvo.config.fast_ao = 3
    galvo.config.slow_ao = 4
    return daq, galvo


def fake_run_raster(captured):
    """Stands in for the one function here that needs a real NI card."""

    def run(**kwargs):
        captured.update(kwargs)
        # what run_raster returns: (scan_data, total_y, x_pixels, pixel_samples)
        return np.zeros((2, 3, 4 * 2), dtype=np.float32), 3, 4, 2

    return run


def test_raster_scan_reads_its_wiring_off_the_devices(monkeypatch):
    """The translation this function exists to do, which otherwise only ever
    runs on the instrument.

    The wiring used to arrive as ``**daq.config, **galvo.config`` splatted into
    a fifteen-wide signature, so a renamed field was a TypeError on the first
    frame. Reading it off the device is what makes that a type error instead --
    and this is the check that it is read off the *right* device.
    """
    captured: dict = {}
    monkeypatch.setattr(raster, "run_raster", fake_run_raster(captured))
    daq, galvo = wired_devices()
    scan = ScanGroup(x_pixels=4, y_pixels=3, extra_left=1, extra_right=2, dwell_time_us=2.0)

    frame = raster.raster_scan(
        daq=daq, galvo=galvo, scan=scan, sample_rate_hz=100_000.0, ttl={}
    )

    assert captured["device_name"] == "Dev7"
    assert captured["ai_channels"] == [2, 5]
    assert (captured["fast_ao"], captured["slow_ao"]) == (3, 4)
    assert (captured["x_pixels"], captured["y_pixels"]) == (4, 3)
    assert (captured["extra_left"], captured["extra_right"]) == (1, 2)
    assert captured["dwell_time_us"] == 2.0
    assert captured["sample_rate_hz"] == 100_000.0
    assert frame.shape == (2, 3, 4)


def test_raster_scan_passes_an_empty_ttl_through_rather_than_none(monkeypatch):
    """``ttl`` lost its ``None`` default: "no mask is bound" is now said, not omitted."""
    captured: dict = {}
    monkeypatch.setattr(raster, "run_raster", fake_run_raster(captured))
    daq, galvo = wired_devices()

    raster.raster_scan(
        daq=daq,
        galvo=galvo,
        scan=ScanGroup(x_pixels=4, y_pixels=3, extra_left=1, extra_right=2, dwell_time_us=2.0),
        sample_rate_hz=100_000.0,
        ttl={},
    )
    assert captured["ttl_signals"] == {}


def test_raster_scan_requires_a_ttl_argument():
    daq, galvo = wired_devices()
    with pytest.raises(TypeError):
        raster.raster_scan(  # type: ignore[call-arg]
            daq=daq, galvo=galvo, scan=ScanGroup(), sample_rate_hz=100_000.0
        )


def test_the_waveform_covers_every_ao_sample_of_the_scan(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(raster, "run_raster", fake_run_raster(captured))
    daq, galvo = wired_devices()
    scan = ScanGroup(x_pixels=4, y_pixels=3, extra_left=1, extra_right=2, dwell_time_us=2.0)

    raster.raster_scan(daq=daq, galvo=galvo, scan=scan, sample_rate_hz=100_000.0, ttl={})

    pixel_samples = raster.pixel_samples(scan.dwell_time_us, 100_000.0)
    assert captured["waveform"].shape == (2, scan.total_x * scan.y_pixels * pixel_samples)
