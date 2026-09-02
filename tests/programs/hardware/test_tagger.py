from __future__ import annotations

import types

import numpy as np
import pytest

from pyrpoc.core.errors import DaqError
from pyrpoc.core.params import ScanGroup, TriggerGroup
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.programs.hardware import tagger as acq


def wired(*, device_name="Dev1", fast_ao=0, slow_ao=1):
    """A DAQ and a galvo carrying the wiring flim_scan reads off them."""
    daq, galvo = DAQ(), Galvo()
    daq.config.device_name = device_name
    galvo.config.fast_ao = fast_ao
    galvo.config.slow_ao = slow_ao
    return daq, galvo


# --------------------------------------------------------------------------- #
# reshape_flim_frame / flim_intensity (pure logic)
# --------------------------------------------------------------------------- #

def test_reshape_flim_frame_folds_and_clips_overscan():
    y_pixels, x_pixels, extra_left, extra_right, n_bins = 2, 3, 1, 2, 4
    total_x = extra_left + x_pixels + extra_right  # 6
    n_pixels = total_x * y_pixels
    # pixel p gets a histogram filled with the value p so we can track it
    flat = np.repeat(np.arange(n_pixels, dtype=np.float32)[:, None], n_bins, axis=1)

    cube = acq.reshape_flim_frame(flat, n_bins, y_pixels, total_x, extra_left, x_pixels)

    assert cube.shape == (y_pixels, x_pixels, n_bins)
    # row 0 keeps total_x columns extra_left..extra_left+x_pixels => pixels 1,2,3
    assert [cube[0, c, 0] for c in range(x_pixels)] == [1.0, 2.0, 3.0]
    # row 1 starts at pixel total_x => clipped cols are total_x+1 .. = 7,8,9
    assert [cube[1, c, 0] for c in range(x_pixels)] == [7.0, 8.0, 9.0]


def test_flim_intensity_sums_over_bins():
    cube = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    intensity = acq.flim_intensity(cube)
    assert intensity.shape == (2, 2)
    assert np.array_equal(intensity, cube.sum(axis=2))


def test_read_flim_frame_reshapes_measurement_output():
    y_pixels, x_pixels, extra_left, extra_right, n_bins = 2, 2, 0, 0, 3
    total_x = x_pixels
    flat = np.arange(total_x * y_pixels * n_bins, dtype=np.float32).reshape(total_x * y_pixels, n_bins)

    fake_frame = types.SimpleNamespace(getHistograms=lambda: flat)
    fake_flim = types.SimpleNamespace(getCurrentFrameEx=lambda: fake_frame)

    cube = acq.read_flim_frame(fake_flim, n_bins, y_pixels, total_x, extra_left, x_pixels)
    assert cube.shape == (y_pixels, x_pixels, n_bins)
    assert np.array_equal(cube.reshape(-1, n_bins), flat)


# --------------------------------------------------------------------------- #
# DAQ scan wiring (mocked nidaqmx) — documents the required marker wiring
# --------------------------------------------------------------------------- #

class FakeTask:
    """A minimal stand-in for nidaqmx.Task that records the calls run_flim_scan
    makes so the test can assert the frame-trigger + pixel-clock wiring."""

    def __init__(self) -> None:
        self.ao_channels = _Recorder()
        self.co_channels = _CoRecorder()
        self.timing = _Recorder()
        self.export_signals = _Recorder()
        self.triggers = types.SimpleNamespace(start_trigger=_Recorder())
        self.calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, *a, **k):
        self.calls.append(("write", a, k))

    def start(self):
        self.calls.append(("start", (), {}))

    def wait_until_done(self, **k):
        self.calls.append(("wait_until_done", (), k))


class _Recorder:
    def __init__(self) -> None:
        self.calls: dict[str, tuple] = {}

    def __getattr__(self, name):
        def record(*a, **k):
            self.calls[name] = (a, k)
        return record


class _CoRecorder(_Recorder):
    def add_co_pulse_chan_ticks(self, *a, **k):
        self.calls["add_co_pulse_chan_ticks"] = (a, k)
        self.channel = types.SimpleNamespace(co_pulse_term=None)
        return self.channel


def run_scan_with_fake_daq(monkeypatch, **overrides):
    created: list[FakeTask] = []

    def make_task():
        task = FakeTask()
        created.append(task)
        return task

    monkeypatch.setattr(acq, "nx", types.SimpleNamespace(Task=make_task))

    kwargs = dict(
        device_name="Dev1",
        sample_rate_hz=1_000_000.0,
        fast_ao=0,
        slow_ao=1,
        raster_waveform=np.zeros((2, 48), dtype=np.float64),
        n_pixels=24,
        pixel_samples=2,
        frame_trigger_pfi=0,
        pixel_clock_ctr=0,
        pixel_clock_pfi=1,
    )
    kwargs.update(overrides)
    acq.run_flim_scan(**kwargs)
    return created


def test_run_flim_scan_exports_frame_trigger(monkeypatch):
    ao_task, _co_task = run_scan_with_fake_daq(monkeypatch)
    args, _ = ao_task.export_signals.calls["export_signal"]
    # the AO start trigger is exported to the configured PFI line (frame marker)
    assert args[1] == "/Dev1/PFI0"


def test_run_flim_scan_pixel_clock_divides_sample_clock(monkeypatch):
    _ao_task, co_task = run_scan_with_fake_daq(monkeypatch, pixel_samples=8)
    _args, kw = co_task.co_channels.calls["add_co_pulse_chan_ticks"]
    # one pulse every pixel_samples ticks of the AO sample clock -> locked, no drift
    assert kw["source_terminal"] == "/Dev1/ao/SampleClock"
    assert kw["high_ticks"] + kw["low_ticks"] == 8
    # pixel clock pulses are routed to the configured PFI output
    assert co_task.co_channels.channel.co_pulse_term == "/Dev1/PFI1"


def test_run_flim_scan_pixel_clock_started_by_ao_and_counts_pixels(monkeypatch):
    _ao_task, co_task = run_scan_with_fake_daq(monkeypatch, n_pixels=24)
    _args, kw = co_task.timing.calls["cfg_implicit_timing"]
    assert kw["samps_per_chan"] == 24
    trig_args, _ = co_task.triggers.start_trigger.calls["cfg_dig_edge_start_trig"]
    assert trig_args[0] == "/Dev1/ao/StartTrigger"


def test_run_flim_scan_wraps_failures(monkeypatch):
    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(acq, "nx", types.SimpleNamespace(Task=boom))
    with pytest.raises(acq.DaqError):
        acq.run_flim_scan(
            device_name="Dev1",
            sample_rate_hz=1e6,
            fast_ao=0,
            slow_ao=1,
            raster_waveform=np.zeros((2, 4), dtype=np.float64),
            n_pixels=2,
            pixel_samples=2,
            frame_trigger_pfi=0,
            pixel_clock_ctr=0,
            pixel_clock_pfi=1,
        )


# --------------------------------------------------------------------------- #
# flim_scan geometry — derives pixel_samples / n_pixels and the waveform
# --------------------------------------------------------------------------- #

def test_flim_scan_computes_geometry_and_clamps_pixel_samples(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(acq, "run_flim_scan", fake_run)

    daq, galvo = wired()
    acq.flim_scan(
        daq=daq,
        galvo=galvo,
        scan=ScanGroup(x_pixels=4, y_pixels=3, extra_left=1, extra_right=2, dwell_time_us=2.0),
        sample_rate_hz=1_000_000.0,
        triggers=TriggerGroup(frame_trigger_pfi=0, pixel_clock_ctr=0, pixel_clock_pfi=1),
    )

    total_x = 1 + 4 + 2
    pixel_samples = int(round(2.0 * 1e-6 * 1_000_000.0))  # 2
    assert captured["pixel_samples"] == pixel_samples
    assert captured["n_pixels"] == total_x * 3
    # waveform has one column per AO sample = total_x * y * pixel_samples
    assert captured["raster_waveform"].shape == (2, total_x * 3 * pixel_samples)


def test_flim_scan_floors_pixel_samples_at_two(monkeypatch):
    captured = {}
    monkeypatch.setattr(acq, "run_flim_scan", lambda **k: captured.update(k))
    # dwell*rate rounds below 2 -> must be clamped so the tick divider stays valid
    daq, galvo = wired()
    acq.flim_scan(
        daq=daq,
        galvo=galvo,
        scan=ScanGroup(x_pixels=8, y_pixels=8, extra_left=0, extra_right=0, dwell_time_us=2.0),
        sample_rate_hz=100_000.0,
        triggers=TriggerGroup(frame_trigger_pfi=0, pixel_clock_ctr=0, pixel_clock_pfi=1),
    )
    assert captured["pixel_samples"] == 2
