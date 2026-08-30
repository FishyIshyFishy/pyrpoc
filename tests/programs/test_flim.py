"""FLIM, checked against the v3.0 modality still in the tree.

The behaviour this phase changes deliberately: the TimeTagger is created and
freed once per run instead of once per frame. Image data must be identical; the
lifecycle must not be.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from pyrpoc.core.streams import Cube3D, Image2D
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices import DAQ, Galvo, TimeTagger
from pyrpoc.programs.flim import FLIM, FlimParams
from pyrpoc.run.runner import Runner


SCAN = dict(
    x_pixels=8, y_pixels=8, extra_left=3, extra_right=2,
    fast_axis_offset=0.0, fast_axis_amplitude=1.0,
    slow_axis_offset=0.0, slow_axis_amplitude=1.0, dwell_time_us=2.0,
)
N_BINS = 5
SAMPLE_RATE = 1_000_000.0


def fake_cube(index: int) -> np.ndarray:
    return np.full((SCAN["y_pixels"], SCAN["x_pixels"], N_BINS), index + 1, np.float32)


class FakeTaggerSdk:
    """Records the lifecycle so the once-per-run change can be asserted."""

    def __init__(self):
        self.created = 0
        self.freed = 0
        self.measurements_started = 0
        self.measurements_stopped = 0
        self.configured = 0

    def install(self, device: TimeTagger):
        sdk = self

        def create_tagger():
            sdk.created += 1
            device.tagger = object()

        def free_tagger():
            if device.tagger is not None:
                sdk.freed += 1
            device.tagger = None

        def configure_for_flim():
            sdk.configured += 1

        def start(**kwargs):
            sdk.measurements_started += 1
            return object()

        def stop(flim):
            if flim is not None:
                sdk.measurements_stopped += 1
            free_tagger()

        device.create_tagger = create_tagger
        device.free_tagger = free_tagger
        device.configure_for_flim = configure_for_flim
        device.start_flim_measurement = start
        device.stop_flim_measurement = stop
        return self


@pytest.fixture
def devices():
    daq, galvo, tagger = DAQ(), Galvo(), TimeTagger()
    return daq, galvo, tagger


def new_params(tmp_path=None, frames=3) -> FlimParams:
    params = FlimParams()
    for name, value in SCAN.items():
        setattr(params.scan, name, value)
    params.daq.sample_rate_hz = SAMPLE_RATE
    params.histogram.histogram_bins = N_BINS
    params.histogram.frame_settle_s = 0.0
    params.num_frames = frames
    if tmp_path is not None:
        params.save.save_enabled = True
        params.save.save_path = str(tmp_path / "acq")
    return params


def run_new(monkeypatch, params, devices, *, continuous=False, fail_on=None):
    daq, galvo, tagger = devices
    sdk = FakeTaggerSdk().install(tagger)
    scans = []

    def fake_scan(**kwargs):
        scans.append(kwargs)
        if fail_on is not None and len(scans) == fail_on:
            raise RuntimeError("NI-DAQ FLIM scan failed: boom")

    monkeypatch.setattr("pyrpoc.programs.flim.flim_scan", fake_scan)
    monkeypatch.setattr(
        "pyrpoc.programs.flim.read_flim_frame",
        lambda flim, **kwargs: fake_cube(len(scans) - 1),
    )

    runner = Runner(DatasetLibrary())
    failures = []
    handle = runner.start(
        FLIM(), params, [daq, galvo, tagger], continuous=continuous,
        program_key="flim", on_failed=failures.append,
    )
    if continuous:
        for _ in range(500):
            if len(scans) >= 3:
                break
            threading.Event().wait(0.005)
        runner.stop()
    handle.thread.join(timeout=10)
    return handle, sdk, scans, failures


# --- the intended change ---------------------------------------------------


def test_the_tagger_is_created_once_per_run_not_once_per_frame(monkeypatch, devices):
    _, sdk, scans, _ = run_new(monkeypatch, new_params(frames=3), devices)
    assert len(scans) == 3
    assert sdk.created == 1
    assert sdk.configured == 1
    assert sdk.measurements_started == 1
    assert sdk.freed == 1


def test_setup_happens_before_the_loop_not_inside_it(monkeypatch, devices):
    """v3.0 called setup_tagger() and teardown_tagger() inside acquire_once, so
    a ten-frame run created and freed the TimeTagger ten times."""
    handle, sdk, scans, _ = run_new(monkeypatch, new_params(frames=10), devices)
    assert len(scans) == 10
    assert (sdk.created, sdk.freed) == (1, 1)


# --- frames ----------------------------------------------------------------


def test_intensity_is_published_once_per_frame(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    dataset = handle.datasets["intensity"]
    assert len(dataset) == 3
    for index in range(3):
        np.testing.assert_array_equal(
            dataset.frame(index), fake_cube(index).sum(axis=2)[np.newaxis]
        )


def test_the_histogram_cube_is_published_unchanged(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    dataset = handle.datasets["histogram"]
    assert len(dataset) == 3
    for index in range(3):
        np.testing.assert_array_equal(dataset.frame(index), fake_cube(index))


def test_intensity_is_the_histogram_sum(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(frames=1), devices)
    cube = handle.datasets["histogram"].frame(0)
    intensity = handle.datasets["intensity"].frame(0)
    np.testing.assert_array_equal(intensity[0], cube.sum(axis=2))


def test_channel_labels_are_the_v30_label(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    assert handle.datasets["intensity"].channel_labels == ["intensity"]


def test_histogram_metadata_is_what_v30_attached_to_every_frame(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    assert handle.datasets["histogram"].metadata == {
        "laser_period_ps": 12500,
        "binwidth_ps": 100,
        "n_bins": N_BINS,
    }


def test_the_scan_receives_the_trigger_wiring(monkeypatch, devices):
    _, _, scans, _ = run_new(monkeypatch, new_params(), devices)
    assert scans[0]["frame_trigger_pfi"] == 0
    assert scans[0]["pixel_clock_ctr"] == 0
    assert scans[0]["pixel_clock_pfi"] == 1


# --- teardown --------------------------------------------------------------


def test_the_tagger_is_freed_when_a_run_is_cancelled(monkeypatch, devices):
    """Cancelled is an exception, so the program's finally does the teardown."""
    _, sdk, _, _ = run_new(monkeypatch, new_params(frames=1), devices, continuous=True)
    assert sdk.created == 1
    assert sdk.freed == 1
    assert sdk.measurements_stopped == 1


def test_the_tagger_is_freed_when_a_scan_fails(monkeypatch, devices):
    _, sdk, _, failures = run_new(monkeypatch, new_params(frames=3), devices, fail_on=2)
    assert failures and "NI-DAQ FLIM scan failed" in failures[0]
    assert sdk.freed == 1
    assert sdk.measurements_stopped == 1


def test_a_failure_mid_run_keeps_the_frames_already_acquired(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(frames=3), devices, fail_on=2)
    assert len(handle.datasets["intensity"]) == 1


# --- saving ----------------------------------------------------------------


def test_the_intensity_tiff_is_written(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    assert (tmp_path / "acq_intensity.tiff").exists()


def test_the_histogram_npz_holds_every_cube(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    with np.load(tmp_path / "acq_histogram.npz", allow_pickle=True) as npz:
        expected = np.stack([fake_cube(i) for i in range(3)], axis=0)
        np.testing.assert_array_equal(npz["frames"], expected)
        assert npz["frames"].dtype == np.float32  # v3.0 wrote dtype=object


def test_the_histogram_filename_and_key_changed_deliberately(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    assert (tmp_path / "acq_histogram.npz").exists()
    assert not (tmp_path / "acq_raw.npz").exists()
    with np.load(tmp_path / "acq_histogram.npz", allow_pickle=True) as data:
        assert "parameters" in data.files
        assert "acquisition_parameters" not in data.files


def test_metadata_records_both_streams(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert sorted(meta["streams"]) == ["histogram", "intensity"]
    assert meta["frames_saved"] == 3
    assert meta["modality_key"] == "flim"


# --- shape -----------------------------------------------------------------


def test_flim_declares_only_the_four_program_attributes():
    defined = {name for name in vars(FLIM) if not name.startswith("__")}
    assert defined == {"uses", "params", "emits", "run"}


def test_flim_uses_all_three_devices():
    assert FLIM.uses == [Galvo, DAQ, TimeTagger]


def test_flim_emits_an_image_and_a_cube():
    assert FLIM.emits == {"intensity": Image2D, "histogram": Cube3D}
