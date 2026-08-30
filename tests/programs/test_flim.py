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

from pyrpoc.modalities.flim.flim import FlimModality


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


def v30_params(tmp_path=None, frames=3) -> dict:
    return {
        "X Pixels": SCAN["x_pixels"], "Y Pixels": SCAN["y_pixels"],
        "Extra Steps Left": SCAN["extra_left"], "Extra Steps Right": SCAN["extra_right"],
        "Fast Axis Offset": SCAN["fast_axis_offset"],
        "Fast Axis Amplitude": SCAN["fast_axis_amplitude"],
        "Slow Axis Offset": SCAN["slow_axis_offset"],
        "Slow Axis Amplitude": SCAN["slow_axis_amplitude"],
        "Dwell Time (us)": SCAN["dwell_time_us"],
        "DAQ Device": "Dev1", "Sample Rate (Hz)": SAMPLE_RATE,
        "Fast Axis AO": 0, "Slow Axis AO": 1,
        "Frame Trigger PFI Line": 0, "Pixel Clock Counter": 0, "Pixel Clock PFI Line": 1,
        "Laser Channel": 1, "Detector Channel": 2, "Pixel Channel": 3, "Frame Channel": 4,
        "Laser Frequency MHz": 80.0, "Histogram Bins": N_BINS,
        "Histogram Bin Width (ps)": 100,
        "Laser Trigger V": 0.05, "Detector Trigger V": 0.2,
        "Pixel Trigger V": 0.5, "Frame Trigger V": 0.5,
        "Laser Input Delay (ps)": 0,
        "save_enabled": tmp_path is not None,
        "save_path": str(tmp_path / "acq") if tmp_path else "acquisition",
        "num_frames": frames,
    }


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


def run_v30(monkeypatch, raw_params, *, frames=3):
    tagger_device = TimeTagger()
    sdk = FakeTaggerSdk()

    class V30TaggerShim:
        """v3.0 called create_tagger/configure_for_flim/create_flim_measurement."""

        def __init__(self):
            self.tagger = None

        def create_tagger(self):
            sdk.created += 1
            self.tagger = object()

        def free_tagger(self):
            if self.tagger is not None:
                sdk.freed += 1
            self.tagger = None

        def configure_for_flim(self, *args, **kwargs):
            sdk.configured += 1

        def create_flim_measurement(self, *args, **kwargs):
            sdk.measurements_started += 1
            return type("Flim", (), {"stop": lambda self: None})()

    del tagger_device
    shim = V30TaggerShim()
    scans = []
    monkeypatch.setattr(
        "pyrpoc.modalities.flim.flim.flim_scan", lambda **kwargs: scans.append(kwargs)
    )
    monkeypatch.setattr(
        "pyrpoc.modalities.flim.flim.read_flim_frame",
        lambda flim, **kwargs: fake_cube(len(scans) - 1),
    )
    monkeypatch.setattr("pyrpoc.modalities.flim.flim.frame_settle_s", 0.0)

    from pyrpoc.instruments.time_tagger import TimeTaggerInstrument

    modality = FlimModality()
    modality.configure(raw_params, {TimeTaggerInstrument: shim}, [])

    emitted = []

    def on_data(acquired):
        emitted.append(acquired)
        if acquired.kind.is_persistent:
            modality.save_acquired_frame(acquired, frame_index=sum(
                1 for a in emitted if a.kind.is_persistent
            ) - 1)

    modality.prepare_acquisition_storage(frame_limit=frames)
    done = threading.Event()
    thread = modality.acquire_continuous(
        on_frame=on_data, frame_limit=frames, should_stop=lambda: False,
        on_error=lambda exc: None, on_finished=lambda count, error: done.set(),
    )
    thread.join(timeout=10)
    done.wait(timeout=5)
    modality.finalize_acquisition_storage(frame_count=frames, frame_limit=frames, error=None)
    return modality, emitted, sdk, scans


# --- the intended change ---------------------------------------------------


def test_the_tagger_is_created_once_per_run_not_once_per_frame(monkeypatch, devices):
    _, sdk, scans, _ = run_new(monkeypatch, new_params(frames=3), devices)
    assert len(scans) == 3
    assert sdk.created == 1
    assert sdk.configured == 1
    assert sdk.measurements_started == 1
    assert sdk.freed == 1


def test_v30_created_the_tagger_once_per_frame(monkeypatch):
    """The behaviour being fixed, asserted so the improvement is measured."""
    _, _, sdk, scans = run_v30(monkeypatch, v30_params(frames=3))
    assert len(scans) == 3
    assert sdk.created == 3
    assert sdk.measurements_started == 3
    assert sdk.freed == 3


# --- frames ----------------------------------------------------------------


def test_intensity_frames_match_the_v30_modality(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    _, emitted, _, _ = run_v30(monkeypatch, v30_params())

    from pyrpoc.backend_utils.acquired_data import DataKind

    old_intensity = [a for a in emitted if a.kind is DataKind.INTENSITY_FRAME]
    dataset = handle.datasets["intensity"]
    assert len(dataset) == len(old_intensity) == 3
    for index, acquired in enumerate(old_intensity):
        np.testing.assert_array_equal(dataset.frame(index), acquired.data)


def test_histogram_cubes_match_the_v30_modality(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    _, emitted, _, _ = run_v30(monkeypatch, v30_params())

    from pyrpoc.backend_utils.acquired_data import DataKind

    old_cubes = [a for a in emitted if a.kind is DataKind.FLIM_RAW_FRAME]
    dataset = handle.datasets["histogram"]
    assert len(dataset) == len(old_cubes) == 3
    for index, acquired in enumerate(old_cubes):
        np.testing.assert_array_equal(dataset.frame(index), acquired.data)


def test_intensity_is_the_histogram_sum(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(frames=1), devices)
    cube = handle.datasets["histogram"].frame(0)
    intensity = handle.datasets["intensity"].frame(0)
    np.testing.assert_array_equal(intensity[0], cube.sum(axis=2))


def test_channel_labels_match_the_v30_modality(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    assert handle.datasets["intensity"].channel_labels == ["intensity"]


def test_histogram_metadata_matches_what_v30_attached(monkeypatch, devices):
    handle, _, _, _ = run_new(monkeypatch, new_params(), devices)
    _, emitted, _, _ = run_v30(monkeypatch, v30_params())

    from pyrpoc.backend_utils.acquired_data import DataKind

    old = next(a for a in emitted if a.kind is DataKind.FLIM_RAW_FRAME)
    assert handle.datasets["histogram"].metadata == old.metadata
    assert handle.datasets["histogram"].metadata["laser_period_ps"] == 12500


def test_the_scan_receives_the_trigger_wiring(monkeypatch, devices):
    _, _, scans, _ = run_new(monkeypatch, new_params(), devices)
    _, _, _, old_scans = run_v30(monkeypatch, v30_params())
    assert scans[0]["frame_trigger_pfi"] == old_scans[0]["frame_trigger_pfi"] == 0
    assert scans[0]["pixel_clock_ctr"] == old_scans[0]["pixel_clock_ctr"] == 0
    assert scans[0]["pixel_clock_pfi"] == old_scans[0]["pixel_clock_pfi"] == 1


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


def test_the_intensity_tiff_matches_v30(monkeypatch, devices, tmp_path):
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    new_dir.mkdir()
    old_dir.mkdir()
    run_new(monkeypatch, new_params(new_dir), devices)
    run_v30(monkeypatch, v30_params(old_dir))
    assert (new_dir / "acq_intensity.tiff").read_bytes() == (
        old_dir / "acq_intensity.tiff"
    ).read_bytes()


def test_the_histogram_npz_holds_the_same_cubes_as_v30(monkeypatch, devices, tmp_path):
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    new_dir.mkdir()
    old_dir.mkdir()
    run_new(monkeypatch, new_params(new_dir), devices)
    run_v30(monkeypatch, v30_params(old_dir))

    with np.load(old_dir / "acq_raw.npz", allow_pickle=True) as old:
        old_frames = np.stack(list(old["frames"]), axis=0)
    with np.load(new_dir / "acq_histogram.npz", allow_pickle=True) as new:
        np.testing.assert_array_equal(new["frames"], old_frames)
        assert new["frames"].dtype == np.float32  # v3.0 wrote dtype=object


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
