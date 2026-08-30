"""Split confocal, checked against the v3.0 modality still in the tree."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from pyrpoc.core.modulation import MaskBinding, save_mask
from pyrpoc.core.streams import Image2D, Samples4D
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.operations.raster import pixel_samples
from pyrpoc.programs.split_confocal import (
    SplitConfocal,
    SplitConfocalParams,
    build_ttl,
    channel_labels,
)
from pyrpoc.run.runner import Runner

from pyrpoc.modalities.split_confocal.split_confocal import SplitConfocalModality


SCAN = dict(
    x_pixels=8, y_pixels=8, extra_left=3, extra_right=2,
    fast_axis_offset=0.0, fast_axis_amplitude=1.0,
    slow_axis_offset=0.0, slow_axis_amplitude=1.0, dwell_time_us=40.0,
)
AI_CHANNELS = (0, 1)
SAMPLE_RATE = 100_000.0
T0, T1 = 1, 1
# Derived, not hardcoded: the raster formula truncates, and 40 us at 100 kHz is
# 3.9999999999999996 -> 3, not 4. Exactly what test_pixel_samples.py pins.
PIXEL_SAMPLES = pixel_samples(SCAN["dwell_time_us"], SAMPLE_RATE)


def fake_split(index: int):
    split = np.full((len(AI_CHANNELS) * 2, SCAN["y_pixels"], SCAN["x_pixels"]), index, np.float32)
    raw = np.full(
        (len(AI_CHANNELS), SCAN["y_pixels"], SCAN["x_pixels"], PIXEL_SAMPLES), index, np.float32
    )
    return split, raw


@pytest.fixture
def devices():
    daq, galvo = DAQ(), Galvo()
    daq.config.ai_channels = AI_CHANNELS
    return daq, galvo


def new_params(tmp_path=None, frames=3, masks=()) -> SplitConfocalParams:
    params = SplitConfocalParams()
    for name, value in SCAN.items():
        setattr(params.scan, name, value)
    params.split.t0_samples = T0
    params.split.t1_samples = T1
    params.num_frames = frames
    params.modulation.masks = tuple(masks)
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
        "Active AI Channels": list(AI_CHANNELS),
        "t0 Samples": T0, "t1 Samples": T1,
        "save_enabled": tmp_path is not None,
        "save_path": str(tmp_path / "acq") if tmp_path else "acquisition",
        "num_frames": frames,
    }


def run_new(monkeypatch, params, devices):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return fake_split(len(calls) - 1)

    monkeypatch.setattr("pyrpoc.programs.split_confocal.split_raster_scan", fake)
    runner = Runner(DatasetLibrary())
    handle = runner.start(
        SplitConfocal(), params, list(devices), program_key="split_confocal"
    )
    handle.thread.join(timeout=10)
    return handle, calls


def run_v30(monkeypatch, raw_params, *, frames=3):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return fake_split(len(calls) - 1)

    monkeypatch.setattr(
        "pyrpoc.modalities.split_confocal.split_confocal.acquire_daq_split_confocal", fake
    )
    modality = SplitConfocalModality()
    modality.configure(raw_params, {}, [])
    modality.prepare_acquisition_storage(frame_limit=frames)

    emitted = []

    def on_data(acquired):
        emitted.append(acquired)
        modality.save_acquired_frame(acquired, frame_index=len(emitted) - 1)

    done = threading.Event()
    thread = modality.acquire_continuous(
        on_frame=on_data, frame_limit=frames, should_stop=lambda: False,
        on_error=lambda exc: None, on_finished=lambda count, error: done.set(),
    )
    thread.join(timeout=10)
    done.wait(timeout=5)
    modality.finalize_acquisition_storage(frame_count=len(emitted), frame_limit=frames, error=None)
    return modality, emitted, calls


# --- frames ----------------------------------------------------------------


def test_intensity_frames_match_the_v30_modality(monkeypatch, devices):
    handle, _ = run_new(monkeypatch, new_params(), devices)
    _, emitted, _ = run_v30(monkeypatch, v30_params())
    dataset = handle.datasets["intensity"]
    assert len(dataset) == len(emitted) == 3
    for index, acquired in enumerate(emitted):
        np.testing.assert_array_equal(dataset.frame(index), acquired.data)


def test_channel_labels_match_the_v30_modality(monkeypatch, devices):
    handle, _ = run_new(monkeypatch, new_params(), devices)
    modality, _, _ = run_v30(monkeypatch, v30_params())
    labels = handle.datasets["intensity"].channel_labels
    assert labels == modality.get_active_channel_labels()
    assert labels == ["ai0_t0", "ai0_t2", "ai1_t0", "ai1_t2"]


def test_channel_labels_interleave_t0_and_t2(devices):
    daq, _ = devices
    daq.config.ai_channels = (2, 5)
    assert channel_labels(daq) == ["ai2_t0", "ai2_t2", "ai5_t0", "ai5_t2"]


def test_split_timing_reaches_the_operation(monkeypatch, devices):
    _, new_calls = run_new(monkeypatch, new_params(), devices)
    _, _, old_calls = run_v30(monkeypatch, v30_params())
    assert new_calls[0]["t0_samples"] == old_calls[0]["t0_samples"] == T0
    assert new_calls[0]["t1_samples"] == old_calls[0]["t1_samples"] == T1


# --- the raw stream --------------------------------------------------------


def test_the_raw_stream_is_a_declared_output_not_a_side_channel(monkeypatch, devices):
    handle, _ = run_new(monkeypatch, new_params(), devices)
    raw = handle.datasets["raw_pixel_stream"]
    assert raw.spec is Samples4D
    assert len(raw) == 3
    assert raw.frame(0).shape == (len(AI_CHANNELS), 8, 8, PIXEL_SAMPLES)


def test_the_raw_stream_is_in_the_library_and_can_be_bound(monkeypatch, devices):
    """v3.0 buffered it in storage where no display could ever see it."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return fake_split(len(calls) - 1)

    monkeypatch.setattr("pyrpoc.programs.split_confocal.split_raster_scan", fake)
    library = DatasetLibrary()
    runner = Runner(library)
    handle = runner.start(SplitConfocal(), new_params(), list(devices))
    handle.thread.join(timeout=10)
    assert library.matching(Samples4D) == [handle.datasets["raw_pixel_stream"]]
    assert library.matching(Image2D) == [handle.datasets["intensity"]]


def test_the_raw_npz_matches_v30_and_keeps_its_filename(monkeypatch, devices, tmp_path):
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    new_dir.mkdir()
    old_dir.mkdir()
    run_new(monkeypatch, new_params(new_dir), devices)
    run_v30(monkeypatch, v30_params(old_dir))

    new_npz = new_dir / "acq_raw_pixel_stream.npz"
    old_npz = old_dir / "acq_raw_pixel_stream.npz"
    assert new_npz.exists() and old_npz.exists()

    with np.load(new_npz, allow_pickle=True) as new, np.load(old_npz, allow_pickle=True) as old:
        assert sorted(new.files) == sorted(old.files)
        np.testing.assert_array_equal(new["frames"], old["frames"])
        np.testing.assert_array_equal(new["frame_indices"], old["frame_indices"])


def test_the_intensity_tiffs_match_v30(monkeypatch, devices, tmp_path):
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    new_dir.mkdir()
    old_dir.mkdir()
    run_new(monkeypatch, new_params(new_dir), devices)
    run_v30(monkeypatch, v30_params(old_dir))

    for label in ("ai0_t0", "ai0_t2", "ai1_t0", "ai1_t2"):
        assert (new_dir / f"acq_{label}.tiff").read_bytes() == (
            old_dir / f"acq_{label}.tiff"
        ).read_bytes()


def test_metadata_counts_frames_against_the_intensity_stream(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["frames_saved"] == 3
    assert sorted(meta["streams"]) == ["intensity", "raw_pixel_stream"]
    assert "raw_pixel_stream" in meta["auxiliary_paths"]


# --- masks -----------------------------------------------------------------


def test_the_ttl_is_gated_to_t0_like_v30(devices, tmp_path):
    from pyrpoc.backend_utils.opto_control_contexts import MaskContext
    from pyrpoc.modalities.split_confocal import acquisition_core as v30_core

    daq, _ = devices
    mask = np.zeros((SCAN["y_pixels"], SCAN["x_pixels"]), np.uint8)
    mask[1:4, 2:6] = 255
    path = save_mask(tmp_path / "m.png", mask)

    new_ttl = build_ttl(new_params(masks=[MaskBinding(path, 0, 3)]), daq)
    old_ttl = v30_core.generate_mask_ttl_signals(
        total_x=SCAN["x_pixels"] + SCAN["extra_left"] + SCAN["extra_right"],
        total_y=SCAN["y_pixels"],
        pixel_samples=PIXEL_SAMPLES,
        extra_left=SCAN["extra_left"],
        extra_right=SCAN["extra_right"],
        device_name="Dev1",
        mask_contexts=[MaskContext("mask", "m", mask, 0, 3)],
        scan_x_pixels=SCAN["x_pixels"],
        t0_samples=T0,
    )
    assert sorted(new_ttl) == sorted(old_ttl)
    for channel, signal in new_ttl.items():
        np.testing.assert_array_equal(signal, old_ttl[channel])


# --- shape -----------------------------------------------------------------


def test_split_confocal_declares_only_the_four_program_attributes():
    defined = {name for name in vars(SplitConfocal) if not name.startswith("__")}
    assert defined == {"uses", "params", "emits", "run"}


def test_split_confocal_emits_two_streams():
    assert SplitConfocal.emits == {"intensity": Image2D, "raw_pixel_stream": Samples4D}
