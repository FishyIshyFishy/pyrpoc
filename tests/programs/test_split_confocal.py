"""Split confocal.

The mask gate is checked against ``tests/reference/phase0_references.npz``;
saved-file compatibility lives in tests/data/test_io.py.
"""

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

from tests.reference.generate_references import reference_path


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


# --- frames ----------------------------------------------------------------


def test_both_streams_publish_what_the_operation_returned(monkeypatch, devices):
    handle, calls = run_new(monkeypatch, new_params(), devices)
    assert len(calls) == 3
    for index in range(3):
        expected_split, expected_raw = fake_split(index)
        np.testing.assert_array_equal(handle.datasets["intensity"].frame(index), expected_split)
        np.testing.assert_array_equal(
            handle.datasets["raw_pixel_stream"].frame(index), expected_raw
        )


def test_channel_labels_are_the_v30_interleaved_labels(monkeypatch, devices):
    handle, _ = run_new(monkeypatch, new_params(), devices)
    assert handle.datasets["intensity"].channel_labels == [
        "ai0_t0", "ai0_t2", "ai1_t0", "ai1_t2"
    ]


def test_channel_labels_interleave_t0_and_t2(devices):
    daq, _ = devices
    daq.config.ai_channels = (2, 5)
    assert channel_labels(daq) == ["ai2_t0", "ai2_t2", "ai5_t0", "ai5_t2"]


def test_split_timing_reaches_the_operation(monkeypatch, devices):
    _, calls = run_new(monkeypatch, new_params(), devices)
    assert calls[0]["t0_samples"] == T0
    assert calls[0]["t1_samples"] == T1


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


def test_the_raw_npz_keeps_its_v30_filename_and_keys(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    written = tmp_path / "acq_raw_pixel_stream.npz"
    assert written.exists(), "the v3.0 filename must be unchanged"
    with np.load(written, allow_pickle=True) as npz:
        assert sorted(npz.files) == ["frame_indices", "frames", "parameters"]
        assert npz["frames"].shape[0] == 3


def test_one_tiff_per_split_channel(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    for label in ("ai0_t0", "ai0_t2", "ai1_t0", "ai1_t2"):
        assert (tmp_path / f"acq_{label}.tiff").exists()


def test_metadata_counts_frames_against_the_intensity_stream(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["frames_saved"] == 3
    assert sorted(meta["streams"]) == ["intensity", "raw_pixel_stream"]
    assert "raw_pixel_stream" in meta["auxiliary_paths"]


# --- masks -----------------------------------------------------------------


def test_the_ttl_gate_matches_the_phase0_golden_array(devices, tmp_path):
    """The t0 gate is the only thing split confocal does differently here."""
    golden = dict(np.load(reference_path))
    daq, _ = devices

    mask = np.zeros((6, 8), np.uint8)
    mask[1:4, 2:6] = 255
    path = save_mask(tmp_path / "m.png", mask)

    params = new_params(masks=[MaskBinding(path, 0, 3)])
    params.scan.x_pixels, params.scan.y_pixels = 8, 6
    params.scan.extra_left, params.scan.extra_right = 3, 2
    params.scan.dwell_time_us, params.daq.sample_rate_hz = 4.0, 1_000_000.0
    assert pixel_samples(4.0, 1_000_000.0) == 4

    ttl = build_ttl(params, daq)
    np.testing.assert_array_equal(ttl["Dev1/port0/line3"], golden["split_mask_ttl"])
    assert ttl["Dev1/port0/line3"].sum() < golden["confocal_mask_ttl"].sum()


# --- shape -----------------------------------------------------------------


def test_split_confocal_declares_only_the_four_program_attributes():
    defined = {name for name in vars(SplitConfocal) if not name.startswith("__")}
    assert defined == {"uses", "params", "emits", "run"}


def test_split_confocal_emits_two_streams():
    assert SplitConfocal.emits == {"intensity": Image2D, "raw_pixel_stream": Samples4D}
