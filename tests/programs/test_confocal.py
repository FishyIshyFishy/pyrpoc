"""Confocal, checked against the v3.0 modality that is still in the tree.

The hardware call is replaced on both sides with the same fake, so everything
above it runs for real: the runner's thread, dataset creation from ``emits``,
publishing, the save policy, and the program's own control flow.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from pyrpoc.core.modulation import MaskBinding, save_mask
from pyrpoc.data.io import load_frames
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.programs.confocal import Confocal, ConfocalParams, build_ttl, channel_labels
from pyrpoc.run.runner import Runner

# v3.0, still present until phase 9.
from pyrpoc.modalities.confocal.confocal import ConfocalModality


SCAN = dict(
    x_pixels=8, y_pixels=8, extra_left=3, extra_right=2,
    fast_axis_offset=0.25, fast_axis_amplitude=1.5,
    slow_axis_offset=-0.5, slow_axis_amplitude=2.0, dwell_time_us=2.0,
)
AI_CHANNELS = (0, 1)


def fake_frame(index: int) -> np.ndarray:
    return np.full((len(AI_CHANNELS), SCAN["y_pixels"], SCAN["x_pixels"]), index, np.float32)


@pytest.fixture
def devices():
    daq, galvo = DAQ(), Galvo()
    daq.config.ai_channels = AI_CHANNELS
    return daq, galvo


def new_params(tmp_path=None, frames=3, masks=()) -> ConfocalParams:
    params = ConfocalParams()
    for name, value in SCAN.items():
        setattr(params.scan, name, value)
    params.num_frames = frames
    params.modulation.masks = tuple(masks)
    if tmp_path is not None:
        params.save.save_enabled = True
        params.save.save_path = str(tmp_path / "acq")
    return params


def v30_params(tmp_path=None, frames=3) -> dict:
    raw = {
        "X Pixels": SCAN["x_pixels"], "Y Pixels": SCAN["y_pixels"],
        "Extra Steps Left": SCAN["extra_left"], "Extra Steps Right": SCAN["extra_right"],
        "Fast Axis Offset": SCAN["fast_axis_offset"],
        "Fast Axis Amplitude": SCAN["fast_axis_amplitude"],
        "Slow Axis Offset": SCAN["slow_axis_offset"],
        "Slow Axis Amplitude": SCAN["slow_axis_amplitude"],
        "Dwell Time (us)": SCAN["dwell_time_us"],
        "DAQ Device": "Dev1", "Sample Rate (Hz)": 100_000.0,
        "Fast Axis AO": 0, "Slow Axis AO": 1,
        "Active AI Channels": list(AI_CHANNELS),
        "save_enabled": tmp_path is not None,
        "save_path": str(tmp_path / "acq") if tmp_path else "acquisition",
        "num_frames": frames,
    }
    return raw


def run_new(monkeypatch, params, devices, *, continuous=False, capture=None):
    calls = []

    def fake_raster_scan(**kwargs):
        calls.append(kwargs)
        return fake_frame(len(calls) - 1)

    monkeypatch.setattr("pyrpoc.programs.confocal.raster_scan", fake_raster_scan)
    runner = Runner(DatasetLibrary())
    handle = runner.start(
        Confocal(), params, list(devices), continuous=continuous, program_key="confocal"
    )
    if continuous:
        for _ in range(500):
            if len(calls) >= 3:
                break
            threading.Event().wait(0.005)
        runner.stop()
    handle.thread.join(timeout=10)
    if capture is not None:
        capture.extend(calls)
    return handle, calls


def run_v30(monkeypatch, raw_params, *, frames=3):
    calls = []

    def fake_acquire(**kwargs):
        calls.append(kwargs)
        return fake_frame(len(calls) - 1)

    monkeypatch.setattr(
        "pyrpoc.modalities.confocal.confocal.acquire_daq_confocal", fake_acquire
    )
    modality = ConfocalModality()
    modality.configure(raw_params, {}, [])
    modality.prepare_acquisition_storage(frame_limit=frames)

    emitted = []

    def on_data(acquired):
        emitted.append(acquired)
        modality.save_acquired_frame(acquired, frame_index=len(emitted) - 1)

    done = threading.Event()
    thread = modality.acquire_continuous(
        on_frame=on_data,
        frame_limit=frames,
        should_stop=lambda: False,
        on_error=lambda exc: None,
        on_finished=lambda count, error: done.set(),
    )
    thread.join(timeout=10)
    done.wait(timeout=5)
    modality.finalize_acquisition_storage(frame_count=len(emitted), frame_limit=frames, error=None)
    return modality, emitted, calls


# --- frames ----------------------------------------------------------------


def test_published_frames_match_the_v30_modality(monkeypatch, devices):
    handle, _ = run_new(monkeypatch, new_params(), devices)
    _, emitted, _ = run_v30(monkeypatch, v30_params())

    dataset = handle.datasets["intensity"]
    assert len(dataset) == len(emitted) == 3
    for index, acquired in enumerate(emitted):
        np.testing.assert_array_equal(dataset.frame(index), acquired.data)


def test_channel_labels_match_the_v30_modality(monkeypatch, devices):
    handle, _ = run_new(monkeypatch, new_params(), devices)
    modality, _, _ = run_v30(monkeypatch, v30_params())
    assert handle.datasets["intensity"].channel_labels == modality.get_active_channel_labels()
    assert handle.datasets["intensity"].channel_labels == ["ai0", "ai1"]


def test_channel_labels_come_from_the_daq_device(devices):
    daq, _ = devices
    daq.config.ai_channels = (0, 3, 7)
    assert channel_labels(daq) == ["ai0", "ai3", "ai7"]


def test_the_operation_receives_the_same_scan_values_v30_passed(monkeypatch, devices):
    _, new_calls = run_new(monkeypatch, new_params(), devices)
    _, _, old_calls = run_v30(monkeypatch, v30_params())

    for key, value in SCAN.items():
        assert new_calls[0][key] == old_calls[0][key] == value
    assert new_calls[0]["device_name"] == old_calls[0]["device_name"] == "Dev1"
    assert new_calls[0]["sample_rate_hz"] == old_calls[0]["sample_rate_hz"] == 100_000.0
    assert new_calls[0]["fast_ao"] == old_calls[0]["fast_axis_ao"] == 0
    assert new_calls[0]["slow_ao"] == old_calls[0]["slow_axis_ao"] == 1
    assert tuple(new_calls[0]["ai_channels"]) == tuple(old_calls[0]["active_ai_channels"])


# --- saving ----------------------------------------------------------------


def test_saved_tiffs_match_the_v30_modality(monkeypatch, devices, tmp_path):
    new_dir = tmp_path / "new"
    old_dir = tmp_path / "old"
    new_dir.mkdir()
    old_dir.mkdir()

    run_new(monkeypatch, new_params(new_dir), devices)
    run_v30(monkeypatch, v30_params(old_dir))

    for label in ("ai0", "ai1"):
        assert (new_dir / f"acq_{label}.tiff").read_bytes() == (
            old_dir / f"acq_{label}.tiff"
        ).read_bytes()


def test_saved_metadata_agrees_with_v30_where_the_keys_overlap(monkeypatch, devices, tmp_path):
    new_dir = tmp_path / "new"
    old_dir = tmp_path / "old"
    new_dir.mkdir()
    old_dir.mkdir()

    run_new(monkeypatch, new_params(new_dir), devices)
    run_v30(monkeypatch, v30_params(old_dir))

    new_meta = json.loads((new_dir / "acq_meta.json").read_text())
    old_meta = json.loads((old_dir / "acq_meta.json").read_text())

    assert new_meta["modality_key"] == old_meta["modality_key"] == "confocal"
    assert new_meta["frames_saved"] == old_meta["frames_saved"] == 3
    assert new_meta["frame_limit"] == old_meta["frame_limit"] == 3
    assert new_meta["last_error"] == old_meta["last_error"] is None
    assert sorted(new_meta["tiff_paths"]) == sorted(old_meta["tiff_paths"]) == ["ai0", "ai1"]


def test_saved_frames_read_back_in_order(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(tmp_path), devices)
    pages = load_frames(tmp_path / "acq_ai0.tiff")
    assert pages.shape == (3, SCAN["y_pixels"], SCAN["x_pixels"])
    np.testing.assert_array_equal(pages[:, 0, 0], [0.0, 1.0, 2.0])


def test_the_run_parameters_are_recorded_in_the_metadata(monkeypatch, devices, tmp_path):
    """A run's saved parameters now fully describe what happened."""
    run_new(monkeypatch, new_params(tmp_path), devices)
    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["parameters"]["scan"]["x_pixels"] == SCAN["x_pixels"]
    assert meta["parameters"]["num_frames"] == 3
    assert meta["devices"]["DAQ"]["ai_channels"] == list(AI_CHANNELS)
    assert meta["devices"]["Galvo"]["fast_ao"] == 0


# --- masks -----------------------------------------------------------------


def test_masks_are_turned_into_ttl_once_before_the_loop(monkeypatch, devices, tmp_path):
    daq, _ = devices
    mask = np.zeros((SCAN["y_pixels"], SCAN["x_pixels"]), np.uint8)
    mask[1:4, 2:6] = 255
    path = save_mask(tmp_path / "m.png", mask)
    binding = MaskBinding(path, port=0, line=3)

    loads = []
    real_load = __import__(
        "pyrpoc.programs.confocal", fromlist=["load_mask"]
    ).load_mask

    def counting_load(p):
        loads.append(p)
        return real_load(p)

    monkeypatch.setattr("pyrpoc.programs.confocal.load_mask", counting_load)

    _, calls = run_new(monkeypatch, new_params(masks=[binding]), devices)

    assert len(loads) == 1, "the mask was loaded per frame instead of once per run"
    assert len(calls) == 3
    ttls = [call["ttl"] for call in calls]
    assert all(ttl is ttls[0] for ttl in ttls), "the TTL was rebuilt per frame"
    assert list(ttls[0]) == ["Dev1/port0/line3"]


def test_the_ttl_matches_what_v30_generated_for_the_same_mask(devices, tmp_path):
    from pyrpoc.backend_utils.opto_control_contexts import MaskContext
    from pyrpoc.modalities.confocal import acquisition_core as v30_core

    daq, _ = devices
    mask = np.zeros((SCAN["y_pixels"], SCAN["x_pixels"]), np.uint8)
    mask[1:4, 2:6] = 255
    path = save_mask(tmp_path / "m.png", mask)

    params = new_params(masks=[MaskBinding(path, port=0, line=3)])
    new_ttl = build_ttl(params, daq)

    samples = max(1, int(SCAN["dwell_time_us"] * 1e-6 * 100_000.0))
    old_ttl = v30_core.generate_mask_ttl_signals(
        total_x=SCAN["x_pixels"] + SCAN["extra_left"] + SCAN["extra_right"],
        total_y=SCAN["y_pixels"],
        pixel_samples=samples,
        extra_left=SCAN["extra_left"],
        extra_right=SCAN["extra_right"],
        device_name="Dev1",
        mask_contexts=[MaskContext("mask", "m", mask, 0, 3)],
        scan_x_pixels=SCAN["x_pixels"],
    )

    assert sorted(new_ttl) == sorted(old_ttl)
    for channel, signal in new_ttl.items():
        np.testing.assert_array_equal(signal, old_ttl[channel])


def test_no_masks_means_no_ttl(devices):
    daq, _ = devices
    assert build_ttl(new_params(), daq) == {}


# --- continuous ------------------------------------------------------------


def test_continuous_keeps_going_past_num_frames(monkeypatch, devices):
    params = new_params(frames=1)
    handle, calls = run_new(monkeypatch, params, devices, continuous=True)
    assert len(calls) >= 3
    assert len(handle.datasets["intensity"]) >= 3
    assert params.num_frames == 1, "continuous must not overwrite the stored frame count"


# --- shape -----------------------------------------------------------------


def test_confocal_declares_only_the_four_program_attributes():
    defined = {name for name in vars(Confocal) if not name.startswith("__")}
    assert defined == {"uses", "params", "emits", "run"}


def test_confocal_uses_the_galvo_and_the_daq():
    assert Confocal.uses == [Galvo, DAQ]


def test_confocal_emits_one_image_stream():
    from pyrpoc.core.streams import Image2D

    assert Confocal.emits == {"intensity": Image2D}
