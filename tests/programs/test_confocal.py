"""Confocal.

The hardware call is replaced with a fake, so everything above it runs for real:
the runner's thread, dataset creation from ``emits``, publishing, the save
policy, and the program's own control flow.

The mask path is checked against ``tests/reference/phase0_references.npz``,
which pins what v3.0 computed. Saved-file compatibility is checked in
tests/data/test_io.py against the frozen v3.0 storage recording.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from pyrpoc.core.modulation import MaskBinding, save_mask
from pyrpoc.data.io import load_frames
from pyrpoc.data.io import SaveTarget
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.programs.confocal import Confocal, ConfocalParams, build_ttl, channel_labels
from pyrpoc.run.runner import Runner
from tests.reference.generate_references import reference_path


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


def saving(tmp_path, name: str = "acq") -> SaveTarget:
    """Saving on, into the test's own directory. Not a parameter any more."""
    return SaveTarget(name=name, directory=str(tmp_path), enabled=True)


def new_params(frames=3, masks=()) -> ConfocalParams:
    params = ConfocalParams()
    for name, value in SCAN.items():
        setattr(params.scan, name, value)
    params.num_frames = frames
    params.modulation.masks = tuple(masks)
    return params


def run_new(monkeypatch, params, devices, *, continuous=False, capture=None, save=None):
    calls = []

    def fake_raster_scan(**kwargs):
        calls.append(kwargs)
        return fake_frame(len(calls) - 1)

    monkeypatch.setattr("pyrpoc.programs.confocal.raster_scan", fake_raster_scan)
    runner = Runner(DatasetLibrary())
    handle = runner.start(
        Confocal(), params, list(devices), continuous=continuous,
        program_key="confocal", save=save,
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


# --- frames ----------------------------------------------------------------


def test_every_frame_the_operation_returns_is_published_unchanged(monkeypatch, devices):
    handle, calls = run_new(monkeypatch, new_params(), devices)
    dataset = handle.datasets["intensity"]
    assert len(dataset) == len(calls) == 3
    for index in range(3):
        np.testing.assert_array_equal(dataset.frame(index), fake_frame(index))


def test_channel_labels_are_the_v30_ai_labels(monkeypatch, devices):
    handle, _ = run_new(monkeypatch, new_params(), devices)
    assert handle.datasets["intensity"].channel_labels == ["ai0", "ai1"]


def test_channel_labels_come_from_the_daq_device(devices):
    daq, _ = devices
    daq.config.ai_channels = (0, 3, 7)
    assert channel_labels(daq) == ["ai0", "ai3", "ai7"]


def test_the_operation_receives_the_scan_the_devices_and_the_run_settings(monkeypatch, devices):
    """v3.0 passed these as fast_axis_ao / slow_axis_ao / active_ai_channels."""
    _, calls = run_new(monkeypatch, new_params(), devices)
    for key, value in SCAN.items():
        assert calls[0][key] == value
    assert calls[0]["device_name"] == "Dev1"
    assert calls[0]["sample_rate_hz"] == 100_000.0
    assert calls[0]["fast_ao"] == 0
    assert calls[0]["slow_ao"] == 1
    assert tuple(calls[0]["ai_channels"]) == AI_CHANNELS


# --- saving ----------------------------------------------------------------


def test_saving_writes_one_tiff_per_channel(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(), devices, save=saving(tmp_path))
    for label in ("ai0", "ai1"):
        assert (tmp_path / f"acq_{label}.tiff").exists()


def test_saved_metadata_keeps_the_v30_shape(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(), devices, save=saving(tmp_path))
    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["modality_key"] == "confocal"   # v3.0 alias, kept for lab scripts
    assert meta["frames_saved"] == 3
    assert meta["frame_limit"] == 3
    assert meta["last_error"] is None
    assert sorted(meta["tiff_paths"]) == ["ai0", "ai1"]


def test_saved_frames_read_back_in_order(monkeypatch, devices, tmp_path):
    run_new(monkeypatch, new_params(), devices, save=saving(tmp_path))
    pages = load_frames(tmp_path / "acq_ai0.tiff")
    assert pages.shape == (3, SCAN["y_pixels"], SCAN["x_pixels"])
    np.testing.assert_array_equal(pages[:, 0, 0], [0.0, 1.0, 2.0])


def test_the_run_parameters_are_recorded_in_the_metadata(monkeypatch, devices, tmp_path):
    """A run's saved parameters now fully describe what happened."""
    run_new(monkeypatch, new_params(), devices, save=saving(tmp_path))
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
    from pyrpoc.core.modulation import load_mask as real_load

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


def test_the_ttl_matches_the_phase0_golden_array(devices, tmp_path):
    """The program's mask path, tied to what v3.0 computed for the same mask.

    Uses the reference geometry, which is smaller than the schema minimum --
    build_ttl is arithmetic, so it does not validate, and that is the point.
    """
    from pyrpoc.operations.raster import pixel_samples

    golden = dict(np.load(reference_path))
    daq, _ = devices

    mask = np.zeros((6, 8), np.uint8)
    mask[1:4, 2:6] = 255
    np.testing.assert_array_equal(mask, golden["mask_input"])
    path = save_mask(tmp_path / "m.png", mask)

    params = new_params(masks=[MaskBinding(path, port=0, line=3)])
    params.scan.x_pixels, params.scan.y_pixels = 8, 6
    params.scan.extra_left, params.scan.extra_right = 3, 2
    params.scan.dwell_time_us, params.daq.sample_rate_hz = 4.0, 1_000_000.0
    assert pixel_samples(4.0, 1_000_000.0) == 4, "the golden array assumes 4 samples per pixel"

    ttl = build_ttl(params, daq)
    assert list(ttl) == ["Dev1/port0/line3"]
    np.testing.assert_array_equal(ttl["Dev1/port0/line3"], golden["confocal_mask_ttl"])


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
