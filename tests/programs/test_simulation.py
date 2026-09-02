"""Simulation.

Nothing is faked here -- there is nothing to fake. The whole stack above the
hardware boundary runs for real, which is exactly what this program exists for:
the runner's thread, dataset creation from ``emits``, publishing, the save
policy and the mask path, all on a machine with no instruments.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from pyrpoc.core.modulation import MaskBinding, save_mask
from pyrpoc.core.streams import Image2D
from pyrpoc.data.io import load_frames
from pyrpoc.data.io import SaveTarget
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.operations.simulation import PATTERNS, combine_masks, synthetic_frame
from pyrpoc.programs.simulation import Simulation, SimulationParams, build_mask
from pyrpoc.run.runner import Runner


def saving(tmp_path, name: str = "acq") -> SaveTarget:
    """Saving on, into the test's own directory. Not a parameter any more."""
    return SaveTarget(name=name, directory=str(tmp_path), enabled=True)


def new_params(*, frames=3, masks=(), pattern="cells") -> SimulationParams:
    params = SimulationParams()
    params.frame.x_pixels = 32
    params.frame.y_pixels = 24
    params.frame.channels = 2
    params.signal.pattern = pattern
    params.num_frames = frames
    params.frame_interval_ms = 0
    params.modulation.masks = tuple(masks)
    return params


def run_new(params, *, continuous=False, save=None):
    runner = Runner(DatasetLibrary())
    handle = runner.start(
        Simulation(), params, [], continuous=continuous,
        program_key="simulation", save=save,
    )
    if continuous:
        dataset = handle.datasets["intensity"]
        for _ in range(500):
            if len(dataset) >= 3:
                break
            threading.Event().wait(0.005)
        runner.stop()
    handle.thread.join(timeout=10)
    return handle


# --- claiming nothing -------------------------------------------------------


def test_it_claims_no_devices():
    assert Simulation.uses == []


def test_it_runs_with_an_empty_inventory():
    handle = run_new(new_params())
    assert len(handle.datasets["intensity"]) == 3


# --- the frames -------------------------------------------------------------


def test_frames_are_the_declared_shape_and_contract():
    handle = run_new(new_params())
    dataset = handle.datasets["intensity"]
    assert dataset.spec is Image2D
    assert dataset.latest().shape == (2, 24, 32)
    assert dataset.latest().dtype == np.float32


def test_channels_are_labelled_and_differ_from_each_other():
    handle = run_new(new_params())
    dataset = handle.datasets["intensity"]
    assert dataset.channel_labels == ["sim0", "sim1"]
    frame = dataset.frame(0)
    assert not np.allclose(frame[0], frame[1])


def test_frames_move():
    handle = run_new(new_params())
    dataset = handle.datasets["intensity"]
    assert not np.allclose(dataset.frame(0), dataset.frame(2))


def test_the_same_seed_gives_the_same_pixels():
    first = run_new(new_params()).datasets["intensity"].frame(1)
    second = run_new(new_params()).datasets["intensity"].frame(1)
    assert np.array_equal(first, second)


def test_a_different_seed_gives_different_pixels():
    params = new_params()
    params.signal.seed = 99
    other = run_new(params).datasets["intensity"].frame(1)
    assert not np.allclose(run_new(new_params()).datasets["intensity"].frame(1), other)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_pattern_produces_a_usable_frame(pattern):
    frame = synthetic_frame(
        x_pixels=16,
        y_pixels=12,
        channels=2,
        pattern=pattern,
        signal_level=1.0,
        noise_level=0.0,
        drift_pixels_per_frame=1.0,
        mask_gain=0.0,
        seed=7,
        frame_index=0,
    )
    assert frame.shape == (2, 12, 16)
    assert np.isfinite(frame).all()
    assert frame.min() >= 0.0
    assert frame.max() > 0.0


def test_an_unknown_pattern_is_an_error():
    with pytest.raises(ValueError):
        synthetic_frame(
            x_pixels=8, y_pixels=8, channels=1, pattern="spiral", signal_level=1.0,
            noise_level=0.0, drift_pixels_per_frame=0.0, mask_gain=0.0, seed=0, frame_index=0,
        )


def test_noise_is_the_only_source_of_frame_to_frame_change_when_nothing_drifts():
    common = dict(
        x_pixels=16, y_pixels=16, channels=1, pattern="cells", signal_level=1.0,
        drift_pixels_per_frame=0.0, mask_gain=0.0, seed=3,
    )
    quiet_a = synthetic_frame(**common, noise_level=0.0, frame_index=0)
    quiet_b = synthetic_frame(**common, noise_level=0.0, frame_index=5)
    noisy = synthetic_frame(**common, noise_level=0.2, frame_index=0)
    assert np.array_equal(quiet_a, quiet_b)
    assert not np.allclose(quiet_a, noisy)


# --- masks ------------------------------------------------------------------


def mask_file(tmp_path: Path) -> Path:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[:12, :] = 255
    return save_mask(tmp_path / "mask.png", mask)


def test_masked_pixels_are_brighter(tmp_path):
    binding = MaskBinding(path=mask_file(tmp_path))
    params = new_params(masks=[binding], pattern="flat")
    params.signal.noise_level = 0.0
    params.signal.mask_gain = 1.0

    frame = run_new(params).datasets["intensity"].frame(0)
    illuminated, dark = frame[0][:12], frame[0][12:]
    assert np.allclose(illuminated, 2.0 * dark.max())
    assert np.allclose(dark, 0.5)


def test_no_masks_means_no_mask_plane():
    assert build_mask(new_params()) is None


def test_masks_are_resized_to_the_frame_and_or_ed_together():
    left = np.zeros((8, 8), np.uint8)
    left[:, :4] = 255
    top = np.zeros((8, 8), np.uint8)
    top[:4, :] = 255

    combined = combine_masks([left, top], y_pixels=16, x_pixels=16)
    assert combined.shape == (16, 16)
    assert combined[:8, :].all() and combined[:, :8].all()
    assert not combined[8:, 8:].any()


# --- the rest of the stack --------------------------------------------------


def test_continuous_ignores_the_frame_count_without_overwriting_it():
    params = new_params(frames=1)
    handle = run_new(params, continuous=True)
    assert len(handle.datasets["intensity"]) >= 3
    assert params.num_frames == 1


def test_saving_writes_one_tiff_per_channel_and_a_metadata_file(tmp_path):
    run_new(new_params(), save=saving(tmp_path))

    meta = json.loads((tmp_path / "acq_meta.json").read_text(encoding="utf-8"))
    assert meta["program_key"] == "simulation"
    assert meta["frames_saved"] == 3
    assert meta["devices"] == {}
    assert sorted(meta["tiff_paths"]) == ["sim0", "sim1"]

    frames = load_frames(Path(meta["tiff_paths"]["sim0"]))
    assert frames.shape == (3, 24, 32)
