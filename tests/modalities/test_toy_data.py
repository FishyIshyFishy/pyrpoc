from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.modalities.helpers.toy_data import (
    apply_masks,
    generate_toy_confocal_frame,
    generate_toy_split_confocal_frame,
    overlay_split_label,
)


def confocal(**overrides):
    kwargs = dict(
        x_pixels=24,
        y_pixels=20,
        active_channels=[0, 1],
        frame_index=0,
        mask_contexts=[],
        fast_axis_offset=0.0,
        fast_axis_amplitude=1.0,
        slow_axis_offset=0.0,
        slow_axis_amplitude=1.0,
    )
    kwargs.update(overrides)
    return generate_toy_confocal_frame(**kwargs)


def test_confocal_shape_dtype_and_range():
    frame = confocal()
    assert frame.shape == (2, 20, 24)
    assert frame.dtype == np.float32
    assert frame.min() >= 0.0
    assert frame.max() <= 1.0 + 1e-5


def test_confocal_is_deterministic():
    assert np.array_equal(confocal(), confocal())


def test_confocal_frame_index_changes_output():
    assert not np.array_equal(confocal(frame_index=0), confocal(frame_index=1))


def test_confocal_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        confocal(x_pixels=0)


def test_confocal_rejects_empty_channels():
    with pytest.raises(ValueError):
        confocal(active_channels=[])


def test_apply_masks_boosts_masked_region():
    frame = np.full((1, 8, 8), 0.5, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    ctx = MaskContext(optocontrol_key="mask", alias="m", mask=mask, daq_port=0, daq_line=0)
    apply_masks(frame, [ctx])
    assert np.all(frame[0, 2:5, 2:5] > 0.5)
    assert np.all(frame[0, 0, 0] == 0.5)


def test_apply_masks_resizes_mismatched_mask():
    frame = np.full((1, 8, 8), 0.5, dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.uint8)
    ctx = MaskContext(optocontrol_key="mask", alias="m", mask=mask, daq_port=0, daq_line=0)
    apply_masks(frame, [ctx])  # should resize 4x4 -> 8x8 without error
    assert np.all(frame[0] > 0.5)


def test_split_confocal_shapes():
    split, raw = generate_toy_split_confocal_frame(
        x_pixels=16,
        y_pixels=12,
        active_channels=[0, 1],
        frame_index=2,
        mask_contexts=[],
        fast_axis_offset=0.0,
        fast_axis_amplitude=1.0,
        slow_axis_offset=0.0,
        slow_axis_amplitude=1.0,
        t0_samples=3,
        t1_samples=2,
        pixel_samples=10,
    )
    assert split.shape == (4, 12, 16)  # 2 channels * 2 split halves
    assert raw.shape == (2, 12, 16, 10)
    assert split.dtype == np.float32


@pytest.mark.parametrize(
    "bad",
    [
        dict(pixel_samples=0),
        dict(t0_samples=0),
        dict(t1_samples=-1),
        dict(t0_samples=6, t1_samples=6, pixel_samples=10),  # t0+t1 >= pixel_samples
    ],
)
def test_split_confocal_validates_timing(bad):
    kwargs = dict(
        x_pixels=8,
        y_pixels=8,
        active_channels=[0],
        frame_index=0,
        mask_contexts=[],
        fast_axis_offset=0.0,
        fast_axis_amplitude=1.0,
        slow_axis_offset=0.0,
        slow_axis_amplitude=1.0,
        t0_samples=3,
        t1_samples=2,
        pixel_samples=10,
    )
    kwargs.update(bad)
    with pytest.raises(ValueError):
        generate_toy_split_confocal_frame(**kwargs)


def test_overlay_split_label_preserves_shape():
    image = np.zeros((40, 40), dtype=np.float32)
    out = overlay_split_label(image, "1")
    assert out.shape == image.shape


def test_overlay_split_label_ignores_tiny_image():
    image = np.zeros((4, 4), dtype=np.float32)
    out = overlay_split_label(image, "1")
    assert np.array_equal(out, image)
