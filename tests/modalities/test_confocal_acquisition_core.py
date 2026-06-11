from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.modalities.confocal.acquisition_core import (
    extract_kept_samples,
    extract_mask_contexts,
    generate_mask_ttl_signals,
    preprocess_mask_to_scan_grid,
    reshape_to_frame,
    resize_mask_nearest,
)
from pyrpoc.optocontrols.mask import MaskOptoControl


# --------------------------------------------------------------------------- #
# resize_mask_nearest
# --------------------------------------------------------------------------- #

def test_resize_upsamples_nearest():
    src = np.array([[True, False]], dtype=bool)
    out = resize_mask_nearest(src, target_h=2, target_w=4)
    assert out.shape == (2, 4)
    assert out.tolist() == [[True, True, False, False], [True, True, False, False]]


def test_resize_empty_source_returns_zeros():
    out = resize_mask_nearest(np.zeros((0, 0), dtype=bool), target_h=3, target_w=3)
    assert out.shape == (3, 3)
    assert not out.any()


# --------------------------------------------------------------------------- #
# preprocess_mask_to_scan_grid
# --------------------------------------------------------------------------- #

def test_preprocess_pads_into_scan_region():
    mask = np.ones((2, 2), dtype=np.uint8)
    padded = preprocess_mask_to_scan_grid(mask, total_x=4, total_y=2, scan_x_pixels=2, extra_left=1, extra_right=1)
    assert padded.shape == (2, 4)
    # only the middle two columns (the scan region) are active
    assert padded[:, 0].sum() == 0
    assert padded[:, 3].sum() == 0
    assert padded[:, 1:3].all()


def test_preprocess_resizes_mismatched_mask():
    mask = np.ones((1, 1), dtype=np.uint8)
    padded = preprocess_mask_to_scan_grid(mask, total_x=4, total_y=2, scan_x_pixels=2, extra_left=1, extra_right=1)
    assert padded[:, 1:3].all()


def test_preprocess_rejects_total_x_mismatch():
    with pytest.raises(ValueError):
        preprocess_mask_to_scan_grid(np.ones((2, 2), np.uint8), total_x=5, total_y=2, scan_x_pixels=2, extra_left=1, extra_right=1)


def test_preprocess_rejects_non_2d_mask():
    with pytest.raises(ValueError):
        preprocess_mask_to_scan_grid(np.ones((2, 2, 2), np.uint8), total_x=4, total_y=2, scan_x_pixels=2, extra_left=1, extra_right=1)


# --------------------------------------------------------------------------- #
# generate_mask_ttl_signals
# --------------------------------------------------------------------------- #

def test_generate_ttl_signal_shape_and_count():
    ctx = MaskContext(optocontrol_key="mask", alias="m", mask=np.ones((2, 2), np.uint8), daq_port=0, daq_line=3)
    signals = generate_mask_ttl_signals(
        total_x=4, total_y=2, pixel_samples=2, extra_left=1, extra_right=1,
        device_name="Dev1", mask_contexts=[ctx], scan_x_pixels=2,
    )
    key = "Dev1/port0/line3"
    assert key in signals
    assert signals[key].shape == (2 * 4 * 2,)
    assert signals[key].sum() == 2 * 2 * 2  # rows * scan_cols * pixel_samples


def test_generate_ttl_skips_none_mask():
    ctx = MaskContext(optocontrol_key="mask", alias="m", mask=None, daq_port=0, daq_line=0)
    signals = generate_mask_ttl_signals(
        total_x=4, total_y=2, pixel_samples=1, extra_left=1, extra_right=1,
        device_name="Dev1", mask_contexts=[ctx], scan_x_pixels=2,
    )
    assert signals == {}


def test_generate_ttl_skips_empty_mask():
    ctx = MaskContext(optocontrol_key="mask", alias="m", mask=np.zeros((2, 2), np.uint8), daq_port=0, daq_line=0)
    signals = generate_mask_ttl_signals(
        total_x=4, total_y=2, pixel_samples=1, extra_left=1, extra_right=1,
        device_name="Dev1", mask_contexts=[ctx], scan_x_pixels=2,
    )
    assert signals == {}


# --------------------------------------------------------------------------- #
# extract_kept_samples / reshape_to_frame
# --------------------------------------------------------------------------- #

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

def test_extract_mask_contexts_requires_prepared_context():
    control = MaskOptoControl()
    with pytest.raises(TypeError):
        extract_mask_contexts([control])


def test_extract_mask_contexts_collects_prepared():
    control = MaskOptoControl()
    control.get_context()  # prepares control.context
    contexts = extract_mask_contexts([control])
    assert len(contexts) == 1
    assert isinstance(contexts[0], MaskContext)


def test_extract_mask_contexts_empty_list():
    assert extract_mask_contexts([]) == []
