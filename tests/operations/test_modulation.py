"""Mask resizing, padding onto the scan grid, and TTL generation.

Moved with the code from tests/modalities/test_confocal_acquisition_core.py.
The tests that exercised extract_mask_contexts went with the optocontrols: a
mask is a MaskBinding plus a loaded array now, not a prepared context object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyrpoc.core.modulation import MaskBinding
from pyrpoc.core.params import ScanGroup
from pyrpoc.operations.modulation import (
    mask_ttl,
    preprocess_mask_to_scan_grid,
    resize_mask_nearest,
    split_mask_ttl,
)


def scan_group(x_pixels=8, y_pixels=6, extra_left=3, extra_right=2) -> ScanGroup:
    return ScanGroup(
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        extra_left=extra_left,
        extra_right=extra_right,
    )


def bind(mask, port=0, line=3):
    return [(MaskBinding(Path("m.png"), port, line), mask)]


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
    scan = scan_group(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1)
    signals = mask_ttl(
        bind(np.ones((2, 2), np.uint8)), scan=scan, pixel_samples=2, device_name="Dev1"
    )
    key = "Dev1/port0/line3"
    assert key in signals
    assert signals[key].shape == (2 * 4 * 2,)
    assert signals[key].sum() == 2 * 2 * 2  # rows * scan_cols * pixel_samples


def test_generate_ttl_skips_none_mask():
    scan = scan_group(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1)
    assert mask_ttl(bind(None), scan=scan, pixel_samples=1, device_name="Dev1") == {}


def test_generate_ttl_skips_empty_mask():
    scan = scan_group(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1)
    signals = mask_ttl(
        bind(np.zeros((2, 2), np.uint8)), scan=scan, pixel_samples=1, device_name="Dev1"
    )
    assert signals == {}


def test_split_gate_is_a_subset_of_the_ungated_ttl():
    """The one thing split confocal does differently: truncate to t0."""
    scan = scan_group(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1)
    masks = bind(np.ones((2, 2), np.uint8))
    ungated = mask_ttl(masks, scan=scan, pixel_samples=4, device_name="Dev1")
    gated = split_mask_ttl(
        bind(np.ones((2, 2), np.uint8)),
        scan=scan,
        pixel_samples=4,
        device_name="Dev1",
        t0_samples=1,
    )
    key = "Dev1/port0/line3"
    assert not np.any(gated[key] & ~ungated[key])
    assert gated[key].sum() == ungated[key].sum() // 4


def test_a_t0_at_or_past_the_pixel_length_gates_nothing():
    scan = scan_group(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1)
    gated = split_mask_ttl(
        bind(np.ones((2, 2), np.uint8)),
        scan=scan,
        pixel_samples=2,
        device_name="Dev1",
        t0_samples=2,
    )
    ungated = mask_ttl(
        bind(np.ones((2, 2), np.uint8)), scan=scan, pixel_samples=2, device_name="Dev1"
    )
    key = "Dev1/port0/line3"
    np.testing.assert_array_equal(gated[key], ungated[key])


# --------------------------------------------------------------------------- #
# extract_kept_samples / reshape_to_frame
# --------------------------------------------------------------------------- #
