"""Transforms, checked against the v3.0 display code they were lifted from."""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.data.transforms import channel_levels, normalize_channels


def v30_normalize(arr):
    """The body of Tiled2DDisplay.get_normalized_data_3d, inlined."""
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.zeros_like(arr, dtype=np.float32)
    for idx in range(arr.shape[0]):
        channel = arr[idx]
        lo = float(np.min(channel))
        hi = float(np.max(channel))
        if hi > lo:
            norm[idx] = (channel - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def test_matches_the_v30_display_normalisation():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(4, 6, 8)).astype(np.float32)
    np.testing.assert_array_equal(normalize_channels(data), v30_normalize(data))


def test_scales_each_channel_independently():
    data = np.stack([np.array([[0.0, 10.0]]), np.array([[-5.0, -1.0]])]).astype(np.float32)
    out = normalize_channels(data)
    np.testing.assert_allclose(out[0], [[0.0, 1.0]])
    np.testing.assert_allclose(out[1], [[0.0, 1.0]])


def test_a_flat_channel_stays_zero():
    out = normalize_channels(np.full((1, 2, 2), 7.0, np.float32))
    np.testing.assert_array_equal(out, np.zeros((1, 2, 2), np.float32))


def test_none_passes_through():
    assert normalize_channels(None) is None


def test_rejects_a_non_chw_array():
    with pytest.raises(ValueError, match="channels-first"):
        normalize_channels(np.zeros((4, 5), np.float32))


def test_channel_levels_never_degenerate():
    lo, hi = channel_levels(np.full((2, 2), 3.0, np.float32))
    assert lo == 3.0 and hi > lo


def test_channel_levels_are_the_min_and_max():
    assert channel_levels(np.array([[1.0, 5.0]], np.float32)) == (1.0, 5.0)


def test_normalisation_is_idempotent():
    data = np.random.default_rng(5).normal(size=(2, 3, 4)).astype(np.float32)
    once = normalize_channels(data)
    np.testing.assert_allclose(normalize_channels(once), once, atol=1e-6)
