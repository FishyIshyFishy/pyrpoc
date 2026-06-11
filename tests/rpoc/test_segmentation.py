from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.rpoc.segmentation_methods import (
    adaptive_threshold,
    apply_morph,
    components_to_labels,
    labels_to_contours,
    segment,
    threshold_components,
    to_u8,
    watershed,
)


def three_level_image() -> np.ndarray:
    """Background 0, a large mid-intensity target, and one bright pixel to set the max.

    to_u8 stretches max->255, so the mid target lands inside a [0.2, 0.8] band.
    """
    image = np.zeros((40, 40), dtype=np.float32)
    image[10:26, 10:26] = 0.5
    image[0, 0] = 1.0
    return image


# --------------------------------------------------------------------------- #
# to_u8
# --------------------------------------------------------------------------- #

def test_to_u8_constant_image_is_zeros():
    out = to_u8(np.full((4, 4), 7.0, dtype=np.float32))
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_to_u8_stretches_to_full_range():
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = to_u8(image)
    assert out[0, 0] == 0
    assert out[0, 2] == 255


def test_to_u8_rejects_non_2d():
    with pytest.raises(ValueError):
        to_u8(np.zeros((3, 4, 5), dtype=np.float32))


# --------------------------------------------------------------------------- #
# apply_morph / components_to_labels
# --------------------------------------------------------------------------- #

def test_apply_morph_returns_uint8_same_shape():
    mask = (np.random.default_rng(0).random((20, 20)) > 0.5).astype(np.uint8) * 255
    out = apply_morph(mask)
    assert out.dtype == np.uint8
    assert out.shape == mask.shape


def test_components_to_labels_counts_and_filters():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:14, 2:14] = 255   # 144 px -> kept
    mask[2:14, 26:38] = 255  # 144 px -> kept
    mask[30:34, 30:34] = 255  # 16 px -> filtered out
    labels = components_to_labels(mask, min_area=120)
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}


# --------------------------------------------------------------------------- #
# threshold_components / watershed / adaptive_threshold
# --------------------------------------------------------------------------- #

def test_threshold_components_detects_target():
    labels = threshold_components(three_level_image(), low=0.1, high=0.9, min_area=50)
    assert labels.dtype == np.int32
    assert labels.max() >= 1


def test_threshold_components_empty_when_nothing_matches():
    labels = threshold_components(np.zeros((32, 32), dtype=np.float32))
    assert labels.max() == 0


def test_watershed_runs_and_returns_int32():
    labels = watershed(three_level_image(), low=0.1, high=0.9, min_area=20)
    assert labels.dtype == np.int32
    assert labels.shape == (40, 40)


def test_watershed_on_blank_image_is_zeros():
    labels = watershed(np.zeros((32, 32), dtype=np.float32))
    assert np.all(labels == 0)


def test_adaptive_threshold_handles_even_block_size():
    labels = adaptive_threshold(three_level_image(), block_size=60, min_area=20)
    assert labels.dtype == np.int32
    assert labels.shape == (40, 40)


# --------------------------------------------------------------------------- #
# segment dispatch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["threshold + components", "Threshold + Components", "WATERSHED", "adaptive threshold"])
def test_segment_dispatches_known_methods(method):
    labels = segment(three_level_image(), method, min_area=20)
    assert labels.dtype == np.int32


def test_segment_unknown_method_raises():
    with pytest.raises(ValueError):
        segment(three_level_image(), "magic")


# --------------------------------------------------------------------------- #
# labels_to_contours
# --------------------------------------------------------------------------- #

def test_labels_to_contours_returns_xy_points():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[5:25, 5:25] = 1  # 400 px
    contours = labels_to_contours(labels, min_area=50)
    assert len(contours) == 1
    assert contours[0].ndim == 2
    assert contours[0].shape[1] == 2


def test_labels_to_contours_filters_small_regions():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[0:3, 0:3] = 1  # 9 px
    assert labels_to_contours(labels, min_area=120) == []


def test_labels_to_contours_requires_2d():
    with pytest.raises(ValueError):
        labels_to_contours(np.zeros((3, 4, 5), dtype=np.int32))
