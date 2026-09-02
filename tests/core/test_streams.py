"""Shape contracts."""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.core.streams import Cube3D, Image2D, Samples4D


def test_image2d_accepts_chw():
    Image2D.validate(np.zeros((3, 4, 5), dtype=np.float32))


def test_image2d_rejects_a_bare_image():
    with pytest.raises(ValueError, match="3 dimensions"):
        Image2D.validate(np.zeros((4, 5), dtype=np.float32))


def test_cube3d_accepts_hwb():
    Cube3D.validate(np.zeros((4, 5, 125), dtype=np.float32))


def test_samples4d_is_four_dimensional():
    """Split confocal's raw stream. The design doc files this as Image2D; it is not."""
    assert Samples4D.ndim == 4
    Samples4D.validate(np.zeros((2, 4, 5, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        Image2D.validate(np.zeros((2, 4, 5, 8), dtype=np.float32))


def test_empty_axis_is_rejected():
    with pytest.raises(ValueError, match="empty axis"):
        Image2D.validate(np.zeros((0, 4, 5), dtype=np.float32))


def test_coerce_casts_to_float32():
    out = Image2D.coerce(np.zeros((1, 2, 2), dtype=np.float64))
    assert out.dtype == np.float32
