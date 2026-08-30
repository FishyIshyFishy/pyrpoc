"""MaskBinding and mask file I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyrpoc.core.modulation import MaskBinding, load_mask, save_mask


def test_channel_string_matches_the_v3_0_format():
    binding = MaskBinding(Path("m.png"), port=0, line=3)
    assert binding.channel("Dev1") == "Dev1/port0/line3"


def test_binding_round_trip():
    binding = MaskBinding(Path("m.png"), 1, 2)
    assert MaskBinding.from_dict(binding.to_dict()) == binding


def test_binding_coerces_a_string_path():
    assert MaskBinding("m.png").path == Path("m.png")  # type: ignore[arg-type]


def test_save_then_load(tmp_path):
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:4, 2:6] = 255
    path = save_mask(tmp_path / "m.png", mask)
    np.testing.assert_array_equal(load_mask(path), mask)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mask(tmp_path / "nope.png")


def test_save_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError):
        save_mask(tmp_path / "m.png", np.zeros((2, 3, 4), dtype=np.uint8))
