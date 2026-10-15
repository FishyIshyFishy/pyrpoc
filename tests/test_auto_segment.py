"""Auto-segment's mask, and a mask output saved beside an image output."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrpoc.data_library.format import META_SUFFIX, load_recording
from pyrpoc.plugins.programs.auto_segment.otsu import otsu_mask
from pyrpoc.structs.data_library.data import Image2D, Mask2D
from pyrpoc.structs.plugins.params import BlockStore
from pyrpoc.structs.plugins.programs.program import Program, RunContext

from .helpers import Recorder


def bright_square() -> np.ndarray:
    """A dim, slightly noisy 32x32 field with a bright 12x12 square in it."""
    rng = np.random.default_rng(0)
    plane = rng.normal(10.0, 1.0, (32, 32)).astype(np.float32)
    plane[8:20, 10:22] += 100.0
    return plane


class SegmentingProgram(Program):
    """Auto-segment's outputs without the hardware: two frames and their masks."""

    display_name = "Segment fixture"
    emits = {"intensity": Image2D, "mask": Mask2D}

    def run(self, ctx: RunContext) -> None:
        for _ in range(2):
            frame = bright_square()[None, :, :]
            ctx.publish("intensity", frame, channels=["ai0"])
            ctx.publish("mask", otsu_mask(frame[0]))


def test_otsu_finds_the_bright_region() -> None:
    mask = otsu_mask(bright_square())

    expected = np.zeros((32, 32), dtype=np.uint8)
    expected[8:20, 10:22] = 255
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, expected)


def test_a_flat_frame_masks_nothing() -> None:
    assert not otsu_mask(np.full((8, 8), 3.0, dtype=np.float32)).any()


def test_mask_saves_and_reloads_beside_the_image(tmp_path: Path) -> None:
    saved = Recorder().record(SegmentingProgram(), "segment", BlockStore(), tmp_path, "seg")
    loaded = {d.output: d for d in load_recording(tmp_path / f"seg{META_SUFFIX}")}

    assert (tmp_path / "seg_mask.tiff").is_file() and (tmp_path / "seg_ai0.tiff").is_file()
    for original in saved:
        reread = loaded[original.output]
        assert reread.spec is original.spec
        for before, after in zip(original.frames(), reread.frames(), strict=True):
            assert after.dtype == before.dtype
            np.testing.assert_array_equal(before, after)
