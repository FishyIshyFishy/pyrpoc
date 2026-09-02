"""Shape contracts for acquired arrays.

A contract says what shape and dtype an array has and what its axes mean. It is
the type half of the old ``DataKind``; the name half is a key in a program's
``emits``. A view declares which contracts it can render, so a binding can be
checked before a run starts rather than inferred from a tag mid-flight.
"""

from __future__ import annotations

import numpy as np


class Stream:
    """Base contract. Subclasses fix ``ndim``, ``axes`` and a human name."""

    name: str = "stream"
    ndim: int = 0
    axes: tuple[str, ...] = ()
    dtype = np.float32

    @classmethod
    def validate(cls, array: np.ndarray) -> None:
        arr = np.asarray(array)
        if arr.ndim != cls.ndim:
            raise ValueError(
                f"{cls.name} expects {cls.ndim} dimensions {cls.axes}, "
                f"got shape {arr.shape}"
            )
        if any(size <= 0 for size in arr.shape):
            raise ValueError(f"{cls.name} received an empty axis: shape {arr.shape}")

    @classmethod
    def coerce(cls, array: np.ndarray) -> np.ndarray:
        cls.validate(array)
        return np.asarray(array, dtype=cls.dtype)


class Image2D(Stream):
    """``(C, H, W)`` float32 — one image per channel."""

    name = "Image2D"
    ndim = 3
    axes = ("channel", "y", "x")


class Cube3D(Stream):
    """``(H, W, B)`` float32 — one value per pixel per bin (FLIM histograms)."""

    name = "Cube3D"
    ndim = 3
    axes = ("y", "x", "bin")


class Samples4D(Stream):
    """``(C, H, W, S)`` float32 — per-pixel raw samples, unaveraged.

    Split confocal's raw pixel stream. The design document files this as
    ``Image2D``; the array ``reshape_to_split_frame`` returns is four
    dimensional, so it gets its own contract rather than a false one.
    """

    name = "Samples4D"
    ndim = 4
    axes = ("channel", "y", "x", "sample")


CONTRACTS: dict[str, type[Stream]] = {
    Image2D.name: Image2D,
    Cube3D.name: Cube3D,
    Samples4D.name: Samples4D,
}
