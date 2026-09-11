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


class Spectrum1D(Stream):
    """``(C, W)`` float32 — one spectrum per channel.

    Channel-first like ``Image2D`` rather than a bare ``(W,)``, for two
    reasons: ``axes[0] == "channel"`` is what makes ``Dataset.append`` fill in
    channel labels, and a spectrometer with more than one detector needs no new
    contract. A single-detector run is one channel, not a different shape.

    The spectral axis carries no units. What a bin means is the spectrometer's
    calibration, which is a device property rather than a shape contract -- so
    it belongs in dataset metadata when there is a real instrument to read it
    from.
    """

    name = "Spectrum1D"
    ndim = 2
    axes = ("channel", "wavelength")


class Mask2D(Stream):
    """``(H, W)`` uint8 -- an authored region, 0 outside and non-zero inside.

    Its own contract rather than a one-channel ``Image2D``, and that is what
    makes the Modulation picker work: a mask is chosen with
    ``DatasetLibrary.matching(Mask2D)``, so a contract shared with acquired
    images would offer every run as a mask. The spec *is* the filter.

    uint8 rather than float32 because a mask is a decision per pixel, not a
    measurement. Nothing downstream minds -- the writers and
    ``normalize_channels`` cast for themselves.
    """

    name = "Mask2D"
    ndim = 2
    axes = ("y", "x")
    dtype = np.uint8


CONTRACTS: dict[str, type[Stream]] = {
    Image2D.name: Image2D,
    Cube3D.name: Cube3D,
    Samples4D.name: Samples4D,
    Spectrum1D.name: Spectrum1D,
    Mask2D.name: Mask2D,
}
