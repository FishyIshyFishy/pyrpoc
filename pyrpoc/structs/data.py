"""The kinds of data the application passes around.

A ``Data`` subclass fixes an array's shape, dtype and axis meanings. Programs
name the kind each output carries and data panels name the kinds they render,
so a binding is checked before a run starts rather than inferred mid-flight.
"""

from __future__ import annotations

import numpy as np


class Data:
    """Base kind. Subclasses fix ``ndim``, ``axes`` and a human name."""

    name: str = "data"
    ndim: int = 0
    axes: tuple[str, ...] = ()
    dtype = np.float32

    @classmethod
    def validate(cls, array: np.ndarray) -> None:
        arr = np.asarray(array)
        if arr.ndim != cls.ndim:
            raise ValueError(
                f"{cls.name} expects {cls.ndim} dimensions {cls.axes}, got shape {arr.shape}"
            )
        if any(size <= 0 for size in arr.shape):
            raise ValueError(f"{cls.name} received an empty axis: shape {arr.shape}")

    @classmethod
    def coerce(cls, array: np.ndarray) -> np.ndarray:
        cls.validate(array)
        return np.asarray(array, dtype=cls.dtype)


class Image2D(Data):
    """``(C, H, W)`` float32: one image per channel."""

    name = "Image2D"
    ndim = 3
    axes = ("channel", "y", "x")


class Cube3D(Data):
    """``(H, W, B)`` float32: one value per pixel per bin (FLIM histograms)."""

    name = "Cube3D"
    ndim = 3
    axes = ("y", "x", "bin")


class Samples4D(Data):
    """``(C, H, W, S)`` float32: per-pixel raw samples, unaveraged."""

    name = "Samples4D"
    ndim = 4
    axes = ("channel", "y", "x", "sample")


class Spectrum1D(Data):
    """``(C, W)`` float32: one spectrum per channel.

    Channel-first like ``Image2D`` so ``Dataset.append`` fills in channel
    labels and a multi-detector spectrometer needs no new kind. Bins carry no
    units: that is the spectrometer's calibration, not a shape contract.
    """

    name = "Spectrum1D"
    ndim = 2
    axes = ("channel", "wavelength")


class Mask2D(Data):
    """``(H, W)`` uint8: an authored region, 0 outside and non-zero inside.

    Its own kind rather than a one-channel ``Image2D``, because pickers filter
    on the kind: sharing one with acquired images would offer every run as a
    mask. uint8 because a mask is a decision per pixel, not a measurement.
    """

    name = "Mask2D"
    ndim = 2
    axes = ("y", "x")
    dtype = np.uint8


# Every kind a saved recording can name, so a file's kind string finds its class.
DATA_KINDS: dict[str, type[Data]] = {
    kind.name: kind for kind in (Image2D, Cube3D, Samples4D, Spectrum1D, Mask2D)
}
