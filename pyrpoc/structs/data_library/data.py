"""The kinds of data the application passes around, one noun each.

A kind is what the data is, not how many axes it has: an ``Image`` and a
``Mask`` are both 2-D, but nothing that wants one should be offered the other.
Programs name the kind each output carries and data panels name the kinds they
render, so a binding is checked before a run starts rather than inferred
mid-flight. ``axes`` and ``dtype`` are how a kind checks a frame it is given.
"""

from __future__ import annotations

import numpy as np


class Data:
    """Base kind. Subclasses fix ``axes`` and, if not float32, ``dtype``."""

    # The class name, set per subclass: what a recording's ``kind`` field holds.
    name: str = "Data"
    axes: tuple[str, ...] = ()
    dtype = np.float32

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.name = cls.__name__

    @classmethod
    def validate(cls, array: np.ndarray) -> None:
        arr = np.asarray(array)
        if arr.ndim != len(cls.axes):
            raise ValueError(f"{cls.name} expects axes {cls.axes}, got shape {arr.shape}")
        if any(size <= 0 for size in arr.shape):
            raise ValueError(f"{cls.name} received an empty axis: shape {arr.shape}")

    @classmethod
    def coerce(cls, array: np.ndarray) -> np.ndarray:
        cls.validate(array)
        return np.asarray(array, dtype=cls.dtype)


class Image(Data):
    """One picture per channel."""

    axes = ("channel", "y", "x")


class DecayHistogram(Data):
    """Photon arrival times binned per pixel: what FLIM measures."""

    axes = ("y", "x", "bin")


class RawSamples(Data):
    """Every sample a detector took at each pixel, before averaging."""

    axes = ("channel", "y", "x", "sample")


class Spectrum(Data):
    """Intensity per wavelength bin, one spectrum per channel.

    Channel-first like ``Image`` so ``Dataset.append`` fills in channel
    labels and a multi-detector spectrometer needs no new kind. Bins carry no
    units: that is the spectrometer's calibration, not part of the kind.
    """

    axes = ("channel", "wavelength")


class Mask(Data):
    """An authored region: 0 outside, non-zero inside.

    uint8 because a mask is a decision per pixel, not a measurement.
    """

    axes = ("y", "x")
    dtype = np.uint8


# Every kind a saved recording can name, so a file's kind string finds its class.
DATA_KINDS: dict[str, type[Data]] = {
    kind.name: kind for kind in (Image, DecayHistogram, RawSamples, Spectrum, Mask)
}
