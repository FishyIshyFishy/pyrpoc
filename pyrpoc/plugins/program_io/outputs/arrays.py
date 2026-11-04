"""Data that is one array with named axes, and its kinds: images, decay histograms, raw
samples and spectra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from pyrpoc.core.program_output import Data, data_kind_registry


@dataclass(frozen=True, eq=False)
class ArrayData(Data):
    """Data that is one array with named axes. A first axis of ``"channel"``
    is labelled by ``channels``, or numbered when they are not given."""

    axes: ClassVar[tuple[str, ...]] = ()
    dtype: ClassVar[type[np.generic]] = np.float32

    array: np.ndarray
    channels: Sequence[str] = ()

    def __post_init__(self) -> None:
        array = np.asarray(self.array, dtype=self.dtype)
        if array.ndim != len(self.axes):
            raise ValueError(f"{self.name} expects axes {self.axes}, got shape {array.shape}")
        if any(size <= 0 for size in array.shape):
            raise ValueError(f"{self.name} received an empty axis: shape {array.shape}")
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "channels", self.labels_for(array))

    def labels_for(self, array: np.ndarray) -> tuple[str, ...]:
        if self.axes[0] != "channel":
            if self.channels:
                raise ValueError(f"{self.name} has no channel axis to label")
            return ()
        count = array.shape[0]
        if not self.channels:
            return tuple(f"channel_{index}" for index in range(count))
        if len(self.channels) != count:
            raise ValueError(f"{self.name} has {count} channels but {len(self.channels)} labels")
        return tuple(self.channels)

    @property
    def nbytes(self) -> int:
        return self.array.nbytes

    def summary(self) -> str:
        shape = " × ".join(map(str, self.array.shape))
        labels = f" ({', '.join(self.channels)})" if self.channels else ""
        return f"{shape} along {', '.join(self.axes)}{labels}"


@data_kind_registry.register("Image")
class Image(ArrayData):
    """One picture per channel."""

    axes = ("channel", "y", "x")


@data_kind_registry.register("DecayHistogram")
class DecayHistogram(ArrayData):
    """Photon arrival times binned per pixel: what FLIM measures."""

    axes = ("y", "x", "bin")


@data_kind_registry.register("RawSamples")
class RawSamples(ArrayData):
    """Every sample a detector took at each pixel, before averaging."""

    axes = ("channel", "y", "x", "sample")


@data_kind_registry.register("Spectrum")
class Spectrum(ArrayData):
    """Intensity per wavelength bin, one spectrum per channel."""

    # Bins carry no units: that is the spectrometer's calibration, not the kind's.
    axes = ("channel", "wavelength")
