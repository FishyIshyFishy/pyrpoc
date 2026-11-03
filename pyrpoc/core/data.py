"""The kinds of data programs produce, one class each."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


class Data:
    """One value of data. ``name`` is the class name, which a recording's
    ``kind`` field holds."""

    name: ClassVar[str] = "Data"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.name = cls.__name__

    @property
    def nbytes(self) -> int:
        """Memory held, in bytes."""
        raise NotImplementedError

    def summary(self) -> str:
        """One line saying what this is."""
        raise NotImplementedError


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


class Image(ArrayData):
    """One picture per channel."""

    axes = ("channel", "y", "x")


class DecayHistogram(ArrayData):
    """Photon arrival times binned per pixel: what FLIM measures."""

    axes = ("y", "x", "bin")


class RawSamples(ArrayData):
    """Every sample a detector took at each pixel, before averaging."""

    axes = ("channel", "y", "x", "sample")


class Spectrum(ArrayData):
    """Intensity per wavelength bin, one spectrum per channel."""

    # Bins carry no units: that is the spectrometer's calibration, not the kind's.
    axes = ("channel", "wavelength")


@dataclass(frozen=True)
class MosaicTile:
    """One stop of a mosaic's stage grid: ``row`` grows along stage y and
    ``col`` along x."""

    row: int
    col: int
    x_um: float
    y_um: float


@dataclass(frozen=True, eq=False)
class Mosaic(Data):
    """Images taken across a stage grid: ``tiles`` is the whole plan in order,
    and ``images[i]`` was taken at ``tiles[i]``."""

    tiles: tuple[MosaicTile, ...]
    images: tuple[Image, ...]

    def __post_init__(self) -> None:
        if len(self.images) > len(self.tiles):
            raise ValueError(f"a mosaic of {len(self.tiles)} tiles got {len(self.images)} images")

    @property
    def nbytes(self) -> int:
        return sum(image.nbytes for image in self.images)

    def summary(self) -> str:
        return f"{len(self.images)}/{len(self.tiles)} tiles"


# Every kind a saved recording can name, so a file's kind string finds its class.
DATA_KINDS: dict[str, type[Data]] = {
    kind.name: kind for kind in (Image, DecayHistogram, RawSamples, Spectrum, Mosaic)
}
