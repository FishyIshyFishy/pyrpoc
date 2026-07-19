"""Parcels: the units of acquired data that flow out of a run.

Distinct types, not one envelope. A display accepts parcel *types*; many modalities
may emit the same type. Types are structural (image, histogram-cube), never
per-modality identity.
"""

from __future__ import annotations

import numpy as np
from attrs import define


@define
class Parcel:
    """Base class for one unit of acquired data."""


@define
class ImageParcel(Parcel):
    """A multi-channel image, shape (channels, height, width)."""

    data: np.ndarray
    channel_labels: list[str]

    @property
    def channel_count(self) -> int:
        return self.data.shape[0]


@define
class ImageFrameParcel(ImageParcel):
    """A complete image frame — the image parcel that gets written to storage."""


@define
class PartialImageParcel(ImageParcel):
    """An in-progress image frame streamed during a scan; never written to storage."""


@define
class HistogramCubeParcel(Parcel):
    """Per-pixel decay histograms, shape (height, width, bins), for lifetime imaging."""

    data: np.ndarray
    bin_width_ps: float
    laser_period_ps: float
