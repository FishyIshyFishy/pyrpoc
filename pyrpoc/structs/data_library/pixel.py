"""A pixel of a dataset: what a click on a display names."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Dataset


@dataclass(frozen=True)
class Pixel:
    """One pixel of ``dataset``'s frames: ``x`` along the fast axis, ``y`` along the slow."""

    dataset: Dataset
    x: int
    y: int
