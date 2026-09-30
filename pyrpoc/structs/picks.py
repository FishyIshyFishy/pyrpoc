"""What a runner can request: a location in some data.

A runner asks for a pick by kind and never learns what provided it; a new
kind of location (a box, a line) is a new subclass here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Dataset


@dataclass(frozen=True)
class Pick:
    """A location in ``dataset``. Subclasses say what kind of location."""

    dataset: Dataset


@dataclass(frozen=True)
class PixelPick(Pick):
    """One pixel of a 2-D frame: ``x`` along the fast axis, ``y`` along the slow."""

    x: int
    y: int
