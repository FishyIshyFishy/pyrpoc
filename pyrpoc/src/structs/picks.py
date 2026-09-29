"""What a display hands back when it is asked to point at something.

A pick is a dataset plus a location in it. Displays produce them; programs
consume them, through an ``ArmAndRun`` runner's apply function. Neither side
knows the other: a display knows how to turn a click into a ``PixelPick``, a
program knows how to turn a ``PixelPick`` into parameter values, and the app in
between only carries it.

A box or a line is a new subclass here, and a display opts in to producing it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Dataset


@dataclass(frozen=True)
class Pick:
    """A location in ``dataset``. Subclasses say what kind of location."""

    dataset: Dataset


@dataclass(frozen=True)
class PixelPick(Pick):
    """One pixel of a 2-D frame: ``x`` along the fast axis, ``y`` along the slow."""

    x: int
    y: int
