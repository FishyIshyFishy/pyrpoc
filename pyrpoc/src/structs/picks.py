"""What a runner can request: a location in some data.

A pick is a dataset plus a location in it. A runner requests one by kind and
receives one back; it never learns what provided it. A program knows how to
turn a ``PixelPick`` into parameter values, whatever produced it knows how to
make one, and the host in between only carries it.

A box or a line is a new subclass here.
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
