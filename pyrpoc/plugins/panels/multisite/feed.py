"""``LoopFeed``: everything the panel reads from a loop and every change it sends one. A fake loop
feeds the demo; a feed built on a run's outputs will feed the real panel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pyrpoc.plugins.program_io.outputs import FrameTag, Placement, Unit

from .state.routines import RoutineKey
from .state.snapshot import Snapshot


@dataclass(frozen=True)
class Backdrop:
    """The survey's ``(C, H, W)`` overview, where it lies on the dish, and its channels."""

    planes: np.ndarray
    placement: Placement
    channels: tuple[str, ...]


@dataclass(frozen=True)
class UnitFrame:
    """A unit's latest ``(C, H, W)`` frame, where it lies, its pixel size in µm, and a count that
    rises with every new frame of the unit."""

    planes: np.ndarray
    tag: FrameTag
    um_per_px: float
    serial: int


class Steer(Protocol):
    """The changes a person can ask of a running loop."""

    def pause(self, unit: int, paused: bool) -> None: ...

    def run_now(self, unit: int) -> None: ...

    def remove(self, unit: int) -> None: ...

    def reorder(self, order: tuple[RoutineKey, ...]) -> None:
        """Run due routines in ``order``, those it leaves out after them."""
        ...

    def add_unit(self, unit: Unit) -> None:
        """Join ``unit``, whose key the panel gave it, so it can tell the answer when it comes."""
        ...


class LoopFeed(Steer, Protocol):
    """A loop as the panel sees it."""

    def snapshot(self) -> Snapshot: ...

    def backdrop(self) -> Backdrop:
        """The same object until the survey changes."""
        ...

    def frames(self) -> Mapping[int, UnitFrame]:
        """Each unit's latest frame, by the unit's number."""
        ...

    def field_um(self) -> float:
        """The side of one scan field, in µm."""
        ...
