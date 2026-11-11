"""What a routine is: a generator, made for one unit, that yields the steps it wants done and is
given back what each scan saw."""

# A routine sees only its own unit and its own images; tracking, focus and the queue are the
# orchestrator's.

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.outputs import Unit

from ...interfaces.microscope import DigitalLine


@dataclass(frozen=True)
class Scan:
    """Image the unit where it is now, with ``lines`` driven high over it."""

    lines: tuple[DigitalLine, ...]


@dataclass(frozen=True)
class Wait:
    """Be due again ``seconds`` from now."""

    seconds: float


@dataclass(frozen=True)
class Record:
    """Log a measurement, one value per channel."""

    name: str
    values: tuple[float, ...]


Step = Scan | Wait | Record


@dataclass(frozen=True)
class Seen:
    """What a scan gave its routine: the ``(C, H, W)`` image centred on the unit, the unit's
    outline as a mask of it, when it was taken, and whether the unit was found in it; if not,
    the image is of where the unit was last seen."""

    image: np.ndarray
    mask: np.ndarray
    time_s: float
    found: bool

    def means(self) -> tuple[float, ...]:
        """Each channel's mean inside the unit."""
        return tuple(float(plane[self.mask].mean()) for plane in self.image)


# Sent ``Seen`` after a scan, and None after any other step.
Routine = Generator[Step, Seen | None, None]


@dataclass(frozen=True)
class Recorded:
    """A measurement a routine recorded earlier: its name, when, and its values."""

    name: str
    time_s: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class RoutineHistory:
    """What a routine already did for its unit before a loop was cut short and resumed: when
    each scan ran, and what it recorded. Empty for a routine starting afresh."""

    scans_s: tuple[float, ...]
    records: tuple[Recorded, ...]


NO_HISTORY = RoutineHistory((), ())


@dataclass(frozen=True)
class RoutineSpec:
    """A named routine, and how to start it for one unit, carrying on from its history."""

    name: str
    start: Callable[[Unit, RoutineHistory], Routine]


def seen_from(result: Seen | None) -> Seen:
    """What a scan sent back, which a routine yielding ``Scan`` is always given."""
    if result is None:
        raise RuntimeError("a scan gave its routine nothing back")
    return result
