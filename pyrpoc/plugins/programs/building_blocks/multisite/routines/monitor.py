"""The monitor routine: treat the unit once if it is in the treated group, then scan it every
period for the duration, recording its mean intensity each time."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import Unit

from ..interfaces.microscope import DigitalLine
from ..stages.loop.routine import Record, Routine, Scan, Wait, seen_from


@dataclass(frozen=True)
class Monitoring:
    """How often and for how long a unit is scanned; asked again before each wait, so a change
    takes effect at the unit's next scan."""

    period_s: float
    duration_min: float


@dataclass(frozen=True)
class Treatment:
    """Which group is treated at the start, on which line."""

    group: str
    line: DigitalLine


def monitor(unit: Unit, now: Callable[[], Monitoring], treatment: Treatment | None) -> Routine:
    """Monitor ``unit``; a ``treatment`` of None treats nobody."""
    if treatment is not None and unit.group == treatment.group:
        yield Scan((treatment.line,))
    started: float | None = None
    while True:
        seen = seen_from((yield Scan(())))
        started = seen.time_s if started is None else started
        yield Record("intensity", seen.means())
        settings = now()
        if seen.time_s - started >= settings.duration_min * 60.0:
            return
        yield Wait(settings.period_s)
