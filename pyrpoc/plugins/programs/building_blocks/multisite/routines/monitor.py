"""The monitor routine: with a treatment, one first scan of every unit, the line driven only for
the treated group's; then a scan every period for the duration, recording the unit's mean
intensity each time it is found."""

# The untreated units' first scan is a sham: the same scan with the line left low. Each group
# then takes the same imaging light before monitoring, so the groups differ only in the treatment.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import Unit

from ..interfaces.microscope import DigitalLine
from ..stages.loop.routine import Record, Routine, RoutineHistory, Scan, Wait, seen_from


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


def monitor(
    unit: Unit,
    history: RoutineHistory,
    now: Callable[[], Monitoring],
    treatment: Treatment | None,
) -> Routine:
    """Monitor ``unit``, carrying on from ``history``; a ``treatment`` of None treats nobody."""
    # Any scan in the history means the first scan, treatment or sham, was already given.
    first = 0 if treatment is None else 1
    if treatment is not None and not history.scans_s:
        yield Scan((treatment.line,) if unit.group == treatment.group else ())
    started = history.scans_s[first] if len(history.scans_s) > first else None
    while True:
        seen = seen_from((yield Scan(())))
        started = seen.time_s if started is None else started
        # A scan that did not find the unit imaged where it was, which is not a measure of it.
        if seen.found:
            yield Record("intensity", seen.means())
        settings = now()
        if seen.time_s - started >= settings.duration_min * 60.0:
            return
        yield Wait(settings.period_s)
