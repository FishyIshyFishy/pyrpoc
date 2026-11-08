"""The order routines run in: run-now first, then those due, in the person's order where they gave
one, then the rest by due time. Nothing runs before it is due unless asked to run now."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from .routines import RoutineKey, RoutineStatus


class Section(Enum):
    """Where a routine sits in the queue, in the order the sections are shown."""

    NOW = "Now"
    RUN_NOW = "Run now"
    DUE = "Due · your order first"
    UPCOMING = "Upcoming"
    PAUSED = "Paused"


SECTIONS = tuple(Section)


@dataclass(frozen=True)
class QueueRow:
    status: RoutineStatus
    section: Section

    @property
    def key(self) -> RoutineKey:
        return self.status.key


def by_priority(
    statuses: Iterable[RoutineStatus], order: Sequence[RoutineKey]
) -> list[RoutineStatus]:
    """Those the person ordered, in their order, then the rest, the earliest due first."""
    rank = {key: index for index, key in enumerate(order)}

    def priority(status: RoutineStatus) -> tuple[int, float]:
        if status.key in rank:
            return (0, float(rank[status.key]))
        return (1, status.due_s)

    return sorted(statuses, key=priority)


def section_of(status: RoutineStatus, now_s: float) -> Section:
    if status.paused:
        return Section.PAUSED
    if status.urgent:
        return Section.RUN_NOW
    return Section.DUE if status.due_s <= now_s else Section.UPCOMING


def queue_rows(
    statuses: Sequence[RoutineStatus],
    running: RoutineKey | None,
    now_s: float,
    order: Sequence[RoutineKey],
) -> tuple[QueueRow, ...]:
    """Every open routine, section by section, each section in the order it will run."""
    rows = [QueueRow(s, Section.NOW) for s in statuses if s.key == running]
    waiting = [s for s in statuses if s.key != running]
    for section in SECTIONS[1:]:
        members = [s for s in waiting if section_of(s, now_s) is section]
        if section is Section.UPCOMING:
            members.sort(key=lambda s: s.due_s)
        elif section is Section.PAUSED:
            members.sort(key=lambda s: s.key)
        else:
            members = by_priority(members, order)
        rows += [QueueRow(s, section) for s in members]
    return tuple(rows)


def next_to_run(
    statuses: Sequence[RoutineStatus], now_s: float, order: Sequence[RoutineKey]
) -> RoutineStatus | None:
    """The routine to run next, or None while nothing is due."""
    for section in (Section.RUN_NOW, Section.DUE):
        members = [s for s in statuses if section_of(s, now_s) is section]
        if members:
            return by_priority(members, order)[0]
    return None
