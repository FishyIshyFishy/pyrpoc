"""Moving a due routine up or down the queue."""

from __future__ import annotations

from collections.abc import Sequence

from .routines import RoutineKey
from .run_order import QueueRow, Section

# Only these sections run in the person's order; the rest run by time or not at all.
ORDERED = (Section.RUN_NOW, Section.DUE)


def movable(rows: Sequence[QueueRow], key: RoutineKey) -> bool:
    return any(row.key == key and row.section in ORDERED for row in rows)


def moved(
    rows: Sequence[QueueRow], order: Sequence[RoutineKey], key: RoutineKey, step: int
) -> tuple[RoutineKey, ...] | None:
    """The person's order with ``key`` moved ``step`` places within its section, or None when it
    cannot move that way."""
    (row,) = [row for row in rows if row.key == key]
    if row.section not in ORDERED:
        return None
    keys = [r.key for r in rows if r.section is row.section]
    at = keys.index(key)
    to = at + step
    if not 0 <= to < len(keys):
        return None
    keys.insert(to, keys.pop(at))
    # The whole section is ranked as now shown, so what the person sees is what runs; ranks of
    # routines elsewhere keep their order after it.
    rest = [each for each in order if each not in keys]
    return (*keys, *rest)
