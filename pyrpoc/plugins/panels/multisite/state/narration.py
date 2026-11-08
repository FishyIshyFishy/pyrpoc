"""What the panel says in words: how long until a routine is due, what the loop is doing, and news
a person should see."""

from __future__ import annotations

from pyrpoc.plugins.program_io.outputs import UnitRefused

from .phase import phase_of
from .routines import RoutineStatus
from .run_order import Section
from .snapshot import Snapshot

# How long a refused change stays on the status line, in seconds.
NOTICE_S = 6.0


def duration(seconds: float) -> str:
    """``seconds`` as a person reads it: 8 s, 2 min 5 s, 1 h 4 min."""
    whole = max(0, round(seconds))
    if whole < 60:
        return f"{whole} s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes} min {secs} s" if secs else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def due_text(status: RoutineStatus, now_s: float) -> str:
    """When ``status`` is due, from ``now_s``."""
    late = now_s - status.due_s
    if late >= 0.5:
        return f"overdue {duration(late)}"
    if late >= 0:
        return "due now"
    return f"in {duration(-late)}"


def doing_text(snapshot: Snapshot) -> str:
    """What the loop is doing, in one line."""
    doing = snapshot.doing
    if doing is not None:
        took = duration(snapshot.now_s - doing.time_s)
        return f"{phase_of(doing).value} · unit {doing.unit} · {doing.routine} · {took}"
    if not snapshot.statuses:
        return "Finished · every routine has ended" if snapshot.view.units else "No units yet"
    coming = snapshot.coming(1)
    if not coming:
        return "Waiting · every unit is paused"
    status = coming[0].status
    when = due_text(status, snapshot.now_s)
    return f"Waiting · next unit {status.unit} · {status.routine} {when}"


def notice(snapshot: Snapshot) -> str | None:
    """The latest refused change, while it is still news."""
    for event in snapshot.view.recent:
        if snapshot.now_s - event.time_s > NOTICE_S:
            return None
        if isinstance(event, UnitRefused):
            return f"Not taken: {event.reason}"
    return None


def row_text(section: Section, status: RoutineStatus, snapshot: Snapshot) -> str:
    """The second line of a queue row: where the routine stands."""
    doing = snapshot.doing
    if section is Section.NOW and doing is not None:
        return f"{phase_of(doing).value.lower()} · {duration(snapshot.now_s - doing.time_s)}"
    if section is Section.PAUSED:
        return "paused"
    if section is Section.RUN_NOW:
        return "runs next"
    lost = " · lost at last scan" if status.lost else ""
    return due_text(status, snapshot.now_s) + lost
