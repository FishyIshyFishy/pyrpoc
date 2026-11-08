"""The loop at a glance: a few figures, each with a line saying what is behind it."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.plugins.program_io.outputs import UnitState

from .history import Tone, last_focus, scans
from .narration import duration
from .phase import phase_of
from .run_order import Section
from .snapshot import Snapshot

# A due routine waiting longer than this is worth a warning: the loop is falling behind.
BEHIND_S = 60.0


@dataclass(frozen=True)
class Stat:
    """One figure: what it is, its value, a line of detail, and a tone when it needs notice."""

    label: str
    value: str
    detail: str
    tone: Tone | None


def summary(snapshot: Snapshot) -> tuple[Stat, ...]:
    return (
        now(snapshot),
        units(snapshot),
        routines(snapshot),
        behind(snapshot),
        scanned(snapshot),
        focus(snapshot),
    )


def now(snapshot: Snapshot) -> Stat:
    doing, elapsed = snapshot.doing, duration(snapshot.now_s - snapshot.view.started_s)
    if doing is not None:
        took = duration(snapshot.now_s - doing.time_s)
        return Stat(
            "Now", phase_of(doing).value, f"unit {doing.unit} · {doing.routine} · {took}", None
        )
    if not snapshot.statuses:
        return Stat("Now", "Finished", f"ran {elapsed}", Tone.GOOD)
    return Stat("Now", "Waiting", f"running {elapsed}", None)


def units(snapshot: Snapshot) -> Stat:
    states = [unit.state for unit in snapshot.view.units]
    present = [state for state in states if state is not UnitState.REMOVED]
    parts = [
        f"{states.count(state)} {state.value}"
        for state in (UnitState.PAUSED, UnitState.LOST, UnitState.DONE)
        if states.count(state)
    ]
    lost = UnitState.LOST in states
    return Stat(
        "Units",
        str(len(present)),
        " · ".join(parts) or "all running",
        Tone.WARNING if lost else None,
    )


def routines(snapshot: Snapshot) -> Stat:
    rows = snapshot.rows
    due = sum(1 for row in rows if row.section in (Section.DUE, Section.RUN_NOW))
    upcoming = sum(1 for row in rows if row.section is Section.UPCOMING)
    return Stat("Routines", str(len(rows)), f"{due} due · {upcoming} upcoming", None)


def behind(snapshot: Snapshot) -> Stat:
    """The longest any due routine has waited past its time."""
    due = [row.status for row in snapshot.rows if row.section is Section.DUE]
    if not due:
        return Stat("Longest wait", "none", "nothing overdue", None)
    worst = min(due, key=lambda status: status.due_s)
    late = snapshot.now_s - worst.due_s
    tone = Tone.WARNING if late > BEHIND_S else None
    return Stat("Longest wait", duration(late), f"unit {worst.unit} · {worst.routine}", tone)


def scanned(snapshot: Snapshot) -> Stat:
    taken, lost = scans(snapshot.events)
    detail = f"{lost} lost their unit" if lost else "every unit found"
    return Stat("Scans", str(taken), detail, Tone.WARNING if lost else None)


def focus(snapshot: Snapshot) -> Stat:
    checked = last_focus(snapshot.events)
    if checked is None:
        return Stat("Focus", "—", "not checked yet", None)
    ago = duration(snapshot.now_s - checked.time_s)
    return Stat("Focus", f"{checked.offset_um:+.1f} µm", f"off the survey · {ago} ago", None)
