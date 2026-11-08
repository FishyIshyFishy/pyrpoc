"""What the panel says in words: when a routine runs, what a unit does next, and what the loop
is doing."""

from __future__ import annotations

from pyrpoc.plugins.program_io.outputs import FocusingAtUnit, LoopActivity, MovingToUnit

from .pending import Change
from .standing import QueueRow, Snapshot, Standing, UnitRow

# What each standing means, for a tooltip.
MEANING = {
    Standing.DRAWN: "Drawn on the dish; the loop has not taken it yet",
    Standing.REFUSED: "Drawn on the dish, but the loop would not take it",
    Standing.RUNNING: "The loop is working on this unit now",
    Standing.WAITING: "Waiting for its next routine to fall due",
    Standing.LOST: "Not found where it was at its last scan; the loop keeps looking for it",
    Standing.PAUSED: "Paused: none of its routines run until it is resumed",
    Standing.DONE: "Every one of its routines has ended",
}


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


def ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


def phase_of(activity: LoopActivity) -> tuple[str, str]:
    """The phase in a word, and as the words before a unit: ``("moving", "Moving to")``."""
    match activity:
        case MovingToUnit():
            return "moving", "Moving to"
        case FocusingAtUnit():
            return "focusing", "Focusing on"
        case _:
            return "scanning", "Scanning"


def due_text(due_s: float, now_s: float) -> str:
    late = now_s - due_s
    if late >= 1:
        return f"overdue {duration(late)}"
    return "due now" if late > -1 else f"in {duration(-late)}"


def when_text(row: QueueRow, snapshot: Snapshot) -> str:
    """When a queued routine runs: now, at once when asked to, or when it falls due."""
    if not snapshot.running:
        # A loop that has stopped keeps no time, so nothing is due in so many seconds.
        return "—"
    doing = snapshot.doing
    if row.place == 0 and doing is not None:
        return f"{phase_of(doing)[0]}, {duration(snapshot.now_s - row.time_s)}"
    if row.urgent:
        return "asked to run now"
    return due_text(row.time_s, snapshot.now_s)


def standing_text(row: UnitRow, pending: Change | None) -> str:
    return row.standing.value if pending is None else pending.value


def next_text(row: UnitRow, snapshot: Snapshot) -> str:
    """What a unit does next, and where that stands in the queue."""
    following = row.next
    if following is not None and following.place == 0:
        return f"{following.routine}, now"
    if following is not None:
        text = f"{following.routine}, {ordinal(following.place)} in the queue"
        return f"{text}, {when_text(following, snapshot)}" if snapshot.running else text
    match row.standing:
        case Standing.PAUSED:
            return "nothing until resumed"
        case Standing.DONE:
            return "nothing: its routines have ended"
        case Standing.REFUSED:
            return snapshot.refused[row.key]
        case Standing.DRAWN if snapshot.running:
            return "joins the loop after its current step"
        case Standing.DRAWN:
            return "starts with the next loop run on this survey"
        case _:
            return ""


def doing_text(snapshot: Snapshot, queue: tuple[QueueRow, ...]) -> str:
    """What the loop is doing, and what it does next, in one line."""
    if snapshot.view is None:
        return "No loop has run over this survey yet. Draw units on the dish, then start a loop."
    waiting = [row for row in queue if row.place > 0]
    if not snapshot.running:
        if waiting:
            return "Stopped with routines left. Run the loop again in this folder to carry on."
        return "Finished: every routine has ended."
    doing = snapshot.doing
    now = "Waiting"
    if doing is not None:
        took = duration(snapshot.now_s - doing.time_s)
        now = f"{phase_of(doing)[1]} unit {doing.unit} for {doing.routine} ({took})"
    if not waiting:
        rest = "Nothing else is queued" if doing is not None else "every unit is paused"
        return f"{now}: {rest}."
    first = waiting[0]
    return f"{now}. Next: unit {first.unit} for {first.routine}, {when_text(first, snapshot)}."
