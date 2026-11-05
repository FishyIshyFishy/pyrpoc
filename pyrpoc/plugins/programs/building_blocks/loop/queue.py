"""The queue: every (unit, routine) pair as a thread with its next due time, the earliest
first."""

from __future__ import annotations

from dataclasses import dataclass

from .steps import Routine, Seen


@dataclass
class Thread:
    """One routine running for one unit: when it is next due, what to send it next, and
    whether it is finished or paused."""

    unit: int
    routine: str
    steps: Routine
    due_s: float
    sending: Seen | None = None
    started: bool = False
    finished: bool = False
    paused: bool = False
    # Asked to run at once: it goes ahead of everything due, until its next scan.
    urgent: bool = False

    @property
    def key(self) -> str:
        return f"{self.unit}:{self.routine}"


class Queue:
    """The threads, and which runs next."""

    def __init__(self) -> None:
        self.threads: list[Thread] = []

    def add(self, thread: Thread) -> None:
        self.threads.append(thread)

    def active(self) -> list[Thread]:
        return [thread for thread in self.threads if not thread.finished]

    def runnable(self) -> list[Thread]:
        """The threads that can run: those asked to run now, then the earliest due."""
        ready = [thread for thread in self.active() if not thread.paused]
        return sorted(ready, key=lambda thread: (not thread.urgent, thread.due_s))

    def of_unit(self, unit: int) -> list[Thread]:
        return [thread for thread in self.active() if thread.unit == unit]
