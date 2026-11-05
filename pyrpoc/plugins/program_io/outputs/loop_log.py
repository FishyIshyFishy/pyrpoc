"""``LoopLogOutput``: everything a loop did, in order, and ``LoopView``, what that says the loop
is doing now."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyrpoc.core.program_output import ProgramOutput, program_output_registry
from pyrpoc.core.recording import Recording

# How many of the latest events a view carries.
RECENT = 12


@dataclass(frozen=True)
class LoopEvent:
    """One thing a loop did: when on the instrument's clock, what (``start``, ``added``,
    ``due``, ``scan``, ``record``, ``focus``, ``lost``, ``done``, ``error``, ...), to which unit
    and in which routine, and the details."""

    time_s: float
    event: str
    unit: int | None
    routine: str
    details: dict[str, Any]


@dataclass(frozen=True)
class UnitView:
    """One unit as the log tells it: its group, outline as last tracked, whether it is
    waiting, paused, lost or done, when it is next due, and each value recorded for it as
    (seconds since the loop started, one value per channel)."""

    id: int
    group: str
    outline: tuple[tuple[float, float], ...]
    state: str
    next_due_s: float | None
    series: tuple[tuple[float, tuple[float, ...]], ...]


@dataclass(frozen=True)
class QueueEntry:
    """A routine step that is due: whose, in which routine, and when, in seconds since the loop
    started."""

    unit: int
    routine: str
    due_s: float


@dataclass(frozen=True)
class LoopView:
    """What a loop is doing as of its latest event: how long it has run, its channels, every
    unit, the queue soonest first, and its latest events, newest first."""

    elapsed_s: float
    channels: tuple[str, ...]
    units: tuple[UnitView, ...]
    queue: tuple[QueueEntry, ...]
    recent: tuple[str, ...]


@program_output_registry.register("loop_log")
class LoopLogOutput(ProgramOutput):
    """A loop's events, each appended to ``<root>_<name>.jsonl`` as it happens, so a loop cut
    short still says what it did."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.events: list[LoopEvent] = []
        self.nbytes_logged = 0

    def log(self, event: LoopEvent) -> None:
        line = json.dumps(asdict(event))
        with self.lock:
            self.events.append(event)
            self.nbytes_logged += len(line)
            if self.root is not None:
                path = self.root.with_name(f"{self.root.name}_{self.name}.jsonl")
                with path.open("a", encoding="utf-8") as log:
                    log.write(line + "\n")
                self.paths = {self.name: path}
        self.changed()

    def view(self) -> LoopView:
        with self.lock:
            events = list(self.events)
        fold = LoopFold()
        for event in events:
            fold.take(event)
        return fold.view()

    @property
    def nbytes(self) -> int:
        return self.nbytes_logged

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> LoopLogOutput:
        output = cls(name=name, recording=recording, root=None)
        if files:
            (path,) = files.values()
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    raw = json.loads(line)
                    output.events.append(LoopEvent(**raw))
                    output.nbytes_logged += len(line)
        return output


@dataclass
class UnitFold:
    """One unit while folding: as added, where it was last tracked, and its threads."""

    group: str
    outline: tuple[tuple[float, float], ...]
    centre: tuple[float, float]
    tracked: tuple[float, float]
    lost: bool = False
    paused: bool = False
    # Each open routine's next due time, and the routines asked to run now.
    due: dict[str, float] = field(default_factory=dict)
    urgent: set[str] = field(default_factory=set)
    series: list[tuple[float, tuple[float, ...]]] = field(default_factory=list)


class LoopFold:
    """The log's events taken in order into what the loop is doing."""

    def __init__(self) -> None:
        self.started_s = 0.0
        self.latest_s = 0.0
        self.channels: tuple[str, ...] = ()
        self.units: dict[int, UnitFold] = {}
        self.recent: list[str] = []

    def take(self, event: LoopEvent) -> None:
        self.latest_s = event.time_s
        if event.event == "start":
            self.started_s = event.time_s
            self.channels = tuple(event.details["channels"])
        elif event.event == "added" and event.unit is not None:
            self.add_unit(event.unit, event.details)
        elif event.unit is not None and event.unit in self.units:
            self.take_unit_event(self.units[event.unit], event)
        who = "" if event.unit is None else f" unit {event.unit}"
        minutes = (event.time_s - self.started_s) / 60.0
        self.recent.append(f"{minutes:5.1f} min  {event.event}{who}")

    def add_unit(self, unit: int, details: dict[str, Any]) -> None:
        outline = tuple((float(x), float(y)) for x, y in details["outline"])
        centre = (float(details["x_um"]), float(details["y_um"]))
        self.units[unit] = UnitFold(str(details["group"]), outline, centre, centre)

    def take_unit_event(self, unit: UnitFold, event: LoopEvent) -> None:
        kind, details = event.event, event.details
        if kind == "due":
            unit.due[event.routine] = float(details["due_s"])
        elif kind == "scan":
            unit.tracked = (float(details["x_um"]), float(details["y_um"]))
            unit.lost = not details["found"]
            unit.urgent.discard(event.routine)
        elif kind == "record":
            values = tuple(float(v) for v in details["values"])
            unit.series.append((event.time_s - self.started_s, values))
        elif kind in ("done", "error"):
            unit.due.pop(event.routine, None)
        elif kind == "removed":
            unit.due.clear()
        elif kind in ("paused", "resumed"):
            unit.paused = kind == "paused"
        elif kind == "run now":
            unit.urgent |= set(unit.due)

    def view(self) -> LoopView:
        return LoopView(
            self.latest_s - self.started_s,
            self.channels,
            tuple(self.unit_view(uid, unit) for uid, unit in self.units.items()),
            self.queue(),
            tuple(reversed(self.recent[-RECENT:])),
        )

    def unit_view(self, uid: int, unit: UnitFold) -> UnitView:
        if not unit.due:
            state = "done"
        elif unit.paused:
            state = "paused"
        elif unit.lost:
            state = "lost"
        else:
            state = "waiting"
        dx, dy = unit.tracked[0] - unit.centre[0], unit.tracked[1] - unit.centre[1]
        outline = tuple((x + dx, y + dy) for x, y in unit.outline)
        due = min(unit.due.values(), default=None)
        next_due = None if due is None else due - self.started_s
        return UnitView(uid, unit.group, outline, state, next_due, tuple(unit.series))

    def queue(self) -> tuple[QueueEntry, ...]:
        """The open, unpaused routines: those asked to run now first, then the earliest due."""
        entries = [
            (routine not in unit.urgent, due, QueueEntry(uid, routine, due - self.started_s))
            for uid, unit in self.units.items()
            if not unit.paused
            for routine, due in unit.due.items()
        ]
        return tuple(entry for _urgent, _due, entry in sorted(entries, key=lambda e: e[:2]))
