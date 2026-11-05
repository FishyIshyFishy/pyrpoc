"""What a loop publishes as it runs: each event for its log, each unit's frames, and the units
as they were when it started."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Self

from pyrpoc.core.program_output import Data, data_kind_registry

from .arrays import Image
from .records import Record


@data_kind_registry.register("LoopEvent")
@dataclass(frozen=True, eq=False)
class LoopEvent(Data):
    """One thing a loop did, in the order it did it: when on the instrument's clock, what
    (``scan``, ``record``, ``focus``, ``lost``, ``edit``, ``error``, ...), to which unit and in
    which routine, and the details."""

    time_s: float
    event: str
    unit: int | None
    routine: str
    details: dict[str, Any]

    @property
    def nbytes(self) -> int:
        return len(self.to_json_line())

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))

    def summary(self) -> str:
        unit = "" if self.unit is None else f" unit {self.unit}"
        return f"{self.event}{unit} at {self.time_s:.1f} s"


@data_kind_registry.register("UnitFrame")
@dataclass(frozen=True, eq=False)
class UnitFrame(Data):
    """One frame a routine took of its unit, centred on where the unit was tracked to, the
    dish position of its top-left pixel, and its place in that unit and routine's stack."""

    image: Image
    unit: int
    routine: str
    index: int
    left_um: float
    top_um: float

    @property
    def nbytes(self) -> int:
        return self.image.nbytes

    def summary(self) -> str:
        return f"unit {self.unit} {self.routine} frame {self.index + 1}"


@dataclass(frozen=True)
class UnitSnapshot:
    """A unit as it was: its id, group and outline in dish µm."""

    id: int
    group: str
    outline: tuple[tuple[float, float], ...]


@data_kind_registry.register("LoopUnits")
@dataclass(frozen=True, eq=False)
class LoopUnits(Record):
    """The units a loop started with."""

    units: tuple[UnitSnapshot, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        return cls(
            tuple(
                UnitSnapshot(
                    int(unit["id"]),
                    str(unit["group"]),
                    tuple((float(x), float(y)) for x, y in unit["outline"]),
                )
                for unit in raw["units"]
            )
        )

    def summary(self) -> str:
        return f"{len(self.units)} units"


@dataclass(frozen=True)
class UnitState:
    """One unit as a loop sees it now: its group, outline as tracked, whether it is waiting,
    paused, lost or done, when it is next due, and each intensity recorded for it as
    (seconds since the loop started, one value per channel)."""

    id: int
    group: str
    outline: tuple[tuple[float, float], ...]
    state: str
    next_due_s: float | None
    series: tuple[tuple[float, tuple[float, ...]], ...]


@dataclass(frozen=True)
class QueueEntry:
    """A step that is due: whose, in which routine, and when, in seconds since the loop
    started."""

    unit: int
    routine: str
    due_s: float


@data_kind_registry.register("LoopState")
@dataclass(frozen=True, eq=False)
class LoopState(Record):
    """What a loop is doing, republished after every step: how long it has run, its channels,
    every unit, the queue soonest first, and its latest events, newest first."""

    elapsed_s: float
    channels: tuple[str, ...]
    units: tuple[UnitState, ...]
    queue: tuple[QueueEntry, ...]
    recent: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        units = tuple(
            UnitState(
                int(u["id"]),
                str(u["group"]),
                tuple((float(x), float(y)) for x, y in u["outline"]),
                str(u["state"]),
                None if u["next_due_s"] is None else float(u["next_due_s"]),
                tuple((float(t), tuple(float(v) for v in values)) for t, values in u["series"]),
            )
            for u in raw["units"]
        )
        queue = tuple(
            QueueEntry(int(q["unit"]), str(q["routine"]), float(q["due_s"])) for q in raw["queue"]
        )
        return cls(
            float(raw["elapsed_s"]),
            tuple(str(c) for c in raw["channels"]),
            units,
            queue,
            tuple(str(r) for r in raw["recent"]),
        )

    def summary(self) -> str:
        return f"{len(self.units)} units, {self.elapsed_s / 60.0:.1f} min"
