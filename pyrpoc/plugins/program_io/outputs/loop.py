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
