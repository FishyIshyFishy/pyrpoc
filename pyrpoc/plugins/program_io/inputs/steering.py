"""Steering a running loop from the Autonomy panel: which units are paused, and which were
asked to run now."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pyrpoc.core.program_input import Field, ParameterError, declare


def int_tuple(label: str, value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
        raise ParameterError(f"{label}: expected a list of whole numbers")
    return tuple(int(each) for each in value)


@dataclass(frozen=True)
class UnitIdsField(Field):
    """Unit ids, in order."""

    def coerce(self, value: Any) -> tuple[int, ...]:
        return int_tuple(self.label, value)

    def encode(self, value: tuple[int, ...]) -> Any:
        return list(value)


@dataclass(frozen=True)
class RunNow:
    """One request to run a unit's routines at once; ``serial`` grows with each request, so a
    loop takes each only once."""

    serial: int
    unit: int


@dataclass(frozen=True)
class RunNowField(Field):
    """The requests made since launch, oldest first."""

    def coerce(self, value: Any) -> tuple[RunNow, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of requests")
        return tuple(
            each if isinstance(each, RunNow) else RunNow(*int_tuple(self.label, each))
            for each in value
        )

    def encode(self, value: tuple[RunNow, ...]) -> Any:
        return [[each.serial, each.unit] for each in value]


# Neither is kept: a pause or a request belongs to the loop it was made in.
PAUSED_UNITS = declare(
    "loop_paused_units",
    UnitIdsField("Paused Units", "Units the loop leaves alone until resumed", shown=False),
    (),
    persisted=False,
)
RUN_NOW = declare(
    "loop_run_now",
    RunNowField("Run Now", "Units asked to run at once", shown=False),
    (),
    persisted=False,
)
