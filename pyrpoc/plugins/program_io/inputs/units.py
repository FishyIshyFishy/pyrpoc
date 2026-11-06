"""``UNITS``, the regions of the dish a loop starts on, as a program input."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pyrpoc.core.program_input import Field, ParameterError, ProgramInput, declare
from pyrpoc.plugins.program_io.outputs import Unit


@dataclass(frozen=True)
class UnitsField(Field):
    """The rules for the units: a tuple of ``Unit`` with distinct keys."""

    def coerce(self, value: Any) -> tuple[Unit, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of units")
        units = tuple(each if isinstance(each, Unit) else Unit.from_dict(each) for each in value)
        if len({unit.key for unit in units}) != len(units):
            raise ParameterError(f"{self.label}: two units share a key")
        return units

    def encode(self, value: tuple[Unit, ...]) -> Any:
        return [unit.to_dict() for unit in value]


UNITS: ProgramInput[tuple[Unit, ...]] = declare(
    "units",
    UnitsField("Units", "The regions drawn over the survey that a loop starts on"),
    (),
    # Thousands of outlines would be rewritten on every autosave. Once a loop has started, its
    # units output is the record of them.
    restored=False,
)
