"""``UNITS``, the regions of the dish a loop experiments on, as a program input."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pyrpoc.core.program_input import Field, ParameterError, declare
from pyrpoc.plugins.program_io.outputs import Unit


@dataclass(frozen=True)
class UnitsField(Field):
    """The rules for the units: a tuple of ``Unit`` with distinct ids."""

    def coerce(self, value: Any) -> tuple[Unit, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of units")
        units = tuple(each if isinstance(each, Unit) else Unit.from_dict(each) for each in value)
        if len({unit.id for unit in units}) != len(units):
            raise ParameterError(f"{self.label}: two units share an id")
        return units

    def encode(self, value: tuple[Unit, ...]) -> Any:
        return [unit.to_dict() for unit in value]


UNITS = declare(
    "units",
    UnitsField("Units", "The regions drawn in the Autonomy panel"),
    (),
    # They belong to the dish, not the session: the Autonomy panel keeps them in units.json in
    # the survey's folder, and the session would rewrite thousands of them on every autosave.
    restored=False,
)
