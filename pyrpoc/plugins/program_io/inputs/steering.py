"""``PAUSED_UNITS``: the units a running loop leaves alone until they are resumed."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pyrpoc.core.program_input import Field, ParameterError, declare


@dataclass(frozen=True)
class UnitIdsField(Field):
    """Unit ids, in order."""

    def coerce(self, value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of whole numbers")
        return tuple(int(each) for each in value)

    def encode(self, value: tuple[int, ...]) -> Any:
        return list(value)


PAUSED_UNITS = declare(
    "loop_paused_units",
    UnitIdsField("Paused Units", "Units the loop leaves alone until resumed"),
    (),
    # A pause belongs to the loop it was made in.
    restored=False,
)
