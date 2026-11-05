"""``Steering``: the changes to its units a loop is asked for, read between its steps."""

from __future__ import annotations

from typing import Protocol

from pyrpoc.plugins.program_io.requests import UnitCommand


class Steering(Protocol):
    """Where a loop's unit commands come from, whoever gives them: a person or software."""

    def commands(self) -> list[UnitCommand]:
        """The commands given since the loop last looked, oldest first."""
        ...
