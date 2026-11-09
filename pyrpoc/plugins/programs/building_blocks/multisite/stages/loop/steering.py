"""``Steering``: the units a loop is asked to run on, read between its steps."""

from __future__ import annotations

from typing import Protocol

from pyrpoc.plugins.program_io.outputs import Unit


class Steering(Protocol):
    """Where a loop learns which units are wanted now, whoever wants them: a person or
    software."""

    def units(self) -> tuple[Unit, ...]:
        """Every unit wanted now, each as it should be run."""
        ...
