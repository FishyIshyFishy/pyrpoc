"""``Steering``: what the user asked of a loop since it last looked, read between its steps."""

from __future__ import annotations

from typing import Protocol

from pyrpoc.plugins.program_io.outputs import Unit


class Steering(Protocol):
    """The units to run now, those paused, and those asked to run at once."""

    def units(self) -> tuple[Unit, ...]: ...

    def paused(self) -> frozenset[int]: ...

    def run_now(self) -> list[int]:
        """The units asked to run at once since the loop last looked."""
        ...
