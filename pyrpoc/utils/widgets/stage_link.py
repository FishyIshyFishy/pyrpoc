"""``StageLink``: what a data panel may do to the stage outside any run, given to every data
panel by whoever holds the devices."""

from __future__ import annotations

from typing import Protocol


class StageLink(Protocol):
    def blocked(self) -> str | None:
        """Why the stage cannot be moved now, or None."""
        ...

    def move_to(self, x_um: float, y_um: float) -> str | None:
        """Start moving the stage to (``x_um``, ``y_um``), or say why it was not moved."""
        ...
