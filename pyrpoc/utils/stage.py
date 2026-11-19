"""``XYStage``: a device that can be sent to an xy position, which is all that is asked of one
when it is moved outside a run."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


# Runtime-checkable so the inventory can tell which of its devices is one, without naming any.
@runtime_checkable
class XYStage(Protocol):
    def move_xy_to(self, x: float, y: float) -> None:
        """Start moving to (``x``, ``y``) µm, returning without waiting for the stage to stop."""
        ...
