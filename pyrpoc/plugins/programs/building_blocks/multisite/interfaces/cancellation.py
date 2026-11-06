"""``Cancellation``: how a stage learns that its run was stopped."""

from __future__ import annotations

from typing import Protocol


class Cancellation(Protocol):
    def check_cancel(self) -> None:
        """Raise if the run was stopped."""
        ...
