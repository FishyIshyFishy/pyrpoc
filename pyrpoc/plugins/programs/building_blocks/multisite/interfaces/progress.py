"""``Progress``: what a stage reports to while it works, and asks whether to stop."""

from __future__ import annotations

from typing import Protocol


class Progress(Protocol):
    """Where a stage says what it is doing, and learns that it was stopped."""

    def status(self, text: str) -> None: ...

    def check_cancel(self) -> None:
        """Raise if the run was stopped."""
        ...
