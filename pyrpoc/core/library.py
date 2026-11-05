"""The ``Library`` protocol: the open recordings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .recording import Recording


class Library(Protocol):
    """The open recordings: add one, close one, list them, and subscribe to changes."""

    def add(self, recording: Recording) -> None: ...

    def close(self, recording: Recording) -> None:
        """Drop it from memory; files already saved stay on disk."""
        ...

    def recordings(self) -> list[Recording]:
        """Every open recording, oldest first."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...
