"""``LibraryStore``: the program outputs currently open, acquired now or loaded
from disk. Not persisted."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

from pyrpoc.core.program_output import Data, ProgramOutput

from .grouping import group_label

D = TypeVar("D", bound=Data)

# Everything open together, in bytes. Big enough for hours of typical imaging,
# small enough to leave a lab PC's RAM for the rest of the session.
LIBRARY_LIMIT_BYTES = 1 << 30


class LibraryFull(Exception):
    """Refused: more data would go past the limit, and auto-purge is off."""


def gib(nbytes: int) -> str:
    return f"{nbytes / (1 << 30):.2f} GiB"


class LibraryStore:
    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        # Close the oldest finished entries whenever the total is over the limit.
        self.auto_purge = False
        self._outputs: list[ProgramOutput[Any]] = []
        self._subscribers: list[Callable[[], None]] = []
        self._lock = threading.RLock()

    def add(self, program_output: ProgramOutput[D]) -> ProgramOutput[D]:
        with self._lock:
            self._outputs.append(program_output)
        self.notify()
        return program_output

    def remove(self, program_output: ProgramOutput[Any]) -> None:
        with self._lock:
            self._outputs.remove(program_output)
        self.notify()

    def all(self) -> list[ProgramOutput[Any]]:
        with self._lock:
            return list(self._outputs)

    def by_id(self, output_id: str, kind: type[D]) -> ProgramOutput[D] | None:
        """The open program output with this id, if it holds ``kind``."""
        # Ids come from saved sessions, so one may name a closed entry or another kind.
        with self._lock:
            return next((d for d in self._outputs if d.id == output_id and d.kind is kind), None)

    def matching(self, kind: type[D]) -> list[ProgramOutput[D]]:
        """Program outputs of this kind, newest first."""
        with self._lock:
            return [d for d in reversed(self._outputs) if d.kind is kind]

    def group_label(self, program_output: ProgramOutput[Any]) -> str:
        return group_label(program_output)

    def subscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._subscribers.remove(callback)

    def notify(self) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback()

    @property
    def nbytes(self) -> int:
        """Memory held by every entry together."""
        with self._lock:
            return sum(program_output.nbytes for program_output in self._outputs)

    @property
    def over_limit(self) -> bool:
        return self.nbytes > self.limit_bytes

    def check_room(self) -> None:
        """Refuse new data while over the limit with nothing to free room."""
        if self.over_limit and not self.auto_purge:
            raise LibraryFull(
                f"The data library holds {gib(self.nbytes)}, over its "
                f"{gib(self.limit_bytes)} limit. Close entries in the Data Library "
                "panel, or turn on auto-purge there, to acquire or load more."
            )

    def purgeable(self) -> list[ProgramOutput[Any]]:
        """What auto-purge may close, oldest first: finished entries, never one
        still being written."""
        with self._lock:
            return [program_output for program_output in self._outputs if program_output.finished]

    def __len__(self) -> int:
        with self._lock:
            return len(self._outputs)
