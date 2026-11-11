"""``LibraryStore``: the recordings currently open, acquired now or loaded from disk. Not
persisted."""

from __future__ import annotations

import threading
from collections.abc import Callable

from pyrpoc.core.recording import Recording

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
        # Close the oldest finished recordings whenever the total is over the limit.
        self.auto_purge = False
        self._recordings: list[Recording] = []
        self._subscribers: list[Callable[[], None]] = []
        self._lock = threading.RLock()

    def add(self, recording: Recording) -> None:
        with self._lock:
            self._recordings.append(recording)
        self.notify()

    def close(self, recording: Recording) -> None:
        with self._lock:
            self._recordings.remove(recording)
        self.notify()

    def recordings(self) -> list[Recording]:
        """Every open recording, oldest first."""
        with self._lock:
            return list(self._recordings)

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
        """Memory held by every recording together."""
        return sum(recording.nbytes for recording in self.recordings())

    @property
    def over_limit(self) -> bool:
        return self.nbytes > self.limit_bytes

    @property
    def full(self) -> bool:
        """Over the limit with nothing to free room, so new data is refused."""
        return self.over_limit and not self.auto_purge

    def check_room(self) -> None:
        """Refuse new data while full."""
        if self.full:
            raise LibraryFull(
                f"The data library holds {gib(self.nbytes)}, over its "
                f"{gib(self.limit_bytes)} limit. Close recordings in the Data Library "
                "panel, or turn on auto-purge there, to acquire or load more."
            )

    def purgeable(self) -> list[Recording]:
        """What auto-purge may close, oldest first: finished recordings, never one still
        being written."""
        return [recording for recording in self.recordings() if recording.finished]

    def __len__(self) -> int:
        with self._lock:
            return len(self._recordings)
