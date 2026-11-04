"""``Notes``: what the user wrote about each run, one note per recording."""

from __future__ import annotations

import threading


class Notes:
    """One note per recording, shared by all its outputs, saved or not. Keyed by
    ``grouping.recording_key``."""

    # Keyed by recording rather than run id: a loaded recording keeps the run id it had when
    # taken, which can repeat one from this session.

    def __init__(self) -> None:
        self._notes: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str:
        with self._lock:
            return self._notes.get(key, "")

    def set(self, key: str, text: str) -> None:
        with self._lock:
            self._notes[key] = text
