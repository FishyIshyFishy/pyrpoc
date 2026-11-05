"""``Recording``: what one run produced, and what made it."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .program_output import ProgramOutput


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Recording:
    """What one run produced: the program, name, start time, input values, devices and
    earlier recordings that made it, where it is saved, its note, and its program outputs."""

    def __init__(
        self,
        *,
        program_key: str,
        name: str,
        started_at: str,
        parameters: dict[str, Any],
        devices: dict[str, Any],
        reads: dict[str, str],
        location: Path | None,
        note: str,
        finished: bool,
    ):
        self.program_key = program_key
        # The save filename, set whether or not it was saved; empty for an unnamed run.
        self.name = name
        self.started_at = started_at
        self.parameters = parameters
        self.devices = devices
        # The earlier recordings it read: their metadata file names in its folder, by name.
        self.reads = reads
        # Its metadata file, or None when it is not saved.
        self.location = location
        # Written by the user before, during or after the run; the saver reads it as it writes.
        self.note = note
        # Nothing more will be added: its run ended, or it was loaded from disk.
        self.finished = finished
        # Filled once by whoever opens it, since each output points back here.
        self.outputs: dict[str, ProgramOutput] = {}

    @property
    def folder(self) -> Path | None:
        """Its Experiment: the folder it is saved in."""
        return None if self.location is None else self.location.parent

    @property
    def nbytes(self) -> int:
        return sum(output.nbytes for output in self.outputs.values())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Recording {self.name or self.program_key} {self.started_at}>"
