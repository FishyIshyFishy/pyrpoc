"""``Recording``: what one run produced, and what made it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .program_output import ProgramOutput


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Outcome(str, Enum):
    """How a run ended: it ran to its end, it was stopped partway, or it failed."""

    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class RecordingPart:
    """A stretch of a recording made with one set of settings: when it began, the program
    input values and device settings it ran with, and the names of the outputs it filled."""

    started_at: str
    parameters: dict[str, Any]
    devices: dict[str, Any]
    outputs: tuple[str, ...]


class Recording:
    """What one run produced: the program, name and start time, its parts, each with the
    settings that made it, where it is saved, its note, how its run ended, and its program
    outputs."""

    def __init__(
        self,
        *,
        program_key: str,
        name: str,
        started_at: str,
        parts: list[RecordingPart],
        location: Path | None,
        note: str,
        finished: bool,
        outcome: Outcome | None,
    ):
        self.program_key = program_key
        # The save filename, set whether or not it was saved; empty for an unnamed run.
        self.name = name
        self.started_at = started_at
        # A new part begins each time the settings of a repeating run change between passes.
        self.parts = parts
        # Its metadata file, or None when it is not saved.
        self.location = location
        # Written by the user before, during or after the run; the saver reads it as it writes.
        self.note = note
        # Nothing more will be added: its run ended, or it was loaded from disk.
        self.finished = finished
        # None while its run goes, and for one on disk whose run never ended.
        self.outcome = outcome
        # Filled once by whoever opens it, since each output points back here.
        self.outputs: dict[str, ProgramOutput] = {}

    def part_of(self, output_name: str) -> RecordingPart:
        """The part that filled the output ``output_name``."""
        return next(part for part in self.parts if output_name in part.outputs)

    @property
    def folder(self) -> Path | None:
        """Its Experiment: the folder it is saved in."""
        return None if self.location is None else self.location.parent

    @property
    def nbytes(self) -> int:
        return sum(output.nbytes for output in self.outputs.values())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Recording {self.name or self.program_key} {self.started_at}>"
