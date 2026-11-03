"""``Recording``: the program outputs and optional saver that runs write into, and
``Series``, consecutive runs that continue one recording."""

# A run on its own gets a fresh recording that closes when it ends. Runs in a series continue its
# recording while the ``RecordingKey`` holds; a run whose parameters, devices or save target
# differ opens a new one. No Qt: the executor's lock guards every mutable field here.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.data_library.saving import RecordingSaver

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.engine.acquisition.executor import Run


@dataclass(frozen=True)
class RecordingKey:
    """What a recording's data depends on: a run whose key differs writes a new recording."""

    program_key: str
    parameters: dict[str, Any]
    devices: dict[str, Any]
    save: SaveTarget


class Recording:
    def __init__(
        self,
        key: RecordingKey,
        outputs: dict[str, ProgramOutput[Any]],
        saver: RecordingSaver | None,
    ):
        self.key = key
        self.outputs = outputs
        self.saver = saver
        # Runs writing into it right now; it closes only once this is zero.
        self.active = 0
        self.closed = False
        self.error: Exception | None = None
        # Where a failure to close is reported: the run that wrote last.
        self.last_run: Run | None = None

    def finalize(self) -> list[str]:
        """Close the program outputs, then the saver, returning every failure's message."""
        # Collected rather than raised: a failed save is still a failure, and the other closers
        # must still run.
        closers: list[Callable[[], None]] = [
            lambda d=program_output: d.finalize(self.error)
            for program_output in self.outputs.values()
        ]
        saver = self.saver
        if saver is not None:
            closers.append(lambda: saver.finalize(self.error))
        errors: list[str] = []
        for close in closers:
            try:
                close()
            except Exception as exc:
                errors.append(str(exc))
        return errors


class Series:
    """Consecutive runs that continue one recording. The runner that opened it
    closes it; the recording closes once its last run has ended."""

    def __init__(self) -> None:
        self.recording: Recording | None = None
        self.closed = False
