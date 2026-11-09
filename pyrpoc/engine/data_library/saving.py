"""``RecordingSaver``: one recording's metadata file, beside the files its outputs write."""

from __future__ import annotations

import threading
from collections.abc import Collection
from pathlib import Path

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording, utc_now
from pyrpoc.engine.data_library.format import (
    OutputRecord,
    RecordingRecord,
    Taken,
    free_root,
    meta_path_for,
    pyrpoc_version,
    write_record,
)

from .notes import save_note


def save_location(root: Path, taken: Collection[Path] = ()) -> tuple[Path, Path]:
    """The first free numbered stem of ``root``, which every file of a recording hangs its
    suffix off, and the metadata file there. A stem is taken by a recording on disk, or by
    one whose metadata file is among ``taken``."""
    free = free_root(root, taken)
    return free, meta_path_for(free)


class RecordingSaver:
    """Writes ``recording``'s metadata file, at its location, as its run goes."""

    def __init__(self, recording: Recording, meta_path: Path):
        self.recording = recording
        self.meta_path = meta_path
        self.taken: list[Taken] = []
        # A run writes from its own thread, and a note edit from the GUI thread.
        self.lock = threading.Lock()

    def begin(self) -> None:
        """Create the folder, and put the recording on disk before its data."""
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_metadata(ended_at=None, last_error=None)
        # A note written before the run is set now, as one edited later is set by the library.
        if self.recording.note:
            save_note(self.recording)

    def log_taken(self, taken: Taken) -> None:
        """Record the run taking something from its inbox, on disk at once."""
        self.taken.append(taken)
        self.write_metadata(ended_at=None, last_error=None)

    def part_begun(self) -> None:
        """Put a new part of the recording, and its outputs, on disk at once."""
        self.write_metadata(ended_at=None, last_error=None)

    def finalize(self, error: Exception | None) -> None:
        """Rewrite the metadata."""
        # The program outputs are closed first, so every output's paths are complete by now.
        self.write_metadata(ended_at=utc_now(), last_error=str(error) if error else None)

    def write_metadata(self, *, ended_at: str | None, last_error: str | None) -> None:
        with self.lock:
            write_record(self.meta_path, self.record(ended_at, last_error))

    def record(self, ended_at: str | None, last_error: str | None) -> RecordingRecord:
        recording = self.recording
        return RecordingRecord(
            pyrpoc_version=pyrpoc_version(),
            program_key=recording.program_key,
            name=recording.name,
            started_at=recording.started_at,
            ended_at=ended_at,
            last_error=last_error,
            parts=list(recording.parts),
            outputs={
                name: output_record(program_output, self.meta_path.parent)
                for name, program_output in recording.outputs.items()
            },
            # Read each time: the note may be edited while the recording runs.
            notes=recording.note,
            taken=list(self.taken),
        )


def output_record(program_output: ProgramOutput, folder: Path) -> OutputRecord:
    """Its type and files, each named from ``folder``, so an output may keep files in folders
    of its own."""
    with program_output.lock:
        files = {
            key: path.relative_to(folder).as_posix() for key, path in program_output.paths.items()
        }
    return OutputRecord(type=program_output.type_key, files=files)
