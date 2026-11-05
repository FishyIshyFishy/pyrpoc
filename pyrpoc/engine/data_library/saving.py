"""``RecordingSaver``: one recording's metadata file, beside the files its outputs write."""

from __future__ import annotations

import threading
from pathlib import Path

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording, utc_now
from pyrpoc.engine.data_library.format import (
    LiveRead,
    OutputRecord,
    RecordingRecord,
    free_root,
    meta_path_for,
    pyrpoc_version,
    write_folder_notes,
    write_record,
)


def save_location(root: Path) -> tuple[Path, Path]:
    """The first free numbered stem of ``root``, which every file of a recording hangs its
    suffix off, and the metadata file there."""
    free = free_root(root)
    return free, meta_path_for(free)


class RecordingSaver:
    """Writes ``recording``'s metadata file, at its location, as its run goes."""

    def __init__(self, recording: Recording, meta_path: Path):
        self.recording = recording
        self.meta_path = meta_path
        self.live_reads: list[LiveRead] = []
        # A run writes from its own thread, and a note edit from the GUI thread.
        self.lock = threading.Lock()
        # The note as the file last held it, so the Experiment's notes are rebuilt only when
        # it changes.
        self.written_note = ""

    def begin(self) -> None:
        """Create the folder, and put the recording on disk before its data."""
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_metadata(ended_at=None, last_error=None)

    def log_live(self, read: LiveRead) -> None:
        """Record a run being given a changed live input, on disk at once."""
        self.live_reads.append(read)
        self.write_metadata(ended_at=None, last_error=None)

    def finalize(self, error: Exception | None) -> None:
        """Rewrite the metadata."""
        # The program outputs are closed first, so every output's paths are complete by now.
        self.write_metadata(ended_at=utc_now(), last_error=str(error) if error else None)

    def write_metadata(self, *, ended_at: str | None, last_error: str | None) -> None:
        with self.lock:
            # Read each time: the note may be edited while the recording runs.
            note = self.recording.note
            write_record(self.meta_path, self.record(ended_at, last_error, note))
            if note != self.written_note:
                write_folder_notes(self.meta_path.parent)
                self.written_note = note

    def record(self, ended_at: str | None, last_error: str | None, note: str) -> RecordingRecord:
        recording = self.recording
        return RecordingRecord(
            pyrpoc_version=pyrpoc_version(),
            program_key=recording.program_key,
            name=recording.name,
            started_at=recording.started_at,
            ended_at=ended_at,
            last_error=last_error,
            parameters=recording.parameters,
            devices=recording.devices,
            outputs={
                name: output_record(program_output)
                for name, program_output in recording.outputs.items()
            },
            notes=note,
            inputs=recording.reads,
            live_reads=list(self.live_reads),
        )


def output_record(program_output: ProgramOutput) -> OutputRecord:
    with program_output.lock:
        files = {key: path.name for key, path in program_output.paths.items()}
    return OutputRecord(type=program_output.type_key, files=files)
