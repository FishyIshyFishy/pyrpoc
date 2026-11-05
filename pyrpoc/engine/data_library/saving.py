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
    TakenRequest,
    free_root,
    meta_path_for,
    pyrpoc_version,
    write_record,
)

from .notes import save_note


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
        self.requests: list[TakenRequest] = []
        # A run writes from its own thread, and a note edit from the GUI thread.
        self.lock = threading.Lock()

    def begin(self) -> None:
        """Create the folder, and put the recording on disk before its data."""
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_metadata(ended_at=None, last_error=None)
        # A note written before the run is set now, as one edited later is set by the library.
        if self.recording.note:
            save_note(self.recording)

    def log_live(self, read: LiveRead) -> None:
        """Record a run being given a changed live input, on disk at once."""
        self.live_reads.append(read)
        self.write_metadata(ended_at=None, last_error=None)

    def log_request(self, taken: TakenRequest) -> None:
        """Record a run taking a request, on disk at once."""
        self.requests.append(taken)
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
            parameters=recording.parameters,
            devices=recording.devices,
            outputs={
                name: output_record(program_output)
                for name, program_output in recording.outputs.items()
            },
            # Read each time: the note may be edited while the recording runs.
            notes=recording.note,
            inputs=recording.reads,
            live_reads=list(self.live_reads),
            requests=list(self.requests),
        )


def output_record(program_output: ProgramOutput) -> OutputRecord:
    with program_output.lock:
        files = {key: path.name for key, path in program_output.paths.items()}
    return OutputRecord(type=program_output.type_key, files=files)
