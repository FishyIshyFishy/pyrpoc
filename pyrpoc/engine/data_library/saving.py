"""``RecordingSaver``: one recording's metadata file, beside the files its outputs write."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from pyrpoc.core.program_output import ProgramOutput, utc_now
from pyrpoc.engine.data_library.format import (
    LiveRead,
    OutputRecord,
    RecordingRecord,
    RunRecord,
    free_root,
    meta_path_for,
    pyrpoc_version,
    write_folder_notes,
    write_record,
)
from pyrpoc.engine.data_library.grouping import experiment_of, recording_key_of
from pyrpoc.engine.data_library.notes import Notes


class RecordingSaver:
    """One recording's metadata file, at the first free numbered stem of ``root``, where its
    outputs write their files too."""

    def __init__(
        self,
        *,
        root: Path,
        program_key: str,
        name: str,
        parameters: dict[str, Any],
        devices: dict[str, Any],
        started_at: str,
        inputs: dict[str, str],
        notes: Notes,
    ):
        self.root = free_root(root)
        self.meta_path = meta_path_for(self.root)
        self.program_key = program_key
        self.name = name
        self.parameters = dict(parameters)
        self.devices = dict(devices)
        self.started_at = started_at
        self.inputs = dict(inputs)
        self.outputs: dict[str, ProgramOutput] = {}
        self.runs: list[RunRecord] = []
        self.live_reads: list[LiveRead] = []
        # Runs write from their own threads, and runs in a series overlap the GUI thread.
        self.lock = threading.Lock()
        self.notes = notes
        # The note as the file last held it, so the Experiment's notes are rebuilt only when
        # it changes.
        self.written_note = ""

    def prepare(self) -> None:
        """Create the folder the recording is saved in."""
        self.root.parent.mkdir(parents=True, exist_ok=True)

    def begin(self, outputs: dict[str, ProgramOutput]) -> None:
        """Take the program outputs the writers feed, which the metadata describes."""
        self.outputs = outputs
        for program_output in outputs.values():
            program_output.meta_path = self.meta_path

    def add_run(self, run_id: int, started_at: str) -> None:
        """Record a run joining, and put the recording on disk before its data."""
        self.runs.append(RunRecord(run_id=run_id, started_at=started_at, ended_at=None, error=None))
        self.write_metadata(ended_at=None, last_error=None)

    def end_run(self, run_id: int, error: Exception | None) -> None:
        run = next(run for run in self.runs if run.run_id == run_id)
        run.ended_at = utc_now()
        run.error = str(error) if error is not None else None
        # A series can run for hours; the file should list every run so far.
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
            note = self.note()
            write_record(self.meta_path, self.record(ended_at, last_error, note))
            if note != self.written_note:
                write_folder_notes(experiment_of(self.meta_path))
                self.written_note = note

    def note(self) -> str:
        """The recording's note now: it may be edited while the recording runs."""
        return self.notes.get(recording_key_of(self.meta_path))

    def record(self, ended_at: str | None, last_error: str | None, note: str) -> RecordingRecord:
        return RecordingRecord(
            pyrpoc_version=pyrpoc_version(),
            program_key=self.program_key,
            name=self.name,
            started_at=self.started_at,
            ended_at=ended_at,
            last_error=last_error,
            runs=list(self.runs),
            parameters=self.parameters,
            devices=self.devices,
            outputs={
                output: output_record(program_output)
                for output, program_output in self.outputs.items()
            },
            notes=note,
            inputs=self.inputs,
            live_reads=list(self.live_reads),
        )


def output_record(program_output: ProgramOutput) -> OutputRecord:
    with program_output.lock:
        files = {key: path.name for key, path in program_output.paths.items()}
    return OutputRecord(type=program_output.type_key, files=files)
