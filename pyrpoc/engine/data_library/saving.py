"""``RecordingSaver``: one recording's metadata file, its log of applied edits and its sidecar
files, beside the files its outputs write."""

from __future__ import annotations

import threading
from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pyrpoc.core.program_input import SidecarProgramInput
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording, utc_now
from pyrpoc.engine.data_library.format import (
    AppliedEdit,
    OutputRecord,
    RecordingRecord,
    applied_edits_path_for,
    free_root,
    meta_path_for,
    pyrpoc_version,
    write_record,
)
from pyrpoc.utils.file_formats.json_file import append_json_line

from .notes import save_note


def save_location(root: Path, taken: Collection[Path] = ()) -> tuple[Path, Path]:
    """The first free numbered stem of ``root``, which every file of a recording hangs its
    suffix off, and the metadata file there. A stem is taken by a recording on disk, or by
    one whose metadata file is among ``taken``."""
    free = free_root(root, taken)
    return free, meta_path_for(free)


class RecordingSaver:
    """Writes ``recording``'s files beside the stem ``root`` as its run goes: the metadata, the
    log of edits it applied, and the sidecar inputs its program saves."""

    def __init__(self, recording: Recording, root: Path):
        self.recording = recording
        self.root = root
        self.meta_path = meta_path_for(root)
        self.edits_path = applied_edits_path_for(root)
        # How many files each sidecar input has been saved to, by key.
        self.sidecar_counts: dict[str, int] = {}
        # A run writes from its own thread, and a note edit from the GUI thread.
        self.lock = threading.Lock()

    def begin(self) -> None:
        """Create the folder, and put the recording on disk before any of its parts."""
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.edits_path.write_text("", encoding="utf-8")
        self.write_metadata(ended_at=None, last_error=None)
        # A note written before the run is set now, as one edited later is set by the library.
        if self.recording.note:
            save_note(self.recording)

    def log_applied(self, edit: AppliedEdit) -> None:
        """Add an edit the run applied to the log, on disk at once."""
        append_json_line(self.edits_path, asdict(edit))

    def save_sidecar(self, program_input: SidecarProgramInput[Any], value: Any) -> str:
        """Write ``value`` to the next file of ``program_input``, returning the file's name."""
        number = self.sidecar_counts.get(program_input.key, 0) + 1
        self.sidecar_counts[program_input.key] = number
        stem = f"{self.root.name}_{program_input.key}_{number}"
        path = self.root.with_name(f"{stem}{program_input.file.suffix}")
        program_input.file.write(path, value)
        return path.name

    def part_begun(self) -> None:
        """Put a new part of the recording, and its outputs, on disk at once."""
        self.write_metadata(ended_at=None, last_error=None)

    def finalize(self, last_error: str | None) -> None:
        """Rewrite the metadata once the run has ended, with its outcome and its first failure."""
        # The program outputs are closed first, so every output's paths are complete by now.
        self.write_metadata(ended_at=utc_now(), last_error=last_error)

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
            outcome=recording.outcome,
            parts=list(recording.parts),
            outputs={
                name: output_record(program_output, self.meta_path.parent)
                for name, program_output in recording.outputs.items()
            },
            # Read each time: the note may be edited while the recording runs.
            notes=recording.note,
            applied_edits=self.edits_path.name,
        )


def save_finished(recording: Recording, root: Path) -> None:
    """Write a finished recording that was never saved beside the stem ``root``, as if its run
    had saved it: every output's files, then the metadata, then its note."""
    saver = RecordingSaver(recording, root)
    saver.meta_path.parent.mkdir(parents=True, exist_ok=True)
    for program_output in recording.outputs.values():
        program_output.save_to(root)
    # A run not saved logged no edits and kept no sidecar inputs, so neither can be written now.
    saver.edits_path.write_text("", encoding="utf-8")
    recording.name = root.name
    # When the run ended and its first failure were not kept, so they are left unknown.
    saver.write_metadata(ended_at=None, last_error=None)
    recording.location = saver.meta_path
    save_note(recording)


def output_record(program_output: ProgramOutput, folder: Path) -> OutputRecord:
    """Its type and files, each named from ``folder``, so an output may keep files in folders
    of its own."""
    with program_output.lock:
        files = {
            key: path.relative_to(folder).as_posix() for key, path in program_output.paths.items()
        }
    return OutputRecord(type=program_output.type_key, files=files)
