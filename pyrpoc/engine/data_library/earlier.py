"""The outputs of finished recordings in an Experiment that later runs can read, and the
recordings there that cannot be read."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyrpoc.core.program_output import ProgramOutput, program_output_registry

from .format import META_SUFFIX, RecordingError, read_record, recordings_in


@dataclass(frozen=True)
class EarlierOutput:
    """One output of a finished recording on disk, as a choice of what a run reads."""

    meta_path: Path
    # The file's stem, which tells apart recordings saved under one name.
    name: str
    started_at: str
    output: str
    output_type: type[ProgramOutput]


@dataclass(frozen=True)
class Unreadable:
    """A recording on disk whose metadata cannot be read, and why."""

    meta_path: Path
    reason: str


def earlier_outputs(folder: Path) -> tuple[list[EarlierOutput], list[Unreadable]]:
    """Every output of a known type in a recording in ``folder`` that ended without an error,
    newest first, and the recordings there that cannot be read."""
    # One still being written, or one that failed, is nothing a later run should build on.
    found: list[EarlierOutput] = []
    unreadable: list[Unreadable] = []
    for meta_path in recordings_in(folder):
        try:
            record = read_record(meta_path)
        except RecordingError as exc:
            unreadable.append(Unreadable(meta_path, str(exc)))
            continue
        if record.ended_at is None or record.last_error:
            continue
        stem = meta_path.name.removesuffix(META_SUFFIX)
        for output, entry in record.outputs.items():
            if entry.type in program_output_registry.keys():
                output_type = program_output_registry.get(entry.type)
                found.append(EarlierOutput(meta_path, stem, record.started_at, output, output_type))
    found.sort(key=lambda each: each.started_at, reverse=True)
    return found, unreadable
