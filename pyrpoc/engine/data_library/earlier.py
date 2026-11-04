"""The finished recordings in an Experiment that later runs can read."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

from .format import META_SUFFIX, readable_records


@dataclass(frozen=True)
class EarlierRecording:
    """One finished recording on disk, as a choice of what a run reads."""

    meta_path: Path
    # The file's stem, which tells apart runs saved under one name.
    name: str
    started_at: str


def earlier_recordings(folder: Path, program_keys: Set[str]) -> dict[str, list[EarlierRecording]]:
    """The recordings in ``folder`` of each of ``program_keys`` that ended without an error,
    newest first."""
    # One still being written, or one that failed, is nothing a later run should build on.
    found: dict[str, list[EarlierRecording]] = {key: [] for key in program_keys}
    for meta_path, record in readable_records(folder):
        if record.program_key not in found or record.ended_at is None or record.last_error:
            continue
        found[record.program_key].append(
            EarlierRecording(meta_path, meta_path.name.removesuffix(META_SUFFIX), record.started_at)
        )
    for recordings in found.values():
        recordings.sort(key=lambda each: each.started_at, reverse=True)
    return found
