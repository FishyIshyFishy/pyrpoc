"""How open recordings group into Experiments, and how that tree sorts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pyrpoc.core.program import program_registry
from pyrpoc.core.recording import Recording

# The group of recordings no folder holds.
UNSAVED = "Unsaved"


def program_display_name(program_key: str) -> str:
    """The program's name as the app lists it, or its key if it is not registered now."""
    # Recordings on disk may name a program this launch lacks, such as a simulated one.
    if program_key in program_registry.entries:
        return program_registry.get(program_key).display_name
    return program_key


def recording_title(recording: Recording) -> str:
    """What the recording was called, or its program when it was not named."""
    return recording.name or program_display_name(recording.program_key)


@dataclass
class ExperimentNode:
    """One Experiment folder and its recordings, or the Unsaved group when ``folder`` is
    None."""

    label: str
    folder: Path | None
    recordings: list[Recording] = field(default_factory=list)

    @property
    def key(self) -> str:
        return UNSAVED if self.folder is None else str(self.folder)

    @property
    def latest(self) -> str:
        """When its newest recording was taken, or empty when it has none."""
        return max((recording.started_at for recording in self.recordings), default="")


class SortBy(Enum):
    NAME = "name"
    DATE = "date"


def build_tree(recordings: list[Recording], folders: list[Path]) -> list[ExperimentNode]:
    """Every recording under its Experiment, plus each opened folder that has none."""
    experiments: dict[Path | None, ExperimentNode] = {
        folder: ExperimentNode(folder.name, folder) for folder in folders
    }
    for recording in recordings:
        folder = recording.folder
        if folder not in experiments:
            experiments[folder] = ExperimentNode(UNSAVED if folder is None else folder.name, folder)
        experiments[folder].recordings.append(recording)
    return list(experiments.values())


def sort_tree(
    experiments: list[ExperimentNode], by: SortBy, descending: bool
) -> list[ExperimentNode]:
    """Experiments and the recordings in each, ordered like a file explorer; Unsaved stays
    on top, and outputs stay in the order their program declares them."""
    for experiment in experiments:
        experiment.recordings.sort(
            key=lambda r: (
                (recording_title(r).casefold(), r.started_at)
                if by is SortBy.NAME
                else (r.started_at, recording_title(r).casefold())
            ),
            reverse=descending,
        )
    folders = sorted(
        (each for each in experiments if each.folder is not None),
        key=lambda e: (e.label.casefold(), e.key) if by is SortBy.NAME else (e.latest, e.key),
        reverse=descending,
    )
    unsaved = [each for each in experiments if each.folder is None]
    return unsaved + folders
