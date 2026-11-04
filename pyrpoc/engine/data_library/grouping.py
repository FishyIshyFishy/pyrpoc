"""How open program outputs group into recordings and Experiments, and how that tree sorts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pyrpoc.core.program import program_registry
from pyrpoc.core.program_output import ProgramOutput

# The group of outputs no recording on disk holds.
UNSAVED = "Unsaved"


def experiment_of(meta_path: Path) -> Path:
    """The Experiment a recording belongs to: the folder it sits in."""
    return meta_path.parent


def group_label(program_output: ProgramOutput[Any]) -> str:
    """The name of the group ``program_output`` is listed under."""
    meta_path = program_output.meta_path
    return UNSAVED if meta_path is None else experiment_of(meta_path).name


def program_display_name(program_key: str) -> str:
    """The program's name as the app lists it, or its key if it is not registered now."""
    # Recordings on disk may name a program this launch lacks, such as a simulated one.
    if program_key in program_registry.entries:
        return program_registry.get(program_key).display_name
    return program_key


@dataclass
class RecordingNode:
    """One run's outputs: a recording on disk, or an unsaved run."""

    key: str
    name: str
    program: str
    started_at: str
    meta_path: Path | None
    outputs: list[ProgramOutput[Any]] = field(default_factory=list)


@dataclass
class ExperimentNode:
    """One Experiment folder and its recordings, or the Unsaved group when ``folder`` is
    None."""

    label: str
    folder: Path | None
    recordings: list[RecordingNode] = field(default_factory=list)

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


def recording_key(program_output: ProgramOutput[Any]) -> str:
    meta_path = program_output.meta_path
    return f"run:{program_output.run_id}" if meta_path is None else str(meta_path)


def build_tree(outputs: list[ProgramOutput[Any]], folders: list[Path]) -> list[ExperimentNode]:
    """Every output under its recording and Experiment, plus each opened folder that has
    none."""
    experiments: dict[str, ExperimentNode] = {}
    for folder in folders:
        experiments[str(folder)] = ExperimentNode(folder.name, folder)
    recordings: dict[str, RecordingNode] = {}
    for program_output in outputs:
        meta_path = program_output.meta_path
        folder = None if meta_path is None else experiment_of(meta_path)
        group_key = UNSAVED if folder is None else str(folder)
        if group_key not in experiments:
            experiments[group_key] = ExperimentNode(group_label(program_output), folder)
        key = recording_key(program_output)
        if key not in recordings:
            provenance = program_output.provenance
            recordings[key] = RecordingNode(
                key=key,
                name=program_output.run_name,
                program=program_display_name(provenance.program_key),
                started_at=provenance.started_at,
                meta_path=meta_path,
            )
            experiments[group_key].recordings.append(recordings[key])
        recordings[key].outputs.append(program_output)
    return list(experiments.values())


def sort_tree(
    experiments: list[ExperimentNode], by: SortBy, descending: bool
) -> list[ExperimentNode]:
    """Experiments and the recordings in each, ordered like a file explorer; Unsaved stays
    on top, and outputs stay in the order their program declares them."""
    for experiment in experiments:
        experiment.recordings.sort(
            key=lambda r: (
                (r.name.casefold(), r.started_at)
                if by is SortBy.NAME
                else (r.started_at, r.name.casefold())
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
