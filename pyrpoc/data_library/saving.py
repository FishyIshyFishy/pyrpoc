"""Saving a recording to disk: one writer per output, chosen by its kind of
``Data`` from ``writer_registry``, plus the recording's metadata file.

    <root>_meta.json        rewritten as each run joins and ends, and at the end

The formats themselves live in ``programs/components/writers/``, and the
metadata file's shape in ``recording_format.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyrpoc.data_library.format import (
    OutputRecord,
    RecordingRecord,
    RunRecord,
    free_root,
    meta_path_for,
    pyrpoc_version,
    write_record,
)
from pyrpoc.structs.data_library.data import Data
from pyrpoc.structs.data_library.dataset import Dataset, utc_now
from pyrpoc.structs.data_library.saving import Writer, writer_registry


class RecordingSaver:
    """One recording's writers and metadata file. ``root`` is moved to the
    first free numbered stem, so an earlier recording is never overwritten."""

    def __init__(
        self,
        *,
        root: Path,
        program_key: str,
        name: str,
        parameters: dict[str, Any],
        devices: dict[str, Any],
        started_at: str,
    ):
        self.root = free_root(root)
        self.meta_path = meta_path_for(self.root)
        self.program_key = program_key
        self.name = name
        self.parameters = dict(parameters)
        self.devices = dict(devices)
        self.started_at = started_at
        self.writers: dict[str, Writer] = {}
        self.datasets: dict[str, Dataset] = {}
        self.runs: list[RunRecord] = []

    def prepare(self, outputs: dict[str, type[Data]]) -> None:
        """Create the output directory and a writer per output."""
        self.root.parent.mkdir(parents=True, exist_ok=True)
        for output, spec in outputs.items():
            self.writers[output] = writer_registry.get(spec.name)(
                self.root, output, self.parameters
            )

    def writer_for(self, output: str) -> Writer:
        return self.writers[output]

    def begin(self, datasets: dict[str, Dataset]) -> None:
        """Take the datasets the writers feed, which the metadata describes."""
        self.datasets = datasets
        for dataset in datasets.values():
            dataset.meta_path = self.meta_path

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

    def finalize(self, error: Exception | None) -> None:
        """Rewrite the metadata. The datasets are finalized first, so every
        writer's paths are complete by now."""
        self.write_metadata(ended_at=utc_now(), last_error=str(error) if error else None)

    def write_metadata(self, *, ended_at: str | None, last_error: str | None) -> None:
        write_record(self.meta_path, self.record(ended_at, last_error))

    def record(self, ended_at: str | None, last_error: str | None) -> RecordingRecord:
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
                output: output_record(dataset, self.writers[output])
                for output, dataset in self.datasets.items()
            },
        )


def output_record(dataset: Dataset, writer: Writer) -> OutputRecord:
    return OutputRecord(
        kind=dataset.spec.name,
        channels=list(dataset.channel_labels),
        frames=len(dataset),
        files={key: path.name for key, path in writer.paths.items()},
        metadata=dict(dataset.metadata),
        notes=dataset.notes,
    )
