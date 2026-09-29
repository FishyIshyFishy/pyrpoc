"""Saving a run to disk: one writer per output, chosen by its kind of ``Data``
from ``writer_registry``, plus one metadata file for the whole run.

    <root>_meta.json        written once when the run starts, once when it ends

The formats themselves live in ``programs/components/writers/``. Nothing here
counts frames: a file's own length already says how much was produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyrpoc.src.structs.data import Data, Writer
from pyrpoc.src.structs.registries import writer_registry


class RunSaver:
    """One run's writers and its single metadata file, written from ``prepare``
    so the run is on disk before any data, and again from ``finalize`` once the
    writers know their paths."""

    def __init__(
        self,
        *,
        root: Path,
        program_key: str,
        parameters: dict[str, Any],
        devices: dict[str, Any],
        run_id: int,
        started_at: str,
    ):
        self.root = root
        self.program_key = program_key
        self.parameters = dict(parameters)
        self.devices = dict(devices)
        self.run_id = run_id
        self.started_at = started_at

        self.json_path = self.root.with_name(f"{self.root.name}_meta.json")
        self.writers: dict[str, Writer] = {}

    def prepare(self, outputs: dict[str, type[Data]]) -> None:
        """Create the output directory and write the metadata stub."""
        self.root.parent.mkdir(parents=True, exist_ok=True)
        for output, spec in outputs.items():
            self.writers[output] = writer_registry.get(spec.name)(
                self.root, output, self.parameters
            )
        self.write_metadata(None)

    def writer_for(self, output: str) -> Writer:
        return self.writers[output]

    def finalize(self, error: Exception | None) -> None:
        """Rewrite the metadata. The executor finalizes every dataset first, so
        every writer's paths are complete by now."""
        self.write_metadata(str(error) if error is not None else None)

    def write_metadata(self, last_error: str | None) -> None:
        paths: dict[str, dict[str, str]] = {"tiff_paths": {}, "auxiliary_paths": {}}
        for writer in self.writers.values():
            paths[writer.metadata_key].update(
                {label: str(path) for label, path in writer.paths.items()}
            )
        payload = {
            "run_id": self.run_id,
            "started": self.started_at,
            "program_key": self.program_key,
            "save_root_path": str(self.root),
            "save_json_path": str(self.json_path),
            "outputs": sorted(self.writers),
            **paths,
            "parameters": self.parameters,
            "devices": self.devices,
            "last_error": last_error,
        }
        self.json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
