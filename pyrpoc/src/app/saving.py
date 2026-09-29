"""Saving a run to disk.

    <root>_<channel>.tiff   appended float32, one page per published array
    <root>_<output>.npz     data / parameters
    <root>_meta.json        written once when the run starts, once when it ends

Nothing here counts frames: a TIFF's page count or an npz's leading axis
already says how much was produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import tifffile

from pyrpoc.src.structs.data import Data, Dataset, Image2D
from pyrpoc.src.structs.params import ParameterError


@dataclass
class SaveTarget:
    """What an acquisition is called, where it goes, and whether it goes.

    Its own argument to the executor rather than a parameter block, because
    where bytes land does not depend on which program produced them. ``name``
    is a bare filename and also names the run in the data panel with saving off.
    """

    name: str = "acquisition"
    directory: str = ""
    enabled: bool = False

    @property
    def filename(self) -> str:
        """``name`` with no directory and no TIFF suffix; the writers append
        their own ``_<channel>.tiff``."""
        stem = Path(self.name.strip()).name
        if stem.lower().endswith((".tif", ".tiff")):
            stem = stem.rsplit(".", 1)[0]
        return stem

    @property
    def folder(self) -> Path:
        """Where files go. No directory means the working directory."""
        text = self.directory.strip()
        return Path(text).expanduser() if text else Path.cwd()

    @property
    def root(self) -> Path:
        """The base path the writers hang their suffixes off."""
        if not self.filename:
            raise ParameterError("Name is required when saving is enabled")
        return self.folder / self.filename


class Writer:
    """Puts one output's arrays on disk."""

    # Which metadata entry lists this writer's files.
    metadata_key: ClassVar[str]

    def __init__(self, saver: RunSaver, output: str):
        self.saver = saver
        self.output = output
        self.paths: dict[str, Path] = {}

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        raise NotImplementedError

    def finalize(self, dataset: Dataset, error: Exception | None) -> None:
        del dataset, error


class TiffWriter(Writer):
    """``Image2D``: one appended TIFF per channel, one page per publish."""

    metadata_key = "tiff_paths"

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        channels = [array[index] for index in range(array.shape[0])]

        if not self.paths:
            labels = dataset.resolved_channel_labels(len(channels))
            root = self.saver.root
            self.paths = {label: root.with_name(f"{root.name}_{label}.tiff") for label in labels}
            for path in self.paths.values():
                path.unlink(missing_ok=True)

        if len(channels) != len(self.paths):
            raise ValueError("channel count does not match the configured save layout")

        for path, channel_plane in zip(self.paths.values(), channels, strict=True):
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(np.asarray(channel_plane, dtype=np.float32))


class NpzWriter(Writer):
    """Everything that is not ``Image2D``: buffered, written once at finalize."""

    metadata_key = "auxiliary_paths"

    def __init__(self, saver: RunSaver, output: str):
        super().__init__(saver, output)
        self._buffer: list[np.ndarray] = []

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        self._buffer.append(np.asarray(array, dtype=np.float32))

    def finalize(self, dataset: Dataset, error: Exception | None) -> None:
        if not self._buffer:
            return
        root = self.saver.root
        path = root.with_name(f"{root.name}_{self.output}.npz")
        # Leading axis is one entry per published array, in publish order.
        np.savez_compressed(
            str(path),
            data=np.stack(self._buffer, axis=0),
            parameters=np.asarray(self.saver.parameters, dtype=object),
        )
        self.paths = {self.output: path}


def make_writer(saver: RunSaver, output: str, spec: type[Data]) -> Writer:
    return TiffWriter(saver, output) if spec is Image2D else NpzWriter(saver, output)


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
            self.writers[output] = make_writer(self, output, spec)
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
