"""Saving: one copy of what modalities/*/storage.py did three times.

The on-disk layout:

    <root>_<channel>.tiff   appended float32, one page per published array
    <root>_<stream>.npz     data / parameters
    <root>_meta.json        written once when the run starts, once when it ends

Nothing here counts. How much a run produced is recoverable from the data
itself -- the page count of a TIFF, the leading axis of an npz -- and how much
was asked for rides in ``parameters`` with the rest of the acquisition
settings. A saver that tracked a total had to nominate one stream of a
multi-stream run as the one worth counting, which is a hierarchy none of these
files needs.

The auxiliary-payload machinery this replaces -- ``_pending_auxiliary``,
``append_auxiliary_payload``, ``flush_auxiliary_payloads`` -- existed only
because split confocal produced a second output and there was no way to declare
one. Streams are declared in ``emits`` now, so they all travel the same path.

``<root>`` comes from a ``SaveTarget``, which is also where the acquisition's
name comes from. It is here rather than in the parameter model because saving
is a property of a run and not of the program that fills it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from pyrpoc.core.errors import ParameterError
from pyrpoc.core.streams import Image2D, Stream

from .dataset import Dataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SaveTarget:
    """What an acquisition is called, where it goes, and whether it goes.

    Not a parameter group. Every program declared an identical ``SaveGroup``
    and the runner reached past the parameter model to find it, which made
    saving look like a decision a program makes. It is not: nothing about
    where bytes land depends on what produced them, so this travels as its own
    argument to the runner and lives once per session rather than once per
    program.

    ``name`` is a bare filename and means something with saving off -- it is
    what the acquisition is called in the data panel, where a full path would
    be both misleading (nothing was written) and too wide to read.
    """

    name: str = "acquisition"
    directory: str = ""
    enabled: bool = False

    @property
    def filename(self) -> str:
        """``name`` as a bare filename: no directory, no TIFF suffix.

        The writers append their own ``_<channel>.tiff``, so a typed ".tiff"
        would land in the middle of the real filename.
        """
        stem = Path((self.name or "").strip()).name
        if stem.lower().endswith((".tif", ".tiff")):
            stem = stem.rsplit(".", 1)[0]
        return stem

    @property
    def folder(self) -> Path:
        """Where files go. No directory means the working directory."""
        text = (self.directory or "").strip()
        return Path(text).expanduser() if text else Path.cwd()

    @property
    def root(self) -> Path:
        """The base path the writers hang their suffixes off."""
        stem = self.filename
        if not stem:
            raise ParameterError("Name is required when saving is enabled")
        return self.folder / stem


class StreamWriter:
    """Base: puts one stream's arrays on disk."""

    def __init__(self, saver: "RunSaver", stream: str):
        self.saver = saver
        self.stream = stream
        self.paths: dict[str, Path] = {}

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        raise NotImplementedError

    def finalize(self, dataset: Dataset, error: Exception | None) -> None:
        pass


class TiffStreamWriter(StreamWriter):
    """``Image2D``: one appended TIFF per channel, one page per publish."""

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        channels = [array[index] for index in range(array.shape[0])]

        if not self.paths:
            labels = dataset.resolved_channel_labels(len(channels))
            root = self.saver.root
            self.paths = {
                label: root.with_name(f"{root.name}_{label}.tiff") for label in labels
            }
            for path in self.paths.values():
                if path.exists():
                    path.unlink()

        if len(channels) != len(self.paths):
            raise ValueError("channel count does not match the configured save layout")

        for path, channel_plane in zip(self.paths.values(), channels):
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(np.asarray(channel_plane, dtype=np.float32))


class NpzStreamWriter(StreamWriter):
    """Everything that is not ``Image2D``: buffered, written once at finalize."""

    def __init__(self, saver: "RunSaver", stream: str):
        super().__init__(saver, stream)
        self._buffer: list[np.ndarray] = []

    def write(self, dataset: Dataset, array: np.ndarray) -> None:
        self._buffer.append(np.asarray(array, dtype=np.float32))

    def finalize(self, dataset: Dataset, error: Exception | None) -> None:
        if not self._buffer:
            return
        root = self.saver.root
        path = root.with_name(f"{root.name}_{self.stream}.npz")
        #: Leading axis is one entry per published array, in publish order.
        payload = np.stack(self._buffer, axis=0)
        np.savez_compressed(
            str(path),
            data=payload,
            parameters=np.asarray(self.saver.parameters, dtype=object),
        )
        self.paths = {self.stream: path}


def writer_for_spec(saver: "RunSaver", stream: str, spec: type[Stream]) -> StreamWriter:
    return (
        TiffStreamWriter(saver, stream)
        if spec is Image2D
        else NpzStreamWriter(saver, stream)
    )


class RunSaver:
    """Owns one run's output: the per-stream writers and the metadata file.

    One metadata file per run rather than per stream, so a multi-stream run
    still describes itself in one place. It is written twice -- once from
    ``prepare`` so the run is on disk before any data is, and once from
    ``finalize`` once the writers know their paths.
    """

    def __init__(
        self,
        *,
        root: Path,
        program_key: str,
        parameters: dict[str, Any],
        devices: dict[str, Any] | None = None,
        run_id: int = 1,
        started_at: str | None = None,
    ):
        self.root = Path(root)
        self.program_key = program_key
        self.parameters = dict(parameters)
        self.devices = dict(devices or {})
        self.run_id = run_id
        self.started_at = started_at or utc_now()

        self.json_path = self.root.with_name(f"{self.root.name}_meta.json")
        self.writers: dict[str, StreamWriter] = {}

    def prepare(self, streams: dict[str, type[Stream]]) -> None:
        """Create the output directory and write the metadata stub."""
        self.root.parent.mkdir(parents=True, exist_ok=True)
        for stream, spec in streams.items():
            self.writers[stream] = writer_for_spec(self, stream, spec)
        self.write_metadata(None)

    def writer_for(self, stream: str) -> StreamWriter | None:
        return self.writers.get(stream)

    def finalize(self, error: Exception | None) -> None:
        """Rewrite the metadata now that every writer knows its paths.

        ``Runner.worker`` finalizes every dataset before it finalizes the
        saver, so ``tiff_paths`` and ``auxiliary_paths`` are both complete by
        the time this runs.
        """
        self.write_metadata(str(error) if error is not None else None)

    # -- metadata ---------------------------------------------------------- #

    def tiff_paths(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for writer in self.writers.values():
            if isinstance(writer, TiffStreamWriter):
                out.update({label: str(path) for label, path in writer.paths.items()})
        return out

    def auxiliary_paths(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for writer in self.writers.values():
            if isinstance(writer, NpzStreamWriter):
                out.update({label: str(path) for label, path in writer.paths.items()})
        return out

    def write_metadata(self, last_error: str | None) -> None:
        payload = {
            "run_id": self.run_id,
            "started": self.started_at,
            "program_key": self.program_key,
            "save_root_path": str(self.root),
            "save_json_path": str(self.json_path),
            "streams": sorted(self.writers),
            "tiff_paths": self.tiff_paths(),
            "auxiliary_paths": self.auxiliary_paths(),
            "parameters": self.parameters,
            "devices": self.devices,
            "last_error": last_error,
        }
        self.json_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
