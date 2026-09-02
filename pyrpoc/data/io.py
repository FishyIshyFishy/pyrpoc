"""Saving: one copy of what modalities/*/storage.py did three times.

The on-disk layout is preserved so existing analysis scripts keep working:

    <root>_<channel>.tiff   appended float32, one page per frame (Image2D)
    <root>_<stream>.npz     frames / parameters / frame_indices (Cube3D, Samples4D)
    <root>_meta.json        rewritten after every frame, so run progress stays
                            readable from disk mid-run

Two deliberate differences, both recorded in the plan: FLIM's histogram file was
``<root>_raw.npz`` with ``frames`` as ``dtype=object`` and an
``acquisition_parameters`` key; it is now ``<root>_histogram.npz`` with a real
float32 array and a ``parameters`` key. Split confocal's raw file is unchanged.

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
    """Base: puts one stream's frames on disk."""

    def __init__(self, saver: "RunSaver", stream: str):
        self.saver = saver
        self.stream = stream
        self.paths: dict[str, Path] = {}

    def write(self, dataset: Dataset, array: np.ndarray, frame_index: int) -> None:
        raise NotImplementedError

    def finalize(self, dataset: Dataset, frame_count: int, error: Exception | None) -> None:
        pass


class TiffStreamWriter(StreamWriter):
    """``Image2D``: one appended TIFF per channel, exactly as v3.0 wrote them."""

    def write(self, dataset: Dataset, array: np.ndarray, frame_index: int) -> None:
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
            raise ValueError("frame channel count does not match the configured save layout")

        for path, channel_frame in zip(self.paths.values(), channels):
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(np.asarray(channel_frame, dtype=np.float32))

        self.saver.on_frame_written(self.stream, frame_index)


class NpzStreamWriter(StreamWriter):
    """``Cube3D`` / ``Samples4D``: buffered, written once at finalize."""

    def __init__(self, saver: "RunSaver", stream: str):
        super().__init__(saver, stream)
        self._buffer: list[np.ndarray] = []

    def write(self, dataset: Dataset, array: np.ndarray, frame_index: int) -> None:
        self._buffer.append(np.asarray(array, dtype=np.float32))
        self.saver.on_frame_written(self.stream, frame_index)

    def finalize(self, dataset: Dataset, frame_count: int, error: Exception | None) -> None:
        if not self._buffer:
            return
        root = self.saver.root
        path = root.with_name(f"{root.name}_{self.stream}.npz")
        payload = np.stack(self._buffer, axis=0)
        np.savez_compressed(
            str(path),
            frames=payload,
            parameters=np.asarray(self.saver.parameters, dtype=object),
            frame_indices=np.arange(payload.shape[0], dtype=np.int32),
        )
        self.paths = {self.stream: path}
        self.saver.write_metadata(str(error) if error is not None else None)


def writer_for_spec(saver: "RunSaver", stream: str, spec: type[Stream]) -> StreamWriter:
    return (
        TiffStreamWriter(saver, stream)
        if spec is Image2D
        else NpzStreamWriter(saver, stream)
    )


class RunSaver:
    """Owns one run's output: the per-stream writers and the metadata file.

    One metadata file per run rather than per stream, so a multi-stream run
    still produces the single ``_meta.json`` v3.0 produced.
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
        frame_limit: int | None = None,
    ):
        self.root = Path(root)
        self.program_key = program_key
        self.parameters = dict(parameters)
        self.devices = dict(devices or {})
        self.run_id = run_id
        self.started_at = started_at or utc_now()
        self.frame_limit = frame_limit

        self.json_path = self.root.with_name(f"{self.root.name}_meta.json")
        self.writers: dict[str, StreamWriter] = {}
        self.primary_stream: str | None = None
        self.frames_saved = 0

    def prepare(self, streams: dict[str, type[Stream]]) -> None:
        """Create the output directory and write the metadata stub."""
        self.root.parent.mkdir(parents=True, exist_ok=True)
        for stream, spec in streams.items():
            self.writers[stream] = writer_for_spec(self, stream, spec)
            if self.primary_stream is None:
                self.primary_stream = stream
        self.write_metadata(None)

    def writer_for(self, stream: str) -> StreamWriter | None:
        return self.writers.get(stream)

    def on_frame_written(self, stream: str, frame_index: int) -> None:
        """Bump the saved-frame count and rewrite the metadata.

        Counted against the first declared stream, which is ``intensity`` for
        all three programs, so ``frames_saved`` means what it meant in v3.0.
        """
        if stream != self.primary_stream:
            return
        self.frames_saved = frame_index + 1
        self.write_metadata(None)

    def finalize(self, frame_count: int, error: Exception | None) -> None:
        self.frames_saved = frame_count
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
            # v3.0 alias, so lab scripts reading it keep working. Remove in 3.2.
            "modality_key": self.program_key,
            "save_root_path": str(self.root),
            "save_json_path": str(self.json_path),
            "streams": sorted(self.writers),
            "tiff_paths": self.tiff_paths(),
            "auxiliary_paths": self.auxiliary_paths(),
            "frames_saved": self.frames_saved,
            "frame_limit": self.frame_limit,
            "parameters": self.parameters,
            "devices": self.devices,
            "last_error": last_error,
        }
        self.json_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frames(path: Path) -> np.ndarray:
    """Read back an appended per-channel TIFF as ``(F, H, W)``.

    Appending one page at a time -- which is how v3.0 wrote these and how they
    are still written -- puts each frame in its own TIFF *series*, so a plain
    ``tifffile.imread(path)`` returns only the first frame. ``key=slice(None)``
    reads every page. Worth knowing before writing an analysis script.
    """
    return np.asarray(tifffile.imread(str(path), key=slice(None)))
