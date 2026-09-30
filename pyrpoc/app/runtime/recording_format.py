"""Recording format: what a recording on disk is, written and read in one place.

A recording is one ``<stem>_meta.json`` with its data files beside it. The
metadata file is the only entry point: it names the files and says everything
needed to rebuild each output as a ``Dataset``. Every recording saved since
format 1 must load in every later version of pyrpoc, so a new format version
adds a decoder to ``DECODERS`` and never removes one; within a version, fields
may be added but never renamed or removed. See ``docs/data-format.md``.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.structs.data import DATA_KINDS, Dataset, Origin, Provenance
from pyrpoc.structs.registries import writer_registry

FORMAT = "pyrpoc-recording"
FORMAT_VERSION = 1
META_SUFFIX = "_meta.json"


class RecordingError(Exception):
    """A recording on disk that cannot be read. The message names the file."""


@dataclass
class RunRecord:
    run_id: int
    started_at: str
    ended_at: str | None
    error: str | None


@dataclass
class OutputRecord:
    kind: str
    channels: list[str]
    frames: int
    # File names beside the metadata file, keyed as the writer keyed them.
    files: dict[str, str]
    metadata: dict[str, Any]
    notes: str


@dataclass
class RecordingRecord:
    pyrpoc_version: str
    program_key: str
    name: str
    started_at: str
    ended_at: str | None
    last_error: str | None
    runs: list[RunRecord]
    parameters: dict[str, Any]
    devices: dict[str, Any]
    outputs: dict[str, OutputRecord]


def pyrpoc_version() -> str:
    return importlib.metadata.version("pyrpoc")


def meta_path_for(root: Path) -> Path:
    return root.with_name(f"{root.name}{META_SUFFIX}")


def free_root(root: Path) -> Path:
    """``root``, or the first of ``root_2``, ``root_3``… with no recording yet,
    so a new recording never overwrites an old one."""
    candidate, number = root, 1
    while meta_path_for(candidate).exists():
        number += 1
        candidate = root.with_name(f"{root.name}_{number}")
    return candidate


def write_record(meta_path: Path, record: RecordingRecord) -> None:
    write_json(meta_path, {"format": FORMAT, "format_version": FORMAT_VERSION, **asdict(record)})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` whole, so a crash mid-write never leaves half a file."""
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(partial, path)


def read_json(meta_path: Path) -> dict[str, Any]:
    """The metadata file's JSON, checked to be a recording this version reads."""
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingError(f"cannot read {meta_path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise RecordingError(f"{meta_path} is not a pyrpoc recording")
    version = raw.get("format_version")
    if not isinstance(version, int) or version not in DECODERS:
        raise RecordingError(
            f"{meta_path} is recording format {version!r}; this version of pyrpoc reads "
            f"formats {sorted(DECODERS)}"
        )
    return raw


def read_record(meta_path: Path) -> RecordingRecord:
    raw = read_json(meta_path)
    try:
        return DECODERS[raw["format_version"]](raw)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RecordingError(f"{meta_path} is malformed: {exc!r}") from exc


def optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def decode_v1(raw: dict[str, Any]) -> RecordingRecord:
    """Unknown fields are ignored: later versions of format 1 may add some."""
    return RecordingRecord(
        pyrpoc_version=str(raw["pyrpoc_version"]),
        program_key=str(raw["program_key"]),
        name=str(raw["name"]),
        started_at=str(raw["started_at"]),
        ended_at=optional_str(raw["ended_at"]),
        last_error=optional_str(raw["last_error"]),
        runs=[
            RunRecord(
                run_id=int(run["run_id"]),
                started_at=str(run["started_at"]),
                ended_at=optional_str(run["ended_at"]),
                error=optional_str(run["error"]),
            )
            for run in raw["runs"]
        ],
        parameters=dict(raw["parameters"]),
        devices=dict(raw["devices"]),
        outputs={
            str(output): OutputRecord(
                kind=str(entry["kind"]),
                channels=[str(label) for label in entry["channels"]],
                frames=int(entry["frames"]),
                files={str(key): str(name) for key, name in entry["files"].items()},
                metadata=dict(entry["metadata"]),
                notes=str(entry["notes"]),
            )
            for output, entry in raw["outputs"].items()
        },
    )


# Keyed by ``format_version``. Entries are added, never removed.
DECODERS: dict[int, Callable[[dict[str, Any]], RecordingRecord]] = {1: decode_v1}


def load_recording(meta_path: Path) -> list[Dataset]:
    """Every output of the recording at ``meta_path``, frames in memory."""
    record = read_record(meta_path)
    return [
        load_output(meta_path, record, output, entry) for output, entry in record.outputs.items()
    ]


def load_output(
    meta_path: Path, record: RecordingRecord, output: str, entry: OutputRecord
) -> Dataset:
    spec = DATA_KINDS.get(entry.kind)
    if spec is None:
        raise RecordingError(f"{meta_path}: output {output!r} is of unknown kind {entry.kind!r}")
    frames = read_frames(meta_path, output, entry)
    dataset = Dataset(
        output=output,
        spec=spec,
        provenance=Provenance(
            program_key=record.program_key,
            started_at=record.started_at,
            name=record.name,
            parameters=record.parameters,
            devices=record.devices,
            run_id=record.runs[0].run_id if record.runs else 0,
        ),
        origin=Origin.LOADED,
    )
    dataset.channel_labels = list(entry.channels)
    dataset.metadata = dict(entry.metadata)
    dataset.notes = entry.notes
    dataset.meta_path = meta_path
    for frame in frames:
        dataset.append(frame)
    return dataset


def read_frames(meta_path: Path, output: str, entry: OutputRecord) -> list[np.ndarray]:
    files = {key: meta_path.parent / name for key, name in entry.files.items()}
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise RecordingError(f"{meta_path}: output {output!r} is missing {', '.join(missing)}")
    if not files:
        return []
    try:
        return writer_registry.get(entry.kind).read(files)
    except (OSError, ValueError, KeyError) as exc:
        raise RecordingError(f"{meta_path}: cannot read output {output!r}: {exc}") from exc


def write_notes(meta_path: Path, output: str, notes: str) -> None:
    """Set one output's notes, the one field changed after a recording ends.
    Edited as raw JSON so fields this version does not know survive."""
    raw = read_json(meta_path)
    outputs = raw.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(output), dict):
        raise RecordingError(f"{meta_path} has no output {output!r}")
    outputs[output]["notes"] = notes
    write_json(meta_path, raw)
