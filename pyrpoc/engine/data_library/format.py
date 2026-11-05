"""The recording format: a ``<stem>_meta.json`` that names its data files, and
how it is written and read."""

from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pyrpoc.core.program_output import ProgramOutput, program_output_registry
from pyrpoc.core.recording import EarlierRead, Recording

FORMAT = "pyrpoc-recording"
FORMAT_VERSION = 8
META_SUFFIX = "_meta.json"


class RecordingError(Exception):
    """A recording on disk that cannot be read. The message names the file."""


@dataclass(frozen=True)
class LiveRead:
    """The run reading a live input whose value changed since it last read it."""

    read_at: str
    key: str
    # As the input's rules encode it.
    value: Any


@dataclass
class OutputRecord:
    # The output's ``type_key`` in ``program_output_registry``.
    type: str
    # File names beside the metadata file, keyed as the output keyed them.
    files: dict[str, str]


@dataclass
class RecordingRecord:
    pyrpoc_version: str
    program_key: str
    name: str
    started_at: str
    ended_at: str | None
    last_error: str | None
    parameters: dict[str, Any]
    devices: dict[str, Any]
    outputs: dict[str, OutputRecord]
    # What the user wrote about the run, before or after it.
    notes: str
    # What it read of earlier recordings, by the name its program reads each under.
    inputs: dict[str, EarlierRead]
    # Each change to a live input a run read, in the order read.
    live_reads: list[LiveRead]


def pyrpoc_version() -> str:
    return importlib.metadata.version("pyrpoc")


def meta_path_for(root: Path) -> Path:
    return root.with_name(f"{root.name}{META_SUFFIX}")


def recordings_in(folder: Path) -> list[Path]:
    """The metadata file of every recording in ``folder``, by name."""
    return sorted(folder.glob(f"*{META_SUFFIX}"))


def free_root(root: Path) -> Path:
    """``root``, or the first of ``root_2``, ``root_3`` and so on that has no recording
    yet."""
    candidate, number = root, 1
    while meta_path_for(candidate).exists():
        number += 1
        candidate = root.with_name(f"{root.name}_{number}")
    return candidate


def write_record(meta_path: Path, record: RecordingRecord) -> None:
    write_json(meta_path, {"format": FORMAT, "format_version": FORMAT_VERSION, **asdict(record)})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, default=str))


def write_text(path: Path, text: str) -> None:
    """Replace ``path`` whole."""
    # So a crash mid-write never leaves half a file.
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(text, encoding="utf-8")
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
    if version != FORMAT_VERSION:
        raise RecordingError(
            f"{meta_path} is recording format {version!r}; this version of pyrpoc reads "
            f"format {FORMAT_VERSION}"
        )
    return raw


def read_record(meta_path: Path) -> RecordingRecord:
    raw = read_json(meta_path)
    try:
        return decode_record(raw)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RecordingError(f"{meta_path} is malformed: {exc!r}") from exc


def optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def decode_record(raw: dict[str, Any]) -> RecordingRecord:
    """The recording a metadata file's JSON describes."""
    # Unknown fields are ignored, so a format may add some.
    return RecordingRecord(
        pyrpoc_version=str(raw["pyrpoc_version"]),
        program_key=str(raw["program_key"]),
        name=str(raw["name"]),
        started_at=str(raw["started_at"]),
        ended_at=optional_str(raw["ended_at"]),
        last_error=optional_str(raw["last_error"]),
        parameters=dict(raw["parameters"]),
        devices=dict(raw["devices"]),
        outputs={
            str(output): OutputRecord(
                type=str(entry["type"]),
                files={str(key): str(name) for key, name in entry["files"].items()},
            )
            for output, entry in raw["outputs"].items()
        },
        notes=str(raw["notes"]),
        inputs={
            str(name): EarlierRead(str(read["recording"]), str(read["output"]))
            for name, read in raw["inputs"].items()
        },
        live_reads=[
            LiveRead(
                read_at=str(read["read_at"]),
                key=str(read["key"]),
                value=read["value"],
            )
            for read in raw["live_reads"]
        ],
    )


def load_recording(meta_path: Path) -> Recording:
    """The recording at ``meta_path``, each output holding all it was given."""
    record = read_record(meta_path)
    recording = Recording(
        program_key=record.program_key,
        name=record.name,
        started_at=record.started_at,
        parameters=record.parameters,
        devices=record.devices,
        reads=record.inputs,
        location=meta_path,
        note=record.notes,
        finished=True,
    )
    recording.outputs = {
        output: load_output(recording, meta_path, output, entry)
        for output, entry in record.outputs.items()
    }
    return recording


def load_output(
    recording: Recording, meta_path: Path, output: str, entry: OutputRecord
) -> ProgramOutput:
    if entry.type not in program_output_registry.keys():
        raise RecordingError(f"{meta_path}: output {output!r} is of unknown type {entry.type!r}")
    files = {key: meta_path.parent / name for key, name in entry.files.items()}
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise RecordingError(f"{meta_path}: output {output!r} is missing {', '.join(missing)}")
    try:
        return program_output_registry.get(entry.type).load(
            name=output, recording=recording, files=files
        )
    except (OSError, ValueError, KeyError) as exc:
        raise RecordingError(f"{meta_path}: cannot read output {output!r}: {exc}") from exc
