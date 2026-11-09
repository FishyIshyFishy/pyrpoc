"""``fill_earlier``: sets each program input that is an output of an earlier recording to the
newest of its type in the save folder."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyrpoc.core.program_input import ProgramInput, program_input_registry
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.engine.data_library.earlier import EarlierOutput, earlier_outputs
from pyrpoc.engine.data_library.format import RecordingError, load_recording
from pyrpoc.engine.data_library.store import LibraryStore
from pyrpoc.utils.value_rules import EarlierOutputField

from .program_inputs import ProgramInputsStore


def earlier_inputs(inputs: Iterable[ProgramInput[Any]]) -> dict[str, EarlierOutputField]:
    """The rule of each of ``inputs`` that is an output of an earlier recording, by key."""
    return {each.key: each.rule for each in inputs if isinstance(each.rule, EarlierOutputField)}


def fill_earlier(
    folder: Path | None, inputs: ProgramInputsStore, library: LibraryStore
) -> dict[str, str]:
    """Set every such input to the newest output of its type in ``folder``, or None when there
    is none, and return why, by key, each whose newest cannot be read."""
    rules = earlier_inputs(program_input_registry.values())
    found = [] if folder is None or not rules else earlier_outputs(folder)[0]
    unreadable: dict[str, str] = {}
    for key, rule in rules.items():
        newest = newest_of(found, rule.output_type)
        current = inputs.get(key)
        value: ProgramOutput | None = None
        if newest is not None and rule.encode(current) == reference(newest):
            # A finished recording's files never change, so the one already set stands.
            continue
        if newest is not None:
            try:
                value = load(newest, library)
            except RecordingError as exc:
                # An earlier recording's files are a boundary: they may not load.
                unreadable[key] = str(exc)
        if rule.encode(value) != rule.encode(current):
            inputs.set(key, value)
    return unreadable


def newest_of(found: list[EarlierOutput], output_type: type[ProgramOutput]) -> EarlierOutput | None:
    """The first of ``found``, which is newest first, that is an ``output_type``."""
    return next((each for each in found if issubclass(each.output_type, output_type)), None)


def reference(each: EarlierOutput) -> dict[str, str]:
    """``each`` as an earlier output's rule encodes it."""
    return {"recording": each.name, "output": each.output}


def load(each: EarlierOutput, library: LibraryStore) -> ProgramOutput:
    """``each``'s content: the library's open copy when it holds one, else read from its files.
    Raises ``RecordingError`` for files that cannot be read."""
    for recording in library.recordings():
        if recording.location == each.meta_path:
            return recording.outputs[each.output]
    # Loaded copies stay out of the library: reading one must not take a place there.
    return load_recording(each.meta_path).outputs[each.output]
