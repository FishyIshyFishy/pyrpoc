"""``EarlierChoices``: the recordings in the save folder that programs read, and which one
each reads."""

from __future__ import annotations

from pathlib import Path

from pyrpoc.core.program import program_registry
from pyrpoc.engine.data_library.earlier import EarlierRecording, earlier_recordings


def read_program_keys() -> set[str]:
    """Every program some registered program reads an earlier run of."""
    return {each.program_key for cls in program_registry.entries.values() for each in cls.reads}


class EarlierChoices:
    """The finished recordings in one folder of each program that others read, and the one
    picked for each. One pick per program key, shared by every program that reads it."""

    def __init__(self) -> None:
        self.found: dict[str, list[EarlierRecording]] = {}
        # Picked by the user; a key with no pick reads the latest.
        self.picked: dict[str, Path] = {}

    def refresh(self, folder: Path | None) -> None:
        """Look again at what ``folder`` holds."""
        keys = read_program_keys()
        self.found = {} if folder is None or not keys else earlier_recordings(folder, keys)
        # A pick that left the folder, or a folder that changed, goes back to the latest.
        held = {each.meta_path for recordings in self.found.values() for each in recordings}
        self.picked = {key: path for key, path in self.picked.items() if path in held}

    def options(self, program_key: str) -> list[EarlierRecording]:
        """The recordings of ``program_key`` to choose from, newest first."""
        return self.found.get(program_key, [])

    def picked_path(self, program_key: str) -> Path | None:
        """The recording the user picked for ``program_key``, or None to read the latest."""
        return self.picked.get(program_key)

    def pick(self, program_key: str, meta_path: Path | None) -> None:
        if meta_path is None:
            self.picked.pop(program_key, None)
        else:
            self.picked[program_key] = meta_path

    def chosen(self, program_key: str) -> Path | None:
        """What a run of a program reading ``program_key`` reads, or None when there is
        nothing to read."""
        options = self.options(program_key)
        if not options:
            return None
        return self.picked.get(program_key, options[0].meta_path)
