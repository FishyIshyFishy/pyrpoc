"""``EarlierChoices``: the earlier outputs in the save folder that programs can read, which one
each read name reads, and each one's content once loaded."""

from __future__ import annotations

from pathlib import Path

from pyrpoc.core.program import program_registry
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.engine.data_library.earlier import EarlierOutput, Unreadable, earlier_outputs
from pyrpoc.engine.data_library.format import load_recording
from pyrpoc.engine.data_library.store import LibraryStore


def any_reads() -> bool:
    """Whether some registered program reads an earlier output."""
    return any(cls.reads for cls in program_registry.entries.values())


class EarlierChoices:
    """The earlier outputs in one folder, the recordings there that cannot be read, the one
    picked for each read name, and those loaded so far. A pick is shared by every program
    reading that name."""

    def __init__(self, library: LibraryStore) -> None:
        self.library = library
        self.found: list[EarlierOutput] = []
        self.unreadable: list[Unreadable] = []
        # Picked by the user; a name with no pick reads the newest.
        self.picked: dict[str, EarlierOutput] = {}
        # Kept, since preflight judges a start again on every edit and a finished recording's files
        # never change.
        self.loaded: dict[EarlierOutput, ProgramOutput] = {}

    def refresh(self, folder: Path | None) -> None:
        """Look again at what ``folder`` holds."""
        if folder is None or not any_reads():
            self.found, self.unreadable = [], []
        else:
            self.found, self.unreadable = earlier_outputs(folder)
        # A pick that left the folder, or a folder that changed, goes back to the latest.
        self.picked = {name: each for name, each in self.picked.items() if each in self.found}
        self.loaded = {each: output for each, output in self.loaded.items() if each in self.found}

    def options(self, output_type: type[ProgramOutput]) -> list[EarlierOutput]:
        """The outputs that are an ``output_type``, its daughter types included, newest
        first."""
        return [each for each in self.found if issubclass(each.output_type, output_type)]

    def picked_for(self, name: str, output_type: type[ProgramOutput]) -> EarlierOutput | None:
        """The output the user picked for ``name``, or None when there is no pick."""
        picked = self.picked.get(name)
        return picked if picked is not None and picked in self.options(output_type) else None

    def pick(self, name: str, choice: EarlierOutput) -> None:
        self.picked[name] = choice

    def chosen(self, name: str, output_type: type[ProgramOutput]) -> EarlierOutput | None:
        """What a run reads for ``name``: the pick, else the newest, or None when there is
        nothing to read."""
        options = self.options(output_type)
        if not options:
            return None
        return self.picked_for(name, output_type) or options[0]

    def load(self, choice: EarlierOutput) -> ProgramOutput:
        """``choice``'s content: the library's open copy when it holds one, else read from its
        files. Raises ``RecordingError`` for files that cannot be read."""
        if choice not in self.loaded:
            open_copy = self.open_copy(choice)
            self.loaded[choice] = self.read(choice) if open_copy is None else open_copy
        return self.loaded[choice]

    def open_copy(self, choice: EarlierOutput) -> ProgramOutput | None:
        for recording in self.library.recordings():
            if recording.location == choice.meta_path:
                return recording.outputs[choice.output]
        return None

    def read(self, choice: EarlierOutput) -> ProgramOutput:
        # Loaded copies stay out of the library: reading one must not take a place there.
        return load_recording(choice.meta_path).outputs[choice.output]
