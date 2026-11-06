"""What a program still needs before a run of it can start, the one answer every caller asks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from pyrpoc.core.check import CheckContext
from pyrpoc.core.device import Device
from pyrpoc.core.program import Program
from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.engine.acquisition.save_target import SaveTarget
from pyrpoc.engine.data_library.earlier import EarlierOutput
from pyrpoc.engine.data_library.format import META_SUFFIX, RecordingError, recordings_in

from . import claims

OutputT = TypeVar("OutputT", bound=ProgramOutput)


class NotReady(Exception):
    """A run cannot start: the message lists what it needs."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("needs " + ", ".join(missing))


def missing(
    program: type[Program],
    inventory: list[Device],
    save: SaveTarget,
    start: StartingRun,
) -> list[str]:
    """What has to be supplied before ``program`` can run: its devices, a folder and name when
    saving, an earlier output chosen for each name it reads, and what its own checks ask for.
    Empty when ready."""
    needs = [cls.display_name for cls in claims.missing(list(program.devices), inventory)]
    if save.enabled and save.folder is None:
        needs.append("a folder to save into")
    if save.enabled and not save.filename:
        needs.append("a name to save under")
    folder = "a folder" if save.folder is None else "this folder"
    unread = [name for name in program.reads if name not in start.chosen]
    for name in unread:
        needs.append(f"a {name} ({program.reads[name].__name__}) saved in {folder}")
    # A check may look into any output the program reads, so it is judged only once all are found.
    if not unread:
        needs += check_blockers(program, start)
    return needs


def check_blockers(program: type[Program], start: StartingRun) -> list[str]:
    """What the program's own checks ask for."""
    try:
        return [need for check in program.checks for need in check.blockers(start)]
    except RecordingError as exc:
        # An earlier recording's files are a boundary: they may not load.
        return [f"an earlier recording that can be read ({exc})"]


class StartingRun(CheckContext):
    """A run about to start as its program's checks see it: the save folder, the program input
    values, and the earlier output chosen for each name it reads, loaded with ``load`` when a
    check asks for it."""

    def __init__(
        self,
        folder: Path | None,
        inputs: ProgramInputValues,
        chosen: Mapping[str, EarlierOutput],
        load: Callable[[EarlierOutput], ProgramOutput],
    ):
        self._folder = folder
        self._inputs = inputs
        self.chosen = chosen
        self.load = load

    @property
    def folder(self) -> Path | None:
        return self._folder

    def recordings_in_folder(self) -> list[str]:
        if self._folder is None:
            return []
        return [path.name.removesuffix(META_SUFFIX) for path in recordings_in(self._folder)]

    def reading(self) -> dict[str, str]:
        return {name: each.name for name, each in self.chosen.items()}

    @property
    def inputs(self) -> ProgramInputValues:
        return self._inputs

    def earlier(self, name: str, output_type: type[OutputT]) -> OutputT:
        program_output = self.load(self.chosen[name])
        # Narrows for the type checker, and catches a check asking for the wrong type.
        if not isinstance(program_output, output_type):
            raise TypeError(
                f"{name!r} is a {type(program_output).__name__}, not {output_type.__name__}"
            )
        return program_output
