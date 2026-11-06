"""What a program still needs before a run of it can start, the one answer every caller asks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pyrpoc.core.check import CheckContext
from pyrpoc.core.device import Device
from pyrpoc.core.program import Program
from pyrpoc.engine.acquisition.save_target import SaveTarget
from pyrpoc.engine.data_library.earlier import EarlierOutput
from pyrpoc.engine.data_library.format import META_SUFFIX, recordings_in

from . import claims


class NotReady(Exception):
    """A run cannot start: the message lists what it needs."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("needs " + ", ".join(missing))


def missing(
    program: type[Program],
    inventory: list[Device],
    save: SaveTarget,
    earlier: Mapping[str, EarlierOutput],
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
    for name in program.reads:
        if name not in earlier:
            needs.append(f"a {name} saved in {folder}")
    context = StartingRun(save.folder, earlier)
    for check in program.checks:
        needs += check.blockers(context)
    return needs


class StartingRun(CheckContext):
    """The save folder and the reads chosen, as a program's checks see them."""

    def __init__(self, folder: Path | None, earlier: Mapping[str, EarlierOutput]):
        self._folder = folder
        self._earlier = earlier

    @property
    def folder(self) -> Path | None:
        return self._folder

    def recordings_in_folder(self) -> list[str]:
        if self._folder is None:
            return []
        return [path.name.removesuffix(META_SUFFIX) for path in recordings_in(self._folder)]

    def reading(self) -> dict[str, str]:
        return {name: each.name for name, each in self._earlier.items()}
