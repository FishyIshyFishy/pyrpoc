"""What a program still needs before a run of it can start, the one answer every caller asks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pyrpoc.core.device import Device
from pyrpoc.core.program import Program
from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.data_library.grouping import program_display_name

from . import claims


class NotReady(Exception):
    """A run cannot start: the message lists what it needs."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("needs " + ", ".join(missing))


def missing(
    program: type[Program], inventory: list[Device], save: SaveTarget, earlier: Mapping[str, Path]
) -> list[str]:
    """What has to be supplied before ``program`` can run: its devices, a folder and name when
    saving, and the earlier recording chosen for each it reads. Empty when ready."""
    needs = [cls.display_name for cls in claims.missing(list(program.devices), inventory)]
    if save.enabled and save.folder is None:
        needs.append("a folder to save into")
    if save.enabled and not save.filename:
        needs.append("a name to save under")
    for each in program.reads:
        if each.program_key not in earlier:
            name = program_display_name(each.program_key)
            folder = "a folder" if save.folder is None else "this folder"
            needs.append(f"{name} saved in {folder}")
    return needs
