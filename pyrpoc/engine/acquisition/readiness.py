"""What a program still needs before a run of it can start, the one answer every caller asks."""

from __future__ import annotations

from collections.abc import Mapping

from pyrpoc.core.device import Device
from pyrpoc.core.program import Program
from pyrpoc.engine.acquisition.save_target import SaveTarget
from pyrpoc.engine.data_library.earlier import EarlierOutput

from . import claims


class NotReady(Exception):
    """A run cannot start: the message lists what it needs."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("needs " + ", ".join(missing))


def missing(
    program: type[Program],
    inventory: list[Device],
    chosen: Mapping[str, Device],
    save: SaveTarget,
    earlier: Mapping[str, EarlierOutput],
) -> list[str]:
    """What has to be supplied before ``program`` can run: its devices, a device chosen for
    each name it chooses, a folder and name when saving, and an earlier output chosen for each
    name it reads. Empty when ready."""
    required = claims.with_backing(list(program.devices), chosen)
    needs = [cls.display_name for cls in claims.missing(required, inventory)]
    for name, device_type in program.chooses.items():
        if name not in chosen:
            needs.append(f"a {device_type.display_name} for {name}")
    if save.enabled and save.folder is None:
        needs.append("a folder to save into")
    if save.enabled and not save.filename:
        needs.append("a name to save under")
    folder = "a folder" if save.folder is None else "this folder"
    for name, output_type in program.reads.items():
        if name not in earlier:
            needs.append(f"a {name} ({output_type.__name__}) saved in {folder}")
    return needs
