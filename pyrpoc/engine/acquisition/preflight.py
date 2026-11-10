"""``Preflight``: the one judge of whether a run can start. The panel asks it whenever something
it looks at changes, and the executor asks it again as it starts the run, so both see one list."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from pyrpoc.core.device import Device
from pyrpoc.core.program import Program
from pyrpoc.core.program_input import ParameterError, ProgramInputValues
from pyrpoc.engine.acquisition.save_target import SaveTarget
from pyrpoc.engine.data_library.format import META_SUFFIX, recordings_in
from pyrpoc.engine.data_library.store import LibraryStore

from . import claims
from .earlier import earlier_inputs
from .program_inputs import ProgramInputsStore


class NotReady(Exception):
    """A run cannot start: the message lists everything it needs."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("needs " + ", ".join(missing))


@dataclass(frozen=True)
class Start:
    """A start asked for: the program, and what the workbench holds for it now. ``unreadable``
    says why, by key, each earlier output input's newest in the folder cannot be read."""

    program_key: str
    program: Program
    inputs: ProgramInputsStore
    inventory: list[Device]
    save: SaveTarget
    unreadable: Mapping[str, str]


class Preflight:
    """One start, judged whole: its devices are there and free, it has somewhere to save, its
    input values are valid, every earlier output it reads is in the folder, the library has room,
    and the program's own ``preflight`` is satisfied. Every need is listed at once."""

    def __init__(self, start: Start, held: Collection[Device], library: LibraryStore):
        self.start = start
        self.held = held
        self.library = library

    def blockers(self) -> list[str]:
        """Everything the run still needs; empty when it may start."""
        return [
            *self.devices(),
            *self.saving(),
            *self.values(),
            *self.earlier_outputs(),
            *self.start.program.preflight(self),
            *self.room(),
        ]

    def devices(self) -> list[str]:
        required = list(self.start.program.devices)
        absent = claims.missing(required, self.start.inventory)
        if absent:
            return [cls.display_name for cls in absent]
        claimed = claims.resolve(required, self.start.inventory).values()
        return [
            f"{each.display_name} free of the run using it" for each in claimed if each in self.held
        ]

    def saving(self) -> list[str]:
        save = self.start.save
        if not save.enabled:
            return []
        needs = [] if save.folder is not None else ["a folder to save into"]
        return needs if save.filename else [*needs, "a name to save under"]

    def values(self) -> list[str]:
        """Each input whose current value its rules refuse."""
        needs: list[str] = []
        for each in self.start.program.inputs:
            try:
                each.rule.coerce(self.start.inputs.get(each.key))
            except ParameterError as exc:
                needs.append(f"a valid {each.rule.label} ({exc})")
        return needs

    def earlier_outputs(self) -> list[str]:
        """Each earlier output input with no value: none of its type in the folder, or the
        newest there cannot be read."""
        where = "a folder" if self.folder is None else "this folder"
        needs: list[str] = []
        for key, rule in earlier_inputs(self.start.program.inputs).items():
            if self.start.inputs.get(key) is not None:
                continue
            if key in self.start.unreadable:
                needs.append(f"a {rule.label} that can be read ({self.start.unreadable[key]})")
            else:
                needs.append(f"a {rule.label} saved in {where}")
        return needs

    def room(self) -> list[str]:
        if not self.library.full:
            return []
        return ["room in the data library (close recordings there, or turn on auto-purge)"]

    @property
    def folder(self) -> Path | None:
        return self.start.save.folder

    def recordings_in_folder(self) -> list[str]:
        if self.folder is None:
            return []
        return [path.name.removesuffix(META_SUFFIX) for path in recordings_in(self.folder)]

    @property
    def inputs(self) -> ProgramInputValues:
        return self.start.inputs.values_for(self.start.program.inputs)
