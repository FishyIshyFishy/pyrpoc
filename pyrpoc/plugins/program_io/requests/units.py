"""``UNIT_COMMAND``: a change to a running loop's units, each naming its unit by key."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pyrpoc.core.program_input import Field, ParameterError
from pyrpoc.core.requests import Request
from pyrpoc.plugins.program_io.outputs import Unit


@dataclass(frozen=True)
class AddUnit:
    """Start running every routine on a unit the loop does not have."""

    word: ClassVar[str] = "add"
    unit: Unit


@dataclass(frozen=True)
class EditUnit:
    """Take a new outline or group for a unit the loop has. A new outline says where the unit
    is now."""

    word: ClassVar[str] = "edit"
    unit: Unit


@dataclass(frozen=True)
class RemoveUnit:
    """Stop running anything on a unit; what it took stays."""

    word: ClassVar[str] = "remove"
    key: str


@dataclass(frozen=True)
class PauseUnit:
    """Leave a unit alone until it is resumed."""

    word: ClassVar[str] = "pause"
    key: str


@dataclass(frozen=True)
class ResumeUnit:
    word: ClassVar[str] = "resume"
    key: str


@dataclass(frozen=True)
class RunUnitNow:
    """Run a unit's routines at once, ahead of everything due."""

    word: ClassVar[str] = "run now"
    key: str


UnitCommand = AddUnit | EditUnit | RemoveUnit | PauseUnit | ResumeUnit | RunUnitNow
# The commands that name their unit by key alone, by the word each is logged under.
KEYED: dict[str, type[RemoveUnit | PauseUnit | ResumeUnit | RunUnitNow]] = {
    cls.word: cls for cls in (RemoveUnit, PauseUnit, ResumeUnit, RunUnitNow)
}


def command_key(command: UnitCommand) -> str:
    """The key of the unit ``command`` names."""
    match command:
        case AddUnit(unit) | EditUnit(unit):
            return unit.key
        case RemoveUnit(key) | PauseUnit(key) | ResumeUnit(key) | RunUnitNow(key):
            return key


@dataclass(frozen=True)
class UnitCommandField(Field):
    """The rules for one unit command."""

    def coerce(self, value: Any) -> UnitCommand:
        if not isinstance(value, (AddUnit, EditUnit, *KEYED.values())):
            raise ParameterError(f"{self.label}: expected a unit command, not {value!r}")
        return value

    def encode(self, value: UnitCommand) -> Any:
        match value:
            case AddUnit(unit) | EditUnit(unit):
                return {"command": value.word, "unit": unit.to_dict()}
            case RemoveUnit(key) | PauseUnit(key) | ResumeUnit(key) | RunUnitNow(key):
                return {"command": value.word, "key": key}

    def decode(self, raw: Any) -> UnitCommand:
        try:
            word = raw["command"]
            if word == AddUnit.word:
                return AddUnit(Unit.from_dict(raw["unit"]))
            if word == EditUnit.word:
                return EditUnit(Unit.from_dict(raw["unit"]))
            return KEYED[word](str(raw["key"]))
        except (KeyError, TypeError) as exc:
            raise ParameterError(f"{self.label}: malformed command {raw!r}") from exc


UNIT_COMMAND = Request[UnitCommand](
    "loop_unit_command",
    UnitCommandField("Unit Command", "Add, edit, remove, pause, resume or run a unit now"),
)
