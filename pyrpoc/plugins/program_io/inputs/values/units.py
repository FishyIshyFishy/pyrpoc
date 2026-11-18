"""The rules for the units a loop runs on, and their file beside a recording."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyrpoc.core.program_input import Field, ParameterError
from pyrpoc.plugins.program_io.outputs import Unit
from pyrpoc.utils.file_formats.json_file import write_json


@dataclass(frozen=True)
class UnitsField(Field):
    """The rules for the units: a tuple of ``Unit`` with distinct keys."""

    def coerce(self, value: Any) -> tuple[Unit, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of units")
        units = tuple(each if isinstance(each, Unit) else Unit.from_dict(each) for each in value)
        if len({unit.key for unit in units}) != len(units):
            raise ParameterError(f"{self.label}: two units share a key")
        return units

    def encode(self, value: tuple[Unit, ...]) -> Any:
        return [unit.to_dict() for unit in value]


class UnitsJson:
    """The units as one ``.json``: ``{"units": [...]}``, each as its ``to_dict``."""

    suffix = ".json"

    def write(self, path: Path, value: tuple[Unit, ...]) -> None:
        write_json(path, {"units": [unit.to_dict() for unit in value]})
