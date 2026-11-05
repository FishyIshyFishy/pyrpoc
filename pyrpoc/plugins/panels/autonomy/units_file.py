"""``units.json``: the units drawn over a survey, kept in the survey's folder."""

from __future__ import annotations

import json
from pathlib import Path

from pyrpoc.plugins.program_io.inputs import UNITS, Unit

UNITS_FILE = "units.json"


def read_units(folder: Path) -> tuple[Unit, ...]:
    """The units saved in ``folder``, or none if it has no units file. Raises
    ``ParameterError`` for a file that does not hold units."""
    path = folder / UNITS_FILE
    if not path.exists():
        return ()
    return UNITS.rule.decode(json.loads(path.read_text(encoding="utf-8"))["units"])


def write_units(folder: Path, units: tuple[Unit, ...]) -> None:
    path = folder / UNITS_FILE
    path.write_text(json.dumps({"units": UNITS.rule.encode(units)}, indent=2), encoding="utf-8")
