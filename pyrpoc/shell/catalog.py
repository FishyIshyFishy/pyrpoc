"""Ways to launch a program: label and grouping.

Presentation data, curated by hand, because what belongs in a dropdown is a
design decision. Keeping labels here rather than on the program is what lets one
program be offered more than once later, and keeps Program from growing
presentation fields.

Adding an experiment is one file in programs/ plus one row here. Deleting one is
deleting that file and that row.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.programs.confocal import Confocal
from pyrpoc.programs.flim import FLIM
from pyrpoc.programs.simulation import Simulation
from pyrpoc.programs.split_confocal import SplitConfocal
from pyrpoc.run.program import Program


@dataclass(frozen=True)
class Entry:
    program: type[Program]
    key: str
    label: str
    group: str = "Imaging"


CATALOG: list[Entry] = [
    Entry(Confocal, "confocal", "Confocal"),
    Entry(SplitConfocal, "split_confocal", "Split Confocal"),
    Entry(FLIM, "flim", "FLIM"),
    Entry(Simulation, "simulation", "Simulation", group="Testing"),
]


def entry_for(key: str) -> Entry:
    for entry in CATALOG:
        if entry.key == key:
            return entry
    raise KeyError(f"no catalog entry for {key!r}")


def keys() -> list[str]:
    return [entry.key for entry in CATALOG]
