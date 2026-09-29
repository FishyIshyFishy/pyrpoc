"""The programs offered in the dropdown, with their labels and grouping.

Curated by hand, since what belongs in a dropdown is a design decision, and
kept off the program so a program never grows presentation fields. Adding an
experiment is one file in programs/ plus one row here.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.src.programs.confocal import Confocal
from pyrpoc.src.programs.flim import FLIM
from pyrpoc.src.programs.pinpoint_raman import PinpointRaman
from pyrpoc.src.programs.simulation import Simulation
from pyrpoc.src.programs.split_confocal import SplitConfocal
from pyrpoc.src.structs.program import Program


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
    Entry(PinpointRaman, "pinpoint_raman", "Pinpoint Raman", group="Spectroscopy"),
    Entry(Simulation, "simulation", "Simulation", group="Testing"),
]


def entry_for(key: str) -> Entry:
    for entry in CATALOG:
        if entry.key == key:
            return entry
    raise KeyError(f"no catalog entry for {key!r}")


def keys() -> list[str]:
    return [entry.key for entry in CATALOG]
