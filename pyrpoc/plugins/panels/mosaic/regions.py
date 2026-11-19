"""The regions marked on a mosaic: squares one field across on the dish, kept by the panel to
come back to later. No Qt."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Region:
    """A square ``side_um`` across, centred at (``x_um``, ``y_um``), the stage position that
    puts it in the middle of the field. ``number`` is its name, never given twice."""

    number: int
    x_um: float
    y_um: float
    side_um: float

    def holds(self, x_um: float, y_um: float) -> bool:
        half = self.side_um / 2.0
        return abs(x_um - self.x_um) <= half and abs(y_um - self.y_um) <= half


def bounds_middle(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The centre of the points' bounds."""
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def span(points: Sequence[tuple[float, float]]) -> float:
    """The longer side of the points' bounds."""
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


class Regions:
    """The regions in the order they were marked, the newest on top."""

    def __init__(self) -> None:
        self.marked: list[Region] = []
        self.next_number = 1

    def add(self, x_um: float, y_um: float, side_um: float) -> Region:
        region = Region(self.next_number, x_um, y_um, side_um)
        self.next_number += 1
        self.marked.append(region)
        return region

    def remove(self, number: int) -> None:
        self.marked = [region for region in self.marked if region.number != number]

    def get(self, number: int) -> Region:
        return next(region for region in self.marked if region.number == number)

    def at(self, x_um: float, y_um: float) -> Region | None:
        """The newest region holding the point, or None."""
        for region in reversed(self.marked):
            if region.holds(x_um, y_um):
                return region
        return None

    def export_state(self) -> dict[str, Any]:
        return {
            "next_number": self.next_number,
            "regions": [
                {"number": r.number, "x_um": r.x_um, "y_um": r.y_um, "side_um": r.side_um}
                for r in self.marked
            ],
        }

    def import_state(self, raw: Any) -> None:
        """Take saved regions from session JSON, a boundary: a malformed entry is skipped."""
        if not isinstance(raw, dict):
            return
        entries = raw.get("regions")
        self.marked = [
            Region(int(e["number"]), float(e["x_um"]), float(e["y_um"]), float(e["side_um"]))
            for e in (entries if isinstance(entries, list) else [])
            if isinstance(e, dict) and {"number", "x_um", "y_um", "side_um"} <= e.keys()
        ]
        highest = max((region.number for region in self.marked), default=0)
        saved = raw.get("next_number")
        self.next_number = max(highest + 1, saved if isinstance(saved, int) else 1)
