"""``Unit``, one region of the dish a loop experiments on, and ``UnitsOutput``, the units a
loop ran on."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.core.program_input import ParameterError
from pyrpoc.core.program_output import ProgramOutput, program_output_registry
from pyrpoc.core.recording import Recording


@dataclass(frozen=True)
class Unit:
    """One region of the dish: an id, an outline in dish µm, so it outlasts a change of
    resolution, and the group it belongs to, such as treated or control."""

    id: int
    outline: tuple[tuple[float, float], ...]
    group: str

    @property
    def centre(self) -> tuple[float, float]:
        """The outline's centre of area."""
        points = np.asarray(self.outline)
        x, y = points[:, 0], points[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2.0
        if abs(area) < 1e-12:
            return float(x.mean()), float(y.mean())
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * area)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * area)
        return float(cx), float(cy)

    def bounds(self) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) in dish µm."""
        points = np.asarray(self.outline)
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )

    def moved_by(self, dx_um: float, dy_um: float) -> Unit:
        return replace(self, outline=tuple((x + dx_um, y + dy_um) for x, y in self.outline))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "outline": [list(point) for point in self.outline],
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Unit:
        """A unit from JSON, a boundary: anything malformed raises ``ParameterError``."""
        try:
            outline = tuple((float(x), float(y)) for x, y in raw["outline"])
            unit = cls(int(raw["id"]), outline, str(raw["group"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterError(f"a unit is malformed: {exc}") from exc
        if len(outline) < 3:
            raise ParameterError(f"unit {unit.id} needs at least three points")
        return unit


@program_output_registry.register("units")
class UnitsOutput(ProgramOutput):
    """Every unit a loop ran on, as drawn when it started on it, saved as
    ``<root>_<name>.json`` each time one is added, so the units outlast the panel they were
    drawn in."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.units: tuple[Unit, ...] = ()
        self.nbytes_saved = 0

    def keep(self, units: tuple[Unit, ...]) -> None:
        text = json.dumps({"units": [unit.to_dict() for unit in units]}, indent=2)
        with self.lock:
            self.units = units
            self.nbytes_saved = len(text)
            if self.root is not None:
                path = self.root.with_name(f"{self.root.name}_{self.name}.json")
                path.write_text(text, encoding="utf-8")
                self.paths = {self.name: path}
        self.changed()

    @property
    def nbytes(self) -> int:
        return self.nbytes_saved

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> UnitsOutput:
        output = cls(name=name, recording=recording, root=None)
        if files:
            (path,) = files.values()
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
            output.units = tuple(Unit.from_dict(each) for each in raw["units"])
            output.nbytes_saved = path.stat().st_size
        return output
