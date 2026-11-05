"""``Unit``, one region of the dish a loop experiments on, ``UnitRecord``, a unit as a loop
knows it, and ``UnitsOutput``, every unit a loop has had."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.core.program_input import ParameterError
from pyrpoc.core.program_output import ProgramOutput, program_output_registry
from pyrpoc.core.recording import Recording

Outline = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Unit:
    """One region of the dish: a key it keeps for life and no other unit ever has, an outline in
    dish µm, so it outlasts a change of resolution, and the group it belongs to, such as
    treated or control."""

    key: str
    outline: Outline
    group: str

    @classmethod
    def new(cls, outline: Outline, group: str) -> Unit:
        """A unit never seen before, with a key of its own."""
        return cls(uuid.uuid4().hex, outline, group)

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

    def moved_to(self, x_um: float, y_um: float) -> Unit:
        """The same unit with its centre at (``x_um``, ``y_um``)."""
        cx, cy = self.centre
        return self.moved_by(x_um - cx, y_um - cy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "outline": [list(point) for point in self.outline],
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Unit:
        """A unit from JSON, a boundary: anything malformed raises ``ParameterError``."""
        try:
            outline = tuple((float(x), float(y)) for x, y in raw["outline"])
            unit = cls(str(raw["key"]), outline, str(raw["group"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterError(f"a unit is malformed: {exc}") from exc
        if len(outline) < 3:
            raise ParameterError(f"unit {unit.key} needs at least three points")
        return unit


@dataclass(frozen=True)
class UnitRecord:
    """A unit as a loop knows it: the number the loop gave it on joining, never given to another
    unit, the unit as last drawn or edited, when it joined and left on the instrument's clock,
    and whether it is paused."""

    number: int
    unit: Unit
    joined_s: float
    left_s: float | None
    paused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "unit": self.unit.to_dict(),
            "joined_s": self.joined_s,
            "left_s": self.left_s,
            "paused": self.paused,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> UnitRecord:
        left = raw["left_s"]
        return cls(
            int(raw["number"]),
            Unit.from_dict(raw["unit"]),
            float(raw["joined_s"]),
            None if left is None else float(left),
            bool(raw["paused"]),
        )


@program_output_registry.register("units")
class UnitsOutput(ProgramOutput):
    """Every unit a loop has had, those that left included, saved whole as
    ``<root>_<name>.json`` whenever one joins, is edited, leaves, or is paused or resumed."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.records: tuple[UnitRecord, ...] = ()
        self.nbytes_saved = 0

    def keep(self, records: tuple[UnitRecord, ...]) -> None:
        text = json.dumps({"units": [record.to_dict() for record in records]}, indent=2)
        with self.lock:
            self.records = records
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
            output.records = tuple(UnitRecord.from_dict(each) for each in raw["units"])
            output.nbytes_saved = path.stat().st_size
        return output
