"""``UNITS``, the regions of the dish a loop experiments on, and ``Unit``, one of them."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pyrpoc.core.program_input import Field, ParameterError, declare


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


@dataclass(frozen=True)
class UnitsField(Field):
    """The rules for the units: a tuple of ``Unit`` with distinct ids."""

    def coerce(self, value: Any) -> tuple[Unit, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of units")
        units = tuple(each if isinstance(each, Unit) else Unit.from_dict(each) for each in value)
        if len({unit.id for unit in units}) != len(units):
            raise ParameterError(f"{self.label}: two units share an id")
        return units

    def encode(self, value: tuple[Unit, ...]) -> Any:
        return [unit.to_dict() for unit in value]


UNITS = declare(
    "units",
    # Drawn and edited in the Autonomy panel, which keeps them in units.json in the survey's
    # folder: they belong to the dish, not the session.
    UnitsField("Units", "The regions drawn in the Autonomy panel", shown=False),
    (),
    persisted=False,
)
