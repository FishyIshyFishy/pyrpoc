"""Small results saved as JSON: the base kind, and the galvo calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Self

from pyrpoc.core.program_output import Data, data_kind_registry


class Record(Data):
    """A small result, saved whole as JSON."""

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        return len(json.dumps(self.to_json()))


@dataclass(frozen=True)
class CalibrationPoint:
    """One measured move: the axis, how far the stage moved, and how far the image moved
    in volts."""

    axis: str
    move_um: float
    shift_volts: float


@data_kind_registry.register("GalvoCalibration")
@dataclass(frozen=True, eq=False)
class GalvoCalibration(Record):
    """How far the beam moves per volt on each axis, signed so that a stage position plus
    volts times this is the dish position the beam is on, and how well the fit held."""

    um_per_volt_x: float
    um_per_volt_y: float
    residual_um_x: float
    residual_um_y: float
    channel: str
    points: tuple[CalibrationPoint, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        points = tuple(
            CalibrationPoint(str(p["axis"]), float(p["move_um"]), float(p["shift_volts"]))
            for p in raw["points"]
        )
        return cls(
            float(raw["um_per_volt_x"]),
            float(raw["um_per_volt_y"]),
            float(raw["residual_um_x"]),
            float(raw["residual_um_y"]),
            str(raw["channel"]),
            points,
        )

    def summary(self) -> str:
        return f"{self.um_per_volt_x:.2f} × {self.um_per_volt_y:.2f} µm/V"
