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


@dataclass(frozen=True)
class FocusPlane:
    """Focus as a plane over stage position, ``z = z0 + tilt_x * x + tilt_y * y`` in µm, the
    spread of the points it was fitted to about it, and how many there were."""

    z0_um: float
    tilt_x: float
    tilt_y: float
    residual_um: float
    points: int

    def z_at(self, x_um: float, y_um: float) -> float:
        return self.z0_um + self.tilt_x * x_um + self.tilt_y * y_um


@dataclass(frozen=True)
class MotionRate:
    """How fast what was imaged moved, in µm per minute: the median over every feature
    followed, the spread about it, and how many there were."""

    um_per_min: float
    spread_um_per_min: float
    samples: int


@dataclass(frozen=True)
class SurveyTile:
    """One tile of a survey: where the stage was, the z it was taken at, when on the
    instrument's clock, whether it held content (and so was focused), and where stitching
    placed its top-left pixel on the dish."""

    x_um: float
    y_um: float
    z_um: float
    time_s: float
    content: bool
    left_um: float
    top_um: float


@data_kind_registry.register("SurveyResult")
@dataclass(frozen=True, eq=False)
class SurveyResult(Record):
    """What a survey found: the region and field it covered and the calibration it used, the
    focus plane, how fast things move (None with nothing to follow), and every tile."""

    left_um: float
    top_um: float
    width_um: float
    height_um: float
    field_um: float
    field_pixels: int
    um_per_volt_x: float
    um_per_volt_y: float
    focus: FocusPlane
    motion: MotionRate | None
    tiles: tuple[SurveyTile, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        motion = raw["motion"]
        return cls(
            float(raw["left_um"]),
            float(raw["top_um"]),
            float(raw["width_um"]),
            float(raw["height_um"]),
            float(raw["field_um"]),
            int(raw["field_pixels"]),
            float(raw["um_per_volt_x"]),
            float(raw["um_per_volt_y"]),
            FocusPlane(**raw["focus"]),
            None if motion is None else MotionRate(**motion),
            tuple(SurveyTile(**tile) for tile in raw["tiles"]),
        )

    def summary(self) -> str:
        content = sum(tile.content for tile in self.tiles)
        moving = "" if self.motion is None else f", {self.motion.um_per_min:.2f} µm/min"
        return f"{len(self.tiles)} tiles, {content} with content{moving}"
