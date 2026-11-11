"""``CalibrationLogOutput``: everything a galvo calibration did, in order, and its events, one
class for each kind. Times are on the instrument's clock."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.core.program_output import program_output_registry

from .log import LogOutput


@dataclass(frozen=True)
class FeatureFound:
    """The view was found to hold structure on ``channel``, which frames are registered on as
    the stage moves."""

    kind: ClassVar[str] = "feature"
    time_s: float
    channel: str

    def text(self) -> str:
        return f"registering the view on {self.channel}"


@dataclass(frozen=True)
class AxisMeasuring:
    """Measuring one axis, ``x`` or ``y``, began."""

    kind: ClassVar[str] = "axis"
    time_s: float
    axis: str

    def text(self) -> str:
        return f"measuring {self.axis}"


@dataclass(frozen=True)
class StageMoved:
    """The stage moved ``move_um`` along an axis and back, and the image shifted ``shift_px``;
    a rough move only sizes the real ones."""

    kind: ClassVar[str] = "moved"
    time_s: float
    axis: str
    move_um: float
    shift_px: float
    rough: bool

    def text(self) -> str:
        rough = " (sizing)" if self.rough else ""
        moved = f"moved {self.move_um:+.2f} µm, image shifted {self.shift_px:+.2f} px"
        return f"{self.axis}: {moved}{rough}"


@dataclass(frozen=True)
class AxisMeasured:
    """One axis's µm per volt, and how far the moves strayed from the fitted line."""

    kind: ClassVar[str] = "measured"
    time_s: float
    axis: str
    um_per_volt: float
    residual_um: float

    def text(self) -> str:
        residual = f"residual {self.residual_um:.2f} µm"
        return f"{self.axis}: {self.um_per_volt:.3f} µm per volt ({residual})"


@dataclass(frozen=True)
class AxisPrecision:
    """How well one axis's scale is known, as a fraction of it at one standard error, after
    ``rounds`` rounds of moves; and how far, in degrees, the image moved off the axis as the
    stage moved along it, and how many pixels that misplaces the field's edge by."""

    kind: ClassVar[str] = "precision"
    time_s: float
    axis: str
    uncertainty: float
    rounds: int
    degrees_off_axis: float
    edge_px: float

    def text(self) -> str:
        known = f"{self.axis}: scale known to ±{self.uncertainty:.2%} in {self.rounds} rounds"
        off = f"{self.degrees_off_axis:+.2f}° off the axis ({self.edge_px:.1f} px at the edge)"
        return f"{known}; the image moves {off}"


CalibrationEvent = FeatureFound | AxisMeasuring | StageMoved | AxisMeasured | AxisPrecision
CALIBRATION_EVENTS: dict[str, type[CalibrationEvent]] = {
    cls.kind: cls for cls in (FeatureFound, AxisMeasuring, StageMoved, AxisMeasured, AxisPrecision)
}


@program_output_registry.register("calibration_log")
class CalibrationLogOutput(LogOutput[CalibrationEvent]):
    """A galvo calibration's events."""

    def kinds(self) -> Mapping[str, type[CalibrationEvent]]:
        return CALIBRATION_EVENTS
