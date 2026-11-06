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
    """A clear feature was found in view, on ``channel``, to follow as the stage moves."""

    kind: ClassVar[str] = "feature"
    time_s: float
    channel: str

    def text(self) -> str:
        return f"following a clear feature on {self.channel}"


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


CalibrationEvent = FeatureFound | AxisMeasuring | StageMoved | AxisMeasured
CALIBRATION_EVENTS: dict[str, type[CalibrationEvent]] = {
    cls.kind: cls for cls in (FeatureFound, AxisMeasuring, StageMoved, AxisMeasured)
}


@program_output_registry.register("calibration_log")
class CalibrationLogOutput(LogOutput[CalibrationEvent]):
    """A galvo calibration's events."""

    def kinds(self) -> Mapping[str, type[CalibrationEvent]]:
        return CALIBRATION_EVENTS
