"""``GalvoCalibrationOutput``: how far the beam moves per volt on each axis, and the moves it was
measured from."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.json_file import JsonFile


@dataclass(frozen=True)
class CalibrationPoint:
    """One measured move: the axis, how far the stage moved, and how far the image moved
    in volts."""

    axis: str
    move_um: float
    shift_volts: float


@dataclass(frozen=True)
class GalvoCalibration:
    """µm per volt on each axis, signed so that a stage position plus volts times this is the
    dish position the beam is on, how well the fit held, and the channel it was measured on."""

    um_per_volt_x: float
    um_per_volt_y: float
    residual_um_x: float
    residual_um_y: float
    channel: str
    points: tuple[CalibrationPoint, ...]


@program_output_registry.register("galvo_calibration")
class GalvoCalibrationOutput(JsonFile):
    """One calibration, saved as ``<root>_<name>.json`` once it is measured."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.calibration: GalvoCalibration | None = None

    def set(self, calibration: GalvoCalibration) -> None:
        with self.lock:
            self.calibration = calibration
            self.save(asdict(calibration))
        self.changed()

    def measured(self) -> GalvoCalibration:
        """The calibration, which a finished calibration run always holds."""
        if self.calibration is None:
            raise ValueError(f"{self.name} holds no calibration: its run did not finish")
        return self.calibration

    def take(self, raw: dict) -> None:
        points = tuple(
            CalibrationPoint(str(p["axis"]), float(p["move_um"]), float(p["shift_volts"]))
            for p in raw["points"]
        )
        self.calibration = GalvoCalibration(
            float(raw["um_per_volt_x"]),
            float(raw["um_per_volt_y"]),
            float(raw["residual_um_x"]),
            float(raw["residual_um_y"]),
            str(raw["channel"]),
            points,
        )
