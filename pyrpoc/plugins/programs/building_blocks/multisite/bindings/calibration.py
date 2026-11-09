"""The calibration stage bound to a run: its outputs, and the call a program makes."""

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import ScanGeometry
from pyrpoc.plugins.program_io.outputs import (
    CalibrationEvent,
    CalibrationLogOutput,
    GalvoCalibrationOutput,
    ImageOutput,
)

from ..interfaces.microscope import Microscope
from ..stages.calibration.measuring import calibrate

# What every calibration fills: the result, every frame it looked at, and what it did.
CALIBRATION_EMITS: dict[str, type[ProgramOutput]] = {
    "calibration": GalvoCalibrationOutput,
    "view": ImageOutput,
    "log": CalibrationLogOutput,
}


class CalibrationOutputs:
    """A run's ``view`` and ``log`` outputs, as the calibration's sink. Each event is also the
    run's status."""

    def __init__(self, ctx: RunContext, channels: list[str]):
        self.ctx = ctx
        self.channels = channels
        self.view = ctx.output("view", ImageOutput)
        self.events = ctx.output("log", CalibrationLogOutput)

    def show(self, frame: np.ndarray) -> None:
        self.view.add(frame, self.channels)

    def log(self, event: CalibrationEvent) -> None:
        self.events.log(event)
        self.ctx.status(event.text())


def run_calibration(ctx: RunContext, scope: Microscope, scan: ScanGeometry) -> None:
    """Measure the scanner's µm per volt with ``scan``."""
    result = calibrate(scope, scan, CalibrationOutputs(ctx, scope.channel_labels))
    ctx.output("calibration", GalvoCalibrationOutput).set(result)
