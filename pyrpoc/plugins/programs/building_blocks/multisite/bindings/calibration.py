"""The calibration stage bound to a run: its outputs, and the call a program makes."""

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import ScanGeometry
from pyrpoc.plugins.program_io.outputs import GalvoCalibrationOutput, ImageOutput

from ..interfaces.microscope import Microscope
from ..stages.calibration.measuring import calibrate

# What every calibration fills: the result, and every frame it looked at.
CALIBRATION_EMITS: dict[str, type[ProgramOutput]] = {
    "calibration": GalvoCalibrationOutput,
    "view": ImageOutput,
}


def run_calibration(ctx: RunContext, scope: Microscope, scan: ScanGeometry) -> None:
    """Measure the scanner's µm per volt with ``scan``."""
    view = ctx.output("view", ImageOutput)

    def show(frame: np.ndarray) -> None:
        view.add(frame, scope.channel_labels)

    result = calibrate(scope, ctx, scan, show)
    ctx.output("calibration", GalvoCalibrationOutput).set(result)
    ctx.status(f"{result.um_per_volt_x:.3f} × {result.um_per_volt_y:.3f} µm per volt")
