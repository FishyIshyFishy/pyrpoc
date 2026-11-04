"""Brightfield: grab frames from the camera and publish them like confocal ones."""

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.core.program_input import InputSection
from pyrpoc.plugins.devices import ToupcamCamera
from pyrpoc.plugins.program_io.inputs import (
    CAMERA_BINNED,
    CAMERA_EXPOSURE_MS,
    CAMERA_GAIN_PCT,
    NUM_FRAMES,
    CameraSettings,
)
from pyrpoc.plugins.program_io.outputs import Image

from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.runners import Continuous, Single


def to_grey(frame: np.ndarray) -> np.ndarray:
    """An ``(H, W, 3)`` colour frame as a one-channel ``(1, H, W)`` image."""
    # An unweighted mean, so the SDK's channel order (BGR or RGB) cannot skew it. TODO: make
    # colour readout configurable, publishing the three planes as R/G/B channels of the Image once
    # the SDK's channel order is checked.
    return frame.mean(axis=2, dtype=np.float32)[None]


@program_registry.register("brightfield")
class Brightfield(Program):
    display_name = "Brightfield"
    group = "Brightfield"
    order = 50
    devices = [ToupcamCamera]
    sections = [
        InputSection("Camera", (CAMERA_EXPOSURE_MS, CAMERA_GAIN_PCT, CAMERA_BINNED)),
        InputSection("Frames", (NUM_FRAMES,)),
    ]
    emits = {"intensity": Image}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        camera = ctx.devices[ToupcamCamera]
        settings = CameraSettings.from_inputs(ctx.inputs)
        camera.start(
            exposure_us=round(settings.exposure_ms * 1000),
            gain_pct=settings.gain_pct,
            binned=settings.binned,
        )
        try:
            for _ in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
                ctx.publish("intensity", Image(to_grey(camera.grab(ctx.sleep)), ["brightfield"]))
        finally:
            camera.stop()
