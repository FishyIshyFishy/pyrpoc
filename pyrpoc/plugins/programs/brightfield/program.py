"""Brightfield: grab frames from the camera and add them to an image output like confocal ones."""

from __future__ import annotations

import numpy as np

from pyrpoc.core.program import Program, RunContext, program_registry
from pyrpoc.plugins.devices import ToupcamCamera
from pyrpoc.plugins.program_io.inputs import (
    CAMERA_BINNED,
    CAMERA_EXPOSURE_MS,
    CAMERA_GAIN_PCT,
    NUM_FRAMES,
    CameraSettings,
)
from pyrpoc.plugins.program_io.outputs import ImageOutput

from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.runners import Continuous, Single


def to_grey(frame: np.ndarray) -> np.ndarray:
    """An ``(H, W, 3)`` colour frame as a one-channel ``(1, H, W)`` image."""
    # An unweighted mean, so the SDK's channel order (BGR or RGB) cannot skew it. TODO: make
    # colour readout configurable, adding the three planes as R/G/B channels of the image once
    # the SDK's channel order is checked.
    return frame.mean(axis=2, dtype=np.float32)[None]


@program_registry.register("brightfield")
class Brightfield(Program):
    display_name = "Brightfield"
    group = "Developmental"
    order = 230
    devices = [ToupcamCamera]
    inputs = [CAMERA_EXPOSURE_MS, CAMERA_GAIN_PCT, CAMERA_BINNED, NUM_FRAMES]
    cards = {
        "Camera": (
            CAMERA_EXPOSURE_MS,
            CAMERA_GAIN_PCT,
            CAMERA_BINNED,
        ),
        "Frames": (NUM_FRAMES,),
    }
    emits = {"intensity": ImageOutput}
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
                frame = to_grey(camera.grab(ctx.sleep))
                ctx.output("intensity", ImageOutput).add(frame, ["brightfield"])
        finally:
            camera.stop()
