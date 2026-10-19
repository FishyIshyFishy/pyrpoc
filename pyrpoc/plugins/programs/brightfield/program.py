"""Brightfield: grab frames from the camera and publish them like confocal ones."""

from __future__ import annotations

import numpy as np

from pyrpoc.plugins.devices import ToupcamCamera
from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.parameter_groups import CameraGroup, FrameCountGroup
from ..building_blocks.runners import Continuous, Single


def to_grey(frame: np.ndarray) -> np.ndarray:
    """An ``(H, W, 3)`` colour frame as a one-channel ``(1, H, W)`` image.

    An unweighted mean, so the SDK's channel order (BGR or RGB) cannot skew it.
    TODO: make colour readout configurable, publishing the three planes as
    R/G/B channels of the Image2D once the SDK's channel order is checked.
    """
    return frame.mean(axis=2, dtype=np.float32)[None]


@program_registry.register("brightfield")
class Brightfield(Program):
    display_name = "Brightfield"
    group = "Brightfield"
    order = 50
    uses = [ToupcamCamera]
    params = [CameraGroup, FrameCountGroup]
    emits = {"intensity": Image2D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        camera = ctx.devices[ToupcamCamera]
        settings = ctx.params[CameraGroup]
        camera.start(
            exposure_us=round(settings.exposure_ms * 1000),
            gain_pct=settings.gain_pct,
            binned=settings.binned,
        )
        try:
            for _ in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
                ctx.publish("intensity", to_grey(camera.grab(ctx.sleep)), channels=["brightfield"])
        finally:
            camera.stop()
