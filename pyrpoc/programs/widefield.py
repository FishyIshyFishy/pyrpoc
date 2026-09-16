"""Widefield: stream frames off the ZWO camera, no galvo or DAQ involved.

The clearest single-device program in the file, structured the same way FLIM
sets up and tears down the TimeTagger around its loop: the camera is opened
and configured once before ``ctx.frames``, not per frame, and closed in a
``finally`` so a stopped run still lets go of the USB handle and turns the
cooler off.
"""

from __future__ import annotations

import numpy as np

from pyrpoc.core.streams import Image2D
from pyrpoc.devices import ZWOCamera
from pyrpoc.run.program import Program

from .components import WidefieldGroup
from .registry import program_registry


@program_registry.register("widefield")
class Widefield(Program):
    uses = [ZWOCamera]
    params = [WidefieldGroup]
    emits = {"intensity": Image2D}

    def run(self, ctx) -> None:
        widefield = ctx.params[WidefieldGroup]
        camera: ZWOCamera = ctx.devices[ZWOCamera]

        ctx.status("opening camera")
        camera.open()
        try:
            camera.configure(
                width=widefield.width,
                height=widefield.height,
                binning=int(widefield.binning),
                bit_depth=int(widefield.bit_depth),
                exposure_s=widefield.exposure_s,
                gain=widefield.gain,
            )
            camera.start_capture()

            total = "" if ctx.continuous else f"/{widefield.num_frames}"
            for index in ctx.frames(widefield.num_frames):
                ctx.status(f"frame {index + 1}{total}")
                frame = camera.get_frame()
                if widefield.auto_gain:
                    camera.auto_adjust_gain(frame)
                ctx.publish("intensity", frame[np.newaxis, :, :].astype(np.float32), channels=["mono"])
                ctx.sleep(max(0.0, widefield.interval_s - widefield.exposure_s))
        finally:
            camera.stop_capture()
            camera.close()
