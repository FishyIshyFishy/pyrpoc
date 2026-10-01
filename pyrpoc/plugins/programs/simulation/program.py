"""Simulation: frames out of thin air, no instruments required.

Everything above the hardware boundary is real (the executor's thread, datasets,
publishing, saving, the panels), so this exercises the software itself on any
machine. A plane is a function of (seed, channel, frame index). Each run draws
its own seed, so repeated runs show a different sample rather than a replay.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np

from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.functions.run_loops import count_off
from ..building_blocks.functions.synthetic_detector import sim_labels, with_noise
from ..building_blocks.parameter_groups import (
    FrameCountGroup,
    FrameGroup,
    Mask,
    ModulationGroup,
    PacingGroup,
    SignalGroup,
)
from ..building_blocks.runners import Continuous, Single
from .planes import PLANES


def lit_pixels(masks: Sequence[Mask], frame_shape: FrameGroup) -> np.ndarray | None:
    """Every bound mask on the frame, OR-ed into one plane, or None for no
    masks. Ports and lines are ignored: there are no digital lines here, but
    which pixels are illuminated can still be checked."""
    planes = [mask.on_grid(frame_shape.y_pixels, frame_shape.x_pixels) for mask in masks]
    return reduce(np.logical_or, planes) if planes else None


def synthetic_frame(
    frame_shape: FrameGroup,
    signal: SignalGroup,
    *,
    seed: int,
    frame_index: int,
    lit: np.ndarray | None,
) -> np.ndarray:
    """One ``(C, H, W)`` float32 frame, as a real scan would have returned it.
    Pixels under a mask are brightened by ``mask_gain``, standing in for the
    stimulation the TTL lines would drive."""
    plane_of = PLANES[signal.pattern]
    frame = np.stack(
        [
            plane_of(
                frame_shape.y_pixels,
                frame_shape.x_pixels,
                channel=channel,
                seed=seed,
                drift=signal.drift_pixels_per_frame,
                frame_index=frame_index,
            )
            for channel in range(frame_shape.channels)
        ]
    ).astype(np.float32)
    frame *= signal.signal_level
    if lit is not None and signal.mask_gain:
        frame = frame * (1.0 + signal.mask_gain * lit.astype(np.float32))
    return with_noise(frame, signal.noise_level, seed=seed, index=frame_index)


@program_registry.register("simulation")
class Simulation(Program):
    display_name = "2D image"
    group = "Simulations"
    order = 20
    uses = []
    params = [FrameGroup, FrameCountGroup, SignalGroup, ModulationGroup, PacingGroup]
    emits = {"intensity": Image2D}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        frame_shape = ctx.params[FrameGroup]
        signal = ctx.params[SignalGroup]
        interval_ms = ctx.params[PacingGroup].frame_interval_ms
        lit = lit_pixels(ctx.params[ModulationGroup].masks, frame_shape)
        labels = sim_labels(frame_shape.channels)
        # Fixed for the run so its frames stay one drifting sample.
        seed = int(np.random.default_rng().integers(2**32))

        for index in count_off(ctx, ctx.params[FrameCountGroup].num_frames, "frame"):
            frame = synthetic_frame(frame_shape, signal, seed=seed, frame_index=index, lit=lit)
            ctx.publish("intensity", frame, channels=labels)
            ctx.sleep(interval_ms / 1000.0)
