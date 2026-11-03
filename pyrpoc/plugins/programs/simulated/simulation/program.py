"""Simulation: frames made up from a seed, with no instruments."""

# Everything above the hardware is real (the executor's thread, program outputs, saving, the
# panels), so this exercises the software on any machine. Each run draws its own seed, so repeated
# runs show a different sample rather than a replay.

from __future__ import annotations

from functools import reduce

import numpy as np

from pyrpoc.core.data import Image
from pyrpoc.core.input_section import InputSection
from pyrpoc.core.mask import MASKS, Mask
from pyrpoc.core.program import Program, RunContext

from ...building_blocks.functions.run_loops import count_off
from ...building_blocks.program_inputs import (
    MASK_LINES,
    NUM_FRAMES,
    X_PIXELS,
    Y_PIXELS,
)
from ...building_blocks.runners import Continuous, Single
from ..frame import SIMULATED_CHANNELS, FrameShape
from ..pacing import FRAME_INTERVAL_MS
from ..signal import (
    DRIFT_PIXELS_PER_FRAME,
    MASK_GAIN,
    SIGNAL_LEVEL,
    SIGNAL_PATTERN,
    SyntheticSignal,
)
from ..synthetic_detector import NOISE_LEVEL, sim_labels, with_noise
from .planes import PLANES


def lit_pixels(masks: tuple[Mask, ...], frame_shape: FrameShape) -> np.ndarray | None:
    """Every drawn mask on the frame, OR-ed into one plane, or None for no masks."""
    # Ports and lines are ignored: there are no digital lines here, but which pixels are
    # illuminated can still be checked.
    planes = [mask.on_grid(frame_shape.y_pixels, frame_shape.x_pixels) for mask in masks]
    return reduce(np.logical_or, planes) if planes else None


def synthetic_frame(
    frame_shape: FrameShape,
    signal: SyntheticSignal,
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


class Simulation(Program):
    display_name = "2D image"
    group = "Simulations"
    order = 20
    devices = []
    sections = [
        InputSection("Frame", (X_PIXELS, Y_PIXELS, SIMULATED_CHANNELS)),
        InputSection("Frames", (NUM_FRAMES,)),
        InputSection(
            "Signal",
            (SIGNAL_PATTERN, SIGNAL_LEVEL, NOISE_LEVEL, DRIFT_PIXELS_PER_FRAME, MASK_GAIN),
        ),
        InputSection("Modulation", (MASKS, MASK_LINES)),
        InputSection("Pacing", (FRAME_INTERVAL_MS,)),
    ]
    emits = {"intensity": Image}
    runners = [Single(), Continuous()]

    def run(self, ctx: RunContext) -> None:
        frame_shape = FrameShape.from_inputs(ctx.inputs)
        signal = SyntheticSignal.from_inputs(ctx.inputs)
        interval_ms = ctx.inputs[FRAME_INTERVAL_MS]
        lit = lit_pixels(ctx.inputs[MASKS], frame_shape)
        labels = sim_labels(frame_shape.channels)
        # Fixed for the run so its frames stay one drifting sample.
        seed = int(np.random.default_rng().integers(2**32))

        for index in count_off(ctx, ctx.inputs[NUM_FRAMES], "frame"):
            frame = synthetic_frame(frame_shape, signal, seed=seed, frame_index=index, lit=lit)
            ctx.publish("intensity", Image(frame, labels))
            ctx.sleep(interval_ms / 1000.0)
