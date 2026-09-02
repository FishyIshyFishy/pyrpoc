"""Simulation: frames out of thin air, no instruments required.

``uses = []``, so it runs on any machine: no DAQ, no galvo, no tagger, nothing
to claim. Everything above the hardware boundary is the real thing -- the
runner's thread, dataset creation from ``emits``, publishing, the save policy,
views and their source picker -- so this is how you exercise the software
itself, and how the UI can be looked at away from the rig.

Its parameter groups live here rather than in ``core/params.py``. Nothing else
will ever scan a simulated galvo, and a group used by exactly one program is
that program's business.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.modulation import load_mask
from pyrpoc.core.params import ModulationGroup, SaveGroup, group, int_field
from pyrpoc.core.streams import Image2D
from .synthetic import (
    PATTERNS,
    FrameGroup,
    SignalGroup,
    combine_masks,
    synthetic_frame,
)
from pyrpoc.run.program import Program

from .registry import program_registry


@dataclass
class SimulationParams:
    frame: FrameGroup = group(FrameGroup, "Frame")
    signal: SignalGroup = group(SignalGroup, "Signal")
    modulation: ModulationGroup = group(ModulationGroup, "Modulation")
    num_frames: int = int_field(
        "Frames", 10, minimum=1, tooltip="Number of frames to capture"
    )
    frame_interval_ms: int = int_field(
        "Frame Interval (ms)",
        100,
        minimum=0,
        tooltip="Pause between frames, standing in for acquisition time",
    )


def channel_labels(params: SimulationParams) -> list[str]:
    return [f"sim{index}" for index in range(params.frame.channels)]


def build_mask(params: SimulationParams):
    """Load the bound masks and flatten them onto the frame grid.

    Same shape as confocal's ``build_ttl``: once before the loop, and here
    rather than in ``synthetic.py``, which may not read files.
    """
    if not params.modulation.masks:
        return None
    loaded = [load_mask(binding.path) for binding in params.modulation.masks]
    return combine_masks(
        loaded, y_pixels=params.frame.y_pixels, x_pixels=params.frame.x_pixels
    )


@program_registry.register("simulation")
class Simulation(Program):
    uses = []
    params = SimulationParams
    emits = {"intensity": Image2D}

    def run(self, ctx) -> None:
        p: SimulationParams = ctx.params

        mask = build_mask(p)
        labels = channel_labels(p)
        total = "" if ctx.continuous else f"/{p.num_frames}"

        for index in ctx.frames(p.num_frames):
            ctx.status(f"frame {index + 1}{total}")
            frame = synthetic_frame(
                frame_shape=p.frame,
                signal=p.signal,
                frame_index=index,
                mask=mask,
            )
            ctx.publish("intensity", frame, channels=labels)
            ctx.sleep(p.frame_interval_ms / 1000.0)
