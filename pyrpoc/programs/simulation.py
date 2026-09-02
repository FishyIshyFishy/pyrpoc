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
from pyrpoc.core.params import (
    Group,
    ModulationGroup,
    choice_field,
    float_field,
    group,
    int_field,
)
from pyrpoc.core.streams import Image2D
from pyrpoc.operations.simulation import PATTERNS, combine_masks, synthetic_frame
from pyrpoc.run.program import Program

from .registry import program_registry


@dataclass
class FrameGroup(Group):
    """The shape of a simulated frame -- what ScanGroup decides on a real rig."""

    x_pixels: int = int_field("X Pixels", 256, minimum=8, tooltip="Frame width in pixels")
    y_pixels: int = int_field("Y Pixels", 256, minimum=8, tooltip="Frame height in pixels")
    channels: int = int_field(
        "Channels", 2, minimum=1, maximum=16, tooltip="How many detector channels to fake"
    )


@dataclass
class SignalGroup(Group):
    """What the fake detector sees."""

    pattern: str = choice_field(
        "Pattern",
        "cells",
        choices=PATTERNS,
        tooltip="cells drift like a sample, the rest are test targets",
    )
    signal_level: float = float_field(
        "Signal Level", 1.0, minimum=0.0, tooltip="Peak brightness before noise"
    )
    noise_level: float = float_field(
        "Noise Level", 0.03, minimum=0.0, step=0.01, tooltip="Gaussian noise added per pixel"
    )
    drift_pixels_per_frame: float = float_field(
        "Drift (px/frame)", 1.5, minimum=0.0, tooltip="How far the pattern moves each frame"
    )
    mask_gain: float = float_field(
        "Mask Gain",
        0.5,
        minimum=0.0,
        tooltip="Extra brightness inside bound masks, standing in for stimulation",
    )
    seed: int = int_field(
        "Seed", 1234, minimum=0, tooltip="Same seed and frame index give the same pixels"
    )


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
    rather than in the operation because ``operations/`` may not read files.
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
                **p.frame,
                **p.signal,
                frame_index=index,
                mask=mask,
            )
            ctx.publish("intensity", frame, channels=labels)
            ctx.sleep(p.frame_interval_ms / 1000.0)
