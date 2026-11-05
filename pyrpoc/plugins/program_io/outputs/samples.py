"""``SamplesOutput``: every sample a detector took at each pixel, before averaging."""

from __future__ import annotations

from pyrpoc.core.program_output import program_output_registry

from .building_blocks.npz import NpzStack


@program_output_registry.register("raw_samples")
class SamplesOutput(NpzStack):
    """``(channel, y, x, sample)`` raw samples, one stack per acquisition."""

    axes = ("channel", "y", "x", "sample")
