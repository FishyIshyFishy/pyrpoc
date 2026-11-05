"""``HistogramOutput``: photon arrival times binned per pixel, what FLIM measures."""

from __future__ import annotations

from pyrpoc.core.program_output import program_output_registry

from .building_blocks.npz import NpzStack


@program_output_registry.register("decay_histogram")
class HistogramOutput(NpzStack):
    """``(y, x, bin)`` decay histograms, one per acquisition."""

    axes = ("y", "x", "bin")
