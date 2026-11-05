"""``Overview``: one image of a whole region, and where it lies on the dish."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.plugins.program_io.outputs import Placement


@dataclass(frozen=True)
class Overview:
    """A ``(C, H, W)`` image of a region, placed on the dish."""

    planes: np.ndarray
    placement: Placement
