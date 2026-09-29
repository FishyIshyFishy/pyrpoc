"""Every parameter block a program can be built from.

A block is a reusable unit of configuration. A program declares the blocks it
wants and the store hands it the one instance of each, so two modalities
declaring ``ScanGroup`` are configuring the same scan -- change the geometry in
confocal and FLIM already has it.

These live here rather than in ``structs/`` because they are what this instrument
is, not what the software is. ``structs/params.py`` holds the machinery; this holds
the content -- one block per file, plus the two field types that are this
instrument's own (``mask_field``, ``point_field``). Importing this package is
what registers every block, so a session can name any of them.
"""

from __future__ import annotations

from .daq import DaqGroup
from .frame import FrameGroup
from .histogram import HistogramGroup
from .mask_field import Mask, MasksField, masks_field
from .modulation import ModulationGroup
from .pacing import PacingGroup
from .point import PointGroup
from .point_field import Point, PointField, point_field
from .scan import ScanGroup
from .signal import PATTERNS, SignalGroup
from .spectrum import SpectrumGroup
from .split import SplitGroup
from .trigger import TriggerGroup

__all__ = [
    "DaqGroup",
    "FrameGroup",
    "HistogramGroup",
    "Mask",
    "masks_field",
    "MasksField",
    "ModulationGroup",
    "PacingGroup",
    "PATTERNS",
    "Point",
    "point_field",
    "PointField",
    "PointGroup",
    "ScanGroup",
    "SignalGroup",
    "SpectrumGroup",
    "SplitGroup",
    "TriggerGroup",
]
